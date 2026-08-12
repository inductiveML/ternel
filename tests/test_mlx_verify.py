"""Gates on the artifact verifier.

A verifier that passes everything is worse than none, so most of what follows
corrupts a good artifact one way at a time and requires the verifier to notice.
The interesting cases are the ones a weaker check would miss: a tensor whose
bytes are right but whose *rows* were permuted wrongly, an index that points a
tensor at the wrong shard, and a code byte that is legal ternary but illegal as
a LUT index.

:func:`ternel_mlx.verify.inverse_permutation` gets its own gate because the
value-head order is not an involution -- applying it twice does not undo it --
and a verifier that assumed otherwise would report agreement on tensors that do
not agree.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from bonsai_tq1.format import BLOCK_BYTES, BLOCK_SIZE, FormatError, encode_tq1_blocks
from ternel_mlx.convert import SidecarTensor
from ternel_mlx.layout import PackedTensorLayout
from ternel_mlx.naming import VALUE_HEAD_REORDER, HeadLayout
from ternel_mlx.packing import pack_blocks, unpack_blocks
from ternel_mlx.shard_writer import PlannedTensor, shard_name, write_index, write_shard
from ternel_mlx.verify import (
    MAX_UNCHUNKED_WEIGHTS,
    ArtifactReader,
    RowChunk,
    TileChunk,
    inverse_permutation,
    sidecar_blocks,
    tile_chunks,
)

BONSAI = HeadLayout(key_heads=16, value_heads=48, value_head_dim=128)

METADATA = {"format": "pt"}


def canonical_blocks(rows: int, columns: int, seed: int) -> np.ndarray:
    """Legal TQ1_G128 blocks for a ``rows x columns`` tensor."""
    groups = columns // BLOCK_SIZE
    rng = np.random.default_rng(seed)
    codes = rng.integers(0, 3, size=(rows * groups, BLOCK_SIZE), dtype=np.uint8)
    scales = rng.integers(0, 256, size=(rows * groups, 2), dtype=np.uint8)
    return encode_tq1_blocks(scales, codes).reshape(rows, groups, BLOCK_BYTES)


def write_artifact(
    directory: Path, tensors: dict[str, np.ndarray], *, shards: int
) -> dict[str, str]:
    """A minimal sharded artifact holding exactly ``tensors``, split evenly."""
    names = list(tensors)
    per_shard = -(-len(names) // shards)
    planned = [
        [PlannedTensor(name, tensors[name].dtype, tensors[name].shape) for name in group]
        for group in (names[i : i + per_shard] for i in range(0, len(names), per_shard))
    ]
    total = 0
    for index, group in enumerate(planned):
        total += write_shard(
            directory / shard_name(index, len(planned)),
            group,
            lambda tensor: tensors[tensor.name],
            metadata=METADATA,
        )
    written = write_index(
        directory / "model.safetensors.index.json", planned, total_bytes=total
    )
    return written["weight_map"]


# --------------------------------------------------------------------------
# the permutation


def test_inverse_permutation_undoes_the_value_head_order_which_is_not_an_involution() -> None:
    """The failure this catches: inverting a permutation by reapplying it.

    mlx-lm head ``j`` is llama.cpp head ``(j % 3) * 16 + j // 3``. Applying that
    twice does *not* return the identity, so a verifier that re-permuted instead
    of inverting would compare two differently wrong orderings and see them
    agree only where the order happens to be a fixed point.
    """
    order = BONSAI.head_order()
    source = np.arange(order.size)
    permuted = source[order]
    assert not np.array_equal(permuted[order], source)
    assert np.array_equal(permuted[inverse_permutation(order)], source)


def test_inverse_permutation_refuses_an_order_that_is_not_a_permutation() -> None:
    with pytest.raises(FormatError):
        inverse_permutation(np.array([0, 1, 1, 3]))
    with pytest.raises(FormatError):
        inverse_permutation(np.array([0, 1, 2, 9]))


def test_inverse_permutation_undoes_a_real_row_order_at_block_granularity() -> None:
    order = BONSAI.source_order(10240, VALUE_HEAD_REORDER["attn_qkv.weight"])
    blocks = canonical_blocks(rows=10240, columns=BLOCK_SIZE, seed=7)
    permuted = blocks[order]
    assert not np.array_equal(permuted, blocks)
    assert np.array_equal(permuted[inverse_permutation(order)], blocks)


def test_inverse_permutation_undoes_a_real_group_order_on_the_reduction_axis() -> None:
    reorder = VALUE_HEAD_REORDER["ssm_out.weight"]
    order = BONSAI.group_source_order(48, reorder, group_size=BLOCK_SIZE)
    blocks = canonical_blocks(rows=64, columns=BLOCK_SIZE * 48, seed=17)
    permuted = blocks[:, order]
    assert not np.array_equal(permuted, blocks)
    assert np.array_equal(permuted[:, inverse_permutation(order)], blocks)


# --------------------------------------------------------------------------
# chunking


def test_tile_chunks_cover_every_tile_exactly_once_and_respect_the_budget() -> None:
    layout = PackedTensorLayout.for_tensor(rows=248320, columns=5120)
    chunks = list(tile_chunks(layout, weights_per_chunk=33_554_432, whole=False))
    assert chunks[0].begin_tile == 0
    assert chunks[-1].end_tile == layout.tiles
    assert all(a.end_tile == b.begin_tile for a, b in zip(chunks, chunks[1:]))
    assert all(chunk.tiles * layout.tile * layout.columns <= 33_554_432 for chunk in chunks)


def test_tile_chunks_yields_a_row_permuted_tensor_whole_because_no_slice_is_self_contained() -> None:
    layout = PackedTensorLayout.for_tensor(rows=10240, columns=5120)
    chunks = list(tile_chunks(layout, weights_per_chunk=1024, whole=True))
    assert chunks == [TileChunk(0, layout.tiles, layout.tile)]
    assert chunks[0].row_range.begin == 0
    assert chunks[0].row_range.end == layout.rows


def test_tile_chunks_refuses_a_whole_read_past_the_memory_ceiling() -> None:
    """The failure this catches: a permuted tensor silently exhausting memory.

    Reading whole is the price of a row permutation, so the price is bounded
    explicitly rather than discovered when the machine swaps.
    """
    columns = 1 << 15
    layout = PackedTensorLayout.for_tensor(rows=MAX_UNCHUNKED_WEIGHTS // columns + 256, columns=columns)
    with pytest.raises(FormatError):
        list(tile_chunks(layout, weights_per_chunk=1 << 20, whole=True))


def test_tile_chunks_refuses_a_non_positive_budget() -> None:
    layout = PackedTensorLayout.for_tensor(rows=256, columns=BLOCK_SIZE)
    with pytest.raises(FormatError):
        list(tile_chunks(layout, weights_per_chunk=0, whole=False))


# --------------------------------------------------------------------------
# reading the artifact back


def test_a_packed_tensor_round_trips_from_the_artifact_back_to_canonical_blocks(
    tmp_path: Path,
) -> None:
    """Storage leg 1 in miniature: pack, write, read, unpack, compare bytes.

    Everything the verifier claims about 26.9 billion weights rests on this
    round trip holding, so it is checked here on bytes small enough to print.
    """
    layout = PackedTensorLayout.for_tensor(rows=512, columns=BLOCK_SIZE * 3)
    blocks = canonical_blocks(rows=512, columns=BLOCK_SIZE * 3, seed=11)
    codes, scales = pack_blocks(blocks, layout=layout)
    write_artifact(tmp_path, {"t.codes": codes, "t.scales": scales}, shards=1)

    artifact = ArtifactReader(tmp_path)
    assert artifact.names == {"t.codes", "t.scales"}
    assert artifact.unclaimed() == set()
    restored = unpack_blocks(
        np.ascontiguousarray(artifact.read("t.codes")),
        np.ascontiguousarray(artifact.read("t.scales")),
        layout=layout,
    )
    assert np.array_equal(restored, blocks)


def test_reading_a_tile_range_matches_reading_the_whole_tensor(tmp_path: Path) -> None:
    layout = PackedTensorLayout.for_tensor(rows=1024, columns=BLOCK_SIZE * 2)
    blocks = canonical_blocks(rows=1024, columns=BLOCK_SIZE * 2, seed=13)
    codes, scales = pack_blocks(blocks, layout=layout)
    write_artifact(tmp_path, {"t.codes": codes, "t.scales": scales}, shards=1)

    artifact = ArtifactReader(tmp_path)
    for chunk in tile_chunks(layout, weights_per_chunk=BLOCK_SIZE * 2 * 256, whole=False):
        window = artifact.read_rows("t.codes", chunk.begin_tile, chunk.end_tile)
        assert np.array_equal(window, codes[chunk.begin_tile : chunk.end_tile])


def test_the_artifact_reader_spans_shards_and_reports_what_the_index_omits(
    tmp_path: Path,
) -> None:
    tensors = {f"t{i}.codes": np.full((2, 3), i, dtype=np.uint8) for i in range(4)}
    write_artifact(tmp_path, tensors, shards=2)

    artifact = ArtifactReader(tmp_path)
    assert len(artifact.shards) == 2
    assert len(set(artifact.owner.values())) == 2
    for name, expected in tensors.items():
        assert np.array_equal(artifact.read(name), expected)

    index_path = tmp_path / "model.safetensors.index.json"
    index = json.loads(index_path.read_text())
    dropped = index["weight_map"].pop("t3.codes")
    index_path.write_text(json.dumps(index))
    assert ArtifactReader(tmp_path).unclaimed() == {"t3.codes"}
    assert dropped


def test_the_artifact_reader_refuses_an_index_that_names_the_wrong_shard(
    tmp_path: Path,
) -> None:
    """The failure this catches: an index that loads, but loads the wrong bytes.

    ``mlx_lm.utils.load_model`` globs the shards and ignores the index, so an
    index pointing a tensor at a shard that does not hold it would go unnoticed
    by every MLX-side check and break every other tool in the ecosystem.
    """
    tensors = {f"t{i}.codes": np.full((2, 3), i, dtype=np.uint8) for i in range(4)}
    write_artifact(tmp_path, tensors, shards=2)
    index_path = tmp_path / "model.safetensors.index.json"
    index = json.loads(index_path.read_text())
    first, second = sorted(set(index["weight_map"].values()))
    index["weight_map"]["t0.codes"] = second
    index_path.write_text(json.dumps(index))
    with pytest.raises(FormatError, match="whose header lacks it"):
        ArtifactReader(tmp_path)

    index["weight_map"]["t0.codes"] = first
    index["weight_map"]["t9.codes"] = "model-00009-of-00002.safetensors"
    index_path.write_text(json.dumps(index))
    with pytest.raises(FormatError, match="which is not in"):
        ArtifactReader(tmp_path)


def test_the_artifact_reader_refuses_an_empty_index(tmp_path: Path) -> None:
    (tmp_path / "model.safetensors.index.json").write_text(
        json.dumps({"metadata": {"total_size": 0}, "weight_map": {}})
    )
    with pytest.raises(FormatError, match="carries no weight map"):
        ArtifactReader(tmp_path)


def test_reading_an_unindexed_tensor_names_the_tensor_rather_than_failing_obscurely(
    tmp_path: Path,
) -> None:
    write_artifact(tmp_path, {"t.codes": np.zeros((2, 3), dtype=np.uint8)}, shards=1)
    with pytest.raises(FormatError, match="absent.weight"):
        ArtifactReader(tmp_path).read("absent.weight")


# --------------------------------------------------------------------------
# the sidecar window


def test_sidecar_blocks_windows_the_requested_rows_and_refuses_to_run_past_the_file(
    tmp_path: Path,
) -> None:
    groups_per_row = 3
    rows = 8
    payload = np.arange(64 + rows * groups_per_row * BLOCK_BYTES, dtype=np.uint8) % 241
    path = tmp_path / "sidecar"
    path.write_bytes(payload.tobytes())
    mapping = np.memmap(path, dtype=np.uint8, mode="r")
    entry = SidecarTensor(
        name="t",
        offset=64,
        groups=rows * groups_per_row,
        elements=rows * groups_per_row * BLOCK_SIZE,
        payload_sha256="",
        logical_weight_sha256="",
        scale_bits_sha256="",
    )

    window = sidecar_blocks(mapping, entry, RowChunk(2, 5), groups_per_row=groups_per_row)
    assert window.shape == (3, groups_per_row, BLOCK_BYTES)
    assert np.array_equal(
        window.reshape(-1),
        payload[64 + 2 * groups_per_row * BLOCK_BYTES : 64 + 5 * groups_per_row * BLOCK_BYTES],
    )

    with pytest.raises(FormatError, match="past the sidecar"):
        sidecar_blocks(mapping, entry, RowChunk(2, rows + 1), groups_per_row=groups_per_row)


# --------------------------------------------------------------------------
# the storage leg, composed


def test_the_storage_leg_recovers_the_source_blocks_only_under_the_right_inverse(
    tmp_path: Path,
) -> None:
    """The failure this catches: a storage check that passes on any permutation.

    This is the whole of leg 1 assembled -- permute, pack, write, read, unpack,
    invert -- and then run a second time with a *different* inverse. The right
    one must recover the source bytes exactly and the wrong one must not, or the
    leg is measuring the round trip and not the ordering.
    """
    reorder = VALUE_HEAD_REORDER["attn_qkv.weight"]
    rows, columns = 10240, BLOCK_SIZE * 2
    layout = PackedTensorLayout.for_tensor(rows=rows, columns=columns)
    source = canonical_blocks(rows=rows, columns=columns, seed=23)
    order = BONSAI.source_order(rows, reorder)

    codes, scales = pack_blocks(np.ascontiguousarray(source[order]), layout=layout)
    write_artifact(tmp_path, {"t.codes": codes, "t.scales": scales}, shards=1)

    artifact = ArtifactReader(tmp_path)
    restored = unpack_blocks(
        np.ascontiguousarray(artifact.read("t.codes")),
        np.ascontiguousarray(artifact.read("t.scales")),
        layout=layout,
    )
    assert np.array_equal(restored[inverse_permutation(order)], source)
    assert not np.array_equal(restored[order], source)


def test_a_single_flipped_code_byte_in_a_shard_is_visible_after_the_round_trip(
    tmp_path: Path,
) -> None:
    """One byte of 5.9 GB, and the comparison has to be elementwise to see it."""
    layout = PackedTensorLayout.for_tensor(rows=256, columns=BLOCK_SIZE * 2)
    source = canonical_blocks(rows=256, columns=BLOCK_SIZE * 2, seed=29)
    codes, scales = pack_blocks(source, layout=layout)
    corrupted = codes.copy()
    corrupted[0, 1, 3, 7] = np.uint8(corrupted[0, 1, 3, 7] + 1)
    write_artifact(tmp_path, {"t.codes": corrupted, "t.scales": scales}, shards=1)

    artifact = ArtifactReader(tmp_path)
    restored = unpack_blocks(
        np.ascontiguousarray(artifact.read("t.codes")),
        np.ascontiguousarray(artifact.read("t.scales")),
        layout=layout,
    )
    assert int(np.count_nonzero(restored != source)) == 1
