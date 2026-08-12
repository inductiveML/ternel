"""Gates on the safetensors writer and the sidecar-to-artifact converter.

The writer is the one place in Ternel that produces bytes MLX will later read,
so it is checked against MLX's own loader rather than only against Ternel's, and
the ``model_file`` shim is imported through the exact ``spec_from_file_location``
call ``mlx_lm.utils.load_model`` makes -- which is the only way to find out
whether a checkpoint's imports actually resolve.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import ml_dtypes
import numpy as np
import pytest

from bonsai_tq1.format import BLOCK_BYTES, BLOCK_SIZE, FormatError
from bonsai_tq1.gguf_utils import TensorInfo
from ternel_mlx import LAYOUT_NAME, LAYOUT_VERSION
from ternel_mlx.convert import (
    CONFIG_NAME,
    MODEL_FILE_NAME,
    QUANTISATION_KEYS,
    SidecarTensor,
    artifact_config,
    read_canonical_blocks,
    reorder_blocks,
    sha256_bytes,
    write_model_file,
)
from ternel_mlx.layout import PackedTensorLayout
from ternel_mlx.naming import HeadLayout, MappedTensor, map_gguf_name
from ternel_mlx.packed_model import ACTIVATION_DTYPE_KEY
from ternel_mlx.remote_safetensors import LocalSafetensors
from ternel_mlx.shard_writer import (
    HEADER_ALIGNMENT,
    PlannedTensor,
    expected_file_bytes,
    header_bytes,
    plan_shards,
    shard_name,
    write_index,
    write_shard,
)

BONSAI = HeadLayout(key_heads=16, value_heads=48, value_head_dim=128)

METADATA = {"format": "pt", "layout_name": LAYOUT_NAME}


def planned(name: str, dtype: str, shape: tuple[int, ...]) -> PlannedTensor:
    return PlannedTensor(name, np.dtype(dtype), shape)


def mapped(name: str, columns: int, rows: int) -> MappedTensor:
    """One Q2_0 tensor as ``build_name_map`` would produce it."""
    elements = rows * columns
    info = TensorInfo(
        index=0,
        name=name,
        tensor_type="Q2_0",
        shape=(columns, rows),
        elements=elements,
        data_offset=0,
        data_bytes=elements * 34 // 128,
    )
    mlx_name, quantised = map_gguf_name(name)
    return MappedTensor(gguf=info, mlx_name=mlx_name, quantised=quantised)


def sidecar_entry(*, offset: int, groups: int, elements: int) -> SidecarTensor:
    """One frozen-sidecar manifest entry, with digests these tests do not read."""
    return SidecarTensor(
        name="t",
        offset=offset,
        groups=groups,
        elements=elements,
        payload_sha256="",
        logical_weight_sha256="",
        scale_bits_sha256="",
    )


# --------------------------------------------------------------------------
# planning


def test_plan_shards_fills_each_shard_to_the_budget_in_order() -> None:
    tensors = [planned(f"t{i}", "u1", (100,)) for i in range(5)]
    shards = plan_shards(tensors, max_shard_bytes=250)
    assert [[t.name for t in shard] for shard in shards] == [
        ["t0", "t1"],
        ["t2", "t3"],
        ["t4"],
    ]


def test_plan_shards_gives_an_oversized_tensor_its_own_shard_rather_than_splitting() -> None:
    tensors = [planned("small", "u1", (10,)), planned("huge", "u1", (1000,))]
    shards = plan_shards(tensors, max_shard_bytes=100)
    assert [[t.name for t in shard] for shard in shards] == [["small"], ["huge"]]
    # The single-tensor shard is over budget because safetensors has no concept
    # of a tensor spanning files; silently truncating it would be the alternative.
    assert shards[1][0].nbytes > 100


def test_plan_shards_refuses_an_empty_set_a_duplicate_name_and_a_zero_budget() -> None:
    with pytest.raises(FormatError):
        plan_shards([], max_shard_bytes=100)
    with pytest.raises(FormatError):
        plan_shards([planned("t", "u1", (1,)), planned("t", "u1", (1,))], max_shard_bytes=100)
    for budget in (0, -1):
        with pytest.raises(FormatError):
            plan_shards([planned("t", "u1", (1,))], max_shard_bytes=budget)


def test_planned_tensor_rejects_an_unnamed_unstorable_or_negative_tensor() -> None:
    with pytest.raises(FormatError):
        planned("", "u1", (1,))
    with pytest.raises(FormatError):
        PlannedTensor("t", np.dtype(np.complex64), (1,))
    with pytest.raises(FormatError):
        planned("t", "u1", (1, -1))


def test_shard_name_is_the_bare_name_alone_and_the_hub_form_in_a_set() -> None:
    assert shard_name(0, 1) == "model.safetensors"
    assert shard_name(0, 3) == "model-00001-of-00003.safetensors"
    assert shard_name(2, 3) == "model-00003-of-00003.safetensors"
    for index, count in ((1, 1), (-1, 2), (2, 2), (0, 0)):
        with pytest.raises(FormatError):
            shard_name(index, count)


def test_header_is_eight_byte_aligned_and_its_prefix_states_its_own_length() -> None:
    raw = header_bytes([planned("a", "<f4", (3, 5))], METADATA)
    length = int(np.frombuffer(raw[:8], dtype="<u8")[0])
    assert len(raw) == 8 + length
    assert length % HEADER_ALIGNMENT == 0
    assert json.loads(raw[8:])["a"]["data_offsets"] == [0, 60]


def test_header_refuses_non_string_metadata_which_safetensors_cannot_hold() -> None:
    with pytest.raises(FormatError):
        header_bytes([planned("a", "u1", (1,))], {"version": 1})


# --------------------------------------------------------------------------
# writing


def test_a_written_shard_reads_back_identically_through_mlx_and_through_ternel(
    tmp_path: Path,
) -> None:
    arrays = {
        "codes": np.arange(2 * 3 * 26 * 4, dtype=np.uint8).reshape(2, 3, 26, 4),
        "scales": np.arange(2 * 3 * 4, dtype="<u2").reshape(2, 3, 4) * 997,
        "norm": (np.arange(8) / 3).astype(ml_dtypes.bfloat16),
        "logits": np.linspace(-2, 2, 12, dtype="<f4").reshape(3, 4),
    }
    tensors = [planned(name, array.dtype, array.shape) for name, array in arrays.items()]
    path = tmp_path / "model.safetensors"
    written = write_shard(path, tensors, lambda t: arrays[t.name], metadata=METADATA)

    assert written == sum(array.nbytes for array in arrays.values())
    assert path.stat().st_size == expected_file_bytes(tensors, METADATA)

    reader = LocalSafetensors(path)
    assert sorted(reader.names()) == sorted(arrays)
    assert reader.metadata == METADATA
    for name, array in arrays.items():
        back = reader.read(name)
        assert back.dtype == array.dtype and back.shape == array.shape
        assert sha256_bytes(back) == sha256_bytes(array)

    mx = pytest.importorskip("mlx.core")
    loaded = mx.load(str(path))
    assert sorted(loaded) == sorted(arrays)
    for name, array in arrays.items():
        assert list(loaded[name].shape) == list(array.shape)
        assert np.array_equal(np.asarray(loaded[name].view(mx.uint8)), array.view(np.uint8))


def test_write_shard_rejects_a_produced_array_that_is_not_what_was_planned(
    tmp_path: Path,
) -> None:
    path = tmp_path / "model.safetensors"
    tensor = planned("a", "<f4", (2, 2))
    for wrong in (np.zeros((2, 2), dtype="<f2"), np.zeros((4,), dtype="<f4")):
        with pytest.raises(FormatError):
            write_shard(path, [tensor], lambda _: wrong, metadata=METADATA)
        assert not path.exists()
        assert not path.with_name(path.name + ".tmp").exists()


def test_write_shard_leaves_nothing_behind_when_the_producer_raises(tmp_path: Path) -> None:
    path = tmp_path / "model.safetensors"

    def explode(tensor: PlannedTensor) -> np.ndarray:
        raise RuntimeError("no bytes today")

    with pytest.raises(RuntimeError):
        write_shard(path, [planned("a", "u1", (4,))], explode, metadata=METADATA)
    assert list(tmp_path.iterdir()) == []


def test_write_index_maps_every_tensor_to_the_shard_holding_it(tmp_path: Path) -> None:
    shards = [[planned("a", "u1", (4,)), planned("b", "u1", (4,))], [planned("c", "u1", (4,))]]
    index = write_index(tmp_path / "model.safetensors.index.json", shards, total_bytes=99)
    assert index["weight_map"] == {
        "a": "model-00001-of-00002.safetensors",
        "b": "model-00001-of-00002.safetensors",
        "c": "model-00002-of-00002.safetensors",
    }
    assert index["metadata"] == {"total_size": 99}
    assert json.loads((tmp_path / "model.safetensors.index.json").read_text()) == index


# --------------------------------------------------------------------------
# converting


def test_reorder_blocks_permutes_whole_rows_for_a_row_axis_tensor() -> None:
    tensor = mapped("blk.0.ssm_alpha.weight", 5120, 48)
    blocks = np.arange(48 * 40 * BLOCK_BYTES, dtype=np.uint8).reshape(48, 40, BLOCK_BYTES)
    ordered = reorder_blocks(blocks, tensor, head_layout=BONSAI)
    assert np.array_equal(ordered, blocks[BONSAI.head_order()])
    assert BONSAI.head_order()[:4].tolist() == [0, 16, 32, 1]


def test_reorder_blocks_permutes_whole_groups_for_a_column_axis_tensor() -> None:
    tensor = mapped("blk.0.ssm_out.weight", 6144, 5120)
    blocks = np.arange(8 * 48 * BLOCK_BYTES, dtype=np.uint8).reshape(8, 48, BLOCK_BYTES)
    ordered = reorder_blocks(blocks, tensor, head_layout=BONSAI)
    # A value head is 128 channels, which is exactly one group, so the group
    # permutation is the bare head permutation.
    assert np.array_equal(ordered, blocks[:, BONSAI.head_order()])


def test_reorder_blocks_leaves_an_unaffected_tensor_untouched() -> None:
    tensor = mapped("blk.0.ffn_gate.weight", 5120, 17408)
    blocks = np.arange(4 * 40 * BLOCK_BYTES, dtype=np.uint8).reshape(4, 40, BLOCK_BYTES)
    assert reorder_blocks(blocks, tensor, head_layout=BONSAI) is blocks


def test_read_canonical_blocks_shapes_the_window_and_refuses_a_disagreeing_entry() -> None:
    layout = PackedTensorLayout.for_tensor(rows=512, columns=BLOCK_SIZE * 3)
    payload = np.arange(layout.canonical_bytes + 64, dtype=np.uint8) % 241
    entry = sidecar_entry(offset=64, groups=layout.groups, elements=512 * 384)
    blocks = read_canonical_blocks(payload, entry, layout=layout)
    assert blocks.shape == (512, 3, BLOCK_BYTES)
    assert np.array_equal(blocks.reshape(-1), payload[64:])

    for broken in (
        sidecar_entry(offset=64, groups=layout.groups + 1, elements=512 * 384),
        sidecar_entry(offset=64, groups=layout.groups, elements=512 * 384 + 1),
        sidecar_entry(offset=128, groups=layout.groups, elements=512 * 384),
    ):
        with pytest.raises(FormatError):
            read_canonical_blocks(payload, broken, layout=layout)


def test_artifact_config_strips_every_claim_the_artifact_does_not_meet() -> None:
    baseline = {
        "model_type": "qwen3_5",
        "quantization": {"group_size": 128, "bits": 2},
        "quantization_config": {"group_size": 128, "bits": 2},
        "vision_config": {"depth": 27},
        "image_token_id": 151655,
        "video_token_id": 151656,
        "vision_start_token_id": 151652,
        "vision_end_token_id": 151653,
        "text_config": {
            "hidden_size": 5120,
            "quantization": {"group_size": 128, "bits": 2},
            "quantization_config": {"group_size": 128, "bits": 2},
        },
    }
    frozen = json.dumps(baseline, sort_keys=True)
    config = artifact_config(baseline, activation_dtype="bfloat16")

    for key in QUANTISATION_KEYS:
        assert key not in config and key not in config["text_config"]
    assert not any(key.startswith("vision") or key.endswith("token_id") for key in config)
    assert config["model_file"] == MODEL_FILE_NAME
    assert config["language_model_only"] is True
    assert config["text_config"][ACTIVATION_DTYPE_KEY] == "bfloat16"
    assert config["text_config"]["hidden_size"] == 5120
    assert json.dumps(baseline, sort_keys=True) == frozen, "the baseline config was mutated"


def test_artifact_config_refuses_a_baseline_without_a_text_config() -> None:
    with pytest.raises(FormatError):
        artifact_config({"model_type": "qwen3_5"}, activation_dtype="bfloat16")


def test_the_model_file_imports_the_way_mlx_lm_imports_it(tmp_path: Path) -> None:
    """The failure this catches: an artifact whose ``model_file`` cannot import.

    ``load_model`` loads the file by path under the name ``custom_model``, with
    no package and without the model directory on ``sys.path``, so sibling
    modules copied beside it would not resolve. Executing it exactly that way is
    the only check that the emitted checkpoint is loadable at all.
    """
    pytest.importorskip("mlx.core")
    record = write_model_file(tmp_path)
    path = tmp_path / MODEL_FILE_NAME
    assert record == {
        "name": MODEL_FILE_NAME,
        "sha256": sha256_bytes(np.frombuffer(path.read_bytes(), dtype=np.uint8)),
        "requires": f"ternel_mlx>={LAYOUT_VERSION}",
    }

    spec = importlib.util.spec_from_file_location("custom_model", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    from ternel_mlx.packed_model import Model, ModelArgs

    assert module.Model is Model
    assert module.ModelArgs is ModelArgs
    assert LAYOUT_NAME in (module.__doc__ or "")


def test_the_emitted_config_names_a_model_file_that_exists(tmp_path: Path) -> None:
    config = artifact_config(
        {"model_type": "qwen3_5", "text_config": {"hidden_size": 5120}},
        activation_dtype="float16",
    )
    (tmp_path / CONFIG_NAME).write_text(json.dumps(config), encoding="utf-8")
    write_model_file(tmp_path)
    named = json.loads((tmp_path / CONFIG_NAME).read_text())["model_file"]
    assert (tmp_path / named).is_file()
