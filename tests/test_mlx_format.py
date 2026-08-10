from __future__ import annotations

import numpy as np
import pytest

from bonsai_tq1.format import BLOCK_BYTES, BLOCK_SIZE, FormatError, encode_tq1_blocks
from bonsai_tq1.lut23_reorder import CODE_SLOTS, reorder_tensor_arrays, restore_tensor_arrays
from ternel_mlx.layout import (
    PREFERRED_TILE,
    PackedTensorLayout,
    SCALE_BYTES,
    select_tile,
)

# Every (rows, columns) pair the text-only Ternary Bonsai 27B graph quantises,
# in MLX convention (rows = output dimension, columns = reduction dimension).
MODEL_TENSOR_SHAPES = (
    ("embed_tokens", 248320, 5120),
    ("lm_head", 248320, 5120),
    ("linear_attn.in_proj_a", 48, 5120),
    ("linear_attn.in_proj_b", 48, 5120),
    ("linear_attn.in_proj_qkv", 10240, 5120),
    ("linear_attn.in_proj_z", 6144, 5120),
    ("linear_attn.out_proj", 5120, 6144),
    ("self_attn.q_proj", 12288, 5120),
    ("self_attn.k_proj", 1024, 5120),
    ("self_attn.v_proj", 1024, 5120),
    ("self_attn.o_proj", 5120, 12288),
    ("mlp.gate_proj", 17408, 5120),
    ("mlp.up_proj", 17408, 5120),
    ("mlp.down_proj", 5120, 17408),
)


def test_selected_tile_divides_every_row_count_exactly() -> None:
    for rows in range(1, 2000):
        tile = select_tile(rows)
        assert 0 < tile <= rows
        assert rows % tile == 0
        assert tile == (PREFERRED_TILE if rows % PREFERRED_TILE == 0 else rows)


@pytest.mark.parametrize("rows", (0, -1, -256))
def test_selected_tile_rejects_non_positive_rows(rows: int) -> None:
    with pytest.raises(FormatError, match="row count must be positive"):
        select_tile(rows)


@pytest.mark.parametrize(("name", "rows", "columns"), MODEL_TENSOR_SHAPES)
def test_model_tensors_pack_without_a_single_padded_row(
    name: str, rows: int, columns: int
) -> None:
    """The whole-model claim: 28 bytes per 128 weights, no padding anywhere."""
    layout = PackedTensorLayout.for_tensor(rows, columns)
    assert layout.tiles * layout.tile == rows, name
    assert layout.tile == (PREFERRED_TILE if rows % PREFERRED_TILE == 0 else rows)
    assert layout.payload_bytes == layout.canonical_bytes
    assert layout.bits_per_weight == pytest.approx(1.75)
    assert layout.codes_shape == (layout.tiles, columns // BLOCK_SIZE, CODE_SLOTS, layout.tile)
    assert layout.scales_shape == (layout.tiles, columns // BLOCK_SIZE, layout.tile)


def test_only_the_ssm_gates_fall_off_the_preferred_tile() -> None:
    off_tile = {name for name, rows, _ in MODEL_TENSOR_SHAPES if rows % PREFERRED_TILE}
    assert off_tile == {"linear_attn.in_proj_a", "linear_attn.in_proj_b"}


def test_scale_bytes_account_for_the_raw_fp16_scale() -> None:
    assert SCALE_BYTES == 2
    assert CODE_SLOTS + SCALE_BYTES == BLOCK_BYTES


@pytest.mark.parametrize(("rows", "columns"), ((48, 5120), (256, 640), (512, 256), (100, 384)))
def test_layout_round_trips_through_the_shared_reorder(rows: int, columns: int) -> None:
    layout = PackedTensorLayout.for_tensor(rows, columns)
    groups_per_row = layout.groups_per_row
    generator = np.random.default_rng(20260810 + rows + columns)
    scales = generator.integers(0, 256, size=(rows * groups_per_row, 2), dtype=np.uint8)
    logical = generator.integers(0, 3, size=(rows * groups_per_row, BLOCK_SIZE), dtype=np.uint8)
    packed = encode_tq1_blocks(scales, logical).reshape(rows, groups_per_row, BLOCK_BYTES)

    codes, reordered_scales = reorder_tensor_arrays(packed, tile=layout.tile)
    assert codes.shape == layout.codes_shape
    assert codes.nbytes == layout.codes_bytes
    assert reordered_scales.nbytes == layout.scales_bytes

    # The artifact stores the scale pair as one little-endian uint16 of raw
    # FP16 bits; that view must be a pure reinterpretation, never a conversion.
    scale_bits = reordered_scales.reshape(-1).view("<u2").reshape(layout.scales_shape)
    assert scale_bits.shape == layout.scales_shape
    np.testing.assert_array_equal(
        scale_bits.reshape(-1).view(np.uint8), reordered_scales.reshape(-1)
    )

    np.testing.assert_array_equal(
        restore_tensor_arrays(codes, reordered_scales, rows, tile=layout.tile), packed
    )


@pytest.mark.parametrize(
    ("rows", "columns", "tile", "message"),
    (
        (0, 5120, 256, "dimensions must be positive"),
        (256, 0, 256, "dimensions must be positive"),
        (256, 5000, 256, "not a multiple"),
        (256, 5120, 0, "tile must be positive"),
        (100, 5120, 256, "does not divide"),
    ),
)
def test_layout_rejects_impossible_geometry(
    rows: int, columns: int, tile: int, message: str
) -> None:
    with pytest.raises(FormatError, match=message):
        PackedTensorLayout(rows=rows, columns=columns, tile=tile)


def test_layout_json_is_self_describing() -> None:
    payload = PackedTensorLayout.for_tensor(5120, 17408).to_json()
    assert payload["layout_name"] == "TQ1_G128_MLX_TILED"
    assert payload["layout_version"] == 1
    assert payload["tile"] == 256
    assert payload["tiles"] == 20
    assert payload["groups_per_row"] == 136
    assert payload["groups"] == 5120 * 136
    assert payload["payload_bytes"] == 5120 * 136 * BLOCK_BYTES
