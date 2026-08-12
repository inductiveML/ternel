from __future__ import annotations

import numpy as np
import pytest

from bonsai_tq1.format import FormatError
from bonsai_tq1.gguf_utils import TensorInfo
from ternel_mlx.crosscheck import RowChunk, reordered_rows, row_chunks, zeros_only_difference
from ternel_mlx.naming import (
    PLAIN_BLOCK_PARAMETERS,
    PLAIN_GLOBAL_PARAMETERS,
    QUANTISED_BLOCK_MODULES,
    QUANTISED_GLOBAL_MODULES,
    VALUE_HEAD_REORDER,
    HeadAxis,
    HeadLayout,
    HeadUnit,
    MappedTensor,
    ValueHeadReorder,
    build_name_map,
    map_gguf_name,
)

# Ternary Bonsai 27B's own geometry, from the official checkpoint's config.json.
BONSAI = HeadLayout(key_heads=16, value_heads=48, value_head_dim=128)


def make_tensor(name: str, columns: int, rows: int, tensor_type: str) -> TensorInfo:
    """A TensorInfo in GGUF convention, where shape is (columns, rows)."""
    elements = rows * columns
    per_element = 34 / 128 if tensor_type == "Q2_0" else 4
    return TensorInfo(
        index=0,
        name=name,
        tensor_type=tensor_type,
        shape=(columns, rows),
        elements=elements,
        data_offset=0,
        data_bytes=int(elements * per_element),
    )


def test_head_layout_rejects_a_geometry_that_does_not_divide() -> None:
    with pytest.raises(FormatError):
        HeadLayout(key_heads=7, value_heads=48, value_head_dim=128)
    for bad in ({"key_heads": 0}, {"value_heads": -1}, {"value_head_dim": 0}):
        fields = {"key_heads": 16, "value_heads": 48, "value_head_dim": 128, **bad}
        with pytest.raises(FormatError):
            HeadLayout(**fields)


def test_head_layout_reads_the_official_config_fields() -> None:
    layout = HeadLayout.from_text_config(
        {"linear_num_key_heads": 16, "linear_num_value_heads": 48, "linear_value_head_dim": 128}
    )
    assert layout == BONSAI
    assert layout.repeat == 3
    assert layout.value_dim == 6144


def test_head_layout_refuses_missing_or_non_integer_config_fields() -> None:
    complete = {
        "linear_num_key_heads": 16,
        "linear_num_value_heads": 48,
        "linear_value_head_dim": 128,
    }
    for key in complete:
        partial = {name: value for name, value in complete.items() if name != key}
        with pytest.raises(FormatError):
            HeadLayout.from_text_config(partial)
    for bad in (16.0, "16", True, None):
        with pytest.raises(FormatError):
            HeadLayout.from_text_config({**complete, "linear_num_key_heads": bad})


def test_head_order_matches_the_measured_permutation() -> None:
    # Measured against the official checkpoint: mlx-lm row 1 holds GGUF row 16,
    # row 2 holds row 32, row 3 holds row 1, row 4 holds row 17.
    order = BONSAI.head_order()
    assert order.tolist()[:5] == [0, 16, 32, 1, 17]
    assert sorted(order.tolist()) == list(range(48))


def test_head_order_is_a_transpose_of_the_two_groupings() -> None:
    order = BONSAI.head_order()
    grid = np.arange(BONSAI.value_heads).reshape(BONSAI.repeat, BONSAI.key_heads)
    assert order.tolist() == grid.T.reshape(-1).tolist()


def test_head_order_is_a_permutation_for_every_plausible_geometry() -> None:
    for key_heads in range(1, 9):
        for repeat in range(1, 7):
            layout = HeadLayout(
                key_heads=key_heads, value_heads=key_heads * repeat, value_head_dim=64
            )
            order = layout.head_order()
            assert sorted(order.tolist()) == list(range(layout.value_heads))


def test_source_order_leaves_the_prefix_alone_and_permutes_the_suffix() -> None:
    reorder = ValueHeadReorder(HeadAxis.ROWS, HeadUnit.CHANNEL)
    order = BONSAI.source_order(10240, reorder)
    assert order[:4096].tolist() == list(range(4096))
    assert sorted(order[4096:].tolist()) == list(range(4096, 10240))
    # The first value channel of mlx head 1 is the first channel of GGUF head 16.
    assert int(order[4096 + 128]) == 4096 + 16 * 128


def test_source_order_covers_the_whole_axis_when_it_is_all_value_heads() -> None:
    for unit, length in ((HeadUnit.HEAD, 48), (HeadUnit.CHANNEL, 6144)):
        order = BONSAI.source_order(length, ValueHeadReorder(HeadAxis.ROWS, unit))
        assert sorted(order.tolist()) == list(range(length))


def test_source_order_refuses_an_axis_shorter_than_the_block() -> None:
    with pytest.raises(FormatError):
        BONSAI.source_order(47, ValueHeadReorder(HeadAxis.ROWS, HeadUnit.HEAD))
    with pytest.raises(FormatError):
        BONSAI.source_order(6143, ValueHeadReorder(HeadAxis.COLUMNS, HeadUnit.CHANNEL))


def test_group_source_order_restates_a_column_reorder_in_quantisation_groups() -> None:
    reorder = ValueHeadReorder(HeadAxis.COLUMNS, HeadUnit.CHANNEL)
    # One value head is exactly one 128-weight group, so the group axis carries
    # one slot per head and the order is the bare head permutation.
    groups = BONSAI.group_source_order(48, reorder, group_size=128)
    assert groups.tolist() == BONSAI.head_order().tolist()
    # Halving the group size doubles the slots per head, and expanding the group
    # order back out to weights must reproduce the weight-level order.
    halved = BONSAI.group_source_order(96, reorder, group_size=64)
    assert (halved[:, None] * 64 + np.arange(64)).reshape(-1).tolist() == BONSAI.source_order(
        6144, reorder
    ).tolist()


def test_group_source_order_refuses_a_head_that_is_not_whole_groups() -> None:
    reorder = ValueHeadReorder(HeadAxis.COLUMNS, HeadUnit.CHANNEL)
    for group_size in (0, -128, 256, 48):
        with pytest.raises(FormatError):
            BONSAI.group_source_order(48, reorder, group_size=group_size)
    # A per-head scalar has no sub-head structure to express in groups at all.
    with pytest.raises(FormatError):
        BONSAI.group_source_order(48, ValueHeadReorder(HeadAxis.COLUMNS, HeadUnit.HEAD), group_size=128)


def test_reordering_twice_is_not_the_identity_but_composing_with_argsort_is() -> None:
    reorder = ValueHeadReorder(HeadAxis.ROWS, HeadUnit.HEAD)
    order = BONSAI.source_order(48, reorder)
    gguf = np.arange(48)
    mlx = gguf[order]
    assert not np.array_equal(mlx, gguf)
    assert np.array_equal(mlx[np.argsort(order)], gguf)


def test_every_reordered_kind_exists_and_every_other_kind_is_untouched() -> None:
    known = (
        QUANTISED_BLOCK_MODULES.keys()
        | PLAIN_BLOCK_PARAMETERS.keys()
        | QUANTISED_GLOBAL_MODULES.keys()
        | PLAIN_GLOBAL_PARAMETERS.keys()
    )
    assert VALUE_HEAD_REORDER.keys() <= known
    # Only the linear-attention tensors carry a value-head axis.
    assert all(
        kind.startswith(("attn_qkv", "attn_gate", "ssm_")) for kind in VALUE_HEAD_REORDER
    )
    for kind in ("ffn_gate.weight", "ffn_up.weight", "attn_q.weight", "token_embd.weight"):
        assert kind not in VALUE_HEAD_REORDER


def test_mapped_tensor_reports_its_kind_and_reordering() -> None:
    qkv = MappedTensor(
        make_tensor("blk.7.attn_qkv.weight", 5120, 10240, "Q2_0"),
        "language_model.model.layers.7.linear_attn.in_proj_qkv",
        True,
    )
    assert qkv.kind == "attn_qkv.weight"
    assert qkv.reorder == ValueHeadReorder(HeadAxis.ROWS, HeadUnit.CHANNEL)

    down = MappedTensor(
        make_tensor("blk.7.ffn_down.weight", 17408, 5120, "Q2_0"),
        "language_model.model.layers.7.mlp.down_proj",
        True,
    )
    assert down.reorder is None

    output = MappedTensor(
        make_tensor("output_norm.weight", 1, 5120, "F32"),
        "language_model.model.norm.weight",
        False,
    )
    assert output.kind == "output_norm.weight"
    assert output.reorder is None


def test_out_proj_reorders_columns_not_rows() -> None:
    out = MappedTensor(
        make_tensor("blk.7.ssm_out.weight", 6144, 5120, "Q2_0"),
        "language_model.model.layers.7.linear_attn.out_proj",
        True,
    )
    assert out.reorder == ValueHeadReorder(HeadAxis.COLUMNS, HeadUnit.CHANNEL)
    # One value head is exactly one 128-weight quantisation group, so the column
    # reordering moves whole groups.
    assert out.columns // 128 == BONSAI.value_heads


def test_name_map_is_injective_and_type_consistent() -> None:
    tensors = [
        make_tensor("blk.0.ffn_gate.weight", 5120, 17408, "Q2_0"),
        make_tensor("blk.0.ffn_up.weight", 5120, 17408, "Q2_0"),
        make_tensor("blk.0.attn_norm.weight", 1, 5120, "F32"),
    ]
    mapped = build_name_map(tensors)
    assert [item.mlx_name for item in mapped] == [
        "language_model.model.layers.0.mlp.gate_proj",
        "language_model.model.layers.0.mlp.up_proj",
        "language_model.model.layers.0.input_layernorm.weight",
    ]
    with pytest.raises(FormatError):
        build_name_map(tensors + [make_tensor("blk.0.ffn_gate.weight", 5120, 17408, "Q2_0")])
    with pytest.raises(FormatError):
        build_name_map([make_tensor("blk.0.ffn_gate.weight", 5120, 17408, "F32")])
    with pytest.raises(FormatError):
        map_gguf_name("blk.0.not_a_real_tensor.weight")
    with pytest.raises(FormatError):
        map_gguf_name("vision_tower.blocks.0.attn.qkv.weight")


def test_row_chunks_tile_the_rows_without_splitting_the_reordered_block() -> None:
    chunks = list(row_chunks(10240, 5120, weights_per_chunk=5_120_000, unsplittable_from=4096))
    assert [chunk.begin for chunk in chunks] == [0, 1000, 2000, 3000, 4000, 4096]
    assert chunks[-1] == RowChunk(4096, 10240)
    assert chunks[0].begin == 0 and chunks[-1].end == 10240
    for earlier, later in zip(chunks, chunks[1:]):
        assert earlier.end == later.begin


def test_row_chunks_split_everywhere_when_nothing_is_unsplittable() -> None:
    chunks = list(row_chunks(1000, 100, weights_per_chunk=30_000, unsplittable_from=1000))
    assert [(chunk.begin, chunk.end) for chunk in chunks] == [
        (0, 300),
        (300, 600),
        (600, 900),
        (900, 1000),
    ]
    assert sum(chunk.rows for chunk in chunks) == 1000


def test_row_chunks_reject_a_boundary_outside_the_tensor() -> None:
    for bad in (-1, 1001):
        with pytest.raises(ValueError):
            list(row_chunks(1000, 100, weights_per_chunk=30_000, unsplittable_from=bad))
    with pytest.raises(ValueError):
        list(row_chunks(0, 100, weights_per_chunk=30_000, unsplittable_from=0))


def test_reordered_rows_accepts_a_self_contained_chunk_and_refuses_a_split_one() -> None:
    order = BONSAI.source_order(10240, ValueHeadReorder(HeadAxis.ROWS, HeadUnit.CHANNEL))
    local = reordered_rows(RowChunk(4096, 10240), order)
    assert local is not None
    assert sorted(local.tolist()) == list(range(6144))
    assert reordered_rows(RowChunk(4096, 7000), order) is None
    assert reordered_rows(RowChunk(0, 4096), order) is not None


def test_zeros_only_difference_isolates_the_sign_of_zero() -> None:
    left = np.array([0x0000, 0x8000, 0x3C00, 0x0000, 0x0001], dtype="<u2")
    right = np.array([0x8000, 0x0000, 0x3C00, 0x0001, 0x0000], dtype="<u2")
    assert zeros_only_difference(left, right).tolist() == [True, True, False, False, False]
    # Every genuinely differing pair of zeros is caught, and nothing else is.
    assert not zeros_only_difference(left, left).any()
