"""Independent CPU decoders that the Metal kernels are gated against.

Three paths compute the same product by deliberately different means, so a
shared bug would have to be implemented three times to go unnoticed:

``restored_matmul``
    Inverts the tile permutation, hands the canonical blocks to the frozen
    ``bonsai_tq1.reference`` decode-then-dot GEMV, and therefore also proves the
    permutation itself. Knows nothing about LUT23.

``lut23_matmul``
    Builds the same L2/L3/tail tables the kernel builds and indexes them with
    the same ``(p * 57) >> 9`` split, accumulating in the kernel's exact order.
    Never materialises a ternary weight, and catches structural errors that a
    decode-then-dot path cannot see.

``oracle_matmul``
    Decodes to a dense float64 matrix and calls BLAS. Slowest, simplest, and the
    only one whose result is essentially free of fp32 accumulation error, so it
    is the exactness reference rather than a peer.

``restored_matmul`` and ``oracle_matmul`` share ``decode_reordered_weights``
only for the trit values; their arithmetic is unrelated.
"""

from __future__ import annotations

import numpy as np

from bonsai_tq1.format import BLOCK_SIZE, FormatError, decode_tq1_blocks
from bonsai_tq1.lut23_reorder import CODE_SLOTS, FULL_CODE_SLOTS
from bonsai_tq1.reference import tq1_g128_reference_gemv

from .kernels import L2_ENTRIES, L3_ENTRIES, TAIL_ENTRIES
from .layout import PackedTensorLayout
from .packing import scale_bits_to_float, unpack_blocks

# Trits per regular code slot, and the activation offset the tail slot starts at.
TRITS_PER_SLOT = 5
TAIL_FIRST_COLUMN = FULL_CODE_SLOTS * TRITS_PER_SLOT

# Guards the float64 oracle against being handed a model-sized tensor by accident.
MAX_ORACLE_WEIGHTS = 1 << 27


def _digit(entries: int, position: int) -> np.ndarray:
    """Digit ``position`` of every table index, read in base 3.

    The table index *is* its own base-3 digit vector, which is what lets the
    kernel skip decoding trits entirely.
    """
    index = np.arange(entries, dtype=np.int64)
    return (index // (3**position)) % 3


def build_group_tables(
    activations: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """The LUT23 tables for one 128-weight group, in float32.

    Returns ``(L2, L3, tail)`` with shapes ``(25, 9)``, ``(25, 27)`` and ``(27,)``.
    """
    x = np.asarray(activations, dtype=np.float32)
    if x.shape != (BLOCK_SIZE,):
        raise FormatError(f"expected {BLOCK_SIZE} activations, got shape {x.shape}")
    slots = x[:TAIL_FIRST_COLUMN].reshape(FULL_CODE_SLOTS, TRITS_PER_SLOT)

    l2_digits = np.stack([_digit(L2_ENTRIES, 0), _digit(L2_ENTRIES, 1)]).astype(np.float32) - 1.0
    l3_digits = np.stack(
        [_digit(L3_ENTRIES, 0), _digit(L3_ENTRIES, 1), _digit(L3_ENTRIES, 2)]
    ).astype(np.float32) - 1.0
    tail_digits = np.stack(
        [_digit(TAIL_ENTRIES, 0), _digit(TAIL_ENTRIES, 1), _digit(TAIL_ENTRIES, 2)]
    ).astype(np.float32) - 1.0

    l2 = (slots[:, 0:1] * l2_digits[0] + slots[:, 1:2] * l2_digits[1]).astype(np.float32)
    l3 = (
        slots[:, 2:3] * l3_digits[0]
        + slots[:, 3:4] * l3_digits[1]
        + slots[:, 4:5] * l3_digits[2]
    ).astype(np.float32)
    tail_x = x[TAIL_FIRST_COLUMN:]
    tail = (
        tail_x[0] * tail_digits[0] + tail_x[1] * tail_digits[1] + tail_x[2] * tail_digits[2]
    ).astype(np.float32)
    return l2, l3, tail


def _fma32(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> np.ndarray:
    """Correctly rounded float32 ``a * b + c``.

    Doing the arithmetic in float64 and rounding once to float32 reproduces a
    hardware fma exactly for float32 inputs: the product needs at most 48 bits,
    and 53 >= 2 * 24 + 2, so the double rounding is provably harmless. This
    matters because the kernel applies each group's scale with a single fma.
    """
    return (a.astype(np.float64) * b.astype(np.float64) + c.astype(np.float64)).astype(np.float32)


def _check_inputs(
    codes: np.ndarray, scales: np.ndarray, x: np.ndarray, *, layout: PackedTensorLayout
) -> np.ndarray:
    if codes.shape != layout.codes_shape:
        raise FormatError(f"codes shape {codes.shape} does not match {layout.codes_shape}")
    if scales.shape != layout.scales_shape:
        raise FormatError(f"scales shape {scales.shape} does not match {layout.scales_shape}")
    activation = np.asarray(x)
    if activation.ndim != 2 or activation.shape[1] != layout.columns:
        raise FormatError(
            f"activations must be (batch, {layout.columns}), got shape {activation.shape}"
        )
    if activation.shape[0] <= 0:
        raise FormatError("activations have an empty batch dimension")
    return activation


def lut23_matmul(
    codes: np.ndarray, scales: np.ndarray, x: np.ndarray, *, layout: PackedTensorLayout
) -> np.ndarray:
    """Reference #2: table-driven, in the Metal kernel's exact summation order."""
    activation = _check_inputs(codes, scales, x, layout=layout)
    batch = activation.shape[0]
    rows = layout.rows
    scale_values = scale_bits_to_float(scales, dtype=np.float32).reshape(
        layout.tiles, layout.groups_per_row, layout.tile
    )

    # Row r lives at tile r // tile, position r % tile, so moving the slot axis
    # in front and flattening the two tile axes yields plain row order.
    full = codes[:, :, :FULL_CODE_SLOTS, :].transpose(2, 1, 0, 3).reshape(
        FULL_CODE_SLOTS, layout.groups_per_row, rows
    )
    tail_codes = codes[:, :, FULL_CODE_SLOTS, :].transpose(1, 0, 2).reshape(
        layout.groups_per_row, rows
    )
    high = (full.astype(np.uint32) * np.uint32(57)) >> np.uint32(9)
    low = full.astype(np.uint32) - np.uint32(L2_ENTRIES) * high
    group_scales = scale_values.transpose(1, 0, 2).reshape(layout.groups_per_row, rows)

    out = np.zeros((batch, rows), dtype=np.float32)
    for sample in range(batch):
        accumulator = np.zeros(rows, dtype=np.float32)
        for group in range(layout.groups_per_row):
            begin = group * BLOCK_SIZE
            l2, l3, tail = build_group_tables(activation[sample, begin : begin + BLOCK_SIZE])
            total = np.zeros(rows, dtype=np.float32)
            for slot in range(FULL_CODE_SLOTS):
                total += l2[slot][low[slot, group]] + l3[slot][high[slot, group]]
            total += tail[tail_codes[group]]
            accumulator = _fma32(group_scales[group], total, accumulator)
        out[sample] = accumulator
    return out


def decode_reordered_weights(
    codes: np.ndarray, scales: np.ndarray, *, layout: PackedTensorLayout, dtype: np.dtype
) -> np.ndarray:
    """Dense ``(rows, columns)`` weights, for the oracle and for parity checks."""
    if layout.rows * layout.columns > MAX_ORACLE_WEIGHTS:
        raise FormatError(
            f"refusing to densify {layout.rows}x{layout.columns} weights; the oracle is "
            f"capped at {MAX_ORACLE_WEIGHTS} elements"
        )
    blocks = unpack_blocks(codes, scales, layout=layout)
    scale_bits, trits = decode_tq1_blocks(blocks.reshape(-1, blocks.shape[-1]), validate=True)
    weights = trits.astype(np.int8).astype(dtype) - dtype.type(1)
    scale_values = np.ascontiguousarray(scale_bits).view("<f2").astype(dtype).reshape(-1, 1)
    return (weights * scale_values).reshape(layout.rows, layout.columns)


def oracle_matmul(
    codes: np.ndarray, scales: np.ndarray, x: np.ndarray, *, layout: PackedTensorLayout
) -> np.ndarray:
    """Reference #3: dense float64 matmul, the exactness oracle."""
    activation = _check_inputs(codes, scales, x, layout=layout)
    weights = decode_reordered_weights(codes, scales, layout=layout, dtype=np.dtype(np.float64))
    return activation.astype(np.float64) @ weights.T


def restored_matmul(
    codes: np.ndarray, scales: np.ndarray, x: np.ndarray, *, layout: PackedTensorLayout
) -> np.ndarray:
    """Reference #1: invert the permutation, then the frozen decode-then-dot GEMV."""
    activation = _check_inputs(codes, scales, x, layout=layout)
    blocks = unpack_blocks(codes, scales, layout=layout)
    return np.stack(
        [
            tq1_g128_reference_gemv(
                blocks, activation[sample], rows=layout.rows, columns=layout.columns
            )
            for sample in range(activation.shape[0])
        ]
    )


def get_rows_reference(
    codes: np.ndarray,
    scales: np.ndarray,
    indices: np.ndarray,
    *,
    layout: PackedTensorLayout,
    dtype: np.dtype,
) -> np.ndarray:
    """Reference for the embedding kernel: decode only the requested rows."""
    token_ids = np.asarray(indices)
    if token_ids.ndim != 1:
        raise FormatError(f"indices must be 1-D, got shape {token_ids.shape}")
    if token_ids.size and (token_ids.min() < 0 or token_ids.max() >= layout.rows):
        raise FormatError(f"indices fall outside [0, {layout.rows})")

    tile_index = token_ids // layout.tile
    row_in_tile = token_ids % layout.tile
    selected = codes[tile_index, :, :, row_in_tile]
    if selected.shape != (token_ids.size, layout.groups_per_row, CODE_SLOTS):
        raise FormatError(f"row gather produced shape {selected.shape}")
    gathered_scales = np.ascontiguousarray(scales[tile_index, :, row_in_tile])
    scale_values = gathered_scales.view(np.float16).astype(dtype)
    scale_values = scale_values.reshape(token_ids.size, layout.groups_per_row, 1)

    trits = np.zeros((token_ids.size, layout.groups_per_row, BLOCK_SIZE), dtype=np.uint8)
    for slot in range(FULL_CODE_SLOTS):
        byte = selected[:, :, slot]
        for digit in range(TRITS_PER_SLOT):
            trits[:, :, slot * TRITS_PER_SLOT + digit] = (byte // (3**digit)) % 3
    tail_byte = selected[:, :, FULL_CODE_SLOTS]
    for digit in range(BLOCK_SIZE - TAIL_FIRST_COLUMN):
        trits[:, :, TAIL_FIRST_COLUMN + digit] = (tail_byte // (3**digit)) % 3

    weights = trits.astype(np.int8).astype(dtype) - dtype.type(1)
    return (weights * scale_values).reshape(token_ids.size, layout.columns)
