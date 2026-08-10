"""Bridge between canonical TQ1_G128 blocks and the MLX tiled arrays.

The canonical block is the frozen 28-byte unit that every hash and every gate in
this repository is defined over. The MLX artifact stores exactly those bytes,
only permuted into code-position-major order so that a Metal threadgroup reads
one code slot across 256 consecutive rows as one coalesced load. Nothing is
recomputed, reinterpreted or re-rounded on the way through: the scale pair
becomes a ``uint16`` by pure byte reinterpretation, which is why the round trip
is exact by construction rather than by tolerance.
"""

from __future__ import annotations

import numpy as np

from bonsai_tq1.format import (
    BLOCK_BYTES,
    MAX_FULL_CODE_BYTE,
    MAX_TAIL_CODE_BYTE,
    FormatError,
)
from bonsai_tq1.lut23_reorder import FULL_CODE_SLOTS, reorder_tensor_arrays, restore_tensor_arrays

from .layout import PackedTensorLayout

SCALE_VIEW_DTYPE = np.dtype("<u2")


def validate_packed_codes(codes: np.ndarray) -> None:
    """Reject any byte that could index past the kernel's LUT23 tables.

    The kernel splits a regular byte as ``hi = (p * 57) >> 9``; an illegal
    ``p = 255`` gives ``hi = 28``, one past the 27-entry L3 table. The fast path
    carries no clamp, so this scan is what makes that safe, and it runs over
    every byte of every tensor at load time rather than sampling.
    """
    array = np.asarray(codes)
    if array.dtype != np.uint8:
        raise FormatError(f"codes must be uint8, got {array.dtype}")
    if array.ndim != 4:
        raise FormatError(f"codes must be 4-D (tiles, groups, slots, tile), got {array.shape}")
    invalid_full = int(np.count_nonzero(array[:, :, :FULL_CODE_SLOTS, :] > MAX_FULL_CODE_BYTE))
    invalid_tail = int(np.count_nonzero(array[:, :, FULL_CODE_SLOTS, :] > MAX_TAIL_CODE_BYTE))
    if invalid_full or invalid_tail:
        raise FormatError(
            f"packed codes contain out-of-range base-3 bytes: regular={invalid_full} "
            f"(max {MAX_FULL_CODE_BYTE}), tail={invalid_tail} (max {MAX_TAIL_CODE_BYTE})"
        )


def pack_blocks(blocks: np.ndarray, *, layout: PackedTensorLayout) -> tuple[np.ndarray, np.ndarray]:
    """Canonical ``(rows, groups, 28)`` blocks -> ``(codes uint8, scales uint16)``."""
    source = np.asarray(blocks)
    if source.dtype != np.uint8:
        raise FormatError(f"canonical blocks must be uint8, got {source.dtype}")
    expected = (layout.rows, layout.groups_per_row, BLOCK_BYTES)
    if source.shape != expected:
        raise FormatError(f"expected canonical blocks of shape {expected}, got {source.shape}")

    codes, scale_pairs = reorder_tensor_arrays(source, tile=layout.tile)
    if codes.shape != layout.codes_shape:
        raise FormatError(f"reorder produced codes of shape {codes.shape}, expected {layout.codes_shape}")
    scales = np.ascontiguousarray(scale_pairs).reshape(-1).view(SCALE_VIEW_DTYPE)
    return codes, scales.reshape(layout.scales_shape)


def unpack_blocks(codes: np.ndarray, scales: np.ndarray, *, layout: PackedTensorLayout) -> np.ndarray:
    """Inverse of :func:`pack_blocks`, back to canonical ``(rows, groups, 28)`` blocks."""
    code_array = np.asarray(codes)
    scale_array = np.asarray(scales)
    if code_array.dtype != np.uint8:
        raise FormatError(f"codes must be uint8, got {code_array.dtype}")
    if scale_array.dtype != SCALE_VIEW_DTYPE:
        raise FormatError(f"scales must be little-endian uint16, got {scale_array.dtype}")
    if code_array.shape != layout.codes_shape:
        raise FormatError(f"codes shape {code_array.shape} does not match {layout.codes_shape}")
    if scale_array.shape != layout.scales_shape:
        raise FormatError(f"scales shape {scale_array.shape} does not match {layout.scales_shape}")

    pairs = np.ascontiguousarray(scale_array).reshape(-1).view(np.uint8)
    pairs = pairs.reshape(*layout.scales_shape, 2)
    return restore_tensor_arrays(code_array, pairs, layout.rows, tile=layout.tile)


def scale_bits_to_float(scales: np.ndarray, *, dtype: np.dtype) -> np.ndarray:
    """Reinterpret raw FP16 scale bits as floats without rounding anything."""
    array = np.ascontiguousarray(scales)
    if array.dtype != SCALE_VIEW_DTYPE:
        raise FormatError(f"scales must be little-endian uint16, got {array.dtype}")
    return array.view(np.float16).astype(dtype)
