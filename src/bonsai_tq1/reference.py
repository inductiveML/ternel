from __future__ import annotations

import numpy as np
import numpy.typing as npt

from .format import BLOCK_BYTES, BLOCK_SIZE, SOURCE_BLOCK_BYTES, decode_q2_blocks, decode_tq1_blocks


def _scales_to_float(scales: np.ndarray) -> np.ndarray:
    contiguous = np.ascontiguousarray(scales)
    return contiguous.view("<f2").reshape(-1).astype(np.float32)


def _reference_gemv(
    blocks: np.ndarray,
    activation: np.ndarray,
    *,
    rows: int,
    columns: int,
    block_bytes: int,
    decoder,
) -> np.ndarray:
    if columns % BLOCK_SIZE:
        raise ValueError("columns must be divisible by 128")
    groups_per_row = columns // BLOCK_SIZE
    if blocks.size != rows * groups_per_row * block_bytes:
        raise ValueError("packed matrix byte count does not match shape")
    x = np.asarray(activation, dtype=np.float32)
    if x.shape != (columns,):
        raise ValueError(f"expected activation shape {(columns,)}, got {x.shape}")
    matrix_blocks = np.asarray(blocks, dtype=np.uint8).reshape(rows, groups_per_row, block_bytes)
    output = np.zeros(rows, dtype=np.float32)
    for row in range(rows):
        scales, codes = decoder(matrix_blocks[row])
        weights = codes.astype(np.int8) - 1
        group_x = x.reshape(groups_per_row, BLOCK_SIZE)
        dot = np.sum(weights.astype(np.float32) * group_x, axis=1, dtype=np.float32)
        output[row] = np.sum(dot * _scales_to_float(scales), dtype=np.float32)
    return output


def q2_g128_reference_gemv(
    blocks: npt.ArrayLike, activation: npt.ArrayLike, *, rows: int, columns: int
) -> np.ndarray:
    return _reference_gemv(
        np.asarray(blocks),
        np.asarray(activation),
        rows=rows,
        columns=columns,
        block_bytes=SOURCE_BLOCK_BYTES,
        decoder=decode_q2_blocks,
    )


def tq1_g128_reference_gemv(
    blocks: npt.ArrayLike, activation: npt.ArrayLike, *, rows: int, columns: int
) -> np.ndarray:
    return _reference_gemv(
        np.asarray(blocks),
        np.asarray(activation),
        rows=rows,
        columns=columns,
        block_bytes=BLOCK_BYTES,
        decoder=decode_tq1_blocks,
    )

