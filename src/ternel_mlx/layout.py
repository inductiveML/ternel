"""Storage layout for the MLX distribution of TQ1_G128.

The frozen CUDA sidecar reorders every tensor with a fixed 256-row tile and
zero-pads the final tile. That padding is harmless on a sidecar written once,
but on Apple unified memory it is model-sized dead weight and it breaks the
claim that the artifact costs exactly 28 bytes per 128 weights.

This module therefore picks the tile per tensor. Every row count in Ternary
Bonsai 27B is either divisible by 256 or is one of the two 48-row SSM gates, so
choosing ``256`` when it divides and ``rows`` otherwise leaves **zero** padded
rows across the whole model while keeping the 256-wide tile that the Metal
threadgroup geometry is built around.

The reorder itself is not reimplemented here: ``lut23_reorder`` already does it
and its inverse, and both now take the tile explicitly.
"""

from __future__ import annotations

from dataclasses import dataclass

from bonsai_tq1.format import BLOCK_BYTES, BLOCK_SIZE, FormatError
from bonsai_tq1.lut23_reorder import CODE_SLOTS, M_TILE

from . import LAYOUT_NAME, LAYOUT_VERSION

# The tile the Metal kernels are built around: 256 rows, one row per thread in
# a 256-thread threadgroup. Inherited from the frozen CUDA layout so the two
# reorders agree byte-for-byte whenever the row count allows it.
PREFERRED_TILE = M_TILE

# Bytes of the 28-byte block that are trit codes; the other two are the raw
# little-endian FP16 scale, which the layout stores separately as uint16.
SCALE_BYTES = BLOCK_BYTES - CODE_SLOTS

# Names of the two arrays each packed tensor contributes to the safetensors file.
CODES_SUFFIX = "codes"
SCALES_SUFFIX = "scales"


def select_tile(rows: int) -> int:
    """Pick the row tile for a tensor so that no row is ever padded.

    ``PREFERRED_TILE`` when it divides ``rows`` exactly, otherwise ``rows``
    itself, which trivially divides and yields a single tile.
    """
    if rows <= 0:
        raise FormatError(f"row count must be positive, got {rows}")
    if rows % PREFERRED_TILE == 0:
        return PREFERRED_TILE
    return rows


@dataclass(frozen=True)
class PackedTensorLayout:
    """Shape and byte accounting for one tensor in the packed MLX artifact.

    ``rows`` is the output dimension of the matmul (MLX row-major convention),
    ``columns`` the reduction dimension. Both come from the logical weight, not
    from the GGUF transposition.
    """

    rows: int
    columns: int
    tile: int

    def __post_init__(self) -> None:
        if self.rows <= 0 or self.columns <= 0:
            raise FormatError(
                f"tensor dimensions must be positive, got {self.rows}x{self.columns}"
            )
        if self.columns % BLOCK_SIZE:
            raise FormatError(
                f"{self.columns} columns is not a multiple of the {BLOCK_SIZE}-weight group"
            )
        if self.tile <= 0:
            raise FormatError(f"tile must be positive, got {self.tile}")
        if self.rows % self.tile:
            raise FormatError(
                f"tile {self.tile} does not divide {self.rows} rows; the MLX layout "
                "does not pad rows"
            )

    @classmethod
    def for_tensor(cls, rows: int, columns: int) -> "PackedTensorLayout":
        return cls(rows=rows, columns=columns, tile=select_tile(rows))

    @property
    def groups_per_row(self) -> int:
        return self.columns // BLOCK_SIZE

    @property
    def tiles(self) -> int:
        return self.rows // self.tile

    @property
    def groups(self) -> int:
        return self.rows * self.groups_per_row

    @property
    def codes_shape(self) -> tuple[int, int, int, int]:
        """``[tile][group][code_slot][row_in_tile]``, uint8."""
        return (self.tiles, self.groups_per_row, CODE_SLOTS, self.tile)

    @property
    def scales_shape(self) -> tuple[int, int, int]:
        """``[tile][group][row_in_tile]``, uint16 holding raw FP16 bits."""
        return (self.tiles, self.groups_per_row, self.tile)

    @property
    def codes_bytes(self) -> int:
        return self.groups * CODE_SLOTS

    @property
    def scales_bytes(self) -> int:
        return self.groups * SCALE_BYTES

    @property
    def payload_bytes(self) -> int:
        return self.codes_bytes + self.scales_bytes

    @property
    def canonical_bytes(self) -> int:
        """What the tensor must cost: exactly 28 bytes per 128 weights."""
        return self.groups * BLOCK_BYTES

    @property
    def bits_per_weight(self) -> float:
        return self.payload_bytes * 8 / (self.rows * self.columns)

    def to_json(self) -> dict[str, object]:
        return {
            "layout_name": LAYOUT_NAME,
            "layout_version": LAYOUT_VERSION,
            "rows": self.rows,
            "columns": self.columns,
            "tile": self.tile,
            "tiles": self.tiles,
            "groups_per_row": self.groups_per_row,
            "groups": self.groups,
            "codes_shape": list(self.codes_shape),
            "scales_shape": list(self.scales_shape),
            "codes_bytes": self.codes_bytes,
            "scales_bytes": self.scales_bytes,
            "payload_bytes": self.payload_bytes,
            "bits_per_weight": self.bits_per_weight,
        }
