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


@dataclass(frozen=True)
class ModelLinearShape:
    """One distinct linear weight shape, and how much of a token it accounts for.

    ``dispatches_per_token`` is what makes this a census rather than a list: a
    shape's cost to a user is its own cost times how often a decoded token
    issues it, and the two vary independently. ``mlp.gate_proj`` is dispatched
    128 times a token; ``lm_head``, which is 14 times the weights, once.
    """

    label: str
    rows: int
    columns: int
    dispatches_per_token: int
    modules: tuple[str, ...]

    def layout(self) -> "PackedTensorLayout":
        return PackedTensorLayout.for_tensor(self.rows, self.columns)


# Every distinct linear shape in Ternary Bonsai 27B, verified against the
# artifact manifest's 498 quantised tensors. Shapes that repeat under more than
# one module name are folded onto one entry, ``modules`` naming all of them --
# ``self_attn.o_proj`` is 5120x6144, the same shape as ``linear_attn.out_proj``,
# and measuring it twice under two names measures nothing new.
#
# The counts are what 48 linear-attention layers of eight matmuls, 16
# full-attention layers of seven, and one output head come to. ``embed_tokens``
# is not here: it shares ``lm_head``'s shape but is gathered, never multiplied.
MODEL_LINEAR_SHAPES: tuple[ModelLinearShape, ...] = (
    ModelLinearShape("linear_attn.in_proj_a", 48, 5120, 96,
                     ("linear_attn.in_proj_a", "linear_attn.in_proj_b")),
    ModelLinearShape("self_attn.k_proj", 1024, 5120, 32,
                     ("self_attn.k_proj", "self_attn.v_proj")),
    ModelLinearShape("linear_attn.out_proj", 5120, 6144, 64,
                     ("linear_attn.out_proj", "self_attn.o_proj")),
    ModelLinearShape("mlp.down_proj", 5120, 17408, 64, ("mlp.down_proj",)),
    ModelLinearShape("linear_attn.in_proj_z", 6144, 5120, 48, ("linear_attn.in_proj_z",)),
    ModelLinearShape("linear_attn.in_proj_qkv", 10240, 5120, 48, ("linear_attn.in_proj_qkv",)),
    ModelLinearShape("self_attn.q_proj", 12288, 5120, 16, ("self_attn.q_proj",)),
    ModelLinearShape("mlp.gate_proj", 17408, 5120, 128, ("mlp.gate_proj", "mlp.up_proj")),
    ModelLinearShape("lm_head", 248320, 5120, 1, ("lm_head",)),
)

# Matmuls a decoded token issues, which is what the counts above must sum to.
DISPATCHES_PER_TOKEN = 497

if sum(shape.dispatches_per_token for shape in MODEL_LINEAR_SHAPES) != DISPATCHES_PER_TOKEN:
    raise FormatError("the shape census no longer accounts for a whole decoded token")
