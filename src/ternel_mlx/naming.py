"""Where each GGUF tensor goes in the mlx-lm checkpoint: its name and its order.

Ternary Bonsai 27B is published twice: as a llama.cpp GGUF, whose tensors are
called ``blk.7.ffn_down.weight``, and as an mlx-lm checkpoint, whose tensors are
called ``language_model.model.layers.7.mlp.down_proj.weight``. Ternel converts
the first into the second, so it needs the correspondence between them, and a
wrong entry here would not raise: it would silently wire a layer's gate
projection into its up projection and produce fluent nonsense.

The map is therefore built from a small structural table and then *checked three
ways* before any byte is written:

* **Counts.** Each GGUF kind must appear as many times as its mlx-lm partner,
  which pins the twelve per-layer kinds against the 48/16/64 layer split.
* **Shapes.** GGUF stores ``(columns, rows)`` and mlx-lm ``(rows, columns)``, so
  a transposed pairing is caught even when the counts agree.
* **Values.** Counts and shapes cannot separate ``ssm_alpha`` from ``ssm_beta``
  -- 48 tensors of 5120x48 each, either way round. Only the weights themselves
  can, and because the official mlx-lm checkpoint turns out to store *exact*
  ternary, the comparison is bit-exact rather than statistical. That check lives
  in :mod:`ternel_mlx.crosscheck`; this module supplies the candidate map and
  the structural half of the evidence.

The third check is what turned up the second half of the correspondence. A name
is not enough: the two publications also **order the linear-attention value
heads differently**. There are ``linear_num_value_heads`` of them grouped over
``linear_num_key_heads``, and llama.cpp lays them out repeat-major while mlx-lm
lays them out key-head-major, so mlx-lm's value head ``j`` is llama.cpp's value
head ``(j % repeat) * key_heads + j // repeat``. Eight tensor kinds carry a
value-head axis and must be reordered; the rest are byte-identical as they
stand. This is measured, not assumed -- :mod:`ternel_mlx.crosscheck` proves it
against all 26,893,352,960 weights.

Nothing here is specific to one revision of the model: the tables are keyed by
the block-local part of the name, the layer indices come from whatever the GGUF
actually contains, and the head counts come from the checkpoint's own config.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum

import numpy as np

from bonsai_tq1.format import FormatError
from bonsai_tq1.gguf_utils import TensorInfo

# Every GGUF tensor is either global or lives in a numbered block.
BLOCK_PATTERN = re.compile(r"^blk\.(\d+)\.(.+)$")

# mlx-lm nests the whole text model under this prefix; the vision tower, which
# Ternel does not ship, sits beside it under ``vision_tower.``.
TEXT_PREFIX = "language_model."
LAYER_PREFIX = TEXT_PREFIX + "model.layers."

# Block-local GGUF name -> module path under ``language_model.model.layers.N.``,
# for the Q2_0 matmul weights. The mlx-lm side is a *module* path, not a tensor
# name: a packed module contributes ``.codes`` and ``.scales`` rather than a
# single ``.weight``.
QUANTISED_BLOCK_MODULES: dict[str, str] = {
    "attn_qkv.weight": "linear_attn.in_proj_qkv",
    "attn_gate.weight": "linear_attn.in_proj_z",
    "ssm_beta.weight": "linear_attn.in_proj_b",
    "ssm_alpha.weight": "linear_attn.in_proj_a",
    "ssm_out.weight": "linear_attn.out_proj",
    "attn_q.weight": "self_attn.q_proj",
    "attn_k.weight": "self_attn.k_proj",
    "attn_v.weight": "self_attn.v_proj",
    "attn_output.weight": "self_attn.o_proj",
    "ffn_gate.weight": "mlp.gate_proj",
    "ffn_up.weight": "mlp.up_proj",
    "ffn_down.weight": "mlp.down_proj",
}

# The two Q2_0 tensors that live outside any block.
QUANTISED_GLOBAL_MODULES: dict[str, str] = {
    "token_embd.weight": TEXT_PREFIX + "model.embed_tokens",
    "output.weight": TEXT_PREFIX + "lm_head",
}

# Block-local GGUF name -> parameter name under ``language_model.model.layers.N.``
# for the tensors that are not quantised. These keep a single ``.weight``-style
# name on both sides, so the mlx-lm side is a full parameter name.
PLAIN_BLOCK_PARAMETERS: dict[str, str] = {
    "attn_norm.weight": "input_layernorm.weight",
    "post_attention_norm.weight": "post_attention_layernorm.weight",
    "attn_q_norm.weight": "self_attn.q_norm.weight",
    "attn_k_norm.weight": "self_attn.k_norm.weight",
    "ssm_a": "linear_attn.A_log",
    "ssm_conv1d.weight": "linear_attn.conv1d.weight",
    "ssm_dt.bias": "linear_attn.dt_bias",
    "ssm_norm.weight": "linear_attn.norm.weight",
}

PLAIN_GLOBAL_PARAMETERS: dict[str, str] = {
    "output_norm.weight": TEXT_PREFIX + "model.norm.weight",
}

# Suffixes a packed module contributes to the artifact.
CODES_PARAMETER = "codes"
SCALES_PARAMETER = "scales"

GGUF_QUANTISED_TYPE = "Q2_0"


class HeadAxis(Enum):
    """Which axis of a tensor carries the value-head structure."""

    ROWS = "rows"
    COLUMNS = "columns"


class HeadUnit(Enum):
    """How many slots along that axis one value head occupies."""

    HEAD = "head"  # one slot per head: a per-head scalar
    CHANNEL = "channel"  # ``value_head_dim`` slots per head: a per-head vector


@dataclass(frozen=True)
class ValueHeadReorder:
    """The value-head block of one tensor kind, as an axis and a unit.

    The block is always the *last* ``value_heads * unit`` slots of the axis:
    tensors that are entirely value-head structured take the whole axis, and
    ``attn_qkv``/``ssm_conv1d``, whose leading slots belong to the query and key
    heads, take a suffix. One rule covers both.
    """

    axis: HeadAxis
    unit: HeadUnit


# The tensor kinds whose layout differs between the two publications. Everything
# absent from this table is already in mlx-lm order.
VALUE_HEAD_REORDER: dict[str, ValueHeadReorder] = {
    "attn_qkv.weight": ValueHeadReorder(HeadAxis.ROWS, HeadUnit.CHANNEL),
    "attn_gate.weight": ValueHeadReorder(HeadAxis.ROWS, HeadUnit.CHANNEL),
    "ssm_out.weight": ValueHeadReorder(HeadAxis.COLUMNS, HeadUnit.CHANNEL),
    "ssm_alpha.weight": ValueHeadReorder(HeadAxis.ROWS, HeadUnit.HEAD),
    "ssm_beta.weight": ValueHeadReorder(HeadAxis.ROWS, HeadUnit.HEAD),
    "ssm_conv1d.weight": ValueHeadReorder(HeadAxis.ROWS, HeadUnit.CHANNEL),
    "ssm_dt.bias": ValueHeadReorder(HeadAxis.ROWS, HeadUnit.HEAD),
    "ssm_a": ValueHeadReorder(HeadAxis.ROWS, HeadUnit.HEAD),
}

_KNOWN_KINDS = (
    QUANTISED_BLOCK_MODULES.keys()
    | PLAIN_BLOCK_PARAMETERS.keys()
    | QUANTISED_GLOBAL_MODULES.keys()
    | PLAIN_GLOBAL_PARAMETERS.keys()
)
_UNKNOWN_REORDER_KINDS = sorted(VALUE_HEAD_REORDER.keys() - _KNOWN_KINDS)
if _UNKNOWN_REORDER_KINDS:
    raise FormatError(f"reorder table names tensor kinds that do not exist: {_UNKNOWN_REORDER_KINDS}")

# Config keys the head layout is read from, so nothing here hard-codes 16/48/128.
KEY_HEADS_KEY = "linear_num_key_heads"
VALUE_HEADS_KEY = "linear_num_value_heads"
VALUE_HEAD_DIM_KEY = "linear_value_head_dim"


@dataclass(frozen=True)
class HeadLayout:
    """The linear-attention head geometry, read from the checkpoint's config."""

    key_heads: int
    value_heads: int
    value_head_dim: int

    def __post_init__(self) -> None:
        if self.key_heads <= 0 or self.value_heads <= 0 or self.value_head_dim <= 0:
            raise FormatError(
                f"head layout must be positive, got key_heads={self.key_heads}, "
                f"value_heads={self.value_heads}, value_head_dim={self.value_head_dim}"
            )
        if self.value_heads % self.key_heads:
            raise FormatError(
                f"{self.value_heads} value heads do not divide evenly over "
                f"{self.key_heads} key heads"
            )

    @classmethod
    def from_text_config(cls, text_config: dict[str, object]) -> HeadLayout:
        values: list[int] = []
        for key in (KEY_HEADS_KEY, VALUE_HEADS_KEY, VALUE_HEAD_DIM_KEY):
            found = text_config.get(key)
            if not isinstance(found, int) or isinstance(found, bool):
                raise FormatError(f"config field {key!r} is {found!r}, expected an integer")
            values.append(found)
        return cls(key_heads=values[0], value_heads=values[1], value_head_dim=values[2])

    @property
    def repeat(self) -> int:
        """Value heads per key head."""
        return self.value_heads // self.key_heads

    @property
    def value_dim(self) -> int:
        return self.value_heads * self.value_head_dim

    def slots_per_head(self, unit: HeadUnit) -> int:
        return 1 if unit is HeadUnit.HEAD else self.value_head_dim

    def block_length(self, unit: HeadUnit) -> int:
        return self.value_heads * self.slots_per_head(unit)

    def head_order(self) -> np.ndarray:
        """For each mlx-lm value head, the GGUF value head holding it.

        mlx-lm walks key head then repeat; llama.cpp walks repeat then key head.
        The mapping is its own inverse only when ``repeat == key_heads``, so the
        direction matters and is stated here once: index by the mlx-lm position.
        """
        heads = np.arange(self.value_heads, dtype=np.int64)
        return (heads % self.repeat) * self.key_heads + heads // self.repeat

    def _order(self, length: int, per_head: int) -> np.ndarray:
        block = self.value_heads * per_head
        if length < block:
            raise FormatError(
                f"axis of {length} slots is shorter than the {block}-slot value-head block"
            )
        within = self.head_order()[:, None] * per_head + np.arange(per_head, dtype=np.int64)
        order = np.arange(length, dtype=np.int64)
        order[length - block :] = within.reshape(-1) + (length - block)
        return order

    def source_order(self, length: int, reorder: ValueHeadReorder) -> np.ndarray:
        """For each mlx-lm slot of an axis of ``length``, the GGUF slot holding it."""
        return self._order(length, self.slots_per_head(reorder.unit))

    def group_source_order(
        self, groups: int, reorder: ValueHeadReorder, *, group_size: int
    ) -> np.ndarray:
        """``source_order`` for an axis measured in quantisation groups, not weights.

        A packed tensor's reduction axis is addressable only in whole groups, so
        reordering it is possible exactly when a value head spans a whole number
        of them. Anything else would mean re-encoding, and is refused rather than
        approximated.
        """
        if reorder.unit is not HeadUnit.CHANNEL:
            raise FormatError(f"a {reorder.unit.value}-unit axis has no group granularity")
        if group_size <= 0 or self.value_head_dim % group_size:
            raise FormatError(
                f"a value head of {self.value_head_dim} channels is not a whole number of "
                f"{group_size}-weight groups, so the reordering cannot move whole groups"
            )
        return self._order(groups, self.value_head_dim // group_size)


@dataclass(frozen=True)
class MappedTensor:
    """One GGUF tensor and where it goes in the mlx-lm checkpoint.

    ``rows`` and ``columns`` are stated in the mlx-lm convention -- ``rows`` is
    the matmul's output dimension -- which is the reverse of the GGUF shape.
    """

    gguf: TensorInfo
    mlx_name: str
    quantised: bool

    @property
    def rows(self) -> int:
        return self.gguf.shape[1]

    @property
    def columns(self) -> int:
        return self.gguf.shape[0]

    @property
    def kind(self) -> str:
        """The part of the GGUF name that is shared across layers."""
        match = BLOCK_PATTERN.match(self.gguf.name)
        return match.group(2) if match is not None else self.gguf.name

    @property
    def layer(self) -> int | None:
        """The block this tensor belongs to, or None for the global tensors."""
        match = BLOCK_PATTERN.match(self.gguf.name)
        return int(match.group(1)) if match is not None else None

    @property
    def reorder(self) -> ValueHeadReorder | None:
        """How this tensor's value heads are reordered, or None if they are not."""
        return VALUE_HEAD_REORDER.get(self.kind)

    @property
    def codes_name(self) -> str:
        if not self.quantised:
            raise FormatError(f"{self.gguf.name} is not quantised and has no codes array")
        return f"{self.mlx_name}.{CODES_PARAMETER}"

    @property
    def scales_name(self) -> str:
        if not self.quantised:
            raise FormatError(f"{self.gguf.name} is not quantised and has no scales array")
        return f"{self.mlx_name}.{SCALES_PARAMETER}"

    @property
    def parameter_names(self) -> tuple[str, ...]:
        """The artifact entries this tensor becomes."""
        if self.quantised:
            return (self.codes_name, self.scales_name)
        return (self.mlx_name,)


def map_gguf_name(name: str) -> tuple[str, bool]:
    """``blk.7.ffn_down.weight`` -> ``(...layers.7.mlp.down_proj, True)``.

    The boolean says whether the mlx-lm side is a packed *module* (True, so the
    caller appends ``.codes``/``.scales``) or a plain parameter (False).
    """
    if name in QUANTISED_GLOBAL_MODULES:
        return QUANTISED_GLOBAL_MODULES[name], True
    if name in PLAIN_GLOBAL_PARAMETERS:
        return PLAIN_GLOBAL_PARAMETERS[name], False
    match = BLOCK_PATTERN.match(name)
    if match is None:
        raise FormatError(f"GGUF tensor {name!r} is neither a known global nor a block tensor")
    index, local = match.group(1), match.group(2)
    if local in QUANTISED_BLOCK_MODULES:
        return f"{LAYER_PREFIX}{index}.{QUANTISED_BLOCK_MODULES[local]}", True
    if local in PLAIN_BLOCK_PARAMETERS:
        return f"{LAYER_PREFIX}{index}.{PLAIN_BLOCK_PARAMETERS[local]}", False
    raise FormatError(f"GGUF tensor {name!r} has no mlx-lm counterpart in the name map")


def build_name_map(tensors: list[TensorInfo]) -> list[MappedTensor]:
    """Map every GGUF tensor, in GGUF order, refusing anything ambiguous.

    Ordering is preserved because the converter streams the sidecar in the same
    order and a stable order makes the artifact reproducible byte for byte.
    """
    mapped: list[MappedTensor] = []
    seen: dict[str, str] = {}
    for tensor in tensors:
        mlx_name, quantised = map_gguf_name(tensor.name)
        if quantised != (tensor.tensor_type == GGUF_QUANTISED_TYPE):
            raise FormatError(
                f"{tensor.name} is stored as {tensor.tensor_type} but the name map treats it "
                f"as {'quantised' if quantised else 'plain'}"
            )
        if len(tensor.shape) != (2 if quantised else len(tensor.shape)):
            raise FormatError(f"{tensor.name} has shape {tensor.shape}, expected two dimensions")
        for parameter in MappedTensor(tensor, mlx_name, quantised).parameter_names:
            previous = seen.get(parameter)
            if previous is not None:
                raise FormatError(
                    f"name map is not injective: {previous} and {tensor.name} both produce "
                    f"{parameter}"
                )
            seen[parameter] = tensor.name
        mapped.append(MappedTensor(tensor, mlx_name, quantised))
    return mapped


def check_against_graph(mapped: list[MappedTensor], parameter_names: set[str]) -> None:
    """Refuse a map that does not exactly cover the parameters the graph declares.

    ``mlx_lm.utils.load_model`` calls ``load_weights(..., strict=True)``, so a
    missing or extra entry is a load-time failure. Catching it here names the
    tensor responsible instead of printing a diff of 1349 strings.
    """
    produced = {parameter for tensor in mapped for parameter in tensor.parameter_names}
    missing = sorted(parameter_names - produced)
    extra = sorted(produced - parameter_names)
    if missing or extra:
        raise FormatError(
            f"name map does not match the graph: {len(missing)} parameters unproduced "
            f"(e.g. {missing[:3]}), {len(extra)} produced but unexpected (e.g. {extra[:3]})"
        )


def check_against_baseline(mapped: list[MappedTensor], baseline_names: set[str]) -> None:
    """Refuse a map whose quantised side disagrees with the official checkpoint.

    The official mlx-lm distribution names a quantised tensor's parts
    ``<module>.weight``, ``<module>.scales`` and ``<module>.biases``. Ternel
    stores different arrays under the same module path, so the *module* paths
    must agree even though the leaf names do not.
    """
    baseline_modules = {
        name[: -len(".scales")] for name in baseline_names if name.endswith(".scales")
    }
    produced = {tensor.mlx_name for tensor in mapped if tensor.quantised}
    missing = sorted(baseline_modules - produced)
    extra = sorted(produced - baseline_modules)
    if missing or extra:
        raise FormatError(
            f"quantised name map disagrees with the official checkpoint: {len(missing)} of its "
            f"modules unmapped (e.g. {missing[:3]}), {len(extra)} mapped but absent there "
            f"(e.g. {extra[:3]})"
        )
    plain_expected = {tensor.mlx_name for tensor in mapped if not tensor.quantised}
    plain_baseline = {
        name
        for name in baseline_names
        if name.startswith(TEXT_PREFIX)
        and not name.endswith((".scales", ".biases"))
        and name.rsplit(".", 1)[0] not in baseline_modules
    }
    missing = sorted(plain_baseline - plain_expected)
    extra = sorted(plain_expected - plain_baseline)
    if missing or extra:
        raise FormatError(
            f"plain name map disagrees with the official checkpoint: {len(missing)} of its "
            f"tensors unmapped (e.g. {missing[:3]}), {len(extra)} mapped but absent there "
            f"(e.g. {extra[:3]})"
        )
