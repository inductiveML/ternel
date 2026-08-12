"""Prove the GGUF and the official mlx-lm checkpoint hold the same 27B model.

Ternel converts ``prism-ml/Ternary-Bonsai-27B-gguf`` into an mlx-lm checkpoint,
and takes the 353 non-quantised tensors from ``prism-ml/Ternary-Bonsai-27B-mlx-2bit``
so they arrive in mlx-lm's own convention. Both halves of that plan rest on a
claim that has to be *measured*: that the two publications are the same weights.

They turn out to be the same weights **bit for bit**. Stage A0 found the official
checkpoint stores exact ternary -- ``bias == -scale``, no fourth code -- so the
comparison is not statistical. For every one of the 498 quantised tensors this
module decodes the GGUF's Q2_0 blocks and the checkpoint's affine 2-bit words and
requires the ternary codes and the raw FP16 scale bits to agree exactly, over all
26,893,352,960 weights and 210,104,320 groups.

Two things make that comparison non-trivial, and both were found by running it:

* **The value heads are ordered differently.** llama.cpp lays the
  ``linear_num_value_heads`` heads out repeat-major, mlx-lm key-head-major. Eight
  tensor kinds carry that axis; see :mod:`ternel_mlx.naming`. Without the
  reordering, ``in_proj_a`` disagrees with ``ssm_alpha`` on 56% of its weights --
  and, tellingly, disagrees with ``ssm_beta`` on 65%, so the *names* were right
  all along and only the order was wrong.
* **A right answer must exclude the wrong ones.** Counts and shapes cannot tell
  ``ssm_alpha`` from ``ssm_beta``, nor ``ffn_gate`` from ``ffn_up``. So every
  tensor that shares its shape with another tensor of the same layer is also
  compared against that other tensor, and the check fails unless those
  comparisons *disagree*. A map that matches everything proves nothing.

The 353 plain tensors are compared the same way, after the value transform that
llama.cpp applied on the way out: ``ssm_a`` is ``-exp(A_log)``, so recovering
mlx-lm's ``A_log`` means ``log(-x)``. That inverse is exact in FP16 except for
the sign of zero, which is reported rather than hidden.

One divergence survives all of that, and it is the reason a bit-exact boolean is
not the only verdict here. Row 178519 of the embedding table -- one row of
248,320, and the only place in the model where mlx-lm's ``bias`` is not exactly
``-scale`` -- is a row both quantisers had to encode from a zero absmax, and each
picked a different degenerate answer: the GGUF a smallest-subnormal scale with
codes split by sign, mlx-lm twice that scale with a bias that recentres it. They
dequantise to weights at most ``1.19e-07`` apart -- two units in the last place
of the smallest FP16 subnormal, with identical row norms. So the comparison
records the *weight-space* gap alongside the bit counts, and the caller states
the gap it will accept rather than the tool inventing one.

Nothing in here uses MLX. The reader is a plain ``numpy.memmap`` over both files,
because a checkpoint written by MLX is poor evidence about itself.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterator

import numpy as np

from bonsai_tq1.format import (
    BLOCK_SIZE,
    SOURCE_BLOCK_BYTES,
    FormatError,
    decode_q2_blocks,
    iter_memmap_blocks,
    write_json_atomic,
)
from bonsai_tq1.gguf_utils import load_tensor_infos

from .baseline_fidelity import unpack_affine_codes
from .naming import (
    HeadAxis,
    HeadLayout,
    MappedTensor,
    ValueHeadReorder,
    build_name_map,
    check_against_baseline,
)
from .remote_safetensors import LocalSafetensors

SCHEMA_VERSION = 1

# FP16 bit patterns for the two zeros. The GGUF derives ``ssm_a = -exp(A_log)``,
# and inverting it with ``log(-x)`` cannot recover which zero ``A_log`` was.
POSITIVE_ZERO_BITS = np.uint16(0x0000)
NEGATIVE_ZERO_BITS = np.uint16(0x8000)

# The value transform each plain GGUF tensor needs to become its mlx-lm partner.
# Kinds absent from this table are stored identically on both sides.
PLAIN_VALUE_TRANSFORMS: dict[str, Callable[[np.ndarray], np.ndarray]] = {
    "ssm_a": lambda values: np.log(-values),
}


def transform_name(kind: str) -> str:
    return "log_negate" if kind in PLAIN_VALUE_TRANSFORMS else "identity"


def float16_bits(values: np.ndarray) -> np.ndarray:
    """The raw FP16 bit pattern of ``values``, rounded once from float64."""
    return np.ascontiguousarray(values.astype(np.float16)).view("<u2")


def zeros_only_difference(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    """Where two FP16 bit arrays differ only in the sign of a zero."""
    both_zero = ((left | NEGATIVE_ZERO_BITS) == NEGATIVE_ZERO_BITS) & (
        (right | NEGATIVE_ZERO_BITS) == NEGATIVE_ZERO_BITS
    )
    return (left != right) & both_zero


@dataclass(frozen=True)
class RowChunk:
    begin: int
    end: int

    @property
    def rows(self) -> int:
        return self.end - self.begin


def row_chunks(
    rows: int, columns: int, *, weights_per_chunk: int, unsplittable_from: int
) -> Iterator[RowChunk]:
    """Chunks of at most ``weights_per_chunk`` weights, in mlx-lm row order.

    A value-head reordering is a transpose of the whole block: no proper
    sub-range of it is self-contained. So the rows from ``unsplittable_from``
    onwards are always yielded as one chunk, however large, and only the rows
    before it are split to the budget. Pass ``rows`` to split everywhere.
    """
    if rows <= 0 or columns <= 0 or weights_per_chunk <= 0:
        raise ValueError("row chunking needs positive rows, columns and budget")
    if not 0 <= unsplittable_from <= rows:
        raise ValueError(f"unsplittable boundary {unsplittable_from} is outside {rows} rows")
    step = max(1, weights_per_chunk // columns)
    begin = 0
    while begin < unsplittable_from:
        end = min(begin + step, unsplittable_from)
        yield RowChunk(begin, end)
        begin = end
    if begin < rows:
        yield RowChunk(begin, rows)


def gguf_quantised_chunk(
    path: Path, tensor: MappedTensor, chunk: RowChunk
) -> tuple[np.ndarray, np.ndarray]:
    """``(scale_bits, codes)`` for one row range, straight out of the GGUF."""
    groups_per_row = tensor.columns // BLOCK_SIZE
    pieces = list(
        iter_memmap_blocks(
            path,
            offset=tensor.gguf.data_offset + chunk.begin * groups_per_row * SOURCE_BLOCK_BYTES,
            block_count=chunk.rows * groups_per_row,
            block_bytes=SOURCE_BLOCK_BYTES,
            chunk_groups=chunk.rows * groups_per_row,
        )
    )
    blocks = pieces[0] if len(pieces) == 1 else np.concatenate(pieces)
    scale_bytes, codes = decode_q2_blocks(blocks)
    return (
        scale_bytes.view("<u2").reshape(chunk.rows, groups_per_row),
        codes.reshape(chunk.rows, tensor.columns),
    )


def baseline_quantised_chunk(
    baseline: LocalSafetensors, module: str, columns: int, chunk: RowChunk
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """``(scale_bits, codes, bias_bits)`` for one row range of the official shard."""
    packed = np.ascontiguousarray(baseline.read_rows(f"{module}.weight", chunk.begin, chunk.end))
    scales = np.ascontiguousarray(baseline.read_rows(f"{module}.scales", chunk.begin, chunk.end))
    biases = np.ascontiguousarray(baseline.read_rows(f"{module}.biases", chunk.begin, chunk.end))
    return scales.view("<u2"), unpack_affine_codes(packed, columns=columns), biases.view("<u2")


def reordered_rows(chunk: RowChunk, order: np.ndarray) -> np.ndarray | None:
    """The GGUF rows that fill mlx-lm rows ``[chunk.begin, chunk.end)``.

    Returns ``None`` when the reordering pulls rows from outside the chunk, which
    tells the caller to widen its chunk rather than compare the wrong rows.
    """
    wanted = order[chunk.begin : chunk.end]
    if wanted.min() < chunk.begin or wanted.max() >= chunk.end:
        return None
    return wanted - chunk.begin


@dataclass(frozen=True)
class QuantisedComparison:
    gguf_name: str
    mlx_name: str
    rows: int
    columns: int
    groups: int
    reorder: str
    code_mismatches: int
    scale_mismatches: int
    bias_mismatches: int
    mismatched_rows: tuple[int, ...]
    max_code: int
    max_weight_difference: float

    @property
    def agrees(self) -> bool:
        """Whether the two encodings are identical bit for bit."""
        return (
            self.code_mismatches == 0
            and self.scale_mismatches == 0
            and self.bias_mismatches == 0
        )

    def within(self, bound: float) -> bool:
        """Whether the two encodings dequantise to the same weights within ``bound``."""
        return self.max_weight_difference <= bound

    def to_json(self) -> dict[str, object]:
        return {
            "gguf_name": self.gguf_name,
            "mlx_name": self.mlx_name,
            "rows": self.rows,
            "columns": self.columns,
            "groups": self.groups,
            "reorder": self.reorder,
            "code_mismatches": self.code_mismatches,
            "scale_mismatches": self.scale_mismatches,
            "bias_mismatches": self.bias_mismatches,
            "mismatched_rows": list(self.mismatched_rows),
            "max_code": self.max_code,
            "max_weight_difference": self.max_weight_difference,
            "agrees": self.agrees,
        }


def dequantised_difference(
    source_codes: np.ndarray,
    source_scale_bits: np.ndarray,
    target_codes: np.ndarray,
    target_scale_bits: np.ndarray,
    target_bias_bits: np.ndarray,
) -> float:
    """Largest weight-space gap between the two encodings of the same rows.

    The GGUF dequantises as ``(code - 1) * scale`` and mlx-lm as
    ``code * scale + bias``. Where the codes and scales agree bit for bit *and*
    ``bias == -scale``, the two expressions are algebraically the same number, so
    only rows that differ somewhere need evaluating -- the rest are exactly zero
    by identity rather than by tolerance.
    """
    groups = source_scale_bits.shape[-1]
    shaped = (source_codes.shape[0], groups, source_codes.shape[1] // groups)
    source = (source_codes.reshape(shaped).astype(np.float64) - 1.0) * source_scale_bits.view(
        "<f2"
    ).astype(np.float64)[:, :, None]
    target = target_codes.reshape(shaped).astype(np.float64) * target_scale_bits.view("<f2").astype(
        np.float64
    )[:, :, None] + target_bias_bits.view("<f2").astype(np.float64)[:, :, None]
    return float(np.abs(source - target).max()) if source.size else 0.0


def describe_reorder(reorder: ValueHeadReorder | None) -> str:
    return "none" if reorder is None else f"{reorder.axis.value}/{reorder.unit.value}"


def compare_quantised(
    gguf_path: Path,
    baseline: LocalSafetensors,
    tensor: MappedTensor,
    module: str,
    *,
    layout: HeadLayout,
    weights_per_chunk: int,
    stop_after_first_mismatch: bool,
    max_recorded_rows: int,
) -> QuantisedComparison:
    """Compare one GGUF tensor against one module of the official checkpoint.

    ``module`` is passed separately from ``tensor.mlx_name`` so the same routine
    serves both the claimed pairing and the deliberately wrong ones.
    """
    rows, columns = tensor.rows, tensor.columns
    groups_per_row = columns // BLOCK_SIZE
    reorder = tensor.reorder
    row_order = (
        layout.source_order(rows, reorder)
        if reorder is not None and reorder.axis is HeadAxis.ROWS
        else None
    )
    # A column reordering is applied to the packed reduction axis, which is
    # addressable in 128-weight groups rather than in weights.
    column_order = (
        layout.group_source_order(groups_per_row, reorder, group_size=BLOCK_SIZE)
        if reorder is not None and reorder.axis is HeadAxis.COLUMNS
        else None
    )
    boundary = rows if row_order is None else rows - layout.block_length(reorder.unit)

    code_mismatches = 0
    scale_mismatches = 0
    bias_mismatches = 0
    mismatched: list[int] = []
    max_code = 0
    max_gap = 0.0
    for chunk in row_chunks(
        rows, columns, weights_per_chunk=weights_per_chunk, unsplittable_from=boundary
    ):
        source_scales, source_codes = gguf_quantised_chunk(gguf_path, tensor, chunk)
        if row_order is not None:
            local_order = reordered_rows(chunk, row_order)
            if local_order is None:
                raise FormatError(
                    f"{tensor.gguf.name}: row reordering crosses the chunk boundary at "
                    f"row {chunk.begin}"
                )
            source_scales = source_scales[local_order]
            source_codes = source_codes[local_order]
        if column_order is not None:
            source_scales = source_scales[:, column_order]
            source_codes = source_codes.reshape(chunk.rows, groups_per_row, BLOCK_SIZE)[
                :, column_order
            ].reshape(chunk.rows, columns)

        target_scales, target_codes, target_biases = baseline_quantised_chunk(
            baseline, module, columns, chunk
        )
        code_bad = source_codes != target_codes
        scale_bad = source_scales != target_scales
        bias_bad = target_biases != (target_scales ^ NEGATIVE_ZERO_BITS)
        code_mismatches += int(np.count_nonzero(code_bad))
        scale_mismatches += int(np.count_nonzero(scale_bad))
        bias_mismatches += int(np.count_nonzero(bias_bad))
        max_code = max(max_code, int(target_codes.max()), int(source_codes.max()))

        rows_bad = np.flatnonzero(
            code_bad.any(axis=1) | scale_bad.any(axis=1) | bias_bad.any(axis=1)
        )
        for row in rows_bad[: max(0, max_recorded_rows - len(mismatched))]:
            mismatched.append(chunk.begin + int(row))
        if rows_bad.size:
            max_gap = max(
                max_gap,
                dequantised_difference(
                    source_codes[rows_bad],
                    source_scales[rows_bad],
                    target_codes[rows_bad],
                    target_scales[rows_bad],
                    target_biases[rows_bad],
                ),
            )
            if stop_after_first_mismatch:
                break

    return QuantisedComparison(
        gguf_name=tensor.gguf.name,
        mlx_name=module,
        rows=rows,
        columns=columns,
        groups=rows * groups_per_row,
        reorder=describe_reorder(reorder),
        code_mismatches=code_mismatches,
        scale_mismatches=scale_mismatches,
        bias_mismatches=bias_mismatches,
        mismatched_rows=tuple(mismatched),
        max_code=max_code,
        max_weight_difference=max_gap,
    )


@dataclass(frozen=True)
class PlainComparison:
    gguf_name: str
    mlx_name: str
    gguf_shape: tuple[int, ...]
    mlx_shape: tuple[int, ...]
    transform: str
    reorder: str
    elements: int
    mismatches: int
    zero_sign_only: int
    max_absolute_difference: float

    @property
    def agrees(self) -> bool:
        return self.mismatches == self.zero_sign_only

    def to_json(self) -> dict[str, object]:
        return {
            "gguf_name": self.gguf_name,
            "mlx_name": self.mlx_name,
            "gguf_shape": list(self.gguf_shape),
            "mlx_shape": list(self.mlx_shape),
            "transform": self.transform,
            "reorder": self.reorder,
            "elements": self.elements,
            "mismatches": self.mismatches,
            "zero_sign_only": self.zero_sign_only,
            "max_absolute_difference": self.max_absolute_difference,
            "agrees": self.agrees,
        }


def compare_plain(
    source: np.ndarray,
    baseline: LocalSafetensors,
    tensor: MappedTensor,
    *,
    layout: HeadLayout,
) -> PlainComparison:
    """Compare one plain GGUF tensor against its mlx-lm partner, bit for bit."""
    target = np.ascontiguousarray(baseline.read(tensor.mlx_name))
    if target.dtype != np.dtype("<f2"):
        raise FormatError(f"{tensor.mlx_name} is {target.dtype}, expected float16")
    if not np.isfinite(target.astype(np.float64)).all():
        raise FormatError(f"{tensor.mlx_name} holds non-finite values")
    if source.size != target.size:
        raise FormatError(
            f"{tensor.gguf.name} has {source.size} elements but {tensor.mlx_name} has {target.size}"
        )

    reorder = tensor.reorder
    flat = source.reshape(source.shape[0], -1)
    if reorder is not None:
        if reorder.axis is not HeadAxis.ROWS:
            raise FormatError(f"{tensor.gguf.name}: plain tensors only reorder along rows")
        flat = flat[layout.source_order(flat.shape[0], reorder)]

    transform = PLAIN_VALUE_TRANSFORMS.get(tensor.kind)
    values = flat.astype(np.float64)
    if transform is not None:
        with np.errstate(all="ignore"):
            values = transform(values)
    if not np.isfinite(values).all():
        raise FormatError(
            f"{tensor.gguf.name}: the {transform_name(tensor.kind)} transform produced "
            f"{int(np.count_nonzero(~np.isfinite(values)))} non-finite values"
        )

    got = float16_bits(values).reshape(-1)
    want = target.view("<u2").reshape(-1)
    differing = got != want
    zero_sign = zeros_only_difference(got, want)
    gap = np.abs(got.view("<f2").astype(np.float64) - want.view("<f2").astype(np.float64))
    return PlainComparison(
        gguf_name=tensor.gguf.name,
        mlx_name=tensor.mlx_name,
        gguf_shape=tuple(int(dim) for dim in source.shape),
        mlx_shape=tuple(int(dim) for dim in target.shape),
        transform=transform_name(tensor.kind),
        reorder=describe_reorder(reorder),
        elements=int(source.size),
        mismatches=int(np.count_nonzero(differing)),
        zero_sign_only=int(np.count_nonzero(zero_sign)),
        max_absolute_difference=float(gap.max()) if gap.size else 0.0,
    )


def shape_collisions(tensors: list[MappedTensor]) -> dict[str, list[str]]:
    """For each quantised tensor, the same-layer tensors it could be confused with.

    Two tensors of one layer with the same shape are indistinguishable by counts
    and shapes alone, so the map is only evidence if their weights disagree.
    """
    groups: dict[tuple[int | None, int, int], list[MappedTensor]] = defaultdict(list)
    for tensor in tensors:
        if not tensor.quantised:
            continue
        groups[(tensor.layer, tensor.rows, tensor.columns)].append(tensor)
    collisions: dict[str, list[str]] = {}
    for members in groups.values():
        if len(members) < 2:
            continue
        for tensor in members:
            collisions[tensor.gguf.name] = [
                other.mlx_name for other in members if other.mlx_name != tensor.mlx_name
            ]
    return collisions


def crosscheck(
    gguf_path: Path,
    baseline_dir: Path,
    *,
    weights_per_chunk: int,
    max_recorded_rows: int,
    max_weight_difference: float,
) -> dict[str, object]:
    shard = baseline_dir / "model.safetensors"
    config = json.loads((baseline_dir / "config.json").read_text())
    text_config = config["text_config"]
    layout = HeadLayout.from_text_config(text_config)

    reader, infos = load_tensor_infos(gguf_path)
    mapped = build_name_map(infos)
    baseline = LocalSafetensors(shard)
    check_against_baseline(mapped, set(baseline.names()))

    gguf_arrays = {tensor.name: tensor.data for tensor in reader.tensors}
    collisions = shape_collisions(mapped)

    quantised: list[QuantisedComparison] = []
    discriminations: list[QuantisedComparison] = []
    plain: list[PlainComparison] = []
    for tensor in mapped:
        if tensor.quantised:
            quantised.append(
                compare_quantised(
                    gguf_path,
                    baseline,
                    tensor,
                    tensor.mlx_name,
                    layout=layout,
                    weights_per_chunk=weights_per_chunk,
                    stop_after_first_mismatch=False,
                    max_recorded_rows=max_recorded_rows,
                )
            )
            for rival in collisions.get(tensor.gguf.name, []):
                discriminations.append(
                    compare_quantised(
                        gguf_path,
                        baseline,
                        tensor,
                        rival,
                        layout=layout,
                        weights_per_chunk=weights_per_chunk,
                        stop_after_first_mismatch=True,
                        max_recorded_rows=0,
                    )
                )
        else:
            plain.append(
                compare_plain(
                    np.asarray(gguf_arrays[tensor.gguf.name]),
                    baseline,
                    tensor,
                    layout=layout,
                )
            )

    quantised_disagreeing = [item for item in quantised if not item.agrees]
    over_bound = [item for item in quantised if not item.within(max_weight_difference)]
    plain_disagreeing = [item for item in plain if not item.agrees]
    vacuous = [item for item in discriminations if item.agrees]
    return {
        "schema_version": SCHEMA_VERSION,
        "source": {
            "gguf": str(gguf_path),
            "baseline": str(baseline_dir),
            "baseline_tensor_count": len(baseline),
            "baseline_metadata": baseline.metadata,
        },
        "head_layout": {
            "key_heads": layout.key_heads,
            "value_heads": layout.value_heads,
            "value_head_dim": layout.value_head_dim,
            "repeat": layout.repeat,
            "mlx_head_to_gguf_head": [int(value) for value in layout.head_order()],
        },
        "quantised": {
            "tensors": len(quantised),
            "bit_identical_tensors": len(quantised) - len(quantised_disagreeing),
            "weights": sum(item.rows * item.columns for item in quantised),
            "groups": sum(item.groups for item in quantised),
            "code_mismatches": sum(item.code_mismatches for item in quantised),
            "scale_mismatches": sum(item.scale_mismatches for item in quantised),
            "bias_mismatches": sum(item.bias_mismatches for item in quantised),
            "max_code": max(item.max_code for item in quantised) if quantised else 0,
            "max_weight_difference": max(
                (item.max_weight_difference for item in quantised), default=0.0
            ),
            "max_weight_difference_allowed": max_weight_difference,
            "disagreeing": [item.to_json() for item in quantised_disagreeing],
            "over_bound": [item.to_json() for item in over_bound],
            "comparisons": [item.to_json() for item in quantised],
        },
        "discrimination": {
            "comparisons": len(discriminations),
            "vacuous": [item.to_json() for item in vacuous],
            "pairs": [
                {"gguf_name": item.gguf_name, "rival": item.mlx_name, "disagrees": not item.agrees}
                for item in discriminations
            ],
        },
        "plain": {
            "tensors": len(plain),
            "elements": sum(item.elements for item in plain),
            "mismatches": sum(item.mismatches for item in plain),
            "zero_sign_only": sum(item.zero_sign_only for item in plain),
            "max_absolute_difference": max((item.max_absolute_difference for item in plain), default=0.0),
            "disagreeing": [item.to_json() for item in plain_disagreeing],
            "comparisons": [item.to_json() for item in plain],
        },
        "verdict": {
            "quantised_bit_identical": not quantised_disagreeing,
            "quantised_within_bound": not over_bound,
            "plain_agree": not plain_disagreeing,
            "map_is_discriminating": not vacuous,
            "pass": not over_bound and not plain_disagreeing and not vacuous,
        },
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--gguf", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--weights-per-chunk", type=int, required=True)
    parser.add_argument("--max-recorded-rows", type=int, required=True)
    parser.add_argument(
        "--max-weight-difference",
        type=float,
        required=True,
        help="largest dequantised weight gap a tensor may show and still pass",
    )
    args = parser.parse_args(argv)

    result = crosscheck(
        args.gguf,
        args.baseline,
        weights_per_chunk=args.weights_per_chunk,
        max_recorded_rows=args.max_recorded_rows,
        max_weight_difference=args.max_weight_difference,
    )
    write_json_atomic(args.result, result)

    quantised = result["quantised"]
    plain = result["plain"]
    verdict = result["verdict"]
    print(
        f"quantised: {quantised['bit_identical_tensors']}/{quantised['tensors']} tensors identical "
        f"bit for bit over {quantised['weights']} weights; {quantised['code_mismatches']} code, "
        f"{quantised['scale_mismatches']} scale and {quantised['bias_mismatches']} bias mismatches, "
        f"largest weight gap {quantised['max_weight_difference']:.6g}"
    )
    print(
        f"plain:     {plain['tensors']} tensors, {plain['elements']} elements, "
        f"{plain['mismatches']} mismatches ({plain['zero_sign_only']} only the sign of zero)"
    )
    print(
        f"discrimination: {result['discrimination']['comparisons']} rival pairings, "
        f"{len(result['discrimination']['vacuous'])} of them indistinguishable"
    )
    print(f"pass = {verdict['pass']}")
    return 0 if verdict["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
