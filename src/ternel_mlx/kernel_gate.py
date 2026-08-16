"""Gate A4: the Metal ops against independent CPU decoders, at model shapes.

The unit tests in ``tests/test_mlx_kernels.py`` prove the *format* exhaustively
-- every legal byte in every slot, every tail byte, every FP16 scale pattern --
on tensors small enough to check by construction. This module asks the different
question the plan reserves for a gate: at the shapes the model actually
contains, with realistic activations and a large number of independent cases,
how far does each kernel drift from a decoder that shares none of its structure?

Three decoders answer, and their disagreements mean different things:

``lut23_matmul``
    Same algorithm, same summation order, different language. Agreement here is
    expected to be near ULP-level, so this is the comparison that catches a
    wrong trit, a swapped slot or a misread scale -- a single wrong weight out of
    5120 is a large error at this tolerance and a small one against the oracle.

``restored_matmul``
    Inverts the tile permutation and uses the frozen decode-then-dot GEMV. It
    knows nothing about LUT23, so it independently confirms the permutation.

``oracle_matmul``
    Dense float64. The exactness reference: what the fp32 kernels are *drifting
    from*, rather than a peer to agree with.

Two limits are stated rather than hidden. The CPU references cost
O(batch x rows x groups x 26) gathers, so at ``lm_head`` a full-shape comparison
would be tens of billions of them; every case therefore runs its references over
a whole-tile prefix of at most :data:`REFERENCE_ROW_BUDGET` rows, and the full
row count is covered separately by cross-configuration agreement and a
finiteness sweep. And the real-tensor portion of the plan's A4 needs the 7 GB
GGUF, which the agreed staging puts after this gate; it is therefore run in
Stage B and its absence here is recorded in the output.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, replace
from pathlib import Path

import ml_dtypes
import mlx.core as mx
import numpy as np

from bonsai_tq1.format import BLOCK_SIZE, FormatError, encode_tq1_blocks, write_json_atomic

from . import LAYOUT_NAME, LAYOUT_VERSION
from .environment import capture_environment
from .kernels import GemmConfig, MatmulConfig, tq1_gemm, tq1_get_rows, tq1_matmul
from .layout import PackedTensorLayout
from .modules import (
    GEMM_LARGE_BATCH_NARROW_BLOCK,
    GEMM_LARGE_BATCH_WIDE_BLOCK,
    GEMM_SMALL_BATCH_NARROW_BLOCK,
    GEMM_SMALL_BATCH_WIDE_BLOCK,
    GET_ROWS_THREADS,
)
from .packing import pack_blocks
from .reference import (
    MAX_ORACLE_WEIGHTS,
    get_rows_reference,
    lut23_matmul,
    oracle_matmul,
    restored_matmul,
)

SCHEMA_VERSION = 1
SEED = 20260810

# Every awkward row count in the text-only model, each paired with the real
# reduction width of the tensor that has it. 48 is the SSM gate that falls off
# the 256-row tile, 248320 the embedding and output head.
GATE_SHAPES: tuple[tuple[str, int, int], ...] = (
    ("linear_attn.in_proj_a", 48, 5120),
    ("self_attn.k_proj", 1024, 5120),
    ("linear_attn.out_proj", 5120, 6144),
    ("mlp.gate_proj", 17408, 5120),
    ("lm_head", 248320, 5120),
)

# The plan's floor: at least this many independent activation vectors per shape,
# before the edge cases are added.
RANDOM_CASES = 128

# Whole-tile prefix the CPU references are run over. Four 256-row tiles is
# everything the row indexing can structurally get wrong -- several tiles, both
# sides of every boundary, a row count that divides no block size cleanly --
# while lm_head's full 248320 rows would cost a thousand times as much per
# sample for no additional structural coverage.
REFERENCE_ROW_BUDGET = 1024

# ``restored_matmul`` decodes and dots one row at a time in Python, so it costs
# rows x samples interpreter iterations. It is the third arithmetic path rather
# than the primary one -- the tile permutation it also proves is already gated
# byte-exactly in ``tests/test_mlx_format.py`` -- so it runs over a stride
# through the cases instead of all of them.
RESTORED_SAMPLES = 8

# fp32 accumulation drift, normalised by the magnitude of the exact result. The
# same bound the kernel unit tests gate on, so the two cannot disagree about
# what "correct" means.
ORACLE_RELATIVE_TOLERANCE = 1e-5

# Kernel against a reference that sums in the *same* order. Only the fma and the
# order of the threadgroup reduction differ, so the gap is a few ULP; a bound
# this tight is what makes a single wrong weight visible.
SAME_ORDER_RELATIVE_TOLERANCE = 1e-6

# One narrowing of the final store, and nothing else, separates a float16 or
# bfloat16 result from the float32 one. bfloat16 carries 8 mantissa bits.
NARROWED_RELATIVE_TOLERANCE = {"float32": 1e-5, "float16": 2e-3, "bfloat16": 1e-2}

SINGLE = MatmulConfig(threads=256, batch_tile=1, padded_lut=True, safe_clamp=False, k_split=1)
BATCHED = MatmulConfig(threads=256, batch_tile=4, padded_lut=False, safe_clamp=False, k_split=1)

# Split-K, which is what decode dispatches. A split threadgroup starts its
# accumulator part-way into the tensor and stops early, so the two things that
# can go wrong are arithmetic in the group range -- a chunk reading a group its
# neighbour already summed, or the last chunk stopping short of the tail -- and
# both surface as a wrong dot product rather than a slow one.
#
# The gate takes the extremes each shape admits rather than the split the rule
# picks, because the extremes are where a group range is most likely to be
# miscomputed: two chunks give the longest range a split can have, and one group
# per chunk the shortest. Anything the rule picks lies between them.

# Structurally different prefill tilings: different output block, different
# simdgroup grid, different register pressure. Agreement between them at the
# full row count is what covers the rows the CPU references cannot reach.
#
# All six kernel forms are gated, not just the one the dispatch happens to pick.
# The staged fill and the register fragments read the packed bytes in genuinely
# different orders -- one byte spending five trits against five lanes each taking
# one -- and index different decode tables to do it, so agreement between them at
# the full row count is a real cross-check rather than a repeat. Between them the
# tilings cover TM of 1, 2 and 4, TN of 2 and 4, and K chunks of 16, 32 and 64,
# which is what walks the slot window across every way it can be clipped.
GEMM_CONFIGS: tuple[GemmConfig, ...] = (
    # Both halves staged: the form that shipped before the register ones existed.
    GemmConfig(batch_block=128, row_block=128, k_block=32, simd_rows=4, simd_columns=8,
               direct_fragments=False, direct_epilogue=False, threadgroup_table=False,
               safe_clamp=False),
    GemmConfig(batch_block=64, row_block=32, k_block=32, simd_rows=2, simd_columns=2,
               direct_fragments=False, direct_epilogue=False, threadgroup_table=False,
               safe_clamp=False),
    GemmConfig(batch_block=32, row_block=32, k_block=32, simd_rows=2, simd_columns=2,
               direct_fragments=False, direct_epilogue=False, threadgroup_table=False,
               safe_clamp=False),
    # Fragments built in registers, result still written back through a strip.
    GemmConfig(batch_block=64, row_block=128, k_block=64, simd_rows=2, simd_columns=4,
               direct_fragments=True, direct_epilogue=False, threadgroup_table=False,
               safe_clamp=False),
    GemmConfig(batch_block=32, row_block=64, k_block=16, simd_rows=1, simd_columns=2,
               direct_fragments=True, direct_epilogue=False, threadgroup_table=False,
               safe_clamp=False),
    # Staged fill, result written straight out of the accumulators.
    GemmConfig(batch_block=128, row_block=128, k_block=32, simd_rows=4, simd_columns=8,
               direct_fragments=False, direct_epilogue=True, threadgroup_table=False,
               safe_clamp=False),
    GemmConfig(batch_block=32, row_block=32, k_block=32, simd_rows=2, simd_columns=2,
               direct_fragments=False, direct_epilogue=True, threadgroup_table=False,
               safe_clamp=False),
    # Neither: no threadgroup memory and no barrier anywhere in the kernel.
    GemmConfig(batch_block=64, row_block=128, k_block=32, simd_rows=2, simd_columns=4,
               direct_fragments=True, direct_epilogue=True, threadgroup_table=False,
               safe_clamp=False),
    # TM=1, where a lane decodes a weight fragment and multiplies it once. Slow,
    # and gated anyway: the dispatch may not pick it but it must still be right.
    GemmConfig(batch_block=32, row_block=64, k_block=32, simd_rows=4, simd_columns=2,
               direct_fragments=True, direct_epilogue=True, threadgroup_table=False,
               safe_clamp=False),
    # The largest legal output block. It only fits at all without staging.
    GemmConfig(batch_block=128, row_block=256, k_block=32, simd_rows=4, simd_columns=8,
               direct_fragments=True, direct_epilogue=True, threadgroup_table=False,
               safe_clamp=False),
    # The two forms that copy the trit table into threadgroup memory first. The
    # table is only read by the register fragments, so these pair with both
    # epilogues but have no staged-fill counterpart. What can go wrong here is
    # specific: the fill is strided across the whole threadgroup and guarded by
    # one barrier, so a tiling whose thread count does not divide the table
    # evenly, or an epilogue that reuses the same threadgroup bytes, would show
    # up as a wrong result rather than a slow one.
    GemmConfig(batch_block=32, row_block=128, k_block=32, simd_rows=1, simd_columns=4,
               direct_fragments=True, direct_epilogue=False, threadgroup_table=True,
               safe_clamp=False),
    GemmConfig(batch_block=128, row_block=256, k_block=32, simd_rows=4, simd_columns=8,
               direct_fragments=True, direct_epilogue=True, threadgroup_table=True,
               safe_clamp=False),
    # The four tilings ``modules.select_gemm`` actually dispatches -- two batch
    # blocks times two row blocks. The spread above is chosen to walk the kernel
    # forms; these are here because they ship, and a gate that covers everything
    # except what runs proves the wrong thing. The two large-batch ones are also
    # the smallest threadgroups in this list: 64 threads filling the 1280-entry
    # trit table, twenty entries a thread where the other table arms fill ten or
    # fewer.
    GEMM_SMALL_BATCH_WIDE_BLOCK,
    GEMM_SMALL_BATCH_NARROW_BLOCK,
    GEMM_LARGE_BATCH_WIDE_BLOCK,
    GEMM_LARGE_BATCH_NARROW_BLOCK,
)

# The batch the full-shape agreement is repeated at, so that ``MFIT`` -- the
# template bool that compiles the batch bounds out when the batch block divides
# the batch -- is exercised in both states. The gate's own batch is
# ``RANDOM_CASES`` plus the edge cases, which divides none of the batch blocks
# above, so without this every gated launch would take the guarded path and a
# wrong bound on the unguarded one would never be reached. 128 is the least
# common multiple of the batch blocks in ``GEMM_CONFIGS``.
FITTING_BATCH = 128

# Each dtype a graph can run in, paired with the numpy dtype holding the same
# values, so a reference can be computed in the kernel's own store dtype rather
# than in float32 and compared across a narrowing.
DTYPES: tuple[tuple[str, mx.Dtype, np.dtype], ...] = (
    ("float32", mx.float32, np.dtype(np.float32)),
    ("float16", mx.float16, np.dtype(np.float16)),
    ("bfloat16", mx.bfloat16, np.dtype(ml_dtypes.bfloat16)),
)


@dataclass(frozen=True)
class Comparison:
    """Every statistic the plan asks for, over one pair of result matrices."""

    reference: str
    elements: int
    cases: int
    max_abs_error: float
    mean_abs_error: float
    rmse: float
    max_relative_error: float
    normalised_max_abs_error: float
    worst_case: str
    min_cosine_similarity: float
    worst_cosine_case: str
    mismatches: int
    mismatched_cases: int
    tolerance: float
    bit_exact: bool
    candidate_non_finite: int
    reference_non_finite: int
    non_finite_pattern_mismatches: int

    @property
    def passed(self) -> bool:
        return self.mismatches == 0 and self.non_finite_pattern_mismatches == 0

    def to_json(self) -> dict[str, object]:
        return {
            "reference": self.reference,
            "elements": self.elements,
            "cases": self.cases,
            "max_abs_error": self.max_abs_error,
            "mean_abs_error": self.mean_abs_error,
            "rmse": self.rmse,
            "max_relative_error": self.max_relative_error,
            "normalised_max_abs_error": self.normalised_max_abs_error,
            "worst_case": self.worst_case,
            "min_cosine_similarity": self.min_cosine_similarity,
            "worst_cosine_case": self.worst_cosine_case,
            "mismatches": self.mismatches,
            "mismatched_cases": self.mismatched_cases,
            "tolerance": self.tolerance,
            "bit_exact": self.bit_exact,
            "candidate_non_finite": self.candidate_non_finite,
            "reference_non_finite": self.reference_non_finite,
            "non_finite_pattern_mismatches": self.non_finite_pattern_mismatches,
            "passed": self.passed,
        }


def _classify(values: np.ndarray) -> np.ndarray:
    """Finite / +inf / -inf / NaN, the class a kernel has to preserve.

    Counting non-finite entries on each side is not enough on its own: a kernel
    that turned one row's NaN into an infinity and another row's infinity into a
    NaN would keep the count and still be wrong. This compares position by
    position.
    """
    classes = np.zeros(values.shape, dtype=np.int8)
    classes[np.isposinf(values)] = 1
    classes[np.isneginf(values)] = 2
    classes[np.isnan(values)] = 3
    return classes


def compare(
    candidate: np.ndarray,
    reference: np.ndarray,
    *,
    name: str,
    tolerance: float,
    labels: tuple[str, ...],
) -> Comparison:
    """Per-case statistics over the finite entries, plus non-finite propagation.

    Both sides are ``(case, output)``, and every relative quantity is normalised
    **within a case**. That is load-bearing rather than cosmetic: the activation
    set deliberately spans twenty orders of magnitude, so normalising the whole
    matrix by its global maximum would let the ``1e18`` case set an absolute
    tolerance of ``1e13`` under which all 128 unit-magnitude cases would pass no
    matter what the kernel returned.

    Non-finite entries are excluded from the error statistics and compared as a
    pattern instead, because a case that deliberately feeds infinities must be
    gated on the kernel *propagating* them the way the reference does, not on
    the arithmetic distance between two infinities.
    """
    got = np.asarray(candidate, dtype=np.float64)
    want = np.asarray(reference, dtype=np.float64)
    if got.shape != want.shape:
        raise FormatError(f"cannot compare {got.shape} against {want.shape}")
    if got.ndim != 2:
        raise FormatError(f"comparisons are per activation case, so both sides must be 2-D, got {got.shape}")
    if len(labels) != got.shape[0]:
        raise FormatError(f"{len(labels)} labels for {got.shape[0]} cases")

    pattern = int(np.count_nonzero(_classify(got) != _classify(want)))
    candidate_non_finite = int(np.count_nonzero(~np.isfinite(got)))
    reference_non_finite = int(np.count_nonzero(~np.isfinite(want)))

    finite = np.isfinite(got) & np.isfinite(want)
    # Zeroed before the subtraction rather than after it, so ``inf - inf`` is
    # never evaluated. Masking afterwards computes the same numbers but raises
    # an invalid-value warning on exactly the cases that are *meant* to be
    # infinite, which would train the reader to ignore those warnings.
    a = np.where(finite, got, 0.0)
    b = np.where(finite, want, 0.0)
    error = np.abs(a - b)
    # Normalising by the largest exact value in the case rather than element-wise
    # keeps a near-cancelling output element -- where the exact answer is ~0 and
    # every method disagrees in relative terms -- from dominating the verdict.
    magnitude = np.abs(b).max(axis=1)
    scale = np.where(magnitude > 0.0, magnitude, 1.0)
    normalised = error / scale[:, None]
    over = normalised > tolerance

    norms = np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1)
    cosine = np.where(
        norms > 0.0,
        np.einsum("ij,ij->i", a, b) / np.where(norms > 0.0, norms, 1.0),
        1.0,
    )

    measured = error[finite]
    denominator = np.where(finite & (np.abs(want) > 0.0), np.abs(want), np.inf)
    return Comparison(
        reference=name,
        elements=int(got.size),
        cases=int(got.shape[0]),
        max_abs_error=float(measured.max()) if measured.size else 0.0,
        mean_abs_error=float(measured.mean()) if measured.size else 0.0,
        rmse=float(np.sqrt(np.mean(np.square(measured)))) if measured.size else 0.0,
        max_relative_error=float((error / denominator).max()),
        normalised_max_abs_error=float(normalised.max()),
        worst_case=labels[int(np.argmax(normalised.max(axis=1)))],
        min_cosine_similarity=float(cosine.min()),
        worst_cosine_case=labels[int(np.argmin(cosine))],
        # How many output elements are out of tolerance, not merely whether any
        # is: one bad row out of five thousand and a systematically wrong kernel
        # produce the same worst-case error but very different counts.
        mismatches=int(np.count_nonzero(over)),
        mismatched_cases=int(np.count_nonzero(over.any(axis=1))),
        tolerance=tolerance,
        bit_exact=bool(measured.size and measured.max() == 0.0 and pattern == 0),
        candidate_non_finite=candidate_non_finite,
        reference_non_finite=reference_non_finite,
        non_finite_pattern_mismatches=pattern,
    )


def random_packed(rows: int, columns: int, *, seed: int):
    """A packed tensor built through the frozen block encoder, then tiled."""
    layout = PackedTensorLayout.for_tensor(rows, columns)
    generator = np.random.default_rng(seed)
    scale_bits = (
        generator.normal(0.0, 0.05, size=layout.groups)
        .astype(np.float16)
        .view(np.uint8)
        .reshape(-1, 2)
    )
    trits = generator.integers(0, 3, size=(layout.groups, BLOCK_SIZE), dtype=np.uint8)
    blocks = encode_tq1_blocks(scale_bits, trits).reshape(rows, layout.groups_per_row, -1)
    codes, scales = pack_blocks(blocks, layout=layout)
    return layout, codes, scales


def activation_cases(columns: int, *, seed: int) -> tuple[np.ndarray, tuple[str, ...]]:
    """``RANDOM_CASES`` independent vectors, then the edge cases, as one matrix.

    They are returned together so a single batched launch covers every case, and
    the labels let the report say which row is which.
    """
    generator = np.random.default_rng(seed)
    random = generator.normal(0.0, 1.0, size=(RANDOM_CASES, columns)).astype(np.float32)
    labels = [f"random_{index}" for index in range(RANDOM_CASES)]

    # fp32 has a max normal near 3.4e38; a dot product of 5120 such terms would
    # overflow whatever the kernel does, so "extreme" means the largest input
    # whose product with a 0.05-ish scale still sums finitely.
    large = np.float32(1e18)
    # Small but firmly normal. Going all the way down to the subnormal range
    # would measure Metal's flush-to-zero policy rather than this kernel, and a
    # gate that fails on a documented platform behaviour tells nobody anything.
    small = np.float32(1e-30)
    edges = {
        "zeros": np.zeros(columns, dtype=np.float32),
        "ones": np.ones(columns, dtype=np.float32),
        "alternating_signs": np.where(
            np.arange(columns) % 2 == 0, np.float32(1.0), np.float32(-1.0)
        ).astype(np.float32),
        "single_hot_first": np.eye(1, columns, 0, dtype=np.float32)[0],
        "single_hot_last": np.eye(1, columns, columns - 1, dtype=np.float32)[0],
        # The first column of the tail slot, which no full slot ever touches.
        "single_hot_tail": np.eye(1, columns, 125, dtype=np.float32)[0],
        "large_positive": np.full(columns, large, dtype=np.float32),
        "large_alternating": (
            np.where(np.arange(columns) % 2 == 0, large, -large).astype(np.float32)
        ),
        "small_normal": np.full(columns, small, dtype=np.float32),
        "mixed_magnitudes": (
            generator.normal(0.0, 1.0, size=columns)
            * np.float32(10.0) ** generator.integers(-12, 12, size=columns)
        ).astype(np.float32),
    }
    labels.extend(edges)
    return np.concatenate([random, np.stack(list(edges.values()))]), tuple(labels)


def non_finite_cases(columns: int) -> tuple[np.ndarray, tuple[str, ...]]:
    """Activations the kernel must propagate rather than quietly absorb.

    A kernel that returned zeros for a NaN input would hide a broken upstream
    layer, so the gate is that the non-finite *pattern* matches the reference's.
    """
    cases = {
        "all_nan": np.full(columns, np.nan, dtype=np.float32),
        "all_positive_inf": np.full(columns, np.inf, dtype=np.float32),
        "single_nan_first": np.zeros(columns, dtype=np.float32),
        "single_nan_tail": np.zeros(columns, dtype=np.float32),
        "single_inf_last": np.zeros(columns, dtype=np.float32),
    }
    cases["single_nan_first"][0] = np.nan
    cases["single_nan_tail"][125] = np.nan
    cases["single_inf_last"][columns - 1] = -np.inf
    return np.stack(list(cases.values())), tuple(cases)


def matmul(codes, scales, x, *, layout, config) -> np.ndarray:
    out = tq1_matmul(
        mx.array(codes), mx.array(scales), mx.array(x),
        groups_per_row=layout.groups_per_row, tile=layout.tile, config=config,
    )
    mx.eval(out)
    return np.array(out, copy=False)


def gemm(codes, scales, x, *, layout, config) -> np.ndarray:
    out = tq1_gemm(
        mx.array(codes), mx.array(scales), mx.array(x),
        groups_per_row=layout.groups_per_row, tile=layout.tile, config=config,
    )
    mx.eval(out)
    return np.array(out, copy=False)


def pair_tolerance(arm: str, reference: str) -> float:
    """The bound this particular pairing is entitled to be judged by.

    Only one pairing sums in the same order: a LUT23 kernel against
    ``lut23_matmul``, which mirrors slot order, the per-slot ``L2 + L3`` add and
    the single per-group ``fma`` exactly. There the two should agree to a few
    ULP or not at all, and a bound that tight is what makes one wrong weight out
    of 5120 visible.

    Everything else differs by summation order and is bounded by ordinary fp32
    drift over a 5120-term dot product: the GEMM arms fold each group's scale
    into the weight and reduce through the matrix units, ``restored`` decodes and
    dots, ``oracle`` works in float64. Holding those to the same-order bound
    would be gating on an accumulation order none of them claims to share.

    A split LUT23 arm is in that second category despite its name. It walks slot
    order identically inside each chunk, but the chunks are summed afterwards,
    so it associates the group sum differently from the reference and cannot
    claim the ULP-level bound the unsplit arms are held to.
    """
    if reference == "lut23" and arm.startswith("lut23") and "_k" not in arm:
        return SAME_ORDER_RELATIVE_TOLERANCE
    return ORACLE_RELATIVE_TOLERANCE


def split_gate_cases(groups_per_row: int) -> tuple[int, ...]:
    """The coarsest and finest splits of a tensor with this many groups.

    Chunk counts, so the smallest divisor above one gives the longest chunks and
    ``groups_per_row`` itself gives one group each. A prime group count collapses
    the two into a single case, which is why the result is a deduplicated set
    rather than a pair.
    """
    divisors = [k for k in range(2, groups_per_row + 1) if groups_per_row % k == 0]
    if not divisors:
        raise FormatError(f"{groups_per_row} groups admit no split at all")
    return tuple(sorted({min(divisors), max(divisors)}))


def reference_prefix(layout: PackedTensorLayout) -> PackedTensorLayout:
    """The whole-tile prefix of ``layout`` the CPU references are run over."""
    tiles = max(1, min(layout.tiles, REFERENCE_ROW_BUDGET // layout.tile))
    return PackedTensorLayout(rows=tiles * layout.tile, columns=layout.columns, tile=layout.tile)


def gate_against_references(
    name: str, rows: int, columns: int
) -> dict[str, object]:
    """Every kernel, every dtype, against every reference, at one model shape."""
    layout, codes, scales = random_packed(rows, columns, seed=SEED + rows + columns)
    prefix = reference_prefix(layout)
    sliced_codes = codes[: prefix.tiles]
    sliced_scales = scales[: prefix.tiles]

    x, labels = activation_cases(columns, seed=SEED + columns)
    non_finite, non_finite_labels = non_finite_cases(columns)

    # A stride rather than a prefix, so the subset ``restored`` sees spans the
    # random cases and the edge cases alike instead of stopping before them.
    stride = max(1, len(x) // RESTORED_SAMPLES)
    references: dict[str, np.ndarray] = {
        "lut23": lut23_matmul(sliced_codes, sliced_scales, x, layout=prefix),
        "restored": restored_matmul(sliced_codes, sliced_scales, x[::stride], layout=prefix),
    }
    subsets = {"restored": stride}
    if prefix.rows * prefix.columns <= MAX_ORACLE_WEIGHTS:
        references["oracle"] = oracle_matmul(sliced_codes, sliced_scales, x, layout=prefix)

    # These cases exist to be infinite, so the reference producing ``inf * 0``
    # and ``inf + -inf`` is the arithmetic under test, not a numerical accident.
    # Declaring that here keeps the invalid-value warnings numpy would otherwise
    # raise from burying a warning that does mean something.
    with np.errstate(invalid="ignore"):
        non_finite_references = {
            "lut23": lut23_matmul(sliced_codes, sliced_scales, non_finite, layout=prefix)
        }

    arms: dict[str, np.ndarray] = {
        "lut23_batched": matmul(sliced_codes, sliced_scales, x, layout=prefix, config=BATCHED),
        # The single-vector path is a different template instantiation, so each
        # case goes through it on its own rather than as a batch.
        "lut23_single": np.concatenate(
            [
                matmul(sliced_codes, sliced_scales, row[None, :], layout=prefix, config=SINGLE)
                for row in x
            ]
        ),
    }
    for base in (SINGLE, BATCHED):
        for split in split_gate_cases(prefix.groups_per_row):
            config = replace(base, k_split=split)
            if base is SINGLE:
                got = np.concatenate([
                    matmul(sliced_codes, sliced_scales, row[None, :], layout=prefix, config=config)
                    for row in x
                ])
            else:
                got = matmul(sliced_codes, sliced_scales, x, layout=prefix, config=config)
            arms[config.benchmark_name] = got
    for config in GEMM_CONFIGS:
        if prefix.tiles > 1 and prefix.tile % config.row_block:
            continue
        arms[config.benchmark_name] = gemm(
            sliced_codes, sliced_scales, x, layout=prefix, config=config
        )

    comparisons = [
        {
            "arm": arm,
            "activations": "float32",
            **compare(
                result[:: subsets.get(key, 1)], reference,
                name=key, tolerance=pair_tolerance(arm, key),
                labels=labels[:: subsets.get(key, 1)],
            ).to_json(),
        }
        for arm, result in arms.items()
        for key, reference in references.items()
    ]

    # Non-finite propagation, on the LUT23 path only: the property under test is
    # whether the kernel masks a NaN, and every arm shares that behaviour or
    # does not, so one arm per family is the whole question.
    for arm, config in (("lut23_batched", BATCHED), ("lut23_single", SINGLE)):
        got = matmul(sliced_codes, sliced_scales, non_finite, layout=prefix, config=config)
        comparisons.append(
            {
                "arm": arm,
                "activations": "non_finite",
                **compare(
                    got, non_finite_references["lut23"],
                    name="lut23", tolerance=SAME_ORDER_RELATIVE_TOLERANCE,
                    labels=non_finite_labels,
                ).to_json(),
            }
        )

    # Narrowed activations, on the random cases only: the 1e18 edge case is past
    # float16's largest normal, so including it would measure an overflow both
    # sides agree on rather than the narrowing under test. A narrowed input is
    # itself exact in float32, so the reference is fed the round-tripped values
    # and only the final store is left to differ.
    narrow_cases = x[:RANDOM_CASES]
    for dtype_name, dtype, _store_dtype in DTYPES:
        if dtype_name == "float32":
            continue
        activations = mx.array(narrow_cases).astype(dtype)
        mx.eval(activations)
        reference = lut23_matmul(
            sliced_codes, sliced_scales,
            np.array(activations.astype(mx.float32), copy=False), layout=prefix,
        )
        got = tq1_matmul(
            mx.array(sliced_codes), mx.array(sliced_scales), activations,
            groups_per_row=prefix.groups_per_row, tile=prefix.tile, config=BATCHED,
        )
        mx.eval(got)
        comparisons.append(
            {
                "arm": "lut23_batched",
                "activations": dtype_name,
                **compare(
                    np.array(got.astype(mx.float32), copy=False), reference,
                    name="lut23", tolerance=NARROWED_RELATIVE_TOLERANCE[dtype_name],
                    labels=labels[:RANDOM_CASES],
                ).to_json(),
            }
        )

    full = full_shape_agreement(codes, scales, x, labels=labels, layout=layout)
    return {
        "tensor": name,
        "rows": rows,
        "columns": columns,
        "tile": layout.tile,
        "tiles": layout.tiles,
        "activation_cases": len(labels),
        "random_cases": RANDOM_CASES,
        "edge_case_labels": [label for label in labels if not label.startswith("random_")],
        "non_finite_case_labels": list(non_finite_labels),
        "reference_rows": prefix.rows,
        "reference_rows_are_full_shape": prefix.rows == layout.rows,
        "restored_sample_stride": stride,
        "comparisons": comparisons,
        "full_shape": full,
        "passed": (
            all(entry["passed"] for entry in comparisons) and bool(full["passed"])
        ),
    }


def full_shape_agreement(
    codes: np.ndarray,
    scales: np.ndarray,
    x: np.ndarray,
    *,
    labels: tuple[str, ...],
    layout: PackedTensorLayout,
) -> dict[str, object]:
    """Cover the rows the CPU references cannot reach, without a CPU reference.

    Every kernel here reaches the same weights by a different route -- one
    thread per row against a decoded threadgroup tile, three different output
    blocks, three different simdgroup grids -- so agreement across all of them
    over the full row count is strong evidence about the rows the bounded
    reference prefix never sees. It is evidence of a different kind from the
    reference comparison, and is reported separately for that reason.

    Each configuration runs twice: once over the whole activation matrix, whose
    row count divides no batch block, and once over a ``FITTING_BATCH`` prefix,
    which divides all of them. Those are different kernels -- the batch bound is
    a compile-time constant -- and only the second reaches the stores that run
    unguarded. Slicing the baseline rather than recomputing it is exact: every
    reference arm here reduces along the columns of one activation row and never
    across rows, which is the same independence ``lut23_single`` is compared
    under below.
    """
    if x.shape[0] < FITTING_BATCH:
        raise FormatError(
            f"the activation matrix has {x.shape[0]} rows, too few to take a "
            f"{FITTING_BATCH}-row prefix; the batch-bound-free kernels would go ungated"
        )
    baseline = matmul(codes, scales, x, layout=layout, config=BATCHED)
    entries = []
    for config in GEMM_CONFIGS:
        if layout.tiles > 1 and layout.tile % config.row_block:
            continue
        for batch, tag in ((x.shape[0], ""), (FITTING_BATCH, "_fitb")):
            got = gemm(codes, scales, x[:batch], layout=layout, config=config)
            entries.append(
                {
                    "arm": config.benchmark_name + tag,
                    "batch": batch,
                    "batch_block_divides_batch": batch % config.batch_block == 0,
                    "row_block_divides_rows": layout.rows % config.row_block == 0,
                    **compare(
                        got, baseline[:batch], name="lut23_batched",
                        tolerance=ORACLE_RELATIVE_TOLERANCE, labels=labels[:batch],
                    ).to_json(),
                }
            )
    single = matmul(codes, scales, x[:1], layout=layout, config=SINGLE)
    entries.append(
        {
            "arm": "lut23_single",
            **compare(
                single, baseline[:1], name="lut23_batched",
                tolerance=SAME_ORDER_RELATIVE_TOLERANCE, labels=labels[:1],
            ).to_json(),
        }
    )
    return {
        "rows": layout.rows,
        "baseline": "lut23_batched",
        "non_finite_outputs": int(np.count_nonzero(~np.isfinite(baseline))),
        "agreements": entries,
        "passed": all(entry["passed"] for entry in entries),
    }


def gate_get_rows(rows: int, columns: int) -> dict[str, object]:
    """The embedding path, which the references *can* cover at full row count.

    A gather decodes only the rows it is asked for, so unlike the matmul there is
    no budget problem here: the whole 248320-row table is in scope.
    """
    layout, codes, scales = random_packed(rows, columns, seed=SEED + rows)
    generator = np.random.default_rng(SEED)
    # Deterministic coverage of the table: both ends, both sides of a tile
    # boundary, and a random spread through the middle.
    boundaries = [0, 1, layout.tile - 1, layout.tile, layout.tile + 1, rows - 1]
    indices = np.array(
        sorted(set(boundaries)) + generator.integers(0, rows, size=RANDOM_CASES).tolist(),
        dtype=np.uint32,
    )

    entries = []
    for dtype_name, dtype, store_dtype in DTYPES:
        out = tq1_get_rows(
            mx.array(codes), mx.array(scales), mx.array(indices),
            groups_per_row=layout.groups_per_row, tile=layout.tile,
            dtype=dtype, threads=GET_ROWS_THREADS, safe_clamp=False,
        )
        mx.eval(out)
        got = np.array(out.astype(mx.float32), copy=False)
        # The reference is computed in the kernel's own store dtype, and the two
        # are then compared in float32, which both widen into exactly.
        #
        # Computing it in float32 and comparing across the narrowing would be
        # wrong for bfloat16: an fp16 scale carries 10 mantissa bits where
        # bfloat16 holds 7, so (t - 1) * scale is exactly representable in
        # float32 and float16 but not in bfloat16, and the rounding is part of
        # what this gates rather than noise around it. Doing the multiply in the
        # store dtype is equivalent to the kernel's round-after-multiply because
        # t - 1 is one of -1, 0, +1 and round-to-nearest-even is sign-symmetric.
        #
        # With that settled this must be equality: the only arithmetic in the
        # whole path is a sign flip, so a tolerance would hide a real defect.
        reference = get_rows_reference(
            codes, scales, indices, layout=layout, dtype=store_dtype
        ).astype(np.float32)
        exact = bool(np.array_equal(got, reference))
        entries.append(
            {
                "dtype": dtype_name,
                "indices": int(indices.size),
                "exact": exact,
                **compare(
                    got, reference, name="get_rows_reference", tolerance=0.0,
                    labels=tuple(f"token_{int(token)}" for token in indices),
                ).to_json(),
            }
        )
    return {
        "tensor": "embed_tokens",
        "rows": rows,
        "columns": columns,
        "tile": layout.tile,
        "boundary_indices": boundaries,
        "results": entries,
        "passed": all(entry["exact"] for entry in entries),
    }


def run(*, shapes: tuple[tuple[str, int, int], ...]) -> dict[str, object]:
    environment = capture_environment()
    matmuls = []
    for name, rows, columns in shapes:
        print(f"  {name:24s} {rows:6d}x{columns:<6d} ", file=sys.stderr, end="", flush=True)
        result = gate_against_references(name, rows, columns)
        print("pass" if result["passed"] else "FAIL", file=sys.stderr)
        matmuls.append(result)

    print("  embed_tokens             248320x5120  ", file=sys.stderr, end="", flush=True)
    rows_result = gate_get_rows(248320, 5120)
    print("pass" if rows_result["passed"] else "FAIL", file=sys.stderr)

    return {
        "schema_version": SCHEMA_VERSION,
        "layout_name": LAYOUT_NAME,
        "layout_version": LAYOUT_VERSION,
        "seed": SEED,
        "random_activation_cases": RANDOM_CASES,
        "reference_row_budget": REFERENCE_ROW_BUDGET,
        "tolerances": {
            "same_order_relative": SAME_ORDER_RELATIVE_TOLERANCE,
            "oracle_relative": ORACLE_RELATIVE_TOLERANCE,
            "narrowed_relative": NARROWED_RELATIVE_TOLERANCE,
            "rule": (
                "a LUT23 kernel against the lut23 reference sums in the same order and is "
                "held to same_order_relative; every other pairing reorders the sum and is "
                "held to oracle_relative; each is normalised within an activation case, "
                "never across cases"
            ),
        },
        "references": {
            "lut23": "numpy LUT23, the kernel's own algorithm and summation order",
            "restored": "tile permutation inverted, then the frozen decode-then-dot GEMV",
            "oracle": "dense float64 matmul, capped at 2^27 weights",
        },
        "scope": {
            "weights": "synthetic tensors at real model shapes",
            "real_tensors": (
                "deferred to Stage B: gating against tensors streamed from the GGUF "
                "requires the 7 GB download that the agreed staging places after this gate"
            ),
        },
        "matmul": matmuls,
        "get_rows": rows_result,
        "environment": environment,
        "passed": all(entry["passed"] for entry in matmuls) and bool(rows_result["passed"]),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Gate the packed TQ1 Metal ops (A4)")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--shape",
        action="append",
        metavar="NAME:ROWS:COLUMNS",
        help="override the model shapes; repeatable",
    )
    args = parser.parse_args(argv)

    shapes = GATE_SHAPES
    if args.shape:
        parsed = []
        for entry in args.shape:
            parts = entry.split(":")
            if len(parts) != 3:
                raise FormatError(f"expected NAME:ROWS:COLUMNS, got {entry!r}")
            parsed.append((parts[0], int(parts[1]), int(parts[2])))
        shapes = tuple(parsed)

    payload = run(shapes=shapes)
    write_json_atomic(args.output, payload)
    print(json.dumps({"output": str(args.output), "passed": payload["passed"]}), file=sys.stderr)
    return 0 if payload["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
