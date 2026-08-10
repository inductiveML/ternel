"""Op-level benchmarks for the packed TQ1_G128 Metal path.

This answers one question and refuses to answer it vaguely: at each shape the
model actually contains, is executing packed ternary weights directly faster or
slower than the best MLX itself can do on the same logical matrix? The baseline
is therefore ``mx.quantized_matmul`` at ``bits=2, group_size=128`` -- the exact
representation the official Ternary Bonsai MLX repo ships, and the ceiling a
user already has today without this project.

Method, all of it inherited from the CUDA phase's reporting discipline:

* **Paired AB/BA alternation.** Each trial runs both arms; odd trials run them
  in the opposite order. A thermal ramp or a background process that drifts
  across the run therefore lands on both arms roughly equally instead of
  becoming a fake speedup.
* **Enqueue-then-synchronise.** A timed sample enqueues ``inner`` independent
  calls and synchronises once. ``inner`` is chosen per case so the timed region
  clears :data:`MIN_SAMPLE_SECONDS`, which puts the measurement well above
  Python dispatch jitter -- and both arms get the identical treatment, so the
  ratio is fair even where the absolute number carries dispatch cost.
* **Medians with intervals, never a bare mean.** ``quantile_summary`` reports
  p5/p95 alongside the median and ``paired_bootstrap_ratio`` puts a 95%
  confidence interval on the speedup, so a ratio whose interval straddles 1.0
  is visibly not a result.
* **Warm, then measured.** ``mx.fast.metal_kernel`` JIT-compiles one Metal
  library per template tuple. That cost is real but it is a cold-start cost, so
  it is measured separately and reported as such rather than smeared into the
  steady-state numbers.

Every case is correctness-checked against the numpy LUT23 reference before it is
timed, so a kernel that has been made fast by being wrong cannot post a number.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import mlx.core as mx
import numpy as np

from bonsai_tq1.format import BLOCK_SIZE, FormatError, encode_tq1_blocks, write_json_atomic
from bonsai_tq1.lut23_reporting import paired_bootstrap_ratio, quantile_summary

from . import LAYOUT_NAME, LAYOUT_VERSION
from .environment import capture_environment
from .kernels import GemmConfig, MatmulConfig, tq1_gemm, tq1_get_rows, tq1_matmul
from .layout import PackedTensorLayout
from .packing import pack_blocks
from .reference import lut23_matmul

SCHEMA_VERSION = 1

# Deterministic everywhere: the same seed must reproduce the same weights, the
# same activations and the same bootstrap resample as the CUDA phase used.
SEED = 20260809

# A timed sample must span at least this long, otherwise `inner` is doubled.
# Below roughly a millisecond, perf_counter jitter and Python dispatch noise are
# a visible fraction of the sample and the medians stop being comparable.
MIN_SAMPLE_SECONDS = 1.0e-3

# Bounds on the enqueue count. The floor keeps every case honest about dispatch
# cost; the ceiling stops a very cheap case from queueing thousands of buffers.
MIN_INNER = 4
MAX_INNER = 256

TRIALS = 40
WARMUP_SAMPLES = 5

# Relative tolerance for the pre-timing correctness check. The kernel is
# bit-exact against this reference in the test suite; this is a tripwire for a
# mis-shaped benchmark harness, not a substitute for the gate.
BENCH_RTOL = 1.0e-5


@dataclass(frozen=True)
class OpCase:
    """One (weight shape, batch) point to measure."""

    label: str
    rows: int
    columns: int
    batch: int

    def layout(self) -> PackedTensorLayout:
        return PackedTensorLayout.for_tensor(self.rows, self.columns)


# The distinct linear shapes of Ternary Bonsai 27B, named for the module they
# come from. Both hybrid layer kinds are represented, plus the tied embedding.
MODEL_SHAPES: tuple[tuple[str, int, int], ...] = (
    ("linear_attn.in_proj_qkv", 10240, 5120),
    ("linear_attn.in_proj_z", 6144, 5120),
    ("linear_attn.in_proj_a", 48, 5120),
    ("linear_attn.out_proj", 5120, 6144),
    ("self_attn.q_proj", 12288, 5120),
    ("self_attn.k_proj", 1024, 5120),
    ("self_attn.o_proj", 5120, 12288),
    ("mlp.gate_proj", 17408, 5120),
    ("mlp.down_proj", 5120, 17408),
    ("lm_head", 248320, 5120),
)

# The configuration sweep the plan requires: batch tile against LUT padding.
# Combinations that overflow threadgroup memory are dropped by MatmulConfig
# itself and reported as unavailable rather than silently skipped.
BATCH_TILES: tuple[int, ...] = (1, 2, 4, 8)
PADDINGS: tuple[bool, ...] = (True, False)
BENCH_THREADS = 256

# Where the register form reads its trit table from. ``constant`` memory serves
# the staged fill well -- it spends all five digits of a code byte at once -- but
# the register form gathers 2*TN entries per K step at data-dependent indices, so
# 32 lanes hit 32 unrelated offsets and the constant path serialises them.
#
# A dedicated cube over (row bounds, batch bounds, table residency), 45 paired
# rows, found the table is not independent of the bounds: without them it loses
# (geometric mean 0.976x, 39 of 45 rows below parity), with the row bound
# compiled out it wins (1.011x, and 1.026x with both bounds gone). But per tiling
# the effect ran 0.999x to 1.025x with 11 of 45 individual rows still below
# parity -- a positive mean with no rule under it. The 1024-thread tiling that
# looked like a clean regression at 17 rows came back to 0.999x at 45.
#
# So it is swept rather than set. It is only legal with ``direct_fragments``:
# the staged fill decodes through the byte-major table instead and never reads
# this one, so the cross below is over the register tilings only. That is a
# structural precondition rather than a budget one, so it is not routed through
# the rejection list -- nothing is learned from recording it five times.
TABLE_RESIDENCY: tuple[bool, ...] = (False, True)

# Prefill tilings, as ``(batch_block, row_block, k_block, simd_rows, simd_cols,
# direct_fragments, direct_epilogue)``.
#
# Two things dominate, and both were measured rather than reasoned about.
#
# The first is how many 8x8 accumulator fragments a lane must keep live. Holding
# the output block and the threadgroup traffic fixed at 128x128 and varying only
# the simdgroup grid over it spread the runtime by 8.7x -- 2x4 simdgroups (8x4
# fragments) took 130798us where 4x8 (4x2 fragments) took 15107us for identical
# arithmetic. The cliff sits between roughly 28 and 48 registers per lane.
#
# The second is TM = batch_block / (simd_rows * 8), which decides whether the
# register form beats the staged one -- see :class:`GemmConfig`. It also splits
# the batch range: below batch 32 the fastest tilings are TM=1 and staged, from
# batch 32 up they are TM=4 and built in registers. Both halves are here because
# the dispatch has to choose between them per shape and batch.
GEMM_TILINGS: tuple[tuple[int, int, int, int, int, bool, bool], ...] = (
    # Small batches. TM=1 leaves the register form paying a whole decode per
    # multiply, and these were the fastest staged tilings measured at 8 and 16.
    (8, 256, 32, 1, 8, False, False),
    (16, 128, 64, 2, 4, False, False),
    (16, 256, 32, 2, 8, False, False),
    # TM=4, where the register form wins. BN and BK vary around the winner so
    # the sweep can separate "TM=4 is what matters" from one lucky tiling.
    (32, 64, 32, 1, 2, True, True),
    (32, 128, 32, 1, 4, True, False),
    (32, 256, 32, 1, 8, True, False),
    (64, 64, 32, 2, 2, True, False),
    (64, 128, 32, 2, 4, True, False),
    (64, 128, 64, 2, 4, True, False),
    (128, 128, 32, 4, 4, True, False),
    # The largest legal output block, which only fits without staging at all.
    (128, 256, 32, 4, 8, True, True),
    # The incumbents, kept so the sweep measures the change rather than
    # replacing the ladder with one that shares no point with the old.
    (128, 128, 32, 4, 8, False, False),
    (128, 32, 32, 4, 2, False, False),
    (128, 32, 32, 4, 2, True, True),
)


@dataclass(frozen=True)
class Variant:
    """One candidate kernel configuration, of either family.

    The two families answer different questions -- ``lut23`` builds an activation
    table once per 128-weight group and reads each weight byte once, ``gemm``
    decodes a weight tile into threadgroup memory and feeds the matrix units --
    so they are measured side by side at every batch rather than one being
    assumed to own a batch range.
    """

    name: str
    family: str
    config: MatmulConfig | GemmConfig

    def supports(self, layout: PackedTensorLayout) -> bool:
        """Whether this variant can run the given tensor at all."""
        if isinstance(self.config, GemmConfig):
            # A row block that straddles two row tiles would read a second
            # tile's codes through the first tile's base pointer.
            return not (layout.tiles > 1 and layout.tile % self.config.row_block)
        return True

    def call(
        self, codes: mx.array, scales: mx.array, x: mx.array, *, layout: PackedTensorLayout
    ) -> mx.array:
        if isinstance(self.config, GemmConfig):
            return tq1_gemm(
                codes,
                scales,
                x,
                groups_per_row=layout.groups_per_row,
                tile=layout.tile,
                config=self.config,
            )
        return tq1_matmul(
            codes,
            scales,
            x,
            groups_per_row=layout.groups_per_row,
            tile=layout.tile,
            config=self.config,
        )

    def to_json(self) -> dict[str, object]:
        common: dict[str, object] = {
            "config": self.name,
            "family": self.family,
            "threads": self.config.threads,
            "threadgroup_bytes": self.config.threadgroup_bytes,
        }
        if isinstance(self.config, GemmConfig):
            return common | {
                "batch_block": self.config.batch_block,
                "row_block": self.config.row_block,
                "k_block": self.config.k_block,
                "simd_rows": self.config.simd_rows,
                "simd_columns": self.config.simd_columns,
                "accumulators_per_thread": self.config.accumulators_per_thread,
                "direct_fragments": self.config.direct_fragments,
                "direct_epilogue": self.config.direct_epilogue,
                "threadgroup_table": self.config.threadgroup_table,
            }
        return common | {
            "batch_tile": self.config.batch_tile,
            "padded_lut": self.config.padded_lut,
        }


def _configs() -> tuple[list[Variant], list[dict[str, object]]]:
    """Build every requested variant, and record the ones the format rejected.

    A configuration that overflows the 32 KiB threadgroup budget raises out of
    its own constructor. Returning those rejections here rather than rebuilding
    a list of expected names elsewhere keeps one source of truth for what the
    sweep asked for: a name spelled a second time drifts the moment a
    configuration field is added, and the reason is worth more than the name.
    """
    variants: list[Variant] = []
    rejected: list[dict[str, object]] = []

    for batch_tile in BATCH_TILES:
        for padded in PADDINGS:
            try:
                config = MatmulConfig(
                    threads=BENCH_THREADS,
                    batch_tile=batch_tile,
                    padded_lut=padded,
                    safe_clamp=False,
                )
            except FormatError as error:
                rejected.append({
                    "family": "lut23",
                    "request": {"batch_tile": batch_tile, "padded_lut": padded},
                    "reason": str(error),
                })
                continue
            variants.append(
                Variant(name=config.benchmark_name, family="lut23", config=config)
            )

    for (
        batch_block,
        row_block,
        k_block,
        simd_rows,
        simd_columns,
        direct_fragments,
        direct_epilogue,
    ) in GEMM_TILINGS:
        tables = TABLE_RESIDENCY if direct_fragments else (False,)
        for threadgroup_table in tables:
            try:
                gemm = GemmConfig(
                    batch_block=batch_block,
                    row_block=row_block,
                    k_block=k_block,
                    simd_rows=simd_rows,
                    simd_columns=simd_columns,
                    direct_fragments=direct_fragments,
                    direct_epilogue=direct_epilogue,
                    threadgroup_table=threadgroup_table,
                    safe_clamp=False,
                )
            except FormatError as error:
                rejected.append({
                    "family": "gemm",
                    "request": {
                        "batch_block": batch_block,
                        "row_block": row_block,
                        "k_block": k_block,
                        "simd_rows": simd_rows,
                        "simd_columns": simd_columns,
                        "direct_fragments": direct_fragments,
                        "direct_epilogue": direct_epilogue,
                        "threadgroup_table": threadgroup_table,
                    },
                    "reason": str(error),
                })
                continue
            variants.append(
                Variant(name=gemm.benchmark_name, family="gemm", config=gemm)
            )
    return variants, rejected


def build_packed_tensor(case: OpCase) -> tuple[PackedTensorLayout, mx.array, mx.array, np.ndarray, np.ndarray]:
    """Synthesise a ternary tensor with realistic FP16 scales, packed as shipped."""
    layout = case.layout()
    generator = np.random.default_rng(SEED + case.rows * 131 + case.columns)
    scale_bits = (
        generator.normal(0.0, 0.05, size=layout.groups)
        .astype(np.float16)
        .view(np.uint8)
        .reshape(-1, 2)
    )
    trits = generator.integers(0, 3, size=(layout.groups, BLOCK_SIZE), dtype=np.uint8)
    blocks = encode_tq1_blocks(scale_bits, trits).reshape(case.rows, layout.groups_per_row, -1)
    codes, scales = pack_blocks(blocks, layout=layout)
    return layout, mx.array(codes), mx.array(scales), codes, scales


def build_affine_baseline(case: OpCase) -> tuple[mx.array, mx.array, mx.array]:
    """The MLX 2-bit affine weight for the same logical shape.

    Quantised from a float tensor by MLX's own quantiser so the baseline is the
    code path a user actually runs, not a hand-built approximation of it.
    """
    generator = np.random.default_rng(SEED + 7 + case.rows)
    dense = mx.array(
        generator.normal(0.0, 0.05, size=(case.rows, case.columns)).astype(np.float16)
    )
    weight, scales, biases = mx.quantize(dense, group_size=BLOCK_SIZE, bits=2)
    mx.eval(weight, scales, biases)
    return weight, scales, biases


def _synchronised(call: Callable[[], mx.array], inner: int) -> float:
    start = time.perf_counter()
    mx.eval([call() for _ in range(inner)])
    mx.synchronize()
    return time.perf_counter() - start


def plan_inner(calls: list[Callable[[], mx.array]]) -> int:
    """Pick one enqueue count that every arm of a case will share.

    Two things this has to get right. It warms each arm before estimating,
    because the first executions of a freshly JIT-compiled Metal library are not
    representative and an estimate taken there picks an ``inner`` far too small.
    And it sizes from the *fastest* arm, so the slowest arm cannot be the only
    one whose sample clears :data:`MIN_SAMPLE_SECONDS` -- a shared ``inner`` is
    what makes the variants of a case comparable to each other and not just each
    to the baseline.
    """
    fastest = float("inf")
    for call in calls:
        for _ in range(WARMUP_SAMPLES):
            _synchronised(call, MIN_INNER)
        fastest = min(fastest, _synchronised(call, MIN_INNER) / MIN_INNER)
    if not fastest > 0.0:
        raise FormatError("timing probe returned a non-positive duration")
    needed = int(-(-MIN_SAMPLE_SECONDS // fastest))
    return max(MIN_INNER, min(MAX_INNER, needed))


def measure_pair(
    baseline: Callable[[], mx.array],
    candidate: Callable[[], mx.array],
    *,
    inner: int,
    trials: int,
) -> tuple[list[float], list[float]]:
    """Time both arms in alternating AB/BA order, returning per-call seconds."""
    for _ in range(WARMUP_SAMPLES):
        _synchronised(baseline, inner)
        _synchronised(candidate, inner)

    baseline_times: list[float] = []
    candidate_times: list[float] = []
    for trial in range(trials):
        if trial % 2 == 0:
            first = _synchronised(baseline, inner)
            second = _synchronised(candidate, inner)
        else:
            second = _synchronised(candidate, inner)
            first = _synchronised(baseline, inner)
        baseline_times.append(first / inner)
        candidate_times.append(second / inner)
    return baseline_times, candidate_times


def _relative_error(got: mx.array, want: np.ndarray) -> float:
    left = np.asarray(got, dtype=np.float64)
    right = np.asarray(want, dtype=np.float64)
    scale = float(np.abs(right).max())
    if scale == 0.0:
        raise FormatError("reference output is identically zero; the case proves nothing")
    return float(np.abs(left - right).max() / scale)


def measure_case(case: OpCase, configs: list[Variant]) -> dict[str, object]:
    layout, codes, scales, codes_np, scales_np = build_packed_tensor(case)
    if codes.nbytes + scales.nbytes != layout.payload_bytes:
        raise FormatError(
            f"packed arrays occupy {codes.nbytes + scales.nbytes} bytes, "
            f"expected exactly {layout.payload_bytes}"
        )

    generator = np.random.default_rng(SEED + case.batch)
    activations = generator.normal(0.0, 1.0, size=(case.batch, case.columns)).astype(np.float32)
    x = mx.array(activations)
    mx.eval(x)
    expected = lut23_matmul(codes_np, scales_np, activations, layout=layout)

    weight, affine_scales, affine_biases = build_affine_baseline(case)

    def baseline() -> mx.array:
        return mx.quantized_matmul(
            x,
            weight,
            affine_scales,
            affine_biases,
            transpose=True,
            group_size=BLOCK_SIZE,
            bits=2,
        )

    mx.eval(baseline())
    mx.synchronize()

    def make_candidate(variant: Variant) -> Callable[[], mx.array]:
        def candidate() -> mx.array:
            return variant.call(codes, scales, x, layout=layout)

        return candidate

    # A variant that cannot run this tensor is named in the emitted document
    # rather than dropped silently, so the sweep's coverage stays auditable.
    runnable = [variant for variant in configs if variant.supports(layout)]
    unavailable = [variant.name for variant in configs if not variant.supports(layout)]
    if not runnable:
        raise FormatError(f"{case.label} has no runnable kernel configuration")
    candidates = [make_candidate(variant) for variant in runnable]

    # JIT every instantiation and prove each one correct before any of them is
    # allowed to post a timing.
    errors: list[float] = []
    for variant, candidate in zip(runnable, candidates, strict=True):
        first = candidate()
        mx.eval(first)
        mx.synchronize()
        error = _relative_error(first, expected)
        if not error <= BENCH_RTOL:
            raise FormatError(
                f"{case.label} batch {case.batch} {variant.name} disagrees with the "
                f"LUT23 reference by {error:.3g} relative"
            )
        errors.append(error)

    inner = plan_inner([baseline, *candidates])

    variants: list[dict[str, object]] = []
    for variant, candidate, error in zip(runnable, candidates, errors, strict=True):
        baseline_times, candidate_times = measure_pair(
            baseline, candidate, inner=inner, trials=TRIALS
        )
        base = np.asarray(baseline_times, dtype=np.float64)
        cand = np.asarray(candidate_times, dtype=np.float64)
        variants.append(variant.to_json() | {
            "inner_calls_per_sample": inner,
            "relative_error_vs_lut23_reference": error,
            "packed_seconds": quantile_summary(cand),
            "affine_2bit_seconds": quantile_summary(base),
            "speedup_vs_affine_2bit": float(np.median(base) / np.median(cand)),
            "bootstrap": paired_bootstrap_ratio(
                candidate_times, baseline_times, samples=100_000, seed=SEED
            ),
        })

    fastest = min(variants, key=lambda entry: entry["packed_seconds"]["median"])
    by_family: dict[str, dict[str, object]] = {}
    for entry in variants:
        family = str(entry["family"])
        current = by_family.get(family)
        if current is None or entry["packed_seconds"]["median"] < current["packed_seconds"]["median"]:
            by_family[family] = entry
    return {
        "label": case.label,
        "rows": case.rows,
        "columns": case.columns,
        "batch": case.batch,
        "tile": layout.tile,
        "tiles": layout.tiles,
        "groups_per_row": layout.groups_per_row,
        "packed_bytes": layout.payload_bytes,
        "affine_2bit_bytes": layout.groups * (BLOCK_SIZE * 2 // 8 + 4),
        "bits_per_weight": layout.bits_per_weight,
        "variants": variants,
        "unavailable_configs": unavailable,
        "best_config": fastest["config"],
        "best_speedup_vs_affine_2bit": fastest["speedup_vs_affine_2bit"],
        # The dispatch threshold in PackedLinear is read off these two numbers,
        # so they are recorded per case rather than inferred from the winner.
        "best_per_family": {
            family: {
                "config": entry["config"],
                "speedup_vs_affine_2bit": entry["speedup_vs_affine_2bit"],
            }
            for family, entry in by_family.items()
        },
    }


def measure_get_rows(rows: int, columns: int, token_counts: tuple[int, ...]) -> list[dict[str, object]]:
    """Time the embedding gather against MLX's dequantise-then-gather equivalent."""
    case = OpCase(label="embed_tokens", rows=rows, columns=columns, batch=1)
    layout, codes, scales, _, _ = build_packed_tensor(case)
    weight, affine_scales, affine_biases = build_affine_baseline(case)
    generator = np.random.default_rng(SEED + 3)

    results: list[dict[str, object]] = []
    for tokens in token_counts:
        indices_np = generator.integers(0, rows, size=tokens).astype(np.uint32)
        indices = mx.array(indices_np)
        mx.eval(indices)

        def baseline() -> mx.array:
            return mx.dequantize(
                weight[indices],
                affine_scales[indices],
                affine_biases[indices],
                group_size=BLOCK_SIZE,
                bits=2,
            )

        def candidate() -> mx.array:
            return tq1_get_rows(
                codes,
                scales,
                indices,
                groups_per_row=layout.groups_per_row,
                tile=layout.tile,
                dtype=mx.float32,
                threads=BENCH_THREADS,
                safe_clamp=False,
            )

        mx.eval(baseline(), candidate())
        mx.synchronize()
        inner = plan_inner([baseline, candidate])
        baseline_times, candidate_times = measure_pair(
            baseline, candidate, inner=inner, trials=TRIALS
        )
        base = np.asarray(baseline_times, dtype=np.float64)
        cand = np.asarray(candidate_times, dtype=np.float64)
        results.append({
            "tokens": tokens,
            "rows": rows,
            "columns": columns,
            "inner_calls_per_sample": inner,
            "packed_seconds": quantile_summary(cand),
            "affine_2bit_seconds": quantile_summary(base),
            "speedup_vs_affine_2bit": float(np.median(base) / np.median(cand)),
            "bootstrap": paired_bootstrap_ratio(
                candidate_times, baseline_times, samples=100_000, seed=SEED
            ),
        })
    return results


def measure_jit_cost(configs: list[Variant]) -> list[dict[str, object]]:
    """Cold-start cost of one Metal library per template tuple.

    Measured on a deliberately tiny tensor so the number is compilation, not
    arithmetic, and on a row count no other case uses so nothing is already
    cached by an earlier measurement. The batch is large enough that every
    prefill tiling has a full batch block to fill, since a tiling that runs
    ragged still compiles the same library but times differently.
    """
    case = OpCase(label="jit_probe", rows=512, columns=256, batch=512)
    layout, codes, scales, _, _ = build_packed_tensor(case)
    x = mx.array(
        np.random.default_rng(SEED + 11)
        .normal(0.0, 1.0, size=(case.batch, case.columns))
        .astype(np.float32)
    )
    mx.eval(codes, scales, x)
    mx.synchronize()

    results: list[dict[str, object]] = []
    for variant in configs:
        if not variant.supports(layout):
            continue
        start = time.perf_counter()
        first = variant.call(codes, scales, x, layout=layout)
        mx.eval(first)
        mx.synchronize()
        cold = time.perf_counter() - start

        start = time.perf_counter()
        warm_result = variant.call(codes, scales, x, layout=layout)
        mx.eval(warm_result)
        mx.synchronize()
        warm = time.perf_counter() - start

        results.append({
            "config": variant.name,
            "family": variant.family,
            "cold_seconds": cold,
            "warm_seconds": warm,
            "compile_seconds": cold - warm,
        })
    return results


def run(*, shapes: tuple[tuple[str, int, int], ...], batches: tuple[int, ...]) -> dict[str, object]:
    configs, unavailable = _configs()

    environment_before = capture_environment()
    jit = measure_jit_cost(configs)

    cases: list[dict[str, object]] = []
    for label, rows, columns in shapes:
        for batch in batches:
            case = OpCase(label=label, rows=rows, columns=columns, batch=batch)
            print(f"  {label:26s} {rows:6d}x{columns:<6d} batch={batch:<4d}", file=sys.stderr, end="")
            result = measure_case(case, configs)
            print(
                f"  best={result['best_config']:<28s} "
                f"{result['best_speedup_vs_affine_2bit']:.2f}x",
                file=sys.stderr,
            )
            cases.append(result)

    get_rows = measure_get_rows(248320, 5120, (1, 8, 64, 512))
    environment_after = capture_environment()

    return {
        "schema_version": SCHEMA_VERSION,
        "layout_name": LAYOUT_NAME,
        "layout_version": LAYOUT_VERSION,
        "seed": SEED,
        "trials_per_arm": TRIALS,
        "warmup_samples": WARMUP_SAMPLES,
        "min_sample_seconds": MIN_SAMPLE_SECONDS,
        "baseline": {
            "name": "mx.quantized_matmul",
            "mode": "affine",
            "bits": 2,
            "group_size": BLOCK_SIZE,
            "bits_per_weight": 2.25,
            "rationale": (
                "the representation prism-ml/Ternary-Bonsai-27B-mlx-2bit ships and the "
                "fastest path MLX offers for this matrix today"
            ),
        },
        "configurations_measured": [variant.to_json() for variant in configs],
        "configurations_unavailable": unavailable,
        "jit": jit,
        "matmul": cases,
        "get_rows": get_rows,
        "environment_before": environment_before,
        "environment_after": environment_after,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Benchmark the packed TQ1 Metal ops")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--batches",
        type=int,
        nargs="+",
        required=True,
        help=(
            "batch sizes to sweep. Batch 1 is token generation; the rest walk into "
            "prompt processing. Include at least one that is not a power of two: the "
            "GEMM compiles its batch bounds out when the batch block divides the batch, "
            "so a sweep of powers of two measures only the kernel that real prompt "
            "lengths mostly will not get"
        ),
    )
    parser.add_argument(
        "--shape",
        action="append",
        required=True,
        metavar="LABEL:ROWS:COLUMNS",
        help="a weight shape to measure; repeat the flag, or pass 'all' for the model's shapes",
    )
    args = parser.parse_args(argv)

    if args.shape == ["all"]:
        shapes = MODEL_SHAPES
    else:
        shapes = tuple(_parse_shape(entry) for entry in args.shape)

    result = run(shapes=shapes, batches=tuple(args.batches))
    write_json_atomic(args.output, result)
    json.dump({"matmul": result["matmul"], "get_rows": result["get_rows"]}, sys.stdout, indent=2)
    sys.stdout.write("\n")
    return 0


def _parse_shape(entry: str) -> tuple[str, int, int]:
    fields = entry.split(":")
    if len(fields) != 3:
        raise FormatError(f"expected LABEL:ROWS:COLUMNS, got {entry!r}")
    return fields[0], int(fields[1]), int(fields[2])


if __name__ == "__main__":
    raise SystemExit(main())
