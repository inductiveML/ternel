"""A6: how far the K axis should be split, measured in the regime decode is in.

``bench_ops`` times ``mx.eval([call() for _ in range(inner)])`` -- ``inner``
*independent* calls sharing an input and writing distinct outputs. MLX encodes
those with a concurrent dispatch encoder and inserts barriers only on buffer
hazards, so with no hazard between them they overlap freely and the machine
fills up however few threadgroups any one dispatch launches. That is the right
shape for a throughput question and the wrong one for this question. A decoded
token issues 497 matmuls in a single read-after-write chain, each waiting on the
one before, so every dispatch meets the GPU alone.

LUT23 walks a tensor's groups strictly in series inside each threadgroup, and
launches one threadgroup per row tile. Alone, a 48-row tensor therefore runs one
chain of 40 group iterations on one core of forty. This file measures each shape
both ways -- overlapped, as ``bench_ops`` does, and in a forced dependency chain
with the chain link's own cost subtracted -- and sweeps every legal split of the
K axis at each of the batch sizes LUT23 serves. The ratio between the two modes
is the serialisation factor: how much more a dispatch costs when nothing else is
resident to hide it.

``mx.quantized_matmul`` runs beside every cell as the same baseline
``bench_ops`` uses. It already splits K, so its serialisation factor is the
control: whatever part of the chained penalty is MLX's encoder rather than this
kernel shows up in both arms.

``modules.select_lut23`` is replayed against this file by
``tests/test_mlx_modules.py``, the same way ``select_gemm`` is replayed against
the A5 sweep, so the shipped rule is checked against the measurement rather than
described by it.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections.abc import Callable
from dataclasses import replace
from pathlib import Path

import mlx.core as mx
import numpy as np

from bonsai_tq1.format import BLOCK_SIZE, FormatError, write_json_atomic
from bonsai_tq1.lut23_reporting import quantile_summary

from . import LAYOUT_NAME, LAYOUT_VERSION
from .bench_ops import SEED, OpCase, build_affine_baseline, build_packed_tensor, relative_error
from .environment import capture_environment, exclusive_gpu
from .kernel_gate import NARROWED_RELATIVE_TOLERANCE
from .kernels import MatmulConfig, tq1_matmul
from .layout import DISPATCHES_PER_TOKEN, MODEL_LINEAR_SHAPES, PackedTensorLayout
from .modules import select_lut23_tile
from .reference import lut23_matmul

SCHEMA_VERSION = 1

# The dtype the artifact's config declares and mlx-lm loads. Only one is swept:
# the split changes how many threadgroups run, not what they read.
#
# It is handed to ``build_affine_baseline`` as well, and must be. The dtype is a
# kernel template parameter on both sides: MLX promotes bfloat16 activations
# against float16 scales to float32, so a baseline left at float16 would run its
# float32 kernel against the packed path's bfloat16 one -- a control in a
# heavier dtype than the shipped 2-bit repo, whose scales are bfloat16, actually
# runs. The first ``results/mlx/a6_split_k.json`` was measured that way; its
# packed-against-packed columns are unaffected, since both arms are the same
# kernel in the same dtype, but its affine column is not comparable and the
# sweep is re-run.
ACTIVATION_DTYPE = mx.bfloat16
ACTIVATION_DTYPE_NAME = "bfloat16"

# How far past the machine's forty cores a split is allowed to go before the
# sweep stops measuring it. Wide enough that the flat part of every curve is
# visible on both sides of its knee, so the rule is chosen against a measured
# turn rather than against the edge of the sweep.
MAX_THREADGROUPS = 2560


def divisors(count: int) -> tuple[int, ...]:
    return tuple(k for k in range(1, count + 1) if count % k == 0)


def legal_splits(layout: PackedTensorLayout, *, threadgroups: int) -> tuple[int, ...]:
    """Splits of this tensor's K axis that stay inside the sweep's ceiling.

    A divisor of ``groups_per_row``, always: a ragged split leaves one
    threadgroup holding a full chunk of groups and another holding none, and a
    dispatch ends when its slowest threadgroup does.

    ``k_split=1`` is not one of the splits being explored -- it is the shipped
    configuration everything else is measured against -- so it is always
    returned. ``MAX_THREADGROUPS`` bounds how far a split may push the grid, and
    at prefill batches the grid can already be past it before any split: lm_head
    presents 970 tiles times four batch tiles, 3880 threadgroups, at batch 16.
    Saying "no split is a candidate for this shape" is a result, and the same one
    the dispatch rule reaches; dropping the control on the way to saying it would
    leave the shape unmeasured instead.
    """
    return tuple(
        k for k in divisors(layout.groups_per_row)
        if k == 1 or threadgroups * k <= MAX_THREADGROUPS
    )


def timed(build: Callable[[], mx.array | list[mx.array]]) -> float:
    """Seconds of GPU time for one already-constructed graph.

    The graph is built before the clock starts, unlike ``bench_ops``, which
    times the Python construction too. Here that construction differs between
    the arms -- a chain of thirty-two builds four ops per link where an
    overlapped batch builds one -- so leaving it in would charge the chained arm
    for work the GPU never does.
    """
    graph = build()
    mx.synchronize()
    start = time.perf_counter()
    mx.eval(graph)
    mx.synchronize()
    return time.perf_counter() - start


def overlapped(call: Callable[[mx.array], mx.array], x: mx.array, *, enqueued: int) -> float:
    """``bench_ops``' shape: independent calls, no hazard, free to run at once."""
    return timed(lambda: [call(x) for _ in range(enqueued)])


def chained(
    call: Callable[[mx.array], mx.array], x: mx.array, zero: mx.array, *, enqueued: int
) -> float:
    """Decode's shape: every call reads what the call before it wrote.

    The link adds zero, so each matmul sees bit-identical activations and does
    identical work; only the hazard is new. The zero arrives as an array rather
    than a literal so the graph builder cannot fold the dependency away.
    """

    def build() -> mx.array:
        current = x
        for _ in range(enqueued):
            current = x + call(current)[:, :1] * zero
        return current

    return timed(build)


def link_only(x: mx.array, zero: mx.array, *, enqueued: int) -> float:
    """The same chain with the matmul removed, so its cost can be subtracted."""

    def build() -> mx.array:
        current = x
        for _ in range(enqueued):
            current = x + current[:, :1] * zero
        return current

    return timed(build)


def measure_case(
    case: OpCase, *, enqueued: int, trials: int, warmup: int, max_foreign_gpu_share: float,
    contention: list[dict[str, object]],
) -> list[dict[str, object]]:
    """Every legal split of one shape at one batch, against affine, both modes."""
    layout, codes, scales, codes_np, scales_np = build_packed_tensor(case)
    base = select_lut23_tile(layout, case.batch)
    batch_tiles = -(-case.batch // base.batch_tile)
    threadgroups = layout.tiles * batch_tiles
    splits = legal_splits(layout, threadgroups=threadgroups)

    activations = (
        np.random.default_rng(SEED + case.batch)
        .normal(0.0, 1.0, size=(case.batch, case.columns))
        .astype(np.float32)
    )
    expected = lut23_matmul(codes_np, scales_np, activations, layout=layout)
    x = mx.array(activations).astype(ACTIVATION_DTYPE)
    zero = mx.array(0.0).astype(ACTIVATION_DTYPE)
    mx.eval(x, zero)

    weight, affine_scales, affine_biases = build_affine_baseline(case, dtype=ACTIVATION_DTYPE)

    def affine(vector: mx.array) -> mx.array:
        return mx.quantized_matmul(
            vector, weight, affine_scales, affine_biases,
            transpose=True, group_size=BLOCK_SIZE, bits=2,
        )

    def packed_for(config: MatmulConfig) -> Callable[[mx.array], mx.array]:
        def call(vector: mx.array) -> mx.array:
            return tq1_matmul(
                codes, scales, vector,
                groups_per_row=layout.groups_per_row, tile=layout.tile, config=config,
            )

        return call

    arms: dict[str, Callable[[mx.array], mx.array]] = {"affine_2bit": affine}
    configs: dict[str, MatmulConfig] = {}
    errors: dict[str, float] = {}
    for split in splits:
        config = replace(base, k_split=split)
        call = packed_for(config)
        # A kernel made fast by being wrong does not get to post a number.
        probe = call(x)
        mx.eval(probe)
        mx.synchronize()
        error = relative_error(probe, expected)
        if error > NARROWED_RELATIVE_TOLERANCE[ACTIVATION_DTYPE_NAME]:
            raise FormatError(
                f"{case.label} batch={case.batch} k_split={split}: "
                f"relative error {error:.3e} over the {ACTIVATION_DTYPE_NAME} bound"
            )
        name = config.benchmark_name
        arms[name] = call
        configs[name] = config
        errors[name] = error

    for call in arms.values():
        for _ in range(warmup):
            overlapped(call, x, enqueued=enqueued)
            chained(call, x, zero, enqueued=enqueued)
    for _ in range(warmup):
        link_only(x, zero, enqueued=enqueued)

    samples: dict[str, list[float]] = {
        f"{name}:{mode}": [] for name in arms for mode in ("overlapped", "chained")
    }
    samples["link_only"] = []
    with exclusive_gpu(
        f"{case.label} {case.rows}x{case.columns} batch={case.batch}",
        max_foreign_share=max_foreign_gpu_share,
        record=contention,
    ):
        # Round-robin rather than arm by arm, so a thermal ramp or a background
        # burst lands on every arm instead of on whichever was measured while it
        # happened. The same discipline ``bench_ops`` uses for its pairs.
        for _ in range(trials):
            for name, call in arms.items():
                samples[f"{name}:overlapped"].append(overlapped(call, x, enqueued=enqueued))
                samples[f"{name}:chained"].append(chained(call, x, zero, enqueued=enqueued))
            samples["link_only"].append(link_only(x, zero, enqueued=enqueued))

    link = quantile_summary(samples["link_only"])["median"] / enqueued
    variants: list[dict[str, object]] = []
    for name in arms:
        over = quantile_summary(samples[f"{name}:overlapped"])
        chain = quantile_summary(samples[f"{name}:chained"])
        per_overlapped = over["median"] / enqueued
        per_chained = chain["median"] / enqueued - link
        config = configs.get(name)
        variants.append(
            {
                "config": name,
                "family": "affine_2bit" if config is None else "lut23",
                "k_split": None if config is None else config.k_split,
                "batch_tile": None if config is None else config.batch_tile,
                "threadgroups": None if config is None else threadgroups * config.k_split,
                "overlapped_seconds": {**over, "per_call": per_overlapped},
                "chained_seconds": {**chain, "per_call": per_chained},
                "serialisation_factor": per_chained / per_overlapped,
                "relative_error": errors.get(name),
            }
        )

    del codes, scales, weight, affine_scales, affine_biases
    return variants


def run(
    *,
    batches: tuple[int, ...],
    enqueued: int,
    trials: int,
    warmup: int,
    max_foreign_gpu_share: float,
) -> dict[str, object]:
    environment_before = capture_environment()
    contention: list[dict[str, object]] = []
    cases: list[dict[str, object]] = []

    for shape in MODEL_LINEAR_SHAPES:
        for batch in batches:
            case = OpCase(label=shape.label, rows=shape.rows, columns=shape.columns, batch=batch)
            layout = shape.layout()
            print(
                f"  {shape.label:26s} {shape.rows:6d}x{shape.columns:<6d} batch={batch:<4d}",
                file=sys.stderr, end="",
            )
            variants = measure_case(
                case, enqueued=enqueued, trials=trials, warmup=warmup,
                max_foreign_gpu_share=max_foreign_gpu_share, contention=contention,
            )
            packed = [v for v in variants if v["family"] == "lut23"]
            unsplit = next(v for v in packed if v["k_split"] == 1)
            best = min(packed, key=lambda v: float(v["chained_seconds"]["per_call"]))
            print(
                f"  best k={best['k_split']:<3d} "
                f"{float(unsplit['chained_seconds']['per_call']) / float(best['chained_seconds']['per_call']):.2f}x "
                f"over unsplit",
                file=sys.stderr,
            )
            cases.append(
                {
                    "label": shape.label,
                    "modules": list(shape.modules),
                    "rows": shape.rows,
                    "columns": shape.columns,
                    "tile": layout.tile,
                    "tiles": layout.tiles,
                    "groups_per_row": layout.groups_per_row,
                    "batch": batch,
                    "dispatches_per_token": shape.dispatches_per_token,
                    "activation_dtype": ACTIVATION_DTYPE_NAME,
                    "variants": variants,
                }
            )

    environment_after = capture_environment()
    return {
        "schema_version": SCHEMA_VERSION,
        "layout_name": LAYOUT_NAME,
        "layout_version": LAYOUT_VERSION,
        "seed": SEED,
        "enqueued_per_sample": enqueued,
        "trials_per_arm": trials,
        "warmup_samples": warmup,
        "activation_dtype": ACTIVATION_DTYPE_NAME,
        "dispatches_per_token": DISPATCHES_PER_TOKEN,
        "max_threadgroups_swept": MAX_THREADGROUPS,
        "max_foreign_gpu_share": max_foreign_gpu_share,
        "gpu_contention": contention,
        "baseline": {
            "name": "mx.quantized_matmul",
            "mode": "affine",
            "bits": 2,
            "group_size": BLOCK_SIZE,
            "bits_per_weight": 2.25,
            "rationale": (
                "already splits K, so its serialisation factor is the control for "
                "whatever part of the chained penalty belongs to MLX's encoder rather "
                "than to this kernel"
            ),
        },
        "matmul": cases,
        "environment_before": environment_before,
        "environment_after": environment_after,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Sweep the LUT23 K-axis split (A6)")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--batches", type=int, nargs="+", required=True,
        help=(
            "batch sizes to sweep. Batch 1 is token generation, which is what the "
            "split exists for; the rest walk to where select_gemm takes over, so the "
            "rule is measured over the whole range LUT23 serves rather than assumed "
            "to extrapolate"
        ),
    )
    parser.add_argument(
        "--enqueued", type=int, required=True,
        help="matmuls per timed sample, in both the overlapped and the chained arm",
    )
    parser.add_argument("--trials", type=int, required=True, help="timed samples per arm")
    parser.add_argument("--warmup", type=int, required=True, help="untimed samples per arm")
    parser.add_argument(
        "--max-foreign-gpu-share", type=float, required=True,
        help="void the run if another process exceeds this share of the GPU while timing",
    )
    args = parser.parse_args(argv)
    if min(args.enqueued, args.trials, args.warmup) < 1:
        raise FormatError("enqueued, trials and warmup must all be at least one")
    if not args.max_foreign_gpu_share > 0.0:
        raise FormatError(f"max foreign share must be positive, got {args.max_foreign_gpu_share}")

    result = run(
        batches=tuple(args.batches),
        enqueued=args.enqueued,
        trials=args.trials,
        warmup=args.warmup,
        max_foreign_gpu_share=args.max_foreign_gpu_share,
    )
    write_json_atomic(args.output, result)
    json.dump({"matmul": result["matmul"]}, sys.stdout, indent=2)
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
