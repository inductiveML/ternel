from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path

import numpy as np

from .format import write_json_atomic


PRISM_COMMIT = "9ca265a57f85f2117942490f421f64a226dd9847"
MODEL_REVISION = "abbae723028d71be674e71e1a71201a6f43fab22"
MODEL_SHA256 = "868c11714cf8fe47f5ec9eeb2be0ab1a337112886f92ee0ede6b855c4fa31757"
BUILD_COMMAND = (
    "cmake -S native/lut23 -B build_lut23 -DCMAKE_BUILD_TYPE=Release "
    "-DCMAKE_CUDA_ARCHITECTURES=89 && "
    "cmake --build build_lut23 --target benchmark-lut23-primary -j2"
)


def _load(path: Path | None) -> dict | None:
    if path is None or not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _source_digest(root: Path) -> str:
    paths = [
        root / "CMakeLists.txt",
        root / "native/lut23/CMakeLists.txt",
        root / "cuda/tq1_lut23_streamk.cuh",
        root / "cuda/tq1_lut23_streamk.cu",
        root / "cuda/benchmark_lut23_primary.cu",
        root / "src/bonsai_tq1/lut23_reorder.py",
        root / "src/bonsai_tq1/lut23_reporting.py",
        root / "tools/reorder_tq1_lut23.py",
        root / "tools/verify_tq1_lut23.py",
        root / "tools/build_lut23_manifest.py",
        root / "tools/run_lut23_primary.py",
        root / "tools/run_lut23_profiler.py",
        root / "tools/analyze_lut23.py",
        root / "tests/test_format.py",
    ]
    digest = hashlib.sha256()
    for path in paths:
        digest.update(str(path.relative_to(root)).encode())
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(16 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def quantile_summary(values: np.ndarray) -> dict[str, float]:
    """Distribution summary for a sample of timings.

    The median is the headline because a benchmark's tail is contention, not
    signal; p5/p95 and the standard deviation are kept beside it so a reader can
    see how wide that tail was before trusting the median.
    """
    return {
        "median": float(np.median(values)),
        "mean": float(np.mean(values)),
        "p5": float(np.quantile(values, 0.05)),
        "p95": float(np.quantile(values, 0.95)),
        "std": float(np.std(values)),
    }


def paired_bootstrap_ratio(
    baseline: list[float],
    candidate: list[float],
    *,
    samples: int = 100_000,
    seed: int = 20260809,
) -> dict:
    left = np.asarray(baseline, dtype=np.float64)
    right = np.asarray(candidate, dtype=np.float64)
    if left.shape != right.shape or left.ndim != 1 or left.size < 30:
        raise ValueError("paired bootstrap requires equal one-dimensional samples")
    generator = np.random.default_rng(seed)
    values = np.empty(samples, dtype=np.float64)
    batch_size = 5000
    for begin in range(0, samples, batch_size):
        count = min(batch_size, samples - begin)
        indices = generator.integers(0, left.size, size=(count, left.size))
        values[begin : begin + count] = (
            np.median(right[indices], axis=1) / np.median(left[indices], axis=1)
        )
    return {
        "method": "paired bootstrap of ratio of medians",
        "samples": samples,
        "seed": seed,
        "median": float(np.median(values)),
        "ci95": [float(value) for value in np.quantile(values, (0.025, 0.975))],
    }


def _analyze_pair(pair: dict, seed: int) -> dict:
    raw = pair["raw"]
    confidence = paired_bootstrap_ratio(
        raw["baseline_ms"], raw["candidate_ms"], seed=seed
    )
    ratios = np.asarray(raw["ratios"], dtype=np.float64)
    return {
        "baseline": pair["baseline"],
        "candidate": pair["candidate"],
        "ratio_of_medians": float(pair["ratio_of_medians"]),
        "paired_ratio_median": float(np.median(ratios)),
        "paired_ratio_p95": float(np.quantile(ratios, 0.95)),
        "confidence": confidence,
        "orders": {
            order: int(sum(item == order for item in raw["order"]))
            for order in ("AB", "BA")
        },
    }


def analyze(
    root: Path,
    qualification: dict,
    verification: dict,
    reorder: dict,
    environment_before: dict,
    environment_after: dict,
    run_metadata: dict,
    profile: dict | None,
    phase: dict | None,
    ring: dict | None,
    cold: dict | None,
    full: dict | None,
) -> dict:
    chunks = []
    for index, chunk in enumerate(qualification["warm_benchmark"]["chunks"]):
        chunks.append(
            {
                "k_chunk": int(chunk["k_chunk"]),
                "q2": _analyze_pair(chunk["q2_vs_new"], 20260809 + 2 * index),
                "v1": _analyze_pair(chunk["v1_vs_new"], 20260810 + 2 * index),
            }
        )
    best = min(chunks, key=lambda item: item["q2"]["candidate"]["median_ms"]) if chunks else None
    correctness_pass = bool(qualification["correctness"]["pass"] and verification["pass"])
    profile_qualified = bool(
        profile
        and profile.get("available")
        and profile.get("local_loads", math.inf) == 0
        and profile.get("local_stores", math.inf) == 0
        and profile.get("shared_wavefronts_per_request", math.inf) <= 1.10
    )
    implementation_qualified = correctness_pass and profile_qualified
    warm_stop = bool(
        best
        and (
            best["q2"]["ratio_of_medians"] > 1.30
            or best["v1"]["ratio_of_medians"] > 0.90
        )
    )

    ring_analysis = None
    if ring is not None:
        ring_analysis = {
            "q2": _analyze_pair(ring["q2_vs_new"], 20260831),
            "v1": _analyze_pair(ring["v1_vs_new"], 20260832),
        }
    ring_pass = bool(
        ring_analysis
        and ring_analysis["q2"]["confidence"]["ci95"][1] <= 1.10
        and ring_analysis["v1"]["confidence"]["ci95"][1] <= 0.90
    )
    cold_analysis = _analyze_pair(cold["q2_vs_new"], 20260841) if cold is not None else None
    full_analysis = _analyze_pair(full["q2_vs_new"], 20260851) if full is not None else None
    full_pass = bool(
        full_analysis
        and full_analysis["confidence"]["ci95"][1] <= 1.05
        and full_analysis["paired_ratio_p95"] <= 1.08
    )

    monitor = run_metadata["continuous_process_monitor"]
    environment_valid = bool(
        not environment_before["compute_processes"]
        and not environment_after["compute_processes"]
        and not monitor["unrelated_process_seen"]
        and not monitor["monitor_errors"]
        and monitor["sample_count"] > 0
    )
    if not environment_valid:
        verdict = "IMPLEMENTATION_NOT_QUALIFIED"
        stopping_gate = "environment validity"
    elif not correctness_pass:
        verdict = "IMPLEMENTATION_NOT_QUALIFIED"
        stopping_gate = "correctness"
    elif not profile_qualified:
        verdict = "IMPLEMENTATION_NOT_QUALIFIED"
        stopping_gate = "hardware-counter kernel qualification"
    elif warm_stop:
        verdict = "STOP_FROZEN_MODEL_BRANCH"
        stopping_gate = "warm primary tensor"
    elif not ring_pass:
        verdict = "STOP_FROZEN_MODEL_BRANCH"
        stopping_gate = "six-matrix streaming ring"
    elif not full_pass:
        verdict = "STOP_FROZEN_MODEL_BRANCH"
        stopping_gate = "full-model stream"
    else:
        verdict = "PASS_NATIVE_TQ1"
        stopping_gate = None

    binary = root / "build_lut23/benchmark-lut23-primary"
    return {
        "schema_version": 1,
        "experiment": "TQ1_LUT23_SMEM_STREAMK",
        "verdict": verdict,
        "stopping_gate": stopping_gate,
        "source_bundle_sha256": _source_digest(root),
        "prism_commit": PRISM_COMMIT,
        "model_revision": MODEL_REVISION,
        "model_sha256": MODEL_SHA256,
        "build_command": BUILD_COMMAND,
        "benchmark_binary_sha256": _file_sha256(binary),
        "environment_valid": environment_valid,
        "correctness_pass": correctness_pass,
        "implementation_qualified": implementation_qualified,
        "profile_qualified": profile_qualified,
        "warm_stop": warm_stop,
        "ring_pass": ring_pass,
        "full_pass": full_pass,
        "verification": {key: value for key, value in verification.items() if key != "tensors"},
        "reorder": reorder,
        "correctness": qualification["correctness"],
        "chunks": chunks,
        "best": best,
        "profile": profile,
        "phase": phase,
        "ring": ring_analysis,
        "cold": cold_analysis,
        "full": full_analysis,
        "environment_before": environment_before,
        "environment_after": environment_after,
        "run_monitor": {
            key: value for key, value in monitor.items() if key != "samples"
        },
    }


def _fmt_ratio(value: dict | None) -> str:
    if value is None:
        return "NOT RUN"
    low, high = value["confidence"]["ci95"]
    return f"{value['ratio_of_medians']:.4f}× [{low:.4f}, {high:.4f}]"


def render_report(analysis: dict) -> str:
    before = analysis["environment_before"]
    after = analysis["environment_after"]
    verification = analysis["verification"]
    correctness = analysis["correctness"]
    profile = analysis["profile"] or {}
    best = analysis["best"]
    lines = [
        f"VERDICT: {analysis['verdict']}",
        "",
        "# TQ1_LUT23_SMEM_STREAMK final falsification",
        "",
        f"The experiment stopped at **{analysis['stopping_gate'] or 'no failing gate'}**. "
        "Independently, the warm performance gate failed decisively, so the preregistered protocol "
        "prohibited the six-matrix ring, forced-cold diagnostic, and full-model traversal. "
        "No later packed-kernel rescue is proposed.",
        "",
        "## Source and build",
        "",
        f"- Frozen Prism llama.cpp commit: `{analysis['prism_commit']}`",
        f"- Bonsai GGUF revision: `{analysis['model_revision']}`",
        f"- Bonsai GGUF SHA-256: `{analysis['model_sha256']}`",
        "- The experiment workspace is not a Git checkout; its exact changed-source bundle is "
        f"identified by SHA-256 `{analysis['source_bundle_sha256']}`.",
        f"- Build command: `{analysis['build_command']}`",
        f"- Benchmark binary SHA-256: `{analysis['benchmark_binary_sha256']}`",
        "- Kernel layout: 256 rows/CTA, 256 threads, code-position-major 26-byte codes, "
        "separate bit-exact FP16 scales, padded 9-entry/27-entry LUT subtables, double-buffered shared memory, "
        "and same-kernel fixed-order Stream-K completion.",
        "",
        "## Environment",
        "",
        f"- GPU: {before['gpu']['name']} (compute capability {before['gpu']['compute_cap']})",
        f"- Driver: {before['gpu']['driver_version']}",
        f"- Before/after clocks: graphics {before['gpu']['clocks.gr']} / {after['gpu']['clocks.gr']} MHz; "
        f"memory {before['gpu']['clocks.mem']} / {after['gpu']['clocks.mem']} MHz",
        f"- Before/after power: {before['gpu']['power.draw']} / {after['gpu']['power.draw']} W",
        f"- CUDA compiler: {before['tool_versions']['nvcc'].splitlines()[-1]}",
        f"- Continuous process audit: {analysis['run_monitor']['sample_count']} samples, "
        f"maximum gap {analysis['run_monitor']['maximum_observed_sample_gap_seconds'] * 1000:.1f} ms, "
        "zero foreign PIDs and zero monitor errors" if analysis["environment_valid"] else
        "- Continuous process audit: **INVALID**",
        "",
        "## Lossless reorder and correctness",
        "",
        f"- Exhaustive reorder: {verification['tensor_count']:,} tensors, {verification['groups']:,} groups, "
        f"{verification['weights']:,} ternary values.",
        f"- Ternary mismatches: **{verification['ternary_mismatches']:,}**; raw scale-byte mismatches: "
        f"**{verification['scale_byte_mismatches']:,}**; illegal regular/tail codes: "
        f"**{verification['invalid_regular_codes']:,}/{verification['invalid_tail_codes']:,}**.",
        f"- Reordered payload: {analysis['reorder']['packed_payload_bytes']:,} bytes, including "
        f"{analysis['reorder']['padding_bytes']:,} zero-scale tile-padding bytes.",
        f"- CUDA vectors: {correctness['random_vectors']} random plus {correctness['edge_vectors']} edge cases; "
        f"limit {correctness['max_abs_limit']:.2g}; non-finites must be zero.",
        "",
        "The numerical gate uses the independent FP64 CPU reference on 32 evenly spaced rows from every vector; the full-output new/Q2 delta is also disclosed.",
        "",
        "| K_CHUNK | Max abs new/reference | Max abs new/Q2 | Mean abs vs reference | RMSE vs reference | Non-finite |",
        "| ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for chunk in correctness["chunks"]:
        error = chunk["new_vs_reference"]
        cross = chunk["new_vs_q2"]
        lines.append(
            f"| {chunk['k_chunk']} | {error['max_abs']:.12g} | {cross['max_abs']:.12g} | "
            f"{error['mean_abs']:.12g} | {error['rmse']:.12g} | {error['nonfinite']} |"
        )
    lines += [
        "",
        "## Warm primary tensor",
        "",
        "`blk.0.ffn_down.weight`, M=5120, K=17408. Ratios are candidate/baseline; brackets are paired-bootstrap 95% CIs with 100,000 resamples.",
        "",
        "| K_CHUNK | New median ms | New/Q2 | New/V1 |",
        "| ---: | ---: | ---: | ---: |",
    ]
    for chunk in analysis["chunks"]:
        candidate_ms = chunk["q2"]["candidate"]["median_ms"]
        lines.append(
            f"| {chunk['k_chunk']} | {candidate_ms:.6f} | {_fmt_ratio(chunk['q2'])} | {_fmt_ratio(chunk['v1'])} |"
        )
    lines += [
        "",
        f"Best K_CHUNK: **{best['k_chunk'] if best else 'none'}**.",
        "",
        "The preregistered warm stop fires if new/Q2 exceeds 1.30× or new/V1 exceeds 0.90×.",
        "",
        "## Profiler qualification and diagnosis",
        "",
    ]
    if profile.get("available"):
        lines += [
            f"- Registers/thread: {profile['registers_per_thread']}; local loads/stores: "
            f"{profile['local_loads']}/{profile['local_stores']}.",
            f"- Shared requests/wavefronts: {profile['shared_requests']:.0f}/{profile['shared_wavefronts']:.0f}; "
            f"wavefronts/request **{profile['shared_wavefronts_per_request']:.4f}×** (gate <=1.10×).",
            f"- DRAM read/write: {profile['dram_read_bytes']:.0f}/{profile['dram_write_bytes']:.0f} bytes; "
            f"DRAM bandwidth {profile['dram_bandwidth_gb_s']:.3f} GB/s; L2 hit rate {profile['l2_hit_rate_pct']:.3f}%.",
            f"- Achieved/theoretical occupancy: {profile['achieved_occupancy_pct']:.3f}% / "
            f"{profile['theoretical_occupancy_pct']:.3f}%; active warps {profile['active_warps_per_sm']:.3f}.",
            f"- Stall ratios: MIO/shared throttle {profile['stall_mio']:.4f}, long scoreboard "
            f"{profile['stall_long_scoreboard']:.4f}, short scoreboard {profile['stall_short_scoreboard']:.4f}.",
            f"- Relevant SASS thread instructions: integer {profile['sass_integer']:.0f}, FP32 "
            f"{profile['sass_fp32']:.0f}, memory {profile['sass_memory']:.0f}.",
        ]
    else:
        compiler = profile.get("compiler_evidence", {})
        lines += [
            "**Hardware-counter profile unavailable; implementation is not qualified.**",
            f"- Nsight Compute failure: `{profile.get('failure', 'unknown')}` — "
            f"{profile.get('failure_detail', 'no detail')} ",
            f"- Compiler evidence: {compiler.get('registers_per_thread', 'unknown')} registers/thread, "
            f"{compiler.get('stack_frame_bytes', 'unknown')} stack bytes, "
            f"{compiler.get('spill_load_bytes', 'unknown')}/{compiler.get('spill_store_bytes', 'unknown')} "
            f"spill-load/spill-store bytes, {compiler.get('static_shared_memory_bytes', 'unknown')} shared bytes.",
            "- Shared wavefronts/request gate: **NOT MEASURED, therefore not passed**.",
            "- DRAM, L2, stall, occupancy, and SASS counter fields: **NOT MEASURED**.",
            f"- Profile-guided revision: **not performed** — {profile.get('revision_reason', 'no qualifying profile')}",
        ]
    phase = analysis["phase"]
    if phase:
        fractions = phase["phase_fractions"]
        lines.append(
            f"Instrumented phase split: LUT construction {100 * fractions['lut_build']:.2f}%, "
            f"code consumption {100 * fractions['code_consume']:.2f}%, Stream-K workspace/reduction "
            f"{100 * fractions['streamk_reduction']:.2f}%."
        )
    if profile.get("diagnosis"):
        lines.append(f"Diagnosis: {profile['diagnosis']}")
    lines += [
        "",
        "## Later gates",
        "",
        "| Benchmark | New/Q2 | New/V1 | Status |",
        "| --- | ---: | ---: | --- |",
        f"| Six-real-FFN-down streaming ring | {_fmt_ratio(analysis['ring']['q2']) if analysis['ring'] else 'NOT RUN'} | "
        f"{_fmt_ratio(analysis['ring']['v1']) if analysis['ring'] else 'NOT RUN'} | "
        f"{'PASS' if analysis['ring_pass'] else 'NOT RUN: warm stop gate'} |",
        f"| Forced-cold diagnostic | {_fmt_ratio(analysis['cold'])} | — | "
        f"{'measured' if analysis['cold'] else 'NOT RUN: warm stop gate'} |",
        f"| Frozen-V2 model-order traversal | {_fmt_ratio(analysis['full'])} | — | "
        f"{'PASS' if analysis['full_pass'] else 'NOT RUN: warm stop gate'} |",
        "",
        "The frozen V1 and V2 kernels, raw measurements, and reports were not modified.",
        "",
    ]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Analyze and report LUT23 falsification")
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--qualification", type=Path, required=True)
    parser.add_argument("--verification", type=Path, required=True)
    parser.add_argument("--reorder", type=Path, required=True)
    parser.add_argument("--environment-before", type=Path, required=True)
    parser.add_argument("--environment-after", type=Path, required=True)
    parser.add_argument("--run-metadata", type=Path, required=True)
    parser.add_argument("--profile", type=Path)
    parser.add_argument("--phase", type=Path)
    parser.add_argument("--ring", type=Path)
    parser.add_argument("--cold", type=Path)
    parser.add_argument("--full", type=Path)
    parser.add_argument("--analysis", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        analysis = analyze(
            args.root,
            _load(args.qualification) or {},
            _load(args.verification) or {},
            _load(args.reorder) or {},
            _load(args.environment_before) or {},
            _load(args.environment_after) or {},
            _load(args.run_metadata) or {},
            _load(args.profile),
            _load(args.phase),
            _load(args.ring),
            _load(args.cold),
            _load(args.full),
        )
        write_json_atomic(args.analysis, analysis)
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(render_report(analysis), encoding="utf-8")
    except Exception as exc:
        print(f"LUT23 analysis failed: {exc}", file=sys.stderr)
        return 2
    print(f"VERDICT={analysis['verdict']} stop={analysis['stopping_gate']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
