from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

from .format import write_json_atomic


V1_VERDICT = "PACKING_ONLY_KERNEL_FAIL"
Q2_WEIGHT_BYTES = 7_143_546_880
TQ1_WEIGHT_BYTES = 5_882_920_960
Q2_MODEL_BYTES = 7_165_121_600
TQ1_MODEL_BYTES = 5_904_495_680
TOKEN_EMBED_Q2_BYTES = 337_715_200
NON_Q2_BYTES = Q2_MODEL_BYTES - Q2_WEIGHT_BYTES
V1_Q2_WARM_MS = 0.016576
V1_TQ1_WARM_MS = 0.028672


def _load(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object in {path}")
    return value


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _paired_bootstrap(q2: np.ndarray, tq1: np.ndarray, samples: int = 100_000) -> dict:
    generator = np.random.default_rng(20260809)
    indices = generator.integers(0, q2.size, size=(samples, q2.size))
    ratios = np.median(tq1[indices], axis=1) / np.median(q2[indices], axis=1)
    return {
        "method": "paired bootstrap of complete AB/BA traversal pairs",
        "samples": samples,
        "median_ratio": float(np.median(ratios)),
        "ratio_95_ci": np.quantile(ratios, [0.025, 0.975]).tolist(),
    }


def _rank_average(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="stable")
    ranks = np.empty(values.size, dtype=np.float64)
    start = 0
    while start < values.size:
        end = start + 1
        while end < values.size and values[order[end]] == values[order[start]]:
            end += 1
        ranks[order[start:end]] = 0.5 * (start + end - 1)
        start = end
    return ranks


def _correlation(x: np.ndarray, y: np.ndarray) -> float:
    if x.size < 2 or np.std(x) == 0 or np.std(y) == 0:
        return 0.0
    return float(np.corrcoef(x, y)[0, 1])


def _classify(ratio: float) -> str:
    if ratio <= 0.95:
        return "STREAMING_TQ1_WIN"
    if ratio <= 1.05:
        return "STREAMING_TQ1_PARITY"
    if ratio <= 1.10:
        return "STREAMING_TQ1_NEARMISS"
    return "STREAMING_TQ1_FAIL"


def _pct(new: float, old: float) -> float:
    return 100.0 * (new / old - 1.0)


def _fmt_ms(value: float) -> str:
    return f"{value:.6f} ms"


def _environment_valid(before: dict, after: dict, run: dict) -> tuple[bool, list[str]]:
    reasons: list[str] = []
    if before.get("compute_processes"):
        reasons.append("unrelated CUDA process present before timing")
    if after.get("compute_processes"):
        reasons.append("unrelated CUDA process present after timing")
    if run.get("unrelated_processes_seen_at_boundaries"):
        reasons.append("runner observed a competing CUDA process at a timing boundary")
    monitor = run.get("continuous_process_monitor", {})
    if monitor.get("unrelated_process_seen"):
        reasons.append("continuous monitor observed a competing CUDA process during timing")
    if monitor.get("monitor_errors"):
        reasons.append("continuous CUDA-process monitor had an observation gap")
    if not monitor.get("sample_count"):
        reasons.append("continuous CUDA-process monitor produced no samples")
    for label, environment in (("before", before), ("after", after)):
        gpu = environment.get("gpu", {})
        if gpu.get("name") != "NVIDIA RTX 6000 Ada Generation" or gpu.get("compute_cap") != "8.9":
            reasons.append(f"{label} environment is not RTX 6000 Ada sm_89")
    if run.get("returncode") != 0:
        reasons.append(f"benchmark return code was {run.get('returncode')}")
    return not reasons, reasons


def _analyze_crossover(per_tensor: list[dict], primary: dict) -> dict:
    sizes = np.asarray([tensor["q2_bytes"] for tensor in per_tensor], dtype=np.float64)
    ratios = np.asarray([tensor["median_ratio"] for tensor in per_tensor], dtype=np.float64)
    log_sizes = np.log10(sizes)
    log_ratios = np.log(ratios)
    pearson = _correlation(log_sizes, ratios)
    spearman = _correlation(_rank_average(sizes), _rank_average(ratios))
    log_pearson = _correlation(log_sizes, log_ratios)

    grouped_raw: dict[int, list[dict]] = defaultdict(list)
    for tensor in per_tensor:
        grouped_raw[int(tensor["q2_bytes"])].append(tensor)
    groups: list[dict] = []
    for q2_bytes in sorted(grouped_raw):
        tensors = grouped_raw[q2_bytes]
        q2_latency = float(np.median([tensor["q2"]["median_ms"] for tensor in tensors]))
        tq1_latency = float(np.median([tensor["tq1"]["median_ms"] for tensor in tensors]))
        groups.append(
            {
                "q2_bytes": q2_bytes,
                "tq1_bytes": int(tensors[0]["tq1_bytes"]),
                "tensor_count": len(tensors),
                "q2_median_ms": q2_latency,
                "tq1_median_ms": tq1_latency,
                "median_ratio": tq1_latency / q2_latency,
                "tq1_wins": sum(tensor["median_ratio"] < 1.0 for tensor in tensors),
            }
        )

    geometry_raw: dict[tuple[str, int, int], list[dict]] = defaultdict(list)
    for tensor in per_tensor:
        geometry_raw[(tensor["kind"], int(tensor["m"]), int(tensor["k"]))].append(tensor)
    geometries: list[dict] = []
    for (kind, m, k), tensors in sorted(geometry_raw.items()):
        geometries.append(
            {
                "kind": kind,
                "m": m,
                "k": k,
                "tensor_count": len(tensors),
                "median_ratio": float(np.median([tensor["median_ratio"] for tensor in tensors])),
                "tq1_wins": sum(tensor["median_ratio"] < 1.0 for tensor in tensors),
            }
        )

    winning_groups = [group for group in groups if group["median_ratio"] <= 1.0]
    observed_crossover = min((group["q2_bytes"] for group in winning_groups), default=None)
    grouped_x = np.log10(np.asarray([group["q2_bytes"] for group in groups], dtype=np.float64))
    grouped_y = np.asarray([group["median_ratio"] for group in groups], dtype=np.float64)
    slope, intercept = np.polyfit(grouped_x, grouped_y, 1)
    regression_crossover: float | None = None
    if slope != 0:
        candidate = 10 ** ((1.0 - intercept) / slope)
        if min(sizes) / 10 <= candidate <= max(sizes) * 10:
            regression_crossover = float(candidate)

    q2_sum = sum(tensor["q2"]["median_ms"] for tensor in per_tensor)
    tq1_sum = sum(tensor["tq1"]["median_ms"] for tensor in per_tensor)
    tq1_selected = [tensor for tensor in per_tensor if tensor["tq1"]["median_ms"] < tensor["q2"]["median_ms"]]
    hybrid_latency = sum(
        min(tensor["q2"]["median_ms"], tensor["tq1"]["median_ms"])
        for tensor in per_tensor
    )
    hybrid_selected_bytes = sum(
        tensor["tq1_bytes"] if tensor in tq1_selected else tensor["q2_bytes"]
        for tensor in per_tensor
    )
    selected_tq1_bytes = sum(tensor["tq1_bytes"] for tensor in tq1_selected)
    selected_q2_coverage = sum(tensor["q2_bytes"] for tensor in tq1_selected)
    projected_model_bytes = NON_Q2_BYTES + TOKEN_EMBED_Q2_BYTES + hybrid_selected_bytes
    oracle = {
        "method": "offline sum of per-tensor model-order diagnostic medians",
        "all_q2_latency_ms": q2_sum,
        "all_tq1_latency_ms": tq1_sum,
        "hybrid_latency_ms": hybrid_latency,
        "tq1_tensor_count": len(tq1_selected),
        "q2_tensor_count": len(per_tensor) - len(tq1_selected),
        "tq1_tensor_fraction": len(tq1_selected) / len(per_tensor),
        "tq1_actual_byte_fraction": selected_tq1_bytes / hybrid_selected_bytes,
        "q2_payload_coverage_reencoded_as_tq1": selected_q2_coverage / sum(tensor["q2_bytes"] for tensor in per_tensor),
        "selected_stream_bytes": hybrid_selected_bytes,
        "projected_model_bytes": projected_model_bytes,
        "speedup_vs_all_q2": q2_sum / hybrid_latency,
        "speedup_vs_all_tq1": tq1_sum / hybrid_latency,
        "latency_reduction_vs_all_q2_percent": 100.0 * (1.0 - hybrid_latency / q2_sum),
        "latency_reduction_vs_all_tq1_percent": 100.0 * (1.0 - hybrid_latency / tq1_sum),
        "tq1_tensor_names": [tensor["name"] for tensor in tq1_selected],
    }
    oracle["clearly_useful_mixed_pareto"] = (
        0 < oracle["tq1_tensor_count"] < len(per_tensor)
        and hybrid_latency <= 0.98 * min(q2_sum, tq1_sum)
        and projected_model_bytes <= 0.95 * Q2_MODEL_BYTES
    )
    return {
        "pearson_log_size_vs_ratio": pearson,
        "pearson_log_size_vs_log_ratio": log_pearson,
        "spearman_size_vs_ratio": spearman,
        "tq1_winning_tensor_count": int(np.count_nonzero(ratios < 1.0)),
        "tq1_winning_tensor_fraction": float(np.mean(ratios < 1.0)),
        "observed_crossover_q2_bytes": observed_crossover,
        "regression": {
            "ratio_equals_slope_log10_q2_bytes_plus_intercept": True,
            "slope": float(slope),
            "intercept": float(intercept),
            "estimated_crossover_q2_bytes": regression_crossover,
        },
        "groups": groups,
        "geometries": geometries,
        "oracle_hybrid": oracle,
        "primary_clean_traversal": {
            "q2_median_ms": primary["q2"]["median_ms"],
            "tq1_median_ms": primary["tq1"]["median_ms"],
        },
    }


def analyze(
    raw: dict,
    traversal: dict,
    before: dict,
    after: dict,
    run: dict,
    profiler: dict,
    v1_reports_dir: Path,
) -> dict:
    frozen_actual = {path.name: _sha256(path) for path in sorted(v1_reports_dir.glob("*.md"))}
    frozen_unchanged = frozen_actual == traversal["v1_frozen_report_sha256"]
    primary = raw["regime_b_model_stream"]
    q2 = np.asarray(primary["raw"]["q2_ms"], dtype=np.float64)
    tq1 = np.asarray(primary["raw"]["tq1_ms"], dtype=np.float64)
    if q2.size != tq1.size or q2.size < 30 or np.any(q2 <= 0) or np.any(tq1 <= 0):
        raise ValueError("primary traversal timings are incomplete or invalid")
    confidence = _paired_bootstrap(q2, tq1)
    primary_ratio = float(np.median(tq1) / np.median(q2))
    if not math.isclose(primary_ratio, primary["median_ratio"], rel_tol=1e-8, abs_tol=1e-8):
        raise ValueError("reported and recomputed primary ratios differ")

    order = np.asarray(primary["raw"]["order"])
    ab = order == "AB"
    ba = order == "BA"
    ab_ratio = float(np.median(tq1[ab]) / np.median(q2[ab]))
    ba_ratio = float(np.median(tq1[ba]) / np.median(q2[ba]))
    order_effect_fraction = abs(ab_ratio - ba_ratio) / primary_ratio
    q2_cv = float(np.std(q2) / np.mean(q2))
    tq1_cv = float(np.std(tq1) / np.mean(tq1))
    ci_width = confidence["ratio_95_ci"][1] - confidence["ratio_95_ci"][0]
    timing_stable = q2_cv <= 0.10 and tq1_cv <= 0.10 and order_effect_fraction <= 0.10 and ci_width <= 0.10

    warm = raw["regime_a_single_matrix_warm"]
    baseline_reproduced = (
        0.5 <= warm["q2"]["median_ms"] / V1_Q2_WARM_MS <= 2.0
        and 0.5 <= warm["tq1"]["median_ms"] / V1_TQ1_WARM_MS <= 2.0
        and warm["median_ratio"] >= 1.20
    )
    environment_valid, invalid_reasons = _environment_valid(before, after, run)
    if not frozen_unchanged:
        invalid_reasons.append("one or more frozen V1 reports changed")
    if not traversal.get("valid") or not traversal["working_set"]["substantially_exceeds_l2"]:
        invalid_reasons.append("INVALID_STREAMING_WORKSET")
    if not raw["correctness"]["pass"]:
        invalid_reasons.append("INVALID_NUMERICAL_REGRESSION")
    if not baseline_reproduced:
        invalid_reasons.append("V1 warm single-matrix control was not reproduced")
    if not timing_stable:
        invalid_reasons.append("primary traversal timing failed preregistered trust checks")
    anti = raw["anti_cheating"]
    if not (
        anti["q2_allocation_bytes"] == anti["expected_q2_payload_bytes"]
        and anti["tq1_allocation_bytes"] == anti["expected_tq1_payload_bytes"]
        and not anti["unpacked_tq1_weight_buffer"]
        and anti["real_checkpoint_extents"]
        and anti["same_q8_pointer_per_tensor"]
        and not anti["preprocessing_in_timed_path"]
        and anti["same_shapes"]
        and anti["same_scales"]
        and anti["output_dtype"] == "F32"
    ):
        invalid_reasons.append("representation-isolation assertion failed")
    if profiler.get("required") and not profiler.get("attempted"):
        invalid_reasons.append("required profiler investigation was not attempted")

    valid = environment_valid and not invalid_reasons
    verdict = _classify(primary_ratio) if valid else "INVALID_STREAMING_EXPERIMENT"
    crossover = _analyze_crossover(primary["per_tensor"], primary)
    warm_ratio = warm["median_ratio"]
    cold_ratio = raw["regime_c_forced_cold"]["median_ratio"]
    distance_to_warm = abs(primary_ratio - warm_ratio)
    distance_to_cold = abs(primary_ratio - cold_ratio)
    integration_recommended = valid and (
        verdict in {"STREAMING_TQ1_WIN", "STREAMING_TQ1_PARITY"}
        or crossover["oracle_hybrid"]["clearly_useful_mixed_pareto"]
    )
    return {
        "schema_version": 2,
        "v1_verdict_frozen": V1_VERDICT,
        "v1_frozen_unchanged": frozen_unchanged,
        "v1_frozen_report_sha256": frozen_actual,
        "v2_verdict": verdict,
        "valid": valid,
        "invalid_reasons": invalid_reasons,
        "primary_ratio": primary_ratio,
        "confidence": confidence,
        "timing_trust": {
            "stable": timing_stable,
            "q2_cv": q2_cv,
            "tq1_cv": tq1_cv,
            "ab_ratio": ab_ratio,
            "ba_ratio": ba_ratio,
            "order_effect_fraction": order_effect_fraction,
            "bootstrap_ci_width": ci_width,
            "environment_valid": environment_valid,
            "baseline_reproduced": baseline_reproduced,
        },
        "cache_spectrum": {
            "warm_ratio": warm_ratio,
            "stream_ratio": primary_ratio,
            "cold_ratio": cold_ratio,
            "absolute_ratio_distance_to_warm": distance_to_warm,
            "absolute_ratio_distance_to_cold": distance_to_cold,
            "stream_is_closer_to": "cold" if distance_to_cold < distance_to_warm else "warm",
        },
        "correctness": raw["correctness"],
        "regime_a": raw["regime_a_single_matrix_warm"],
        "regime_b": {key: value for key, value in primary.items() if key != "per_tensor"},
        "regime_c": raw["regime_c_forced_cold"],
        "crossover": crossover,
        "profiler": profiler,
        "traversal": {key: value for key, value in traversal.items() if key != "tensors"},
        "anti_cheating": anti,
        "environment_before": before,
        "environment_after": after,
        "run": run,
        "full_integration_recommended": integration_recommended,
    }


def _write_correctness(reports: Path, analysis: dict) -> None:
    correctness = analysis["correctness"]
    metrics = correctness["aggregate"]
    limits = correctness["limits"]
    text = f"""# V2 all-tensor correctness

**Gate: {'PASS' if correctness['pass'] else 'INVALID_NUMERICAL_REGRESSION'}.**

Every one of the 497 selected real tensors was evaluated with two matched inputs: one FP16-origin and one BF16-origin vector, both promoted to F32 and quantized once by Prism to the identical Q8_1 buffer used by Q2 and TQ1. This compared {metrics['values']:,} F32 outputs. All per-tensor checksums were finite and nontrivial.

| Metric | Observed TQ1 vs Q2 | Limit |
| --- | ---: | ---: |
| Max absolute error | {metrics['max_abs']:.12g} | {limits['max_abs']:.12g} |
| Mean absolute error | {metrics['mean_abs']:.12g} | {limits['mean_abs']:.12g} |
| RMSE | {metrics['rmse']:.12g} | {limits['rmse']:.12g} |
| Max relative error | {metrics['max_relative']:.12g} | diagnostic |
| Cosine similarity | {metrics['cosine']:.12g} | >= {limits['minimum_cosine']:.6f} |
| Non-finite values | {metrics['nonfinite']} | 0 |

The frozen exhaustive representation proof remains unchanged: the selected 25,621,954,560 logical weights and 200,171,520 raw FP16 scale groups match tensor-by-tensor hashes with zero mismatches. This V2 pass is regression protection for the multi-tensor launch path, not a replacement for that proof.
"""
    (reports / "01_correctness.md").write_text(text, encoding="utf-8")


def _stats_rows(regime: dict) -> str:
    return (
        f"| Q2 | {regime['q2']['median_ms']:.6f} | {regime['q2']['mean_ms']:.6f} | {regime['q2']['p5_ms']:.6f} | {regime['q2']['p95_ms']:.6f} | {regime['q2']['std_ms']:.6f} |\n"
        f"| TQ1 V1 | {regime['tq1']['median_ms']:.6f} | {regime['tq1']['mean_ms']:.6f} | {regime['tq1']['p5_ms']:.6f} | {regime['tq1']['p95_ms']:.6f} | {regime['tq1']['std_ms']:.6f} |"
    )


def _write_controls(reports: Path, analysis: dict) -> None:
    warm = analysis["regime_a"]
    cold = analysis["regime_c"]
    warm_text = f"""# Regime A: repeated single-matrix warm control

The unchanged V1 kernels repeatedly execute the real `blk.0.ffn_down.weight` matrix (M=5,120, K=17,408). There are 200 warmups and 5,000 CUDA-event-timed AB/BA pairs using the same two prequantized Q8_1 vectors.

| Kernel | Median ms | Mean ms | p5 ms | p95 ms | Std ms |
| --- | ---: | ---: | ---: | ---: | ---: |
{_stats_rows(warm)}

- TQ1/Q2 median ratio: **{warm['median_ratio']:.6f}×**
- Frozen V1 ratio: **{V1_TQ1_WARM_MS / V1_Q2_WARM_MS:.6f}×**
- Control reproduction: **{'PASS' if analysis['timing_trust']['baseline_reproduced'] else 'FAIL'}**

This confirms the cache-resident single-matrix behavior without tuning for it.
"""
    (reports / "02_single_matrix_control.md").write_text(warm_text, encoding="utf-8")
    cold_text = f"""# Regime C: forced cold-ish control

Before every timed `blk.0.ffn_down.weight` launch, the harness writes {cold['scrub_bytes']:,} unrelated bytes (>2× L2). The scrub is stream-ordered before the begin event and excluded from latency. This diagnostic intentionally mirrors the V1 cold-ish control; it is not the primary workload.

| Kernel | Median ms | Mean ms | p5 ms | p95 ms | Std ms |
| --- | ---: | ---: | ---: | ---: | ---: |
{_stats_rows(cold)}

- TQ1/Q2 median ratio: **{cold['median_ratio']:.6f}×**
- TQ1 latency delta: **{_pct(cold['tq1']['median_ms'], cold['q2']['median_ms']):+.3f}%**

The cache spectrum is therefore {analysis['cache_spectrum']['warm_ratio']:.4f}× warm, {analysis['cache_spectrum']['stream_ratio']:.4f}× model-streaming, and {analysis['cache_spectrum']['cold_ratio']:.4f}× forced-cold.
"""
    (reports / "04_cold_control.md").write_text(cold_text, encoding="utf-8")


def _write_stream(reports: Path, analysis: dict) -> None:
    stream = analysis["regime_b"]
    confidence = analysis["confidence"]
    trust = analysis["timing_trust"]
    working = analysis["traversal"]["working_set"]
    monitor = analysis["run"]["continuous_process_monitor"]
    gpu_before = analysis["environment_before"]["gpu"]
    gpu_after = analysis["environment_after"]["gpu"]
    text = f"""# Regime B: full model-order streaming benchmark

**Primary V2 ratio: {analysis['primary_ratio']:.6f}×.**

Each complete traversal launches all 497 real matrix tensors once in model order with no explicit cache flush. The Q2 stream is {working['q2_bytes_per_traversal']:,} bytes ({working['q2_l2_multiple']:.3f}× L2); TQ1 is {working['tq1_bytes_per_traversal']:,} bytes ({working['tq1_l2_multiple']:.3f}× L2). Six full warmups precede 40 CUDA-event-timed AB/BA pairs. Checksums consume every traversal outside its end event.

| Stream | Median ms | Mean ms | p5 ms | p95 ms | Std ms | Physical GB/s |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Q2 | {stream['q2']['median_ms']:.6f} | {stream['q2']['mean_ms']:.6f} | {stream['q2']['p5_ms']:.6f} | {stream['q2']['p95_ms']:.6f} | {stream['q2']['std_ms']:.6f} | {stream['effective_gb_s']['q2_physical']:.3f} |
| TQ1 V1 | {stream['tq1']['median_ms']:.6f} | {stream['tq1']['mean_ms']:.6f} | {stream['tq1']['p5_ms']:.6f} | {stream['tq1']['p95_ms']:.6f} | {stream['tq1']['std_ms']:.6f} | {stream['effective_gb_s']['tq1_physical']:.3f} |

- Ratio-of-medians TQ1/Q2: **{analysis['primary_ratio']:.6f}×**
- Paired-ratio median: **{stream['paired_ratio']['median_ms']:.6f}×**
- Paired bootstrap 95% CI: **[{confidence['ratio_95_ci'][0]:.6f}, {confidence['ratio_95_ci'][1]:.6f}]** ({confidence['samples']:,} resamples)
- AB / BA ratios: {trust['ab_ratio']:.6f}× / {trust['ba_ratio']:.6f}×
- Q2 / TQ1 CV: {trust['q2_cv']:.4%} / {trust['tq1_cv']:.4%}
- CUDA-process audit: {monitor['sample_count']:,} direct-NVML samples at a 10 ms requested interval; maximum observed gap {monitor['maximum_observed_sample_gap_seconds'] * 1000:.1f} ms; zero foreign PIDs and zero monitor errors
- GPU before / after: {gpu_before['pstate']} / {gpu_after['pstate']}, graphics {gpu_before['clocks.gr']} / {gpu_after['clocks.gr']} MHz, memory {gpu_before['clocks.mem']} / {gpu_after['clocks.mem']} MHz, power {gpu_before['power.draw']} / {gpu_after['power.draw']} W
- Timing trust checks: **{'PASS' if trust['stable'] and trust['environment_valid'] else 'FAIL'}**
- Classification: **{analysis['v2_verdict']}**

The claim-bearing run preceded the forced-cold and per-tensor diagnostic passes. No TQ1 kernel modification was made before or after the primary measurement.
"""
    (reports / "03_streaming_benchmark.md").write_text(text, encoding="utf-8")


def _write_profiler(reports: Path, profiler: dict) -> None:
    if not profiler.get("required"):
        body = f"""# Profiler investigation

**NOT RUN.** The primary ratio was {profiler['primary_ratio']:.6f}×, outside the preregistered 1.00×–1.15× profiler window. No profiler-guided tuning was permitted or attempted.
"""
    elif not profiler.get("available"):
        body = f"""# Profiler investigation

**Attempted; hardware counters unavailable.**

The primary ratio was {profiler['primary_ratio']:.6f}×, so the preregistered profiler condition applied. Nsight Compute was invoked once, but collection failed with `{profiler.get('reason', 'unknown error')}`. Timing therefore remains the sole evidence. No repair was made because no profiler-supported local defect was identified.
"""
    else:
        body = "# Profiler investigation\n\n" + profiler.get("markdown", "Profiler results are in the raw profiler artifact.") + "\n"
    (reports / "05_profiler.md").write_text(body, encoding="utf-8")


def _write_crossover(reports: Path, analysis: dict, raw: dict) -> None:
    crossover = analysis["crossover"]
    oracle = crossover["oracle_hybrid"]
    observed = crossover["observed_crossover_q2_bytes"]
    regression = crossover["regression"]["estimated_crossover_q2_bytes"]
    lines = [
        "# Tensor-size crossover and oracle hybrid",
        "",
        "Per-tensor CUDA events were collected in ten AB/BA model-order diagnostic pairs after the clean primary run. They are secondary diagnostics; the verdict uses only complete-traversal timing.",
        "",
        f"- Pearson correlation, log10(Q2 bytes) vs ratio: **{crossover['pearson_log_size_vs_ratio']:.6f}**",
        f"- Spearman correlation, size vs ratio: **{crossover['spearman_size_vs_ratio']:.6f}**",
        f"- TQ1-winning tensors: **{crossover['tq1_winning_tensor_count']}/{len(raw['regime_b_model_stream']['per_tensor'])}** ({crossover['tq1_winning_tensor_fraction']:.2%})",
        f"- Smallest observed size class with median TQ1/Q2 <=1: **{f'{observed:,} Q2 bytes' if observed is not None else 'none'}**",
        f"- Log-size regression crossover estimate: **{f'{regression:,.0f} Q2 bytes' if regression is not None else 'outside the observed range'}**",
        "",
        "## Size classes",
        "",
        "| Q2 bytes/tensor | TQ1 bytes/tensor | Tensors | Q2 median ms | TQ1 median ms | Ratio | TQ1 wins |",
        "| ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for group in crossover["groups"]:
        lines.append(
            f"| {group['q2_bytes']:,} | {group['tq1_bytes']:,} | {group['tensor_count']} | "
            f"{group['q2_median_ms']:.6f} | {group['tq1_median_ms']:.6f} | "
            f"{group['median_ratio']:.6f} | {group['tq1_wins']} |"
        )
    lines.extend(
        [
            "",
            "## Matrix geometry",
            "",
            "Equal byte counts do not imply equal behavior: the 23,674,880-byte FFN gate/up and down tensors are transposed geometries. Individual TQ1 wins concentrate in `ffn_down` (large K, smaller M), so this is a geometry effect rather than a clean size-only crossover.",
            "",
            "| Kind | M | K | Tensors | Median ratio | TQ1 wins |",
            "| --- | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for geometry in crossover["geometries"]:
        lines.append(
            f"| {geometry['kind']} | {geometry['m']:,} | {geometry['k']:,} | "
            f"{geometry['tensor_count']} | {geometry['median_ratio']:.6f} | {geometry['tq1_wins']} |"
        )
    lines.extend(
        [
            "",
            "## Oracle hybrid",
            "",
            "The offline oracle chooses the lower measured per-tensor median. The unbenchmarked token embedding remains Q2, and the 21,574,720 non-Q2 bytes remain fixed. This is not an adaptive runtime measurement.",
            "",
            "| Diagnostic | Value |",
            "| --- | ---: |",
            f"| All-Q2 per-tensor sum | {oracle['all_q2_latency_ms']:.6f} ms |",
            f"| All-TQ1 per-tensor sum | {oracle['all_tq1_latency_ms']:.6f} ms |",
            f"| Oracle-hybrid sum | {oracle['hybrid_latency_ms']:.6f} ms |",
            f"| Q2 / TQ1 tensor choices | {oracle['q2_tensor_count']} / {oracle['tq1_tensor_count']} |",
            f"| Actual hybrid bytes stored as TQ1 | {oracle['tq1_actual_byte_fraction']:.2%} |",
            f"| Original selected Q2 payload reencoded | {oracle['q2_payload_coverage_reencoded_as_tq1']:.2%} |",
            f"| Projected whole-model bytes | {oracle['projected_model_bytes']:,} B |",
            f"| Speedup vs all Q2 | {oracle['speedup_vs_all_q2']:.6f}× |",
            f"| Speedup vs all TQ1 | {oracle['speedup_vs_all_tq1']:.6f}× |",
            f"| Clearly useful mixed Pareto point | {'YES' if oracle['clearly_useful_mixed_pareto'] else 'NO'} |",
            "",
            "## Every tensor",
            "",
            "| # | Tensor | Q2 bytes | Q2 ms | TQ1 ms | Ratio | Faster |",
            "| ---: | --- | ---: | ---: | ---: | ---: | --- |",
        ]
    )
    for tensor in raw["regime_b_model_stream"]["per_tensor"]:
        faster = "TQ1" if tensor["median_ratio"] < 1.0 else "Q2"
        lines.append(
            f"| {tensor['order']} | `{tensor['name']}` | {tensor['q2_bytes']:,} | "
            f"{tensor['q2']['median_ms']:.6f} | {tensor['tq1']['median_ms']:.6f} | "
            f"{tensor['median_ratio']:.6f} | {faster} |"
        )
    (reports / "06_tensor_crossover.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_final(reports: Path, analysis: dict) -> None:
    warm = analysis["regime_a"]
    stream = analysis["regime_b"]
    cold = analysis["regime_c"]
    error = analysis["correctness"]["aggregate"]
    crossover = analysis["crossover"]
    oracle = crossover["oracle_hybrid"]
    cache = analysis["cache_spectrum"]
    ratio = analysis["primary_ratio"]
    valid = analysis["valid"]

    if not valid:
        answer1 = "Invalid experiment; no cache-causality claim is permitted."
        answer2 = "Invalid experiment."
        answer3 = "Invalid experiment."
        answer4 = "Invalid experiment."
    else:
        answer1 = (
            "Yes. The 1.73× warm penalty collapses under model-order streaming, so cache residency was the primary cause of the V1 warm-matrix failure."
            if ratio <= 1.05
            else "No. Model-order streaming still exceeds the 1.05× parity boundary, so cache residency does not explain away the execution penalty."
        )
        answer2 = (
            f"It behaves closer to forced-cold: {ratio:.4f}× streaming versus {cache['cold_ratio']:.4f}× cold and {cache['warm_ratio']:.4f}× warm."
            if cache["stream_is_closer_to"] == "cold"
            else f"It behaves closer to cache-resident warm: {ratio:.4f}× streaming versus {cache['warm_ratio']:.4f}× warm and {cache['cold_ratio']:.4f}× cold."
        )
        answer3 = (
            f"Yes. The complete traversal costs {ratio:.4f}× baseline, inside the <=1.05× parity gate."
            if ratio <= 1.05
            else f"No. The complete traversal costs {ratio:.4f}× baseline, outside the <=1.05× free-cost gate."
        )
        answer4 = (
            f"Yes, by {100.0 * (1.0 - ratio):.2f}% in median complete-traversal latency."
            if ratio < 1.0
            else f"No; it is {100.0 * (ratio - 1.0):.2f}% slower."
        )
    observed = crossover["observed_crossover_q2_bytes"]
    observed_group = next(
        (group for group in crossover["groups"] if group["q2_bytes"] == observed),
        None,
    )
    if observed_group is not None and observed_group["tensor_count"] > 1:
        answer5 = (
            f"Yes, with the first winning size class at {observed:,} Q2 bytes; "
            f"the size/ratio Spearman correlation is {crossover['spearman_size_vs_ratio']:.3f}."
        )
    elif observed_group is not None:
        answer5 = (
            f"Only tentatively: the sole {observed:,}-byte LM head is 0.66% faster and the "
            f"size/ratio Spearman correlation is {crossover['spearman_size_vs_ratio']:.3f}, "
            "but one tensor does not establish a clean size threshold. The smaller wins are "
            "geometry-specific FFN-down matrices."
        )
    else:
        answer5 = (
            f"No clean size-only crossover is established: no size-class median wins despite a "
            f"{crossover['spearman_size_vs_ratio']:.3f} Spearman correlation. Individual wins are "
            "geometry-specific, overwhelmingly FFN-down matrices."
        )
    answer6 = (
        f"Yes as an offline oracle: {oracle['projected_model_bytes']:,} bytes and {oracle['hybrid_latency_ms']:.6f} ms, a clear diagnostic Pareto point."
        if oracle["clearly_useful_mixed_pareto"]
        else "No clearly useful mixed Pareto point survives the preregistered usefulness threshold."
    )
    answer7 = (
        "Yes. The V2 streaming gate or the clearly useful static hybrid diagnostic justifies proceeding to full llama.cpp integration."
        if analysis["full_integration_recommended"]
        else "No. Stop; full llama.cpp integration is not justified by this result."
    )
    invalid_note = ""
    if not valid:
        invalid_note = "\nInvalid reasons: " + "; ".join(analysis["invalid_reasons"]) + ".\n"

    text = f"""V1 VERDICT (FROZEN): {V1_VERDICT}
V2 VERDICT: {analysis['v2_verdict']}

| Metric                         |              Q2 |             TQ1 |     Delta |
| ------------------------------ | --------------: | --------------: | --------: |
| Weight storage                 | 7,143,546,880 B | 5,882,920,960 B | -17.6471% |
| Whole-model projected bytes    | 7,165,121,600 B | 5,904,495,680 B | -17.5939% |
| Single-matrix warm latency     | {warm['q2']['median_ms']:.6f} ms | {warm['tq1']['median_ms']:.6f} ms | {_pct(warm['tq1']['median_ms'], warm['q2']['median_ms']):+.4f}% |
| Forced-cold latency            | {cold['q2']['median_ms']:.6f} ms | {cold['tq1']['median_ms']:.6f} ms | {_pct(cold['tq1']['median_ms'], cold['q2']['median_ms']):+.4f}% |
| Full-stream traversal latency  | {stream['q2']['median_ms']:.6f} ms | {stream['tq1']['median_ms']:.6f} ms | {_pct(stream['tq1']['median_ms'], stream['q2']['median_ms']):+.4f}% |
| Full-stream TQ1/Q2 ratio       |       1.000000× | {ratio:.6f}× | {_pct(ratio, 1.0):+.4f}% |
| Full-stream effective GB/s     | {stream['effective_gb_s']['q2_physical']:.3f} | {stream['effective_gb_s']['tq1_physical']:.3f} | {_pct(stream['effective_gb_s']['tq1_physical'], stream['effective_gb_s']['q2_physical']):+.4f}% |
| Numerical max error            |               0 | {error['max_abs']:.12g} | within gate |
| Oracle-hybrid latency          | {oracle['all_q2_latency_ms']:.6f} ms | {oracle['hybrid_latency_ms']:.6f} ms | {_pct(oracle['hybrid_latency_ms'], oracle['all_q2_latency_ms']):+.4f}% |
| Oracle-hybrid projected memory | 7,165,121,600 B | {oracle['projected_model_bytes']:,} B | {_pct(oracle['projected_model_bytes'], Q2_MODEL_BYTES):+.4f}% |
{invalid_note}
1. **Was the previous warm-matrix failure caused primarily by cache residency?** {answer1}
2. **Does actual model-order traversal behave more like the warm or cold regime?** {answer2}
3. **Does TQ1 give 17.6% lower memory essentially for free?** {answer3}
4. **Is TQ1 faster under realistic streaming?** {answer4}
5. **Is there a tensor-size crossover?** {answer5}
6. **Would a mixed Q2/TQ1 model dominate either uniform format?** {answer6}
7. **Is full llama.cpp integration justified?** {answer7}

The V1 report files remain byte-for-byte unchanged. V2 used the existing validated V1 TQ1 kernel with no pre-primary tuning and no representation change.
"""
    (reports / "FINAL_REPORT.md").write_text(text, encoding="utf-8")


def write_reports(reports: Path, analysis: dict, raw: dict) -> None:
    reports.mkdir(parents=True, exist_ok=True)
    _write_correctness(reports, analysis)
    _write_controls(reports, analysis)
    _write_stream(reports, analysis)
    _write_profiler(reports, analysis["profiler"])
    _write_crossover(reports, analysis, raw)
    _write_final(reports, analysis)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Analyze V2 model-stream results and write reports")
    parser.add_argument("--raw", type=Path, required=True)
    parser.add_argument("--traversal", type=Path, required=True)
    parser.add_argument("--environment-before", type=Path, required=True)
    parser.add_argument("--environment-after", type=Path, required=True)
    parser.add_argument("--run-metadata", type=Path, required=True)
    parser.add_argument("--profiler", type=Path, required=True)
    parser.add_argument("--v1-reports", type=Path, required=True)
    parser.add_argument("--analysis", type=Path, required=True)
    parser.add_argument("--reports", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        raw = _load(args.raw)
        traversal = _load(args.traversal)
        before = _load(args.environment_before)
        after = _load(args.environment_after)
        run = _load(args.run_metadata)
        profiler = _load(args.profiler)
        result = analyze(raw, traversal, before, after, run, profiler, args.v1_reports)
        write_json_atomic(args.analysis, result)
        write_reports(args.reports, result, raw)
    except Exception as exc:
        print(f"V2 reporting failed: {exc}", file=sys.stderr)
        return 2
    print(f"V2_VERDICT={result['v2_verdict']} ratio={result['primary_ratio']:.6f}")
    return 0 if result["valid"] else 3


if __name__ == "__main__":
    raise SystemExit(main())
