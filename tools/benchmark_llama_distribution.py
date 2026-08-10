#!/usr/bin/env python3
"""Run paired, alternating llama-bench trials for Q2 and distributed TQ1."""

from __future__ import annotations

import argparse
import json
import re
import statistics
import subprocess
import threading
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np


MEMORY_RE = re.compile(r"^(?P<pid>\d+),\s*(?P<memory>\d+)\s+MiB$")


@dataclass
class MonitoredRun:
    returncode: int
    stdout: str
    stderr: str
    elapsed_seconds: float
    peak_vram_mib: int
    gpu_samples: list[dict[str, float]]


def command_output(command: list[str]) -> str:
    return subprocess.run(command, check=True, text=True, capture_output=True).stdout.strip()


def gpu_snapshot() -> dict[str, Any]:
    fields = [
        "name",
        "driver_version",
        "memory.total",
        "memory.used",
        "utilization.gpu",
        "pstate",
        "clocks.sm",
        "power.draw",
    ]
    raw = command_output(
        ["nvidia-smi", f"--query-gpu={','.join(fields)}", "--format=csv,noheader,nounits"]
    )
    values = [item.strip() for item in raw.split(",")]
    return dict(zip(fields, values, strict=True))


def monitor_process(command: list[str], poll_seconds: float = 1.0) -> MonitoredRun:
    process = subprocess.Popen(command, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    peak_vram_mib = 0
    gpu_samples: list[dict[str, float]] = []
    stop = threading.Event()

    def poll() -> None:
        nonlocal peak_vram_mib
        while not stop.is_set():
            try:
                apps = command_output(
                    [
                        "nvidia-smi",
                        "--query-compute-apps=pid,used_memory",
                        "--format=csv,noheader",
                    ]
                )
                for line in apps.splitlines():
                    match = MEMORY_RE.match(line.strip())
                    if match and int(match.group("pid")) == process.pid:
                        peak_vram_mib = max(peak_vram_mib, int(match.group("memory")))
                gpu = command_output(
                    [
                        "nvidia-smi",
                        "--query-gpu=utilization.gpu,clocks.sm,power.draw",
                        "--format=csv,noheader,nounits",
                    ]
                )
                util, clock, power = (float(item.strip()) for item in gpu.split(","))
                gpu_samples.append({"utilization_pct": util, "clock_mhz": clock, "power_w": power})
            except (OSError, subprocess.SubprocessError, ValueError):
                pass
            stop.wait(poll_seconds)

    monitor = threading.Thread(target=poll, daemon=True)
    started = time.monotonic()
    monitor.start()
    stdout, stderr = process.communicate()
    elapsed = time.monotonic() - started
    stop.set()
    monitor.join(timeout=2)
    return MonitoredRun(process.returncode, stdout, stderr, elapsed, peak_vram_mib, gpu_samples)


def summarize(values: list[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "count": int(array.size),
        "median": float(np.median(array)),
        "p5": float(np.percentile(array, 5)),
        "p95": float(np.percentile(array, 95)),
        "mean": float(np.mean(array)),
    }


def bootstrap_median_ci(values: list[float], seed: int = 128, iterations: int = 20_000) -> list[float]:
    array = np.asarray(values, dtype=np.float64)
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, array.size, size=(iterations, array.size))
    medians = np.median(array[indices], axis=1)
    return [float(np.percentile(medians, 2.5)), float(np.percentile(medians, 97.5))]


def test_name(entry: dict[str, Any]) -> str:
    prompt = int(entry["n_prompt"])
    generation = int(entry["n_gen"])
    if prompt:
        return f"pp{prompt}"
    return f"tg{generation}"


def gpu_sample_summary(samples: list[dict[str, float]]) -> dict[str, dict[str, float]]:
    if not samples:
        return {}
    return {
        key: summarize([sample[key] for sample in samples])
        for key in ("utilization_pct", "clock_mhz", "power_w")
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--binary", type=Path, default=Path("build/bin/llama-bench"))
    parser.add_argument(
        "--baseline", type=Path,
        default=Path("artifacts/models/Ternary-Bonsai-27B-Q2_0.gguf"),
    )
    parser.add_argument(
        "--tq1", type=Path,
        default=Path("artifacts/models/Ternary-Bonsai-27B-TQ1_G128.gguf"),
    )
    parser.add_argument("--output", type=Path, default=Path("artifacts/results_dist/llama_bench.json"))
    parser.add_argument("--repetitions", type=int, default=5)
    parser.add_argument("--rounds", type=int, default=2)
    parser.add_argument("--prompt-tokens", type=int, default=32)
    parser.add_argument("--generation-tokens", type=int, default=64)
    parser.add_argument("--threads", type=int, default=24)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.rounds < 1 or args.repetitions < 1:
        raise SystemExit("rounds and repetitions must be positive")
    for path in (args.binary, args.baseline, args.tq1):
        if not path.is_file():
            raise SystemExit(f"missing required file: {path}")

    models = {"q2": args.baseline, "tq1": args.tq1}
    before = gpu_snapshot()
    records: list[dict[str, Any]] = []
    run_index = 0
    for round_index in range(args.rounds):
        order = ["q2", "tq1"] if round_index % 2 == 0 else ["tq1", "q2"]
        for model_name in order:
            run_index += 1
            command = [
                str(args.binary),
                "-m", str(models[model_name]),
                "-p", "0",
                "-n", "0",
                "-pg", f"{args.prompt_tokens},0",
                "-pg", f"0,{args.generation_tokens}",
                "-ngl", "99",
                "-sm", "none",
                "-mmp", "0",
                "-t", str(args.threads),
                "-r", str(args.repetitions),
                "-o", "json",
            ]
            print(
                f"[{datetime.now(timezone.utc).isoformat()}] "
                f"round={round_index + 1} model={model_name}",
                flush=True,
            )
            monitored = monitor_process(command)
            record: dict[str, Any] = {
                "run_index": run_index,
                "round": round_index + 1,
                "model": model_name,
                "order_in_round": order.index(model_name) + 1,
                "command": command,
                "returncode": monitored.returncode,
                "elapsed_seconds": monitored.elapsed_seconds,
                "peak_process_vram_mib": monitored.peak_vram_mib,
                "gpu": gpu_sample_summary(monitored.gpu_samples),
                "stderr": monitored.stderr,
            }
            try:
                record["benchmarks"] = json.loads(monitored.stdout)
            except json.JSONDecodeError:
                record["stdout"] = monitored.stdout
            records.append(record)
            if monitored.returncode != 0 or "benchmarks" not in record:
                args.output.parent.mkdir(parents=True, exist_ok=True)
                args.output.write_text(json.dumps({"status": "FAIL", "runs": records}, indent=2) + "\n")
                raise SystemExit(f"llama-bench failed for {model_name} (run {run_index})")

    ratios_by_test: dict[str, list[float]] = {}
    latency_by_model_test: dict[str, dict[str, list[float]]] = {"q2": {}, "tq1": {}}
    throughput_by_model_test: dict[str, dict[str, list[float]]] = {"q2": {}, "tq1": {}}
    for record in records:
        for entry in record["benchmarks"]:
            name = test_name(entry)
            latency_by_model_test[record["model"]].setdefault(name, []).extend(
                float(value) / 1e6 for value in entry["samples_ns"]
            )
            throughput_by_model_test[record["model"]].setdefault(name, []).extend(
                float(value) for value in entry["samples_ts"]
            )

    for round_index in range(1, args.rounds + 1):
        pair = {record["model"]: record for record in records if record["round"] == round_index}
        q2_entries = {test_name(item): item for item in pair["q2"]["benchmarks"]}
        tq1_entries = {test_name(item): item for item in pair["tq1"]["benchmarks"]}
        for name in sorted(q2_entries.keys() & tq1_entries.keys()):
            q2_ns = q2_entries[name]["samples_ns"]
            tq1_ns = tq1_entries[name]["samples_ns"]
            if len(q2_ns) != len(tq1_ns):
                raise RuntimeError(f"sample count mismatch for {name}")
            ratios_by_test.setdefault(name, []).extend(
                float(tq1) / float(q2) for q2, tq1 in zip(q2_ns, tq1_ns, strict=True)
            )

    tests: dict[str, Any] = {}
    for name, ratios in ratios_by_test.items():
        tests[name] = {
            "latency_ratio_tq1_over_q2": summarize(ratios),
            "latency_ratio_bootstrap_95_ci": bootstrap_median_ci(ratios),
            "q2_latency_ms": summarize(latency_by_model_test["q2"][name]),
            "tq1_latency_ms": summarize(latency_by_model_test["tq1"][name]),
            "q2_tokens_per_second": summarize(throughput_by_model_test["q2"][name]),
            "tq1_tokens_per_second": summarize(throughput_by_model_test["tq1"][name]),
        }

    peaks = {
        model: [record["peak_process_vram_mib"] for record in records if record["model"] == model]
        for model in models
    }
    peak_summary = {
        model: {
            "runs_mib": values,
            "median_mib": statistics.median(values),
            "min_mib": min(values),
            "max_mib": max(values),
        }
        for model, values in peaks.items()
    }
    q2_peak = float(peak_summary["q2"]["median_mib"])
    tq1_peak = float(peak_summary["tq1"]["median_mib"])
    peak_summary["saved_mib"] = q2_peak - tq1_peak
    peak_summary["saved_percent_of_q2"] = 100.0 * (q2_peak - tq1_peak) / q2_peak

    output = {
        "status": "PASS",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "method": {
            "paired_alternating_rounds": args.rounds,
            "repetitions_per_round": args.repetitions,
            "warmup": "llama-bench default warmup enabled",
            "mmap": False,
            "gpu_layers": 99,
            "split_mode": "none",
            "bootstrap_iterations": 20_000,
        },
        "gpu_before": before,
        "gpu_after": gpu_snapshot(),
        "models": {name: str(path.resolve()) for name, path in models.items()},
        "tests": tests,
        "process_peak_vram": peak_summary,
        "runs": records,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps({"tests": tests, "process_peak_vram": peak_summary}, indent=2))


if __name__ == "__main__":
    main()
