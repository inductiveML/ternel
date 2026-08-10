from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

from run_v2_benchmark import (
    _NvmlProcessMonitor,
    _compute_processes,
    _gpu_snapshot,
    _tool_versions,
    _write_json,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run clean monitored LUT23 primary qualification")
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--q2", type=Path, required=True)
    parser.add_argument("--tq1", type=Path, required=True)
    parser.add_argument("--reordered", type=Path, required=True)
    parser.add_argument("--reorder-result", type=Path, required=True)
    parser.add_argument("--verification", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--environment-before", type=Path, required=True)
    parser.add_argument("--environment-after", type=Path, required=True)
    parser.add_argument("--run-metadata", type=Path, required=True)
    parser.add_argument("--warmups", type=int, default=200)
    parser.add_argument("--pairs", type=int, default=200)
    args = parser.parse_args(argv)

    try:
        verification = json.loads(args.verification.read_text(encoding="utf-8"))
        reorder = json.loads(args.reorder_result.read_text(encoding="utf-8"))
        if not verification.get("pass"):
            raise RuntimeError("exhaustive reordered-sidecar verification did not pass")
        if args.reordered.stat().st_size != int(reorder["destination_file_bytes"]):
            raise RuntimeError("reordered sidecar byte count changed after verification")
        if verification["source_sha256"] != reorder["source_sha256"]:
            raise RuntimeError("reorder and verification source identities differ")

        before_processes = _compute_processes()
        before_gpu = _gpu_snapshot()
        before = {
            "gpu": before_gpu,
            "compute_processes": before_processes,
            "tool_versions": _tool_versions(),
        }
        _write_json(args.environment_before, before)
        if before_processes:
            details = ", ".join(
                f"PID {item['pid']} {item['process_name']} ({item['used_memory_mib']} MiB)"
                for item in before_processes
            )
            print(f"GPU_BUSY: {details}", file=sys.stderr)
            return 75
        if before_gpu["name"] != "NVIDIA RTX 6000 Ada Generation" or before_gpu["compute_cap"] != "8.9":
            raise RuntimeError("target is not the preregistered RTX 6000 Ada sm_89")

        command = [
            str(args.binary),
            "--manifest", str(args.manifest),
            "--q2", str(args.q2),
            "--tq1", str(args.tq1),
            "--reordered", str(args.reordered),
            "--output", str(args.output),
            "--mode", "qualify",
            "--warmups", str(args.warmups),
            "--pairs", str(args.pairs),
        ]
        started = datetime.now(UTC)
        nvml = _NvmlProcessMonitor()
        process = subprocess.Popen(command, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        samples: list[dict] = []
        foreign: dict[str, dict[str, str]] = {}
        errors: list[str] = []
        last_sample: float | None = None
        maximum_gap = 0.0
        while process.poll() is None:
            captured_at = datetime.now(UTC).isoformat()
            monotonic = time.monotonic()
            if last_sample is not None:
                maximum_gap = max(maximum_gap, monotonic - last_sample)
            last_sample = monotonic
            try:
                observed = nvml.processes()
                samples.append({"captured_at_utc": captured_at, "processes": observed})
                for item in observed:
                    if int(item["pid"]) != process.pid:
                        foreign[item["pid"]] = item
            except Exception as exc:
                errors.append(f"{captured_at}: {exc}")
            time.sleep(0.01)
        stdout, stderr = process.communicate()
        nvml.close()
        finished = datetime.now(UTC)
        after_processes = _compute_processes()
        after_gpu = _gpu_snapshot()
        after = {
            "gpu": after_gpu,
            "compute_processes": after_processes,
            "tool_versions": _tool_versions(),
        }
        _write_json(args.environment_after, after)
        metadata = {
            "command": command,
            "started_at_utc": started.isoformat(),
            "finished_at_utc": finished.isoformat(),
            "elapsed_seconds": (finished - started).total_seconds(),
            "returncode": process.returncode,
            "benchmark_pid": process.pid,
            "stdout": stdout,
            "stderr": stderr,
            "before_compute_processes": before_processes,
            "after_compute_processes": after_processes,
            "continuous_process_monitor": {
                "method": "direct NVML nvmlDeviceGetComputeRunningProcesses_v3",
                "requested_poll_interval_seconds": 0.01,
                "maximum_observed_sample_gap_seconds": maximum_gap,
                "sample_count": len(samples),
                "samples": samples,
                "monitor_errors": errors,
                "foreign_processes": list(foreign.values()),
                "unrelated_process_seen": bool(foreign),
            },
        }
        _write_json(args.run_metadata, metadata)
        if stdout:
            print(stdout, end="")
        if stderr:
            print(stderr, end="", file=sys.stderr)
        if process.returncode != 0:
            return process.returncode
        if after_processes or foreign or errors:
            print("INVALID_EXPERIMENT: competing GPU process or monitor failure", file=sys.stderr)
            return 76
        json.loads(args.output.read_text(encoding="utf-8"))
        return 0
    except Exception as exc:
        print(f"LUT23 primary runner failed: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
