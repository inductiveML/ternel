from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def _processes() -> list[str]:
    completed = subprocess.run(
        [
            "nvidia-smi",
            "--query-compute-apps=pid,process_name,used_memory",
            "--format=csv,noheader",
        ],
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    return [line.strip() for line in completed.stdout.splitlines() if line.strip()]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Apply the preregistered V2 profiler condition")
    parser.add_argument("--raw", type=Path, required=True)
    parser.add_argument("--traversal", type=Path, required=True)
    parser.add_argument("--benchmark-gemv", type=Path, required=True)
    parser.add_argument("--q2", type=Path, required=True)
    parser.add_argument("--tq1", type=Path, required=True)
    parser.add_argument("--probe-output", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)

    raw = json.loads(args.raw.read_text(encoding="utf-8"))
    traversal = json.loads(args.traversal.read_text(encoding="utf-8"))
    ratio = float(raw["regime_b_model_stream"]["median_ratio"])
    required = 1.00 <= ratio <= 1.15
    if not required:
        result = {
            "schema_version": 2,
            "primary_ratio": ratio,
            "required": False,
            "attempted": False,
            "available": None,
            "reason": "primary ratio outside preregistered 1.00x-1.15x profiler window",
            "repair_attempted": False,
        }
        _write_json(args.output, result)
        print(result["reason"])
        return 0

    if _processes():
        print("GPU_BUSY: profiler attempt deferred", file=sys.stderr)
        return 75
    primary = next(tensor for tensor in traversal["tensors"] if tensor["name"] == "blk.0.ffn_down.weight")
    command = [
        "ncu",
        "--target-processes", "all",
        "--set", "basic",
        "--kernel-name", "regex:.*tq1_g128_gemv_v1_kernel.*",
        "--launch-count", "1",
        str(args.benchmark_gemv),
        "--q2", str(args.q2),
        "--tq1", str(args.tq1),
        "--q2-offset", str(primary["q2_file_offset"]),
        "--tq1-offset", str(primary["tq1_file_offset"]),
        "--rows", str(primary["m"]),
        "--columns", str(primary["k"]),
        "--kernel", "v1",
        "--output", str(args.probe_output),
    ]
    completed = subprocess.run(
        command,
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    combined = completed.stdout + "\n" + completed.stderr
    permission_denied = "ERR_NVGPUCTRPERM" in combined or "permission" in combined.lower()
    available = completed.returncode == 0 and not permission_denied
    result = {
        "schema_version": 2,
        "primary_ratio": ratio,
        "required": True,
        "attempted": True,
        "available": available,
        "reason": (
            "ERR_NVGPUCTRPERM: performance-counter access denied by host"
            if permission_denied
            else ("Nsight Compute collection succeeded" if available else f"ncu exited {completed.returncode}")
        ),
        "command": command,
        "returncode": completed.returncode,
        "stdout": completed.stdout,
        "stderr": completed.stderr,
        "repair_attempted": False,
        "repair_reason": "no profiler-supported trivial defect identified",
    }
    _write_json(args.output, result)
    print(result["reason"])
    if available:
        print(
            "Profiler counters are unexpectedly available; complete the three-way profile comparison before reporting.",
            file=sys.stderr,
        )
        return 4
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
