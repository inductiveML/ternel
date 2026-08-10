from __future__ import annotations

import argparse
import ctypes
import hashlib
import json
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path


GPU_FIELDS = (
    "name",
    "uuid",
    "driver_version",
    "compute_cap",
    "pstate",
    "clocks.gr",
    "clocks.mem",
    "power.draw",
    "power.limit",
    "utilization.gpu",
    "utilization.memory",
    "memory.used",
    "memory.free",
)


class _NvmlProcessInfo(ctypes.Structure):
    _fields_ = [
        ("pid", ctypes.c_uint),
        ("used_gpu_memory", ctypes.c_ulonglong),
        ("gpu_instance_id", ctypes.c_uint),
        ("compute_instance_id", ctypes.c_uint),
    ]


class _NvmlProcessMonitor:
    SUCCESS = 0
    INSUFFICIENT_SIZE = 7

    def __init__(self) -> None:
        self.library = ctypes.CDLL("libnvidia-ml.so.1")
        self.library.nvmlInit_v2.restype = ctypes.c_int
        self.library.nvmlShutdown.restype = ctypes.c_int
        self.library.nvmlDeviceGetHandleByIndex_v2.argtypes = [
            ctypes.c_uint,
            ctypes.POINTER(ctypes.c_void_p),
        ]
        self.library.nvmlDeviceGetHandleByIndex_v2.restype = ctypes.c_int
        self.library.nvmlDeviceGetComputeRunningProcesses_v3.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_uint),
            ctypes.POINTER(_NvmlProcessInfo),
        ]
        self.library.nvmlDeviceGetComputeRunningProcesses_v3.restype = ctypes.c_int
        if self.library.nvmlInit_v2() != self.SUCCESS:
            raise RuntimeError("NVML initialization failed")
        self.handle = ctypes.c_void_p()
        if self.library.nvmlDeviceGetHandleByIndex_v2(0, ctypes.byref(self.handle)) != self.SUCCESS:
            self.library.nvmlShutdown()
            raise RuntimeError("NVML device lookup failed")

    def close(self) -> None:
        if self.library.nvmlShutdown() != self.SUCCESS:
            raise RuntimeError("NVML shutdown failed")

    def processes(self) -> list[dict[str, str]]:
        count = ctypes.c_uint(0)
        status = self.library.nvmlDeviceGetComputeRunningProcesses_v3(
            self.handle, ctypes.byref(count), None
        )
        if status == self.SUCCESS and count.value == 0:
            return []
        if status != self.INSUFFICIENT_SIZE:
            raise RuntimeError(f"NVML process-count query failed with status {status}")
        capacity = max(count.value + 8, 16)
        while True:
            entries = (_NvmlProcessInfo * capacity)()
            populated = ctypes.c_uint(capacity)
            status = self.library.nvmlDeviceGetComputeRunningProcesses_v3(
                self.handle, ctypes.byref(populated), entries
            )
            if status == self.INSUFFICIENT_SIZE:
                capacity = max(capacity * 2, populated.value + 8)
                continue
            if status != self.SUCCESS:
                raise RuntimeError(f"NVML process query failed with status {status}")
            return [
                {
                    "pid": str(entries[index].pid),
                    "used_memory_mib": str(entries[index].used_gpu_memory // (1024 * 1024)),
                }
                for index in range(populated.value)
            ]


def _run(command: list[str], *, check: bool = True) -> subprocess.CompletedProcess[str]:
    return subprocess.run(command, check=check, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)


def _gpu_snapshot() -> dict:
    query = ",".join(GPU_FIELDS)
    completed = _run(
        ["nvidia-smi", f"--query-gpu={query}", "--format=csv,noheader,nounits"]
    )
    lines = [line for line in completed.stdout.splitlines() if line.strip()]
    if len(lines) != 1:
        raise RuntimeError(f"expected exactly one GPU, found {len(lines)}")
    values = [value.strip() for value in lines[0].split(",")]
    if len(values) != len(GPU_FIELDS):
        raise RuntimeError("unexpected nvidia-smi GPU field count")
    result = dict(zip(GPU_FIELDS, values, strict=True))
    result["captured_at_utc"] = datetime.now(UTC).isoformat()
    return result


def _compute_processes() -> list[dict[str, str]]:
    completed = _run(
        [
            "nvidia-smi",
            "--query-compute-apps=pid,process_name,used_memory",
            "--format=csv,noheader,nounits",
        ]
    )
    result = []
    for line in completed.stdout.splitlines():
        if not line.strip():
            continue
        values = [value.strip() for value in line.split(",")]
        if len(values) != 3:
            raise RuntimeError("unexpected nvidia-smi process field count")
        result.append({"pid": values[0], "process_name": values[1], "used_memory_mib": values[2]})
    return result


def _tool_versions() -> dict[str, str]:
    return {
        "nvcc": _run(["nvcc", "--version"]).stdout.strip(),
        "cxx": _run(["g++", "--version"]).stdout.splitlines()[0],
        "cmake": _run(["cmake", "--version"]).stdout.splitlines()[0],
    }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def _validate_frozen_v1(traversal_path: Path, reports_dir: Path) -> None:
    traversal = json.loads(traversal_path.read_text(encoding="utf-8"))
    expected = traversal["v1_frozen_report_sha256"]
    actual = {path.name: _sha256(path) for path in sorted(reports_dir.glob("*.md"))}
    if actual != expected:
        raise RuntimeError("V1 report hashes changed after the V2 freeze point")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the gated V2 model-stream CUDA benchmark")
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--traversal-json", type=Path, required=True)
    parser.add_argument("--q2", type=Path, required=True)
    parser.add_argument("--tq1", type=Path, required=True)
    parser.add_argument("--v1-reports", type=Path, required=True)
    parser.add_argument("--bridge-result", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--environment-before", type=Path, required=True)
    parser.add_argument("--environment-after", type=Path, required=True)
    parser.add_argument("--run-metadata", type=Path, required=True)
    parser.add_argument("--traversal-warmups", type=int, default=6)
    parser.add_argument("--traversal-pairs", type=int, default=40)
    parser.add_argument("--diagnostic-pairs", type=int, default=10)
    args = parser.parse_args(argv)

    try:
        _validate_frozen_v1(args.traversal_json, args.v1_reports)
        bridge = json.loads(args.bridge_result.read_text(encoding="utf-8"))
        if not bridge.get("pass"):
            raise RuntimeError("frozen Prism public-graph bridge validation did not pass")
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
        if float(before_gpu["memory.used"]) > 2048:
            raise RuntimeError("GPU has more than 2 GiB allocated despite an empty compute-process list")

        command = [
            str(args.binary),
            "--manifest", str(args.manifest),
            "--q2", str(args.q2),
            "--tq1", str(args.tq1),
            "--output", str(args.output),
            "--traversal-warmups", str(args.traversal_warmups),
            "--traversal-pairs", str(args.traversal_pairs),
            "--diagnostic-pairs", str(args.diagnostic_pairs),
        ]
        started = datetime.now(UTC)
        nvml = _NvmlProcessMonitor()
        process = subprocess.Popen(command, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        process_samples: list[dict] = []
        foreign_processes: dict[str, dict[str, str]] = {}
        monitor_errors: list[str] = []
        last_sample_monotonic: float | None = None
        maximum_sample_gap = 0.0
        while process.poll() is None:
            captured_at = datetime.now(UTC).isoformat()
            captured_monotonic = time.monotonic()
            if last_sample_monotonic is not None:
                maximum_sample_gap = max(maximum_sample_gap, captured_monotonic - last_sample_monotonic)
            last_sample_monotonic = captured_monotonic
            try:
                observed = nvml.processes()
                process_samples.append({"captured_at_utc": captured_at, "processes": observed})
                for item in observed:
                    if int(item["pid"]) != process.pid:
                        foreign_processes[item["pid"]] = item
            except Exception as exc:
                monitor_errors.append(f"{captured_at}: {exc}")
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
            "unrelated_processes_seen_at_boundaries": bool(before_processes or after_processes),
            "continuous_process_monitor": {
                "method": "direct NVML nvmlDeviceGetComputeRunningProcesses_v3",
                "poll_interval_seconds": 0.01,
                "maximum_observed_sample_gap_seconds": maximum_sample_gap,
                "samples": process_samples,
                "sample_count": len(process_samples),
                "monitor_errors": monitor_errors,
                "foreign_processes": list(foreign_processes.values()),
                "unrelated_process_seen": bool(foreign_processes),
            },
        }
        _write_json(args.run_metadata, metadata)
        if stdout:
            print(stdout, end="")
        if stderr:
            print(stderr, end="", file=sys.stderr)
        if process.returncode != 0:
            return process.returncode
        if after_processes or foreign_processes or monitor_errors:
            print(
                "INVALID_STREAMING_EXPERIMENT: competing process or process-monitor gap during timing",
                file=sys.stderr,
            )
            return 76
        if not args.output.exists():
            raise RuntimeError("benchmark completed without writing its raw result")
        json.loads(args.output.read_text(encoding="utf-8"))
        return 0
    except Exception as exc:
        print(f"V2 runner failed: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
