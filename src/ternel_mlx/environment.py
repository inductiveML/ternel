"""Capture the Apple Silicon machine and toolchain a benchmark ran on.

``bonsai_tq1.environment`` asks ``nvidia-smi`` questions that have no answer on
a Mac, so this is the macOS counterpart rather than a rewrite: the probe runner
is imported from there, and the emitted document keeps the same
``schema_version`` / ``captured_at_utc`` / raw-command shape so both sets of
results read the same way.

Two things it deliberately records beyond the hardware. ``pmset`` power mode and
thermal state, because a low-power or thermally-limited run is not comparable to
a nominal one; and the machine's load average and heaviest processes, which is
the closest honest analogue of the CUDA capture's ``compute_processes`` gate --
macOS exposes no per-process GPU accounting, so contention is reported as CPU
pressure and the reader is told that is what it is.
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
from datetime import UTC, datetime
from pathlib import Path

import mlx.core as mx

from bonsai_tq1.environment import run_command
from bonsai_tq1.format import FormatError, write_json_atomic

SCHEMA_VERSION = 1

# Queried one per invocation so a single unsupported key cannot blank the rest.
SYSCTL_KEYS = (
    "machdep.cpu.brand_string",
    "hw.model",
    "hw.memsize",
    "hw.ncpu",
    "hw.perflevel0.logicalcpu",
    "hw.perflevel1.logicalcpu",
    "hw.pagesize",
    "vm.loadavg",
)

# How many of the busiest processes to record. Enough to spot a competing build
# or another model server, short enough to stay readable in the report.
BUSIEST_PROCESS_COUNT = 8


def _sysctl() -> dict[str, object]:
    values: dict[str, object] = {}
    for key in SYSCTL_KEYS:
        probe = run_command(["sysctl", "-n", key])
        values[key] = probe["stdout"] if probe["returncode"] == 0 else None
    return values


def _gpu() -> dict[str, object]:
    """Core count and Metal family, from ``system_profiler``.

    Only the adapter's own fields are kept. The ``spdisplays_ndrvs`` list under
    it describes attached monitors and carries display serial numbers, which are
    both irrelevant to a compute benchmark and not something to write into a
    published result file.
    """
    probe = run_command(["system_profiler", "SPDisplaysDataType", "-json"])
    adapters: list[dict[str, object]] = []
    if probe["returncode"] == 0:
        try:
            document = json.loads(str(probe["stdout"]))
        except json.JSONDecodeError:
            document = {}
        entries = document.get("SPDisplaysDataType", []) if isinstance(document, dict) else []
        for entry in entries:
            if isinstance(entry, dict):
                adapters.append({
                    key: value for key, value in entry.items() if key != "spdisplays_ndrvs"
                })
    return {"returncode": probe["returncode"], "adapters": adapters}


def _busiest_processes() -> dict[str, object]:
    probe = run_command(["ps", "-A", "-r", "-o", "pid=,pcpu=,rss=,comm="])
    if probe["returncode"] != 0:
        return {"returncode": probe["returncode"], "processes": []}
    processes: list[dict[str, object]] = []
    for line in str(probe["stdout"]).splitlines()[:BUSIEST_PROCESS_COUNT]:
        fields = line.split(None, 3)
        if len(fields) != 4:
            continue
        processes.append({
            "pid": int(fields[0]),
            "cpu_percent": float(fields[1]),
            "rss_kib": int(fields[2]),
            "command": fields[3],
        })
    return {"returncode": probe["returncode"], "processes": processes}


def _mlx() -> dict[str, object]:
    # mx.metal.device_info is deprecated in 0.32.0 in favour of mx.device_info.
    info = mx.device_info()
    return {
        "mlx_version": mx.__version__,
        "default_device": str(mx.default_device()),
        "metal_available": mx.metal.is_available(),
        "device_info": {key: value for key, value in info.items()},
        "active_memory_bytes": mx.get_active_memory(),
        "peak_memory_bytes": mx.get_peak_memory(),
        "cache_memory_bytes": mx.get_cache_memory(),
    }


def capture_environment() -> dict[str, object]:
    return {
        "schema_version": SCHEMA_VERSION,
        "captured_at_utc": datetime.now(UTC).isoformat(),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "sw_vers": run_command(["sw_vers"]),
        "sysctl": _sysctl(),
        "gpu": _gpu(),
        "power": run_command(["pmset", "-g"]),
        "thermal": run_command(["pmset", "-g", "therm"]),
        "metal_compiler": run_command(["xcrun", "-sdk", "macosx", "metal", "--version"]),
        "xcode": run_command(["xcodebuild", "-version"]),
        "uv": run_command(["uv", "--version"]),
        "mlx": _mlx(),
        "busiest_processes": _busiest_processes(),
    }


def gpu_core_count(environment: dict[str, object]) -> int:
    """Read the GPU core count out of a captured document.

    Raises rather than guessing: a benchmark report that silently prints a
    made-up core count is worse than one that fails to render.
    """
    gpu = environment.get("gpu")
    if not isinstance(gpu, dict):
        raise FormatError("captured environment has no gpu section")
    for adapter in gpu.get("adapters", []):
        cores = adapter.get("sppci_cores") if isinstance(adapter, dict) else None
        if cores is not None:
            return int(cores)
    raise FormatError("captured environment reports no GPU core count")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Capture the Apple Silicon benchmark environment")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    result = capture_environment()
    write_json_atomic(args.output, result)
    json.dump(result, sys.stdout, indent=2, sort_keys=True)
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
