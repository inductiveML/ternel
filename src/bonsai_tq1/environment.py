from __future__ import annotations

import argparse
import json
import platform
import subprocess
from datetime import UTC, datetime
from pathlib import Path

from .inspect_model import write_json_atomic


def _run(command: list[str]) -> dict[str, object]:
    completed = subprocess.run(command, text=True, capture_output=True, check=False)
    return {
        "command": command,
        "returncode": completed.returncode,
        "stdout": completed.stdout.strip(),
        "stderr": completed.stderr.strip(),
    }


def capture_environment() -> dict:
    gpu_fields = (
        "name,uuid,driver_version,compute_cap,pstate,clocks.current.graphics,"
        "clocks.current.memory,power.draw,power.limit,utilization.gpu,"
        "utilization.memory,memory.used,memory.free"
    )
    return {
        "schema_version": 1,
        "captured_at_utc": datetime.now(UTC).isoformat(),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "gpu": _run([
            "nvidia-smi",
            f"--query-gpu={gpu_fields}",
            "--format=csv,noheader,nounits",
        ]),
        "compute_processes": _run([
            "nvidia-smi",
            "--query-compute-apps=pid,process_name,used_memory",
            "--format=csv,noheader,nounits",
        ]),
        "nvcc": _run(["nvcc", "--version"]),
        "compiler": _run(["c++", "--version"]),
        "cmake": _run(["cmake", "--version"]),
        "uv": _run(["uv", "--version"]),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Capture experiment hardware and toolchain")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    result = capture_environment()
    write_json_atomic(args.output, result)
    print(json.dumps(result, indent=2))
    return 0

