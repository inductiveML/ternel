from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

from huggingface_hub import hf_hub_download

from .constants import (
    BUILD_DIR,
    MODEL_DIR,
    MODEL_FILENAME,
    MODEL_PATH,
    MODEL_REPO,
    MODEL_REVISION,
    MODEL_SHA256,
    MODEL_SIZE,
    PACKED_PATH,
    REPORTS_DIR,
    RESULTS_DIR,
    ROOT,
    TARGET_COMPUTE_CAPABILITY,
    TARGET_GPU_NAME,
)
from .cuda_reporting import analyze_cuda, write_cuda_reports
from .environment import capture_environment
from .format import read_sidecar, sha256_file
from .inspect_model import inspect_model, write_json_atomic
from .pack_model import pack_model
from .reporting import write_audit_reports
from .verify_model import verify_model, write_verification_report


AUDIT_PATH = RESULTS_DIR / "format_audit.json"
PACKING_PATH = RESULTS_DIR / "packing.json"
VERIFICATION_PATH = RESULTS_DIR / "lossless_verification.json"
BRIDGE_PATH = RESULTS_DIR / "baseline_bridge_validation.json"
V0_PATH = RESULTS_DIR / "cuda_v0.json"
V1_PATH = RESULTS_DIR / "cuda_v1.json"
ENVIRONMENT_PATH = RESULTS_DIR / "environment.json"
FINAL_ANALYSIS_PATH = RESULTS_DIR / "final_analysis.json"


class InvalidExperiment(RuntimeError):
    pass


def _run(command: list[str], *, timeout: int | None = None) -> None:
    print("+", " ".join(command), flush=True)
    subprocess.run(command, cwd=ROOT, check=True, timeout=timeout)


def _ensure_model(force: bool) -> None:
    if force or not MODEL_PATH.exists():
        MODEL_DIR.mkdir(parents=True, exist_ok=True)
        downloaded = Path(
            hf_hub_download(
                repo_id=MODEL_REPO,
                filename=MODEL_FILENAME,
                revision=MODEL_REVISION,
                local_dir=MODEL_DIR,
                force_download=force,
            )
        )
        if downloaded.resolve() != MODEL_PATH.resolve():
            raise RuntimeError(f"unexpected download path {downloaded}")
    if MODEL_PATH.stat().st_size != MODEL_SIZE:
        raise RuntimeError("downloaded GGUF size mismatch")
    digest = sha256_file(MODEL_PATH)
    if digest != MODEL_SHA256:
        raise RuntimeError(f"downloaded GGUF SHA-256 mismatch: {digest}")


def _audit(force: bool) -> dict:
    _ensure_model(force=False)
    if force or not AUDIT_PATH.exists():
        result = inspect_model(MODEL_PATH)
        write_json_atomic(AUDIT_PATH, result)
    else:
        result = json.loads(AUDIT_PATH.read_text(encoding="utf-8"))
    write_audit_reports(AUDIT_PATH)
    if not result["gate_1"]["pass"]:
        raise InvalidExperiment(result["gate_1"]["verdict_if_stopped"])
    return result


def _pack(force: bool) -> dict:
    if force or not PACKED_PATH.exists() or not PACKING_PATH.exists():
        result = pack_model(MODEL_PATH, AUDIT_PATH, PACKED_PATH)
        write_json_atomic(PACKING_PATH, result)
    else:
        result = json.loads(PACKING_PATH.read_text(encoding="utf-8"))
    return result


def _verify(force: bool) -> dict:
    if force or not VERIFICATION_PATH.exists():
        result = verify_model(MODEL_PATH, PACKED_PATH)
        write_json_atomic(VERIFICATION_PATH, result)
    else:
        result = json.loads(VERIFICATION_PATH.read_text(encoding="utf-8"))
    write_verification_report(result)
    if not result["pass"]:
        raise InvalidExperiment("FAIL_NOT_LOSSLESS")
    return result


def _capture_and_validate_idle_gpu(output_path: Path = ENVIRONMENT_PATH) -> dict:
    environment = capture_environment()
    write_json_atomic(output_path, environment)
    gpu = str(environment["gpu"]["stdout"])
    if TARGET_GPU_NAME not in gpu or f", {TARGET_COMPUTE_CAPABILITY}," not in gpu:
        raise InvalidExperiment("target GPU is not RTX 6000 Ada sm_89")
    processes = str(environment["compute_processes"]["stdout"]).strip()
    if processes:
        raise InvalidExperiment(
            "GPU has competing compute processes; CUDA correctness/timing must run on an idle target"
        )
    return environment


def _tensor_offsets() -> tuple[int, int, int, int]:
    audit = json.loads(AUDIT_PATH.read_text(encoding="utf-8"))
    source = next(item for item in audit["tensors"] if item["name"] == "blk.0.ffn_down.weight")
    _, manifest = read_sidecar(PACKED_PATH)
    packed = next(item for item in manifest["tensors"] if item["name"] == source["name"])
    columns, rows = source["shape_gguf"]
    return int(source["source_offset"]), int(packed["packed_offset"]), int(rows), int(columns)


def _benchmark_arguments(output: Path, kernel: str) -> list[str]:
    source_offset, packed_offset, rows, columns = _tensor_offsets()
    return [
        str(BUILD_DIR / "benchmark-gemv"),
        "--q2", str(MODEL_PATH),
        "--tq1", str(PACKED_PATH),
        "--q2-offset", str(source_offset),
        "--tq1-offset", str(packed_offset),
        "--rows", str(rows),
        "--columns", str(columns),
        "--kernel", kernel,
        "--output", str(output),
    ]


def _cuda(force: bool, jobs: int) -> None:
    _capture_and_validate_idle_gpu(RESULTS_DIR / "environment_before_valid_run.json")
    _run(["uv", "run", "python", "tools/bootstrap_prism.py"])
    _run([
        "cmake", "-S", ".", "-B", str(BUILD_DIR), "-G", "Ninja",
        "-DCMAKE_BUILD_TYPE=Release", "-DCMAKE_CUDA_ARCHITECTURES=89",
    ])
    _run(["cmake", "--build", str(BUILD_DIR), "--target", "benchmark-gemv", "validate-prism-bridge", "-j", str(jobs)])
    source_offset, _, rows, columns = _tensor_offsets()
    if force or not BRIDGE_PATH.exists():
        _run([
            str(BUILD_DIR / "validate-prism-bridge"),
            "--q2", str(MODEL_PATH), "--q2-offset", str(source_offset),
            "--rows", str(rows), "--columns", str(columns),
            "--output", str(BRIDGE_PATH),
        ])
    bridge = json.loads(BRIDGE_PATH.read_text(encoding="utf-8"))
    if not bridge["pass"]:
        raise InvalidExperiment("benchmark bridge differs from public GGML graph")
    if force or not V0_PATH.exists():
        _run(_benchmark_arguments(V0_PATH, "v0"))
    v0 = json.loads(V0_PATH.read_text(encoding="utf-8"))
    if v0["benchmark"]["worst_ratio"] > 1.05:
        profile_status = {
            "schema_version": 1,
            "attempted": True,
            "command_succeeded": False,
            "reason": None,
        }
        profile_command = [
            "ncu", "--kernel-name-base", "demangled",
            "-k", "regex:.*tq1_g128_gemv_v0_kernel.*", "-c", "1", "--kill", "yes",
            "--section", "SpeedOfLight", "--section", "Occupancy",
            "--section", "WarpStateStats", "--section", "MemoryWorkloadAnalysis",
            "-o", str(RESULTS_DIR / "tq1_v0_profile"), "-f",
            *_benchmark_arguments(RESULTS_DIR / "profile_probe.json", "v0"),
        ]
        try:
            _run(profile_command, timeout=120)
            profile_status["command_succeeded"] = True
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as error:
            profile_status["reason"] = str(error)
        write_json_atomic(RESULTS_DIR / "profile_status.json", profile_status)
        if force or not V1_PATH.exists():
            _run(_benchmark_arguments(V1_PATH, "v1"))
    elif force or not V1_PATH.exists():
        # Keep one canonical final result even if V0 happens to pass on a future toolchain.
        _run(_benchmark_arguments(V1_PATH, "v1"))
    _capture_and_validate_idle_gpu(ENVIRONMENT_PATH)


def _reports() -> dict:
    analysis = analyze_cuda(
        AUDIT_PATH,
        VERIFICATION_PATH,
        V0_PATH,
        V1_PATH,
        BRIDGE_PATH,
        ENVIRONMENT_PATH,
        FINAL_ANALYSIS_PATH,
    )
    write_cuda_reports(
        FINAL_ANALYSIS_PATH,
        AUDIT_PATH,
        VERIFICATION_PATH,
        V1_PATH,
        ENVIRONMENT_PATH,
        REPORTS_DIR,
    )
    return analysis


def _write_invalid_status(reason: str) -> None:
    write_json_atomic(
        RESULTS_DIR / "run_status.json",
        {"schema_version": 1, "verdict": "INVALID_EXPERIMENT", "reason": reason},
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the gated BONSAI-TQ1-G128 experiment")
    parser.add_argument(
        "stage",
        nargs="?",
        default="all",
        choices=("download", "audit", "pack", "verify", "cuda", "reports", "all"),
    )
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--jobs", type=int, default=max(1, min(24, os.cpu_count() or 1)))
    args = parser.parse_args(argv)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    try:
        if args.stage in ("download", "all"):
            _ensure_model(args.force)
        if args.stage in ("audit", "pack", "verify", "cuda", "all"):
            _audit(args.force if args.stage in ("audit", "all") else False)
        if args.stage in ("pack", "verify", "cuda", "all"):
            _pack(args.force if args.stage in ("pack", "all") else False)
        if args.stage in ("verify", "cuda", "all"):
            _verify(args.force if args.stage in ("verify", "all") else False)
        if args.stage in ("cuda", "all"):
            _cuda(args.force, args.jobs)
        if args.stage in ("reports", "all"):
            result = _reports()
            print(f"VERDICT: {result['verdict']}")
    except InvalidExperiment as error:
        _write_invalid_status(str(error))
        print(f"INVALID_EXPERIMENT: {error}", file=sys.stderr)
        return 4
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
