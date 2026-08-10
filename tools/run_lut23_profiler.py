from __future__ import annotations

import argparse
import csv
import json
import re
import subprocess
import sys
import time
from pathlib import Path

from run_v2_benchmark import _NvmlProcessMonitor, _compute_processes


METRICS = (
    "gpu__time_duration.sum",
    "dram__bytes_read.sum",
    "dram__bytes_write.sum",
    "dram__bytes.sum.per_second",
    "lts__t_sector_hit_rate.pct",
    "smsp__sass_inst_executed_op_shared_ld.sum",
    "smsp__sass_inst_executed_op_shared_st.sum",
    "l1tex__data_pipe_lsu_wavefronts_mem_shared_op_ld.sum",
    "l1tex__data_pipe_lsu_wavefronts_mem_shared_op_st.sum",
    "l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum",
    "l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_st.sum",
    "smsp__sass_inst_executed_op_local_ld.sum",
    "smsp__sass_inst_executed_op_local_st.sum",
    "sm__warps_active.avg.pct_of_peak_sustained_active",
    "sm__warps_active.avg.per_cycle_active",
    "sm__maximum_warps_per_active_cycle_pct",
    "launch__registers_per_thread",
    "launch__shared_mem_per_block",
    "smsp__average_warps_issue_stalled_mio_throttle_per_issue_active.ratio",
    "smsp__average_warps_issue_stalled_long_scoreboard_per_issue_active.ratio",
    "smsp__average_warps_issue_stalled_short_scoreboard_per_issue_active.ratio",
    "smsp__sass_thread_inst_executed_op_integer_pred_on.sum",
    "smsp__sass_thread_inst_executed_op_fp32_pred_on.sum",
    "smsp__sass_thread_inst_executed_op_memory_pred_on.sum",
)


def _number(value: str) -> float:
    cleaned = value.strip().replace(",", "")
    if cleaned in {"", "n/a", "N/A"}:
        return 0.0
    return float(cleaned)


def parse_csv(text: str) -> dict[str, float]:
    lines = text.splitlines()
    header_index = next(
        (index for index, line in enumerate(lines) if "Metric Name" in line and "Metric Value" in line),
        None,
    )
    if header_index is None:
        raise ValueError("Nsight Compute CSV header not found")
    reader = csv.DictReader(lines[header_index:])
    result: dict[str, float] = {}
    for row in reader:
        name = (row.get("Metric Name") or "").strip()
        if name:
            result[name] = _number(row.get("Metric Value") or "0")
    if not result:
        raise ValueError("Nsight Compute returned no metric rows")
    return result


def _is_descendant(pid: int, root: int) -> bool:
    current = pid
    for _ in range(32):
        if current == root:
            return True
        try:
            stat = Path(f"/proc/{current}/stat").read_text(encoding="utf-8")
            current = int(stat[stat.rfind(")") + 2 :].split()[1])
        except (OSError, ValueError, IndexError):
            return False
        if current <= 1:
            return False
    return False


def _run_profile_monitored(command: list[str]) -> tuple[subprocess.CompletedProcess[str], dict]:
    before = _compute_processes()
    if before:
        raise RuntimeError(f"GPU busy before profiler: {before}")
    nvml = _NvmlProcessMonitor()
    process = subprocess.Popen(command, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    foreign: dict[str, dict[str, str]] = {}
    errors: list[str] = []
    samples = 0
    maximum_gap = 0.0
    last_sample: float | None = None
    while process.poll() is None:
        now = time.monotonic()
        if last_sample is not None:
            maximum_gap = max(maximum_gap, now - last_sample)
        last_sample = now
        try:
            observed = nvml.processes()
            samples += 1
            for item in observed:
                pid = int(item["pid"])
                if not _is_descendant(pid, process.pid):
                    foreign[item["pid"]] = item
        except Exception as exc:
            errors.append(str(exc))
        time.sleep(0.01)
    stdout, stderr = process.communicate()
    nvml.close()
    after = _compute_processes()
    completed = subprocess.CompletedProcess(command, process.returncode, stdout, stderr)
    monitor = {
        "method": "direct NVML with profiler-child ancestry allowlist",
        "sample_count": samples,
        "maximum_observed_sample_gap_seconds": maximum_gap,
        "foreign_processes": list(foreign.values()),
        "monitor_errors": errors,
        "after_processes": after,
    }
    if foreign or errors or after:
        raise RuntimeError(f"profiler GPU process audit failed: {monitor}")
    return completed, monitor


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Profile the selected LUT23 K_CHUNK exactly once")
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--q2", type=Path, required=True)
    parser.add_argument("--tq1", type=Path, required=True)
    parser.add_argument("--reordered", type=Path, required=True)
    parser.add_argument("--k-chunk", type=int, required=True, choices=(512, 1024, 2048))
    parser.add_argument("--phase-output", type=Path, required=True)
    parser.add_argument("--ncu-csv", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    base = [
        str(args.binary),
        "--manifest", str(args.manifest),
        "--q2", str(args.q2),
        "--tq1", str(args.tq1),
        "--reordered", str(args.reordered),
        "--output", str(args.phase_output),
        "--mode", "profile",
        "--k-chunk", str(args.k_chunk),
        "--warmups", "20",
        "--pairs", "30",
    ]
    try:
        command = [
            "ncu",
            "--csv",
            "--page", "raw",
            "--kernel-name-base", "demangled",
            "--kernel-name", "regex:.*tq1_lut23_smem_streamk_kernel.*",
            "--launch-count", "1",
            "--metrics", ",".join(METRICS),
            *base,
        ]
        profiled, monitor = _run_profile_monitored(command)
        combined = profiled.stdout + "\n" + profiled.stderr
        args.ncu_csv.parent.mkdir(parents=True, exist_ok=True)
        args.ncu_csv.write_text(combined, encoding="utf-8")
        if profiled.returncode != 0:
            raise RuntimeError(
                f"Nsight Compute exited {profiled.returncode}: {profiled.stderr[-2000:]}"
            )
        metrics = parse_csv(combined)
        phase, phase_monitor = _run_profile_monitored(base)
        if phase.returncode != 0:
            raise RuntimeError(f"instrumented phase run exited {phase.returncode}: {phase.stderr}")
        value = lambda name: float(metrics.get(name, 0.0))
        requests = value("smsp__sass_inst_executed_op_shared_ld.sum") + value(
            "smsp__sass_inst_executed_op_shared_st.sum"
        )
        wavefronts = value("l1tex__data_pipe_lsu_wavefronts_mem_shared_op_ld.sum") + value(
            "l1tex__data_pipe_lsu_wavefronts_mem_shared_op_st.sum"
        )
        conflicts = value("l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum") + value(
            "l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_st.sum"
        )
        ratio = wavefronts / requests if requests else float("inf")
        phase_data = json.loads(args.phase_output.read_text(encoding="utf-8"))
        fractions = phase_data["phase_fractions"]
        dominant = max(fractions, key=fractions.get)
        diagnosis = (
            f"{dominant.replace('_', ' ')} dominates the instrumented phase accounting at "
            f"{100 * fractions[dominant]:.2f}%. Shared wavefronts/request is {ratio:.4f}x "
            f"with {conflicts:.0f} reported bank conflicts."
        )
        result = {
            "schema_version": 1,
            "available": True,
            "k_chunk": args.k_chunk,
            "command": command,
            "ncu_version": subprocess.run(
                ["ncu", "--version"], check=True, text=True, capture_output=True
            ).stdout.strip(),
            "kernel_name_regex": ".*tq1_lut23_smem_streamk_kernel.*",
            "process_monitor": monitor,
            "phase_process_monitor": phase_monitor,
            "duration_ns": value("gpu__time_duration.sum"),
            "dram_read_bytes": value("dram__bytes_read.sum"),
            "dram_write_bytes": value("dram__bytes_write.sum"),
            "dram_bandwidth_gb_s": value("dram__bytes.sum.per_second") / 1e9,
            "l2_hit_rate_pct": value("lts__t_sector_hit_rate.pct"),
            "shared_requests": requests,
            "shared_wavefronts": wavefronts,
            "shared_wavefronts_per_request": ratio,
            "shared_bank_conflicts": conflicts,
            "local_loads": value("smsp__sass_inst_executed_op_local_ld.sum"),
            "local_stores": value("smsp__sass_inst_executed_op_local_st.sum"),
            "registers_per_thread": int(value("launch__registers_per_thread")),
            "shared_memory_bytes": value("launch__shared_mem_per_block"),
            "achieved_occupancy_pct": value("sm__warps_active.avg.pct_of_peak_sustained_active"),
            "theoretical_occupancy_pct": value("sm__maximum_warps_per_active_cycle_pct"),
            "active_warps_per_sm": value("sm__warps_active.avg.per_cycle_active"),
            "stall_mio": value(
                "smsp__average_warps_issue_stalled_mio_throttle_per_issue_active.ratio"
            ),
            "stall_long_scoreboard": value(
                "smsp__average_warps_issue_stalled_long_scoreboard_per_issue_active.ratio"
            ),
            "stall_short_scoreboard": value(
                "smsp__average_warps_issue_stalled_short_scoreboard_per_issue_active.ratio"
            ),
            "sass_integer": value("smsp__sass_thread_inst_executed_op_integer_pred_on.sum"),
            "sass_fp32": value("smsp__sass_thread_inst_executed_op_fp32_pred_on.sum"),
            "sass_memory": value("smsp__sass_thread_inst_executed_op_memory_pred_on.sum"),
            "phase_command_stdout": phase.stdout,
            "diagnosis": diagnosis,
            "raw_metrics": metrics,
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    except Exception as exc:
        result = {
            "schema_version": 1,
            "available": False,
            "k_chunk": args.k_chunk,
            "error": str(exc),
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(f"LUT23 profiler failed: {exc}", file=sys.stderr)
        return 2
    print(
        f"wavefronts_per_request={ratio:.6f} registers={result['registers_per_thread']} "
        f"local={result['local_loads']}/{result['local_stores']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
