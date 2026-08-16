"""Capture the Apple Silicon machine and toolchain a benchmark ran on.

``bonsai_tq1.environment`` asks ``nvidia-smi`` questions that have no answer on
a Mac, so this is the macOS counterpart rather than a rewrite: the probe runner
is imported from there, and the emitted document keeps the same
``schema_version`` / ``captured_at_utc`` / raw-command shape so both sets of
results read the same way.

Two things it deliberately records beyond the hardware. ``pmset`` power mode and
thermal state, because a low-power or thermally-limited run is not comparable to
a nominal one; and per-process GPU occupancy, which is this file's analogue of
the CUDA capture's ``compute_processes`` gate.

That second one was originally a guess -- the machine's load average and
heaviest processes, on the belief that macOS exposes no per-process GPU
accounting. It does. Every Metal client appears in the IO registry as an
``AGXDeviceUserClient`` carrying an ``AppUsage`` record with
``accumulatedGPUTime``, so sampling twice and differencing gives each process's
share of the GPU over the interval. The CPU-pressure proxy is kept alongside it,
because a competitor can be CPU-bound in a way that still perturbs a benchmark,
but it is no longer the only thing standing between a contended run and a
published number. That gap cost a full 39-minute benchmark: a second MLX process
took 85% of the device and the result read as a 7x throughput collapse.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import plistlib
import sys
import time
from collections.abc import Iterator
from contextlib import contextmanager
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

# The IO registry class every Metal client is registered under on Apple silicon.
GPU_CLIENT_CLASS = "AGXDeviceUserClient"

# Long enough that a competitor between kernel launches is still caught, short
# enough that capturing the environment stays a quick operation.
CONTENTION_WINDOW_SECONDS = 3.0


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


def gpu_client_usage() -> dict[int, tuple[str, int]]:
    """Every process holding a Metal client, and the GPU nanoseconds it has used.

    One process can hold several clients -- MLX opens one, a window opens
    another -- so the per-client totals are summed per pid. Absolute totals are
    meaningless on their own, since they count from process start; the caller is
    expected to difference two samples.
    """
    probe = run_command(["ioreg", "-r", "-c", GPU_CLIENT_CLASS, "-a", "-l", "-w0"])
    if probe["returncode"] != 0:
        raise FormatError(f"ioreg failed with returncode {probe['returncode']}")
    document = str(probe["stdout"])
    if not document.strip():
        raise FormatError(f"ioreg reported no {GPU_CLIENT_CLASS} entries; GPU accounting is absent")
    entries = plistlib.loads(document.encode())
    usage: dict[int, tuple[str, int]] = {}
    for entry in entries:
        creator = entry.get("IOUserClientCreator")
        records = entry.get("AppUsage")
        if not isinstance(creator, str) or not isinstance(records, list):
            continue
        # "pid 12804, python3.11"
        head, _, name = creator.partition(", ")
        if not head.startswith("pid "):
            continue
        pid = int(head.removeprefix("pid "))
        nanoseconds = sum(
            record["accumulatedGPUTime"]
            for record in records
            if isinstance(record, dict) and isinstance(record.get("accumulatedGPUTime"), int)
        )
        previous = usage.get(pid)
        usage[pid] = (name, nanoseconds + (previous[1] if previous else 0))
    return usage


def foreign_gpu_share(
    before: dict[int, tuple[str, int]],
    after: dict[int, tuple[str, int]],
    *,
    seconds: float,
    own_pids: frozenset[int],
) -> dict[str, object]:
    """How much of the interval *other* processes spent on the GPU.

    Both the total and the busiest single client are returned, and the total is
    the one worth gating on. These counters partition the device rather than
    overlapping it: measured over a 20s window on a busy desktop, every client's
    share summed to 99.97%, so a microsecond charged to somebody else is a
    microsecond this process did not get. Three competitors at 20% each leave a
    third of the machine and no single one of them looks alarming.

    A process that appears only in ``after`` is skipped rather than counted from
    zero: its accumulated time predates the window, so differencing it would
    charge the window for work it did not contain. A process that exits during
    the window disappears from the registry with it -- which is why the gate
    samples per timed unit rather than once around a whole run.
    """
    if seconds <= 0:
        raise FormatError(f"cannot apportion a {seconds}s window")
    ranked: list[dict[str, object]] = []
    for pid, (name, nanoseconds) in after.items():
        if pid in own_pids:
            continue
        earlier = before.get(pid)
        if earlier is None:
            continue
        share = (nanoseconds - earlier[1]) / (seconds * 1e9)
        if share > 0:
            ranked.append({"pid": pid, "command": name, "gpu_share": share})
    ranked.sort(key=lambda entry: float(entry["gpu_share"]), reverse=True)
    return {
        "seconds": seconds,
        "total_share": sum(float(entry["gpu_share"]) for entry in ranked),
        "busiest_share": float(ranked[0]["gpu_share"]) if ranked else 0.0,
        "clients": ranked[:BUSIEST_PROCESS_COUNT],
    }


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


def sampled_gpu_usage() -> tuple[float, dict[int, tuple[str, int]]]:
    """The client counters, with the instant they were read.

    ``gpu_client_usage`` shells out to ``ioreg``, which is not free -- a few
    hundred milliseconds on a busy desktop. The counters therefore accumulate
    over the interval between the two registry *reads*, not over the interval
    between the calls returning, and dividing by the latter overstates every
    share by the sampling overhead. Over a forty-minute benchmark that is
    nothing; over a block that lasts ten milliseconds it reported 210% of the
    device in use and voided a run that had the machine to itself.

    The instant returned is the midpoint of the call. Where inside it the
    registry was actually read is not observable, so the midpoint is the
    estimator with the smallest worst-case error -- half the call, in either
    direction, instead of a whole call in one.
    """
    started = time.perf_counter()
    usage = gpu_client_usage()
    return (started + time.perf_counter()) / 2.0, usage


class GpuContentionError(FormatError):
    """A timed block shared the GPU, so its numbers mean nothing.

    Distinct from every other :class:`FormatError` because it is the one failure
    a caller can legitimately answer by measuring again: the block is void, not
    wrong. Everything else -- a shape that does not match, an arm that produced
    two different continuations -- is a fact about the run that repeating it
    would only repeat. A caller that retried on the message text would also
    retry on those.
    """

    def __init__(self, message: str, *, share: dict[str, object]) -> None:
        super().__init__(message)
        self.share = share


@contextmanager
def exclusive_gpu(
    label: str, *, max_foreign_share: float, record: list[dict[str, object]]
) -> Iterator[None]:
    """Fail the run if other processes were on the GPU while this block was timed.

    What is gated is the share taken by everybody else together, not by the worst
    one of them: the counters partition the device, so it is the sum that says
    how much of the machine this measurement actually had.

    Wrapped around each timed unit rather than around a whole benchmark, because
    a competitor that appears for four minutes of a forty-minute run is still
    fatal to the four minutes it touched, and averaging it over the run would
    hide it.

    This is not a hypothetical. A 39-minute model benchmark was run while an
    unrelated MLX process held 85% of the device; the packed arm read as 2.9
    tok/s against its true 20.9, and nothing in the harness objected, because the
    only contention signal was CPU pressure and a GPU-bound competitor barely
    shows up there.

    The window the share is apportioned over is the one the counters cover --
    see :func:`sampled_gpu_usage` -- which is wider than the block itself. The
    block's own duration is recorded beside it so a reader can see how much of
    the window was the sampling rather than the measurement.
    """
    first_at, before = sampled_gpu_usage()
    started = time.perf_counter()
    yield
    elapsed = time.perf_counter() - started
    second_at, after = sampled_gpu_usage()
    share = foreign_gpu_share(
        before, after, seconds=second_at - first_at, own_pids=frozenset({os.getpid()})
    )
    record.append({"label": label, "timed_seconds": elapsed} | share)
    total = float(share["total_share"])
    if total > max_foreign_share:
        busiest = share["clients"][0]
        raise GpuContentionError(
            f"{label} was timed while other processes used {total:.1%} of the GPU, over "
            f"the {max_foreign_share:.1%} this run allows; the busiest was pid "
            f"{busiest['pid']} ({busiest['command']}) at "
            f"{float(busiest['gpu_share']):.1%}; the measurement is void",
            share=share,
        )


def measure_gpu_contention(*, seconds: float) -> dict[str, object]:
    """Sample GPU occupancy over a short window, excluding this process."""
    first_at, before = sampled_gpu_usage()
    time.sleep(seconds)
    second_at, after = sampled_gpu_usage()
    return foreign_gpu_share(
        before, after, seconds=second_at - first_at, own_pids=frozenset({os.getpid()})
    )


# How long one occupancy sample covers. Two seconds is long enough that the two
# ``ioreg`` reads bracketing it dominate nothing -- see :func:`sampled_gpu_usage`
# -- and short enough to notice a display that has just gone back to sleep.
CONTENTION_SAMPLE_SECONDS = 2.0


def await_quiet_gpu(
    *, max_foreign_share: float, timeout_seconds: float, sample_seconds: float
) -> dict[str, object]:
    """Poll until nobody else is on the GPU, or the budget runs out.

    An efficiency device and nothing more: no measurement's validity rests on
    this, because :func:`exclusive_gpu` still voids any block that shared the
    device whatever this returned. What it buys is not re-timing a twenty-minute
    benchmark straight back into the contention that just voided it -- on this
    machine the usual competitor is a display that woke up, which goes away on
    its own in a minute or two if something waits for it.

    Returns the last sample taken, with ``settled`` saying whether it came in
    under the bar or the budget simply expired.
    """
    if timeout_seconds < 0.0:
        raise FormatError(f"a wait cannot be negative, got {timeout_seconds} seconds")
    deadline = time.perf_counter() + timeout_seconds
    while True:
        share = measure_gpu_contention(seconds=sample_seconds)
        if float(share["total_share"]) <= max_foreign_share:
            return share | {"settled": True}
        if time.perf_counter() >= deadline:
            return share | {"settled": False}


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
        "gpu_contention": measure_gpu_contention(seconds=CONTENTION_WINDOW_SECONDS),
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
