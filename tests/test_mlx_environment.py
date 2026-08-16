"""Gates on the GPU-exclusivity check every timed run is wrapped in.

This file exists because of a specific failure. A 39-minute model benchmark was
run while an unrelated MLX process held 85% of the device; the packed arm read
as 2.9 tok/s against its true 20.9, and the harness reported it as a result,
because the only contention signal at the time was CPU pressure -- which a
GPU-bound competitor barely moves. What is gated here is the instrument that
replaced it, and the arithmetic it uses to decide a run is void.
"""

from __future__ import annotations

import time
from unittest import mock

import pytest

from bonsai_tq1.format import FormatError
from ternel_mlx import environment
from ternel_mlx.environment import (
    GpuContentionError,
    await_quiet_gpu,
    exclusive_gpu,
    foreign_gpu_share,
    gpu_client_usage,
    measure_gpu_contention,
)


def test_a_competitor_on_the_gpu_voids_the_measurement_it_overlapped():
    samples = iter([
        {999: ("competitor", 0)},
        {999: ("competitor", 850_000_000)},
    ])
    record: list[dict[str, object]] = []
    with pytest.raises(FormatError, match=r"pid 999 \(competitor\)"):
        with mock.patch.object(environment, "gpu_client_usage", lambda: next(samples)):
            with exclusive_gpu("pp32", max_foreign_share=0.05, record=record):
                time.sleep(1.0)
    # Recorded before it raised, so a void measurement is still auditable.
    assert record[0]["label"] == "pp32"
    assert record[0]["busiest_share"] > 0.05


def test_a_quiet_gpu_passes_the_gate_and_is_recorded_as_quiet():
    steady = {999: ("competitor", 12_345)}
    record: list[dict[str, object]] = []
    with mock.patch.object(environment, "gpu_client_usage", lambda: dict(steady)):
        with exclusive_gpu("pp32", max_foreign_share=0.05, record=record):
            pass
    assert record[0]["busiest_share"] == 0.0
    assert record[0]["total_share"] == 0.0
    assert record[0]["clients"] == []


def test_several_small_competitors_together_void_a_run_none_of_them_would():
    """The counters partition the device, so what matters is how much of it this
    process was left, not whether any single rival looked alarming. Four clients
    at a fifth each leave a fifth of the machine while every one of them clears a
    25% bar."""
    samples = iter([
        {n: (f"rival{n}", 0) for n in range(4)},
        {n: (f"rival{n}", 200_000_000) for n in range(4)},
    ])
    record: list[dict[str, object]] = []
    with pytest.raises(FormatError, match="other processes used 7[0-9].[0-9]% of the GPU"):
        with mock.patch.object(environment, "gpu_client_usage", lambda: next(samples)):
            with exclusive_gpu("pp32", max_foreign_share=0.25, record=record):
                time.sleep(1.0)
    # No single rival is over the bar; together they are, and together is what counts.
    assert float(record[0]["busiest_share"]) < 0.25
    assert float(record[0]["total_share"]) > 0.25


def test_the_total_counts_competitors_that_are_too_small_to_be_listed():
    """``clients`` is truncated for readability; a total computed from the
    truncated list would under-report a machine death by a thousand cuts."""
    crowd = environment.BUSIEST_PROCESS_COUNT + 6
    share = foreign_gpu_share(
        {n: (f"rival{n}", 0) for n in range(crowd)},
        {n: (f"rival{n}", 10_000_000) for n in range(crowd)},
        seconds=1.0,
        own_pids=frozenset(),
    )
    assert len(share["clients"]) == environment.BUSIEST_PROCESS_COUNT
    assert float(share["total_share"]) == pytest.approx(0.01 * crowd)


def test_a_competitor_under_the_ceiling_is_recorded_without_voiding_the_run():
    """A compositor drawing the screen is real GPU use and cannot be avoided;
    the gate has to distinguish it from another model, not from zero."""
    samples = iter([
        {42: ("WindowServer", 0)},
        {42: ("WindowServer", 20_000_000)},
    ])
    record: list[dict[str, object]] = []
    with mock.patch.object(environment, "gpu_client_usage", lambda: next(samples)):
        with exclusive_gpu("pp32", max_foreign_share=0.25, record=record):
            time.sleep(1.0)
    assert 0.0 < float(record[0]["busiest_share"]) < 0.25


def test_this_process_is_never_counted_as_its_own_competitor():
    share = foreign_gpu_share(
        {7: ("us", 0)}, {7: ("us", 10_000_000_000)}, seconds=1.0, own_pids=frozenset({7})
    )
    assert share["busiest_share"] == 0.0


def test_a_process_that_appeared_mid_window_is_not_charged_for_its_whole_life():
    """Its accumulated time predates the window; differencing it from zero would
    bill this measurement for GPU work done before the window opened."""
    share = foreign_gpu_share(
        {}, {8: ("latecomer", 10_000_000_000)}, seconds=1.0, own_pids=frozenset()
    )
    assert share["busiest_share"] == 0.0


def test_several_clients_held_by_one_process_are_charged_to_it_once():
    """MLX opens a client and a window opens another; a process that is half the
    GPU across two clients is half the GPU, not two quarters."""
    share = foreign_gpu_share(
        {9: ("both", 0)}, {9: ("both", 500_000_000)}, seconds=1.0, own_pids=frozenset()
    )
    assert share["busiest_share"] == pytest.approx(0.5)
    assert len(share["clients"]) == 1


def test_clients_are_ranked_so_the_busiest_is_the_one_reported():
    share = foreign_gpu_share(
        {1: ("quiet", 0), 2: ("loud", 0)},
        {1: ("quiet", 100_000_000), 2: ("loud", 900_000_000)},
        seconds=1.0,
        own_pids=frozenset(),
    )
    assert [entry["command"] for entry in share["clients"]] == ["loud", "quiet"]
    assert share["busiest_share"] == pytest.approx(0.9)
    assert share["total_share"] == pytest.approx(1.0)


def test_an_empty_window_cannot_be_apportioned():
    with pytest.raises(FormatError, match="cannot apportion"):
        foreign_gpu_share({}, {}, seconds=0.0, own_pids=frozenset())


def test_the_machine_really_does_expose_per_process_gpu_accounting():
    """The gate rests on this being present, so it is asserted against the real
    IO registry rather than only against fixtures."""
    usage = gpu_client_usage()
    assert usage, "no Metal clients at all means the accounting source moved"
    for pid, (name, nanoseconds) in usage.items():
        assert isinstance(pid, int) and pid > 0
        assert isinstance(name, str)
        assert isinstance(nanoseconds, int) and nanoseconds >= 0


def test_a_live_contention_sample_apportions_a_real_window():
    measured = measure_gpu_contention(seconds=0.5)
    assert measured["seconds"] >= 0.5
    assert 0.0 <= float(measured["busiest_share"]) <= float(measured["total_share"])
    for entry in measured["clients"]:
        assert float(entry["gpu_share"]) > 0.0


def test_the_window_covers_the_sampling_and_not_just_the_block():
    """A short block must not be charged for the time ``ioreg`` itself took.

    Reading the registry costs a few hundred milliseconds. The counters
    accumulate across both reads, so apportioning that delta over the block
    alone divides a wide numerator by a narrow denominator -- which is not a
    small error on a short block: it reported 210% of a device that this process
    had entirely to itself, and voided the run.

    Each read here takes half a second, so the two of them span a window of
    roughly 0.5s around a block that lasts a millisecond. The competitor uses
    50ms of GPU in that window, which is a tenth of it. Charged to the block
    instead, the same 50ms would read as fifty times the machine.
    """
    slow_registry = iter([
        {999: ("competitor", 0)},
        {999: ("competitor", 50_000_000)},
    ])

    def sample() -> dict[int, tuple[str, int]]:
        # Half the cost before the read and half after, which is where the
        # midpoint estimator assumes the registry is actually sampled.
        time.sleep(0.25)
        entry = next(slow_registry)
        time.sleep(0.25)
        return entry

    record: list[dict[str, object]] = []
    with mock.patch.object(environment, "gpu_client_usage", sample):
        with exclusive_gpu("short block", max_foreign_share=0.25, record=record):
            time.sleep(0.001)

    assert float(record[0]["total_share"]) == pytest.approx(0.10, abs=0.02)
    # Both durations are kept, so a reader can tell a quiet run from one whose
    # window was mostly sampling overhead.
    assert float(record[0]["timed_seconds"]) < 0.1
    assert float(record[0]["seconds"]) > 0.4


def test_contention_is_a_distinct_failure_a_caller_may_answer_by_measuring_again():
    """A void block is the one failure worth repeating, so it has its own type.

    Everything else ``exclusive_gpu``'s callers raise is a fact about the run --
    a shape that does not match, an arm that changed its answer -- and repeating
    it would only reproduce it. A caller that retried on the message text would
    retry on those too, so the distinction is carried by the exception rather
    than by its wording, and the share that voided the block travels with it so
    a retry loop can record what it was up against.
    """
    samples = iter([
        {999: ("competitor", 0)},
        {999: ("competitor", 850_000_000)},
    ])
    record: list[dict[str, object]] = []
    with mock.patch.object(environment, "gpu_client_usage", lambda: next(samples)):
        with pytest.raises(GpuContentionError) as raised:
            with exclusive_gpu("block", max_foreign_share=0.25, record=record):
                time.sleep(0.01)

    assert isinstance(raised.value, FormatError)
    assert float(raised.value.share["total_share"]) > 0.25
    assert raised.value.share["clients"][0]["command"] == "competitor"


# The counters are nanoseconds of GPU time, so a share is a delta divided by the
# window it covers. A fifth of a second is long enough that the jitter around
# ``time.sleep`` is a rounding error on the shares these tests assert.
WAIT_SAMPLE_SECONDS = 0.2
BUSY_NANOSECONDS = int(WAIT_SAMPLE_SECONDS * 1e9 * 0.9)
QUIET_NANOSECONDS = int(WAIT_SAMPLE_SECONDS * 1e9 * 0.01)


def scripted_reads(deltas: tuple[int, ...]):
    """One ``gpu_client_usage`` per read, paired into before/after per sample."""
    reads: list[dict[int, tuple[str, int]]] = []
    for delta in deltas:
        reads.append({999: ("competitor", 0)})
        reads.append({999: ("competitor", delta)})
    return lambda: reads.pop(0)


def test_the_retry_wait_returns_as_soon_as_the_gpu_is_quiet():
    """A wait that outlasted the contention would be pure delay.

    Two samples are scripted and only two are supplied, so a third poll would
    fail on an empty list rather than quietly pass.
    """
    with mock.patch.object(
        environment, "gpu_client_usage", scripted_reads((BUSY_NANOSECONDS, QUIET_NANOSECONDS))
    ):
        settled = await_quiet_gpu(
            max_foreign_share=0.25,
            timeout_seconds=30.0,
            sample_seconds=WAIT_SAMPLE_SECONDS,
        )

    assert settled["settled"] is True
    assert float(settled["total_share"]) < 0.25


def test_the_retry_wait_gives_up_rather_than_blocking_forever():
    """The budget is a bound, and the sample it gives up on is reported.

    Giving up is not a failure: the retry it precedes is gated by
    :func:`exclusive_gpu` regardless, so a wait that expires costs one more
    voided attempt rather than a wrong number.
    """
    with mock.patch.object(
        environment, "gpu_client_usage", scripted_reads((BUSY_NANOSECONDS,))
    ):
        gave_up = await_quiet_gpu(
            max_foreign_share=0.25,
            timeout_seconds=0.0,
            sample_seconds=WAIT_SAMPLE_SECONDS,
        )

    assert gave_up["settled"] is False
    assert float(gave_up["total_share"]) > 0.25


def test_a_negative_wait_is_rejected():
    with pytest.raises(FormatError, match="cannot be negative"):
        await_quiet_gpu(max_foreign_share=0.25, timeout_seconds=-1.0, sample_seconds=0.01)
