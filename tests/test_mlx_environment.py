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
