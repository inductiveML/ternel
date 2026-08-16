"""Gates on the full-model benchmark's bookkeeping.

The timing itself needs two 27B checkpoints and twenty minutes, so what is
gated here is everything around it that can quietly make a benchmark lie: a
workload whose decode segment has nothing in it, a rate divided by the wrong
token count, a prompt built to the wrong length, and the anti-cheat that is
supposed to notice when an arm stops saying the same thing.
"""

from __future__ import annotations

import itertools
import subprocess
import sys
import threading
from contextlib import contextmanager
from pathlib import Path

import pytest

from bonsai_tq1.format import FormatError
from ternel_mlx import bench_model
from ternel_mlx.bench_model import (
    RSS_POLL_SECONDS,
    TRIALS,
    WORKLOADS,
    Workload,
    common_prefix_length,
    corpus_tokens,
    main,
    measure_isolated,
    prompt_digest,
    rates,
    sample_resident_bytes,
    workload_by_label,
)
from ternel_mlx.environment import GpuContentionError
from ternel_mlx.graph_gate import PROMPTS


class WordTokenizer:
    """Enough of a tokenizer to build a corpus from, with stable ids."""

    def __init__(self) -> None:
        self.vocabulary: dict[str, int] = {}

    def encode(self, text: str) -> list[int]:
        return [self.vocabulary.setdefault(word, len(self.vocabulary)) for word in text.split()]


def test_a_workload_with_nothing_to_decode_is_rejected():
    """The decode rate divides by ``generate_tokens - 1``, so one token is zero."""
    with pytest.raises(FormatError, match="nothing to time"):
        Workload(label="w", prompt_tokens=8, generate_tokens=1)


def test_a_workload_with_an_empty_prompt_is_rejected():
    with pytest.raises(FormatError, match="not one"):
        Workload(label="w", prompt_tokens=0, generate_tokens=8)


def test_every_shipped_workload_is_reachable_by_its_label():
    for workload in WORKLOADS:
        assert workload_by_label(workload.label) is workload
    with pytest.raises(FormatError, match="unknown workload"):
        workload_by_label("pp1_tg1")


def test_shipped_workload_labels_are_distinct():
    """The probe subprocess is addressed by label, so a duplicate would silently
    measure the wrong shape."""
    labels = [workload.label for workload in WORKLOADS]
    assert len(set(labels)) == len(labels)


def test_rates_divide_the_prompt_by_prefill_and_the_rest_by_decode():
    workload = Workload(label="w", prompt_tokens=100, generate_tokens=11)
    measured = rates(workload, 2.0, 5.0)
    assert measured["prompt_tokens_per_second"] == pytest.approx(50.0)
    # Ten tokens were generated after the first, which the prefill segment paid for.
    assert measured["generated_tokens_per_second"] == pytest.approx(2.0)
    assert measured["time_to_first_token_seconds"] == 2.0
    assert measured["decode_seconds"] == 5.0


def test_the_corpus_reaches_the_requested_length_exactly():
    tokenizer = WordTokenizer()
    for length in (1, 32, 512, 2048):
        assert len(corpus_tokens(tokenizer, at_least=length)) == length


def test_the_corpus_is_a_prefix_of_every_longer_corpus():
    """Each workload slices the same token list, so the short prompts have to be
    the openings of the long ones or the arms would read different text."""
    tokenizer = WordTokenizer()
    long = corpus_tokens(tokenizer, at_least=2048)
    assert corpus_tokens(tokenizer, at_least=512) == long[:512]
    assert corpus_tokens(tokenizer, at_least=32) == long[:32]


def test_the_corpus_repeats_rather_than_running_out():
    """The prompts are a few hundred tokens and the longest workload wants 2048."""
    tokenizer = WordTokenizer()
    passage = "\n\n".join(text for _, text in PROMPTS)
    once = len(tokenizer.encode(passage))
    assert once < 2048
    corpus = corpus_tokens(WordTokenizer(), at_least=2048)
    assert corpus[once : once * 2] == corpus[:once]


def test_the_common_prefix_stops_at_the_first_divergence():
    assert common_prefix_length((1, 2, 3, 4), (1, 2, 3, 4)) == 4
    assert common_prefix_length((1, 2, 9, 4), (1, 2, 3, 4)) == 2
    assert common_prefix_length((9, 2), (1, 2)) == 0
    assert common_prefix_length((), (1, 2)) == 0


def test_the_resident_sampler_reads_a_live_process_and_stops_when_told():
    process = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"])
    stop = threading.Event()
    samples: list[int] = []
    watcher = threading.Thread(target=lambda: samples.extend(sample_resident_bytes(process.pid, stop)))
    watcher.start()
    try:
        watcher.join(timeout=RSS_POLL_SECONDS * 12)
        assert watcher.is_alive()
        stop.set()
        watcher.join(timeout=RSS_POLL_SECONDS * 12)
        assert not watcher.is_alive()
    finally:
        stop.set()
        process.kill()
        process.wait()
    assert samples
    # A Python interpreter is worth more than a megabyte and less than a gigabyte,
    # which is enough to prove the number is bytes and not the kibibytes ``ps``
    # actually prints.
    assert all(1 << 20 < sample < 1 << 30 for sample in samples)


def test_the_resident_sampler_survives_a_process_that_is_already_gone():
    process = subprocess.Popen([sys.executable, "-c", ""])
    process.wait()
    stop = threading.Event()
    threading.Timer(RSS_POLL_SECONDS * 4, stop.set).start()
    assert sample_resident_bytes(process.pid, stop) == []


def test_the_prompt_digest_separates_sequences_and_survives_a_round_trip():
    """The arms build their corpora in different processes, so the parent
    compares digests rather than trusting that two tokenizers agreed."""
    assert prompt_digest((1, 2, 3)) == prompt_digest((1, 2, 3))
    assert prompt_digest((1, 2, 3)) != prompt_digest((1, 2, 4))
    assert prompt_digest((1, 2, 3)) != prompt_digest((1, 2, 3, 0))
    assert prompt_digest(()) != prompt_digest((0,))


def test_a_single_round_cannot_alternate_and_is_rejected():
    """Alternation is the whole reason rounds exist; one round is one order."""
    with pytest.raises(FormatError, match="cannot alternate"):
        measure_isolated(
            Path("artifact"),
            Path("baseline"),
            rounds=1,
            trials_per_round=64,
            max_foreign_gpu_share=0.05,
            round_attempts=1,
            round_retry_wait_seconds=0.0,
        )


def test_too_few_isolated_samples_are_rejected_before_anything_is_launched():
    """``paired_bootstrap_ratio`` refuses fewer than 30, and finding that out
    after a half-hour of subprocesses would be an expensive way to learn it."""
    with pytest.raises(FormatError, match=f"refuses fewer than {TRIALS}"):
        measure_isolated(
            Path("artifact"),
            Path("baseline"),
            rounds=4,
            trials_per_round=7,
            max_foreign_gpu_share=0.05,
            round_attempts=1,
            round_retry_wait_seconds=0.0,
        )


class ScriptedGpu:
    """A GPU that is busy on the attempts named, and quiet on the rest.

    Stands in for ``exclusive_gpu`` so the retry can be exercised without two
    27B checkpoints. It records what it was asked to time, which is how the
    tests below tell a re-measured round from a half-measured one.
    """

    def __init__(self, busy_labels: set[str]) -> None:
        self.busy_labels = busy_labels
        self.timed: list[str] = []

    @contextmanager
    def __call__(self, label, *, max_foreign_share, record):
        self.timed.append(label)
        yield
        share = {"total_share": 0.9, "clients": [{"pid": 1, "command": "other"}]}
        record.append({"label": label} | share)
        if label in self.busy_labels:
            raise GpuContentionError(f"{label} is void", share=share)


def scripted_isolated(monkeypatch, gpu: ScriptedGpu, *, rounds: int, round_attempts: int):
    """``measure_isolated`` with the GPU and the subprocess both stubbed out.

    Each solo run reports a decode time no other run reports, so the series that
    reach the summary say exactly which runs were kept. Returns the payload and
    those series, keyed by arm.
    """
    monkeypatch.setattr(bench_model, "exclusive_gpu", gpu)
    monkeypatch.setattr(bench_model, "capture_environment", lambda: {})

    counter = itertools.count(1)

    def solo(path: Path, *, trials: int) -> dict[str, object]:
        stamp = float(next(counter))
        return {
            "path": str(path),
            "workloads": {
                workload.label: {
                    "prefill_seconds": [1.0] * trials,
                    "decode_seconds": [stamp] * trials,
                    "prompt_digest": "digest",
                    "tokens": [7, 8, 9],
                }
                for workload in WORKLOADS
            },
        }

    monkeypatch.setattr(bench_model, "run_solo", solo)

    kept: list[list[float]] = []
    real_summary = bench_model.quantile_summary

    def recording_summary(values):
        kept.append([float(value) for value in values])
        return real_summary(values)

    monkeypatch.setattr(bench_model, "quantile_summary", recording_summary)

    payload = measure_isolated(
        Path("artifact"),
        Path("baseline"),
        rounds=rounds,
        trials_per_round=10,
        max_foreign_gpu_share=0.25,
        round_attempts=round_attempts,
        round_retry_wait_seconds=0.0,
    )
    # Four series per workload, in the order the payload builds them: prefill
    # candidate, prefill reference, decode candidate, decode reference.
    return payload, {"candidate": sorted(set(kept[2])), "reference": sorted(set(kept[3]))}


def test_a_round_that_shared_the_gpu_is_measured_again_instead_of_ending_the_run(monkeypatch):
    """A desktop that wakes for ten seconds should cost ten seconds, not the run.

    The bar itself does not move: the block that shared the GPU is still void.
    What changes is that the rounds already collected, and the model loads that
    produced them, survive it.
    """
    gpu = ScriptedGpu({"round 2 attempt 1 reference"})
    payload, _ = scripted_isolated(monkeypatch, gpu, rounds=4, round_attempts=3)

    assert [entry["round"] for entry in payload["voided_rounds"]] == [2]
    assert payload["voided_rounds"][0]["attempt"] == 1
    assert payload["round_attempts"] == 3
    # Four rounds of two arms, plus the one that had to be done twice.
    assert len(gpu.timed) == 9


def test_both_arms_of_a_voided_round_are_discarded_even_when_only_one_was_hit(monkeypatch):
    """The bootstrap pairs samples by position, so a half-round misaligns it.

    Round 2 runs the reference first; the *candidate* is the one that shares the
    GPU, so the reference's own block passed and its numbers are sitting there,
    tempting. Keeping them would give that arm one more sample than its partner
    and shift every pair after it.

    The solo runs are stamped 1 upward in launch order, so round 2's first
    attempt is runs 3 (reference) and 4 (candidate). Neither may appear: 3 is
    the tempting one, and 4 is the void itself.
    """
    gpu = ScriptedGpu({"round 2 attempt 1 candidate"})
    payload, kept = scripted_isolated(monkeypatch, gpu, rounds=4, round_attempts=3)

    assert kept["reference"] == [2.0, 5.0, 8.0, 9.0]
    assert kept["candidate"] == [1.0, 6.0, 7.0, 10.0]
    assert payload["samples_per_arm"] == 40


def test_a_round_that_never_gets_a_quiet_gpu_still_fails_the_run(monkeypatch):
    """The retry is a budget, not a licence to keep going until it passes."""
    gpu = ScriptedGpu({f"round 1 attempt {n} candidate" for n in (1, 2, 3)})
    with pytest.raises(FormatError, match="shared the GPU on all 3 attempts"):
        scripted_isolated(monkeypatch, gpu, rounds=4, round_attempts=3)


def test_a_failure_that_is_not_contention_is_not_retried(monkeypatch):
    """Repeating a shape mismatch would only reproduce it, slowly."""

    def broken(path, *, trials):
        raise FormatError("the arms were not timed on the same prompt")

    gpu = ScriptedGpu(set())
    monkeypatch.setattr(bench_model, "exclusive_gpu", gpu)
    monkeypatch.setattr(bench_model, "capture_environment", lambda: {})
    monkeypatch.setattr(bench_model, "run_solo", broken)
    with pytest.raises(FormatError, match="not timed on the same prompt"):
        measure_isolated(
            Path("artifact"),
            Path("baseline"),
            rounds=4,
            trials_per_round=10,
            max_foreign_gpu_share=0.25,
            round_attempts=3,
            round_retry_wait_seconds=0.0,
        )
    assert len(gpu.timed) == 1


def test_a_round_needs_at_least_one_attempt():
    with pytest.raises(FormatError, match="at least one attempt"):
        measure_isolated(
            Path("artifact"),
            Path("baseline"),
            rounds=4,
            trials_per_round=10,
            max_foreign_gpu_share=0.25,
            round_attempts=0,
            round_retry_wait_seconds=0.0,
        )


@pytest.mark.parametrize(
    "argv",
    [
        ["--artifact", "a", "--baseline", "b"],
        ["--artifact", "a", "--baseline", "b", "--output", "o"],
        ["--artifact", "a", "--baseline", "b", "--isolated-output", "i", "--rounds", "6"],
        ["--artifact", "a", "--baseline", "b", "--isolated-output", "i", "--trials-per-round", "5"],
        # No retry budget: a voided round would have nowhere to go.
        [
            "--artifact", "a", "--baseline", "b", "--isolated-output", "i",
            "--rounds", "6", "--trials-per-round", "5", "--max-foreign-gpu-share", "0.25",
        ],
        # No wait budget between retries: the run would not know how patient to be.
        [
            "--artifact", "a", "--baseline", "b", "--isolated-output", "i",
            "--rounds", "6", "--trials-per-round", "5", "--max-foreign-gpu-share", "0.25",
            "--round-attempts", "4",
        ],
        ["--baseline", "b", "--isolated-output", "i"],
        ["--solo", "a"],
        ["--probe", "a"],
        # No contention ceiling: the run would not know what it tolerates.
        ["--artifact", "a", "--baseline", "b", "--output", "o", "--trials", "30"],
    ],
)
def test_an_incompletely_specified_run_is_refused_rather_than_defaulted(argv):
    """Every knob is explicit, so a missing one is an error and never a default."""
    with pytest.raises(SystemExit) as exit_info:
        main(argv)
    assert exit_info.value.code == 2
