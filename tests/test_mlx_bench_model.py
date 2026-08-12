"""Gates on the full-model benchmark's bookkeeping.

The timing itself needs two 27B checkpoints and twenty minutes, so what is
gated here is everything around it that can quietly make a benchmark lie: a
workload whose decode segment has nothing in it, a rate divided by the wrong
token count, a prompt built to the wrong length, and the anti-cheat that is
supposed to notice when an arm stops saying the same thing.
"""

from __future__ import annotations

import subprocess
import sys
import threading
from pathlib import Path

import pytest

from bonsai_tq1.format import FormatError
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
        )


@pytest.mark.parametrize(
    "argv",
    [
        ["--artifact", "a", "--baseline", "b"],
        ["--artifact", "a", "--baseline", "b", "--output", "o"],
        ["--artifact", "a", "--baseline", "b", "--isolated-output", "i", "--rounds", "6"],
        ["--artifact", "a", "--baseline", "b", "--isolated-output", "i", "--trials-per-round", "5"],
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
