"""Gates on the full-graph gate.

The B7 run itself takes two 27B checkpoints and the better part of a minute per
prompt, so what can be tested cheaply is the arithmetic that turns two logit
matrices into a verdict. That arithmetic is where a graph gate goes wrong
quietly: a KL divergence computed in the wrong direction, a top-k overlap that
counts positions instead of tokens, or a free-running comparison that scores
tokens generated after the two models stopped seeing the same context.

The representation checks get their own gates because they are the only thing
standing between "the kernels are correct" and "the model silently dequantised
itself and was correct for the wrong reason".
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from bonsai_tq1.format import FormatError
from ternel_mlx.convert import MANIFEST_NAME
from ternel_mlx.graph_gate import (
    MAX_FLOAT_PARAMETER_ELEMENTS,
    PromptRun,
    characteristic_drift,
    check_representation,
    compare_prompt,
    kl_divergence,
    log_softmax,
    top_k_overlap,
)

VOCABULARY = 64


def logits(*, positions: int, seed: int, spread: float = 4.0) -> np.ndarray:
    return np.random.default_rng(seed).normal(0.0, spread, size=(positions, VOCABULARY)).astype(
        np.float32
    )


def run(
    logit_matrix: np.ndarray, *, label: str = "p", continuation: tuple[int, ...] = (1, 2, 3, 4)
) -> PromptRun:
    tokens = tuple(range(logit_matrix.shape[0]))
    return PromptRun(
        label=label, tokens=tokens, logits=logit_matrix, continuation=continuation
    )


def census(
    *, packed_bytes: int, largest_unpacked: int = 1024, linear: int = 497, embedding: int = 1
) -> dict[str, object]:
    return {
        "by_dtype": {},
        "packed_bytes": packed_bytes,
        "unpacked_bytes": 0,
        "total_bytes": packed_bytes,
        "packed_modules": {"PackedLinear": linear, "PackedEmbedding": embedding},
        "largest_unpacked_parameter": {"name": "norm.weight", "elements": largest_unpacked},
    }


def artifact_with_manifest(directory: Path, *, payload_bytes: int) -> Path:
    (directory / MANIFEST_NAME).write_text(
        json.dumps({"quantised": {"payload_bytes": payload_bytes}}), encoding="utf-8"
    )
    return directory


def test_log_softmax_normalises_and_survives_a_logit_range_that_overflows_exp():
    # 800 is far past the fp64 exponent limit, so a log-softmax that exponentiates
    # before shifting returns inf/inf here rather than a distribution.
    values = np.array([[800.0, 799.0, 0.0], [1.0, 2.0, 3.0]])
    result = log_softmax(values)
    assert np.all(np.isfinite(result))
    assert np.allclose(np.exp(result).sum(axis=-1), 1.0)
    assert np.argmax(result, axis=-1).tolist() == [0, 2]


def test_kl_divergence_is_zero_only_when_the_distributions_agree():
    reference = logits(positions=8, seed=1)
    assert np.allclose(kl_divergence(reference, reference), 0.0, atol=1e-12)
    # A constant offset is invisible through the softmax, which is exactly why
    # a raw logit distance alone would be the wrong thing to gate behaviour on.
    assert np.allclose(kl_divergence(reference, reference + 3.0), 0.0, atol=1e-12)
    assert np.all(kl_divergence(reference, reference * 0.5) > 0.0)


def test_kl_divergence_weights_by_the_reference_distribution():
    """The direction is load-bearing, so a swap has to be visible."""
    reference = np.array([[10.0, 0.0, 0.0]])
    candidate = np.array([[10.0, 0.0, 6.0]])
    forward = float(kl_divergence(reference, candidate)[0])
    backward = float(kl_divergence(candidate, reference)[0])
    assert forward != pytest.approx(backward)
    # The reference puts almost no mass on the token the candidate disagrees
    # about, so the loss measured from the reference is the smaller of the two.
    assert forward < backward


def test_top_k_overlap_counts_shared_tokens_not_shared_order():
    reference = np.array([[5.0, 4.0, 3.0, 0.0], [5.0, 4.0, 3.0, 0.0]])
    candidate = np.array([[4.0, 5.0, 3.0, 0.0], [5.0, 4.0, 0.0, 3.0]])
    assert top_k_overlap(reference, candidate, 2).tolist() == [1.0, 1.0]
    assert top_k_overlap(reference, candidate, 1).tolist() == [0.0, 1.0]
    assert top_k_overlap(reference, candidate, 3).tolist() == [1.0, pytest.approx(2 / 3)]


def test_top_k_overlap_rejects_a_non_positive_depth():
    with pytest.raises(FormatError, match="positive"):
        top_k_overlap(logits(positions=2, seed=2), logits(positions=2, seed=3), 0)


def test_a_prompt_run_rejects_logits_that_do_not_cover_its_tokens():
    with pytest.raises(FormatError, match="logit rows"):
        PromptRun(label="p", tokens=(1, 2, 3), logits=logits(positions=2, seed=4), continuation=())


def test_identical_logits_compare_as_bit_exact_with_perfect_agreement():
    matrix = logits(positions=12, seed=5)
    result = compare_prompt(run(matrix), run(matrix), logit_tolerance=0.0, typical_drift=0.0)
    assert result.logits["bit_exact"] is True
    assert result.greedy_agreement == 1.0
    assert result.first_greedy_disagreement is None
    assert result.max_kl_divergence == pytest.approx(0.0, abs=1e-12)
    assert result.candidate_nll == pytest.approx(result.reference_nll)
    assert result.top_k_overlap == {"top_1": 1.0, "top_5": 1.0, "top_10": 1.0}
    assert result.continuation_common_prefix == 4


def test_the_comparison_refuses_two_models_that_were_asked_different_questions():
    """Two tokenizers that disagree would make every downstream number a fiction."""
    matrix = logits(positions=6, seed=6)
    candidate = PromptRun(label="p", tokens=tuple(range(6)), logits=matrix, continuation=())
    reference = PromptRun(label="p", tokens=tuple(range(10, 16)), logits=matrix, continuation=())
    with pytest.raises(FormatError, match="same question"):
        compare_prompt(candidate, reference, logit_tolerance=0.0, typical_drift=0.0)


def test_a_single_flipped_argmax_is_located_rather_than_merely_counted():
    reference = logits(positions=10, seed=7)
    candidate = reference.copy()
    loser = int(np.argsort(reference[6])[-2])
    candidate[6, loser] = reference[6].max() + 1.0
    result = compare_prompt(
        run(candidate), run(reference), logit_tolerance=1e-3, typical_drift=0.0
    )
    assert result.greedy_agreements == 9
    assert result.first_greedy_disagreement == 6
    assert result.logits["worst_case"] == "position_6"


def test_a_flip_across_a_margin_the_typical_drift_cannot_reach_is_unexplained():
    """A reference that was not close, flipped anyway, is a defect."""
    reference = np.zeros((1, VOCABULARY), dtype=np.float32)
    reference[0, 0] = 10.0
    candidate = reference.copy()
    candidate[0, 1] = 11.0
    result = compare_prompt(
        run(candidate), run(reference), logit_tolerance=1.0, typical_drift=0.01
    )
    assert (result.drift_explained_flips, result.unexplained_flips) == (0, 1)
    assert result.smallest_flipped_margin == pytest.approx(10.0)


def test_a_flip_across_a_margin_within_twice_the_typical_drift_is_explained():
    reference = np.zeros((1, VOCABULARY), dtype=np.float32)
    reference[0, 0] = 10.0
    reference[0, 1] = 9.99
    candidate = reference.copy()
    candidate[0, 1] = 10.005
    result = compare_prompt(
        run(candidate), run(reference), logit_tolerance=1.0, typical_drift=0.01
    )
    assert (result.drift_explained_flips, result.unexplained_flips) == (1, 0)
    assert result.smallest_flipped_margin == pytest.approx(0.01, abs=1e-5)


def test_the_classifier_is_not_satisfied_by_the_flip_it_is_judging():
    """The rule this replaced compared a margin against the drift at its own
    position, which every flipped argmax satisfies by construction: reordering
    two logits ``m`` apart requires an error above ``m/2`` right there. A
    yardstick taken from the flip itself would call this defect explained."""
    reference = np.zeros((1, VOCABULARY), dtype=np.float32)
    reference[0, 0] = 40.0
    candidate = reference.copy()
    candidate[0, 1] = 60.0
    local = float(np.abs(candidate[0] - reference[0]).max())
    margin = 40.0
    assert margin <= 2.0 * local  # the discredited rule would explain it away
    result = compare_prompt(
        run(candidate), run(reference), logit_tolerance=1e9, typical_drift=0.05
    )
    assert result.unexplained_flips == 1


def test_the_typical_drift_is_a_median_no_single_position_can_move():
    positions = 21
    matrix = logits(positions=positions, seed=11)
    candidate = matrix.copy()
    candidate += 0.01  # a uniform 0.01 of drift everywhere
    candidate[7, 3] += 1000.0  # one wildly broken position
    drift = characteristic_drift([run(candidate)], [run(matrix)])
    assert drift == pytest.approx(0.01, abs=1e-4)


def test_the_typical_drift_spans_every_prompt():
    first = logits(positions=4, seed=12)
    second = logits(positions=4, seed=13)
    candidates = [run(first + 0.2, label="a"), run(second + 0.4, label="b")]
    references = [run(first, label="a"), run(second, label="b")]
    assert characteristic_drift(candidates, references) == pytest.approx(0.3, abs=1e-4)


def test_the_continuation_prefix_stops_at_the_first_divergence():
    matrix = logits(positions=4, seed=8)
    candidate = run(matrix, continuation=(1, 2, 9, 4, 5))
    reference = run(matrix, continuation=(1, 2, 3, 4, 5))
    result = compare_prompt(candidate, reference, logit_tolerance=0.0, typical_drift=0.0)
    assert result.continuation_common_prefix == 2
    # Four of five tokens match, but only the first two are a claim about the
    # kernels; everything after position 2 was generated from a context the
    # other model never saw.
    assert result.continuation_agreements == 4


def test_a_uniform_logit_shift_leaves_behaviour_untouched_but_moves_the_raw_distance():
    reference = logits(positions=9, seed=9)
    result = compare_prompt(
        run(reference + 2.5), run(reference), logit_tolerance=1.0, typical_drift=0.0
    )
    assert result.greedy_agreement == 1.0
    assert result.max_kl_divergence == pytest.approx(0.0, abs=1e-12)
    assert result.candidate_nll == pytest.approx(result.reference_nll)
    assert result.logits["max_abs_error"] == pytest.approx(2.5)


def test_the_teacher_forced_likelihood_scores_the_token_that_actually_follows():
    positions = 7
    matrix = logits(positions=positions, seed=10)
    result = compare_prompt(run(matrix), run(matrix), logit_tolerance=0.0, typical_drift=0.0)
    # tokens are 0..6, so position i is scored against token i+1 and the final
    # position -- which predicts nothing inside the prompt -- is excluded.
    expected = -log_softmax(matrix[:-1])[np.arange(positions - 1), np.arange(1, positions)].mean()
    assert result.candidate_nll == pytest.approx(float(expected))


def test_a_clean_packed_census_reports_no_representation_problems(tmp_path):
    directory = artifact_with_manifest(tmp_path, payload_bytes=5_882_920_960)
    report = {"quantization_declared": False, "census": census(packed_bytes=5_882_920_960)}
    assert check_representation(report, directory) == []


def test_a_declared_quantisation_is_a_representation_problem(tmp_path):
    """mlx-lm would call nn.quantize and build an affine copy of every weight."""
    directory = artifact_with_manifest(tmp_path, payload_bytes=100)
    report = {"quantization_declared": True, "census": census(packed_bytes=100)}
    assert any("nn.quantize" in problem for problem in check_representation(report, directory))


def test_a_matmul_weight_that_did_not_pack_is_a_representation_problem(tmp_path):
    directory = artifact_with_manifest(tmp_path, payload_bytes=100)
    report = {
        "quantization_declared": False,
        "census": census(
            packed_bytes=100, largest_unpacked=MAX_FLOAT_PARAMETER_ELEMENTS + 1
        ),
    }
    problems = check_representation(report, directory)
    assert any("did not pack" in problem for problem in problems)


def test_packed_bytes_that_disagree_with_the_manifest_are_a_representation_problem(tmp_path):
    directory = artifact_with_manifest(tmp_path, payload_bytes=5_882_920_960)
    report = {"quantization_declared": False, "census": census(packed_bytes=5_882_920_959)}
    problems = check_representation(report, directory)
    assert any("the manifest records" in problem for problem in problems)


def test_a_graph_with_no_packed_modules_is_a_representation_problem(tmp_path):
    directory = artifact_with_manifest(tmp_path, payload_bytes=0)
    report = {
        "quantization_declared": False,
        "census": census(packed_bytes=0, linear=0, embedding=0),
    }
    problems = check_representation(report, directory)
    assert any("packed embedding" in problem for problem in problems)
    assert any("no packed linear layers" in problem for problem in problems)
