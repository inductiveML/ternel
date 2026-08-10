"""The A4 gate's own instrument, checked before it is used to judge kernels.

``kernel_gate.compare`` is what decides whether a Metal kernel is correct, so a
weakness in it is invisible in exactly the way that matters: every arm passes and
nothing is learned. These tests are therefore adversarial about the comparator
rather than about the kernels -- they hand it results that are definitely wrong
and require that it says so.
"""

from __future__ import annotations

import numpy as np
import pytest

from bonsai_tq1.format import FormatError
from ternel_mlx.kernel_gate import (
    ORACLE_RELATIVE_TOLERANCE,
    RANDOM_CASES,
    REFERENCE_ROW_BUDGET,
    SAME_ORDER_RELATIVE_TOLERANCE,
    activation_cases,
    compare,
    non_finite_cases,
    pair_tolerance,
    reference_prefix,
)
from ternel_mlx.layout import PackedTensorLayout


def labels_for(rows: int) -> tuple[str, ...]:
    return tuple(f"case_{index}" for index in range(rows))


def test_a_huge_case_cannot_excuse_a_wrong_small_one() -> None:
    """The defect this comparator is normalised per case to prevent.

    The gate's activation set spans twenty orders of magnitude on purpose. Under
    a single global normalisation the 1e18 case sets an absolute tolerance near
    1e13, and every unit-magnitude case then passes no matter what came back.
    """
    reference = np.array([[1e18, -1e18], [1.0, -1.0]])
    candidate = reference.copy()
    candidate[1, 0] = 2.0  # a 100% error on the small case

    result = compare(
        candidate, reference, name="ref", tolerance=1e-5, labels=("huge", "small")
    )
    assert result.mismatches == 1
    assert result.mismatched_cases == 1
    assert result.worst_case == "small"
    assert not result.passed


def test_an_error_scaled_to_its_own_case_is_tolerated() -> None:
    """Drift proportional to the case's own magnitude is accumulation, not a bug."""
    reference = np.array([[1e18, -1e18], [1.0, -1.0]])
    candidate = reference * (1.0 + 1e-7)

    result = compare(
        candidate, reference, name="ref", tolerance=1e-5, labels=("huge", "small")
    )
    assert result.mismatches == 0
    assert result.passed


def test_identical_results_are_reported_as_bit_exact() -> None:
    reference = np.array([[1.0, 2.0], [3.0, 4.0]])
    result = compare(
        reference.copy(), reference, name="ref", tolerance=0.0, labels=("a", "b")
    )
    assert result.bit_exact
    assert result.max_abs_error == 0.0
    assert result.min_cosine_similarity == pytest.approx(1.0)
    assert result.passed


def test_a_zero_reference_case_still_catches_a_nonzero_result() -> None:
    """The all-zero activation case would otherwise divide by its own magnitude."""
    reference = np.zeros((1, 4))
    candidate = np.zeros((1, 4))
    candidate[0, 2] = 1e-3

    result = compare(candidate, reference, name="ref", tolerance=1e-5, labels=("zeros",))
    assert result.mismatches == 1
    assert not result.passed


def test_relocating_a_nan_keeps_the_count_and_fails_anyway() -> None:
    """Counting non-finite entries per side is not enough; positions must agree."""
    reference = np.array([[np.nan, 1.0, 2.0]])
    candidate = np.array([[1.0, np.nan, 2.0]])

    result = compare(candidate, reference, name="ref", tolerance=1e-5, labels=("moved",))
    assert result.candidate_non_finite == result.reference_non_finite == 1
    assert result.non_finite_pattern_mismatches == 2
    assert not result.passed


def test_a_nan_where_an_infinity_belongs_fails() -> None:
    reference = np.array([[np.inf, 0.0]])
    candidate = np.array([[np.nan, 0.0]])

    result = compare(candidate, reference, name="ref", tolerance=1e-5, labels=("swap",))
    assert result.non_finite_pattern_mismatches == 1
    assert not result.passed


def test_a_sign_flipped_infinity_fails() -> None:
    reference = np.array([[np.inf, 0.0]])
    candidate = np.array([[-np.inf, 0.0]])

    result = compare(candidate, reference, name="ref", tolerance=1e-5, labels=("sign",))
    assert result.non_finite_pattern_mismatches == 1
    assert not result.passed


def test_a_case_that_is_entirely_non_finite_is_measurable() -> None:
    """The all-NaN activation produces an all-NaN row with no finite entry left."""
    reference = np.full((1, 8), np.nan)
    result = compare(
        reference.copy(), reference, name="ref", tolerance=1e-5, labels=("all_nan",)
    )
    assert result.candidate_non_finite == 8
    assert result.non_finite_pattern_mismatches == 0
    assert result.max_abs_error == 0.0
    # Nothing finite was measured, so nothing was proven exact either.
    assert not result.bit_exact
    assert result.passed


def test_a_kernel_that_swallows_a_nan_fails() -> None:
    """Returning zeros for a NaN input would hide a broken upstream layer."""
    reference = np.full((1, 4), np.nan)
    candidate = np.zeros((1, 4))

    result = compare(candidate, reference, name="ref", tolerance=1e-5, labels=("eaten",))
    assert result.non_finite_pattern_mismatches == 4
    assert not result.passed


def test_comparisons_require_a_label_per_case() -> None:
    reference = np.zeros((2, 3))
    with pytest.raises(FormatError, match="labels"):
        compare(reference.copy(), reference, name="ref", tolerance=0.0, labels=("only",))


def test_comparisons_reject_a_flat_array() -> None:
    """A 1-D result would be normalised as a single case and lose the whole point."""
    reference = np.zeros(4)
    with pytest.raises(FormatError, match="2-D"):
        compare(reference.copy(), reference, name="ref", tolerance=0.0, labels=("flat",))


def test_comparisons_reject_a_shape_mismatch() -> None:
    with pytest.raises(FormatError, match="cannot compare"):
        compare(
            np.zeros((2, 3)), np.zeros((2, 4)), name="ref", tolerance=0.0,
            labels=("a", "b"),
        )


def test_only_the_same_order_pairing_gets_the_tight_bound() -> None:
    """The GEMM path folds the scale into the weight; it never claimed ULP parity."""
    assert pair_tolerance("lut23_batched", "lut23") == SAME_ORDER_RELATIVE_TOLERANCE
    assert pair_tolerance("lut23_single", "lut23") == SAME_ORDER_RELATIVE_TOLERANCE
    assert pair_tolerance("gemm_128x128", "lut23") == ORACLE_RELATIVE_TOLERANCE
    assert pair_tolerance("lut23_batched", "oracle") == ORACLE_RELATIVE_TOLERANCE
    assert pair_tolerance("lut23_batched", "restored") == ORACLE_RELATIVE_TOLERANCE


@pytest.mark.parametrize(
    ("rows", "columns", "expected"),
    [
        (48, 5120, 48),          # one short tile, taken whole
        (1024, 5120, 1024),      # exactly the budget
        (5120, 5120, 1024),      # four 256-row tiles out of twenty
        (248320, 5120, 1024),    # four out of nine hundred and seventy
    ],
)
def test_the_reference_prefix_is_always_a_whole_number_of_tiles(
    rows: int, columns: int, expected: int
) -> None:
    layout = PackedTensorLayout.for_tensor(rows, columns)
    prefix = reference_prefix(layout)
    assert prefix.rows == expected
    assert prefix.rows <= max(REFERENCE_ROW_BUDGET, layout.tile)
    assert prefix.rows % prefix.tile == 0
    assert prefix.tile == layout.tile
    assert prefix.tiles <= layout.tiles


def test_the_activation_set_carries_one_label_per_row() -> None:
    columns = 5120
    x, labels = activation_cases(columns, seed=1)
    assert x.shape == (len(labels), columns)
    assert x.dtype == np.float32
    assert sum(label.startswith("random_") for label in labels) == RANDOM_CASES
    # Named edges the plan asks for, each one exercising a different structure.
    assert {"zeros", "single_hot_tail", "large_alternating", "small_normal"} <= set(labels)
    assert len(set(labels)) == len(labels)


def test_every_activation_case_is_finite() -> None:
    """Non-finite inputs are a separate, separately-gated experiment."""
    x, _ = activation_cases(5120, seed=2)
    assert np.isfinite(x).all()


def test_the_non_finite_set_puts_a_nan_in_the_tail_slot() -> None:
    """Column 125 is the first the tail byte covers and no full slot touches."""
    x, labels = non_finite_cases(5120)
    assert x.shape == (len(labels), 5120)
    assert np.isnan(x[labels.index("single_nan_tail"), 125])
    assert np.isfinite(x[labels.index("single_nan_tail"), 124])
    assert np.isneginf(x[labels.index("single_inf_last"), 5119])
