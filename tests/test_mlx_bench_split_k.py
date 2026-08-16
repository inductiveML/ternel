"""Gates on which configurations the split sweep agrees to measure.

The sweep itself needs a GPU, so what is gated here is the decision that runs
before it: ``legal_splits`` chooses the arms, and a shape it declines to return
anything for is a shape the run never measures.

That decision has one edge, and the first prefill run fell straight into it.
``MAX_THREADGROUPS`` bounds how far a split may push the grid past the machine's
forty cores, and at decode batches every shape sits far below it -- the widest,
lm_head at batch 8, presents 1940 threadgroups unsplit. At prefill batches it
does not: lm_head at batch 16 is 970 tiles times four batch tiles, 3880
threadgroups, before any split at all. The sweep then had no arms to build and
raised, taking the whole run down at its first shape.

"No split is a candidate here" is a result, and the same one the dispatch rule
reaches at that batch. The control has to survive being told so, which is what
these assert.
"""

from __future__ import annotations

import pytest

from ternel_mlx.bench_split_k import MAX_THREADGROUPS, divisors, legal_splits
from ternel_mlx.layout import MODEL_LINEAR_SHAPES, PackedTensorLayout

# The three group counts the model actually presents; layout.groups_per_row is
# always one of these, so a split rule is fully exercised by covering them.
GROUP_COUNTS = (40, 48, 136)


def layout_with_groups(groups: int) -> PackedTensorLayout:
    """A real model layout whose rows carry ``groups`` groups each.

    Built through ``ModelLinearShape.layout()``, the same call the sweep uses,
    so a change to how a shape maps to a layout reaches this file too rather
    than leaving it asserting against a second, private construction.
    """
    for shape in MODEL_LINEAR_SHAPES:
        layout = shape.layout()
        if layout.groups_per_row == groups:
            return layout
    raise AssertionError(f"no model shape has {groups} groups per row")


@pytest.mark.parametrize("groups", GROUP_COUNTS)
@pytest.mark.parametrize("threadgroups", (1, 40, 970, 3880, MAX_THREADGROUPS * 100))
def test_the_unsplit_control_is_always_measured(groups: int, threadgroups: int) -> None:
    """k=1 is the shipped configuration, not a candidate to be filtered out."""
    assert 1 in legal_splits(layout_with_groups(groups), threadgroups=threadgroups)


@pytest.mark.parametrize("groups", GROUP_COUNTS)
def test_a_saturated_grid_yields_the_control_alone(groups: int) -> None:
    """Past the ceiling unsplit, no split is a candidate -- and it does not raise.

    This is the case that killed the first prefill run.
    """
    layout = layout_with_groups(groups)
    assert legal_splits(layout, threadgroups=MAX_THREADGROUPS + 1) == (1,)


@pytest.mark.parametrize("groups", GROUP_COUNTS)
@pytest.mark.parametrize("threadgroups", (1, 4, 20, 40, 68, 136, 970, 1940))
def test_every_split_returned_is_a_divisor_within_the_ceiling(
    groups: int, threadgroups: int
) -> None:
    """A ragged split strands a threadgroup, so only divisors are ever offered."""
    layout = layout_with_groups(groups)
    for split in legal_splits(layout, threadgroups=threadgroups):
        assert groups % split == 0, f"{split} does not divide {groups}"
        if split != 1:
            assert threadgroups * split <= MAX_THREADGROUPS


# Worked by hand at threadgroups=20 against MAX_THREADGROUPS=2560, so this is an
# oracle rather than a restatement of the filter: every divisor of 40 and of 48
# leaves the grid inside the ceiling, and of 136 only the whole-row split does
# not -- 20 x 136 is 2720.
OFFERED_AT_20_THREADGROUPS = {
    40: (1, 2, 4, 5, 8, 10, 20, 40),
    48: (1, 2, 3, 4, 6, 8, 12, 16, 24, 48),
    136: (1, 2, 4, 8, 17, 34, 68),
}


@pytest.mark.parametrize("groups", GROUP_COUNTS)
def test_no_legal_split_is_silently_dropped(groups: int) -> None:
    """Everything that fits is offered, so the sweep cannot miss the best split."""
    assert MAX_THREADGROUPS == 2560, "the table below was worked against 2560"
    layout = layout_with_groups(groups)
    assert legal_splits(layout, threadgroups=20) == OFFERED_AT_20_THREADGROUPS[groups]


@pytest.mark.parametrize("groups", GROUP_COUNTS)
def test_the_offered_splits_are_exactly_the_divisors_that_fit(groups: int) -> None:
    """The hand-worked table above agrees with what a divisor scan finds."""
    assert OFFERED_AT_20_THREADGROUPS[groups] == tuple(
        k for k in divisors(groups) if k == 1 or 20 * k <= MAX_THREADGROUPS
    )


def test_the_decode_batches_already_measured_are_unaffected() -> None:
    """The committed sweep's widest shape still gets exactly what it got.

    lm_head at batch 8 sits at 1940 threadgroups and 40 groups per row, where
    only the control fits -- 1940 x 2 is already past the ceiling. It recorded a
    single variant before this rule changed and must record a single one after,
    or results/mlx/a6_split_k.json stops describing what the code does.
    """
    assert legal_splits(layout_with_groups(40), threadgroups=1940) == (1,)
