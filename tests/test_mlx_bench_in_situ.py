"""Gates on what the in-situ sweep agrees to measure, and on what it recorded.

The sweep itself needs a GPU and two 27B models, so what is gated here is
everything decided before either is touched: which tilings a shape is offered,
which of them the shipped rule is, and whether the run's own arguments make
sense. A candidate set that omits the shipped tiling is the failure that matters
most -- the run would still produce a file, and every ranking in it would be
against a tiling the model does not use.

The recorded run is replayed too, when there is one. ``select_gemm`` is the
thing this sweep exists to fit, so a change to it re-dates every number in the
file; asserting that the recorded shipped choices are still today's choices is
what makes that visible instead of silent.
"""

from __future__ import annotations

import json
from pathlib import Path

import mlx.core as mx
import numpy as np
import pytest

from bonsai_tq1.format import BLOCK_SIZE, FormatError, encode_tq1_blocks
from ternel_mlx.bench_in_situ import (
    AFFINE,
    BUCKET,
    KEEP_ONE,
    LADDER,
    LUT23,
    SCHEMA_VERSION,
    WHOLE_PASS,
    block_name,
    bucket_block,
    bucket_members,
    build,
    decision_candidates,
    dispatch_threadgroups,
    fits,
    ladder_candidates,
    main,
    relevant,
    rotate,
    rung,
    shipped_config,
    shipped_for_layout,
)
from ternel_mlx.kernels import GemmConfig, MatmulConfig, tq1_gemm
from ternel_mlx.layout import MODEL_LINEAR_SHAPES, PackedTensorLayout
from ternel_mlx.modules import (
    GEMM_LADDER_LARGE_BATCH,
    GEMM_LADDER_SMALL_BATCH,
    GEMM_LARGE_MIN_BATCH,
    GEMM_MIN_BATCH,
    select_gemm,
    select_lut23,
    threadgroups,
)
from ternel_mlx.packing import pack_blocks
from ternel_mlx.reference import oracle_matmul

# The widths the sweep is run at: the four the rule was fitted over, the three
# the model actually prefills at that are cheap enough to sweep, and batch 8 for
# the bucket question below ``GEMM_MIN_BATCH``.
SWEPT_BATCHES = (8, 16, 31, 32, 64, 128)

MEASUREMENTS = Path("results/mlx")
KEEP_ONE_RUN = MEASUREMENTS / "b9_in_situ_keep_one.json"
WHOLE_PASS_RUN = MEASUREMENTS / "b9_in_situ_whole_pass.json"
BUCKET_RUN = MEASUREMENTS / "b9_in_situ_bucket.json"

# The per-shape files. The bucket file records arms over *sets* of shapes, so
# the gates keyed on a per-shape census do not apply to it and it is listed
# separately rather than parametrized in beside them.
PER_SHAPE_RUNS = (KEEP_ONE_RUN, WHOLE_PASS_RUN)
ALL_RUNS = (*PER_SHAPE_RUNS, BUCKET_RUN)


# --------------------------------------------------------------------------
# The ladder
# --------------------------------------------------------------------------


@pytest.mark.parametrize(("name", "spec"), LADDER, ids=[name for name, _ in LADDER])
def test_every_rung_constructs_and_is_named_after_its_block(
    name: str, spec: tuple[int, int, int, int, int, bool, bool, bool]
) -> None:
    """A rung whose name disagrees with its tiling mislabels every reading."""
    config = build(spec)
    assert name == f"gemm_{config.batch_block}x{config.row_block}"
    assert config.benchmark_name.startswith(name)
    assert not config.safe_clamp, "the clamped variant is for fuzzing, not for timing"


@pytest.mark.parametrize("ladder", (GEMM_LADDER_SMALL_BATCH, GEMM_LADDER_LARGE_BATCH))
def test_the_shipped_rungs_are_all_in_the_swept_ladder(
    ladder: tuple[GemmConfig, ...]
) -> None:
    """The sweep has to measure what ships, or it cannot score the rule.

    Compared by value across every field, which is the only comparison that
    catches the failure this guards: two tilings can agree on block shape and
    disagree on which of the six kernel forms they run, and those are different
    kernels with different rankings.
    """
    swept = {build(spec) for _, spec in LADDER}
    for config in ladder:
        assert config in swept, f"{config.benchmark_name} ships but is never measured"


def test_the_ladder_has_no_duplicate_rungs() -> None:
    """A repeated tiling would take two slots in every round and rotate wrong."""
    names = [name for name, _ in LADDER]
    assert len(set(names)) == len(names)
    configs = [build(spec) for _, spec in LADDER]
    assert len(set(configs)) == len(configs)


@pytest.mark.parametrize("batch", (5, 16))
@pytest.mark.parametrize(("name", "spec"), LADDER, ids=[name for name, _ in LADDER])
def test_every_rung_computes_the_right_answer(
    name: str, spec: tuple[int, int, int, int, int, bool, bool, bool], batch: int
) -> None:
    """A ranking over kernels that compute the wrong product ranks nothing.

    ``tests/test_mlx_kernels.py`` crosses all six kernel forms against three
    tilings, and every rung the shipped rule can dispatch is gated again by
    ``kernel_gate``. Neither covers the three eight-deep rungs, which exist only
    here: they are the counterfactual a bucket below
    :data:`~ternel_mlx.modules.GEMM_MIN_BATCH` would dispatch, so nothing that
    ships has a reason to instantiate them. Both batches leave the eight-deep
    rungs' final batch tile part empty, which is where the guarded stores are.
    """
    rows, columns = 256, 640
    layout = PackedTensorLayout.for_tensor(rows, columns)
    generator = np.random.default_rng(spec[0] * 1000 + spec[1])
    scale_bits = (
        generator.normal(0.0, 0.05, size=layout.groups)
        .astype(np.float16).view(np.uint8).reshape(-1, 2)
    )
    trits = generator.integers(0, 3, size=(layout.groups, BLOCK_SIZE), dtype=np.uint8)
    blocks = encode_tq1_blocks(scale_bits, trits).reshape(rows, layout.groups_per_row, -1)
    codes, scales = pack_blocks(blocks, layout=layout)
    x = generator.normal(0.0, 1.0, size=(batch, columns)).astype(np.float32)

    out = tq1_gemm(
        mx.array(codes), mx.array(scales), mx.array(x),
        groups_per_row=layout.groups_per_row, tile=layout.tile, config=build(spec),
    )
    mx.eval(out)
    oracle = oracle_matmul(codes, scales, x, layout=layout)
    got = np.array(out, copy=False).astype(np.float64)
    magnitude = max(float(np.abs(oracle).max()), 1e-30)
    error = float(np.abs(got - oracle).max()) / magnitude
    assert error < 1e-5, f"{name} at batch {batch} is off the oracle by {error:.2e}"


# --------------------------------------------------------------------------
# Which candidates a shape is offered
# --------------------------------------------------------------------------


@pytest.mark.parametrize("batch", SWEPT_BATCHES)
@pytest.mark.parametrize("batch_block", tuple(sorted({spec[0] for _, spec in LADDER})))
def test_relevance_admits_one_step_past_the_batch(batch: int, batch_block: int) -> None:
    """The boundary is where the ladder's own threshold sits, so it stays in view."""
    assert relevant(batch, batch_block) == (batch_block <= 2 * batch)


@pytest.mark.parametrize("batch", SWEPT_BATCHES)
def test_the_shipped_choice_is_always_a_candidate(batch: int) -> None:
    """Otherwise every ratio in the file is against a tiling nothing dispatches."""
    for shape in MODEL_LINEAR_SHAPES:
        incumbent = shipped_config(shape, batch)
        for offered in (ladder_candidates(shape, batch), decision_candidates(shape, batch)):
            assert any(config == incumbent for _, config in offered), (
                f"{shape.label} at batch {batch} ships {incumbent.benchmark_name}, "
                f"which is not among {[name for name, _ in offered]}"
            )


@pytest.mark.parametrize("batch", SWEPT_BATCHES)
def test_lut23_is_the_last_candidate_and_appears_once(batch: int) -> None:
    """The runner appends the affine control after it, so its position is fixed."""
    for shape in MODEL_LINEAR_SHAPES:
        for offered in (ladder_candidates(shape, batch), decision_candidates(shape, batch)):
            names = [name for name, _ in offered]
            assert names.count(LUT23) == 1
            assert names[-1] == LUT23
            assert AFFINE not in names, "the affine control is the runner's, not the set's"


@pytest.mark.parametrize("batch", SWEPT_BATCHES)
def test_lut23_is_offered_exactly_what_the_shipped_path_would_run(batch: int) -> None:
    """The LUT23 arm is the shipped configuration, not a re-derivation of it."""
    for shape in MODEL_LINEAR_SHAPES:
        for offered in (ladder_candidates(shape, batch), decision_candidates(shape, batch)):
            config = dict(offered)[LUT23]
            assert config == select_lut23(shape.layout(), batch)


@pytest.mark.parametrize("batch", SWEPT_BATCHES)
def test_the_decision_set_is_the_rule_s_own_alternatives(batch: int) -> None:
    """Whole-pass measures what the rule chooses between, and nothing wider.

    A handful of entries, which is what makes a mode costing a whole forward
    pass per reading affordable at all. Every rung of the batch's shipped ladder
    that can legally run the shape is in, in the ladder's own order, because a
    rung the rule could pick and this mode never measures is a rung the two
    modes cannot disagree about.
    """
    ladder = (
        GEMM_LADDER_LARGE_BATCH if batch >= GEMM_LARGE_MIN_BATCH
        else GEMM_LADDER_SMALL_BATCH
    )
    for shape in MODEL_LINEAR_SHAPES:
        offered = decision_candidates(shape, batch)
        assert len(offered) <= len(LADDER) + 1
        gemm = [config for _, config in offered if isinstance(config, GemmConfig)]
        legal = [config for config in ladder if fits(shape, config)]
        assert gemm[len(gemm) - len(legal):] == legal


@pytest.mark.parametrize("batch", SWEPT_BATCHES)
def test_only_a_batch_below_the_floor_is_offered_a_bucket(batch: int) -> None:
    """Above the floor the rule already has a bucket, so there is nothing to test.

    Below it every shape goes to LUT23 and the counterfactual -- what a GEMM
    would have cost -- appears on no shipped ladder, so the sweep's own rungs at
    that depth go in front. Running a 16-deep rung at batch 8 is not that
    counterfactual: it wastes half its lanes and buys no threadgroups.
    """
    block = bucket_block(batch)
    assert (block is None) == (batch >= GEMM_MIN_BATCH)
    if block is None:
        return
    assert block <= batch
    for shape in MODEL_LINEAR_SHAPES:
        offered = decision_candidates(shape, batch)
        gemm = [config for _, config in offered if isinstance(config, GemmConfig)]
        bucket = [config for config in gemm if config.batch_block == block]
        assert bucket, f"{shape.label} at batch {batch} is offered no {block}-deep rung"
        assert gemm[: len(bucket)] == bucket, "the bucket rungs come first"
        assert bucket == [
            build(spec) for _, spec in LADDER
            if spec[0] == block and fits(shape, build(spec))
        ]


@pytest.mark.parametrize("batch", SWEPT_BATCHES)
def test_a_candidate_set_never_records_two_arms_under_one_name(batch: int) -> None:
    """The name is the key both modes are joined on, so a collision loses a reading."""
    for shape in MODEL_LINEAR_SHAPES:
        for offered in (ladder_candidates(shape, batch), decision_candidates(shape, batch)):
            names = [name for name, _ in offered]
            assert len(set(names)) == len(names)


@pytest.mark.parametrize("batch", SWEPT_BATCHES)
def test_the_two_modes_name_a_shared_tiling_identically(batch: int) -> None:
    """Reading whole-pass against keep-one is a join on this name and nothing else."""
    for shape in MODEL_LINEAR_SHAPES:
        swept = dict(ladder_candidates(shape, batch))
        for name, config in decision_candidates(shape, batch):
            if name in swept:
                assert swept[name] == config, (
                    f"{shape.label} at batch {batch} calls two different tilings {name}"
                )


@pytest.mark.parametrize("batch", SWEPT_BATCHES)
def test_no_candidate_straddles_two_row_tiles(batch: int) -> None:
    """A row block over a tile boundary reads the next tile through this one's base.

    ``select_gemm`` refuses that pairing; a sweep that measured it anyway would
    be timing a kernel that reads out of bounds.
    """
    for shape in MODEL_LINEAR_SHAPES:
        layout = shape.layout()
        for offered in (ladder_candidates(shape, batch), decision_candidates(shape, batch)):
            for name, config in offered:
                if isinstance(config, GemmConfig):
                    assert layout.tiles == 1 or layout.tile % config.row_block == 0, (
                        f"{name} straddles {shape.label}'s {layout.tile}-row tiles"
                    )


def test_fits_agrees_with_the_dispatch_rule_s_own_refusal() -> None:
    """The sweep's legality test and the rule's are one test, not two."""
    for shape in MODEL_LINEAR_SHAPES:
        for ladder in (GEMM_LADDER_SMALL_BATCH, GEMM_LADDER_LARGE_BATCH):
            for config in ladder:
                if not fits(shape, config):
                    layout = shape.layout()
                    assert select_gemm(layout, config.batch_block) != config


# --------------------------------------------------------------------------
# What is recorded beside each reading
# --------------------------------------------------------------------------


@pytest.mark.parametrize("batch", SWEPT_BATCHES)
def test_gemm_threadgroups_are_counted_by_the_shipped_function(batch: int) -> None:
    """The occupancy column is the quantity the rule decides on, not a copy of it."""
    for shape in MODEL_LINEAR_SHAPES:
        for _, config in ladder_candidates(shape, batch):
            if isinstance(config, GemmConfig):
                assert dispatch_threadgroups(shape, batch, config) == threadgroups(
                    shape.layout(), batch, config
                )


@pytest.mark.parametrize("batch", SWEPT_BATCHES)
def test_lut23_threadgroups_include_its_split(batch: int) -> None:
    """LUT23's parallelism comes from the K axis, so a count that drops it lies.

    This is the whole reason the two families can be compared: a 1024-row tensor
    offers a GEMM eight threadgroups and offers LUT23 six hundred and forty.
    """
    for shape in MODEL_LINEAR_SHAPES:
        layout = shape.layout()
        config = select_lut23(layout, batch)
        assert isinstance(config, MatmulConfig)
        counted = dispatch_threadgroups(shape, batch, config)
        assert counted % config.k_split == 0
        assert counted >= layout.tiles * config.k_split


def test_threadgroups_are_refused_for_anything_that_is_not_a_kernel_config() -> None:
    with pytest.raises(FormatError, match="cannot count threadgroups"):
        dispatch_threadgroups(MODEL_LINEAR_SHAPES[0], 16, None)


@pytest.mark.parametrize("batch", SWEPT_BATCHES)
def test_the_shipped_choice_is_never_none(batch: int) -> None:
    """``select_gemm`` returning None means LUT23, which is an answer, not a gap."""
    for shape in MODEL_LINEAR_SHAPES:
        layout = shape.layout()
        config = shipped_for_layout(layout, batch)
        assert config is not None
        expected = select_gemm(layout, batch)
        assert config == (select_lut23(layout, batch) if expected is None else expected)


# --------------------------------------------------------------------------
# The round's rotation
# --------------------------------------------------------------------------


@pytest.mark.parametrize("size", (1, 2, 3, 7, 15))
def test_rotation_is_a_permutation_that_moves_every_item_to_the_front(size: int) -> None:
    """A fixed order parks drift on whichever candidates sit late in a round."""
    work = tuple(range(size))
    first = []
    for index in range(size):
        rotated = rotate(work, index)
        assert sorted(rotated) == list(work)
        first.append(rotated[0])
    assert sorted(first) == list(work)


def test_rotation_wraps_past_the_work_list() -> None:
    """Rounds outnumber candidates, so the index has to keep meaning something."""
    work = ("a", "b", "c")
    assert rotate(work, 4) == rotate(work, 1)


# --------------------------------------------------------------------------
# The bucket the rule would ship
# --------------------------------------------------------------------------


@pytest.mark.parametrize(("name", "spec"), LADDER, ids=[name for name, _ in LADDER])
def test_every_rung_resolves_from_the_name_it_is_recorded_under(
    name: str, spec: tuple[int, int, int, int, int, bool, bool, bool]
) -> None:
    """A bucket run names its tiling; the name has to reach the same rung.

    The bucket file and the whole-pass file it is read against are joined on this
    string. Resolving it to a different tiling than the one whole-pass measured
    would compare two arms and call the difference additivity.
    """
    assert rung(name) == build(spec)
    assert block_name(rung(name)) == name


def test_a_rung_that_is_not_on_the_ladder_is_refused() -> None:
    """Silently building an unswept tiling would measure something ungated."""
    with pytest.raises(FormatError, match="no ladder rung"):
        rung("gemm_8x64")


@pytest.mark.parametrize("batch", SWEPT_BATCHES)
def test_bucket_membership_is_monotone_in_the_threshold(batch: int) -> None:
    """Raising the occupancy bar can only drop shapes, never add them.

    The threshold is the rule's one free parameter, so a membership that is not
    monotone in it would mean the fitted value cannot be read as "this much
    occupancy or better".
    """
    config = rung("gemm_8x32")
    previous = set(bucket_members(batch, config, 1))
    for bar in (32, 160, 320, 544, 8000):
        members = set(bucket_members(batch, config, bar))
        assert members <= previous, f"raising the bar to {bar} added a shape"
        previous = members


@pytest.mark.parametrize("batch", SWEPT_BATCHES)
def test_every_bucket_member_reaches_the_bar_on_a_tiling_it_can_run(batch: int) -> None:
    """Membership is on threadgroups and legality, and both are load-bearing.

    A shape that cannot legally run the tiling has no threadgroup count on it to
    compare, and dispatching it anyway would read the second row tile's codes
    through the first tile's pointer.
    """
    config = rung("gemm_8x32")
    bar = 320
    members = bucket_members(batch, config, bar)
    for shape in members:
        assert fits(shape, config), f"{shape.label} cannot run the bucket's tiling"
        assert dispatch_threadgroups(shape, batch, config) >= bar
    excluded = set(MODEL_LINEAR_SHAPES) - set(members)
    for shape in excluded:
        assert (
            not fits(shape, config)
            or dispatch_threadgroups(shape, batch, config) < bar
        ), f"{shape.label} qualifies but was left out"


def test_the_bucket_keeps_its_null_control_when_every_shape_moves() -> None:
    """``lm_head`` is in the combined arm, and its true contribution is zero.

    A prefill chunk never issues it, so a bucket that moves it moves 248320 rows
    of nothing. That is the property that lets the combined arm be read at all:
    without a term of known value inside it, a combined reading has no scale.
    """
    config = rung("gemm_8x32")
    members = bucket_members(8, config, 320)
    labels = [shape.label for shape in members]
    assert "lm_head" in labels
    assert dispatch_threadgroups(
        next(shape for shape in members if shape.label == "lm_head"), 8, config
    ) > 320


# --------------------------------------------------------------------------
# The run's own arguments
# --------------------------------------------------------------------------


def arguments(**overrides: object) -> list[str]:
    """A complete argument list, so each test changes exactly one thing."""
    base: dict[str, object] = {
        "--mode": KEEP_ONE,
        "--artifact": "artifacts/ternel-mlx",
        "--baseline": "artifacts/baseline-mlx-2bit",
        "--output": "results/mlx/unused.json",
        "--batches": "16",
        "--rounds": "5",
        "--inner": "3",
        "--warmup": "2",
        "--max-foreign-gpu-share": "0.25",
    }
    base.update(overrides)
    argv: list[str] = []
    for flag, value in base.items():
        if value is None:
            continue
        argv.extend([flag, str(value)])
    return argv


def test_keep_one_without_a_baseline_is_refused() -> None:
    """It is the denominator of every ratio the mode reports."""
    with pytest.raises(SystemExit):
        main(arguments(**{"--baseline": None}))


@pytest.mark.parametrize("mode", (WHOLE_PASS, BUCKET))
def test_a_baseline_outside_keep_one_is_refused(mode: str) -> None:
    """There is no affine arm, so a baseline would be loaded and never used."""
    with pytest.raises(SystemExit):
        main(arguments(**{"--mode": mode}))


@pytest.mark.parametrize("missing", ("--bucket-arm", "--bucket-min-threadgroups"))
def test_a_bucket_run_missing_half_its_rule_is_refused(missing: str) -> None:
    """Both halves name the rule; either alone describes no bucket at all."""
    argv = {
        "--mode": BUCKET, "--baseline": None,
        "--bucket-arm": "gemm_8x32", "--bucket-min-threadgroups": "320",
        missing: None,
    }
    with pytest.raises(SystemExit):
        main(arguments(**argv))


@pytest.mark.parametrize("mode", (KEEP_ONE, WHOLE_PASS))
def test_bucket_arguments_outside_bucket_mode_are_refused(mode: str) -> None:
    """Accepting them silently would let a run claim a rule it never applied."""
    argv: dict[str, object] = {"--mode": mode, "--bucket-arm": "gemm_8x32"}
    if mode != KEEP_ONE:
        argv["--baseline"] = None
    with pytest.raises(SystemExit):
        main(arguments(**argv))


def test_a_non_positive_occupancy_threshold_is_refused() -> None:
    """A bar of zero admits every shape including the ones the tiling starves."""
    with pytest.raises(FormatError, match="occupancy threshold must be positive"):
        main(arguments(**{
            "--mode": BUCKET, "--baseline": None,
            "--bucket-arm": "gemm_8x32", "--bucket-min-threadgroups": "0",
        }))


@pytest.mark.parametrize("flag", ("--rounds", "--inner", "--warmup"))
def test_a_run_with_nothing_to_measure_is_refused(flag: str) -> None:
    with pytest.raises(FormatError, match="at least one"):
        main(arguments(**{flag: "0"}))


def test_a_non_positive_batch_is_refused() -> None:
    with pytest.raises(FormatError, match="batches must be positive"):
        main(arguments(**{"--batches": "0"}))


def test_a_non_positive_contention_bound_is_refused() -> None:
    """A bound of zero would void every run; a negative one is meaningless."""
    with pytest.raises(FormatError, match="max foreign share must be positive"):
        main(arguments(**{"--max-foreign-gpu-share": "0"}))


def test_an_unknown_mode_is_refused() -> None:
    with pytest.raises(SystemExit):
        main(arguments(**{"--mode": "stub-one-out"}))


# --------------------------------------------------------------------------
# Replaying what was recorded
# --------------------------------------------------------------------------


def recorded(path: Path) -> dict[str, object]:
    if not path.exists():
        pytest.skip(f"{path} has not been produced yet")
    return json.loads(path.read_text())


@pytest.mark.parametrize("path", PER_SHAPE_RUNS)
def test_the_recorded_run_still_describes_today_s_rule(path: Path) -> None:
    """A rule change re-dates every ranking in the file, and has to say so.

    The file names the tiling it believed each shape shipped at each batch. If
    ``select_gemm`` now chooses differently, the file's "shipped" column is a
    label from a build that no longer exists, and every ``shipped_over_best``
    computed from it is a comparison against the wrong thing.
    """
    document = recorded(path)
    by_label = {shape.label: shape for shape in MODEL_LINEAR_SHAPES}
    checked = 0
    for batch_text, measured in document["measured"].items():
        batch = int(batch_text)
        for label, entry in measured["shapes"].items():
            offered = entry["candidates"]
            marked = [name for name, row in offered.items() if row["shipped"]]
            assert len(marked) == 1, f"{label} at batch {batch} marks {marked} as shipped"
            assert entry["shipped"] == marked[0]
            expected = shipped_config(by_label[label], batch)
            assert offered[marked[0]]["config"] == expected.benchmark_name, (
                f"{label} at batch {batch} was measured against "
                f"{offered[marked[0]]['config']}, the rule now dispatches "
                f"{expected.benchmark_name}"
            )
            checked += 1
    assert checked == len(MODEL_LINEAR_SHAPES) * len(document["batches"])


@pytest.mark.parametrize("path", PER_SHAPE_RUNS)
def test_the_recorded_run_measured_every_shape_at_every_batch(path: Path) -> None:
    """A shape that raised mid-run would otherwise leave a quiet hole."""
    document = recorded(path)
    labels = {shape.label for shape in MODEL_LINEAR_SHAPES}
    assert set(document["measured"]) == {str(batch) for batch in document["batches"]}
    for measured in document["measured"].values():
        assert set(measured["shapes"]) == labels
        for entry in measured["shapes"].values():
            assert entry["candidates"], "a shape with no candidates measured nothing"
            for row in entry["candidates"].values():
                assert len(row["samples"]) == document["rounds"]


@pytest.mark.parametrize("path", PER_SHAPE_RUNS)
def test_the_recorded_run_is_written_at_today_s_schema(path: Path) -> None:
    """An older file is missing columns the current module reports, silently.

    Both modes record differences. A file whose candidates carry no reference
    pass records them with nothing to read them against, and eight milliseconds
    is a result or a rounding error depending on which.
    """
    document = recorded(path)
    assert document["schema_version"] == SCHEMA_VERSION


@pytest.mark.parametrize("path", PER_SHAPE_RUNS)
def test_every_delta_is_recorded_beside_the_pass_it_came_from(path: Path) -> None:
    """One reference per candidate, paired round for round with its own deltas."""
    document = recorded(path)
    for measured in document["measured"].values():
        for entry in measured["shapes"].values():
            for name, row in entry["candidates"].items():
                assert len(row["reference_samples"]) == document["rounds"]
                assert min(row["reference_samples"]) > 0.0, f"{name} timed a pass at zero"
                assert row["seconds"] + row["reference_seconds"] > 0.0, (
                    f"{name} costs less than nothing"
                )


@pytest.mark.parametrize("path", PER_SHAPE_RUNS)
def test_the_recorded_run_had_the_gpu_to_itself(path: Path) -> None:
    """Every timed block is gated, so the file carries its own proof."""
    document = recorded(path)
    bound = document["max_foreign_gpu_share"]
    assert document["gpu_contention"], "no contention was sampled around the timings"
    for sample in document["gpu_contention"]:
        assert float(sample["total_share"]) <= bound, (
            f"{sample['label']} was timed with {float(sample['total_share']):.1%} "
            f"of the GPU elsewhere"
        )


def test_the_keep_one_run_evaluated_the_logits() -> None:
    """Every one of a shape's dispatches has to execute for its cost to be its cost."""
    document = recorded(KEEP_ONE_RUN)
    assert document["mode"] == KEEP_ONE
    assert document["evaluated"] == "logits"
    for measured in document["measured"].values():
        for entry in measured["shapes"].values():
            assert entry["affine_seconds"] > 0.0
            assert len(entry["affine_samples"]) == document["rounds"]


@pytest.mark.parametrize("path", ALL_RUNS)
def test_the_recorded_run_is_written_at_today_s_schema_whatever_its_mode(path: Path) -> None:
    """Applies to the bucket file too, whose per-arm rows are the same shape."""
    assert recorded(path)["schema_version"] == SCHEMA_VERSION


@pytest.mark.parametrize("path", ALL_RUNS)
def test_every_recorded_run_had_the_gpu_to_itself(path: Path) -> None:
    """Every timed block is gated, so each file carries its own proof."""
    document = recorded(path)
    bound = document["max_foreign_gpu_share"]
    assert document["gpu_contention"], "no contention was sampled around the timings"
    for sample in document["gpu_contention"]:
        assert float(sample["total_share"]) <= bound, (
            f"{sample['label']} was timed with {float(sample['total_share']):.1%} "
            f"of the GPU elsewhere"
        )


def test_the_bucket_run_moved_the_rule_it_names() -> None:
    """The recorded membership has to be what today's rule would select.

    The file names an arm and a threshold; those two are the whole rule, and
    recomputing the membership from them is what stops the file from being read
    as evidence for a bucket other than the one it measured.
    """
    document = recorded(BUCKET_RUN)
    assert document["mode"] == BUCKET
    assert document["evaluated"] == "cache"
    config = rung(document["bucket_arm"])
    bar = document["bucket_min_threadgroups"]
    assert set(document["measured"]) == {str(batch) for batch in document["batches"]}
    for batch_text, entry in document["measured"].items():
        expected = bucket_members(int(batch_text), config, bar)
        assert entry["members"] == [shape.label for shape in expected], (
            f"batch {batch_text} recorded {entry['members']}, the rule now selects "
            f"{[shape.label for shape in expected]}"
        )
        assert entry["block"] == document["bucket_arm"]
        assert entry["arm"] == config.benchmark_name


def test_the_bucket_run_measured_every_member_alone_and_together() -> None:
    """A missing part turns the additivity ratio into a comparison with a hole."""
    document = recorded(BUCKET_RUN)
    for batch_text, entry in document["measured"].items():
        assert set(entry["arms"]) == {BUCKET, *entry["members"]}, batch_text
        for name, row in entry["arms"].items():
            assert len(row["samples"]) == document["rounds"], name
            assert len(row["reference_samples"]) == document["rounds"], name
            assert min(row["reference_samples"]) > 0.0, f"{name} timed a pass at zero"


def test_the_bucket_run_s_parts_sum_to_what_it_reports() -> None:
    """``additivity`` is the whole point of the mode, so its denominator is gated."""
    document = recorded(BUCKET_RUN)
    for batch_text, entry in document["measured"].items():
        parts = sum(
            sorted(entry["arms"][label]["samples"])[len(entry["arms"][label]["samples"]) // 2]
            for label in entry["members"]
        )
        assert entry["sum_of_parts_seconds"] == pytest.approx(parts, abs=1e-9), batch_text
        combined = entry["arms"][BUCKET]["seconds"]
        assert entry["additivity"] == pytest.approx(combined / parts, rel=1e-9), batch_text


def test_the_bucket_run_kept_its_null_control() -> None:
    """``lm_head`` is inside the combined arm and its true contribution is zero.

    Bounded against its own reference pass rather than against the other arms,
    so the gate is on the measurement and not on the result: a bucket that turns
    out not to help is a finding, a clock that reads 5% of a pass off a matmul
    the prefill never issues is a broken measurement.
    """
    document = recorded(BUCKET_RUN)
    for batch_text, entry in document["measured"].items():
        assert "lm_head" in entry["members"], batch_text
        null = entry["arms"]["lm_head"]
        share = abs(null["seconds"]) / null["reference_seconds"]
        assert share < 0.02, (
            f"batch {batch_text}: retiling lm_head moved a prefill that never "
            f"issues it by {share:.1%} of the pass"
        )


def test_the_whole_pass_run_evaluated_the_cache() -> None:
    """It is the prefill mlx-lm runs; timing the logits adds a matmul it skips."""
    document = recorded(WHOLE_PASS_RUN)
    assert document["mode"] == WHOLE_PASS
    assert document["evaluated"] == "cache"
    for measured in document["measured"].values():
        for entry in measured["shapes"].values():
            assert "affine_seconds" not in entry
            assert entry["candidates"][entry["shipped"]]["shipped"] is True
