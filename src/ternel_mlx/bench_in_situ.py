"""B9: rank the prefill tilings inside the graph that runs them.

``bench_ops`` times a shape by enqueueing many *independent* copies of one
matmul. MLX encodes those with a concurrent dispatch encoder and inserts
barriers only on buffer hazards, so with no hazard between them they overlap and
the machine fills from the queue however few threadgroups any one dispatch
launches. A forward pass has no such queue: its 497 matmuls form one
read-after-write chain and each meets the GPU alone. So a tiling whose only
defect is that it cannot fill the machine is exactly the tiling ``bench_ops``
cannot penalise, and that defect turned out to be the whole prefill deficit --
9.29x affine at ``self_attn.k_proj``'s eight threadgroups against 1.11x at
``lm_head``'s 1940.

This file measures the same candidates in the real graph, three ways, because
each answers a different part of the question and none is sufficient alone.

**keep-one** stubs every linear, restores one shape, forces that shape's layers
onto a chosen tiling, and differences against the all-stubbed pass *inside the
round*. It is what fitted :data:`~ternel_mlx.modules.GEMM_MIN_WAVES` and
:data:`~ternel_mlx.modules.GEMM_WIDE_BLOCK_WAVES`. It replaced an earlier design
that stubbed one shape *out* and subtracted from the full pass, which asked for
a 1% difference between two large numbers and returned per-shape costs larger
than the whole pass and negative on six of nine shapes.

**whole-pass** stubs nothing. Every packed linear is pinned to what the shipped
rule chooses, one shape is moved to a candidate, and the whole prefill pass is
timed. It exists because keep-one may *inflate* what it measures: stubbing every
other linear leaves a starved dispatch with nothing beside it, but q/k/v and
gate/up are independent branches whose dispatches could fill the machine for
each other in a real pass. That is ``bench_ops``' confound inverted -- one
over-fills the queue and hides starvation, the other empties it and may
exaggerate the same effect. Reading the two modes against each other at one
batch, in milliseconds rather than in ratios, is what says which.

**bucket** moves every shape over an occupancy threshold onto one tiling at
once, and each of those shapes alone in the same rounds. whole-pass prices one
retiling; a bucket is a rule that fires on several shapes in the same pass, and
the two are only the same number if the savings add. They need not: shapes that
were starved of cores are competing for the same cores once they are not, and
the ones on a dependency chain cannot overlap whatever the occupancy. Summing
nine separate whole-pass readings and calling the total a bucket's saving
assumes the answer; this mode measures it, and reports the combined arm over the
sum of its parts.

Both modes difference within the round and rotate the order of the work each
round. A fixed order gives every candidate a fixed position, so drift within a
round -- the machine warming as it proceeds -- lands on the same candidates
every time and is arithmetically indistinguishable from those candidates being
slower. It is the within-round form of the AB/BA alternation the paired
benchmarks use between arms.

The two modes stop the clock at different points, and the difference is
load-bearing rather than an inconsistency:

  keep-one evaluates the model's **logits**. It is measuring one shape's
  matmuls, so every one of that shape's dispatches has to execute. Under a lazy
  graph, evaluating the cache alone prunes the last layer's tail and ``lm_head``
  entirely, which would leave part of the census unmeasured and ``lm_head``
  unmeasurable. The logits cost is a constant that both the reference and the
  variant pay, so it cancels in the difference for every shape except the one it
  belongs to.

  whole-pass evaluates the **cache**, because that is the prefill mlx-lm runs.
  ``generate_step`` calls the model, throws the return value away, and evaluates
  only ``[c.state for c in cache]``. Timing ``mx.eval(out)`` instead charges the
  pass for a 248320-row matmul the model never issues on a prefill chunk.

Which means ``lm_head`` is whole-pass's null control rather than one of its
results: a prefill chunk does not compute logits, so retiling ``lm_head`` must
move a whole-pass reading by nothing, and a mode that reports otherwise is
reporting its own noise floor.

The batches are the caller's. Two facts about which ones are worth measuring:
the batch a prompt presents is ``prompt_tokens - 1``, since ``generate_step``
prefills every token but the last, so a 32-token prompt is a batch-31 pass; and
:data:`~ternel_mlx.modules.GEMM_MIN_BATCH` sends everything below 16 to LUT23,
so a batch under it measures a counterfactual bucket rather than what ships.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
import time
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import numpy as np
from mlx.utils import tree_map_with_path
from mlx_lm.models.cache import make_prompt_cache
from mlx_lm.utils import load_model

from bonsai_tq1.format import BLOCK_SIZE, FormatError, write_json_atomic

from . import LAYOUT_NAME, LAYOUT_VERSION
from .environment import capture_environment, exclusive_gpu
from .kernels import GemmConfig, MatmulConfig, tq1_gemm, tq1_matmul
from .layout import (
    DISPATCHES_PER_TOKEN,
    MODEL_LINEAR_SHAPES,
    ModelLinearShape,
    PackedTensorLayout,
)
from .modules import (
    GEMM_LADDER_LARGE_BATCH,
    GEMM_LADDER_SMALL_BATCH,
    GEMM_LARGE_MIN_BATCH,
    GEMM_MIN_BATCH,
    select_gemm,
    select_lut23,
    threadgroups,
)

# 2 adds ``reference_seconds`` and ``reference_samples`` to every candidate: the
# pass each delta was differenced against. A version 1 file records differences
# with no scale to read them at.
SCHEMA_VERSION = 2

SEED = 20260809

KEEP_ONE = "keep-one"
WHOLE_PASS = "whole-pass"
BUCKET = "bucket"
MODES = (KEEP_ONE, WHOLE_PASS, BUCKET)

# The name the affine control is recorded under in keep-one mode. It is one more
# entry in the round's rotation rather than a coda after it: it is the
# denominator of every ratio the round reports, and a denominator pinned to the
# last slot carries whatever that slot does to a reading.
AFFINE = "affine"
LUT23 = "lut23"

# What a linear leaf looks like in either arm. ``PackedLinear`` is ours,
# ``QuantizedLinear`` is the affine baseline's, and ``Linear`` covers the layers
# mlx-lm leaves unquantised in both.
LINEAR_TYPES = ("PackedLinear", "QuantizedLinear", "Linear")

# Vocabulary of Ternary Bonsai 27B. Only used to draw token ids, so what matters
# is that every id is in range for both arms' embedding tables.
VOCAB = 151936

# batch_block, row_block, k_block, simd_rows, simd_columns, direct_fragments,
# direct_epilogue, threadgroup_table.
#
# The set keep-one sweeps. It is built around threadgroup count at the measured
# batch -- a tiling's dispatch offers ``ceil(rows/row_block) *
# ceil(batch/batch_block)`` threadgroups -- so it deliberately holds pairs that
# agree on that product and disagree on block shape: 16x128 against 8x256
# against 32x128 at batch 16, 32x128 against 64x64 at every batch. Those pairs
# are the controls. If they move the ranking, threadgroup count is not the whole
# story, which is what happened at batch 32 when 8x256 came out 1.9x worse than
# the incumbent at identical occupancy: how many activation fragments a lane
# holds matters too.
#
# The two 8-deep rungs are here for the batch-8 bucket question. Each is the
# form of the shipped rung it shadows with the batch block halved and
# ``simd_rows`` dropped to keep TM where that rung had it -- 8x128 the staged
# form of ``GEMM_SMALL_BATCH_WIDE_BLOCK``, 8x32 the register form of
# ``GEMM_SMALL_BATCH_NARROW_BLOCK``. A bucket keyed at batch 8 would dispatch
# one of these, so asking whether the bucket is worth having means measuring
# them rather than the 16-deep rungs run half empty.
LADDER: tuple[tuple[str, tuple[int, int, int, int, int, bool, bool, bool]], ...] = (
    ("gemm_8x32", (8, 32, 32, 1, 2, True, True, True)),
    ("gemm_8x128", (8, 128, 64, 1, 4, False, False, False)),
    ("gemm_8x256", (8, 256, 32, 1, 8, False, False, False)),
    ("gemm_16x32", (16, 32, 32, 2, 2, True, True, True)),
    ("gemm_16x128", (16, 128, 64, 2, 4, False, False, False)),
    ("gemm_32x32", (32, 32, 32, 1, 2, True, True, True)),
    ("gemm_32x64", (32, 64, 32, 1, 2, True, True, True)),
    ("gemm_32x128", (32, 128, 32, 1, 4, True, False, False)),
    ("gemm_64x64", (64, 64, 32, 2, 2, True, False, False)),
    ("gemm_64x128", (64, 128, 32, 2, 4, True, False, False)),
    ("gemm_128x32", (128, 32, 32, 4, 2, True, True, False)),
    ("gemm_128x128", (128, 128, 32, 4, 4, True, False, False)),
    ("gemm_128x256", (128, 256, 32, 4, 8, True, True, False)),
)


def build(spec: tuple[int, int, int, int, int, bool, bool, bool]) -> GemmConfig:
    (
        batch_block, row_block, k_block, simd_rows, simd_columns,
        direct_fragments, direct_epilogue, threadgroup_table,
    ) = spec
    return GemmConfig(
        batch_block=batch_block, row_block=row_block, k_block=k_block,
        simd_rows=simd_rows, simd_columns=simd_columns,
        direct_fragments=direct_fragments, direct_epilogue=direct_epilogue,
        threadgroup_table=threadgroup_table, safe_clamp=False,
    )


def block_name(config: GemmConfig | MatmulConfig) -> str:
    """The key a candidate is recorded and compared under.

    Deliberately coarser than ``benchmark_name``, which also encodes the k block
    and which of the six kernel forms a tiling runs. Those belong in the record
    -- every candidate carries its ``benchmark_name`` beside its timing -- but
    not in the key, because the key has to mean the same thing in both modes and
    at every batch for the two to be read against each other. LUT23's
    ``benchmark_name`` carries its batch tile and K split, so keying on it would
    give the same arm a different name at every shape it is measured on.
    """
    if isinstance(config, GemmConfig):
        return f"gemm_{config.batch_block}x{config.row_block}"
    return LUT23


def relevant(batch: int, batch_block: int) -> bool:
    """Is a tiling worth a slot at this batch?

    A batch block deeper than the batch runs part empty and buys no
    threadgroups, since the batch still splits one way -- but one step past the
    batch stays in, because that is where the ladder's own boundary sits and a
    sweep that cannot see over its threshold cannot move it.
    """
    return batch_block <= 2 * batch


def fits(shape: ModelLinearShape, config: GemmConfig) -> bool:
    """Can this tiling run this tensor at all?

    A row block straddling two row tiles would read the second tile's codes
    through the first tile's base pointer, so :func:`select_gemm` refuses that
    pairing -- and so does this, rather than measuring an unsafe kernel.
    """
    layout = shape.layout()
    return layout.tiles == 1 or layout.tile % config.row_block == 0


def ladder_candidates(
    shape: ModelLinearShape, batch: int
) -> tuple[tuple[str, GemmConfig | MatmulConfig], ...]:
    """Every tiling keep-one sweeps for one shape at one batch, LUT23 last."""
    out: list[tuple[str, GemmConfig | MatmulConfig]] = []
    for _, spec in LADDER:
        if not relevant(batch, spec[0]):
            continue
        config = build(spec)
        if not fits(shape, config):
            continue
        out.append((block_name(config), config))
    out.append((LUT23, select_lut23(shape.layout(), batch)))
    return tuple(out)


def bucket_block(batch: int) -> int | None:
    """The batch block a bucket keyed at this batch would dispatch, or ``None``.

    Only defined below :data:`~ternel_mlx.modules.GEMM_MIN_BATCH`. Above it the
    rule already has a bucket for this batch and the ladder rungs *are* what it
    would dispatch; below it every shape goes to LUT23, so the counterfactual a
    whole-pass reading has to test is a tiling that appears on no shipped
    ladder. A 16-deep rung run at batch 8 is not that tiling -- it wastes half
    its lanes and buys no threadgroups, since the batch still splits one way --
    so measuring it and calling the answer "what a bucket would cost" would
    understate the bucket.

    ``None`` when the sweep's ladder holds nothing that shallow, which is a
    refusal to guess rather than a fallback: the answer would be about a tiling
    this file never measured.
    """
    if batch >= GEMM_MIN_BATCH:
        return None
    blocks = sorted(block for _, spec in LADDER if (block := spec[0]) <= batch)
    return blocks[-1] if blocks else None


def decision_candidates(
    shape: ModelLinearShape, batch: int
) -> tuple[tuple[str, GemmConfig | MatmulConfig], ...]:
    """The set the rule is actually choosing between, LUT23 last.

    Narrower than the sweep's ladder on purpose. A whole-pass reading costs a
    whole forward pass, so what it can afford to measure is the rule's own
    alternatives rather than the space the rule was fitted over -- and those
    alternatives are what a disagreement between the two modes would be about.

    The ladder is picked by batch the way :func:`select_gemm` picks it, but
    without its :data:`~ternel_mlx.modules.GEMM_MIN_BATCH` floor, and below that
    floor :func:`bucket_block`'s rungs go in front of it.
    """
    ladder = (
        GEMM_LADDER_LARGE_BATCH if batch >= GEMM_LARGE_MIN_BATCH
        else GEMM_LADDER_SMALL_BATCH
    )
    block = bucket_block(batch)
    bucket = [build(spec) for _, spec in LADDER if block is not None and spec[0] == block]
    out: list[tuple[str, GemmConfig | MatmulConfig]] = []
    for config in (*bucket, *ladder):
        if not fits(shape, config):
            continue
        name = block_name(config)
        if name in {seen for seen, _ in out}:
            raise FormatError(f"two candidates for {shape.label} would record as {name}")
        out.append((name, config))
    out.append((LUT23, select_lut23(shape.layout(), batch)))
    return tuple(out)


def rung(name: str) -> GemmConfig:
    """The ladder rung recorded under ``name``.

    Named rather than described so a bucket run and the whole-pass run it is
    checked against cannot drift apart in the tiling they mean: both resolve the
    same string through the same table.
    """
    for _, spec in LADDER:
        config = build(spec)
        if block_name(config) == name:
            return config
    raise FormatError(f"no ladder rung is recorded as {name}")


def bucket_members(
    batch: int, config: GemmConfig, min_threadgroups: int
) -> tuple[ModelLinearShape, ...]:
    """Every shape a bucket keyed on occupancy would move onto ``config``.

    The threshold is on threadgroups rather than on rows because rows are only a
    proxy: what a starved dispatch lacks is threadgroups to hand the cores, and
    the same row count supplies a different number of them at a different row
    block. Shapes the tiling cannot legally run are excluded, not clamped -- a
    row block straddling two row tiles reads the second tile through the first
    tile's pointer, so there is no such thing as running it anyway.

    ``lm_head`` clears any threshold that matters here and is left in for that
    reason. A prefill chunk never issues it, so it contributes a term whose true
    value is zero to the combined arm: the bucket keeps its null control even
    when every shape moves at once.
    """
    return tuple(
        shape for shape in MODEL_LINEAR_SHAPES
        if fits(shape, config)
        and dispatch_threadgroups(shape, batch, config) >= min_threadgroups
    )


def shipped_for_layout(
    layout: PackedTensorLayout, batch: int
) -> GemmConfig | MatmulConfig:
    """What the shipped rule dispatches for this layout at this batch.

    :func:`select_gemm` returning ``None`` is not a refusal to answer, it is the
    answer "LUT23", so the two calls belong together wherever the question is
    what actually runs.
    """
    config = select_gemm(layout, batch)
    return select_lut23(layout, batch) if config is None else config


def shipped_config(shape: ModelLinearShape, batch: int) -> GemmConfig | MatmulConfig:
    """What the shipped rule dispatches for this shape at this batch."""
    return shipped_for_layout(shape.layout(), batch)


def dispatch_threadgroups(shape: ModelLinearShape, batch: int, config: object) -> int:
    """The parallelism a candidate's dispatch offers, whichever kernel it is.

    A GEMM's supply comes from tiling the output; LUT23's comes from its row
    tiles times its batch tiles times its K-axis split. Both are counted so the
    two families can be compared on the axis the rule decides on.
    """
    layout = shape.layout()
    if isinstance(config, GemmConfig):
        return threadgroups(layout, batch, config)
    if isinstance(config, MatmulConfig):
        return (
            layout.tiles
            * math.ceil(max(batch, 1) / config.batch_tile)
            * config.k_split
        )
    raise FormatError(f"cannot count threadgroups for {type(config).__name__}")


class StubLinear(nn.Module):
    """A linear layer's shape without its arithmetic.

    Broadcasts the activation's first column to the output width and scales it,
    which materialises a real array of exactly the shape the real layer would
    return. Keeping it real matters: a view would make every downstream op read
    less memory than it does in the shipped graph, and downstream is precisely
    what this measurement holds constant.
    """

    def __init__(self, out_features: int) -> None:
        super().__init__()
        self.out_features = out_features
        self.scale = mx.array(1.0)

    def __call__(self, x: mx.array) -> mx.array:
        wide = mx.broadcast_to(x[..., :1], (*x.shape[:-1], self.out_features))
        return wide * self.scale.astype(x.dtype)


class ForcedLinear(nn.Module):
    """One packed layer pinned to a chosen tiling, sharing the loaded weights.

    Holds the source module's own ``codes`` and ``scales`` rather than copies,
    so switching configuration between rounds costs nothing and cannot change
    what is resident.
    """

    def __init__(self, source: nn.Module, config: GemmConfig | MatmulConfig) -> None:
        super().__init__()
        self.layout = source.layout
        self.codes = source.codes
        self.scales = source.scales
        self.config = config

    def __call__(self, x: mx.array) -> mx.array:
        flat = x.reshape(-1, self.layout.columns)
        if isinstance(self.config, GemmConfig):
            out = tq1_gemm(
                self.codes, self.scales, flat,
                groups_per_row=self.layout.groups_per_row,
                tile=self.layout.tile, config=self.config,
            )
        else:
            out = tq1_matmul(
                self.codes, self.scales, flat,
                groups_per_row=self.layout.groups_per_row,
                tile=self.layout.tile, config=self.config,
            )
        return out.reshape(*x.shape[:-1], self.layout.rows)


def shape_of(module: nn.Module) -> tuple[int, int]:
    """A linear leaf's (rows, columns), read the same way for either arm."""
    layout = getattr(module, "layout", None)
    if layout is not None:
        return int(layout.rows), int(layout.columns)
    return int(module.weight.shape[0]), int(module.scales.shape[1]) * BLOCK_SIZE


def stub_all_but(
    model: nn.Module,
    original: dict[str, object],
    keep: tuple[int, int] | None,
    forced: GemmConfig | MatmulConfig | None,
) -> int:
    """Stub every linear; restore shape ``keep``, optionally onto ``forced``.

    Returns how many layers were kept, which the caller checks against the
    census. A shape that silently loses a layer would report a cost for fewer
    dispatches than it has.
    """
    kept = 0

    def replace(_path: str, module: nn.Module) -> nn.Module:
        nonlocal kept
        if type(module).__name__ not in LINEAR_TYPES:
            return module
        if keep is not None and shape_of(module) == keep:
            kept += 1
            return module if forced is None else ForcedLinear(module, forced)
        return StubLinear(shape_of(module)[0])

    model.update_modules(tree_map_with_path(replace, original, is_leaf=nn.Module.is_module))
    return kept


def pin_all(
    model: nn.Module,
    original: dict[str, object],
    batch: int,
    moves: dict[tuple[int, int], GemmConfig | MatmulConfig],
) -> tuple[int, int]:
    """Pin every packed linear to the shipped choice, then apply ``moves``.

    Returns (layers pinned, layers moved off the shipped choice). Pinning even
    the untouched layers matters: it removes ``select_gemm`` from the timed path
    entirely, so the two passes being differenced differ in the tiling of the
    moved shapes and in nothing else -- not in a cache lookup, not in a branch.
    An empty ``moves`` is the reference pass every delta is taken against.

    ``moves`` is a mapping rather than one shape because a bucket is a rule over
    several shapes at once, and whether those shapes' savings *add* is not
    something a run that moves them one at a time can answer. Two starved
    dispatches made healthy in the same pass may each have been waiting on the
    same idle cores, in which case fixing both buys less than the sum of fixing
    each; or they may serialise behind a dependency, in which case it buys the
    sum. Only moving them together says which.

    The census is keyed on (rows, columns) rather than on rows alone because two
    of its nine entries share a row count: ``linear_attn.out_proj`` is 5120x6144
    and ``mlp.down_proj`` is 5120x17408.
    """
    known = {(shape.rows, shape.columns) for shape in MODEL_LINEAR_SHAPES}
    unknown = set(moves) - known
    if unknown:
        raise FormatError(f"asked to move {sorted(unknown)}, which the census does not hold")
    pinned = 0
    moved = 0

    def replace(_path: str, module: nn.Module) -> nn.Module:
        nonlocal pinned, moved
        if type(module).__name__ not in LINEAR_TYPES:
            return module
        layout = getattr(module, "layout", None)
        if layout is None:
            raise FormatError("whole-pass mode expects the packed model, not the affine one")
        rows, columns = int(layout.rows), int(layout.columns)
        if (rows, columns) not in known:
            raise FormatError(f"the model holds a {rows}x{columns} linear the census does not")
        config = shipped_for_layout(layout, batch)
        forced = moves.get((rows, columns))
        if forced is not None:
            if forced != config:
                moved += 1
            config = forced
        pinned += 1
        return ForcedLinear(module, config)

    model.update_modules(tree_map_with_path(replace, original, is_leaf=nn.Module.is_module))
    return pinned, moved


def logits_pass(model: nn.Module, tokens: mx.array) -> float:
    """Seconds for one forward pass with the logits evaluated.

    keep-one's clock. Evaluating the whole output is what forces every one of
    the 497 dispatches to run: under a lazy graph, evaluating the cache alone
    prunes the last layer's tail and ``lm_head``, and a shape whose layers were
    partly pruned would report a cost for fewer dispatches than it has.
    """
    cache = make_prompt_cache(model)
    mx.synchronize()
    start = time.perf_counter()
    out = model(tokens, cache=cache)
    mx.eval(out)
    mx.synchronize()
    elapsed = time.perf_counter() - start
    del cache, out
    mx.clear_cache()
    return elapsed


def prefill_pass(model: nn.Module, tokens: mx.array) -> float:
    """Seconds for one prefill chunk, cache built outside the clock.

    whole-pass's clock, and the prefill mlx-lm runs. ``generate_step``'s chunk
    loop calls the model, throws the return value away, and evaluates only
    ``[c.state for c in cache]``; under a lazy graph that means ``lm_head``
    never executes on a prefill chunk, and neither does the last layer's tail,
    which nothing downstream of the cache depends on. Timing ``mx.eval(out)``
    instead charges the pass for a 248320-row matmul the model never issues.
    """
    cache = make_prompt_cache(model)
    mx.synchronize()
    start = time.perf_counter()
    model(tokens, cache=cache)
    mx.eval([c.state for c in cache])
    mx.synchronize()
    elapsed = time.perf_counter() - start
    del cache
    mx.clear_cache()
    return elapsed


def timed(clock, model: nn.Module, tokens: mx.array, *, inner: int) -> float:
    """Median of ``inner`` passes -- one reading, cheap enough to sit in a round.

    One untimed pass first. Interleaving two 27B models means a reading can
    follow a pass through the *other* one, so the first pass back pays for
    weights that model evicted; priming keeps that off the clock instead of
    relying on the median to discard it.
    """
    clock(model, tokens)
    return statistics.median(clock(model, tokens) for _ in range(inner))


def rotate(work: tuple, index: int) -> tuple:
    """The round's order. Rotating is what keeps drift off any one candidate."""
    position = index % len(work)
    return work[position:] + work[:position]


def summarise(
    deltas: dict[str, list[float]],
    references: dict[str, list[float]],
    options: tuple[tuple[str, GemmConfig | MatmulConfig], ...],
    shape: ModelLinearShape,
    batch: int,
    *,
    control: float | None,
) -> dict[str, object]:
    """Per-candidate medians, and which one won.

    ``control`` is the affine cost of the same shape in the same rounds, and
    only keep-one has one: it measures a shape's absolute cost, so a ratio
    against the thing the shape has to beat is meaningful. whole-pass measures a
    *difference* from the shipped pass, which straddles zero by construction --
    the shipped candidate's own readings are the mode's noise floor -- so it
    reports milliseconds and no ratio at all.

    ``references`` is the other half of every difference: the pass each delta
    was subtracted from, kept per candidate because each candidate has its own,
    measured in its own round. A difference with the thing it is a difference
    from thrown away cannot be read -- eight milliseconds off a prefill is a
    result or a rounding error depending entirely on how long that prefill was,
    and only this column says which.
    """
    incumbent = shipped_config(shape, batch)
    rows: dict[str, object] = {}
    best_name, best_cost = None, None
    for name, config in options:
        cost = statistics.median(deltas[name])
        entry: dict[str, object] = {
            "seconds": cost,
            "samples": deltas[name],
            "reference_seconds": statistics.median(references[name]),
            "reference_samples": references[name],
            "threadgroups": dispatch_threadgroups(shape, batch, config),
            "family": "lut23" if isinstance(config, MatmulConfig) else "gemm",
            "config": config.benchmark_name,
            "shipped": config == incumbent,
        }
        if control is not None:
            entry["over_affine"] = cost / control
        rows[name] = entry
        if best_cost is None or cost < best_cost:
            best_name, best_cost = name, cost
    shipped_name = next((n for n, c in options if c == incumbent), None)
    summary: dict[str, object] = {
        "label": shape.label,
        "modules": list(shape.modules),
        "rows": shape.rows,
        "columns": shape.columns,
        "dispatches_per_token": shape.dispatches_per_token,
        "candidates": rows,
        "best": best_name,
        "shipped": shipped_name,
    }
    if control is not None:
        summary["affine_seconds"] = control
        summary["affine_samples"] = deltas[AFFINE]
        summary["shipped_over_best"] = (
            None if shipped_name is None or not best_cost
            else statistics.median(deltas[shipped_name]) / best_cost
        )
    return summary


def run_keep_one(
    packed: nn.Module,
    affine: nn.Module,
    originals: dict[str, dict[str, object]],
    *,
    batches: tuple[int, ...],
    rounds: int,
    inner: int,
    warmup: int,
    max_foreign_gpu_share: float,
    contention: list[dict[str, object]],
) -> dict[str, object]:
    """Every legal tiling for every shape, differenced against an empty pass."""
    generator = np.random.default_rng(SEED)
    corpus = mx.array(
        generator.integers(0, VOCAB, size=max(batches), dtype=np.int32), dtype=mx.int32
    )
    mx.eval(corpus)

    record: dict[str, object] = {}
    for batch in batches:
        tokens = corpus[:batch][None]
        for name, model in (("packed", packed), (AFFINE, affine)):
            stub_all_but(model, originals[name], None, None)
            for _ in range(warmup):
                logits_pass(model, tokens)

        print(f"\n=== batch {batch}, keep-one, {rounds} rounds", file=sys.stderr, flush=True)
        per_batch: dict[str, object] = {}
        for shape in MODEL_LINEAR_SHAPES:
            keep = (shape.rows, shape.columns)
            options = ladder_candidates(shape, batch)
            # Compile every kernel this shape will use before any is timed, so
            # the first round does not charge one candidate for a Metal library
            # the rest already have.
            for _, config in options:
                stub_all_but(packed, originals["packed"], keep, config)
                logits_pass(packed, tokens)
            stub_all_but(affine, originals[AFFINE], keep, None)
            logits_pass(affine, tokens)

            work = (*options, (AFFINE, None))
            deltas: dict[str, list[float]] = {name: [] for name, _ in work}
            references: dict[str, list[float]] = {name: [] for name, _ in work}
            with exclusive_gpu(
                f"keep-one {shape.label} batch={batch}",
                max_foreign_share=max_foreign_gpu_share,
                record=contention,
            ):
                for index in range(rounds):
                    for name, config in rotate(work, index):
                        model, original = (
                            (affine, originals[AFFINE]) if name == AFFINE
                            else (packed, originals["packed"])
                        )
                        stub_all_but(model, original, None, None)
                        bare = timed(logits_pass, model, tokens, inner=inner)
                        kept = stub_all_but(model, original, keep, config)
                        if kept != shape.dispatches_per_token:
                            raise FormatError(
                                f"{shape.label}: kept {kept} layers, the census says "
                                f"{shape.dispatches_per_token}"
                            )
                        references[name].append(bare)
                        deltas[name].append(
                            timed(logits_pass, model, tokens, inner=inner) - bare
                        )

            control = statistics.median(deltas[AFFINE])
            entry = summarise(deltas, references, options, shape, batch, control=control)
            per_batch[shape.label] = entry
            best = min(options, key=lambda option: statistics.median(deltas[option[0]]))[0]
            print(
                f"  {shape.label:24s} {shape.rows:6d}x{shape.columns:<5d} "
                f"stub {statistics.median(references[best]) * 1e3:6.1f}ms   "
                f"affine {control * 1e3:7.1f}ms   best {best} at "
                f"{statistics.median(deltas[best]) / control:.2f}x affine, "
                f"shipped {entry['shipped']}",
                file=sys.stderr, flush=True,
            )
        record[str(batch)] = {"shapes": per_batch}

    return record


def run_whole_pass(
    packed: nn.Module,
    original: dict[str, object],
    *,
    batches: tuple[int, ...],
    rounds: int,
    inner: int,
    warmup: int,
    max_foreign_gpu_share: float,
    contention: list[dict[str, object]],
) -> dict[str, object]:
    """One shape moved off the shipped tiling, the whole prefill pass timed."""
    generator = np.random.default_rng(SEED)
    corpus = mx.array(
        generator.integers(0, VOCAB, size=max(batches), dtype=np.int32), dtype=mx.int32
    )
    mx.eval(corpus)

    record: dict[str, object] = {}
    for batch in batches:
        tokens = corpus[:batch][None]
        pinned, _ = pin_all(packed, original, batch, {})
        if pinned != DISPATCHES_PER_TOKEN:
            raise FormatError(
                f"pinned {pinned} linears, the census says {DISPATCHES_PER_TOKEN}"
            )
        for _ in range(warmup):
            prefill_pass(packed, tokens)

        print(f"\n=== batch {batch}, whole-pass, {rounds} rounds", file=sys.stderr, flush=True)
        per_batch: dict[str, object] = {}
        for shape in MODEL_LINEAR_SHAPES:
            keep = (shape.rows, shape.columns)
            incumbent = shipped_config(shape, batch)
            options = decision_candidates(shape, batch)
            for _, config in options:
                pin_all(packed, original, batch, {keep: config})
                prefill_pass(packed, tokens)

            deltas: dict[str, list[float]] = {name: [] for name, _ in options}
            references: dict[str, list[float]] = {name: [] for name, _ in options}
            with exclusive_gpu(
                f"whole-pass {shape.label} batch={batch}",
                max_foreign_share=max_foreign_gpu_share,
                record=contention,
            ):
                for index in range(rounds):
                    for name, config in rotate(options, index):
                        pin_all(packed, original, batch, {})
                        base = timed(prefill_pass, packed, tokens, inner=inner)
                        _, moved = pin_all(packed, original, batch, {keep: config})
                        expected = 0 if config == incumbent else shape.dispatches_per_token
                        if moved != expected:
                            raise FormatError(
                                f"{shape.label}: moved {moved} layers onto {name}, "
                                f"expected {expected}"
                            )
                        references[name].append(base)
                        deltas[name].append(
                            timed(prefill_pass, packed, tokens, inner=inner) - base
                        )

            entry = summarise(deltas, references, options, shape, batch, control=None)
            per_batch[shape.label] = entry
            shipped_name = entry["shipped"]
            print(
                f"  {shape.label:24s} "
                f"pass {statistics.median(references[shipped_name]) * 1e3:6.1f}ms   "
                + "  ".join(
                    f"{name}{'*' if name == shipped_name else ''} "
                    f"{statistics.median(deltas[name]) * 1e3:+7.1f}ms"
                    for name, _ in options
                ),
                file=sys.stderr, flush=True,
            )
        record[str(batch)] = {"shapes": per_batch}

    return record


def summarise_bucket(
    deltas: dict[str, list[float]],
    references: dict[str, list[float]],
    members: tuple[ModelLinearShape, ...],
    batch: int,
    config: GemmConfig,
) -> dict[str, object]:
    """The combined arm, its parts, and the gap between the two.

    ``sum_of_parts_seconds`` is what a run that moves one shape at a time can
    offer, and ``additivity`` is the combined arm divided by it. The whole point
    of the mode is that this number is not assumed to be 1: a bucket that moves
    four starved dispatches is not four independent experiments, since they
    contend for the same cores and sit on the same dependency chain.
    """
    rows: dict[str, object] = {}
    for name in (BUCKET, *(shape.label for shape in members)):
        rows[name] = {
            "seconds": statistics.median(deltas[name]),
            "samples": deltas[name],
            "reference_seconds": statistics.median(references[name]),
            "reference_samples": references[name],
        }
    for shape in members:
        entry = rows[shape.label]
        if not isinstance(entry, dict):
            raise FormatError(f"{shape.label} did not record a mapping")
        entry["rows"] = shape.rows
        entry["columns"] = shape.columns
        entry["dispatches_per_token"] = shape.dispatches_per_token
        entry["threadgroups"] = dispatch_threadgroups(shape, batch, config)
        entry["shipped"] = shipped_config(shape, batch).benchmark_name
    combined = statistics.median(deltas[BUCKET])
    parts = sum(statistics.median(deltas[shape.label]) for shape in members)
    return {
        "arm": config.benchmark_name,
        "block": block_name(config),
        "members": [shape.label for shape in members],
        "layers_moved": sum(
            shape.dispatches_per_token for shape in members
            if shipped_config(shape, batch) != config
        ),
        "arms": rows,
        "sum_of_parts_seconds": parts,
        "additivity": None if not parts else combined / parts,
    }


def run_bucket(
    packed: nn.Module,
    original: dict[str, object],
    *,
    batches: tuple[int, ...],
    rounds: int,
    inner: int,
    warmup: int,
    max_foreign_gpu_share: float,
    contention: list[dict[str, object]],
    arm: str,
    min_threadgroups: int,
) -> dict[str, object]:
    """Every qualifying shape moved at once, against each moved alone.

    whole-pass mode answers "what does retiling this one shape cost", which is
    the right question for ranking candidates and the wrong one for shipping a
    bucket: a bucket moves every qualifying shape in the same pass, and the only
    honest way to price that is to run it. The per-shape arms are measured in the
    *same* rounds as the combined one so the comparison is not across machine
    states -- the additivity question is a couple of milliseconds wide, which is
    the same width as the drift between two runs.
    """
    generator = np.random.default_rng(SEED)
    corpus = mx.array(
        generator.integers(0, VOCAB, size=max(batches), dtype=np.int32), dtype=mx.int32
    )
    mx.eval(corpus)
    config = rung(arm)

    record: dict[str, object] = {}
    for batch in batches:
        tokens = corpus[:batch][None]
        members = bucket_members(batch, config, min_threadgroups)
        if not members:
            raise FormatError(
                f"no shape reaches {min_threadgroups} threadgroups on {arm} at batch {batch}"
            )
        collision = [shape.label for shape in members if shape.label == BUCKET]
        if collision:
            raise FormatError(f"a shape labelled {BUCKET} would overwrite the combined arm")
        every = {(shape.rows, shape.columns): config for shape in members}
        options: tuple[tuple[str, dict[tuple[int, int], GemmConfig]], ...] = (
            (BUCKET, every),
            *(
                (shape.label, {(shape.rows, shape.columns): config})
                for shape in members
            ),
        )

        pinned, _ = pin_all(packed, original, batch, {})
        if pinned != DISPATCHES_PER_TOKEN:
            raise FormatError(
                f"pinned {pinned} linears, the census says {DISPATCHES_PER_TOKEN}"
            )
        for _ in range(warmup):
            prefill_pass(packed, tokens)
        for _, moves in options:
            pin_all(packed, original, batch, moves)
            prefill_pass(packed, tokens)

        print(
            f"\n=== batch {batch}, bucket {arm} over {len(members)} shapes at "
            f">={min_threadgroups} threadgroups, {rounds} rounds",
            file=sys.stderr, flush=True,
        )
        deltas: dict[str, list[float]] = {name: [] for name, _ in options}
        references: dict[str, list[float]] = {name: [] for name, _ in options}
        with exclusive_gpu(
            f"bucket {arm} batch={batch}",
            max_foreign_share=max_foreign_gpu_share,
            record=contention,
        ):
            for index in range(rounds):
                for name, moves in rotate(options, index):
                    pin_all(packed, original, batch, {})
                    base = timed(prefill_pass, packed, tokens, inner=inner)
                    _, moved = pin_all(packed, original, batch, moves)
                    expected = sum(
                        shape.dispatches_per_token for shape in members
                        if (shape.rows, shape.columns) in moves
                        and shipped_config(shape, batch) != config
                    )
                    if moved != expected:
                        raise FormatError(
                            f"{name}: moved {moved} layers, expected {expected}"
                        )
                    references[name].append(base)
                    deltas[name].append(
                        timed(prefill_pass, packed, tokens, inner=inner) - base
                    )

        entry = summarise_bucket(deltas, references, members, batch, config)
        record[str(batch)] = entry
        print(
            f"  pass {statistics.median(references[BUCKET]) * 1e3:6.1f}ms   "
            f"bucket {statistics.median(deltas[BUCKET]) * 1e3:+7.2f}ms   "
            f"parts {entry['sum_of_parts_seconds'] * 1e3:+7.2f}ms   "
            + "  ".join(
                f"{shape.label} {statistics.median(deltas[shape.label]) * 1e3:+6.2f}"
                for shape in members
            ),
            file=sys.stderr, flush=True,
        )

    return record


def run(
    *,
    mode: str,
    artifact: Path,
    baseline: Path | None,
    batches: tuple[int, ...],
    rounds: int,
    inner: int,
    warmup: int,
    max_foreign_gpu_share: float,
    bucket_arm: str | None,
    bucket_min_threadgroups: int | None,
) -> dict[str, object]:
    environment_before = capture_environment()
    contention: list[dict[str, object]] = []

    packed, _ = load_model(artifact)
    packed_original = packed.leaf_modules()
    if mode == KEEP_ONE:
        if baseline is None:
            raise FormatError("keep-one mode needs the affine baseline it divides by")
        affine, _ = load_model(baseline)
        measured = run_keep_one(
            packed, affine,
            {"packed": packed_original, AFFINE: affine.leaf_modules()},
            batches=batches, rounds=rounds, inner=inner, warmup=warmup,
            max_foreign_gpu_share=max_foreign_gpu_share, contention=contention,
        )
    elif mode == BUCKET:
        if bucket_arm is None or bucket_min_threadgroups is None:
            raise FormatError("bucket mode needs both the arm and the occupancy threshold")
        measured = run_bucket(
            packed, packed_original,
            batches=batches, rounds=rounds, inner=inner, warmup=warmup,
            max_foreign_gpu_share=max_foreign_gpu_share, contention=contention,
            arm=bucket_arm, min_threadgroups=bucket_min_threadgroups,
        )
    else:
        measured = run_whole_pass(
            packed, packed_original,
            batches=batches, rounds=rounds, inner=inner, warmup=warmup,
            max_foreign_gpu_share=max_foreign_gpu_share, contention=contention,
        )

    environment_after = capture_environment()
    return {
        "schema_version": SCHEMA_VERSION,
        "layout_name": LAYOUT_NAME,
        "layout_version": LAYOUT_VERSION,
        "seed": SEED,
        "mode": mode,
        "artifact": str(artifact),
        "baseline": None if baseline is None else str(baseline),
        "batches": list(batches),
        "rounds": rounds,
        "inner_passes_per_reading": inner,
        "warmup_passes": warmup,
        "bucket_arm": bucket_arm,
        "bucket_min_threadgroups": bucket_min_threadgroups,
        "evaluated": "logits" if mode == KEEP_ONE else "cache",
        "dispatches_per_token": DISPATCHES_PER_TOKEN,
        "gemm_min_batch": GEMM_MIN_BATCH,
        "gemm_large_min_batch": GEMM_LARGE_MIN_BATCH,
        "max_foreign_gpu_share": max_foreign_gpu_share,
        "gpu_contention": contention,
        "measured": measured,
        "environment_before": environment_before,
        "environment_after": environment_after,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Rank the prefill tilings inside the forward pass (B9)"
    )
    parser.add_argument(
        "--mode", choices=MODES, required=True,
        help=(
            "keep-one stubs every linear but one shape and differences against the "
            "empty pass; whole-pass stubs nothing and moves one shape off the shipped "
            "tiling inside the real graph; bucket moves every shape over an occupancy "
            "threshold at once, and each of them alone in the same rounds, so the "
            "combined saving can be read against the sum of the separate ones"
        ),
    )
    parser.add_argument("--artifact", type=Path, required=True, help="the packed model")
    parser.add_argument(
        "--baseline", type=Path, required=False,
        help="the affine 2-bit model; keep-one divides by it, whole-pass does not use it",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--batches", type=int, nargs="+", required=True,
        help=(
            "activation rows per pass. A prompt of N tokens prefills at N-1, so these "
            "are prompt lengths minus one if the question is end-to-end"
        ),
    )
    parser.add_argument(
        "--rounds", type=int, required=True,
        help="paired readings per candidate, each differenced within the round",
    )
    parser.add_argument(
        "--inner", type=int, required=True, help="timed passes per reading, medianed"
    )
    parser.add_argument(
        "--warmup", type=int, required=True, help="untimed passes before a batch is measured"
    )
    parser.add_argument(
        "--max-foreign-gpu-share", type=float, required=True,
        help="void the run if another process exceeds this share of the GPU while timing",
    )
    parser.add_argument(
        "--bucket-arm", type=str, required=False,
        help="bucket mode only: the ladder rung the bucket dispatches, e.g. gemm_8x32",
    )
    parser.add_argument(
        "--bucket-min-threadgroups", type=int, required=False,
        help="bucket mode only: the occupancy a shape must reach on that rung to join",
    )
    args = parser.parse_args(argv)

    if args.mode == KEEP_ONE and args.baseline is None:
        parser.error("--baseline is required in keep-one mode: it is the ratio's denominator")
    if args.mode != KEEP_ONE and args.baseline is not None:
        parser.error(f"--baseline is not used in {args.mode} mode; there is no affine arm")
    needs_bucket = args.mode == BUCKET
    has_bucket = args.bucket_arm is not None or args.bucket_min_threadgroups is not None
    if needs_bucket and not (args.bucket_arm and args.bucket_min_threadgroups is not None):
        parser.error("bucket mode needs both --bucket-arm and --bucket-min-threadgroups")
    if has_bucket and not needs_bucket:
        parser.error(f"--bucket-arm and --bucket-min-threadgroups mean nothing in {args.mode} mode")
    if needs_bucket and args.bucket_min_threadgroups < 1:
        raise FormatError(
            f"the occupancy threshold must be positive, got {args.bucket_min_threadgroups}"
        )
    if min(args.rounds, args.inner, args.warmup) < 1:
        raise FormatError("rounds, inner and warmup must all be at least one")
    if not args.batches:
        raise FormatError("at least one batch must be measured")
    if min(args.batches) < 1:
        raise FormatError(f"batches must be positive, got {sorted(args.batches)}")
    if not args.max_foreign_gpu_share > 0.0:
        raise FormatError(f"max foreign share must be positive, got {args.max_foreign_gpu_share}")

    result = run(
        mode=args.mode,
        artifact=args.artifact,
        baseline=args.baseline,
        batches=tuple(args.batches),
        rounds=args.rounds,
        inner=args.inner,
        warmup=args.warmup,
        max_foreign_gpu_share=args.max_foreign_gpu_share,
        bucket_arm=args.bucket_arm,
        bucket_min_threadgroups=args.bucket_min_threadgroups,
    )
    write_json_atomic(args.output, result)
    json.dump({"measured": result["measured"]}, sys.stdout, indent=2)
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
