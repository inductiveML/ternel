"""MLX modules that execute packed TQ1_G128 weights without unpacking them.

A :class:`PackedLinear` holds exactly what the artifact stores -- a ``uint8``
code array and a ``uint16`` array of raw FP16 scale bits -- and hands both
straight to a Metal kernel. There is no dequantised weight, no ``float16`` view,
no cached expansion: the 28 bytes per 128 weights that the file contains are the
same 28 bytes the GPU reads, for the entire life of the process. That is the
whole claim of this project, and it is enforced here rather than asserted, since
these two arrays are the only parameters the module owns.

Two kernels can service a matmul and they win in different regimes, so the
module dispatches on both the batch presented and the shape of the weight:

* ``tq1_matmul`` builds a LUT23 activation table once per 128-weight group and
  then reads each weight byte exactly once. Its cost barely moves with batch
  until the table build stops amortising, which makes it the right kernel for
  token generation and short prompts -- up to 2.70x ``mx.quantized_matmul`` at
  batch 8 on the output head.
* ``tq1_gemm`` decodes weight fragments straight into a lane's registers and
  feeds them to Apple's matrix units. It pays a decode cost the LUT path does
  not, and earns it back once there are enough rows to fill the GPU with
  threadgroups: 1.38x LUT23 at batch 16 and 2.05x at batch 32, on the geometric
  mean over the nine wide-tile shapes.

Nothing here is assumed. Every threshold and every configuration below is read
off a committed sweep, and ``tests/test_mlx_modules.py`` replays those sweeps
through the dispatch functions to check the rule still picks what was measured
to be fastest.

There are two of them, and which one governs which threshold matters.
``results/mlx/a5_op_benchmarks.json`` measures both families over all ten
distinct tensor shapes in the model at eleven batch sizes, 110 paired cases, by
timing each shape on its own in a loop. That settles which *family* runs and how
deep its batch block is. It cannot settle the row block, because its regime
manufactures the parallelism the row block exists to supply: the loop enqueues
many independent copies of one matmul, so a dispatch that offers sixteen
threadgroups is submitted a hundred and twenty-eight at a time and the GPU fills
from the queue. In a forward pass each matmul waits on the one before it and
that same dispatch runs a fifth of the machine alone. The row block is therefore
read off the in-situ sweep, which times every legal tiling for every shape
*inside the model graph* with every other linear stubbed out -- and which ranks
the tilings differently enough that the shape A5 was happy with cost 9.29x the
affine baseline there.

Three caveats govern how those files are read here.

The first is that each configuration was timed against its own interleaved
``mx.quantized_matmul`` run, which is what makes the reported speedups
trustworthy, but it means comparing two of *our* kernels through their speedups
divides by two independently noisy baselines. At the smallest shape, the 48-row
tensor at batch 64, the identical affine workload spanned 2.34x across the
thirty runs of it in that file -- 24.63us to 57.72us for the same work. Every
threshold below is therefore chosen on absolute packed time, and
where absolute time could not separate two configurations the comparison was
re-run head to head as a single paired measurement rather than settled from the
sweep.

The second is that a sweep weights every shape equally and the model does not.
So a rule is scored on the matmul time of a whole forward pass: each measured
shape multiplied by how many tensors of it the frozen audit in
``reports/00_format_audit.md`` counts -- 128 of 17408x5120, 96 of 48x5120, one
output head -- and summed. That reordering matters. The 48-row tensors are 0.23%
to 0.93% of that sum at the batches where their tiling is in question, so a rule
can be badly wrong about them and barely move; the in-situ sweep weights them
correctly by construction, since it times each shape as the model actually
dispatches it, all 96 of them per pass.

The third is that the in-situ sweep is small: nine shapes at four batch widths,
36 points, against A5's 110. Two thresholds fitted to 36 points can fit their
noise. Both are therefore taken on a plateau rather than at an argmin, and the
rule family was refitted four times with one batch width held out and scored on
the width it had not seen. That is the strongest check four widths admit, and
what it says is recorded above :data:`GEMM_MIN_WAVES`. What it does not cover is
prefill wider than 128, where the rule extrapolates -- in the direction the
mechanism argues for, since more batch means more threadgroups and every
threshold here is crossed upward, but extrapolates.
"""

from __future__ import annotations

import math
from dataclasses import replace
from functools import lru_cache

import mlx.core as mx
import mlx.nn as nn

from bonsai_tq1.format import FormatError

from .kernels import (
    INDEX_DTYPE,
    GemmConfig,
    MatmulConfig,
    tq1_gemm,
    tq1_get_rows,
    tq1_matmul,
)
from .layout import PackedTensorLayout

# One thread per output row of a tile, so a tile narrower than this leaves the
# rest of the threadgroup idle. Every tensor in the model tiles at 256 rows
# except ssm_alpha/ssm_beta, whose 48 rows use 18.75% of a threadgroup -- which
# is the entire reason those two behave unlike every other shape below.
LUT23_THREADS = 256

# LUT23 configurations, chosen by how many activation vectors there are to
# amortise the table build over. A batch tile wider than the batch itself does
# the full amount of work for the vectors that are not there, and that is the
# most expensive mistake available here: a fixed tile of 4 costs 3.00x at batch
# 1 and 1.67x at batch 2 in total packed time over the nine wide-tile shapes,
# peaking at 3.46x on the output head. Of those eighteen gaps, seventeen are
# resolvable on p5/p95.
LUT23_SINGLE = MatmulConfig(
    threads=LUT23_THREADS, batch_tile=1, padded_lut=True, safe_clamp=False, k_split=1
)
LUT23_PAIR = MatmulConfig(
    threads=LUT23_THREADS, batch_tile=2, padded_lut=False, safe_clamp=False, k_split=1
)
LUT23_BATCHED = MatmulConfig(
    threads=LUT23_THREADS, batch_tile=4, padded_lut=False, safe_clamp=False, k_split=1
)

# Where the ladder stops. A tile of 8 needs unpadded tables to fit threadgroup
# memory at all and was never the fastest variant at any batch, so the cap is
# between 2 and 4 -- and there the measurement genuinely cannot choose. Timing
# the two against each other directly, 120 paired trials at batches 4 and 8 on
# all nine wide-tile shapes, resolved none of the eighteen points; the tile of 4
# is taken because it is 1.1% ahead on the geometric mean of that run and has
# the lower mean regret over the sweep, not because it was shown faster.
LUT23_MAX_BATCH_TILE = 4

# Prefill tilings, on two axes. All four keep their accumulator fragments per
# lane below the measured register cliff: holding the output block fixed at
# 128x128 and varying only the simdgroup grid over it spread runtime by 8.7x,
# entirely because a lane that must keep 8x4 fragments live spills them out of
# registers where one keeping 4x2 does not.
#
# ``batch_block`` follows the batch, because ``TM = batch_block / (simd_rows *
# 8)`` is how many activation fragments a lane holds and a small batch cannot
# afford a large block. Which of the six kernel forms each tiling runs is
# measured rather than reasoned about, and it tracks that same TM: the batch-16
# wide-block tiling is staged at TM=1, and every TM=4 tiling here builds its
# fragments in registers instead.
#
# ``row_block`` follows the *row count*, which is the axis the rule this
# replaces left out entirely and the whole reason it was wrong. A threadgroup
# owns one ``batch_block`` by ``row_block`` corner of the output, so a dispatch
# offers ``ceil(rows / row_block) * ceil(batch / batch_block)`` threadgroups to
# the 40 cores of this part -- and a 128-row block on the 1024-row k projection
# at batch 16 offers eight of them, running a fifth of the machine.
#
# It has a ceiling as well as that floor, and the two ladders sit on opposite
# sides of it. Occupancy sets the floor; it does not follow that a block wide
# enough to fill the machine is the fastest one, and past some width the wider
# block is simply worse. The large-batch ladder therefore tops out at 64 rows
# where the small-batch ladder goes to 128, because at the batches this model
# actually prefills at, 128 is over the top. Fitted at 511 and 2047 over the
# eight shapes a prefill dispatches, dispatching those two ladders' rungs by
# this rule costs 1.49% and 0.18% more of the summed linear time with a 128-row
# wide rung than with this one, and of sixteen 64-against-128 readings the only
# two whose ranges separate both say 64: the 17408-row gate projection, 42% of
# that time on its own, runs 1053.1-1058.1ms against 1081.9-1087.0ms at batch
# 511, and the 1024-row k projection 17.9-19.3ms against 21.4-25.3ms. Nothing
# separates the other way at either batch.
GEMM_SMALL_BATCH_WIDE_BLOCK = GemmConfig(
    batch_block=16, row_block=128, k_block=64, simd_rows=2, simd_columns=4,
    direct_fragments=False, direct_epilogue=False, threadgroup_table=False,
    safe_clamp=False,
)
GEMM_SMALL_BATCH_NARROW_BLOCK = GemmConfig(
    batch_block=16, row_block=32, k_block=32, simd_rows=2, simd_columns=2,
    direct_fragments=True, direct_epilogue=True, threadgroup_table=True,
    safe_clamp=False,
)
GEMM_LARGE_BATCH_WIDE_BLOCK = GemmConfig(
    batch_block=32, row_block=64, k_block=32, simd_rows=1, simd_columns=2,
    direct_fragments=True, direct_epilogue=True, threadgroup_table=True,
    safe_clamp=False,
)
GEMM_LARGE_BATCH_NARROW_BLOCK = GemmConfig(
    batch_block=32, row_block=32, k_block=32, simd_rows=1, simd_columns=2,
    direct_fragments=True, direct_epilogue=True, threadgroup_table=True,
    safe_clamp=False,
)

# The two ladders, widest row block first. :func:`select_gemm` walks one of them
# and stops at the first rung that both divides the tile and still fills the GPU,
# so the order is load-bearing rather than cosmetic.
GEMM_LADDER_SMALL_BATCH = (GEMM_SMALL_BATCH_WIDE_BLOCK, GEMM_SMALL_BATCH_NARROW_BLOCK)
GEMM_LADDER_LARGE_BATCH = (GEMM_LARGE_BATCH_WIDE_BLOCK, GEMM_LARGE_BATCH_NARROW_BLOCK)

# Where the prefill kernel takes over, and where its second tiling does. Both
# come from the A5 re-run -- thirty configurations over ten shapes at eleven
# batches -- read as absolute packed time rather than as speedups, since ranking
# our own kernels through a per-variant affine baseline divides by two
# independently noisy denominators.
#
# The family crossover is sharp and it is not where the previous rule put it.
# Comparing each family's fastest configuration over the nine wide-tile shapes:
# at batch 8 the prefill kernel is 0.956x LUT23 on the geometric mean, spanning
# 0.78x to 1.12x with six of the nine cases' p5/p95 bands overlapping -- a wash
# this file declines to dispatch on. At batch 16 it is 1.380x, spanning 1.14x to
# 1.57x with nothing overlapping, and at batch 32 it is 2.053x. So 16.
#
# The batch block widens at 32 because TM does: the register form loses at TM=1
# and wins at TM=4, and TM is ``batch_block / (simd_rows * 8)``. At batch 16 the
# staged TM=1 tiling was the fastest configuration A5 measured at every one of
# the nine wide shapes; from 32 up the register form leads.
#
# Those two thresholds survive. What did not is everything A5 said about *which
# row block* to run, and the reason is structural rather than a misreading.
# ``bench_ops`` times a shape by enqueueing many independent copies of the same
# matmul, so a dispatch offering 16 threadgroups is submitted 128 at a time and
# the GPU fills from the queue. A tiling whose only defect is that it cannot
# fill the machine on its own is exactly the tiling that regime scores as fine --
# and in a forward pass, where each matmul waits on the one before it, that same
# tiling costs 9.29x. So the row block below is read off the in-situ sweep
# instead, and A5's authority here is limited to the two thresholds above.
GEMM_MIN_BATCH = 16
GEMM_LARGE_MIN_BATCH = 32

# The GPU this part carries. A dispatch is fed to it in threadgroups, and both
# thresholds below are counted in whole passes over these cores.
GPU_CORES = 40

# How full a dispatch has to be for the prefill kernel to be worth running at
# all, and for its wide row block to beat its narrow one.
#
# Both are read off the in-situ sweep: every legal tiling for every shape in the
# model, timed inside the forward pass at batches 16, 32, 64 and 128, each
# reading taken as the difference between a pass with that one shape live and a
# pass with every linear stubbed -- so what is measured is the shape's own
# contribution rather than a small difference between two large totals.
#
# Below :data:`GEMM_MIN_WAVES` even the narrow block leaves cores idle, and
# LUT23 wins because its parallelism does not come from rows at all: it splits
# the K axis to :data:`SPLIT_K_THREADGROUPS` and manufactures the occupancy the
# GEMM cannot find. That is worth 6.51x on the 1024-row k projection at batch 16
# and 5.30x on the 48-row alpha projection at batch 64, both of which the
# previous rule handed to a GEMM running an eighth of the machine. It also
# subsumes the narrow-tile bucket that rule carried: 48 rows yield two row
# blocks, so a narrow tensor cannot reach this floor until batch 1280, and no
# special case for it is needed.
#
# Above :data:`GEMM_WIDE_BLOCK_WAVES` there are enough threadgroups that the
# wide block's arithmetic intensity binds instead, and it takes over. Between
# them the narrow block runs, which is where the previous rule lost the most in
# bulk: at batch 32 the 12288-row q projection went from 3.45x affine to 2.03x,
# and at batch 16 the two 5120-row shapes from 2.63x and 2.65x to 1.17x and
# 1.10x.
#
# Both are taken on a plateau rather than at an argmin. Scored as regret against
# the best tiling measured at each of the 36 points, the split-K floor is flat
# at 1.042 anywhere from 1 to 4 waves, and the wide-block step is flat at 1.042
# across 2.5 and 3 waves -- which dispatch identically on all 36 points -- with
# clear walls either side, 1.087 at 2 waves and 1.111 at 4.
#
# Fitting two thresholds to 36 points can fit their noise, so the family was
# fitted four more times with one batch width held out and scored on the width
# it had not seen. All four folds chose this rule, and its held-out regret came
# to 1.012, 1.061, 1.029 and 1.032 against the previous rule's 1.640, 1.288,
# 1.115 and 1.045 at the same widths.
#
# The null model is worth stating because it is close: running the narrow block
# for everything, keeping only the split-K floor, scores 1.134 on its worst
# width against these two thresholds' 1.042. The wide block is what the second
# threshold buys, and without it the 17408-row and 248320-row tensors -- most of
# the model's weight -- give up 25% and 12% at batch 16.
GEMM_MIN_WAVES = 2
GEMM_WIDE_BLOCK_WAVES = 3

# One thread per output column of a decoded embedding row. The lookup is a pure
# gather with no cooperation between threads, so this only has to be a legal
# threadgroup width; the kernel narrows it to the column count when a row is
# shorter than that.
GET_ROWS_THREADS = 256

# How many threadgroups a decode dispatch is split until it has. LUT23 walks a
# tensor's groups strictly in series inside each threadgroup, so a tensor whose
# rows only yield a few tiles runs one short chain on an otherwise idle machine.
# That is invisible to a benchmark loop, whose independent calls overlap and
# fill the GPU between them, and it is the whole cost in decode, where 497
# dispatches form one read-after-write chain and each waits alone.
#
# The A6 sweep measured it directly, timing every legal split of every shape in
# the model both overlapped and in a forced dependency chain. Chained, the
# unsplit kernel pays 1.01x to 8.32x its own overlapped time, worst on the
# narrow tensors that fill a handful of threadgroups; splitting to this ceiling
# leaves 0.76x to 1.11x, and one decoded token's 497 matmuls fall from 40.20ms
# to 19.80ms -- from 2.00x ``mx.quantized_matmul`` to 0.99x of it.
#
# 800 is twenty threadgroups per core on this 40-core part. It is chosen out of
# sample, which the number it replaced was not: the sweep has been run twice,
# and past the point where the serialisation factor reaches 1.0 the curve is
# flat enough that whichever ceiling wins on one run generally loses on the
# other. Scored against the run that did not pick it, every ceiling from 320 to
# 2560 gives up between 2.6% and 3.6% of a decoded token over knowing each
# shape's argmin in advance, so the first sweep's apparent 1.8% at 320 was a fit
# to its own noise. 800 is taken because it is the one value that beats 320 on
# both runs and on both measures -- 1.58%/2.71% of the token against 1.84%/3.32%,
# and 9.1%/9.5% on the worst single shape against 16.5%/15.7% -- while needing
# 23 kernel instantiations over the swept points where 320 needs 27, since a
# wider budget lands more shapes on the same divisors.
#
# The costs that grow with the ceiling are real but small at this size: one
# extra reduction dispatch, an fp32 partial buffer of k*batch*rows, and
# threadgroups competing with whatever else is in flight. They are what keeps
# this from being uncapped rather than what sets it to 800.
#
# No separate bound on that partial buffer is needed: a tensor with enough rows
# to make it large has enough row tiles to exhaust the ceiling on its own, so
# the widest split lands on the narrowest tensors. lm_head's 970 tiles take
# k=1 and allocate nothing.
SPLIT_K_THREADGROUPS = 800

# Ten distinct layouts times the batches a decode or a short prefill presents.
SELECT_CACHE_ENTRIES = 256


def threadgroups(layout: PackedTensorLayout, batch: int, config: GemmConfig) -> int:
    """How many threadgroups dispatching ``config`` on this shape would offer.

    One threadgroup owns one ``batch_block`` by ``row_block`` corner of the
    output, so this is the whole supply of parallelism the GPU gets -- and it is
    the quantity :func:`select_gemm` decides on, which is why it is a named
    function rather than an expression buried in the rule.
    """
    return math.ceil(layout.rows / config.row_block) * math.ceil(batch / config.batch_block)


def select_gemm(layout: PackedTensorLayout, batch: int) -> GemmConfig | None:
    """The prefill tiling for ``batch`` activation rows, or ``None`` for LUT23.

    The batch decides the batch block, off the A5 sweeps. The *row count*
    decides the row block, off the in-situ sweep, and that is the change: the
    rule this replaces read only the tile and the batch, so it dispatched the
    same 128-row block to a 1024-row projection and a 248320-row output head.
    Those two offer 8 and 1940 threadgroups to 40 cores, and the first of them
    cost 9.29x the affine baseline in a real forward pass while the second cost
    1.11x. ``tests/test_mlx_modules.py`` replays both sweeps through this
    function, so the rule is checked against the measurements rather than
    described by them.

    What is measured is occupancy, so that is what is computed: walk the ladder
    from the widest row block down, and take the first rung that both divides
    the row tile and still offers :data:`GEMM_WIDE_BLOCK_WAVES` passes over the
    GPU. If no rung does, the narrowest legal one runs -- unless even it falls
    below :data:`GEMM_MIN_WAVES`, at which point the shape cannot fill the
    machine along its rows at any tiling and LUT23's K-axis split takes it.

    ``None`` also comes back when no tiling can run the tensor at all: a row
    block straddling two row tiles would read the second tile's codes through
    the first tile's base pointer, so the pairing is refused, not approximated.
    """
    if batch < GEMM_MIN_BATCH:
        return None
    ladder = (
        GEMM_LADDER_LARGE_BATCH if batch >= GEMM_LARGE_MIN_BATCH
        else GEMM_LADDER_SMALL_BATCH
    )
    legal = tuple(
        config for config in ladder
        if layout.tiles == 1 or layout.tile % config.row_block == 0
    )
    if not legal:
        return None
    if threadgroups(layout, batch, legal[-1]) < GEMM_MIN_WAVES * GPU_CORES:
        return None
    for config in legal:
        if threadgroups(layout, batch, config) >= GEMM_WIDE_BLOCK_WAVES * GPU_CORES:
            return config
    return legal[-1]


def select_lut23_tile(layout: PackedTensorLayout, batch: int) -> MatmulConfig:
    """The LUT23 batch tiling for ``batch`` activation rows, before any split.

    The batch tile tracks the batch up to :data:`LUT23_MAX_BATCH_TILE`, snapped
    down to a tile that was actually measured and compiled. A narrow tile is
    held at one vector regardless: with 48 rows the threadgroup is already
    mostly idle, and a batch tile buys parallelism it has no threads to spend
    while multiplying the table footprint by the tile width.

    Separate from :func:`select_lut23` because the two knobs were measured by
    two different sweeps and each is replayed against its own file: A5 swept the
    batch tile with the K axis whole, A6 swept the split at the batch tiles this
    function returns. They meet only through the grid size, which
    :func:`select_lut23` computes from this function's answer rather than
    assuming.
    """
    if layout.tile < LUT23_THREADS or batch < 2:
        return LUT23_SINGLE
    return LUT23_PAIR if batch < LUT23_MAX_BATCH_TILE else LUT23_BATCHED


def largest_split(groups_per_row: int, *, threadgroups: int) -> int:
    """The largest divisor of ``groups_per_row`` that keeps the grid under target.

    A divisor, not a ceiling division: a ragged split leaves one threadgroup
    holding a full chunk of groups while another holds none, and since the
    dispatch ends when its slowest threadgroup does, the extra threadgroups
    would buy nothing.

    The search always succeeds -- one divides every group count, and the budget
    never falls below one -- so a tensor whose row tiles already reach the
    ceiling comes back unsplit rather than rejected.
    """
    budget = max(1, min(SPLIT_K_THREADGROUPS // max(threadgroups, 1), groups_per_row))
    return next(split for split in range(budget, 0, -1) if groups_per_row % split == 0)


@lru_cache(maxsize=SELECT_CACHE_ENTRIES)
def select_lut23(layout: PackedTensorLayout, batch: int) -> MatmulConfig:
    """The LUT23 configuration for ``batch`` activation rows, split included.

    Cached because it is consulted once per matmul and a decoded token issues
    497 of them, while the model only ever presents ten distinct layouts.
    """
    base = select_lut23_tile(layout, batch)
    threadgroups = layout.tiles * math.ceil(max(batch, 1) / base.batch_tile)
    split = largest_split(layout.groups_per_row, threadgroups=threadgroups)
    return base if split == 1 else replace(base, k_split=split)


def packed_matmul(
    codes: mx.array, scales: mx.array, x: mx.array, *, layout: PackedTensorLayout
) -> mx.array:
    """``x @ W.T`` for a packed ``W``, choosing the kernel by activation rows.

    Shared by :class:`PackedLinear` and by the tied-head path on
    :class:`PackedEmbedding`, so the two cannot drift apart.
    """
    if x.shape[-1] != layout.columns:
        raise FormatError(
            f"activation has {x.shape[-1]} columns, weight expects {layout.columns}"
        )
    flat = x.reshape(-1, layout.columns)
    config = select_gemm(layout, flat.shape[0])
    if config is not None:
        out = tq1_gemm(
            codes,
            scales,
            flat,
            groups_per_row=layout.groups_per_row,
            tile=layout.tile,
            config=config,
        )
    else:
        out = tq1_matmul(
            codes,
            scales,
            flat,
            groups_per_row=layout.groups_per_row,
            tile=layout.tile,
            config=select_lut23(layout, flat.shape[0]),
        )
    return out.reshape(*x.shape[:-1], layout.rows)


class PackedLinear(nn.Module):
    """A bias-free linear layer whose weight stays in TQ1_G128 for its lifetime.

    ``codes`` and ``scales`` are created empty and unevaluated. MLX arrays are
    lazy, so the placeholders cost nothing as long as the real ones arrive --
    via ``load_weights`` -- before anything forces evaluation. Loading with
    ``strict=True`` is what guarantees they do.
    """

    def __init__(self, layout: PackedTensorLayout) -> None:
        super().__init__()
        self.layout = layout
        self.codes = mx.zeros(layout.codes_shape, dtype=mx.uint8)
        self.scales = mx.zeros(layout.scales_shape, dtype=mx.uint16)

    @classmethod
    def from_linear(cls, linear: nn.Linear) -> "PackedLinear":
        """Take the shape from a built ``nn.Linear`` and drop its random weight.

        The shape arithmetic for this architecture lives in mlx-lm and is not
        repeated here; this reads the answer off the module mlx-lm built. The
        discarded weight is never evaluated, so it never occupies memory.
        """
        if "bias" in linear:
            raise FormatError("a packed linear layer cannot carry a bias")
        rows, columns = linear.weight.shape
        return cls(PackedTensorLayout.for_tensor(rows, columns))

    def _extra_repr(self) -> str:
        return (
            f"{self.layout.columns} -> {self.layout.rows}, tile={self.layout.tile}, "
            f"{self.layout.bits_per_weight:.3f} bits/weight"
        )

    def __call__(self, x: mx.array) -> mx.array:
        return packed_matmul(self.codes, self.scales, x, layout=self.layout)


class PackedEmbedding(nn.Module):
    """A token embedding decoded one row at a time, straight out of the codes.

    A gather over a packed table cannot reuse the matmul path: only the rows the
    batch actually names are wanted, and decoding the whole table to take a few
    hundred rows would materialise exactly the model-sized array this project
    exists to avoid. ``tq1_get_rows`` decodes precisely the requested rows.

    Unlike every other module here the output dtype cannot be inferred from an
    input -- the input is an index vector -- and there is no float weight to
    read it from either. It is therefore stated explicitly at construction,
    from the dtype the rest of the graph runs in.
    """

    def __init__(self, layout: PackedTensorLayout, *, dtype: mx.Dtype) -> None:
        super().__init__()
        self.layout = layout
        self.dtype = dtype
        self.codes = mx.zeros(layout.codes_shape, dtype=mx.uint8)
        self.scales = mx.zeros(layout.scales_shape, dtype=mx.uint16)

    @classmethod
    def from_embedding(cls, embedding: nn.Embedding, *, dtype: mx.Dtype) -> "PackedEmbedding":
        rows, columns = embedding.weight.shape
        return cls(PackedTensorLayout.for_tensor(rows, columns), dtype=dtype)

    def _extra_repr(self) -> str:
        return f"{self.layout.rows} x {self.layout.columns}, tile={self.layout.tile}, {self.dtype}"

    def __call__(self, indices: mx.array) -> mx.array:
        """Gather embedding rows for ``indices`` of any shape, as ``nn.Embedding`` does.

        mlx-lm hands this a ``(batch, sequence)`` array of the tokenizer's own
        integer dtype, while the kernel takes a flat ``uint32`` list, so the
        reshape and the cast happen here rather than being imposed on callers.
        """
        flat = indices.reshape(-1).astype(INDEX_DTYPE)
        rows = tq1_get_rows(
            self.codes,
            self.scales,
            flat,
            groups_per_row=self.layout.groups_per_row,
            tile=self.layout.tile,
            dtype=self.dtype,
            threads=GET_ROWS_THREADS,
            safe_clamp=False,
        )
        return rows.reshape(*indices.shape, self.layout.columns)

    def as_linear(self, x: mx.array) -> mx.array:
        """The tied-output-head path: the embedding table used as a matmul."""
        return packed_matmul(self.codes, self.scales, x, layout=self.layout)
