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
off ``results/mlx/a5_op_benchmarks.json`` -- both families measured over all ten
distinct tensor shapes in the model at eleven batch sizes, 110 paired cases --
and ``tests/test_mlx_modules.py`` replays that file through the dispatch
functions to check the rule still picks what was measured to be fastest.

Two caveats govern how that file is read here.

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

The second is that the file weights every shape equally and the model does not.
So the rule was picked on the predicted matmul time of a whole forward pass:
each measured shape multiplied by how many tensors of it the frozen audit in
``reports/00_format_audit.md`` counts -- 128 of 17408x5120, 96 of 48x5120, one
output head -- and summed. That reordering matters: the 48-row tensors are
0.23% to 0.93% of that sum at the batches where their tiling is even in
question, and the two candidate rules that differ only in which tiling they take
differ by 14 points of worst per-case regret (28.54% against 14.52%) and by 0.04
points of forward-pass geometric mean (1.0304 against 1.0301). Against a
per-case oracle the shipped rule predicts 1.030x on the geometric mean over the
eleven batches; the rule it replaces predicts 1.426x and LUT23 everywhere
predicts 1.570x.
"""

from __future__ import annotations

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
    threads=LUT23_THREADS, batch_tile=1, padded_lut=True, safe_clamp=False
)
LUT23_PAIR = MatmulConfig(
    threads=LUT23_THREADS, batch_tile=2, padded_lut=False, safe_clamp=False
)
LUT23_BATCHED = MatmulConfig(
    threads=LUT23_THREADS, batch_tile=4, padded_lut=False, safe_clamp=False
)

# Where the ladder stops. A tile of 8 needs unpadded tables to fit threadgroup
# memory at all and was never the fastest variant at any batch, so the cap is
# between 2 and 4 -- and there the measurement genuinely cannot choose. Timing
# the two against each other directly, 120 paired trials at batches 4 and 8 on
# all nine wide-tile shapes, resolved none of the eighteen points; the tile of 4
# is taken because it is 1.1% ahead on the geometric mean of that run and has
# the lower mean regret over the sweep, not because it was shown faster.
LUT23_MAX_BATCH_TILE = 4

# Prefill tilings. All three keep their accumulator fragments per lane below the
# measured register cliff: holding the output block fixed at 128x128 and varying
# only the simdgroup grid over it spread runtime by 8.7x, entirely because a
# lane that must keep 8x4 fragments live spills them out of registers where one
# keeping 4x2 does not.
#
# Which of the six kernel forms each one runs is measured, not reasoned about.
# The two that serve the wide tile are not even the same form: the batch-16
# tiling is staged at TM=1, the batch-32-and-up tiling builds its fragments in
# registers at TM=4 and reads its trit table out of threadgroup memory. That
# split is the same one the tiling sweep found -- the register form loses at
# TM=1 and wins at TM=4 -- arriving as a dispatch boundary because TM is
# batch_block / (simd_rows * 8) and a small batch cannot afford a large batch
# block.
GEMM_SMALL_BATCH = GemmConfig(
    batch_block=16, row_block=128, k_block=64, simd_rows=2, simd_columns=4,
    direct_fragments=False, direct_epilogue=False, threadgroup_table=False,
    safe_clamp=False,
)
GEMM_LARGE_BATCH = GemmConfig(
    batch_block=32, row_block=64, k_block=32, simd_rows=1, simd_columns=2,
    direct_fragments=True, direct_epilogue=True, threadgroup_table=True,
    safe_clamp=False,
)
GEMM_NARROW_TILE = GemmConfig(
    batch_block=128, row_block=32, k_block=32, simd_rows=4, simd_columns=2,
    direct_fragments=True, direct_epilogue=True, threadgroup_table=False,
    safe_clamp=False,
)

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
# The second threshold is a tiling change, not a family change. At batch 16 the
# staged TM=1 tiling above was the fastest configuration measured at every one
# of the nine wide shapes; from batch 32 up the register TM=4 tiling costs a
# mean 2.56% against the per-case best over all 54 wide cases, worst 14.52%.
# Holding the narrow rule fixed and running one of them across the whole range
# instead costs mean per-case regret of 4.38% (worst 39.66%) for the TM=4 tiling
# and 20.03% (worst 80.36%) for the TM=1 one, against 2.43% and 14.52% for the
# two buckets together -- so the second bucket is worth its JIT instantiation
# and a third is not: a bucket added at 512 improved the predicted forward pass
# by 0.1%.
#
# Those regrets were scored on one sweep, later found to have been captured on a
# contended machine. They survive it because every candidate rule was scored on
# the same file, so a distortion they share cancels out of the comparison, and
# because the margins are far wider than the distortion: re-scoring the shipped
# rule on the floor of three sweeps moves its mean regret by 0.45 points, against
# the 17.6 that separate it from the TM=1 rule.
GEMM_MIN_BATCH = 16
GEMM_LARGE_MIN_BATCH = 32

# The narrow-tile tensors cross over four times later, and not because the
# prefill kernel does well on them -- 48 rows fill one or two threadgroups on a
# 40-core GPU whatever the tiling, so every prefill configuration there is
# launch-bound and the sweep resolves none of *those* against each other, only
# the family question below. It is that LUT23 is
# measurably ahead through batch 32 (0.82x, non-overlapping) and still nominally
# ahead at 40 (0.91x, overlapping); from 64 up the prefill kernel leads at every
# batch measured, reaching 2.31x at 512.
#
# Which tiling it takes there moves the predicted forward pass by 0.04 points of
# geometric mean, since the 96 tensors of this shape are 0.23% to 0.93% of the
# matmul time the model spends at those batches. This one is taken because among
# the two shipped candidates it was fastest at the narrow tile at batches 64,
# 128 and 200; at 512 :data:`GEMM_LARGE_BATCH` was, by 4%, and none of the four
# gaps resolves.
NARROW_TILE_GEMM_MIN_BATCH = 64

# One thread per output column of a decoded embedding row. The lookup is a pure
# gather with no cooperation between threads, so this only has to be a legal
# threadgroup width; the kernel narrows it to the column count when a row is
# shorter than that.
GET_ROWS_THREADS = 256


def select_gemm(layout: PackedTensorLayout, batch: int) -> GemmConfig | None:
    """The prefill tiling for ``batch`` activation rows, or ``None`` for LUT23.

    Read off ``results/mlx/a5_op_benchmarks.json``, which measured both families
    over every row count in the model at eleven batch sizes.
    ``tests/test_mlx_modules.py`` replays that file through this function, so
    the rule is checked against the measurement rather than described by it.

    Only the tile and the batch decide. The row count does not appear, which is
    a change: the previous rule held tensors under 5120 rows on LUT23 at every
    batch, on the reasoning that 1024 rows fill 8 threadgroups and the prefill
    kernel had never won there. It wins there now -- what this rule dispatches
    beats the fastest LUT23 by 1.14x at batch 16 and 2.08x at 512, with
    non-overlapping p5/p95 bands at every batch from 16 up -- because the tilings
    that win at those batches use 64- and 128-row blocks rather than 128 wide by
    128 tall, so the same 1024 rows yield 8 to 16 row blocks against a batch
    split several ways as well.

    ``None`` also comes back when no tiling can run the tensor at all: a row
    block straddling two row tiles would read the second tile's codes through
    the first tile's base pointer, so the pairing is refused, not approximated.
    """
    if layout.tile < LUT23_THREADS:
        config = GEMM_NARROW_TILE if batch >= NARROW_TILE_GEMM_MIN_BATCH else None
    elif batch >= GEMM_LARGE_MIN_BATCH:
        config = GEMM_LARGE_BATCH
    elif batch >= GEMM_MIN_BATCH:
        config = GEMM_SMALL_BATCH
    else:
        config = None
    if config is None:
        return None
    if layout.tiles > 1 and layout.tile % config.row_block:
        return None
    return config


def select_lut23(layout: PackedTensorLayout, batch: int) -> MatmulConfig:
    """The LUT23 configuration for ``batch`` activation rows.

    The batch tile tracks the batch up to :data:`LUT23_MAX_BATCH_TILE`, snapped
    down to a tile that was actually measured and compiled. A narrow tile is
    held at one vector regardless: with 48 rows the threadgroup is already
    mostly idle, and a batch tile buys parallelism it has no threads to spend
    while multiplying the table footprint by the tile width.
    """
    if layout.tile < LUT23_THREADS or batch < 2:
        return LUT23_SINGLE
    return LUT23_PAIR if batch < LUT23_MAX_BATCH_TILE else LUT23_BATCHED


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
