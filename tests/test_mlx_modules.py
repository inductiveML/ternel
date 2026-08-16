"""Graph gates: the swap reaches every weight, and no float copy survives it.

The kernels are proven elsewhere. What is at stake here is the claim that a
whole *model* runs packed -- that walking mlx-lm's Qwen3.5 graph replaces every
matmul weight rather than most of them, that the artifact's key convention is
the one ``load_weights(strict=True)`` demands, and that after the swap the
parameter tree contains no float matrix at all. The last of those is the
project's central assertion, and it is checkable exactly: a model-sized float
copy would have to appear as a parameter to be reachable, so its absence from
``model.parameters()`` is proof rather than evidence.
"""

from __future__ import annotations

import json
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import numpy as np
import pytest
from mlx.utils import tree_flatten

from bonsai_tq1.format import BLOCK_SIZE, FormatError, encode_tq1_blocks
from ternel_mlx.kernel_gate import ORACLE_RELATIVE_TOLERANCE, compare
from ternel_mlx.kernels import INDEX_DTYPE, GemmConfig
from ternel_mlx.layout import (
    CODES_SUFFIX,
    MODEL_LINEAR_SHAPES,
    SCALES_SUFFIX,
    PackedTensorLayout,
)
from ternel_mlx.modules import (
    GEMM_LADDER_LARGE_BATCH,
    GEMM_LADDER_SMALL_BATCH,
    GEMM_LARGE_BATCH_NARROW_BLOCK,
    GEMM_LARGE_BATCH_WIDE_BLOCK,
    GEMM_LARGE_MIN_BATCH,
    GEMM_MIN_BATCH,
    GEMM_MIN_WAVES,
    GEMM_SMALL_BATCH_NARROW_BLOCK,
    GEMM_SMALL_BATCH_WIDE_BLOCK,
    GEMM_WIDE_BLOCK_WAVES,
    GPU_CORES,
    LUT23_BATCHED,
    LUT23_PAIR,
    LUT23_SINGLE,
    LUT23_THREADS,
    PackedEmbedding,
    PackedLinear,
    packed_matmul,
    select_gemm,
    select_lut23,
    select_lut23_tile,
    threadgroups,
)
from ternel_mlx.packed_model import (
    ACTIVATION_DTYPE_KEY,
    Model,
    ModelArgs,
    replace_with_packed,
    resolve_activation_dtype,
)
from ternel_mlx.packing import pack_blocks
from ternel_mlx.reference import get_rows_reference, oracle_matmul, restored_matmul

# A miniature of the real hybrid: eight layers on a period of four, so two are
# full attention and six recurrent, which is the same interleaving the 64-layer
# model has. Small enough to build in milliseconds, structurally identical.
LAYERS = 8
FULL_ATTENTION_INTERVAL = 4
HIDDEN = 512
VOCAB = 2048

# Linear layers each layer kind contributes. Recurrent: in_proj_qkv, in_proj_z,
# in_proj_b, in_proj_a, out_proj plus the three MLP projections. Full attention:
# q, k, v, o plus the same three.
RECURRENT_LINEARS = 8
FULL_ATTENTION_LINEARS = 7


def text_config(**overrides: object) -> dict[str, object]:
    config: dict[str, object] = {
        "model_type": "qwen3_5",
        "hidden_size": HIDDEN,
        "intermediate_size": 2 * HIDDEN,
        "num_hidden_layers": LAYERS,
        "num_attention_heads": 8,
        "num_key_value_heads": 2,
        "vocab_size": VOCAB,
        "linear_num_value_heads": 16,
        "linear_num_key_heads": 4,
        "linear_key_head_dim": 128,
        "linear_value_head_dim": 128,
        "full_attention_interval": FULL_ATTENTION_INTERVAL,
        "head_dim": 128,
        "tie_word_embeddings": False,
        ACTIVATION_DTYPE_KEY: "bfloat16",
    }
    config.update(overrides)
    return config


def build_model(**overrides: object) -> Model:
    return Model(ModelArgs.from_dict({"model_type": "qwen3_5", "text_config": text_config(**overrides)}))


def packed_modules(model: nn.Module) -> dict[str, PackedLinear | PackedEmbedding]:
    return {
        path: module
        for path, module in model.named_modules()
        if isinstance(module, (PackedLinear, PackedEmbedding))
    }


def random_packed(rows: int, columns: int, seed: int):
    """A real packed tensor: encoded through the frozen block encoder, then tiled."""
    layout = PackedTensorLayout.for_tensor(rows, columns)
    generator = np.random.default_rng(seed)
    scale_bits = (
        generator.normal(0.0, 0.05, size=layout.groups)
        .astype(np.float16)
        .view(np.uint8)
        .reshape(-1, 2)
    )
    trits = generator.integers(0, 3, size=(layout.groups, BLOCK_SIZE), dtype=np.uint8)
    blocks = encode_tq1_blocks(scale_bits, trits).reshape(rows, layout.groups_per_row, -1)
    codes, scales = pack_blocks(blocks, layout=layout)
    return layout, codes, scales


def load_packed(module: PackedLinear | PackedEmbedding, codes: np.ndarray, scales: np.ndarray) -> None:
    module.load_weights([(CODES_SUFFIX, mx.array(codes)), (SCALES_SUFFIX, mx.array(scales))])


# --------------------------------------------------------------------------
# The swap
# --------------------------------------------------------------------------


def test_every_matmul_weight_in_the_graph_is_replaced() -> None:
    model = build_model()

    full_attention = LAYERS // FULL_ATTENTION_INTERVAL
    recurrent = LAYERS - full_attention
    expected_linears = (
        recurrent * RECURRENT_LINEARS
        + full_attention * FULL_ATTENTION_LINEARS
        + 1  # lm_head, untied in this configuration
    )
    assert model.packed_module_counts == {"linear": expected_linears, "embedding": 1}

    leftovers = [
        path
        for path, module in model.named_modules()
        if isinstance(module, (nn.Linear, nn.Embedding))
        and not isinstance(module, (PackedLinear, PackedEmbedding))
    ]
    assert leftovers == []
    assert len(packed_modules(model)) == expected_linears + 1


def test_a_tied_head_leaves_one_embedding_and_no_lm_head() -> None:
    """Tying is mlx-lm's decision; the swap must survive either answer."""
    untied = build_model(tie_word_embeddings=False)
    tied = build_model(tie_word_embeddings=True)
    assert tied.packed_module_counts["embedding"] == 1
    assert tied.packed_module_counts["linear"] == untied.packed_module_counts["linear"] - 1
    assert not any(path.endswith("lm_head") for path in packed_modules(tied))


def test_no_float_matrix_parameter_survives_the_swap() -> None:
    """The anti-cheat assertion: a dequantised weight would have to be a parameter.

    Everything left in float is a norm, a bias or the small depthwise conv1d.
    What separates those from a weight is not their size but their *shape*: a
    matmul weight is quadratic in the model width, while every legitimate float
    parameter is a vector, or a vector with a couple of tiny trailing axes. So
    the bound below is on element count against the largest axis, which holds at
    any model size rather than at the one this fixture happens to use.
    """
    model = build_model()
    packed = 0
    for name, value in tree_flatten(model.parameters()):
        if value.dtype in (mx.uint8, mx.uint16):
            packed += 1
            continue
        assert value.size <= 8 * max(value.shape), (
            f"{name} is a {value.shape} float array that grows with the square of the "
            "model width; the swap missed a weight"
        )
    assert packed == 2 * len(packed_modules(model))


# --------------------------------------------------------------------------
# The artifact contract
# --------------------------------------------------------------------------


def test_packed_keys_are_exactly_two_per_packed_module() -> None:
    """What the converter must name, derived from the modules rather than a list."""
    model = build_model()
    modules = packed_modules(model)
    expected = {
        f"{path}.{suffix}" for path in modules for suffix in (CODES_SUFFIX, SCALES_SUFFIX)
    }
    found = {
        name
        for name, _ in tree_flatten(model.parameters())
        if name.endswith((f".{CODES_SUFFIX}", f".{SCALES_SUFFIX}"))
    }
    assert found == expected
    assert all(path.startswith("language_model.") for path in modules)


def test_a_strict_load_of_artifact_shaped_weights_succeeds() -> None:
    """An artifact carrying exactly these names, shapes and dtypes must load.

    ``strict=True`` is what makes the lazy placeholders safe: it fails unless
    every parameter is supplied, so no zero-filled stand-in can survive into
    generation.
    """
    model = build_model()
    weights = []
    for name, value in tree_flatten(model.parameters()):
        weights.append((name, mx.zeros(value.shape, dtype=value.dtype)))
    model.load_weights(weights, strict=True)

    embedding = packed_modules(model)["language_model.model.embed_tokens"]
    assert embedding.codes.dtype == mx.uint8
    assert embedding.scales.dtype == mx.uint16
    assert tuple(embedding.codes.shape) == embedding.layout.codes_shape
    assert tuple(embedding.scales.shape) == embedding.layout.scales_shape


def test_a_strict_load_rejects_a_mis_shaped_packed_tensor() -> None:
    model = build_model()
    weights = []
    for name, value in tree_flatten(model.parameters()):
        shape = value.shape
        if name == "language_model.model.embed_tokens.codes":
            shape = (shape[0], shape[1], shape[2], shape[3] // 2)
        weights.append((name, mx.zeros(shape, dtype=value.dtype)))
    with pytest.raises(ValueError, match="[Ss]hape"):
        model.load_weights(weights, strict=True)


def test_a_strict_load_rejects_a_missing_packed_tensor() -> None:
    model = build_model()
    weights = [
        (name, mx.zeros(value.shape, dtype=value.dtype))
        for name, value in tree_flatten(model.parameters())
        if name != "language_model.model.embed_tokens.scales"
    ]
    with pytest.raises(ValueError, match="[Mm]issing"):
        model.load_weights(weights, strict=True)


# --------------------------------------------------------------------------
# Refusals
# --------------------------------------------------------------------------


def test_the_activation_dtype_must_be_stated_not_guessed() -> None:
    config = text_config()
    del config[ACTIVATION_DTYPE_KEY]
    with pytest.raises(FormatError, match=ACTIVATION_DTYPE_KEY):
        resolve_activation_dtype(config)


@pytest.mark.parametrize("name", ("float64", "int8", "bf16", ""))
def test_an_unsupported_activation_dtype_is_refused(name: str) -> None:
    with pytest.raises(FormatError, match="unsupported activation dtype"):
        resolve_activation_dtype(text_config(**{ACTIVATION_DTYPE_KEY: name}))


@pytest.mark.parametrize(
    ("name", "expected"),
    (("bfloat16", mx.bfloat16), ("float16", mx.float16), ("float32", mx.float32)),
)
def test_each_supported_activation_dtype_resolves(name: str, expected: mx.Dtype) -> None:
    assert resolve_activation_dtype(text_config(**{ACTIVATION_DTYPE_KEY: name})) is expected


def test_a_linear_with_a_bias_cannot_be_packed() -> None:
    """TQ1_G128 stores a scale and trits and nothing else; a bias has no home."""
    with pytest.raises(FormatError, match="cannot carry a bias"):
        PackedLinear.from_linear(nn.Linear(BLOCK_SIZE, BLOCK_SIZE, bias=True))


def test_replacement_reports_exactly_what_it_replaced() -> None:
    """The count is what lets ``Model.__init__`` refuse a graph it did not expect."""
    model = nn.Sequential(nn.Linear(BLOCK_SIZE, BLOCK_SIZE, bias=False))
    assert replace_with_packed(model, dtype=mx.float32) == {"linear": 1, "embedding": 0}
    assert isinstance(model.layers[0], PackedLinear)


def test_a_column_mismatch_is_refused_before_the_kernel() -> None:
    layout, codes, scales = random_packed(rows=BLOCK_SIZE, columns=BLOCK_SIZE, seed=1)
    with pytest.raises(FormatError, match="activation has"):
        packed_matmul(
            mx.array(codes),
            mx.array(scales),
            mx.zeros((2, 2 * BLOCK_SIZE), dtype=mx.float32),
            layout=layout,
        )


# --------------------------------------------------------------------------
# Kernel selection
# --------------------------------------------------------------------------


SWEEP_DIRECTORY = Path("results/mlx")
SWEEP_GLOB = "a5_op_benchmarks*.json"

# The dtype the dispatch rule is gated in, which is the dtype the model runs.
GATED_ACTIVATION_DTYPE = "bfloat16"

# How many runs of the identical protocol -- same seed, same trials per arm, the
# same points -- the floor is taken over. Three, because no single sweep on this
# machine can rank one of our kernels against another; see ``fastest_seen``.
SWEEPS_REQUIRED = 3


def gated_sweeps() -> tuple[Path, ...]:
    """Every A5 sweep measured in the dtype the rule is gated in.

    Selected by the dtype each file records, not by its name. A filename glob
    stood here while every sweep ran one protocol, and stopped being correct the
    moment the activation dtype became an input: it is a kernel template
    parameter, so two dtypes are two different sets of compiled kernels and a
    floor taken across them is a floor over nothing.

    Files recording no dtype are excluded rather than assumed into a group. They
    predate the flag that now requires one, and what they were measured in is
    Stage A's third correction rather than something to infer here.
    """
    found: list[Path] = []
    for path in sorted(SWEEP_DIRECTORY.glob(SWEEP_GLOB)):
        document = json.loads(path.read_text())
        if "activation_dtype" not in document:
            continue
        if document["activation_dtype"] == GATED_ACTIVATION_DTYPE:
            found.append(path)
    return tuple(found)

# What the dispatch rule may give up against the fastest configuration measured
# for each case, judged on absolute packed time.
#
# The mean is the real bound: it is an average over the 78 cases this file can
# still score, so the per-case drift discussed below averages out of it, and the
# shipped rule sits at 2.2% with 27 of the 78 exactly right. It read 2.1% over
# all 99 when the previous rule was scored on the whole sweep, and the scope is
# what changed rather than the bounds -- neither number here has been moved.
#
# Which 21 points dropped out, and why, is argued at :func:`occupancy_refused`
# and asserted rather than described. It has to be argued because it is not a
# neutral exclusion: on the eleven the rule refuses on occupancy, A5 ranks a
# prefill tiling first and the rule takes LUT23, which by this file's estimator
# alone is 0.2% to 178% given up. That is the disagreement the in-situ sweep
# exists to settle, and it settles it the other way -- the tilings A5 prefers on
# those two shapes cost 12.28x and 20.73x the affine baseline inside a real
# forward pass against LUT23's 1.89x and 3.91x. Both files are right about what
# they measured; only one of them measured the regime the model runs in.
#
# The per-case maximum is deliberately loose, because this file cannot resolve
# one of our configurations against another. Each was timed in its own paired
# run against the affine baseline, so two configurations' packed times come from
# runs minutes apart and slow drift separates them on its own. An earlier attempt
# to settle such a case by re-running two configurations as the two arms of a
# single paired comparison -- 120 trials, batches 4 and 8, all nine wide-tile
# shapes -- resolved none of the eighteen points and reversed the sign of the
# worst one. So this bound is set to catch a structural mistake rather than to
# arbitrate noise: every genuine dispatch error available here is far larger, a
# fixed batch tile of 4 costing 213% at batch 1.
#
# It has caught one. Scored on bfloat16 the rule gave up 29.86% at the 48-row
# tile at batch 64, dispatching a tiling whose 128-deep batch block is half empty
# there; that was a threshold read off float32 and it is now two buckets. The
# bound is what surfaced it, so it is not widened to accommodate anything.
MAX_DISPATCH_REGRET = 0.20
MEAN_DISPATCH_REGRET = 0.03

# Both bounds are judged on the floor of the three bfloat16 sweeps rather than on
# any one of them, because regret is systematically inflated by contention rather
# than merely scattered by it. It divides by the fastest configuration measured,
# so a rival that happens to get a quiet moment lowers the denominator for
# everybody, and averaging over 78 cases does not remove a bias. Measured on the
# same 78 points: mean regret reads 2.52%, 4.64% and 2.59% across the three
# sweeps and the worst case reads 17.23%, 79.97% and 19.15%, against 2.22% and
# 11.92% on the floor.
#
# The floor is bfloat16 and not float32 because the packed kernels take their
# dtype as a template parameter, so two dtypes rank two different sets of compiled
# kernels -- which is exactly how the narrow-tile error above survived. The
# float32 sweeps remain on disk and still reproduce what the report quotes of
# them, which is how this floor was checked before anything was published from it.
#
# What corrupts the ranking is how much the foreign load *varies*, not how large
# it is. The dirtiest-looking sweep is the cleanest of the three: its competitor
# was a saturating MLX process holding 85% of the device continuously, which
# scales every configuration alike and cancels in a ratio. The two captured
# against a desktop compositor bursting between 0.4% and 95% are the unusable
# ones.

# The shape the dispatch tests below build. It is a real row count -- ssm_out,
# attn_output and every attention projection but q are 5120 rows -- and it tiles
# at 256, so it exercises the multi-tile divisibility check rather than the
# single-tile escape hatch.
DISPATCH_ROWS = 5120


def measured_cases() -> list[dict[str, object]]:
    """The swept points whose shape the model actually contains.

    The float32 sweeps still on disk carry one that it does not: an early copy
    of the shape list recorded ``self_attn.o_proj`` as 5120x12288, where the
    model's o_proj is 5120x6144 and ``linear_attn.out_proj`` already covers it.
    That entry is gone from :data:`MODEL_LINEAR_SHAPES`, so the gated sweeps
    never measured it and this filter drops nothing from them.

    It is kept anyway, because what it asserts is not a fact about today's files
    but the precondition of the whole replay: a dispatch rule scored against a
    shape the model has no instance of proves nothing either way, whichever
    direction the two lists drift.
    """
    sweeps = gated_sweeps()
    if not sweeps:
        pytest.skip(f"no {GATED_ACTIVATION_DTYPE} sweep has been produced yet")
    real = {(shape.rows, shape.columns) for shape in MODEL_LINEAR_SHAPES}
    return [
        case
        for case in json.loads(sweeps[0].read_text())["matmul"]
        if (int(case["rows"]), int(case["columns"])) in real
    ]


def fastest_seen() -> dict[tuple[str, int], dict[str, float]]:
    """Each configuration's fastest time across every repeat, per measured point.

    The estimator is a minimum because the noise it is fighting is one-sided. A
    competing process can only take GPU time away from a kernel, never give it
    any, so of several timings of the same configuration the smallest is the one
    nearest its uncontended speed, and repeating the sweep can only improve the
    estimate. This is the same reason a benchmark reports best-of-N rather than
    mean-of-N when its interference is additive.

    That it is recovering signal rather than manufacturing a flattering number is
    checkable, and was checked: run over the three float32 sweeps instead, the
    same construction reproduces the figures the report carried from them -- 43 of
    99 exact, mean 2.04%, worst 14.52% at ssm_out batch 512 -- which no individual
    sweep among them reads. A floor that could not recover a known answer would
    not be trusted with an unknown one.
    """
    sweeps = gated_sweeps()
    if len(sweeps) != SWEEPS_REQUIRED:
        raise FormatError(
            f"the floor needs {SWEEPS_REQUIRED} independent {GATED_ACTIVATION_DTYPE} "
            f"sweeps and {len(sweeps)} are present; a floor over fewer is just the "
            "contended reading of whichever survived"
        )
    floor: dict[tuple[str, int], dict[str, float]] = {}
    for path in sweeps:
        for case in json.loads(path.read_text())["matmul"]:
            point = floor.setdefault((str(case["label"]), int(case["batch"])), {})
            for variant in case["variants"]:
                config, seconds = str(variant["config"]), packed_median(variant)
                point[config] = min(point.get(config, seconds), seconds)
    return floor


def packed_median(variant: dict[str, object]) -> float:
    """Absolute time, which is what 'which of our kernels is faster' asks.

    Not ``speedup_vs_affine_2bit``: that is the right statistic for reporting a
    headline against MLX, and the wrong one for ranking our own configurations,
    since it divides by a baseline that was re-measured for every variant.
    """
    return float(variant["packed_seconds"]["median"])


def case_layout(case: dict[str, object]) -> PackedTensorLayout:
    return PackedTensorLayout(
        rows=int(case["rows"]), columns=int(case["columns"]), tile=int(case["tile"])
    )


def dispatched_name(case: dict[str, object]) -> str:
    """The benchmark label of the configuration the shipped rule would pick."""
    layout, batch = case_layout(case), int(case["batch"])
    gemm = select_gemm(layout, batch)
    if gemm is not None:
        return gemm.benchmark_name
    # The tiling, not the split: A5 swept the batch tile with the K axis whole,
    # so it is the only one of the two knobs this file can speak to. The split
    # is replayed against its own sweep in test_the_split_rule_* below.
    return select_lut23_tile(layout, batch).benchmark_name


def occupancy_refused(case: dict[str, object]) -> bool:
    """Did the rule send this point to LUT23 because no tiling could fill the GPU?

    This is the one question A5 is not allowed to be asked, and the restriction
    is structural rather than a convenience. ``bench_ops`` times a shape by
    enqueueing many *independent* copies of the same matmul, so a dispatch that
    offers sixteen threadgroups is submitted a hundred at a time and the GPU
    fills itself from the queue. A tiling whose only defect is that it cannot
    occupy the machine alone is therefore exactly the tiling that regime scores
    as fine -- and in a forward pass, where each matmul waits on the one before
    it, the same tiling runs an eighth of the part.

    So where ``select_gemm`` refuses a legal tiling on the occupancy floor, this
    file's verdict is measuring the confound and not the rule. Two shapes reach
    it: the 48-row alpha and beta projections at every prefill batch swept, and
    the 1024-row k and v projections up to batch 64. A5 resolves five of those
    eleven points for the prefill kernel, by 1.34x to 2.54x. The in-situ sweep
    puts the same tilings at 12.28x and 20.73x the affine baseline against
    LUT23's 1.89x and 3.91x, and it is the one that timed them in the graph.

    A batch below :data:`GEMM_MIN_BATCH` is not refused here, because the rule
    never reaches the occupancy test there -- the family crossover decides it,
    and that threshold is A5's own.
    """
    layout, batch = case_layout(case), int(case["batch"])
    if batch < GEMM_MIN_BATCH:
        return False
    ladder = (
        GEMM_LADDER_LARGE_BATCH if batch >= GEMM_LARGE_MIN_BATCH
        else GEMM_LADDER_SMALL_BATCH
    )
    legal = [
        config for config in ladder
        if layout.tiles == 1 or layout.tile % config.row_block == 0
    ]
    return bool(legal) and threadgroups(layout, batch, legal[-1]) < GEMM_MIN_WAVES * GPU_CORES


# The two shapes whose rows cannot fill this GPU at a prefill batch, and the
# batches at which each is still short. Enumerated rather than derived so that a
# threshold edit that quietly widens the set A5 is no longer asked about fails
# here instead of passing quietly with less to be wrong about.
OCCUPANCY_REFUSED_POINTS = {
    ("linear_attn.in_proj_a", batch) for batch in (16, 32, 40, 64, 128, 200, 512)
} | {("self_attn.k_proj", batch) for batch in (16, 32, 40, 64)}

# The narrow row block is the in-situ sweep's finding and A5 never swept it, so
# the points that dispatch it cannot be scored here at all. Both tilings are the
# 32-row rung of their ladder; nothing else in the file is unmeasured.
UNMEASURED_ROW_BLOCK = 32


def test_the_dispatch_rule_reproduces_the_measured_crossover() -> None:
    """Replay every measured point through the rule that ships.

    This is the test the rule exists to satisfy. ``select_gemm`` and
    ``select_lut23`` encode thresholds read off the A5 sweep, and a threshold
    read off a file drifts silently from it the moment either is edited. So
    rather than restate the thresholds here, this feeds the measured
    (shape, batch) pairs back through the functions and checks what they pick
    against what was actually fastest.

    Two of A5's three thresholds survived the in-situ sweep and one did not, so
    the replay is now a partition rather than a sweep over everything.
    Seventy-eight of the 99 points are scored. Ten dispatch a row block A5 never
    swept and cannot be scored here; they are replayed against the sweep that
    chose them. Eleven turn on occupancy, which A5's enqueue pattern supplies for
    free -- see :func:`occupancy_refused`. Both excluded sets are asserted
    exactly, because an exclusion nobody counts is how a rule stops being tested.

    The scored set grew by eight when the large-batch ladder's wide rung came
    down from 128 rows to 64, and the direction is worth stating: those eight
    points became scorable because the rung the in-situ fit chose is a tiling A5
    had measured all along, so a second protocol gets a vote on it. It votes the
    same way. Mean regret over the scored set fell from 2.7% to 2.2% and the
    count of exactly-right dispatches rose from 22 to 27, on a file that had no
    part in choosing the rung.
    """
    floor = fastest_seen()
    regrets: list[tuple[float, str]] = []
    refused: set[tuple[str, int]] = set()
    unmeasured: list[str] = []
    for case in measured_cases():
        label, batch = str(case["label"]), int(case["batch"])
        chosen = dispatched_name(case)
        times = floor[(label, batch)]
        if occupancy_refused(case):
            refused.add((label, batch))
            continue
        if chosen not in times:
            # Every unscorable point has to be unscorable for the one stated
            # reason. A dispatch that fell off the sweep for any other reason is
            # a defect rather than an exclusion, so the row block is checked
            # here rather than trusted.
            config = select_gemm(case_layout(case), batch)
            assert config is not None and config.row_block == UNMEASURED_ROW_BLOCK, (
                f"{chosen} was dispatched for {label} at batch {batch} but never "
                "measured there, and it is not the row block A5 skipped"
            )
            unmeasured.append(chosen)
            continue
        best = min(times.values())
        winner = min(times, key=lambda config: times[config])
        regrets.append((
            times[chosen] / best - 1.0,
            f"{label} rows={case['rows']} batch={batch}: "
            f"{chosen} at {times[chosen] * 1e6:.2f}us, "
            f"best {winner} at {best * 1e6:.2f}us",
        ))

    assert refused == OCCUPANCY_REFUSED_POINTS, (
        f"the occupancy floor now excludes {sorted(refused ^ OCCUPANCY_REFUSED_POINTS)} "
        "differently than it did when these bounds were set"
    )
    assert len(unmeasured) == 10

    # Nine real shapes at eleven batches, less the 21 above. Pinned so a sweep
    # that silently stops covering a shape cannot pass this by having less to be
    # wrong about.
    assert len(regrets) == 78
    worst, where = max(regrets)
    assert worst <= MAX_DISPATCH_REGRET, f"{worst:.1%} given up on {where}"
    mean = sum(regret for regret, _ in regrets) / len(regrets)
    assert mean <= MEAN_DISPATCH_REGRET, f"mean regret {mean:.2%}"


def resolvable_family_winner(case: dict[str, object]) -> tuple[str, str, float] | None:
    """Which family won this case, or ``None`` where the sweep cannot tell.

    The comparison is between each family's fastest measured configuration, on
    absolute time -- taken from ``variants`` rather than from the file's own
    ``best_per_family``, which records only the name and the affine ratio.
    Overlapping p5/p95 bands mean the sweep could not separate the families
    here, and reading a winner off that would be reading noise.
    """
    fastest: dict[str, dict[str, object]] = {}
    for variant in case["variants"]:
        family = str(variant["family"])
        if family not in fastest or packed_median(variant) < packed_median(fastest[family]):
            fastest[family] = variant
    if set(fastest) != {"lut23", "gemm"}:
        return None  # one family could not run this tensor at all
    lut, gemm = fastest["lut23"], fastest["gemm"]
    if (
        lut["packed_seconds"]["p5"] <= gemm["packed_seconds"]["p95"]
        and gemm["packed_seconds"]["p5"] <= lut["packed_seconds"]["p95"]
    ):
        return None
    winner = "lut23" if packed_median(lut) < packed_median(gemm) else "gemm"
    detail = (
        f"{case['label']} rows={case['rows']} batch={case['batch']}: "
        f"lut23 {packed_median(lut) * 1e6:.2f}us against gemm "
        f"{packed_median(gemm) * 1e6:.2f}us"
    )
    return winner, detail, packed_median(lut) / packed_median(gemm)


def dispatched_family(case: dict[str, object]) -> str:
    """Which kernel family the rule sends this point to.

    Read off the rule rather than looked up by name in the sweep. The two are
    the same thing wherever the sweep measured the tiling the rule picks, and
    where it did not -- the narrow row block, which A5 never swept -- the lookup
    raises where the question still has a perfectly good answer.
    """
    return "lut23" if select_gemm(case_layout(case), int(case["batch"])) is None else "gemm"


# The one (tile class, batch) group where the shapes that resolve do not agree
# with each other. Batch 8 is the crossover's own width: gate_proj resolves for
# the prefill kernel at 1.12x while o_proj and down_proj resolve against it at
# 0.87x and 0.84x, and six of the nine wide shapes do not resolve at all. No
# rule that sees only the tile and the batch can satisfy all three, so this
# names the ambiguity instead of letting it hide. Everything else is unanimous.
SPLIT_FAMILY_GROUPS = {("wide", 8)}

# How many groups survive dropping the occupancy-refused points. Seventeen
# resolve at all; the two that go are the narrow tile at batches 200 and 512,
# whose only member is the 48-row shape that cannot fill this GPU at any tiling
# until batch 1249. Pinned because a widened exclusion would announce itself by
# emptying groups, and an empty group demands nothing.
RESOLVABLE_FAMILY_GROUPS = 15


def test_the_dispatch_rule_picks_the_winning_family_wherever_it_is_resolvable() -> None:
    """Family choice is judged where the measurement can tell the families apart.

    This is the assertion that actually pins the dispatch rule down. Which of
    the LUT23 batch tiles runs a tensor moves a few percent and the sweep cannot
    resolve it; which *family* runs it is the decision worth several tens of
    percent, and it is the one the shipped thresholds encode.

    The judgement is at the granularity the rule decides at, on the points this
    file is entitled to judge. ``select_gemm`` sees a whole (tile class, batch)
    group at once and can only be held to what that group's resolvable cases
    agree on. Where they are unanimous -- fourteen of the fifteen groups,
    including every batch from 16 up -- that is a hard demand, and a rule on the
    wrong side of the crossover fails it. Where they contradict each other the
    demand is the majority, and the split is asserted to be exactly the one
    group already known to be split, so a second one appearing is a failure
    rather than a silent relaxation.

    Occupancy-refused points are dropped from the vote as well as from the
    demand, for the reason :func:`occupancy_refused` gives: A5's enqueue pattern
    hides the starvation that decides them, so their verdict here is a reading
    of the harness. Dropping them from the vote and not the demand would be
    worse than either -- it would hold the rule to a majority assembled from
    shapes the majority was not allowed to include.
    """
    winners: dict[tuple[str, int], list[tuple[str, str, float]]] = {}
    for case in measured_cases():
        if occupancy_refused(case):
            continue
        resolved = resolvable_family_winner(case)
        if resolved is None:
            continue
        tile_class = "narrow" if int(case["tile"]) < LUT23_THREADS else "wide"
        winners.setdefault((tile_class, int(case["batch"])), []).append(resolved)

    assert len(winners) == RESOLVABLE_FAMILY_GROUPS, (
        f"{len(winners)} groups resolve where {RESOLVABLE_FAMILY_GROUPS} did; a "
        "group that empties stops demanding anything"
    )
    split = {
        group
        for group, resolved in winners.items()
        if len({winner for winner, _, _ in resolved}) > 1
    }
    assert split == SPLIT_FAMILY_GROUPS, f"family verdicts split in {sorted(split)}"

    # A group split evenly demands nothing -- there is no majority to obey. None
    # of the fifteen is, but leaving the tie undefined rather than resolving it
    # by set-iteration order keeps the demand a measured one.
    majority: dict[tuple[str, int], str] = {}
    for group, resolved in winners.items():
        for_gemm = sum(winner == "gemm" for winner, _, _ in resolved)
        if for_gemm * 2 != len(resolved):
            majority[group] = "gemm" if for_gemm * 2 > len(resolved) else "lut23"
    disagreements: list[str] = []
    for case in measured_cases():
        if occupancy_refused(case):
            continue
        tile_class = "narrow" if int(case["tile"]) < LUT23_THREADS else "wide"
        group = (tile_class, int(case["batch"]))
        if group not in majority:
            continue
        chosen = dispatched_family(case)
        if chosen != majority[group]:
            resolved = resolvable_family_winner(case)
            detail = resolved[1] if resolved else f"{case['label']} batch={case['batch']}"
            disagreements.append(
                f"dispatched {chosen} where {tile_class} tiles at batch "
                f"{case['batch']} resolve for {majority[group]} -- {detail}"
            )
    assert not disagreements, "\n".join(disagreements)


def test_generation_never_dispatches_the_prefill_kernel() -> None:
    """One token at a time is where LUT23 wins by the largest margin measured."""
    for rows, columns in ((248320, 5120), (17408, 5120), (5120, 5120), (48, 5120)):
        layout = PackedTensorLayout.for_tensor(rows, columns)
        assert select_gemm(layout, 1) is None
        assert select_lut23_tile(layout, 1) is LUT23_SINGLE
        assert select_lut23(layout, 1).batch_tile == 1


def test_the_batch_tile_never_exceeds_the_batch_it_serves() -> None:
    """A tile wider than the batch pays full price for vectors that do not exist."""
    layout = PackedTensorLayout.for_tensor(5120, 5120)
    for batch in range(1, GEMM_MIN_BATCH):
        assert select_lut23(layout, batch).batch_tile <= max(batch, 1)
    assert select_lut23_tile(layout, 1) is LUT23_SINGLE
    assert select_lut23_tile(layout, 2) is LUT23_PAIR
    assert select_lut23_tile(layout, 3) is LUT23_PAIR
    assert select_lut23_tile(layout, 4) is LUT23_BATCHED
    assert select_lut23_tile(layout, 64) is LUT23_BATCHED


def test_a_narrow_tile_is_never_batch_tiled_on_lut23() -> None:
    """48 of 256 threads busy is not a shortage of vectors to work on.

    The prefill kernel has the mirror-image problem on the same tensor, and it is
    worse. 48 rows yield two 32-row blocks and one 128-row block, so however wide
    the prompt gets the row axis contributes almost nothing: at batch 128 the
    widest legal tiling dispatches a single threadgroup onto 40 cores, and the
    in-situ sweep prices that at 8.65x the affine baseline against LUT23's 2.96x.
    The batch axis has to make up the whole shortfall on its own, which takes
    until batch 1249 -- forty batch blocks of 32, times the two row blocks, is
    the first pairing to reach :data:`GEMM_MIN_WAVES` passes over the GPU.

    So the special case the previous rule carried for narrow tiles is gone, and
    nothing replaced it. The occupancy floor already says everything that rule
    said, which is what the arithmetic below asserts: it is derived from the
    threadgroup count rather than pinned to a remembered batch.
    """
    narrow = PackedTensorLayout.for_tensor(rows=48, columns=5120)
    assert narrow.tile == 48
    for batch in (1, 2, 4, 16, 128, 511, 512):
        assert select_lut23_tile(narrow, batch) is LUT23_SINGLE
        assert select_lut23(narrow, batch).batch_tile == 1

    narrowest = GEMM_LADDER_LARGE_BATCH[-1]
    crossover = next(
        batch for batch in range(1, 4097)
        if threadgroups(narrow, batch, narrowest) >= GEMM_MIN_WAVES * GPU_CORES
    )
    assert crossover == 1249
    for batch in (1, 2, 4, 16, 32, 64, 128, 200, 512, crossover - 1):
        assert select_gemm(narrow, batch) is None
    # And when it finally arrives it arrives narrow. 48 rows are one 64-row
    # block, so the wide rung has to buy all 120 of its threadgroups on the batch
    # axis and does not get there until 3809 -- past the longest prefill chunk
    # this project measures, and stated here rather than called unreachable.
    for batch in (crossover, 2048, 3808):
        assert select_gemm(narrow, batch) is GEMM_LARGE_BATCH_NARROW_BLOCK
    assert select_gemm(narrow, 3809) is GEMM_LARGE_BATCH_WIDE_BLOCK


def test_a_thousand_rows_stay_on_lut23_until_the_batch_supplies_the_threadgroups()\
        -> None:
    """1024 rows are 32 blocks, so the batch has to carry the occupancy.

    This is the point the previous rule got wrong in both directions. It first
    held every tensor under 5120 rows on LUT23 forever, which A5 refuted; then it
    sent 1024 rows to a 128-row block from batch 16, which offers eight
    threadgroups to 40 cores. Timed in the graph rather than in a queue of
    independent copies, that cost 12.28x the affine baseline at batch 16 where
    LUT23 cost 1.89x -- the single largest error the in-situ sweep found, and the
    reason the row count is now an input.

    What replaces both is one quantity. At 32 rows per block the k projection
    offers 32 threadgroups per batch block, so it needs three batch blocks to
    clear the floor and reaches the prefill kernel at batch 65; the wide block
    offers 16 per batch block, needs eight of them, and waits until 225.
    """
    layout = PackedTensorLayout.for_tensor(rows=1024, columns=5120)
    assert layout.tile == 256
    assert layout.tiles == 4
    for batch in (GEMM_MIN_BATCH - 1, GEMM_MIN_BATCH, GEMM_LARGE_MIN_BATCH, 64):
        assert select_gemm(layout, batch) is None
    for batch in (65, 128, 224):
        assert select_gemm(layout, batch) is GEMM_LARGE_BATCH_NARROW_BLOCK
    for batch in (225, 4096):
        assert select_gemm(layout, batch) is GEMM_LARGE_BATCH_WIDE_BLOCK

    # The comparison the rule is built to make. Three shapes at batch 16, one
    # dispatch width apart: 1024 rows cannot fill the machine at any tiling and
    # go to LUT23, 12288 rows fill it on a 32-row block but only reach 96 of the
    # 120 threadgroups a 128-row block would need, and 17408 rows clear it.
    for rows, expected in (
        (12288, GEMM_SMALL_BATCH_NARROW_BLOCK),
        (17408, GEMM_SMALL_BATCH_WIDE_BLOCK),
    ):
        assert select_gemm(
            PackedTensorLayout.for_tensor(rows=rows, columns=5120), GEMM_MIN_BATCH
        ) is expected


def test_the_large_batch_ladder_stops_a_rung_below_the_small_batch_one() -> None:
    """Filling the machine is a floor on the row block, not an argument for width.

    The occupancy rule reads that floor as if it were the whole story, which
    holds at batch 16 and stops holding at 511. There every shape a prefill
    dispatches offers hundreds of threadgroups at every rung, so the floor
    separates nothing and the walk returns whichever rung it is offered first --
    and offered a 128-row block it hands one to all eight of them. That is not a
    threshold that needs moving. It is a rung that does not belong on this
    ladder, and the assertion that matters is that it is absent rather than
    merely unreachable.

    Fitted at 511 and 2047 instead of extrapolated from the batch-128 sweep, a
    128-row wide rung costs 1.49% and 0.18% of the summed linear time against
    this 64-row one. Of the sixteen 64-against-128 readings only two separate on
    their ranges and both say 64: the gate projection, 42% of that time on its
    own, at 1053.1-1058.1ms against 1081.9-1087.0ms, and the k projection at
    17.9-19.3ms against 21.4-25.3ms. The three shapes that read faster at 128
    all overlap, which is why the 0.83% still between this ladder and knowing
    each shape's argmin is left on the table rather than fitted to.
    """
    assert GEMM_LARGE_BATCH_WIDE_BLOCK.row_block < GEMM_SMALL_BATCH_WIDE_BLOCK.row_block
    assert all(config.row_block <= 64 for config in GEMM_LADDER_LARGE_BATCH)

    # B8's two long workloads prefill at ``prompt_tokens - 1``. lm_head is not
    # among them: mlx-lm's chunk loop discards the model's return value and
    # evaluates only the cache, so a prefill pass never issues the output head.
    prefill = tuple(shape for shape in MODEL_LINEAR_SHAPES if shape.label != "lm_head")
    assert len(prefill) == 8
    for batch in (511, 2047):
        for shape in prefill:
            selected = select_gemm(shape.layout(), batch)
            if shape.rows == 48:
                # One row block at either rung, so the batch axis carries the
                # whole dispatch -- and at 511 it is still short, which is the
                # one place this ladder and the rule it replaced disagree in the
                # narrow direction rather than the wide one.
                assert selected is (
                    None if batch == 511 else GEMM_LARGE_BATCH_NARROW_BLOCK
                )
            else:
                assert selected is GEMM_LARGE_BATCH_WIDE_BLOCK


def test_a_multi_tile_tensor_only_takes_tilings_that_divide_its_tile() -> None:
    """A row block straddling two tiles would read the wrong tile's codes."""
    ordinary = PackedTensorLayout.for_tensor(rows=DISPATCH_ROWS, columns=BLOCK_SIZE)
    assert ordinary.tiles > 1
    assert ordinary.tile % GEMM_SMALL_BATCH_NARROW_BLOCK.row_block == 0
    assert select_gemm(ordinary, GEMM_MIN_BATCH) is GEMM_SMALL_BATCH_NARROW_BLOCK

    # One tile has nothing to straddle, so an indivisible tile is fine there --
    # which is the case the check has to let through, not merely tolerate.
    single = PackedTensorLayout.for_tensor(rows=DISPATCH_ROWS + 80, columns=BLOCK_SIZE)
    assert single.tiles == 1
    assert single.tile % GEMM_SMALL_BATCH_NARROW_BLOCK.row_block
    assert select_gemm(single, GEMM_MIN_BATCH) is GEMM_SMALL_BATCH_NARROW_BLOCK

    # A tile of 80 divides neither rung of either ladder, so both batch buckets
    # have to refuse it outright. This is the only case left that reaches
    # ``None`` through illegality rather than through the occupancy floor.
    strange = PackedTensorLayout(rows=DISPATCH_ROWS, columns=BLOCK_SIZE, tile=80)
    assert strange.tiles > 1
    for ladder in (GEMM_LADDER_SMALL_BATCH, GEMM_LADDER_LARGE_BATCH):
        assert all(strange.tile % config.row_block for config in ladder)
    assert select_gemm(strange, GEMM_MIN_BATCH) is None
    assert select_gemm(strange, GEMM_LARGE_MIN_BATCH) is None

    # A tile divisible by the narrow rung and not the wide one takes the narrow
    # rung rather than falling off the ladder. The previous rule had no ladder to
    # walk and refused this outright, dropping a legal tensor to LUT23 for a
    # reason that was never measured.
    partial = PackedTensorLayout(rows=DISPATCH_ROWS, columns=BLOCK_SIZE, tile=160)
    assert partial.tile % GEMM_LARGE_BATCH_WIDE_BLOCK.row_block
    assert partial.tile % GEMM_LARGE_BATCH_NARROW_BLOCK.row_block == 0
    assert select_gemm(partial, 4096) is GEMM_LARGE_BATCH_NARROW_BLOCK


# --------------------------------------------------------------------------
# The K-axis split
# --------------------------------------------------------------------------


SPLIT_BENCHMARKS = Path("results/mlx/a6_split_k.json")

# What the split rule may give up against the fastest split measured, judged on
# chained per-call time -- the regime decode is in and the only one where the
# split does anything at all.
#
# Two bounds because they answer different questions. The ledger bound is the
# one that matters to a user: it weights each shape by how many times a decoded
# token dispatches it, so it is the regret in tokens per second. The per-case
# bound catches a shape being mis-split even where the model barely uses it.
#
# Both are set from measurement rather than chosen, and specifically from the
# worse of the two A6 runs rather than the committed one. That distinction is
# the whole point: an earlier pair of bounds was set just above where the sweep
# that had tuned the ceiling landed, which made them a gate on that sweep's
# noise -- they passed at 1.84% and then failed at 3.32% when the sweep was
# repeated with nothing about the rule changed. Scored across both runs the
# shipped ceiling gives up 9.1% and 9.5% on its worst single shape and 1.58%
# and 2.71% over the token, so the bounds sit just above the higher of each.
# The assertion messages print where the worst case landed, so a regression
# names its own shape.
MAX_SPLIT_REGRET = 0.10
MAX_LEDGER_REGRET = 0.03


def split_cases() -> list[dict[str, object]]:
    if not SPLIT_BENCHMARKS.exists():
        pytest.skip(f"{SPLIT_BENCHMARKS} has not been produced yet")
    return json.loads(SPLIT_BENCHMARKS.read_text())["matmul"]


def chained_per_call(variant: dict[str, object]) -> float:
    """Seconds this configuration costs a dispatch that nothing overlaps.

    ``chained_seconds.per_call`` already has the dependency link's own cost
    subtracted, so it is the matmul alone. The overlapped column is kept in the
    file for the serialisation ratio and is deliberately not used here: it is
    the regime that hides the very latency the split exists to shorten, and a
    rule tuned on it would pick no split anywhere.
    """
    return float(variant["chained_seconds"]["per_call"])


def split_choice(case: dict[str, object]) -> tuple[str, int]:
    """The configuration name and split the shipped rule picks for this point.

    Asserts on the way past that LUT23 is what would run here at all. The split
    only exists inside that family, so a batch the prefill GEMM claims is a
    batch where replaying ``select_lut23`` would be scoring a rule the model
    never reaches -- and the sweep's batch list is what keeps that from
    happening, so it is checked rather than assumed.
    """
    layout = PackedTensorLayout(
        rows=int(case["rows"]), columns=int(case["columns"]), tile=int(case["tile"])
    )
    batch = int(case["batch"])
    gemm = select_gemm(layout, batch)
    assert gemm is None, (
        f"{case['label']} at batch {batch} dispatches {gemm.benchmark_name}, not LUT23; "
        "the split sweep is measuring a family that would not run"
    )
    config = select_lut23(layout, batch)
    return config.benchmark_name, config.k_split


def test_the_split_rule_reproduces_the_measured_optimum() -> None:
    """Replay every measured (shape, batch, split) through the rule that ships.

    ``select_lut23`` encodes one number -- the threadgroup ceiling the split
    aims at -- and a number read off a sweep drifts from it silently. So rather
    than restate the ceiling here, this feeds every point back through the
    function and checks the split it picks against the split that was fastest.
    """
    regrets: list[tuple[float, str]] = []
    for case in split_cases():
        chosen, split = split_choice(case)
        times = {
            str(variant["config"]): chained_per_call(variant)
            for variant in case["variants"]
            if variant["family"] == "lut23"
        }
        assert chosen in times, (
            f"{chosen} was dispatched for {case['label']} at batch {case['batch']} "
            f"but never measured there; the sweep holds {sorted(times)}"
        )
        assert split == 1 or int(case["groups_per_row"]) % split == 0, (
            f"split {split} does not divide {case['groups_per_row']} groups"
        )
        best = min(times.values())
        winner = min(times, key=lambda config: times[config])
        regrets.append((
            times[chosen] / best - 1.0,
            f"{case['label']} rows={case['rows']} batch={case['batch']}: "
            f"{chosen} at {times[chosen] * 1e6:.2f}us, "
            f"best {winner} at {best * 1e6:.2f}us",
        ))

    assert regrets, "the sweep produced no LUT23 points to replay"
    worst, where = max(regrets)
    assert worst <= MAX_SPLIT_REGRET, f"{worst:.1%} given up on {where}"


def test_the_split_rule_costs_little_over_a_whole_decoded_token() -> None:
    """The same regret, weighted by how often a token actually issues each shape.

    A shape the rule mis-splits by 30% matters exactly as much as the model
    dispatches it, and the per-case bound above cannot see that. This one
    rebuilds the token: every shape's chained cost times its dispatch count,
    under the rule and under the oracle that knows each shape's best split.
    """
    cases = [case for case in split_cases() if int(case["batch"]) == 1]
    assert cases, "the sweep must contain batch 1; that is what decode does"
    dispatches = sum(int(case["dispatches_per_token"]) for case in cases)
    assert dispatches == 497, (
        f"the swept shapes cover {dispatches} of a token's 497 matmuls; a ledger "
        "over part of a token is not a ledger"
    )

    ruled = oracle = unsplit = 0.0
    for case in cases:
        times = {
            str(variant["config"]): chained_per_call(variant)
            for variant in case["variants"]
            if variant["family"] == "lut23"
        }
        chosen, _ = split_choice(case)
        count = int(case["dispatches_per_token"])
        ruled += count * times[chosen]
        oracle += count * min(times.values())
        unsplit += count * times[select_lut23_tile(
            PackedTensorLayout(
                rows=int(case["rows"]), columns=int(case["columns"]), tile=int(case["tile"])
            ),
            1,
        ).benchmark_name]

    regret = ruled / oracle - 1.0
    assert regret <= MAX_LEDGER_REGRET, (
        f"the rule gives up {regret:.2%} of a decoded token against the best split "
        f"per shape: {ruled * 1e3:.2f}ms against {oracle * 1e3:.2f}ms"
    )
    # The split has to be worth having at all, not merely well chosen. This is
    # the claim the whole mechanism exists to make, so it is asserted, not noted.
    assert ruled < unsplit, (
        f"splitting cost {ruled * 1e3:.2f}ms against {unsplit * 1e3:.2f}ms unsplit"
    )


# --------------------------------------------------------------------------
# Numerics
# --------------------------------------------------------------------------


# One (rows, batch, tiling) triple per regime the rule can select, and the
# tiling is named rather than derived so that a rule change which quietly stops
# reaching one of the seven kernels fails here instead of testing six of them
# twice. The row count is a parameter because it now decides as much as the
# batch does: 5120 rows at batch 16 fill 160 threadgroups on a 32-row block and
# only 40 on a 128-row one, so the wide block needs a taller tensor to appear at
# all, and 17408 is the shape 128 of the model's 497 matmuls actually have.
DISPATCH_REGIMES = (
    (DISPATCH_ROWS, 1, None),
    (DISPATCH_ROWS, 2, None),
    (DISPATCH_ROWS, 4, None),
    (DISPATCH_ROWS, GEMM_MIN_BATCH, GEMM_SMALL_BATCH_NARROW_BLOCK),
    (17408, GEMM_MIN_BATCH, GEMM_SMALL_BATCH_WIDE_BLOCK),
    (DISPATCH_ROWS, GEMM_LARGE_MIN_BATCH, GEMM_LARGE_BATCH_NARROW_BLOCK),
    (DISPATCH_ROWS, GEMM_MIN_BATCH + 172, GEMM_LARGE_BATCH_WIDE_BLOCK),
)


@pytest.mark.parametrize(("rows", "batch", "expected"), DISPATCH_REGIMES)
def test_a_packed_linear_matches_the_oracle_across_every_dispatch_regime(
    rows: int, batch: int, expected: GemmConfig | None
) -> None:
    """Same module, same weights, every regime the rule can select, one oracle.

    The cases are picked to land one in each bucket the rule has: the three
    below :data:`GEMM_MIN_BATCH` cover all three LUT23 configurations, and the
    four above it cover both rungs of both prefill ladders. A shape the rule
    always answers the same way would leave one of the kernels untested through
    the module, which is the only place the dispatch and the arithmetic are
    exercised together.

    Both paths are held to :data:`ORACLE_RELATIVE_TOLERANCE` through the same
    :func:`compare` the A4 gate uses, rather than a flat absolute bound. The two
    kernels sum in different orders -- LUT23 accumulates one group at a time
    per row, the prefill kernel over a tiled K -- so their last bits differ from
    each other and from the oracle by an amount that scales with the output
    magnitude, which is exactly what a per-case relative bound measures and what
    an absolute one mistakes for a defect at large batch.
    """
    columns = 2 * BLOCK_SIZE
    layout, codes, scales = random_packed(rows, columns, seed=20260810 + batch)
    module = PackedLinear(layout)
    load_packed(module, codes, scales)
    assert select_gemm(layout, batch) is expected

    generator = np.random.default_rng(4242 + batch)
    x = generator.normal(0.0, 1.0, size=(batch, columns)).astype(np.float32)
    out = np.array(module(mx.array(x)), copy=False)

    assert out.shape == (batch, rows)
    assert np.isfinite(out).all()
    against_oracle = compare(
        out,
        oracle_matmul(codes, scales, x, layout=layout),
        name="oracle_matmul",
        tolerance=ORACLE_RELATIVE_TOLERANCE,
        labels=tuple(f"row{index}" for index in range(batch)),
    )
    assert against_oracle.mismatches == 0, against_oracle

    # ``restored_matmul`` decodes and dots one row at a time in Python, so it
    # runs over a few samples where the float64 oracle above covers all of them.
    # It is here for its independence, not its coverage: it reaches the weights
    # through the shipped block decoder rather than through a LUT.
    stride = max(1, batch // 4)
    against_restored = compare(
        out[::stride],
        restored_matmul(codes, scales, x[::stride], layout=layout),
        name="restored_matmul",
        tolerance=ORACLE_RELATIVE_TOLERANCE,
        labels=tuple(f"row{index}" for index in range(0, batch, stride)),
    )
    assert against_restored.mismatches == 0, against_restored


def test_a_packed_linear_keeps_the_leading_dimensions_of_its_input() -> None:
    """mlx-lm hands these layers ``(batch, sequence, hidden)``, not a flat matrix."""
    rows, columns = 256, BLOCK_SIZE
    layout, codes, scales = random_packed(rows, columns, seed=77)
    module = PackedLinear(layout)
    load_packed(module, codes, scales)

    generator = np.random.default_rng(77)
    x = generator.normal(0.0, 1.0, size=(2, 3, columns)).astype(np.float32)
    out = module(mx.array(x))
    assert tuple(out.shape) == (2, 3, rows)

    flat = module(mx.array(x.reshape(-1, columns)))
    np.testing.assert_array_equal(
        np.array(out, copy=False).reshape(-1, rows), np.array(flat, copy=False)
    )


def test_a_packed_embedding_gathers_the_rows_the_reference_does() -> None:
    rows, columns = 512, 2 * BLOCK_SIZE
    layout, codes, scales = random_packed(rows, columns, seed=909)
    module = PackedEmbedding(layout, dtype=mx.float32)
    load_packed(module, codes, scales)

    # 2-D and of the tokenizer's signed dtype, which is what mlx-lm passes.
    tokens = np.array([[0, 1, 255, 256], [rows - 1, 7, 128, 42]], dtype=np.int32)
    out = module(mx.array(tokens))
    assert tuple(out.shape) == (2, 4, columns)

    expected = get_rows_reference(
        codes, scales, tokens.reshape(-1), layout=layout, dtype=np.dtype(np.float32)
    )
    np.testing.assert_array_equal(
        np.array(out, copy=False).reshape(-1, columns), expected
    )


def test_a_packed_embedding_used_as_a_head_agrees_with_a_packed_linear() -> None:
    """The tied-head path must be the same arithmetic, not a parallel implementation."""
    rows, columns = 256, BLOCK_SIZE
    layout, codes, scales = random_packed(rows, columns, seed=31337)
    embedding = PackedEmbedding(layout, dtype=mx.float32)
    linear = PackedLinear(layout)
    load_packed(embedding, codes, scales)
    load_packed(linear, codes, scales)

    generator = np.random.default_rng(31337)
    x = mx.array(generator.normal(0.0, 1.0, size=(4, columns)).astype(np.float32))
    np.testing.assert_array_equal(
        np.array(embedding.as_linear(x), copy=False), np.array(linear(x), copy=False)
    )


def test_an_out_of_range_token_id_reads_zero_rather_than_out_of_bounds() -> None:
    """Memory safety, not a supported input: it cannot arise from tokenizer or sampler."""
    rows, columns = 256, BLOCK_SIZE
    layout, codes, scales = random_packed(rows, columns, seed=5)
    module = PackedEmbedding(layout, dtype=mx.float32)
    load_packed(module, codes, scales)

    out = np.array(module(mx.array(np.array([rows, rows + 1000, 0], dtype=np.int64))), copy=False)
    assert not out[0].any()
    assert not out[1].any()
    assert out[2].any()
    assert np.array(mx.array(np.array([0], dtype=np.int64)).astype(INDEX_DTYPE))[0] == 0
