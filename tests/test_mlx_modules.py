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
from ternel_mlx.kernels import INDEX_DTYPE
from ternel_mlx.layout import CODES_SUFFIX, SCALES_SUFFIX, PackedTensorLayout
from ternel_mlx.modules import (
    GEMM_LARGE_BATCH,
    GEMM_LARGE_MIN_BATCH,
    GEMM_MIN_BATCH,
    GEMM_NARROW_TILE,
    GEMM_SMALL_BATCH,
    LUT23_BATCHED,
    LUT23_PAIR,
    LUT23_SINGLE,
    LUT23_THREADS,
    NARROW_TILE_GEMM_MIN_BATCH,
    PackedEmbedding,
    PackedLinear,
    packed_matmul,
    select_gemm,
    select_lut23,
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


BENCHMARKS = Path("results/mlx/a5_op_benchmarks.json")

# What the dispatch rule may give up against the fastest configuration measured
# for each case, judged on absolute packed time.
#
# The mean is the real bound: it is an average over 110 cases, so the per-case
# drift discussed below averages out of it, and the shipped rule sits at 2.4%.
#
# The per-case maximum is deliberately loose, because this file cannot resolve
# one of our configurations against another. Each was timed in its own paired
# run against the affine baseline, so two configurations' packed times come from
# runs minutes apart and slow drift separates them on its own. Measured: the
# largest per-case gap the rule concedes here is 14.5%, on ssm_out at batch 512,
# and the configuration it loses to is its own twin -- the same 32x64 tiling with
# the threadgroup trit table compiled out. Their p5/p95 bands sit inside one
# another (3203-3757us against 3428-4045us), so the sweep cannot say which is
# faster there, and an earlier attempt to settle a case like it by re-running the
# two as the two arms of a single paired comparison -- 120 trials, batches 4 and
# 8, all nine wide-tile shapes -- resolved none of the eighteen points and
# reversed the sign of the worst one. So this bound is set to catch a structural
# mistake rather than to arbitrate noise: every genuine dispatch error available
# here is far larger, a fixed batch tile of 4 costing 213% at batch 1.
MAX_DISPATCH_REGRET = 0.20
MEAN_DISPATCH_REGRET = 0.03

# The shape the dispatch tests below build. It is a real row count -- ssm_out,
# attn_output and every attention projection but q are 5120 rows -- and it tiles
# at 256, so it exercises the multi-tile divisibility check rather than the
# single-tile escape hatch.
DISPATCH_ROWS = 5120


def measured_cases() -> list[dict[str, object]]:
    if not BENCHMARKS.exists():
        pytest.skip(f"{BENCHMARKS} has not been produced yet")
    return json.loads(BENCHMARKS.read_text())["matmul"]


def packed_median(variant: dict[str, object]) -> float:
    """Absolute time, which is what 'which of our kernels is faster' asks.

    Not ``speedup_vs_affine_2bit``: that is the right statistic for reporting a
    headline against MLX, and the wrong one for ranking our own configurations,
    since it divides by a baseline that was re-measured for every variant.
    """
    return float(variant["packed_seconds"]["median"])


def dispatched_name(case: dict[str, object]) -> str:
    """The benchmark label of the configuration the shipped rule would pick."""
    layout = PackedTensorLayout(
        rows=int(case["rows"]), columns=int(case["columns"]), tile=int(case["tile"])
    )
    batch = int(case["batch"])
    gemm = select_gemm(layout, batch)
    if gemm is not None:
        return gemm.benchmark_name
    return select_lut23(layout, batch).benchmark_name


def test_the_dispatch_rule_reproduces_the_measured_crossover() -> None:
    """Replay every measured point through the rule that ships.

    This is the test the rule exists to satisfy. ``select_gemm`` and
    ``select_lut23`` encode thresholds read off the A5 sweep, and a threshold
    read off a file drifts silently from it the moment either is edited. So
    rather than restate the thresholds here, this feeds all 110 measured
    (shape, batch) pairs back through the functions and checks what they pick
    against what was actually fastest.
    """
    regrets: list[tuple[float, str]] = []
    for case in measured_cases():
        chosen = dispatched_name(case)
        times = {str(v["config"]): packed_median(v) for v in case["variants"]}
        assert chosen in times, (
            f"{chosen} was dispatched for {case['label']} at batch {case['batch']} "
            "but never measured there"
        )
        best = min(times.values())
        regrets.append((
            times[chosen] / best - 1.0,
            f"{case['label']} rows={case['rows']} batch={case['batch']}: "
            f"{chosen} at {times[chosen] * 1e6:.2f}us, "
            f"best {case['best_config']} at {best * 1e6:.2f}us",
        ))

    assert len(regrets) == 110
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
    families = {str(v["config"]): str(v["family"]) for v in case["variants"]}
    return families[dispatched_name(case)]


# The one (tile class, batch) group where the shapes that resolve do not agree
# with each other. Batch 8 is the crossover's own width: gate_proj resolves for
# the prefill kernel at 1.12x while o_proj and down_proj resolve against it at
# 0.87x and 0.84x, and six of the nine wide shapes do not resolve at all. No
# rule that sees only the tile and the batch can satisfy all three, so this
# names the ambiguity instead of letting it hide. Everything else is unanimous.
SPLIT_FAMILY_GROUPS = {("wide", 8)}


def test_the_dispatch_rule_picks_the_winning_family_wherever_it_is_resolvable() -> None:
    """Family choice is judged where the measurement can tell the families apart.

    This is the assertion that actually pins the dispatch rule down. Which of
    the LUT23 batch tiles runs a tensor moves a few percent and the sweep cannot
    resolve it; which *family* runs it is the decision worth several tens of
    percent, and it is the one the shipped thresholds encode.

    The judgement is at the granularity the rule decides at. ``select_gemm``
    sees the tile and the batch, never the individual tensor, so it answers a
    whole (tile class, batch) group at once and can only be held to what that
    group's resolvable cases agree on. Where they are unanimous -- seventeen of
    the eighteen groups, including every batch from 16 up, nine shapes to nine
    -- that is a hard demand, and a rule on the wrong side of the crossover
    fails it nine to zero. Where they contradict each other the demand is the
    majority, and the split is asserted to be exactly the one group already
    known to be split, so a second one appearing is a failure rather than a
    silent relaxation.
    """
    winners: dict[tuple[str, int], list[tuple[str, str, float]]] = {}
    for case in measured_cases():
        resolved = resolvable_family_winner(case)
        if resolved is None:
            continue
        tile_class = "narrow" if int(case["tile"]) < LUT23_THREADS else "wide"
        winners.setdefault((tile_class, int(case["batch"])), []).append(resolved)

    split = {
        group
        for group, resolved in winners.items()
        if len({winner for winner, _, _ in resolved}) > 1
    }
    assert split == SPLIT_FAMILY_GROUPS, f"family verdicts split in {sorted(split)}"

    # A group split evenly demands nothing -- there is no majority to obey. None
    # of the eighteen is, but leaving the tie undefined rather than resolving it
    # by set-iteration order keeps the demand a measured one.
    majority: dict[tuple[str, int], str] = {}
    for group, resolved in winners.items():
        for_gemm = sum(winner == "gemm" for winner, _, _ in resolved)
        if for_gemm * 2 != len(resolved):
            majority[group] = "gemm" if for_gemm * 2 > len(resolved) else "lut23"
    disagreements: list[str] = []
    for case in measured_cases():
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
        assert select_lut23(layout, 1) is LUT23_SINGLE


def test_the_batch_tile_never_exceeds_the_batch_it_serves() -> None:
    """A tile wider than the batch pays full price for vectors that do not exist."""
    layout = PackedTensorLayout.for_tensor(5120, 5120)
    for batch in range(1, GEMM_MIN_BATCH):
        assert select_lut23(layout, batch).batch_tile <= max(batch, 1)
    assert select_lut23(layout, 1) is LUT23_SINGLE
    assert select_lut23(layout, 2) is LUT23_PAIR
    assert select_lut23(layout, 3) is LUT23_PAIR
    assert select_lut23(layout, 4) is LUT23_BATCHED
    assert select_lut23(layout, 64) is LUT23_BATCHED


def test_a_narrow_tile_is_never_batch_tiled_on_lut23() -> None:
    """48 of 256 threads busy is not a shortage of vectors to work on.

    The prefill kernel does not inherit that constraint: it tiles rows at 32 and
    the batch at 128, so a 48-row tensor is a launch-count problem for it rather
    than a thread-occupancy one, and it does eventually win. But it wins late --
    LUT23 is measurably ahead through batch 32 and nominally ahead at 40, so the
    threshold here is four times the one wide tiles get.
    """
    narrow = PackedTensorLayout.for_tensor(rows=48, columns=5120)
    assert narrow.tile == 48
    for batch in (1, 2, 4, 16, 128, 511, 512):
        assert select_lut23(narrow, batch) is LUT23_SINGLE

    for batch in (1, 2, 4, 16, 32, NARROW_TILE_GEMM_MIN_BATCH - 1):
        assert select_gemm(narrow, batch) is None
    for batch in (NARROW_TILE_GEMM_MIN_BATCH, 128, 512):
        assert select_gemm(narrow, batch) is GEMM_NARROW_TILE


def test_a_thousand_rows_reach_the_prefill_kernel_at_the_same_batch_as_the_rest() -> None:
    """No row-count floor: 1024 rows take the prefill kernel from batch 16.

    This replaces the opposite assertion. The rule that shipped before held every
    tensor under 5120 rows on LUT23 forever, and against the current tilings that
    is simply false -- at 1024 rows the dispatched kernel beats the fastest LUT23
    configuration by 1.14x at batch 16, rising to 2.08x at 512, with p5/p95 bands
    that do not overlap at any batch from 16 up.
    """
    layout = PackedTensorLayout.for_tensor(rows=1024, columns=5120)
    assert layout.tile == 256
    assert layout.tiles == 4  # few enough that the old rule called it too small
    assert select_gemm(layout, GEMM_MIN_BATCH - 1) is None
    assert select_gemm(layout, GEMM_MIN_BATCH) is GEMM_SMALL_BATCH
    assert select_gemm(layout, GEMM_LARGE_MIN_BATCH) is GEMM_LARGE_BATCH
    assert select_gemm(layout, 4096) is GEMM_LARGE_BATCH


def test_a_multi_tile_tensor_only_takes_tilings_that_divide_its_tile() -> None:
    """A row block straddling two tiles would read the wrong tile's codes."""
    ordinary = PackedTensorLayout.for_tensor(rows=DISPATCH_ROWS, columns=BLOCK_SIZE)
    assert ordinary.tiles > 1
    assert ordinary.tile % GEMM_SMALL_BATCH.row_block == 0
    assert select_gemm(ordinary, GEMM_MIN_BATCH) is GEMM_SMALL_BATCH

    # One tile has nothing to straddle, so an indivisible tile is fine there --
    # which is the case the check has to let through, not merely tolerate.
    single = PackedTensorLayout.for_tensor(rows=DISPATCH_ROWS + 80, columns=BLOCK_SIZE)
    assert single.tiles == 1
    assert single.tile % GEMM_SMALL_BATCH.row_block
    assert select_gemm(single, GEMM_MIN_BATCH) is GEMM_SMALL_BATCH

    # 160 divides neither shipped wide-tile row block -- 128 nor 64 -- so both
    # batch buckets have to refuse it. A tile of 320 would not test this: 64
    # divides it, so the large-batch bucket would run and only the small one
    # would refuse.
    strange = PackedTensorLayout(rows=DISPATCH_ROWS, columns=BLOCK_SIZE, tile=160)
    assert strange.tiles > 1
    assert strange.tile % GEMM_SMALL_BATCH.row_block
    assert strange.tile % GEMM_LARGE_BATCH.row_block
    assert select_gemm(strange, GEMM_MIN_BATCH) is None
    assert select_gemm(strange, GEMM_LARGE_MIN_BATCH) is None


# --------------------------------------------------------------------------
# Numerics
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "batch", (1, 2, 4, GEMM_MIN_BATCH, GEMM_LARGE_MIN_BATCH, GEMM_MIN_BATCH + 172)
)
def test_a_packed_linear_matches_the_oracle_across_every_dispatch_regime(
    batch: int,
) -> None:
    """Same module, same weights, every regime the rule can select, one oracle.

    The batches are picked to land one in each bucket the rule has for a wide
    tile: the three below :data:`GEMM_MIN_BATCH` cover all three LUT23
    configurations, :data:`GEMM_MIN_BATCH` covers the small-batch prefill tiling,
    and the two at or above :data:`GEMM_LARGE_MIN_BATCH` cover the large-batch
    one. A shape the rule always answers the same way would leave one of the
    kernels untested through the module.

    Both paths are held to :data:`ORACLE_RELATIVE_TOLERANCE` through the same
    :func:`compare` the A4 gate uses, rather than a flat absolute bound. The two
    kernels sum in different orders -- LUT23 accumulates one group at a time
    per row, the prefill kernel over a tiled K -- so their last bits differ from
    each other and from the oracle by an amount that scales with the output
    magnitude, which is exactly what a per-case relative bound measures and what
    an absolute one mistakes for a defect at large batch.
    """
    rows, columns = DISPATCH_ROWS, 2 * BLOCK_SIZE
    layout, codes, scales = random_packed(rows, columns, seed=20260810 + batch)
    module = PackedLinear(layout)
    load_packed(module, codes, scales)
    expected = None
    if batch >= GEMM_LARGE_MIN_BATCH:
        expected = GEMM_LARGE_BATCH
    elif batch >= GEMM_MIN_BATCH:
        expected = GEMM_SMALL_BATCH
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
