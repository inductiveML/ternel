"""Metal kernel gates: exhaustive byte and scale coverage, plus reference parity.

The positional tests below are exact-integer, not tolerance-based. Feeding the
activation ``[1, 3, 9, 27, 81]`` into a slot makes the group sum equal
``byte - 121`` for every legal byte, so a kernel that swapped two trit
positions, mis-derived ``hi = (p * 57) >> 9``, or read the wrong table would
produce a *different integer* — not a small numerical drift that a tolerance
could absorb.
"""

from __future__ import annotations

import mlx.core as mx
import numpy as np
import pytest

from bonsai_tq1.format import (
    BLOCK_SIZE,
    MAX_FULL_CODE_BYTE,
    MAX_TAIL_CODE_BYTE,
    FormatError,
    encode_tq1_blocks,
)
from bonsai_tq1.lut23_reorder import CODE_SLOTS, FULL_CODE_SLOTS
from ternel_mlx.kernels import (
    # Private on purpose: the lane-map test has to exercise the exact source
    # string the GEMM compiles, and a copy of it would pass forever.
    _GEMM_LANE_MAP,
    CODE_TABLE_ENTRIES,
    MATRIX_TILE,
    THREADGROUP_MEMORY_BYTES,
    TRITS_PER_FULL_BYTE,
    GemmConfig,
    MatmulConfig,
    gemm_threadgroup_bytes,
    lut_floats,
    tq1_gemm,
    tq1_get_rows,
    tq1_matmul,
)
from ternel_mlx.layout import PackedTensorLayout
from ternel_mlx.packing import pack_blocks, unpack_blocks, validate_packed_codes
from ternel_mlx.reference import (
    decode_reordered_weights,
    get_rows_reference,
    lut23_matmul,
    oracle_matmul,
    restored_matmul,
)

TRITS_PER_SLOT = 5
POWERS = (1, 3, 9, 27, 81)
# The value of an all-zero-trit slot: every trit decodes to weight -1, so the
# positional dot product of byte p is p - 121.
NEUTRAL_BYTE = sum(POWERS)
TAIL_TRITS = BLOCK_SIZE - FULL_CODE_SLOTS * TRITS_PER_SLOT
NEUTRAL_TAIL = sum(POWERS[:TAIL_TRITS])

FAST = MatmulConfig(threads=256, batch_tile=1, padded_lut=True, safe_clamp=False, k_split=1)


def build(trits: np.ndarray, scale_bits: np.ndarray, *, rows: int, columns: int):
    """Pack logical trits and raw FP16 scale bits into the MLX layout."""
    layout = PackedTensorLayout.for_tensor(rows, columns)
    blocks = encode_tq1_blocks(scale_bits, trits).reshape(
        rows, layout.groups_per_row, -1
    )
    codes, scales = pack_blocks(blocks, layout=layout)
    return layout, codes, scales


def ones_scale(groups: int) -> np.ndarray:
    return np.full(groups, 1.0, dtype=np.float16).view(np.uint8).reshape(-1, 2)


def run(codes, scales, x, *, layout, config=FAST):
    out = tq1_matmul(
        mx.array(codes),
        mx.array(scales),
        mx.array(x),
        groups_per_row=layout.groups_per_row,
        tile=layout.tile,
        config=config,
    )
    mx.eval(out)
    return np.array(out, copy=False)


def random_tensor(rows: int, columns: int, seed: int):
    layout = PackedTensorLayout.for_tensor(rows, columns)
    generator = np.random.default_rng(seed)
    scale_bits = (
        generator.normal(0.0, 0.05, size=layout.groups)
        .astype(np.float16)
        .view(np.uint8)
        .reshape(-1, 2)
    )
    trits = generator.integers(0, 3, size=(layout.groups, BLOCK_SIZE), dtype=np.uint8)
    return build(trits, scale_bits, rows=rows, columns=columns)


# --------------------------------------------------------------------------
# Exhaustive byte coverage
# --------------------------------------------------------------------------


def test_all_243_regular_bytes_decode_at_every_slot_position() -> None:
    """Every legal byte, in every one of the 25 regular slots, exactly.

    Group ``g`` carries the byte under test in slot ``g``, and batch sample
    ``g`` has a positional activation that is nonzero only inside group ``g``,
    so one launch covers all 25 slots x 243 bytes with no cross-talk.
    """
    rows = MAX_FULL_CODE_BYTE + 1
    groups = FULL_CODE_SLOTS
    columns = groups * BLOCK_SIZE

    trits = np.ones((rows, groups, BLOCK_SIZE), dtype=np.uint8)
    for value in range(rows):
        digits = np.array([(value // power) % 3 for power in POWERS], dtype=np.uint8)
        for slot in range(groups):
            begin = slot * TRITS_PER_SLOT
            trits[value, slot, begin : begin + TRITS_PER_SLOT] = digits

    layout, codes, scales = build(
        trits.reshape(-1, BLOCK_SIZE), ones_scale(rows * groups), rows=rows, columns=columns
    )
    x = np.zeros((groups, columns), dtype=np.float32)
    for slot in range(groups):
        begin = slot * BLOCK_SIZE + slot * TRITS_PER_SLOT
        x[slot, begin : begin + TRITS_PER_SLOT] = POWERS

    got = run(codes, scales, x, layout=layout)
    want = np.tile(np.arange(rows, dtype=np.float32) - NEUTRAL_BYTE, (groups, 1))
    np.testing.assert_array_equal(got, want)


def test_all_27_tail_bytes_decode_exactly() -> None:
    rows = MAX_TAIL_CODE_BYTE + 1
    columns = BLOCK_SIZE
    trits = np.ones((rows, BLOCK_SIZE), dtype=np.uint8)
    tail_begin = FULL_CODE_SLOTS * TRITS_PER_SLOT
    for value in range(rows):
        for digit, power in enumerate(POWERS[:TAIL_TRITS]):
            trits[value, tail_begin + digit] = (value // power) % 3

    layout, codes, scales = build(trits, ones_scale(rows), rows=rows, columns=columns)
    x = np.zeros((1, columns), dtype=np.float32)
    x[0, tail_begin : tail_begin + TAIL_TRITS] = POWERS[:TAIL_TRITS]

    got = run(codes, scales, x, layout=layout)
    want = (np.arange(rows, dtype=np.float32) - NEUTRAL_TAIL).reshape(1, rows)
    np.testing.assert_array_equal(got, want)


def test_every_fp16_scale_bit_pattern_reaches_the_output_unchanged() -> None:
    """All 65,536 FP16 patterns, including subnormals, infinities and NaNs."""
    rows = 1 << 16
    columns = BLOCK_SIZE
    trits = np.ones((rows, BLOCK_SIZE), dtype=np.uint8)
    trits[:, 0] = 2  # a single +1 weight, so the group sum is exactly 1.0
    scale_bits = np.arange(rows, dtype=np.uint16).view(np.uint8).reshape(-1, 2)

    layout, codes, scales = build(trits, scale_bits, rows=rows, columns=columns)
    x = np.zeros((1, columns), dtype=np.float32)
    x[0, 0] = 1.0

    got = run(codes, scales, x, layout=layout)[0]
    with np.errstate(invalid="ignore"):  # the NaN patterns are the point
        want = np.arange(rows, dtype=np.uint16).view(np.float16).astype(np.float32)

    finite = np.isfinite(want)
    np.testing.assert_array_equal(got[finite], want[finite])
    assert np.all(np.isnan(got[np.isnan(want)]))
    infinite = np.isinf(want)
    np.testing.assert_array_equal(got[infinite], want[infinite])
    # A -0.0 scale comes back as +0.0 because the accumulator starts at +0.0;
    # that is the correct IEEE result of fma(-0.0, 1.0, 0.0) and compares equal.
    assert int(np.count_nonzero(np.isnan(want))) > 0, "NaN patterns must be covered"
    assert int(np.count_nonzero(infinite)) == 2, "both infinities must be covered"


# --------------------------------------------------------------------------
# Reference parity
# --------------------------------------------------------------------------

SHAPES = ((48, 5120), (256, 640), (512, 384), (1024, 5120), (300, 256))


@pytest.mark.parametrize(("rows", "columns"), SHAPES)
@pytest.mark.parametrize("batch", (1, 2, 3, 8))
@pytest.mark.parametrize("batch_tile", (1, 2, 4))
def test_kernel_is_bit_exact_against_the_lut23_reference(
    rows: int, columns: int, batch: int, batch_tile: int
) -> None:
    layout, codes, scales = random_tensor(rows, columns, seed=rows * 31 + columns)
    generator = np.random.default_rng(rows + columns + batch)
    x = generator.normal(0.0, 1.0, size=(batch, columns)).astype(np.float32)
    config = MatmulConfig(
        threads=256, batch_tile=batch_tile, padded_lut=True, safe_clamp=False, k_split=1
    )
    got = run(codes, scales, x, layout=layout, config=config)
    np.testing.assert_array_equal(got, lut23_matmul(codes, scales, x, layout=layout))


@pytest.mark.parametrize("padded_lut", (True, False))
def test_padded_and_unpadded_tables_agree_exactly(padded_lut: bool) -> None:
    layout, codes, scales = random_tensor(512, 640, seed=77)
    generator = np.random.default_rng(77)
    x = generator.normal(0.0, 1.0, size=(4, 640)).astype(np.float32)
    config = MatmulConfig(
        threads=256, batch_tile=2, padded_lut=padded_lut, safe_clamp=False, k_split=1
    )
    got = run(codes, scales, x, layout=layout, config=config)
    np.testing.assert_array_equal(got, lut23_matmul(codes, scales, x, layout=layout))


@pytest.mark.parametrize("safe_clamp", (True, False))
def test_safe_clamp_does_not_change_results_on_legal_data(safe_clamp: bool) -> None:
    layout, codes, scales = random_tensor(256, 384, seed=91)
    generator = np.random.default_rng(91)
    x = generator.normal(0.0, 1.0, size=(2, 384)).astype(np.float32)
    config = MatmulConfig(
        threads=256, batch_tile=2, padded_lut=True, safe_clamp=safe_clamp, k_split=1
    )
    got = run(codes, scales, x, layout=layout, config=config)
    np.testing.assert_array_equal(got, lut23_matmul(codes, scales, x, layout=layout))


def test_all_three_references_agree_within_fp32_rounding() -> None:
    layout, codes, scales = random_tensor(512, 640, seed=5)
    generator = np.random.default_rng(5)
    x = generator.normal(0.0, 1.0, size=(4, 640)).astype(np.float32)

    oracle = oracle_matmul(codes, scales, x, layout=layout)
    magnitude = float(np.abs(oracle).max())
    for name, value in (
        ("restored", restored_matmul(codes, scales, x, layout=layout)),
        ("lut23", lut23_matmul(codes, scales, x, layout=layout)),
        ("metal", run(codes, scales, x, layout=layout)),
    ):
        error = float(np.abs(value.astype(np.float64) - oracle).max())
        assert error / magnitude < 1e-6, f"{name} drifted {error} from the float64 oracle"


@pytest.mark.parametrize("dtype", (mx.float32, mx.float16, mx.bfloat16))
def test_every_activation_dtype_matches_a_reference_in_that_dtype(dtype) -> None:
    layout, codes, scales = random_tensor(256, 384, seed=13)
    generator = np.random.default_rng(13)
    raw = generator.normal(0.0, 1.0, size=(3, 384)).astype(np.float32)
    # Round-trip through the target dtype so the reference sees the same inputs.
    activations = mx.array(raw).astype(dtype)
    mx.eval(activations)
    x = np.array(activations.astype(mx.float32), copy=False)

    out = tq1_matmul(
        mx.array(codes),
        mx.array(scales),
        activations,
        groups_per_row=layout.groups_per_row,
        tile=layout.tile,
        config=FAST,
    )
    mx.eval(out)
    assert out.dtype == dtype
    got = np.array(out.astype(mx.float32), copy=False).astype(np.float64)
    want = lut23_matmul(codes, scales, x, layout=layout).astype(np.float64)
    magnitude = max(float(np.abs(want).max()), 1e-30)
    # Only the final store is narrowed, so the gap is one rounding of the result.
    assert float(np.abs(got - want).max()) / magnitude < 1e-2


@pytest.mark.parametrize(
    "fill", (0.0, 1.0, -1.0, 65504.0, -65504.0, 6.103515625e-05, 1e-8)
)
def test_extreme_and_degenerate_activations_match_the_reference(fill: float) -> None:
    layout, codes, scales = random_tensor(256, 384, seed=29)
    x = np.full((2, 384), fill, dtype=np.float32)
    got = run(codes, scales, x, layout=layout)
    np.testing.assert_array_equal(got, lut23_matmul(codes, scales, x, layout=layout))


@pytest.mark.parametrize("bad", (np.nan, np.inf, -np.inf))
def test_non_finite_activations_propagate_rather_than_corrupt(bad: float) -> None:
    layout, codes, scales = random_tensor(256, 384, seed=31)
    x = np.zeros((1, 384), dtype=np.float32)
    x[0, 0] = bad
    got = run(codes, scales, x, layout=layout)
    assert int(np.count_nonzero(np.isfinite(got))) == 0, (
        "a non-finite activation must reach every output row, not be silently dropped"
    )


# --------------------------------------------------------------------------
# Prefill GEMM
# --------------------------------------------------------------------------

# Three shapes of threadgroup budget -- the small high-occupancy block, a square
# one that stages a quarter of a group, and one whose simdgroups tile the row
# axis four ways -- crossed with all six kernel forms.
#
# The cross is not padding. ``direct_fragments``, ``direct_epilogue`` and
# ``threadgroup_table`` select genuinely different code: one form decodes a byte
# once and spends all five of its trits through a threadgroup tile, the other has
# five lanes each take one trit out of the same byte through a transposed table;
# the two epilogues differ in whether a lane's result reaches device memory
# through threadgroup memory or straight out of the accumulator it already holds;
# and the table flag changes which address space that transposed table is read
# from, which in Metal is part of the pointer's type and so is a different
# fragment loop rather than a different constant. Every one of them must land on
# the same float64 oracle at every shape in :data:`SHAPES`.
#
# ``threadgroup_table`` is only legal with ``direct_fragments`` -- the staged
# fill reads the byte-major table instead and never touches this one -- so the
# cross is six forms, not eight.
GEMM_TILINGS = (
    (32, 32, 16, 2, 2),
    (64, 64, 32, 2, 2),
    (32, 64, 32, 2, 4),
)

GEMM_CONFIGS = tuple(
    GemmConfig(
        batch_block=batch_block,
        row_block=row_block,
        k_block=k_block,
        simd_rows=simd_rows,
        simd_columns=simd_columns,
        direct_fragments=direct_fragments,
        direct_epilogue=direct_epilogue,
        threadgroup_table=threadgroup_table,
        safe_clamp=False,
    )
    for batch_block, row_block, k_block, simd_rows, simd_columns in GEMM_TILINGS
    for direct_fragments in (False, True)
    for direct_epilogue in (False, True)
    for threadgroup_table in ((False, True) if direct_fragments else (False,))
)


def run_gemm(codes, scales, x, *, layout, config):
    out = tq1_gemm(
        mx.array(codes),
        mx.array(scales),
        mx.array(x),
        groups_per_row=layout.groups_per_row,
        tile=layout.tile,
        config=config,
    )
    mx.eval(out)
    return np.array(out, copy=False)


@pytest.mark.parametrize(("rows", "columns"), SHAPES)
@pytest.mark.parametrize("batch", (1, 5, 33, 64, 130))
@pytest.mark.parametrize("config", GEMM_CONFIGS, ids=lambda c: c.benchmark_name)
def test_gemm_matches_the_float64_oracle(
    rows: int, columns: int, batch: int, config: GemmConfig
) -> None:
    """A tolerance, not equality -- and deliberately so.

    ``tq1_matmul`` sums each group's trit contributions and applies the scale
    once at the end; the GEMM folds the scale into every decoded weight and sums
    through the matrix units. Both are honest fp32 evaluations of the same
    product, so they are compared against the float64 oracle rather than against
    each other.

    Both bounds states are covered on both axes, because both are compile-time
    constants and so select different kernels rather than different branches.
    Four of the batches divide no batch block, exercising the zero-filled tail of
    the activation tile and the guarded stores; 64 divides every one of them, so
    the stores run unguarded. On the row axis :data:`SHAPES` supplies 48 and 300,
    which divide no row block, against 256, 512 and 1024, which divide all of
    them.
    """
    layout, codes, scales = random_tensor(rows, columns, seed=rows * 31 + columns)
    if layout.tiles > 1 and layout.tile % config.row_block:
        pytest.skip(f"row block {config.row_block} would straddle two {layout.tile}-row tiles")
    generator = np.random.default_rng(rows + columns + batch)
    x = generator.normal(0.0, 1.0, size=(batch, columns)).astype(np.float32)

    oracle = oracle_matmul(codes, scales, x, layout=layout)
    got = run_gemm(codes, scales, x, layout=layout, config=config)
    magnitude = max(float(np.abs(oracle).max()), 1e-30)
    assert float(np.abs(got.astype(np.float64) - oracle).max()) / magnitude < 1e-5


@pytest.mark.parametrize("safe_clamp", (True, False))
def test_gemm_safe_clamp_does_not_change_results_on_legal_data(safe_clamp: bool) -> None:
    layout, codes, scales = random_tensor(512, 640, seed=97)
    generator = np.random.default_rng(97)
    x = generator.normal(0.0, 1.0, size=(9, 640)).astype(np.float32)
    fast, safe = (
        GemmConfig(
            batch_block=32, row_block=32, k_block=16, simd_rows=2, simd_columns=2,
            direct_fragments=False, direct_epilogue=False, threadgroup_table=False,
            safe_clamp=flag,
        )
        for flag in (False, safe_clamp)
    )
    np.testing.assert_array_equal(
        run_gemm(codes, scales, x, layout=layout, config=fast),
        run_gemm(codes, scales, x, layout=layout, config=safe),
    )


@pytest.mark.parametrize("dtype", (mx.float32, mx.float16, mx.bfloat16))
def test_gemm_returns_the_activation_dtype(dtype) -> None:
    layout, codes, scales = random_tensor(256, 384, seed=101)
    generator = np.random.default_rng(101)
    raw = generator.normal(0.0, 1.0, size=(3, 384)).astype(np.float32)
    activations = mx.array(raw).astype(dtype)
    mx.eval(activations)

    out = tq1_gemm(
        mx.array(codes),
        mx.array(scales),
        activations,
        groups_per_row=layout.groups_per_row,
        tile=layout.tile,
        config=GEMM_CONFIGS[0],
    )
    mx.eval(out)
    assert out.dtype == dtype
    oracle = oracle_matmul(
        codes, scales, np.array(activations.astype(mx.float32), copy=False), layout=layout
    )
    got = np.array(out.astype(mx.float32), copy=False).astype(np.float64)
    # Only the final store is narrowed, so the gap is one rounding of the result.
    assert float(np.abs(got - oracle).max()) / max(float(np.abs(oracle).max()), 1e-30) < 1e-2


def test_gemm_rejects_a_row_block_that_would_straddle_two_tiles() -> None:
    layout, codes, scales = random_tensor(1024, 384, seed=103)
    assert layout.tiles > 1
    # 192 rows is three quarters of a 256-row tile, so the second block of every
    # tile starts inside the next one. It is wider than the threadgroup, which is
    # what keeps the weight fill's row-lane mapping legal -- the point of the
    # test is the tile boundary, not a second constraint failing first.
    config = GemmConfig(
        batch_block=32, row_block=192, k_block=16, simd_rows=1, simd_columns=2,
        direct_fragments=False, direct_epilogue=False, threadgroup_table=False,
        safe_clamp=False,
    )
    assert layout.tile % config.row_block
    with pytest.raises(FormatError, match="straddle"):
        tq1_gemm(
            mx.array(codes),
            mx.array(scales),
            mx.array(np.zeros((1, 384), dtype=np.float32)),
            groups_per_row=layout.groups_per_row,
            tile=layout.tile,
            config=config,
        )


def test_gemm_rejects_mismatched_activations() -> None:
    layout, codes, scales = random_tensor(256, 384, seed=107)
    with pytest.raises(FormatError, match="columns"):
        tq1_gemm(
            mx.array(codes),
            mx.array(scales),
            mx.array(np.zeros((1, 256), dtype=np.float32)),
            groups_per_row=layout.groups_per_row,
            tile=layout.tile,
            config=GEMM_CONFIGS[0],
        )


# --------------------------------------------------------------------------
# Embedding kernel
# --------------------------------------------------------------------------


@pytest.mark.parametrize(("rows", "columns"), ((1024, 5120), (48, 384), (300, 256)))
def test_get_rows_is_bit_exact_against_the_reference(rows: int, columns: int) -> None:
    layout, codes, scales = random_tensor(rows, columns, seed=rows + 3)
    generator = np.random.default_rng(rows)
    token_ids = np.unique(
        np.concatenate(
            [
                np.array([0, rows - 1, rows // 2], dtype=np.uint32),
                generator.integers(0, rows, size=16, dtype=np.uint32),
            ]
        )
    ).astype(np.uint32)

    out = tq1_get_rows(
        mx.array(codes),
        mx.array(scales),
        mx.array(token_ids),
        groups_per_row=layout.groups_per_row,
        tile=layout.tile,
        dtype=mx.float32,
        threads=256,
        safe_clamp=False,
    )
    mx.eval(out)
    want = get_rows_reference(
        codes, scales, token_ids, layout=layout, dtype=np.dtype(np.float32)
    )
    np.testing.assert_array_equal(np.array(out, copy=False), want)


def test_get_rows_reproduces_repeated_indices_identically() -> None:
    layout, codes, scales = random_tensor(512, 256, seed=44)
    token_ids = np.array([7, 7, 511, 0, 7], dtype=np.uint32)
    out = tq1_get_rows(
        mx.array(codes),
        mx.array(scales),
        mx.array(token_ids),
        groups_per_row=layout.groups_per_row,
        tile=layout.tile,
        dtype=mx.float32,
        threads=256,
        safe_clamp=False,
    )
    mx.eval(out)
    got = np.array(out, copy=False)
    np.testing.assert_array_equal(got[0], got[1])
    np.testing.assert_array_equal(got[0], got[4])


# --------------------------------------------------------------------------
# Fail-closed validation
# --------------------------------------------------------------------------


def test_load_time_scan_rejects_a_byte_that_would_overrun_the_l3_table() -> None:
    layout, codes, scales = random_tensor(256, 256, seed=101)
    validate_packed_codes(codes)
    corrupted = codes.copy()
    corrupted[0, 0, 0, 0] = MAX_FULL_CODE_BYTE + 1
    with pytest.raises(FormatError, match="out-of-range base-3 bytes"):
        validate_packed_codes(corrupted)


def test_load_time_scan_rejects_an_illegal_tail_byte() -> None:
    layout, codes, scales = random_tensor(256, 256, seed=103)
    corrupted = codes.copy()
    corrupted[0, 0, FULL_CODE_SLOTS, 5] = MAX_TAIL_CODE_BYTE + 1
    with pytest.raises(FormatError, match="tail=1"):
        validate_packed_codes(corrupted)


def test_safe_clamp_keeps_malformed_bytes_inside_the_tables() -> None:
    """The fuzz build must stay in bounds where the fast path is undefined."""
    layout, codes, scales = random_tensor(256, 256, seed=107)
    corrupted = codes.copy()
    corrupted[:, :, :FULL_CODE_SLOTS, :] = 255
    corrupted[:, :, FULL_CODE_SLOTS, :] = 255
    x = np.ones((1, 256), dtype=np.float32)
    config = MatmulConfig(threads=256, batch_tile=1, padded_lut=True, safe_clamp=True, k_split=1)
    got = run(corrupted, scales, x, layout=layout, config=config)

    clamped = corrupted.copy()
    clamped[:, :, :FULL_CODE_SLOTS, :] = MAX_FULL_CODE_BYTE
    clamped[:, :, FULL_CODE_SLOTS, :] = MAX_TAIL_CODE_BYTE
    np.testing.assert_array_equal(got, lut23_matmul(clamped, scales, x, layout=layout))


def test_pack_round_trip_is_exact_for_arbitrary_scale_bits() -> None:
    rows, columns = 256, 384
    layout = PackedTensorLayout.for_tensor(rows, columns)
    generator = np.random.default_rng(211)
    scale_bits = generator.integers(0, 256, size=(layout.groups, 2), dtype=np.uint8)
    trits = generator.integers(0, 3, size=(layout.groups, BLOCK_SIZE), dtype=np.uint8)
    blocks = encode_tq1_blocks(scale_bits, trits).reshape(rows, layout.groups_per_row, -1)

    codes, scales = pack_blocks(blocks, layout=layout)
    np.testing.assert_array_equal(unpack_blocks(codes, scales, layout=layout), blocks)

    # The stored uint16 is the block's first two bytes read little-endian, with
    # no conversion anywhere: row r of the tensor lives at tile r // tile,
    # position r % tile, so undoing that permutation must reproduce them.
    stored = scales.transpose(0, 2, 1).reshape(rows, layout.groups_per_row)
    original = np.ascontiguousarray(blocks[:, :, :2]).view("<u2").reshape(rows, layout.groups_per_row)
    np.testing.assert_array_equal(stored, original)


def test_packing_is_deterministic() -> None:
    first = random_tensor(256, 384, seed=311)
    second = random_tensor(256, 384, seed=311)
    np.testing.assert_array_equal(first[1], second[1])
    np.testing.assert_array_equal(first[2], second[2])


# --------------------------------------------------------------------------
# Configuration guards
# --------------------------------------------------------------------------


def test_batch_tile_is_capped_by_threadgroup_memory() -> None:
    assert MatmulConfig(threads=256, batch_tile=4, padded_lut=True, safe_clamp=False, k_split=1)
    with pytest.raises(FormatError, match="threadgroup memory"):
        MatmulConfig(threads=256, batch_tile=8, padded_lut=True, safe_clamp=False, k_split=1)
    # Unpadded tables are small enough that a batch tile of 8 does fit.
    assert MatmulConfig(threads=256, batch_tile=8, padded_lut=False, safe_clamp=False, k_split=1)
    with pytest.raises(FormatError, match="threadgroup memory"):
        MatmulConfig(threads=256, batch_tile=16, padded_lut=False, safe_clamp=False, k_split=1)


# --------------------------------------------------------------------------
# The K-axis split
# --------------------------------------------------------------------------

# The group counts the model actually has, and for each *every* legal split of
# it -- a split has to divide the group count, so this is the complete set the
# kernel can ever be asked for at that count.
#
# Exhaustive rather than sampled because the alternative is a coverage claim
# with a moving target. Which splits ship is decided by
# ``SPLIT_K_THREADGROUPS`` against each tensor's threadgroup count, so a sample
# chosen to match today's rule silently stops covering it the moment that
# ceiling moves -- and it has moved once already, from 320 to 800, which changed
# five of the nine shapes' splits. Enumerating the divisors costs 17 distinct
# template instantiations per batch tile and makes the coverage independent of
# the rule.
GROUP_COUNTS = (40, 48, 136)
SPLIT_CASES = tuple(
    (groups, tuple(k for k in range(2, groups + 1) if groups % k == 0))
    for groups in GROUP_COUNTS
)


def split_config(k_split: int, *, batch_tile: int = 1) -> MatmulConfig:
    return MatmulConfig(
        threads=256, batch_tile=batch_tile, padded_lut=True, safe_clamp=False, k_split=k_split
    )


@pytest.mark.parametrize(("groups_per_row", "splits"), SPLIT_CASES)
def test_splitting_the_k_axis_does_not_change_exact_arithmetic(
    groups_per_row: int, splits: tuple[int, ...]
) -> None:
    """Where the sum is exact, reordering it must change nothing at all.

    Splitting K reassociates the accumulation: the unsplit kernel adds all ``G``
    group sums into one fp32 accumulator in order, while a split of ``k`` adds
    ``k`` contiguous runs and then adds those. Reassociating fp32 addition is
    normally visible in the last bits, which would let a real boundary bug hide
    inside a tolerance.

    So the arithmetic is made exact instead. Every scale is 1.0 and every
    activation is +1 or -1, which makes each group sum an integer in
    ``[-128, 128]`` and each row total an integer no larger than ``136 * 128``
    -- far inside fp32's exactly-representable range. Any difference between a
    split and the unsplit answer is then a difference in *which* weights were
    summed, not in how they rounded, and ``assert_array_equal`` catches it.
    """
    rows, columns = 256, groups_per_row * BLOCK_SIZE
    layout = PackedTensorLayout.for_tensor(rows, columns)
    generator = np.random.default_rng(groups_per_row)
    ones = np.full(layout.groups, 1.0, dtype=np.float16).view(np.uint8).reshape(-1, 2)
    trits = generator.integers(0, 3, size=(layout.groups, BLOCK_SIZE), dtype=np.uint8)
    layout, codes, scales = build(trits, ones, rows=rows, columns=columns)

    x = generator.choice((-1.0, 1.0), size=(3, columns)).astype(np.float32)
    whole = run(codes, scales, x, layout=layout, config=split_config(1))
    # The premise of the construction: nothing here needed rounding.
    assert np.array_equal(whole, np.rint(whole))
    assert np.abs(whole).max() <= groups_per_row * BLOCK_SIZE

    for k_split in splits:
        got = run(codes, scales, x, layout=layout, config=split_config(k_split))
        np.testing.assert_array_equal(got, whole, err_msg=f"k_split={k_split}")


@pytest.mark.parametrize(("groups_per_row", "splits"), SPLIT_CASES)
@pytest.mark.parametrize("batch_tile", (1, 4))
def test_a_split_stays_inside_the_fp32_summation_bound(
    groups_per_row: int, splits: tuple[int, ...], batch_tile: int
) -> None:
    """On ordinary data a split rounds differently, and the difference is bounded.

    A tolerance taken from the unsplit run would be circular here, and one taken
    from the answer's own magnitude would be wrong: these dot products cancel, so
    the terms are far larger than the total and the rounding is set by the terms.
    The bound used instead is the textbook one for summing ``n`` floats in
    sequence -- ``n * eps * sum|terms|`` -- with ``sum|terms|`` computed exactly
    as ``|W| @ |x|`` in float64 and ``n`` the longest chain either arm can have,
    the 128 weights of a group plus the ``G`` group sums after it. Splitting only
    reassociates that chain, and never lengthens it, so the same bound holds for
    every arm and is not tuned to any of them.

    It is loose by construction -- the real error uses under a tenth of a percent
    of it, because rounding accumulates as a walk rather than in one direction --
    and it still has teeth. A chunk boundary that drops or repeats a group is
    caught if any single element notices, and measuring every group of these
    three shapes that way, even the quietest lands 154x outside its element's
    bound. The exact-arithmetic test above is what pins the boundaries down;
    this one asks whether ordinary floating-point data stays sane.
    """
    rows, columns = 256, groups_per_row * BLOCK_SIZE
    layout, codes, scales = random_tensor(rows, columns, seed=groups_per_row * 7 + batch_tile)
    generator = np.random.default_rng(groups_per_row + batch_tile)
    x = generator.normal(0.0, 1.0, size=(4, columns)).astype(np.float32)

    oracle = oracle_matmul(codes, scales, x, layout=layout)
    weights = decode_reordered_weights(codes, scales, layout=layout, dtype=np.dtype(np.float64))
    terms = np.abs(x.astype(np.float64)) @ np.abs(weights).T
    bound = (BLOCK_SIZE + groups_per_row) * float(np.finfo(np.float32).eps) * terms

    whole = run(codes, scales, x, layout=layout, config=split_config(1, batch_tile=batch_tile))
    unsplit = float(np.abs(whole - oracle).max())
    assert unsplit <= float(bound.max()), f"the unsplit arm itself is outside the bound: {unsplit}"

    for k_split in splits:
        got = run(
            codes, scales, x, layout=layout, config=split_config(k_split, batch_tile=batch_tile)
        )
        error = np.abs(got - oracle)
        assert (error <= bound).all(), (
            f"k_split={k_split} bt={batch_tile} exceeded the summation bound by "
            f"{float((error / bound).max()):.3f}x; the unsplit arm reached {unsplit:.3e}"
        )


@pytest.mark.parametrize("k_split", (2, 8, 40))
def test_a_split_result_is_bit_identical_between_runs(k_split: int) -> None:
    """The reduction is ``mx.sum``, not a float atomic, for exactly this reason.

    Atomic adds arrive in whatever order the scheduler produces, so a rerun of
    the same prompt would round differently and a greedy continuation could
    diverge from itself. The artifact gate asserts identical continuations, so
    that would not be a small cost.
    """
    layout, codes, scales = random_tensor(256, 5120, seed=k_split + 900)
    generator = np.random.default_rng(k_split)
    x = generator.normal(0.0, 1.0, size=(2, 5120)).astype(np.float32)
    config = split_config(k_split)
    first = run(codes, scales, x, layout=layout, config=config)
    for _ in range(3):
        np.testing.assert_array_equal(run(codes, scales, x, layout=layout, config=config), first)


def test_a_split_that_does_not_divide_the_group_count_is_rejected() -> None:
    """Ragged chunks are refused rather than padded.

    A ceiling division would leave the last threadgroup holding fewer groups --
    or none -- while the rest hold a full chunk, and a dispatch ends when its
    slowest threadgroup does. The extra threadgroups would shorten nothing, so
    the caller is told its split is not one rather than quietly given it.
    """
    layout, codes, scales = random_tensor(256, 5120, seed=311)
    x = np.zeros((1, 5120), dtype=np.float32)
    assert layout.groups_per_row == 40
    with pytest.raises(FormatError, match="does not divide the tensor's 40 groups"):
        run(codes, scales, x, layout=layout, config=split_config(3))


def test_a_split_names_itself_apart_from_the_sweep_it_was_added_after() -> None:
    assert split_config(1).benchmark_name == "lut23_bt1_padded"
    assert split_config(8).benchmark_name == "lut23_bt1_padded_k8"


@pytest.mark.parametrize("k_split", (0, -1))
def test_a_non_positive_split_is_rejected(k_split: int) -> None:
    with pytest.raises(FormatError, match="k split must be positive"):
        split_config(k_split)


def test_threadgroup_budget_matches_the_documented_table_sizes() -> None:
    assert lut_floats(padded=True) == FULL_CODE_SLOTS * 32 * 2 + 27
    assert lut_floats(padded=False) == FULL_CODE_SLOTS * 9 + FULL_CODE_SLOTS * 27 + 27
    assert lut_floats(padded=True) * 4 < THREADGROUP_MEMORY_BYTES


@pytest.mark.parametrize("threads", (0, -32, 100))
def test_thread_count_must_be_a_whole_number_of_simdgroups(threads: int) -> None:
    with pytest.raises(FormatError, match="simdgroup"):
        MatmulConfig(threads=threads, batch_tile=1, padded_lut=True, safe_clamp=False, k_split=1)


def test_matmul_rejects_mismatched_shapes_and_dtypes() -> None:
    layout, codes, scales = random_tensor(256, 384, seed=401)
    x = np.zeros((1, 384), dtype=np.float32)
    with pytest.raises(FormatError, match="columns"):
        tq1_matmul(
            mx.array(codes),
            mx.array(scales),
            mx.array(np.zeros((1, 256), dtype=np.float32)),
            groups_per_row=layout.groups_per_row,
            tile=layout.tile,
            config=FAST,
        )
    with pytest.raises(FormatError, match="scales must be uint16"):
        tq1_matmul(
            mx.array(codes),
            mx.array(scales.astype(np.uint32)),
            mx.array(x),
            groups_per_row=layout.groups_per_row,
            tile=layout.tile,
            config=FAST,
        )
    with pytest.raises(FormatError, match="codes shape"):
        tq1_matmul(
            mx.array(codes),
            mx.array(scales),
            mx.array(x),
            groups_per_row=layout.groups_per_row + 1,
            tile=layout.tile,
            config=FAST,
        )
    with pytest.raises(FormatError, match="must be 2-D"):
        tq1_matmul(
            mx.array(codes),
            mx.array(scales),
            mx.array(x.reshape(1, 1, 384)),
            groups_per_row=layout.groups_per_row,
            tile=layout.tile,
            config=FAST,
        )


def test_get_rows_rejects_wrong_index_dtype() -> None:
    layout, codes, scales = random_tensor(256, 256, seed=409)
    with pytest.raises(FormatError, match="indices must be"):
        tq1_get_rows(
            mx.array(codes),
            mx.array(scales),
            mx.array(np.array([0, 1], dtype=np.int32)),
            groups_per_row=layout.groups_per_row,
            tile=layout.tile,
            dtype=mx.float32,
            threads=256,
            safe_clamp=False,
        )


def test_gemm_k_block_must_divide_a_whole_weight_group() -> None:
    """A chunk that straddled two groups would need two scales at once."""
    assert GemmConfig(
        batch_block=32, row_block=32, k_block=BLOCK_SIZE, simd_rows=2, simd_columns=2,
        direct_fragments=False, direct_epilogue=False, threadgroup_table=False,
        safe_clamp=False,
    )
    with pytest.raises(FormatError, match="does not divide"):
        GemmConfig(
            batch_block=32, row_block=32, k_block=48, simd_rows=2, simd_columns=2,
            direct_fragments=False, direct_epilogue=False, threadgroup_table=False,
            safe_clamp=False,
        )
    with pytest.raises(FormatError, match="matrix tile"):
        GemmConfig(
            batch_block=32, row_block=32, k_block=4, simd_rows=2, simd_columns=2,
            direct_fragments=False, direct_epilogue=False, threadgroup_table=False,
            safe_clamp=False,
        )


def test_gemm_blocks_must_cover_their_simdgroups_in_whole_matrix_tiles() -> None:
    with pytest.raises(FormatError, match="batch block"):
        GemmConfig(
            batch_block=8, row_block=32, k_block=16, simd_rows=2, simd_columns=2,
            direct_fragments=False, direct_epilogue=False, threadgroup_table=False,
            safe_clamp=False,
        )
    with pytest.raises(FormatError, match="row block"):
        GemmConfig(
            batch_block=32, row_block=16, k_block=16, simd_rows=2, simd_columns=4,
            direct_fragments=False, direct_epilogue=False, threadgroup_table=False,
            safe_clamp=False,
        )
    with pytest.raises(FormatError, match="simd columns must be positive"):
        GemmConfig(
            batch_block=32, row_block=32, k_block=16, simd_rows=2, simd_columns=0,
            direct_fragments=False, direct_epilogue=False, threadgroup_table=False,
            safe_clamp=False,
        )


def test_gemm_threadgroup_budget_is_the_larger_of_staging_and_write_back() -> None:
    base = {
        "batch_block": 64, "row_block": 64, "k_block": 32,
        "simd_rows": 2, "simd_columns": 2,
        "direct_fragments": False, "direct_epilogue": False,
        "threadgroup_table": False, "safe_clamp": False,
    }

    def config(**overrides: int | bool) -> GemmConfig:
        return GemmConfig(**(base | overrides))

    # Staging dominates: a quarter-group of both tiles beats one 8-row strip.
    # Activations cost four bytes each and the decoded weights two.
    staging = config()
    assert staging.threadgroup_bytes == 32 * 64 * 4 + 32 * 64 * 2
    assert staging.threads == 128
    assert staging.accumulators_per_thread == 64 * 64 // 128

    # Write-back dominates: an 8-trit K chunk stages less than the strips need.
    strips = config(row_block=128, k_block=8, simd_columns=4)
    assert strips.threadgroup_bytes == 2 * MATRIX_TILE * 128 * 4

    assert (
        gemm_threadgroup_bytes(
            batch_block=64, row_block=64, k_block=32, simd_rows=2,
            direct_fragments=False, direct_epilogue=False, threadgroup_table=False,
        )
        == staging.threadgroup_bytes
    )

    # Each direct form drops its own claim, and with both dropped the kernel
    # allocates no threadgroup memory at all.
    assert config(direct_fragments=True).threadgroup_bytes == 2 * MATRIX_TILE * 64 * 4
    assert config(direct_epilogue=True).threadgroup_bytes == 32 * 64 * 4 + 32 * 64 * 2
    assert config(direct_fragments=True, direct_epilogue=True).threadgroup_bytes == 0

    # The trit table is not one of the alternatives. Staging and the write-back
    # strips take turns using the same bytes, so the budget is the larger of the
    # two; the table is live across the whole kernel and so adds to whichever
    # won. With both direct forms on it is the only claim left.
    table = TRITS_PER_FULL_BYTE * CODE_TABLE_ENTRIES * 2
    assert config(direct_fragments=True, threadgroup_table=True).threadgroup_bytes == (
        2 * MATRIX_TILE * 64 * 4 + table
    )
    assert config(
        direct_fragments=True, direct_epilogue=True, threadgroup_table=True
    ).threadgroup_bytes == table

    # And it cannot be asked for by a form that never reads it.
    with pytest.raises(FormatError, match="register form"):
        config(threadgroup_table=True)

    with pytest.raises(FormatError, match="threadgroup memory"):
        GemmConfig(
            batch_block=128, row_block=128, k_block=BLOCK_SIZE, simd_rows=2,
            simd_columns=4, direct_fragments=False, direct_epilogue=False,
            threadgroup_table=False, safe_clamp=False,
        )
    # ...and that same tiling is legal once nothing is staged, which is what
    # makes the largest output blocks reachable at all.
    assert GemmConfig(
        batch_block=128, row_block=128, k_block=BLOCK_SIZE, simd_rows=2,
        simd_columns=4, direct_fragments=True, direct_epilogue=True,
        threadgroup_table=False, safe_clamp=False,
    ).threadgroup_bytes == 0


def test_code_slot_count_is_the_frozen_28_byte_block() -> None:
    assert CODE_SLOTS == FULL_CODE_SLOTS + 1
    assert FULL_CODE_SLOTS * TRITS_PER_SLOT + TAIL_TRITS == BLOCK_SIZE


# The two register forms of the GEMM stand on one hardware fact: in an 8x8
# simdgroup fragment, lane L owns row `fm` and columns `fn`, `fn + 1` under the
# map in `_GEMM_LANE_MAP`. Nothing in Metal's contract promises that map, and if
# it were wrong the kernel would still compile and still produce plausible
# numbers. So it is measured on this device, from the shipped source string
# rather than a copy of it, in both directions the kernel uses:
#
#   fill   -- lay values into registers by the map, multiply through the matrix
#             unit, and store with the hardware's own `simdgroup_store`. This is
#             what `_GEMM_DIRECT_FRAGMENTS` does.
#   read   -- take the accumulator's elements straight out of registers and
#             place them by the same map. This is what `_GEMM_DIRECT_EPILOGUE`
#             does, and it is the inverse claim.
#
# The two outputs are compared against the same host product, so a map that was
# self-consistently wrong in both directions still fails: `simdgroup_store` is
# the hardware's opinion, and it is the tiebreaker.
_LANE_MAP_PROBE = mx.fast.metal_kernel(
    name="tq1_lane_map_probe",
    input_names=["a", "b"],
    output_names=["stored", "owned"],
    header="#include <metal_simdgroup_matrix>\n",
    source=_GEMM_LANE_MAP + """
    simdgroup_float8x8 afrag;
    simdgroup_float8x8 bfrag;
    simdgroup_float8x8 acc = simdgroup_float8x8(0);

    thread float2 &avals = reinterpret_cast<thread float2 &>(afrag.thread_elements());
    thread float2 &bvals = reinterpret_cast<thread float2 &>(bfrag.thread_elements());
    avals.x = a[fm * 8u + fn];
    avals.y = a[fm * 8u + fn + 1u];
    bvals.x = b[fm * 8u + fn];
    bvals.y = b[fm * 8u + fn + 1u];

    simdgroup_multiply_accumulate(acc, afrag, bfrag, acc);

    simdgroup_store(acc, stored, 8);

    thread float2 &cvals = reinterpret_cast<thread float2 &>(acc.thread_elements());
    owned[lane * 2u] = cvals.x;
    owned[lane * 2u + 1u] = cvals.y;
""",
)


def test_simdgroup_fragment_lane_map_holds_on_this_device() -> None:
    generator = np.random.default_rng(20260810)
    a = generator.normal(0.0, 1.0, size=(8, 8)).astype(np.float32)
    b = generator.normal(0.0, 1.0, size=(8, 8)).astype(np.float32)

    stored, owned = _LANE_MAP_PROBE(
        inputs=[mx.array(a), mx.array(b)],
        output_shapes=[(8, 8), (32, 2)],
        output_dtypes=[mx.float32, mx.float32],
        grid=(32, 1, 1),
        threadgroup=(32, 1, 1),
    )
    mx.eval(stored, owned)

    expected = a @ b
    # The matrix unit accumulates in its own order, so this is a tolerance and
    # not an equality -- but a permuted lane map is off by whole matrix entries,
    # which no tolerance this tight absorbs.
    assert np.allclose(np.asarray(stored), expected, rtol=1e-5, atol=1e-5)

    # And the same product read straight out of the accumulator registers.
    from_registers = np.empty((8, 8), dtype=np.float32)
    owned_np = np.asarray(owned)
    for lane in range(32):
        quad = lane // 4
        fm = (quad & 4) + ((lane // 2) % 4)
        fn = (quad & 2) * 2 + (lane % 2) * 2
        from_registers[fm, fn] = owned_np[lane, 0]
        from_registers[fm, fn + 1] = owned_np[lane, 1]
    assert np.allclose(from_registers, expected, rtol=1e-5, atol=1e-5)

    # Every element was claimed by exactly one lane: 32 lanes x 2 elements with
    # no collision covers the whole 8x8 tile, so the map is a bijection and not
    # merely right at the positions this product happened to distinguish.
    positions = [
        (
            ((lane // 4) & 4) + ((lane // 2) % 4),
            ((lane // 4) & 2) * 2 + (lane % 2) * 2 + offset,
        )
        for lane in range(32)
        for offset in (0, 1)
    ]
    assert len(set(positions)) == MATRIX_TILE * MATRIX_TILE
