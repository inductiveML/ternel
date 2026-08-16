"""Metal kernels that execute TQ1_G128 packed weights directly.

Nothing here ever materialises a ternary weight, a dequantised row, or an
unpacked copy of any tensor: the only model-sized arrays that exist are the
``codes`` (uint8) and ``scales`` (uint16) buffers of the packed artifact.

**LUT23.** A regular code byte ``p <= 242`` packs five trits with weights
``1, 3, 9, 27, 81``. Splitting it as ``p = lo + 9 * hi`` separates the first two
trits (``lo``, 9 possibilities) from the last three (``hi``, 27 possibilities),
so the byte's contribution to a dot product is ``L2[slot][lo] + L3[slot][hi]``
where the two tables are built once per 128-weight group from the activations.
``hi = (p * 57) >> 9`` is exactly ``p / 9`` for every ``p <= 511``. The tail byte
``p <= 26`` packs three trits and indexes a single 27-entry table directly.
Five ternary weights are therefore never materialised, and a whole 128-weight
group costs 26 byte loads and 51 table lookups instead of 128 of each.

**Precision.** Table entries, the per-group sum and the accumulator are all
fp32, and the scale is applied exactly once per 128-weight group as
``fma(as_type<half>(scale_bits), group_sum, acc)`` — the raw FP16 scale bits
travel from the source file into the fma without ever being converted. The
summation order below is the contract that ``reference.py`` mirrors.

**Bounds safety.** ``hi`` for an illegal ``p = 255`` would be 28, reading past
the 27-entry L3 table. Legality is enforced fail-closed *outside* the kernel —
at conversion by ``decode_tq1_blocks(validate=True)`` and at artifact load by a
full streaming scan — so the fast path carries no clamp. ``SAFE`` compiles one
in for fuzz tests that deliberately feed malformed bytes.

The plan's "kernel 1 (gemv)" and "kernel 2 (batched gemm)" are the ``BT = 1``
and ``BT > 1`` instantiations of the single source below. Each template tuple is
a separately JIT-compiled Metal library either way, so splitting the source in
two would have produced the same machine code from twice the surface area.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import mlx.core as mx

from bonsai_tq1.format import (
    BLOCK_SIZE,
    MAX_FULL_CODE_BYTE,
    MAX_TAIL_CODE_BYTE,
    FormatError,
)
from bonsai_tq1.lut23_reorder import CODE_SLOTS, FULL_CODE_SLOTS

# Table sizes: two trits -> 9 entries, three trits -> 27.
L2_ENTRIES = 9
L3_ENTRIES = 27
TAIL_ENTRIES = 27

# Padding each slot's table to a 32-float stride puts its 9 or 27 entries in
# distinct threadgroup-memory banks, so a simdgroup gathering different lo/hi
# broadcasts instead of serialising. It costs threadgroup memory, which is what
# caps the batch tile, so both variants are built and the choice is measured.
PADDED_STRIDE = 32

# Apple GPUs expose 32 KiB of threadgroup memory per threadgroup.
THREADGROUP_MEMORY_BYTES = 32 * 1024

# One thread per output row; 256 is a full 8 simdgroups.
DEFAULT_THREADS = 256

# Metal's simdgroup width, and the edge of the 8x8 matrices its matrix units
# multiply. Both are fixed by the hardware, not chosen.
SIMD_WIDTH = 32
MATRIX_TILE = 8

ACTIVATION_DTYPES = (mx.float32, mx.float16, mx.bfloat16)
INDEX_DTYPE = mx.uint32

# Every value a code byte can take, so the byte itself is the index.
CODE_TABLE_ENTRIES = 256

# Trits in a regular code byte. The tail byte carries three of the five.
TRITS_PER_FULL_BYTE = 5


def _trit_table_source() -> str:
    """The base-3 decode as a constant table, indexed by the code byte.

    The prefill kernel unpacked each byte with five divisions. The divisors are
    literals so the compiler folded them into multiply-shifts, but that is still
    a dependent chain per byte, re-run for every K step that touches the byte.
    A table replaces it with one vector load and one scalar load.

    Entries hold the digits already in ``t - 1`` form, so the kernel's only
    remaining arithmetic is the multiply by the group scale -- which stays out
    of the table, since the table is per byte and the scale is per row.

    Bytes above the largest legal one cannot occur: conversion and artifact load
    both reject them before any kernel runs. They are filled with zero anyway,
    the one value that could not be mistaken for a decoded weight.
    """
    low = []
    high = []
    for code in range(CODE_TABLE_ENTRIES):
        digits = (
            (code % 3, (code // 3) % 3, (code // 9) % 3, (code // 27) % 3, code // 81)
            if code <= MAX_FULL_CODE_BYTE
            else (1, 1, 1, 1, 1)
        )
        low.append("half4({:.1f}h, {:.1f}h, {:.1f}h, {:.1f}h)".format(*[d - 1.0 for d in digits[:4]]))
        high.append(f"{digits[4] - 1.0:.1f}h")

    def rows(entries: list[str], per_line: int) -> str:
        return "\n".join(
            "    " + ", ".join(entries[i : i + per_line]) + ","
            for i in range(0, len(entries), per_line)
        )

    return (
        f"constant half4 TQ1_TRIT_LO[{CODE_TABLE_ENTRIES}] = {{\n"
        + rows(low, 4)
        + f"\n}};\n\nconstant half TQ1_TRIT_HI[{CODE_TABLE_ENTRIES}] = {{\n"
        + rows(high, 16)
        + "\n};"
    )


def _direct_trit_table_source() -> str:
    """The same decode transposed, so the digit picks the table and not the row.

    ``TQ1_TRIT_LO``/``TQ1_TRIT_HI`` are byte-major, which is what the staged fill
    wants: one lookup hands it all five digits of a byte and it spends all five.

    A lane building a fragment in registers wants the opposite. Its position in
    the 8x8 fixes one K column, so it needs one digit -- the same digit -- across
    many different code bytes, and that digit is a runtime value. Indexing a
    ``half4`` by a runtime digit would force the vector into memory on every use.
    Transposed, the digit selects a table pointer once per K step and each row
    costs a single scalar load from it.

    Illegal bytes decode to zero here exactly as they do in the byte-major table,
    so the two agree on every input rather than only on legal ones.
    """
    tables = []
    for digit in range(TRITS_PER_FULL_BYTE):
        entries = [
            f"{((code // 3**digit) % 3) - 1.0:.1f}h" if code <= MAX_FULL_CODE_BYTE else "0.0h"
            for code in range(CODE_TABLE_ENTRIES)
        ]
        body = "\n".join(
            "        " + ", ".join(entries[i : i + 16]) + ("," if i + 16 < len(entries) else "")
            for i in range(0, len(entries), 16)
        )
        tables.append("    {\n" + body + "\n    }")
    return (
        f"constant half TQ1_DIRECT_TRIT[{TRITS_PER_FULL_BYTE}][{CODE_TABLE_ENTRIES}] = {{\n"
        + ",\n".join(tables)
        + "\n};"
    )


_HEADER = f"""
// The prefill kernel multiplies through the GPU's matrix units.
#include <metal_simdgroup_matrix>

constant int TQ1_BLOCK = {BLOCK_SIZE};
constant int TQ1_CODE_SLOTS = {CODE_SLOTS};
constant int TQ1_FULL_SLOTS = {FULL_CODE_SLOTS};
constant int TQ1_L2_ENTRIES = {L2_ENTRIES};
constant int TQ1_L3_ENTRIES = {L3_ENTRIES};
constant int TQ1_TAIL_ENTRIES = {TAIL_ENTRIES};
constant int TQ1_PADDED_STRIDE = {PADDED_STRIDE};

// Largest legal byte values; only the SAFE fuzz build consults them.
constant uint TQ1_MAX_FULL_BYTE = {MAX_FULL_CODE_BYTE}u;
constant uint TQ1_MAX_TAIL_BYTE = {MAX_TAIL_CODE_BYTE}u;

constant uint TQ1_POW3[5] = {{1u, 3u, 9u, 27u, 81u}};

{_trit_table_source()}

constexpr int tq1_l2_stride(bool padded) {{
    return padded ? TQ1_PADDED_STRIDE : TQ1_L2_ENTRIES;
}}
constexpr int tq1_l3_stride(bool padded) {{
    return padded ? TQ1_PADDED_STRIDE : TQ1_L3_ENTRIES;
}}
constexpr int tq1_l3_base(bool padded) {{
    return TQ1_FULL_SLOTS * tq1_l2_stride(padded);
}}
constexpr int tq1_tail_base(bool padded) {{
    return tq1_l3_base(padded) + TQ1_FULL_SLOTS * tq1_l3_stride(padded);
}}
constexpr int tq1_lut_floats(bool padded) {{
    return tq1_tail_base(padded) + TQ1_TAIL_ENTRIES;
}}

// Build one group's LUT23 tables from the 128 activations of that group.
// Entry index *is* the base-3 digit vector it stands for, so no trit is ever
// decoded here either: entry `c` of L2 holds (c%3-1)*x0 + (c/3-1)*x1.
template <typename T, bool PAD>
inline void tq1_build_lut(
    threadgroup float *lut,
    const device T *activations,
    uint lane,
    uint lanes)
{{
    const int l2_stride = tq1_l2_stride(PAD);
    const int l3_stride = tq1_l3_stride(PAD);
    const int l3_base = tq1_l3_base(PAD);
    const int tail_base = tq1_tail_base(PAD);

    for (uint e = lane; e < uint(TQ1_FULL_SLOTS * TQ1_L2_ENTRIES); e += lanes) {{
        uint slot = e / uint(TQ1_L2_ENTRIES);
        uint code = e - slot * uint(TQ1_L2_ENTRIES);
        float x0 = float(activations[5u * slot]);
        float x1 = float(activations[5u * slot + 1u]);
        lut[slot * uint(l2_stride) + code] =
            (float(code % 3u) - 1.0f) * x0 + (float(code / 3u) - 1.0f) * x1;
    }}
    for (uint e = lane; e < uint(TQ1_FULL_SLOTS * TQ1_L3_ENTRIES); e += lanes) {{
        uint slot = e / uint(TQ1_L3_ENTRIES);
        uint code = e - slot * uint(TQ1_L3_ENTRIES);
        float x2 = float(activations[5u * slot + 2u]);
        float x3 = float(activations[5u * slot + 3u]);
        float x4 = float(activations[5u * slot + 4u]);
        lut[uint(l3_base) + slot * uint(l3_stride) + code] =
            (float(code % 3u) - 1.0f) * x2
            + (float((code / 3u) % 3u) - 1.0f) * x3
            + (float(code / 9u) - 1.0f) * x4;
    }}
    // The tail byte covers activations 125..127 and indexes its table directly.
    for (uint e = lane; e < uint(TQ1_TAIL_ENTRIES); e += lanes) {{
        float x0 = float(activations[125]);
        float x1 = float(activations[126]);
        float x2 = float(activations[127]);
        lut[uint(tail_base) + e] =
            (float(e % 3u) - 1.0f) * x0
            + (float((e / 3u) % 3u) - 1.0f) * x1
            + (float(e / 9u) - 1.0f) * x2;
    }}
}}
"""

# Template parameters: T activation/output dtype, G groups per row, TILE rows
# per tile, TG threads per threadgroup, BT vectors per batch tile, PAD whether
# the tables are bank-padded, SAFE whether illegal bytes are clamped.
_MATMUL_SOURCE = """
    const int lut_floats = tq1_lut_floats(PAD);
    const int l2_stride = tq1_l2_stride(PAD);
    const int l3_stride = tq1_l3_stride(PAD);
    const int l3_base = tq1_l3_base(PAD);
    const int tail_base = tq1_tail_base(PAD);
    const int rows_per_thread = (TILE + TG - 1) / TG;
    const uint columns = uint(G) * uint(TQ1_BLOCK);
    // Split-K. A threadgroup owns one contiguous run of groups instead of all G
    // of them, so a tensor whose row count only yields a handful of row tiles
    // still launches KS times as many threadgroups. Nothing is recomputed: the
    // chunks partition the K axis, so each group's table is still built exactly
    // once per row tile. What it costs is that the KS partial sums have to be
    // added up afterwards, by the caller, in a second pass.
    //
    // At KS = 1 every expression below folds to what it was: one chunk starting
    // at group zero and ending at G, written at output offset zero.
    const int groups_per_chunk = (G + KS - 1) / KS;

    threadgroup float lut[BT * tq1_lut_floats(PAD)];

    const uint lane = thread_position_in_threadgroup.x;
    const uint tile_index = threadgroup_position_in_grid.x;
    const uint batch_tile = threadgroup_position_in_grid.y;
    const uint k_chunk = threadgroup_position_in_grid.z;
    const uint first_group = k_chunk * uint(groups_per_chunk);
    const uint last_group = min(uint(G), first_group + uint(groups_per_chunk));
    const uint batch = uint(x_shape[0]);
    const uint rows = uint(codes_shape[0]) * uint(TILE);

    float acc[rows_per_thread][BT];
    for (int r = 0; r < rows_per_thread; ++r) {
        for (int j = 0; j < BT; ++j) { acc[r][j] = 0.0f; }
    }

    for (uint g = first_group; g < last_group; ++g) {
        // Guards the previous iteration's table reads against this one's writes.
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (int j = 0; j < BT; ++j) {
            // A tail batch tile reads a real row and discards it at write time,
            // which keeps the branch out of the table build entirely.
            uint sample = min(batch_tile * uint(BT) + uint(j), batch - 1u);
            auto activations = x + ulong(sample) * ulong(columns) + ulong(g) * TQ1_BLOCK;
            tq1_build_lut<T, PAD>(lut + j * lut_floats, activations, lane, uint(TG));
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        const ulong group_base = (ulong(tile_index) * uint(G) + g);
        auto group_codes = codes + group_base * uint(TQ1_CODE_SLOTS) * uint(TILE);
        auto group_scales = scales + group_base * uint(TILE);

        for (int r = 0; r < rows_per_thread; ++r) {
            uint row_in_tile = lane + uint(r) * uint(TG);
            if (row_in_tile >= uint(TILE)) { break; }

            float sum[BT];
            for (int j = 0; j < BT; ++j) { sum[j] = 0.0f; }

            #pragma clang loop unroll(full)
            for (int s = 0; s < TQ1_FULL_SLOTS; ++s) {
                uint code = uint(group_codes[uint(s) * uint(TILE) + row_in_tile]);
                if (SAFE) { code = min(code, TQ1_MAX_FULL_BYTE); }
                uint hi = (code * 57u) >> 9;
                uint lo = code - 9u * hi;
                for (int j = 0; j < BT; ++j) {
                    threadgroup const float *table = lut + j * lut_floats;
                    sum[j] += table[uint(s * l2_stride) + lo]
                        + table[uint(l3_base + s * l3_stride) + hi];
                }
            }

            uint tail = uint(group_codes[uint(TQ1_FULL_SLOTS) * uint(TILE) + row_in_tile]);
            if (SAFE) { tail = min(tail, TQ1_MAX_TAIL_BYTE); }
            // The raw FP16 scale bits reach the fma without a conversion step.
            float scale = float(as_type<half>(group_scales[row_in_tile]));
            for (int j = 0; j < BT; ++j) {
                sum[j] += lut[j * lut_floats + uint(tail_base) + tail];
                acc[r][j] = fma(scale, sum[j], acc[r][j]);
            }
        }
    }

    for (int r = 0; r < rows_per_thread; ++r) {
        uint row_in_tile = lane + uint(r) * uint(TG);
        if (row_in_tile >= uint(TILE)) { break; }
        uint row = tile_index * uint(TILE) + row_in_tile;
        if (row >= rows) { break; }
        for (int j = 0; j < BT; ++j) {
            uint sample = batch_tile * uint(BT) + uint(j);
            if (sample < batch) {
                // At KS = 1 the chunk term vanishes and this is the plain
                // (sample, row) store the single-pass kernel always did. Above
                // that, out carries a leading chunk axis for the caller to sum.
                out[(ulong(k_chunk) * ulong(batch) + ulong(sample)) * ulong(rows) + row]
                    = static_cast<OT>(acc[r][j]);
            }
        }
    }
"""

# Template parameters: T activation/output dtype, G groups per row, TILE rows
# per tile, BM batch rows per threadgroup, BN weight rows per threadgroup, TG
# threads per threadgroup, SAFE whether illegal bytes are clamped.
#
# One 128-weight group is one K step, so the group's scale is folded into the
# decoded weight and the K loop needs no per-group epilogue.
# The K loop and the write-back each have two forms, and which one pays is set
# by TM = BM / (SGM * 8): how many activation fragments a lane holds, and so how
# many times a decoded weight fragment is multiplied before it is thrown away.
#
# The staged form decodes a code byte once and spends all five of its trits into
# a threadgroup tile every simdgroup then reads back. The direct form builds both
# 8x8 fragments straight in registers, which costs one decode per trit -- five
# times the decode work, on three times the byte traffic -- but stages nothing,
# reads nothing back and executes no barrier. Measured on this machine: at TM = 1
# direct loses by up to 35 percent, at TM = 2 it is roughly even, and at TM = 4
# it wins by 10 to 100 percent and carries the kernel past mx.quantized_matmul.
# Neither dominates, so both are built and the dispatch picks per tiling.
#
# The TM = 1 figure holds only where it was taken, which is at row blocks of 128
# and 256 -- the A5 ladder has no narrow TM = 1 tiling to have measured. At row
# block 32 the ranking inverts and direct wins by 9 percent in a whole prefill
# pass; see GemmConfig.direct_fragments. Read TM as the axis worth checking
# first, not as the answer.
#
# The two forms are emitted as separate sources rather than as one source
# branching on a template constant. A barrier inside a branch is well defined
# only if the branch is uniform across the threadgroup; resolving it here makes
# that a property of the text instead of a bet on the Metal compiler folding a
# constant before it checks.
#
# TM = 4 with TN = 4 is not a tuning result, it is the corner of the feasible
# region, and both walls were measured. Counting the simdgroup-matrix registers
# a lane holds -- 2*TM*TN for the accumulators, 2*TM for the activation
# fragments, TN for the weight fragments -- puts every tiling swept on one line:
#
#     TM4/TN4   32 +  8 + 4 = 44   the incumbent
#     TM8/TN2   32 + 16 + 2 = 50   0.093x: spilled
#     TM8/TN4   64 + 16 + 4 = 84   0.012x: spilled harder
#     TM16/TN1  32 + 32 + 1 = 65   0.063x: spilled
#
# so the cliff sits between 44 and 50, and TM4/TN4 is already against it. The
# activation fragments do not have to be live together -- each is dead after its
# TN multiplies -- and building one at a time inside the multiply loop cuts that
# term from 2*TM to 2, which does rescue the spilled tilings exactly as the
# model predicts (TM8/TN2 went 0.093x to 0.683x, TM16/TN1 0.063x to 0.397x).
# They still lose, for the other reason: a lane issues 2/TN activation loads per
# multiply, so the fused arms land at 0.90x for TN=4, 0.68-0.73x for TN=2 and
# 0.40x for TN=1. Fusing costs 10 percent even at TM4/TN4, because TM loads
# issued together overlap their latency and one load issued at its point of use
# does not.
#
# Raising TM needs registers that are not there; lowering TN costs more than the
# decode it saves. Both walls are real, and the corner between them is here.
#
# With the tiling pinned, the last lever left was instruction count. A lane runs
# 24 loads against 16 multiplies per K step, and both of the pairs it loads are
# adjacent and provably aligned: two consecutive activations, and two consecutive
# code bytes. Loading each pair as a pair was measured, bit-identical output on
# every arm, and it does not pay. Hoisting the activation bounds test out of the
# two loads lands at 0.98-1.02x -- no sign, on any shape or batch -- and folding
# the two code bytes into one aligned ushort costs 4 to 9 percent. Both arms
# replace two predicated loads with a branch, and a branch inside a fully
# unrolled loop is worth more than the load it removes: the compiler already
# issues these as pairs when nothing stops it, and the branch stops it.
_GEMM_HEAD = """
    constexpr int TG = SGM * SGN * 32;
    constexpr int M_PER_SIMD = BM / SGM;
    constexpr int N_PER_SIMD = BN / SGN;
    constexpr int TM = M_PER_SIMD / 8;
    constexpr int TN = N_PER_SIMD / 8;
    constexpr int K_STEPS = BK / 8;
    constexpr int CHUNKS = TQ1_BLOCK / BK;
"""

# How the weight fill splits the threadgroup: ROW_LANES threads take consecutive
# rows of the tile, and the SLOT_LANES groups of those divide the slot range
# between them. Every (slot, row) pair is claimed by exactly one thread only if
# the two factor the thread count exactly. This is a constraint of the staged
# fill alone -- a form that builds fragments in registers maps lanes by the
# hardware's own fragment layout and is under no such restriction.
_GEMM_FILL_LANES = """    constexpr int ROW_LANES = TG < BN ? TG : BN;
    constexpr int SLOT_LANES = TG / ROW_LANES;
    static_assert(ROW_LANES * SLOT_LANES == TG,
                  "threads per group must divide into row lanes by slot lanes");
"""

# The staging tiles, and the allocation they share with the write-back strips.
# A form that builds its fragments in registers emits none of this, so a kernel
# that stages nothing declares nothing rather than reserving a tile no line of it
# reads -- the reservation alone would cost the occupancy the form exists to win.
_GEMM_STAGE_TILES = """    // The decoded weight tile is stored as half. Not an approximation: every
    // value in it is (t - 1) * scale, which is exactly -scale, 0 or +scale, and
    // scale is itself an fp16 value read straight out of the block. So the tile
    // is exactly representable in half for every tensor, and Metal accumulates a
    // float x half product into a float matrix at full float accuracy. Measured
    // against an all-float tile the results are bit-identical.
    constexpr int WTILE_FLOATS = (BK * BN) / 2;
    constexpr int STAGING = BM * BK + WTILE_FLOATS;
"""

# The epilogue writes back one 8-row strip per simdgroup row at a time, so it
# costs a fraction of the output tile rather than all of it. That is what lets BN
# grow, and BN is what divides the activation traffic.
_GEMM_STAGE_STRIPS = """    constexpr int EPILOGUE = SGM * 8 * BN;
"""

_GEMM_DIMS = """
    const uint columns = uint(G) * uint(TQ1_BLOCK);
    const uint rows = uint(codes_shape[0]) * uint(TILE);
    const uint batch = uint(x_shape[0]);
"""

# The staging tiles and the write-back strips never both need to exist: the K
# loop's last barrier separates them. So a form that emits both gives them one
# allocation sized by the larger, and a form that emits only one sizes it by that
# one alone. The size expression is resolved here rather than in MSL because a
# form that stages nothing never declares STAGING to compare against.
_GEMM_ALLOCATION = """
    threadgroup float stage[<<STAGE>>];
"""

_GEMM_STAGE_POINTERS = """    threadgroup float *xtile = stage;
    threadgroup half *wtile = (threadgroup half *)(stage + BM * BK);
"""

_GEMM_STRIP_POINTER = """    threadgroup float *strip = stage;
"""

_GEMM_NO_ALLOCATION = """
    // Nothing is staged and nothing is written back through threadgroup memory,
    // so this form allocates none of it and executes no barrier. Every weight,
    // activation and result lives in the registers of the lane that owns it, and
    // occupancy is set by register pressure alone.
"""

# Where a lane's two elements sit in an 8x8 fragment: it owns row `fm`, columns
# `fn` and `fn + 1`. Fixed by the hardware, and checked on this device by the
# kernel tests rather than taken on trust. For a weight fragment the rows are K
# and the columns are output rows; for an activation fragment the rows are batch
# and the columns are K. Emitted only by the forms that read registers directly.
_GEMM_LANE_MAP = """    const uint lane = thread_index_in_simdgroup;
    const uint quad = lane / 4u;
    const uint fm = (quad & 4u) + ((lane / 2u) % 4u);
    const uint fn = (quad & 2u) * 2u + (lane % 2u) * 2u;
"""

_GEMM_INDEXING = """    const uint tid = thread_position_in_threadgroup.x;
    const uint simd_id = simdgroup_index_in_threadgroup;
    const uint n0 = threadgroup_position_in_grid.x * uint(BN);
    const uint m0 = threadgroup_position_in_grid.y * uint(BM);
    // Simdgroups tile the output block in both axes: splitting only the batch
    // axis would force one simdgroup to hold BN/8 column fragments and reload
    // every one of them on each K step.
    const uint simd_m = (simd_id / uint(SGN)) * uint(M_PER_SIMD);
    const uint simd_n = (simd_id % uint(SGN)) * uint(N_PER_SIMD);

    // A BN block never straddles two row tiles: the host requires either that
    // BN divides TILE or that the tensor has a single tile.
    const uint tile_index = n0 / uint(TILE);
    const uint row_base = n0 - tile_index * uint(TILE);

    simdgroup_float8x8 acc[TM][TN];
    for (int i = 0; i < TM; ++i) {
        for (int j = 0; j < TN; ++j) { acc[i][j] = simdgroup_float8x8(0); }
    }
"""

# Copy the digit-major table into threadgroup memory before the group loop.
#
# The staged fill reads this decode once per code byte and spends all five
# digits, which is what `constant` memory is good at. A lane building fragments
# in registers inverts that: it gathers 2*TN entries per K step at indices that
# are whatever the weights happen to be, so a simdgroup's 32 lanes hit 32
# unrelated offsets and the constant path serialises them. Threadgroup memory is
# banked for exactly that.
#
# It only pays once the row bounds are gone, which is why this is a flag and not
# simply how the register form reads its table. Over the full (NFIT, MFIT, TGTAB)
# cube -- 45 paired rows, five tilings, three shapes, three batches -- the table
# on its own is 0.976 (39 of 45 rows below parity): with the bounds still there
# the gather hides behind the predicated loads and the fill's barrier is pure
# cost. Compile the row bound out and it inverts, to 1.011, and 1.026 with the
# batch bound gone too, because the gather is then exposed.
#
# What the cube did not produce is a rule. Per tiling the effect ran 0.999 to
# 1.025 with 11 of 45 individual rows still below parity, and the one clean story
# available -- that the 1024-thread tiling regressed because its fill barrier
# costs more -- did not survive: it read 0.983 over the first 17 rows and 0.999
# over all 45. So this is swept per shape and batch in the A5 benchmark rather
# than set here from a mechanism, and the dispatch reads the answer off that.
#
# The fill is placed before the accumulators are live and before the group loop,
# where every thread of the threadgroup reaches it unconditionally -- a barrier
# any thread can skip is undefined, and this kernel has no early return above it.
_GEMM_TABLE_FILL = f"""
    threadgroup half tgtrit[{TRITS_PER_FULL_BYTE}][{CODE_TABLE_ENTRIES}];
    for (uint idx = tid; idx < {TRITS_PER_FULL_BYTE * CODE_TABLE_ENTRIES}u; idx += uint(TG)) {{
        const uint d = idx / {CODE_TABLE_ENTRIES}u;
        tgtrit[d][idx - d * {CODE_TABLE_ENTRIES}u] =
            TQ1_DIRECT_TRIT[d][idx - d * {CODE_TABLE_ENTRIES}u];
    }}
    threadgroup_barrier(mem_flags::mem_threadgroup);
"""

_GEMM_GROUP_LOOP = """
    for (uint step_index = 0; step_index < uint(G) * uint(CHUNKS); ++step_index) {
        const uint g = step_index / uint(CHUNKS);
        const uint chunk = step_index - g * uint(CHUNKS);
        const uint k0 = chunk * uint(BK);

        // The group's code and scale bases. (t - 1) * scale is exactly
        // representable wherever scale is, so folding the scale into the weight
        // costs no accuracy and leaves a K step scale-free.
        const ulong group_base = (ulong(tile_index) * uint(G) + g);
        auto group_codes = codes + group_base * uint(TQ1_CODE_SLOTS) * uint(TILE);
        auto group_scales = scales + group_base * uint(TILE);
"""

# A row's scale does not change across the group, so a lane building fragments in
# registers reads it once per group rather than once per K step: two per lane,
# for the column pair it owns.
_GEMM_DIRECT_ROWSCALE = """
        half2 rowscale[TN];
        #pragma clang loop unroll(full)
        for (int j = 0; j < TN; ++j) {
            const uint n = simd_n + uint(j * 8) + fn;
            rowscale[j].x = NFIT || n0 + n < rows
                ? as_type<half>(group_scales[row_base + n]) : 0.0h;
            rowscale[j].y = NFIT || n0 + n + 1u < rows
                ? as_type<half>(group_scales[row_base + n + 1u]) : 0.0h;
        }
"""

_GEMM_STAGED_FILL = """
        // Guards the previous iteration's simdgroup_load against this fill.
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Activations, zero-filled past the end of a short final batch tile so
        // the matrix units never see uninitialised memory.
        for (uint idx = tid; idx < uint(BM) * uint(BK); idx += uint(TG)) {
            uint m = idx / uint(BK);
            uint k = idx - m * uint(BK);
            uint sample = m0 + m;
            xtile[idx] = MFIT || sample < batch
                ? float(x[ulong(sample) * ulong(columns) + ulong(g) * TQ1_BLOCK + k0 + k])
                : 0.0f;
        }

        // Weights, decoded into K-major order with the group scale folded in.
        //
        // One thread per (code byte, row), not per (column, row). A byte
        // carries five columns, so reading it once and decoding all five costs
        // one load where walking the columns cost five.
        //
        // A K chunk is not slot aligned -- 128 columns are 25 slots of five
        // trits plus a tail slot of three -- so the window's first and last
        // slot contribute only part of their trits and the range test drops the
        // rest. Every column in the window still belongs to exactly one slot in
        // [first_slot, last_slot], so the tile is written whole.
        const uint first_slot = k0 >= 125u ? uint(TQ1_FULL_SLOTS) : k0 / 5u;
        const uint last_column = k0 + uint(BK) - 1u;
        const uint last_slot = last_column >= 125u
            ? uint(TQ1_FULL_SLOTS) : last_column / 5u;

        // A thread walks the slots of one row, so the row's scale is read once
        // instead of once per slot. The scale is a per-row value and the slot
        // is a per-column one, so pairing them was loading the same two bytes
        // eight times over at a 32-column chunk -- and those two bytes cost
        // twice what the code byte beside them did.
        //
        // Rows still come first in the thread index, so consecutive threads
        // take consecutive n and the code read stays one coalesced run across
        // the tile. Where the threadgroup is wider than the tile the surplus
        // threads divide the slot range rather than idle, which is why the
        // thread count has to divide the tile width -- ROW_LANES enforces it.
        const uint slots = last_slot - first_slot + 1u;
        const uint slot_lane = tid / uint(ROW_LANES);

        for (uint n = tid - slot_lane * uint(ROW_LANES); n < uint(BN); n += uint(ROW_LANES)) {
            // A row past the end of the tensor is zeroed through its scale
            // rather than by skipping the write: (t - 1) * 0 is 0 for every t,
            // and skipping would leave the previous chunk's weights in the tile.
            const bool live = NFIT || n0 + n < rows;
            const half scale = live ? as_type<half>(group_scales[row_base + n]) : 0.0h;

            for (uint s = slot_lane; s < slots; s += uint(SLOT_LANES)) {
                uint slot = first_slot + s;
                uint code = 0u;
                if (live) {
                    code = uint(group_codes[slot * uint(TILE) + row_base + n]);
                    if (SAFE) {
                        code = min(code, slot == uint(TQ1_FULL_SLOTS)
                            ? TQ1_MAX_TAIL_BYTE : TQ1_MAX_FULL_BYTE);
                    }
                }

                // Two table loads instead of five divisions. A row past the end
                // of the tensor still decodes, to zero, because its scale is zero.
                half4 trit_lo = TQ1_TRIT_LO[code];
                half trit_hi = TQ1_TRIT_HI[code];
                half decoded[5] = {
                    trit_lo.x * scale, trit_lo.y * scale, trit_lo.z * scale,
                    trit_lo.w * scale, trit_hi * scale,
                };

                bool tail = slot == uint(TQ1_FULL_SLOTS);
                int base = (tail ? 125 : int(slot) * 5) - int(k0);
                int digits = tail ? 3 : 5;

                // Only the window's first and last slot can be clipped, so the
                // common case is a whole slot landing inside the tile. Testing
                // that once and then storing unconditionally is worth more than
                // the decode it sits next to: the per-digit test was costing
                // more than the entire base-3 unpack it guarded.
                if (!tail && base >= 0 && base + 4 < int(BK)) {
                    #pragma clang loop unroll(full)
                    for (int d = 0; d < 5; ++d) {
                        wtile[uint(base + d) * uint(BN) + n] = decoded[d];
                    }
                } else {
                    #pragma clang loop unroll(full)
                    for (int d = 0; d < 5; ++d) {
                        int column = base + d;
                        if (d < digits && column >= 0 && column < int(BK)) {
                            wtile[uint(column) * uint(BN) + n] = decoded[d];
                        }
                    }
                }
            }
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);
"""

_GEMM_STEP_OPEN = """
        #pragma clang loop unroll(full)
        for (int step = 0; step < K_STEPS; ++step) {
            simdgroup_float8x8 a[TM];
            simdgroup_half8x8 b[TN];
"""

_GEMM_STAGED_LOADS = """            for (int i = 0; i < TM; ++i) {
                simdgroup_load(a[i], xtile + (simd_m + uint(i * 8)) * BK + step * 8, BK);
            }
            for (int j = 0; j < TN; ++j) {
                simdgroup_load(b[j], wtile + uint(step * 8) * uint(BN) + simd_n + uint(j * 8), BN);
            }
"""

_GEMM_DIRECT_FRAGMENTS = """            #pragma clang loop unroll(full)
            for (int i = 0; i < TM; ++i) {
                // The lane's two elements are adjacent in K, so this is one pair
                // of consecutive floats out of the activation row.
                const uint sample = m0 + simd_m + uint(i * 8) + fm;
                const ulong at = ulong(sample) * ulong(columns)
                    + ulong(g) * TQ1_BLOCK + k0 + uint(step * 8) + fn;
                thread float2 &owned = reinterpret_cast<thread float2 &>(a[i].thread_elements());
                owned.x = MFIT || sample < batch ? float(x[at]) : 0.0f;
                owned.y = MFIT || sample < batch ? float(x[at + 1u]) : 0.0f;
            }

            // The lane's K position fixes its slot and digit for every row
            // fragment of this step, so the base-3 position is resolved once and
            // only the code byte changes across j.
            const uint q = k0 + uint(step * 8) + fm;
            const uint slot = q < 125u ? q / 5u : uint(TQ1_FULL_SLOTS);
            const uint digit = q < 125u ? q - 5u * slot : q - 125u;
            auto slot_codes = group_codes + slot * uint(TILE) + row_base;
            <<TRIT>>

            #pragma clang loop unroll(full)
            for (int j = 0; j < TN; ++j) {
                const uint n = simd_n + uint(j * 8) + fn;
                uint lo = NFIT || n0 + n < rows ? uint(slot_codes[n]) : 0u;
                uint hi = NFIT || n0 + n + 1u < rows ? uint(slot_codes[n + 1u]) : 0u;
                if (SAFE) {
                    const uint limit = slot == uint(TQ1_FULL_SLOTS)
                        ? TQ1_MAX_TAIL_BYTE : TQ1_MAX_FULL_BYTE;
                    lo = min(lo, limit);
                    hi = min(hi, limit);
                }
                // A row past the end of the tensor decodes to zero through its
                // scale, exactly as the staged fill zeroed it.
                thread half2 &owned = reinterpret_cast<thread half2 &>(b[j].thread_elements());
                owned.x = trit[lo] * rowscale[j].x;
                owned.y = trit[hi] * rowscale[j].y;
            }
"""

_GEMM_STEP_CLOSE = """            for (int i = 0; i < TM; ++i) {
                for (int j = 0; j < TN; ++j) {
                    simdgroup_multiply_accumulate(acc[i][j], a[i], b[j], acc[i][j]);
                }
            }
        }
    }
"""

# Each lane already holds two of the eight columns of one row of each fragment,
# so this writes them where they belong and the round trip disappears with the
# barriers that guarded it.
_GEMM_DIRECT_EPILOGUE = """
    #pragma clang loop unroll(full)
    for (int i = 0; i < TM; ++i) {
        #pragma clang loop unroll(full)
        for (int j = 0; j < TN; ++j) {
            thread float2 &owned = reinterpret_cast<thread float2 &>(acc[i][j].thread_elements());
            uint m = m0 + simd_m + uint(i * 8) + fm;
            uint n = n0 + simd_n + uint(j * 8) + fn;
            if (MFIT || m < batch) {
                if (NFIT || n < rows) {
                    out[ulong(m) * ulong(rows) + ulong(n)] = static_cast<T>(owned.x);
                }
                if (NFIT || n + 1u < rows) {
                    out[ulong(m) * ulong(rows) + ulong(n) + 1ul] = static_cast<T>(owned.y);
                }
            }
        }
    }
"""

_GEMM_STAGED_EPILOGUE = """
    // Write back one 8-row strip per simdgroup row at a time. The strip index
    // has to stay a compile-time constant: indexing the accumulator array with
    // a runtime value would spill it out of registers.
    threadgroup float *my_strip = strip + (simd_id / uint(SGN)) * 8u * uint(BN);
    #pragma clang loop unroll(full)
    for (int i = 0; i < TM; ++i) {
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (int j = 0; j < TN; ++j) {
            simdgroup_store(acc[i][j], my_strip + simd_n + uint(j * 8), BN);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint idx = tid; idx < uint(SGM) * 8u * uint(BN); idx += uint(TG)) {
            uint flat = idx / uint(BN);
            uint n = idx - flat * uint(BN);
            uint sgm = flat / 8u;
            uint m = m0 + sgm * uint(M_PER_SIMD) + uint(i * 8) + (flat - sgm * 8u);
            if ((MFIT || m < batch) && (NFIT || n0 + n < rows)) {
                out[ulong(m) * ulong(rows) + ulong(n0 + n)] = static_cast<T>(strip[idx]);
            }
        }
    }
"""


def _gemm_source(
    *, direct_fragments: bool, direct_epilogue: bool, threadgroup_table: bool
) -> str:
    """Assemble one of the six GEMM forms.

    ``direct_fragments`` builds both operand fragments in registers instead of
    staging them; ``direct_epilogue`` writes the accumulators straight to device
    instead of through threadgroup memory. They are independent: each removes a
    different claim on threadgroup memory, and with both removed and no
    threadgroup table the kernel allocates none of it and executes no barrier.

    ``threadgroup_table`` moves the digit-major trit table out of ``constant``
    memory, which only the register form reads and so only it can ask for. It is
    a form rather than a template constant because the two differ by the address
    space of a pointer, and an address space is part of a type in Metal: a
    runtime branch would need the whole fragment loop written twice.
    """
    if threadgroup_table and not direct_fragments:
        raise FormatError(
            "the threadgroup trit table is only read by the register form, and the staged "
            "fill decodes through the byte-major table instead"
        )
    pieces = [_GEMM_HEAD]
    if not direct_fragments:
        pieces.append(_GEMM_FILL_LANES)
        pieces.append(_GEMM_STAGE_TILES)
    if not direct_epilogue:
        pieces.append(_GEMM_STAGE_STRIPS)
    pieces.append(_GEMM_DIMS)

    if direct_fragments and direct_epilogue:
        pieces.append(_GEMM_NO_ALLOCATION)
    else:
        if direct_fragments:
            size = "EPILOGUE"
        elif direct_epilogue:
            size = "STAGING"
        else:
            size = "STAGING > EPILOGUE ? STAGING : EPILOGUE"
        pieces.append(_GEMM_ALLOCATION.replace("<<STAGE>>", size))
        if not direct_fragments:
            pieces.append(_GEMM_STAGE_POINTERS)
        if not direct_epilogue:
            pieces.append(_GEMM_STRIP_POINTER)

    if direct_fragments or direct_epilogue:
        pieces.append(_GEMM_LANE_MAP)
    pieces.append(_GEMM_INDEXING)
    if threadgroup_table:
        pieces.append(_GEMM_TABLE_FILL)
    pieces.append(_GEMM_GROUP_LOOP)
    pieces.append(_GEMM_DIRECT_ROWSCALE if direct_fragments else _GEMM_STAGED_FILL)
    pieces.append(_GEMM_STEP_OPEN)
    if direct_fragments:
        pieces.append(
            _GEMM_DIRECT_FRAGMENTS.replace(
                "<<TRIT>>",
                "threadgroup half *trit = tgtrit[digit];"
                if threadgroup_table
                else "constant half *trit = TQ1_DIRECT_TRIT[digit];",
            )
        )
    else:
        pieces.append(_GEMM_STAGED_LOADS)
    pieces.append(_GEMM_STEP_CLOSE)
    pieces.append(_GEMM_DIRECT_EPILOGUE if direct_epilogue else _GEMM_STAGED_EPILOGUE)
    return "".join(pieces)


# Embedding lookup: decode exactly the requested rows, one trit per output
# element, with no table and no intermediate row buffer.
_GET_ROWS_SOURCE = """
    const uint columns = uint(G) * uint(TQ1_BLOCK);
    const uint column = thread_position_in_grid.x;
    const uint token = thread_position_in_grid.y;
    if (column >= columns || token >= uint(indices_shape[0])) { return; }

    // A token id past the end of the table would index another allocation
    // entirely. It cannot happen in the shipped path -- prompt ids come from the
    // tokenizer and sampled ids from an argmax over exactly `rows` logits -- so
    // this is memory safety against a caller bug, not a supported input. Such a
    // row reads as all-zero, which is a defined value that no real embedding row
    // has and which cannot poison the rest of the graph the way a NaN would.
    const uint rows = uint(codes_shape[0]) * uint(TILE);
    const uint row = uint(indices[token]);
    if (row >= rows) {
        out[ulong(token) * ulong(columns) + column] = static_cast<T>(0);
        return;
    }
    const uint tile_index = row / uint(TILE);
    const uint row_in_tile = row - tile_index * uint(TILE);
    const uint group = column / uint(TQ1_BLOCK);
    const uint offset = column - group * uint(TQ1_BLOCK);

    // Slots 0..24 hold five trits each and cover columns 0..124; the tail slot
    // holds the remaining three.
    uint slot = offset / 5u;
    uint digit = offset - 5u * slot;
    if (offset >= 125u) { slot = uint(TQ1_FULL_SLOTS); digit = offset - 125u; }

    const ulong group_base = ulong(tile_index) * uint(G) + group;
    uint code = uint(codes[group_base * uint(TQ1_CODE_SLOTS) * uint(TILE)
        + slot * uint(TILE) + row_in_tile]);
    if (SAFE) {
        code = min(code, slot == uint(TQ1_FULL_SLOTS) ? TQ1_MAX_TAIL_BYTE : TQ1_MAX_FULL_BYTE);
    }
    const uint trit = (code / TQ1_POW3[digit]) % 3u;
    const float scale = float(as_type<half>(scales[group_base * uint(TILE) + row_in_tile]));
    out[ulong(token) * ulong(columns) + column] = static_cast<T>((float(trit) - 1.0f) * scale);
"""

_MATMUL_KERNEL = mx.fast.metal_kernel(
    name="tq1_matmul",
    input_names=["codes", "scales", "x"],
    output_names=["out"],
    source=_MATMUL_SOURCE,
    header=_HEADER,
)

# All six forms are built at import, but building one costs only string work:
# mx.fast.metal_kernel compiles nothing until a template instantiation is called,
# and a form the dispatch never selects is never compiled. The threadgroup table
# is only a form of the register kernel, so the staged pair has no third axis.
_GEMM_FORMS = tuple(
    (direct_fragments, direct_epilogue, threadgroup_table)
    for direct_fragments in (False, True)
    for direct_epilogue in (False, True)
    for threadgroup_table in ((False, True) if direct_fragments else (False,))
)

_GEMM_KERNELS = {
    form: mx.fast.metal_kernel(
        name=(
            "tq1_gemm"
            + ("_dfrag" if form[0] else "")
            + ("_depi" if form[1] else "")
            + ("_tgtab" if form[2] else "")
        ),
        input_names=["codes", "scales", "x"],
        output_names=["out"],
        source=_gemm_source(
            direct_fragments=form[0], direct_epilogue=form[1], threadgroup_table=form[2]
        ),
        # The digit-major table is what the register form indexes; the staged
        # form never names it, so it is not put in front of the compiler.
        header=_HEADER + (_direct_trit_table_source() if form[0] else ""),
    )
    for form in _GEMM_FORMS
}

_GET_ROWS_KERNEL = mx.fast.metal_kernel(
    name="tq1_get_rows",
    input_names=["codes", "scales", "indices"],
    output_names=["out"],
    source=_GET_ROWS_SOURCE,
    header=_HEADER,
)


def lut_floats(*, padded: bool) -> int:
    """Floats of threadgroup memory one group's LUT23 tables occupy."""
    l2_stride = PADDED_STRIDE if padded else L2_ENTRIES
    l3_stride = PADDED_STRIDE if padded else L3_ENTRIES
    return FULL_CODE_SLOTS * l2_stride + FULL_CODE_SLOTS * l3_stride + TAIL_ENTRIES


def threadgroup_bytes(*, batch_tile: int, padded: bool) -> int:
    return batch_tile * lut_floats(padded=padded) * 4


@dataclass(frozen=True)
class MatmulConfig:
    """The tuning knobs swept at A5; every one is a Metal template parameter."""

    threads: int
    batch_tile: int
    padded_lut: bool
    safe_clamp: bool
    k_split: int

    def __post_init__(self) -> None:
        if self.threads <= 0 or self.threads % 32:
            raise FormatError(
                f"threads must be a positive multiple of the 32-wide simdgroup, got {self.threads}"
            )
        if self.batch_tile <= 0:
            raise FormatError(f"batch tile must be positive, got {self.batch_tile}")
        if self.k_split <= 0:
            raise FormatError(f"k split must be positive, got {self.k_split}")
        used = threadgroup_bytes(batch_tile=self.batch_tile, padded=self.padded_lut)
        if used > THREADGROUP_MEMORY_BYTES:
            raise FormatError(
                f"batch tile {self.batch_tile} with "
                f"{'padded' if self.padded_lut else 'unpadded'} tables needs {used} bytes of "
                f"threadgroup memory, over the {THREADGROUP_MEMORY_BYTES} byte limit"
            )

    @property
    def threadgroup_bytes(self) -> int:
        return threadgroup_bytes(batch_tile=self.batch_tile, padded=self.padded_lut)

    @property
    def benchmark_name(self) -> str:
        """The label this configuration is recorded under in the A5 sweep.

        Defined on the configuration rather than in the benchmark so that a rule
        read off ``results/mlx/a5_op_benchmarks.json`` can be replayed against
        it, rather than a test re-deriving a naming scheme and agreeing with
        itself.

        The unsplit label is spelled exactly as the committed sweep spells it,
        so adding split-K did not rename anything already measured.
        """
        name = f"lut23_bt{self.batch_tile}_{'padded' if self.padded_lut else 'packed'}"
        return name if self.k_split == 1 else f"{name}_k{self.k_split}"


def gemm_threadgroup_bytes(
    *,
    batch_block: int,
    row_block: int,
    k_block: int,
    simd_rows: int,
    direct_fragments: bool,
    direct_epilogue: bool,
    threadgroup_table: bool,
) -> int:
    """One allocation, reused by the staging tiles and then by the epilogue.

    The K loop needs ``batch_block x k_block`` activations at four bytes beside
    ``k_block x row_block`` weights at two, the weights being exactly
    representable in half. The epilogue, which runs after the last barrier,
    writes back one 8-row strip per simdgroup row at a time and so needs only
    ``simd_rows x 8 x row_block`` floats. Whichever is larger is what the
    threadgroup actually costs -- and a form that builds its fragments in
    registers or writes its result straight to device drops that claim to zero,
    which is the point of it: the tile it would otherwise reserve costs
    occupancy whether or not a line of the kernel reads it.

    The trit table, when it is staged, is not part of that reuse: it is written
    once before the group loop and read until the kernel ends, so it overlaps
    neither the staging tiles nor the epilogue strip and its bytes add.
    """
    staging = 0 if direct_fragments else k_block * batch_block * 4 + k_block * row_block * 2
    epilogue = 0 if direct_epilogue else simd_rows * MATRIX_TILE * row_block * 4
    table = TRITS_PER_FULL_BYTE * CODE_TABLE_ENTRIES * 2 if threadgroup_table else 0
    return max(staging, epilogue) + table


@dataclass(frozen=True)
class GemmConfig:
    """Tiling for the prefill path.

    ``batch_block`` by ``row_block`` is the output block a threadgroup owns, and
    ``simd_rows`` by ``simd_columns`` is how its simdgroups tile that block --
    each one keeping ``(batch_block / simd_rows / 8) * (row_block / simd_columns
    / 8)`` 8x8 accumulators in registers.

    ``k_block`` is how much of a 128-weight group is staged at a time. It does
    not change the arithmetic, only the threadgroup footprint, and through it how
    many threadgroups a GPU core can keep resident to hide memory latency. That
    was measured to dominate: staging a whole group cost the entire 32 KiB budget
    and ran the kernel three times slower than staging a quarter of one.

    ``direct_fragments``, ``direct_epilogue`` and ``threadgroup_table`` select
    which of the six kernel forms this tiling runs, and none dominates:

    - ``direct_fragments`` is decided by ``TM = batch_block / (simd_rows * 8)``,
      how many activation fragments a lane holds and so how many times a decoded
      weight fragment is multiplied before it is discarded. Building fragments in
      registers costs one decode per trit where the staged fill costs one per
      byte, so at ``TM = 1`` it loses by up to 35 percent and at ``TM = 4`` it
      wins by 10 to 100 and carries the kernel past ``mx.quantized_matmul``.
      TM does not settle it alone, and the exception is one this file ships: the
      35 percent is a *wide*-block figure, since every ``TM = 1`` tiling the A5
      ladder swept has a row block of 128 or 256. At row block 32 the ranking
      inverts. Timed in a whole prefill pass with three shapes -- out_proj,
      down_proj, in_proj_z, 176 of the 497 dispatches a token costs -- routed
      into ``batch_block`` 16 by row block 32 and nothing else varied, the
      register form ran 256.7 ms against the staged twin's 280.5 ms, ten rounds
      in both orders with the two ranges not touching. So ``TM = 1`` is a reason
      to check the form rather than a rule for choosing it, and
      ``GEMM_SMALL_BATCH_NARROW_BLOCK`` builds its fragments in registers on
      that measurement rather than in spite of it.
    - ``direct_epilogue`` is worth about four percent either way at ``TM = 4``,
      but it is what frees the last claim on threadgroup memory, so it decides
      the largest tilings: at 128x256 the staged epilogue needs the whole 32 KiB
      budget and the direct form measured 1.15 to 1.26 times it.
    - ``threadgroup_table`` moves the digit-major trit table the register form
      gathers from into threadgroup memory, and is only legal with
      ``direct_fragments`` because nothing else reads that table. It is the one
      axis a mechanism did not settle: over the bounds cube it was 0.976 with the
      row bounds present and 1.011 to 1.026 with them gone, but per tiling it ran
      0.999 to 1.025 with a quarter of individual rows still below parity. See
      the comment above ``_GEMM_TABLE_FILL``.

    All three are part of the tiling rather than a global switch because the
    dispatch picks them per shape and batch off the measured ladder.
    """

    batch_block: int
    row_block: int
    k_block: int
    simd_rows: int
    simd_columns: int
    direct_fragments: bool
    direct_epilogue: bool
    threadgroup_table: bool
    safe_clamp: bool

    def __post_init__(self) -> None:
        if self.threadgroup_table and not self.direct_fragments:
            raise FormatError(
                "the threadgroup trit table is only read by the register form, so a staged "
                "tiling cannot ask for it"
            )
        for name, value in (
            ("simd rows", self.simd_rows),
            ("simd columns", self.simd_columns),
        ):
            if value <= 0:
                raise FormatError(f"{name} must be positive, got {value}")
        for name, block, groups in (
            ("batch block", self.batch_block, self.simd_rows),
            ("row block", self.row_block, self.simd_columns),
        ):
            span = groups * MATRIX_TILE
            if block <= 0 or block % span:
                raise FormatError(
                    f"{name} {block} is not a multiple of {span}, the {groups} simdgroups "
                    f"across it times the {MATRIX_TILE}x{MATRIX_TILE} matrix tile"
                )
        if self.k_block <= 0 or self.k_block % MATRIX_TILE:
            raise FormatError(
                f"k block {self.k_block} is not a multiple of the {MATRIX_TILE}x{MATRIX_TILE} "
                "matrix tile"
            )
        if BLOCK_SIZE % self.k_block:
            raise FormatError(
                f"k block {self.k_block} does not divide the {BLOCK_SIZE}-weight group, so a "
                "chunk would straddle two groups and two scales"
            )
        # The weight fill gives each thread one row of the tile and a stride
        # through the slots, so that the row's scale is read once rather than
        # once per slot. That mapping covers every (slot, row) pair exactly once
        # only when the row lanes divide the threadgroup evenly. The register
        # form has no fill and no such mapping, so the constraint goes with it.
        row_lanes = min(self.threads, self.row_block)
        if not self.direct_fragments and self.threads % row_lanes:
            raise FormatError(
                f"{self.threads} threads do not divide into row lanes of {row_lanes}, so the "
                f"weight fill for row block {self.row_block} would leave some rows unwritten"
            )
        used = self.threadgroup_bytes
        if used > THREADGROUP_MEMORY_BYTES:
            raise FormatError(
                f"batch block {self.batch_block} by row block {self.row_block} at k block "
                f"{self.k_block} needs {used} bytes of threadgroup memory, over the "
                f"{THREADGROUP_MEMORY_BYTES} byte limit"
            )

    @property
    def threads(self) -> int:
        return self.simd_rows * self.simd_columns * SIMD_WIDTH

    @property
    def accumulators_per_thread(self) -> int:
        """Floats each thread holds in registers for the output block."""
        return self.batch_block * self.row_block // self.threads

    @property
    def threadgroup_bytes(self) -> int:
        return gemm_threadgroup_bytes(
            batch_block=self.batch_block,
            row_block=self.row_block,
            k_block=self.k_block,
            simd_rows=self.simd_rows,
            direct_fragments=self.direct_fragments,
            direct_epilogue=self.direct_epilogue,
            threadgroup_table=self.threadgroup_table,
        )

    @property
    def benchmark_name(self) -> str:
        """The label this tiling is recorded under in the A5 sweep.

        Only the forms that are on are named, so an unsuffixed name means the
        fully staged kernel -- which is what the name meant before the register
        forms existed, and keeps older recorded ladders comparable.
        """
        return (
            f"gemm_{self.batch_block}x{self.row_block}k{self.k_block}"
            f"s{self.simd_rows}x{self.simd_columns}"
            + ("_frag" if self.direct_fragments else "")
            + ("_epi" if self.direct_epilogue else "")
            + ("_tgt" if self.threadgroup_table else "")
        )


def _check_packed(codes: mx.array, scales: mx.array, *, groups_per_row: int, tile: int) -> int:
    if codes.dtype != mx.uint8:
        raise FormatError(f"codes must be uint8, got {codes.dtype}")
    if scales.dtype != mx.uint16:
        raise FormatError(f"scales must be uint16 of raw FP16 bits, got {scales.dtype}")
    if groups_per_row <= 0:
        raise FormatError(f"groups per row must be positive, got {groups_per_row}")
    if tile <= 0:
        raise FormatError(f"tile must be positive, got {tile}")
    if codes.ndim != 4 or codes.shape[1:] != (groups_per_row, CODE_SLOTS, tile):
        raise FormatError(
            f"codes shape {tuple(codes.shape)} does not match "
            f"(tiles, {groups_per_row}, {CODE_SLOTS}, {tile})"
        )
    tiles = codes.shape[0]
    if tuple(scales.shape) != (tiles, groups_per_row, tile):
        raise FormatError(
            f"scales shape {tuple(scales.shape)} does not match ({tiles}, {groups_per_row}, {tile})"
        )
    if tiles <= 0:
        raise FormatError("packed tensor has no tiles")
    return tiles


def tq1_matmul(
    codes: mx.array,
    scales: mx.array,
    x: mx.array,
    *,
    groups_per_row: int,
    tile: int,
    config: MatmulConfig,
) -> mx.array:
    """``x @ W.T`` for a TQ1_G128 packed ``W`` of shape ``(tiles * tile, groups_per_row * 128)``.

    ``x`` is 2-D ``(batch, columns)``; the caller flattens any leading
    dimensions. The returned array is ``(batch, rows)`` in ``x``'s dtype.

    A ``k_split`` above one splits the K axis across that many threadgroups per
    row tile and sums the partials in a second pass. It exists for the narrow
    tensors, where the row count alone cannot fill the machine and a chained
    dispatch pays the whole per-group latency serially. The partials are fp32
    and the reduction is an ordinary ``mx.sum``: a float atomic would be one
    dispatch cheaper but its addition order varies between runs, which would
    cost the bit-identical greedy continuations the artifact gate relies on.
    """
    tiles = _check_packed(codes, scales, groups_per_row=groups_per_row, tile=tile)
    batch = _check_activations(x, groups_per_row=groups_per_row)
    split = config.k_split
    if groups_per_row % split:
        # Ragged chunks would leave one threadgroup holding a full chunk while
        # another holds nothing, and the slowest chunk is what the caller waits
        # for, so an uneven split buys threadgroups that do not shorten anything.
        raise FormatError(
            f"k split {split} does not divide the tensor's {groups_per_row} groups per row"
        )

    rows = tiles * tile
    batch_tiles = math.ceil(batch / config.batch_tile)
    partial_dtype = x.dtype if split == 1 else mx.float32
    outputs = _MATMUL_KERNEL(
        inputs=[codes, scales, x],
        template=[
            ("T", x.dtype),
            ("OT", partial_dtype),
            ("G", groups_per_row),
            ("TILE", tile),
            ("TG", config.threads),
            ("BT", config.batch_tile),
            ("PAD", config.padded_lut),
            ("SAFE", config.safe_clamp),
            ("KS", split),
        ],
        grid=(tiles * config.threads, batch_tiles, split),
        threadgroup=(config.threads, 1, 1),
        output_shapes=[(batch, rows) if split == 1 else (split, batch, rows)],
        output_dtypes=[partial_dtype],
    )
    if split == 1:
        return outputs[0]
    return mx.sum(outputs[0], axis=0).astype(x.dtype)


def _check_activations(x: mx.array, *, groups_per_row: int) -> int:
    if x.dtype not in ACTIVATION_DTYPES:
        raise FormatError(f"activations must be one of {ACTIVATION_DTYPES}, got {x.dtype}")
    if x.ndim != 2:
        raise FormatError(f"activations must be 2-D (batch, columns), got shape {tuple(x.shape)}")
    columns = groups_per_row * BLOCK_SIZE
    if x.shape[1] != columns:
        raise FormatError(f"activations have {x.shape[1]} columns, expected {columns}")
    if x.shape[0] <= 0:
        raise FormatError("activations have an empty batch dimension")
    return x.shape[0]


def tq1_gemm(
    codes: mx.array,
    scales: mx.array,
    x: mx.array,
    *,
    groups_per_row: int,
    tile: int,
    config: GemmConfig,
) -> mx.array:
    """``x @ W.T`` for a packed ``W``, multiplied on the GPU's matrix units.

    The prefill counterpart of :func:`tq1_matmul`. LUT23 wins when each weight is
    touched once, because it replaces 128 decodes with 51 table reads; at prompt
    batch sizes each weight is touched once per sample instead, and decoding a
    weight once and reusing it across the whole batch block through
    ``simdgroup_multiply_accumulate`` is far cheaper. Nothing larger than one
    block's worth of weights is ever unpacked -- at most 32 KiB of threadgroup
    memory in the staged form and nothing at all in the register form, never a
    model-sized array, and never anything that outlives the K chunk that made it.

    The two paths therefore sum in different orders and do not agree bit for bit:
    ``tq1_matmul`` sums trit contributions and applies the scale once per group,
    while this folds the scale into each weight and sums through the matrix
    units. Both are honest fp32 evaluations of the same product.

    ``NFIT`` and ``MFIT`` say that the row block divides the row count and that
    the batch block divides the batch, which is decided here and not in the
    kernel: both are known before dispatch, and when they hold, no thread can
    address a row or a sample past the end. Every bounds test in the kernel is
    then a compile-time true and folds away -- sixteen of the twenty-four loads
    in a K step carry one, plus two per group in the row scales and one or three
    in the epilogue. This costs instantiations: ``NFIT`` is fixed per tensor but
    ``MFIT`` follows the batch, so a shape run at both a fitting and a
    non-fitting prompt length compiles the kernel twice. The A5 sweep reports
    that cold-start cost rather than hiding it.
    """
    tiles = _check_packed(codes, scales, groups_per_row=groups_per_row, tile=tile)
    batch = _check_activations(x, groups_per_row=groups_per_row)

    rows = tiles * tile
    if tiles > 1 and tile % config.row_block:
        raise FormatError(
            f"row block {config.row_block} does not divide the {tile}-row tile, so a block "
            f"would straddle two tiles of this {rows}-row tensor"
        )
    row_blocks = math.ceil(rows / config.row_block)
    batch_blocks = math.ceil(batch / config.batch_block)
    kernel = _GEMM_KERNELS[
        (config.direct_fragments, config.direct_epilogue, config.threadgroup_table)
    ]
    outputs = kernel(
        inputs=[codes, scales, x],
        template=[
            ("T", x.dtype),
            ("G", groups_per_row),
            ("TILE", tile),
            ("BM", config.batch_block),
            ("BN", config.row_block),
            ("BK", config.k_block),
            ("SGM", config.simd_rows),
            ("SGN", config.simd_columns),
            ("SAFE", config.safe_clamp),
            ("NFIT", rows % config.row_block == 0),
            ("MFIT", batch % config.batch_block == 0),
        ],
        grid=(row_blocks * config.threads, batch_blocks, 1),
        threadgroup=(config.threads, 1, 1),
        output_shapes=[(batch, rows)],
        output_dtypes=[x.dtype],
    )
    return outputs[0]


def tq1_get_rows(
    codes: mx.array,
    scales: mx.array,
    indices: mx.array,
    *,
    groups_per_row: int,
    tile: int,
    dtype: mx.Dtype,
    threads: int,
    safe_clamp: bool,
) -> mx.array:
    """Decode the rows named by ``indices`` out of a packed embedding table."""
    _check_packed(codes, scales, groups_per_row=groups_per_row, tile=tile)
    if dtype not in ACTIVATION_DTYPES:
        raise FormatError(f"output dtype must be one of {ACTIVATION_DTYPES}, got {dtype}")
    if indices.dtype != INDEX_DTYPE:
        raise FormatError(f"indices must be {INDEX_DTYPE}, got {indices.dtype}")
    if indices.ndim != 1:
        raise FormatError(f"indices must be 1-D, got shape {tuple(indices.shape)}")
    if indices.size <= 0:
        raise FormatError("indices are empty")
    if threads <= 0 or threads % 32:
        raise FormatError(f"threads must be a positive multiple of 32, got {threads}")

    columns = groups_per_row * BLOCK_SIZE
    outputs = _GET_ROWS_KERNEL(
        inputs=[codes, scales, indices],
        template=[
            ("T", dtype),
            ("G", groups_per_row),
            ("TILE", tile),
            ("SAFE", safe_clamp),
        ],
        grid=(columns, indices.size, 1),
        threadgroup=(min(threads, columns), 1, 1),
        output_shapes=[(indices.size, columns)],
        output_dtypes=[dtype],
    )
    return outputs[0]
