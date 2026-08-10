#include "tq1_lut23_streamk.cuh"

#include <cuda_fp16.h>

#include <cstddef>
#include <stdexcept>

namespace bonsai {
namespace {

constexpr int FULL_SLOTS = 25;

struct alignas(128) LutBuffer {
    alignas(128) float l2[FULL_SLOTS][32];
    alignas(128) float l3[FULL_SLOTS][32];
    alignas(128) float tail[32];
};

static_assert(sizeof(LutBuffer) % 128 == 0);
static_assert(offsetof(LutBuffer, l2) % 128 == 0);
static_assert(offsetof(LutBuffer, l3) % 128 == 0);
static_assert(offsetof(LutBuffer, tail) % 128 == 0);

__device__ __forceinline__ float half_bits_to_float(const std::uint16_t bits) {
    const __half_raw raw{bits};
    return __half2float(raw);
}

__device__ __forceinline__ float load_q8_value(
        const block_q8_1_compat * __restrict__ activation,
        const int logical) {
    const block_q8_1_compat & block = activation[logical >> 5];
    return half_bits_to_float(block.d) * static_cast<float>(block.qs[logical & 31]);
}

__device__ __forceinline__ void compute_lut23(
        const block_q8_1_compat * __restrict__ activation,
        const int group,
        const float trit0,
        const float trit1,
        const float trit2,
        LutBuffer & lut) {
    const int tid = threadIdx.x;
    const int warp = tid >> 5;
    const int lane = tid & 31;

    // Give every warp a contiguous activation interval.  A lane loads one
    // Q8_1 value and warp broadcasts replace the old shared activation stage.
    // This preserves the padded per-slot LUT layout while removing one CTA
    // barrier from every g128 group.
    const int first_slot = warp == 0 ? 0 : 4 + (warp - 1) * 3;
    const int slot_count = warp == 0 ? 4 : 3;
    const int activation_count = 5 * slot_count + (warp == 7 ? 3 : 0);
    const int first_logical = group * 128 + 5 * first_slot;
    const float x = lane < activation_count
        ? load_q8_value(activation, first_logical + lane)
        : 0.0f;

#pragma unroll
    for (int local_slot = 0; local_slot < 4; ++local_slot) {
        if (local_slot >= slot_count) break;
        const int slot = first_slot + local_slot;
        const int code = lane;
        const int source = 5 * local_slot;
        const float x0 = __shfl_sync(0xffffffffu, x, source + 0);
        const float x1 = __shfl_sync(0xffffffffu, x, source + 1);
        const float x2 = __shfl_sync(0xffffffffu, x, source + 2);
        const float x3 = __shfl_sync(0xffffffffu, x, source + 3);
        const float x4 = __shfl_sync(0xffffffffu, x, source + 4);
        if (lane < 9) {
            lut.l2[slot][code] = trit0 * x0 + trit1 * x1;
        }
        if (lane < 27) {
            lut.l3[slot][code] =
                trit0 * x2 + trit1 * x3 + trit2 * x4;
        }
    }

    if (warp == 7) {
        const float x0 = __shfl_sync(0xffffffffu, x, 15);
        const float x1 = __shfl_sync(0xffffffffu, x, 16);
        const float x2 = __shfl_sync(0xffffffffu, x, 17);
        if (lane < 27) {
            lut.tail[lane] =
                trit0 * x0 + trit1 * x1 + trit2 * x2;
        }
    }
}

__device__ __forceinline__ void build_lut23(
        const block_q8_1_compat * __restrict__ activation,
        const int group,
        const float trit0,
        const float trit1,
        const float trit2,
        LutBuffer & lut) {
    compute_lut23(activation, group, trit0, trit1, trit2, lut);
    __syncthreads();
}

__device__ __forceinline__ float consume_code(
        const LutBuffer & lut,
        const int slot,
        const int packed) {
    const int upper = (packed * 57) >> 9;
    const int lower = packed - 9 * upper;
    return lut.l2[slot][lower] + lut.l3[slot][upper];
}

// K_CHUNK=512 launches 680 CTAs for the development tensor. Five resident
// 256-thread CTAs per Ada SM keep that grid in one wave while constraining
// ptxas to the spill-free register envelope needed for slot unrolling.
template <
    int GROUPS_PER_CHUNK,
    int FIXED_CHUNKS = 0,
    int FIXED_GROUPS = 0,
    bool PROFILE = false>
__global__ __launch_bounds__(LUT23_M_TILE, 5) void tq1_lut23_smem_streamk_kernel(
        const std::uint8_t * __restrict__ codes,
        const std::uint16_t * __restrict__ scales,
        const block_q8_1_compat * __restrict__ activation,
        float * __restrict__ output,
        float * __restrict__ workspace,
        unsigned int * __restrict__ tile_counters,
        const int rows,
        const int groups_per_row,
        const int chunks_per_row,
        lut23_phase_counters * __restrict__ phase_cycles) {
    __shared__ LutBuffer lut[2];
    __shared__ int final_chunk;

    const int tid = threadIdx.x;
    const int groups = FIXED_GROUPS > 0 ? FIXED_GROUPS : groups_per_row;
    const int chunks = FIXED_CHUNKS > 0 ? FIXED_CHUNKS : chunks_per_row;
    const int tile = blockIdx.x / chunks;
    const int chunk = blockIdx.x - tile * chunks;
    const int row = tile * LUT23_M_TILE + tid;
    const int first_group = chunk * GROUPS_PER_CHUNK;
    const int final_group = min(first_group + GROUPS_PER_CHUNK, groups);
    const int lane = tid & 31;
    const int lane_div3 = lane / 3;
    const float trit0 = static_cast<float>(lane - 3 * lane_div3 - 1);
    const float trit1 = static_cast<float>(lane_div3 - 3 * (lane_div3 / 3) - 1);
    const float trit2 = static_cast<float>(lane / 9 - 1);
    float partial = 0.0f;

    if (first_group < final_group) {
        unsigned long long phase_started = 0;
        if constexpr (PROFILE) {
            if (tid == 0) phase_started = clock64();
        }
        build_lut23(activation, first_group, trit0, trit1, trit2, lut[0]);
        if constexpr (PROFILE) {
            if (tid == 0) {
                atomicAdd(&phase_cycles->lut_build_cycles, clock64() - phase_started);
            }
        }

        for (int group = first_group; group < final_group; ++group) {
            const int local_group = group - first_group;
            const LutBuffer & current = lut[local_group & 1];
            if constexpr (PROFILE) {
                if (tid == 0) phase_started = clock64();
            }
            const std::size_t group_code_base =
                (static_cast<std::size_t>(tile) * groups + group) *
                LUT23_CODE_SLOTS * LUT23_M_TILE + tid;
            float group_sum = 0.0f;
#pragma unroll
            for (int batch = 0; batch < FULL_SLOTS / 5; ++batch) {
                const int first_slot = batch * 5;
                const int packed0 = codes[
                    group_code_base + (first_slot + 0) * LUT23_M_TILE];
                const int packed1 = codes[
                    group_code_base + (first_slot + 1) * LUT23_M_TILE];
                const int packed2 = codes[
                    group_code_base + (first_slot + 2) * LUT23_M_TILE];
                const int packed3 = codes[
                    group_code_base + (first_slot + 3) * LUT23_M_TILE];
                const int packed4 = codes[
                    group_code_base + (first_slot + 4) * LUT23_M_TILE];
                group_sum += consume_code(current, first_slot + 0, packed0);
                group_sum += consume_code(current, first_slot + 1, packed1);
                group_sum += consume_code(current, first_slot + 2, packed2);
                group_sum += consume_code(current, first_slot + 3, packed3);
                group_sum += consume_code(current, first_slot + 4, packed4);
            }
            const int tail_code = codes[group_code_base + 25 * LUT23_M_TILE];
            group_sum += current.tail[tail_code];
            const std::size_t scale_index =
                (static_cast<std::size_t>(tile) * groups + group) *
                LUT23_M_TILE + tid;
            partial = fmaf(half_bits_to_float(scales[scale_index]), group_sum, partial);
            if constexpr (PROFILE) {
                if (tid == 0) {
                    atomicAdd(&phase_cycles->code_consume_cycles, clock64() - phase_started);
                }
            }

            const int next_group = group + 1;
            if (next_group < final_group) {
                if constexpr (PROFILE) {
                    if (tid == 0) phase_started = clock64();
                }
                build_lut23(
                    activation, next_group, trit0, trit1, trit2,
                    lut[(local_group + 1) & 1]);
                if constexpr (PROFILE) {
                    if (tid == 0) {
                        atomicAdd(&phase_cycles->lut_build_cycles, clock64() - phase_started);
                    }
                }
            }
        }
    }

    unsigned long long reduction_started = 0;
    if constexpr (PROFILE) {
        if (tid == 0) reduction_started = clock64();
    }
    const std::size_t workspace_index =
        (static_cast<std::size_t>(tile) * chunks + chunk) * LUT23_M_TILE + tid;
    workspace[workspace_index] = partial;
    __threadfence();
    __syncthreads();
    if (tid == 0) {
        final_chunk = atomicAdd(&tile_counters[tile], 1u) ==
            static_cast<unsigned int>(chunks - 1);
    }
    __syncthreads();
    if (final_chunk) {
        float result = 0.0f;
        if constexpr (FIXED_CHUNKS > 0) {
#pragma unroll
            for (int index = 0; index < FIXED_CHUNKS; ++index) {
                result += workspace[
                    (static_cast<std::size_t>(tile) * FIXED_CHUNKS + index) *
                    LUT23_M_TILE + tid];
            }
        } else {
            for (int index = 0; index < chunks; ++index) {
                result += workspace[
                    (static_cast<std::size_t>(tile) * chunks + index) *
                    LUT23_M_TILE + tid];
            }
        }
        if (row < rows) output[row] = result;
        __syncthreads();
        if (tid == 0) atomicExch(&tile_counters[tile], 0u);
    }
    if constexpr (PROFILE) {
        if (tid == 0) {
            atomicAdd(&phase_cycles->reduction_cycles, clock64() - reduction_started);
        }
    }
}

template <int GROUPS_PER_CHUNK>
void launch_specialized(
        const std::uint8_t * codes,
        const std::uint16_t * scales,
        const block_q8_1_compat * activation,
        float * output,
        float * partial_workspace,
        unsigned int * tile_counters,
        int rows,
        int columns,
        cudaStream_t stream,
        lut23_phase_counters * phase_counters) {
    const int groups = columns / TQ1_BLOCK_SIZE;
    const int tiles = (rows + LUT23_M_TILE - 1) / LUT23_M_TILE;
    const int chunks = (groups + GROUPS_PER_CHUNK - 1) / GROUPS_PER_CHUNK;
    if constexpr (GROUPS_PER_CHUNK == 4) {
        if (groups == 136 && chunks == 34) {
            if (phase_counters != nullptr) {
                tq1_lut23_smem_streamk_kernel<4, 34, 136, true>
                    <<<tiles * 34, LUT23_M_TILE, 0, stream>>>(
                        codes, scales, activation, output, partial_workspace,
                        tile_counters, rows, 136, 34, phase_counters);
            } else {
                tq1_lut23_smem_streamk_kernel<4, 34, 136, false>
                    <<<tiles * 34, LUT23_M_TILE, 0, stream>>>(
                        codes, scales, activation, output, partial_workspace,
                        tile_counters, rows, 136, 34, nullptr);
            }
            return;
        }
    }
    if constexpr (GROUPS_PER_CHUNK == 8) {
        if (groups == 136 && chunks == 17) {
            if (phase_counters != nullptr) {
                tq1_lut23_smem_streamk_kernel<8, 17, 136, true>
                    <<<tiles * 17, LUT23_M_TILE, 0, stream>>>(
                        codes, scales, activation, output, partial_workspace,
                        tile_counters, rows, 136, 17, phase_counters);
            } else {
                tq1_lut23_smem_streamk_kernel<8, 17, 136, false>
                    <<<tiles * 17, LUT23_M_TILE, 0, stream>>>(
                        codes, scales, activation, output, partial_workspace,
                        tile_counters, rows, 136, 17, nullptr);
            }
            return;
        }
    }
    if (phase_counters != nullptr) {
        tq1_lut23_smem_streamk_kernel<GROUPS_PER_CHUNK, 0, 0, true>
            <<<tiles * chunks, LUT23_M_TILE, 0, stream>>>(
                codes, scales, activation, output, partial_workspace,
                tile_counters, rows, groups, chunks, phase_counters);
    } else {
        tq1_lut23_smem_streamk_kernel<GROUPS_PER_CHUNK, 0, 0, false>
            <<<tiles * chunks, LUT23_M_TILE, 0, stream>>>(
                codes, scales, activation, output, partial_workspace,
                tile_counters, rows, groups, chunks, nullptr);
    }
}

}  // namespace

std::size_t lut23_workspace_bytes(int rows, int columns, int k_chunk) {
    if (rows <= 0 || columns <= 0 || columns % TQ1_BLOCK_SIZE != 0 ||
        (k_chunk != 512 && k_chunk != 1024 && k_chunk != 2048)) {
        throw std::invalid_argument("invalid LUT23 workspace geometry");
    }
    const std::size_t tiles = (rows + LUT23_M_TILE - 1) / LUT23_M_TILE;
    const std::size_t chunks = (columns + k_chunk - 1) / k_chunk;
    return tiles * chunks * LUT23_M_TILE * sizeof(float);
}

std::size_t lut23_counter_bytes(int rows) {
    if (rows <= 0) throw std::invalid_argument("invalid LUT23 row count");
    return static_cast<std::size_t>((rows + LUT23_M_TILE - 1) / LUT23_M_TILE) *
        sizeof(unsigned int);
}

void launch_tq1_lut23_smem_streamk(
        const std::uint8_t * codes,
        const std::uint16_t * scales,
        const block_q8_1_compat * activation,
        float * output,
        float * partial_workspace,
        unsigned int * tile_counters,
        int rows,
        int columns,
        int k_chunk,
        cudaStream_t stream,
        lut23_phase_counters * phase_counters) {
    if (columns % TQ1_BLOCK_SIZE != 0) {
        throw std::invalid_argument("LUT23 columns must be divisible by 128");
    }
    switch (k_chunk) {
        case 512:
            launch_specialized<4>(codes, scales, activation, output, partial_workspace,
                                  tile_counters, rows, columns, stream, phase_counters);
            break;
        case 1024:
            launch_specialized<8>(codes, scales, activation, output, partial_workspace,
                                  tile_counters, rows, columns, stream, phase_counters);
            break;
        case 2048:
            launch_specialized<16>(codes, scales, activation, output, partial_workspace,
                                   tile_counters, rows, columns, stream, phase_counters);
            break;
        default:
            throw std::invalid_argument("K_CHUNK must be 512, 1024, or 2048");
    }
}

}  // namespace bonsai
