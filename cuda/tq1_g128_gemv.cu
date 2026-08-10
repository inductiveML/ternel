#include "tq1_g128.cuh"

#include <cuda_fp16.h>

#include <array>
#include <mutex>
#include <stdexcept>

namespace bonsai {
namespace {

// V0 deliberately uses constant memory.  Divergent indices serialize within a
// warp and the measured V0 result establishes why this seemingly natural LUT
// placement is a poor fit for checkpoint data.
__constant__ std::uint64_t decode_lut_constant[243];

// V1 is the single focused optimization: retain the compact packed LUT, but
// serve it through the read-only global/L1 path and consume each five-trit
// entry exactly once with a rolling byte buffer and signed DP4A.
__device__ std::uint64_t decode_lut_global[243];

__device__ __forceinline__ float warp_sum(float value) {
#pragma unroll
    for (int offset = 16; offset > 0; offset /= 2) {
        value += __shfl_down_sync(0xffffffffu, value, offset);
    }
    return value;
}

__global__ __launch_bounds__(128, 1) void tq1_g128_gemv_v0_kernel(
        const block_tq1_g128 * __restrict__ weights,
        const block_q8_1_compat * __restrict__ activation,
        float * __restrict__ output,
        const int rows,
        const int groups_per_row) {
    const int row = blockIdx.x;
    if (row >= rows) {
        return;
    }
    const int tid = threadIdx.x;
    const int lane = tid & 31;
    const int warp = tid >> 5;
    const int chunk = tid & 3;
    float partial = 0.0f;
    const block_tq1_g128 * row_weights = weights + static_cast<std::size_t>(row) * groups_per_row;

    for (int group = tid >> 2; group < groups_per_row; group += 32) {
        const block_tq1_g128 & block = row_weights[group];
        const block_q8_1_compat & q8 = activation[group * 4 + chunk];
        const int start = chunk * 32;
        int integer_dot = 0;
#pragma unroll
        for (int local = 0; local < 32; ++local) {
            const int logical = start + local;
            const int byte_index = logical / 5;
            const int digit = logical - byte_index * 5;
            const std::uint64_t decoded = decode_lut_constant[block.qs[byte_index]];
            const std::int8_t weight = static_cast<std::int8_t>(decoded >> (digit * 8));
            integer_dot += static_cast<int>(weight) * static_cast<int>(q8.qs[local]);
        }
        const __half weight_scale = *reinterpret_cast<const __half *>(&block.d);
        const __half activation_scale = *reinterpret_cast<const __half *>(&q8.d);
        partial += __half2float(weight_scale) * __half2float(activation_scale) * integer_dot;
    }

    partial = warp_sum(partial);
    __shared__ float warp_partials[4];
    if (lane == 0) {
        warp_partials[warp] = partial;
    }
    __syncthreads();
    if (warp == 0) {
        float value = lane < 4 ? warp_partials[lane] : 0.0f;
        value = warp_sum(value);
        if (lane == 0) {
            output[row] = value;
        }
    }
}

__device__ __forceinline__ int tq1_dot_chunk_v1(
        const block_tq1_g128 & block,
        const block_q8_1_compat & q8,
        const int chunk) {
    const int start = chunk * 32;
    int byte_index = start / 5;
    const int skipped = start - byte_index * 5;
    std::uint64_t buffer = __ldg(&decode_lut_global[block.qs[byte_index++]]) >> (8 * skipped);
    int available = 5 - skipped;
    int integer_dot = 0;
    const int * activation4 = reinterpret_cast<const int *>(q8.qs);
#pragma unroll
    for (int quad = 0; quad < 8; ++quad) {
        if (available < 4) {
            const std::uint64_t decoded = __ldg(&decode_lut_global[block.qs[byte_index++]]);
            buffer |= decoded << (8 * available);
            available += 5;
        }
        integer_dot = __dp4a(static_cast<int>(static_cast<std::uint32_t>(buffer)), activation4[quad], integer_dot);
        buffer >>= 32;
        available -= 4;
    }
    return integer_dot;
}

__global__ __launch_bounds__(128, 1) void tq1_g128_gemv_v1_kernel(
        const block_tq1_g128 * __restrict__ weights,
        const block_q8_1_compat * __restrict__ activation,
        float * __restrict__ output,
        const int rows,
        const int groups_per_row) {
    const int row = blockIdx.x;
    if (row >= rows) return;

    const int tid = threadIdx.x;
    const int lane = tid & 31;
    const int warp = tid >> 5;
    const int chunk = tid & 3;
    float partial = 0.0f;
    const block_tq1_g128 * row_weights = weights + static_cast<std::size_t>(row) * groups_per_row;

    for (int group = tid >> 2; group < groups_per_row; group += 32) {
        const block_tq1_g128 & block = row_weights[group];
        const block_q8_1_compat & q8 = activation[group * 4 + chunk];
        const int integer_dot = tq1_dot_chunk_v1(block, q8, chunk);
        const __half weight_scale = *reinterpret_cast<const __half *>(&block.d);
        const __half activation_scale = *reinterpret_cast<const __half *>(&q8.d);
        partial += __half2float(weight_scale) * __half2float(activation_scale) * integer_dot;
    }

    partial = warp_sum(partial);
    __shared__ float warp_partials[4];
    if (lane == 0) warp_partials[warp] = partial;
    __syncthreads();
    if (warp == 0) {
        float value = lane < 4 ? warp_partials[lane] : 0.0f;
        value = warp_sum(value);
        if (lane == 0) output[row] = value;
    }
}

}  // namespace

void initialize_tq1_lut() {
    static std::once_flag once;
    std::call_once(once, [] {
        std::array<std::uint64_t, 243> host{};
        for (int encoded = 0; encoded < 243; ++encoded) {
            int value = encoded;
            std::uint64_t packed = 0;
            for (int digit = 0; digit < 5; ++digit) {
                const std::int8_t symbol = static_cast<std::int8_t>(value % 3 - 1);
                packed |= static_cast<std::uint64_t>(static_cast<std::uint8_t>(symbol)) << (8 * digit);
                value /= 3;
            }
            host[encoded] = packed;
        }
        cudaError_t error = cudaMemcpyToSymbol(decode_lut_constant, host.data(), sizeof(host));
        if (error != cudaSuccess) {
            throw std::runtime_error(cudaGetErrorString(error));
        }
        error = cudaMemcpyToSymbol(decode_lut_global, host.data(), sizeof(host));
        if (error != cudaSuccess) {
            throw std::runtime_error(cudaGetErrorString(error));
        }
    });
}

void launch_tq1_g128_gemv_v1(
        const block_tq1_g128 * weights,
        const block_q8_1_compat * activation,
        float * output,
        int rows,
        int columns,
        cudaStream_t stream) {
    if (columns % TQ1_BLOCK_SIZE != 0) {
        throw std::invalid_argument("TQ1 columns must be divisible by 128");
    }
    tq1_g128_gemv_v1_kernel<<<rows, 128, 0, stream>>>(weights, activation, output, rows, columns / TQ1_BLOCK_SIZE);
}

void launch_tq1_g128_gemv_v0(
        const block_tq1_g128 * weights,
        const block_q8_1_compat * activation,
        float * output,
        int rows,
        int columns,
        cudaStream_t stream) {
    if (columns % TQ1_BLOCK_SIZE != 0) {
        throw std::invalid_argument("TQ1 columns must be divisible by 128");
    }
    tq1_g128_gemv_v0_kernel<<<rows, 128, 0, stream>>>(weights, activation, output, rows, columns / TQ1_BLOCK_SIZE);
}

}  // namespace bonsai
