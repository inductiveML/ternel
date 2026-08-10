#pragma once

#include "tq1_g128.cuh"

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>

namespace bonsai {

constexpr int LUT23_M_TILE = 256;
constexpr int LUT23_CODE_SLOTS = 26;

struct lut23_phase_counters {
    unsigned long long lut_build_cycles;
    unsigned long long code_consume_cycles;
    unsigned long long reduction_cycles;
};

std::size_t lut23_workspace_bytes(int rows, int columns, int k_chunk);
std::size_t lut23_counter_bytes(int rows);

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
    lut23_phase_counters * phase_counters = nullptr);

}  // namespace bonsai
