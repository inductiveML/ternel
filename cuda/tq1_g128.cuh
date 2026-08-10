#pragma once

#include <cuda_runtime.h>
#include <cstddef>
#include <cstdint>

namespace bonsai {

constexpr int TQ1_BLOCK_SIZE = 128;
constexpr int TQ1_BLOCK_BYTES = 28;
constexpr int Q8_BLOCK_SIZE = 32;

#pragma pack(push, 1)
struct block_tq1_g128 {
    std::uint16_t d;
    std::uint8_t qs[26];
};
#pragma pack(pop)

struct block_q8_1_compat {
    std::uint16_t d;
    std::uint16_t s;
    std::int8_t qs[32];
};

static_assert(sizeof(block_tq1_g128) == TQ1_BLOCK_BYTES);
static_assert(sizeof(block_q8_1_compat) == 36);

void initialize_tq1_lut();

void launch_tq1_g128_gemv_v0(
    const block_tq1_g128 * weights,
    const block_q8_1_compat * activation,
    float * output,
    int rows,
    int columns,
    cudaStream_t stream);

void launch_tq1_g128_gemv_v1(
    const block_tq1_g128 * weights,
    const block_q8_1_compat * activation,
    float * output,
    int rows,
    int columns,
    cudaStream_t stream);

}  // namespace bonsai
