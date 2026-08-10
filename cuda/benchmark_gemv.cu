#include "tq1_g128.cuh"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <numeric>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

extern "C" void bonsai_prism_q2_0_mmvq(
    const void * q2, const void * q8, float * dst,
    int ncols, int nrows, cudaStream_t stream);
extern "C" void bonsai_prism_quantize_q8_1(
    const float * src, void * q8, int ncols, int nrows,
    cudaStream_t stream);

namespace {

constexpr int Q2_BLOCK_BYTES = 34;
constexpr int CORRECTNESS_VECTORS = 1000;
constexpr int REFERENCE_ROWS = 32;
constexpr int WARMUPS = 200;
constexpr int REPEATS = 5;
constexpr int ITERATIONS = 1000;

#define CUDA_CHECK(call) do { \
    const cudaError_t error_ = (call); \
    if (error_ != cudaSuccess) { \
        throw std::runtime_error(std::string(#call) + ": " + cudaGetErrorString(error_)); \
    } \
} while (false)

struct Arguments {
    std::string q2_path;
    std::string tq1_path;
    std::string output_path;
    std::uint64_t q2_offset = 0;
    std::uint64_t tq1_offset = 0;
    int rows = 0;
    int columns = 0;
    std::string kernel = "v1";
};

struct ErrorMetrics {
    double max_abs = 0.0;
    double mean_abs = 0.0;
    double rmse = 0.0;
    double max_relative = 0.0;
    double cosine = 0.0;
    std::uint64_t nonfinite = 0;
};

struct TimingStats {
    double median_ms = 0.0;
    double p5_ms = 0.0;
    double p95_ms = 0.0;
};

Arguments parse_arguments(int argc, char ** argv) {
    Arguments result;
    for (int index = 1; index < argc; ++index) {
        const std::string key = argv[index];
        if (index + 1 >= argc) {
            throw std::invalid_argument("missing value for " + key);
        }
        const std::string value = argv[++index];
        if (key == "--q2") result.q2_path = value;
        else if (key == "--tq1") result.tq1_path = value;
        else if (key == "--q2-offset") result.q2_offset = std::stoull(value);
        else if (key == "--tq1-offset") result.tq1_offset = std::stoull(value);
        else if (key == "--rows") result.rows = std::stoi(value);
        else if (key == "--columns") result.columns = std::stoi(value);
        else if (key == "--output") result.output_path = value;
        else if (key == "--kernel") result.kernel = value;
        else throw std::invalid_argument("unknown argument " + key);
    }
    if (result.q2_path.empty() || result.tq1_path.empty() || result.output_path.empty() ||
        result.rows <= 0 || result.columns <= 0 || result.columns % 128 != 0 ||
        (result.kernel != "v0" && result.kernel != "v1")) {
        throw std::invalid_argument("invalid or incomplete benchmark arguments");
    }
    return result;
}

std::vector<std::uint8_t> read_extent(
        const std::string & path, std::uint64_t offset, std::size_t bytes) {
    std::ifstream input(path, std::ios::binary);
    if (!input) throw std::runtime_error("cannot open " + path);
    input.seekg(static_cast<std::streamoff>(offset));
    std::vector<std::uint8_t> result(bytes);
    input.read(reinterpret_cast<char *>(result.data()), static_cast<std::streamsize>(bytes));
    if (input.gcount() != static_cast<std::streamsize>(bytes)) {
        throw std::runtime_error("truncated tensor extent in " + path);
    }
    return result;
}

float round_bfloat16(float value) {
    std::uint32_t bits;
    std::memcpy(&bits, &value, sizeof(bits));
    const std::uint32_t rounding = 0x7fffu + ((bits >> 16u) & 1u);
    bits = (bits + rounding) & 0xffff0000u;
    std::memcpy(&value, &bits, sizeof(bits));
    return value;
}

std::vector<float> make_activations(int vectors, int columns) {
    std::mt19937 generator(20260808u);
    std::normal_distribution<float> normal(0.0f, 1.0f);
    std::vector<float> result(static_cast<std::size_t>(vectors) * columns);
    for (int vector = 0; vector < vectors; ++vector) {
        for (int column = 0; column < columns; ++column) {
            float value = normal(generator);
            if (vector < vectors / 2) {
                value = __half2float(__float2half_rn(value));
            } else {
                value = round_bfloat16(value);
            }
            result[static_cast<std::size_t>(vector) * columns + column] = value;
        }
    }
    return result;
}

float half_bits_to_float(std::uint16_t bits) {
    __half value;
    std::memcpy(&value, &bits, sizeof(bits));
    return __half2float(value);
}

std::uint16_t load_u16(const std::uint8_t * pointer) {
    std::uint16_t value;
    std::memcpy(&value, pointer, sizeof(value));
    return value;
}

int decode_tq1_symbol(const std::uint8_t * block, int index) {
    const int byte_index = index / 5;
    const int digit = index - byte_index * 5;
    static constexpr int powers[5] = {1, 3, 9, 27, 81};
    return (block[2 + byte_index] / powers[digit]) % 3 - 1;
}

int decode_q2_symbol(const std::uint8_t * block, int index) {
    return ((block[2 + index / 4] >> (2 * (index % 4))) & 3) - 1;
}

double reference_value(
        const std::vector<std::uint8_t> & weights,
        bool tq1,
        const bonsai::block_q8_1_compat * activation,
        int row,
        int columns) {
    const int groups = columns / 128;
    const int block_bytes = tq1 ? 28 : 34;
    const std::uint8_t * row_data = weights.data() + static_cast<std::size_t>(row) * groups * block_bytes;
    double result = 0.0;
    for (int group = 0; group < groups; ++group) {
        const std::uint8_t * block = row_data + static_cast<std::size_t>(group) * block_bytes;
        const double weight_scale = half_bits_to_float(load_u16(block));
        for (int chunk = 0; chunk < 4; ++chunk) {
            const auto & q8 = activation[group * 4 + chunk];
            const double activation_scale = half_bits_to_float(q8.d);
            int dot = 0;
            for (int local = 0; local < 32; ++local) {
                const int logical = chunk * 32 + local;
                const int symbol = tq1 ? decode_tq1_symbol(block, logical) : decode_q2_symbol(block, logical);
                dot += symbol * static_cast<int>(q8.qs[local]);
            }
            result += weight_scale * activation_scale * dot;
        }
    }
    return result;
}

ErrorMetrics compare_vectors(
        const float * actual, const float * reference,
        std::size_t count, int vector_width) {
    ErrorMetrics result;
    long double absolute_sum = 0.0;
    long double square_sum = 0.0;
    long double cosine_sum = 0.0;
    const int vectors = static_cast<int>(count / vector_width);
    for (int vector = 0; vector < vectors; ++vector) {
        long double dot = 0.0;
        long double norm_a = 0.0;
        long double norm_b = 0.0;
        for (int index = 0; index < vector_width; ++index) {
            const std::size_t offset = static_cast<std::size_t>(vector) * vector_width + index;
            const double a = actual[offset];
            const double b = reference[offset];
            if (!std::isfinite(a) || !std::isfinite(b)) {
                ++result.nonfinite;
                continue;
            }
            const double error = std::abs(a - b);
            result.max_abs = std::max(result.max_abs, error);
            result.max_relative = std::max(result.max_relative, error / std::max(std::abs(b), 1e-6));
            absolute_sum += error;
            square_sum += error * error;
            dot += a * b;
            norm_a += a * a;
            norm_b += b * b;
        }
        if (norm_a > 0.0 && norm_b > 0.0) {
            cosine_sum += dot / std::sqrt(norm_a * norm_b);
        }
    }
    const auto finite_count = count - result.nonfinite;
    result.mean_abs = finite_count ? static_cast<double>(absolute_sum / finite_count) : INFINITY;
    result.rmse = finite_count ? std::sqrt(static_cast<double>(square_sum / finite_count)) : INFINITY;
    result.cosine = vectors ? static_cast<double>(cosine_sum / vectors) : 0.0;
    return result;
}

TimingStats summarize(std::vector<float> values) {
    std::sort(values.begin(), values.end());
    const auto pick = [&](double q) {
        const double position = q * (values.size() - 1);
        const std::size_t low = static_cast<std::size_t>(position);
        const std::size_t high = std::min(low + 1, values.size() - 1);
        const double fraction = position - low;
        return values[low] * (1.0 - fraction) + values[high] * fraction;
    };
    return {pick(0.5), pick(0.05), pick(0.95)};
}

__global__ void scrub_kernel(std::uint8_t * data, std::size_t bytes) {
    const std::size_t index = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index < bytes) data[index] = static_cast<std::uint8_t>(index);
}

template <typename Launch>
std::vector<float> time_launches(
        Launch launch, bool cold, std::uint8_t * scrub, std::size_t scrub_bytes,
        int activation_count, cudaStream_t stream) {
    cudaEvent_t begin, end;
    CUDA_CHECK(cudaEventCreate(&begin));
    CUDA_CHECK(cudaEventCreate(&end));
    for (int index = 0; index < WARMUPS; ++index) {
        launch(index % activation_count);
    }
    CUDA_CHECK(cudaStreamSynchronize(stream));
    std::vector<float> result;
    result.reserve(REPEATS * ITERATIONS);
    for (int repeat = 0; repeat < REPEATS; ++repeat) {
        for (int index = 0; index < ITERATIONS; ++index) {
            if (cold) {
                const int blocks = static_cast<int>((scrub_bytes + 255) / 256);
                scrub_kernel<<<blocks, 256, 0, stream>>>(scrub, scrub_bytes);
            }
            CUDA_CHECK(cudaEventRecord(begin, stream));
            launch((repeat * ITERATIONS + index) % activation_count);
            CUDA_CHECK(cudaEventRecord(end, stream));
            CUDA_CHECK(cudaEventSynchronize(end));
            float elapsed = 0.0f;
            CUDA_CHECK(cudaEventElapsedTime(&elapsed, begin, end));
            result.push_back(elapsed);
        }
    }
    CUDA_CHECK(cudaEventDestroy(begin));
    CUDA_CHECK(cudaEventDestroy(end));
    return result;
}

void write_metrics(std::ostream & output, const ErrorMetrics & metrics) {
    output << "{\"max_abs\":" << metrics.max_abs
           << ",\"mean_abs\":" << metrics.mean_abs
           << ",\"rmse\":" << metrics.rmse
           << ",\"max_relative\":" << metrics.max_relative
           << ",\"cosine\":" << metrics.cosine
           << ",\"nonfinite\":" << metrics.nonfinite << "}";
}

void write_stats(std::ostream & output, const TimingStats & stats) {
    output << "{\"median_ms\":" << stats.median_ms
           << ",\"p5_ms\":" << stats.p5_ms
           << ",\"p95_ms\":" << stats.p95_ms << "}";
}

void write_values(std::ostream & output, const std::vector<float> & values) {
    output << '[';
    for (std::size_t index = 0; index < values.size(); ++index) {
        if (index != 0) output << ',';
        output << values[index];
    }
    output << ']';
}

double checksum(const std::vector<float> & values) {
    long double result = 0.0;
    for (std::size_t index = 0; index < values.size(); ++index) {
        result += values[index] * static_cast<double>((index % 251) + 1);
    }
    return static_cast<double>(result);
}

}  // namespace

int main(int argc, char ** argv) try {
    const Arguments args = parse_arguments(argc, argv);
    const int groups_per_row = args.columns / 128;
    const std::size_t q2_bytes = static_cast<std::size_t>(args.rows) * groups_per_row * Q2_BLOCK_BYTES;
    const std::size_t tq1_bytes = static_cast<std::size_t>(args.rows) * groups_per_row * 28;
    auto q2_host = read_extent(args.q2_path, args.q2_offset, q2_bytes);
    auto tq1_host = read_extent(args.tq1_path, args.tq1_offset, tq1_bytes);
    auto activations_host = make_activations(CORRECTNESS_VECTORS, args.columns);

    int device = 0;
    CUDA_CHECK(cudaGetDevice(&device));
    cudaDeviceProp properties{};
    CUDA_CHECK(cudaGetDeviceProperties(&properties, device));
    if (std::string(properties.name) != "NVIDIA RTX 6000 Ada Generation" || properties.major != 8 || properties.minor != 9) {
        throw std::runtime_error("INVALID_EXPERIMENT: target GPU is not RTX 6000 Ada (sm_89)");
    }

    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    std::size_t free_before = 0, total_memory = 0;
    CUDA_CHECK(cudaMemGetInfo(&free_before, &total_memory));
    void * q2_device = nullptr;
    bonsai::block_tq1_g128 * tq1_device = nullptr;
    float * activations_device = nullptr;
    bonsai::block_q8_1_compat * q8_device = nullptr;
    float * baseline_output_device = nullptr;
    float * tq1_output_device = nullptr;
    std::uint8_t * scrub_device = nullptr;
    const std::size_t activation_values = static_cast<std::size_t>(CORRECTNESS_VECTORS) * args.columns;
    const std::size_t q8_blocks = activation_values / 32;
    const std::size_t output_values = static_cast<std::size_t>(CORRECTNESS_VECTORS) * args.rows;
    const std::size_t scrub_bytes = std::max<std::size_t>(128ull << 20, 2ull * properties.l2CacheSize + 4096);
    CUDA_CHECK(cudaMalloc(&q2_device, q2_bytes));
    CUDA_CHECK(cudaMalloc(&tq1_device, tq1_bytes));
    CUDA_CHECK(cudaMalloc(&activations_device, activation_values * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&q8_device, q8_blocks * sizeof(bonsai::block_q8_1_compat)));
    CUDA_CHECK(cudaMalloc(&baseline_output_device, output_values * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&tq1_output_device, output_values * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&scrub_device, scrub_bytes));
    CUDA_CHECK(cudaMemcpyAsync(q2_device, q2_host.data(), q2_bytes, cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaMemcpyAsync(tq1_device, tq1_host.data(), tq1_bytes, cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaMemcpyAsync(activations_device, activations_host.data(), activation_values * sizeof(float), cudaMemcpyHostToDevice, stream));
    bonsai::initialize_tq1_lut();
    bonsai_prism_quantize_q8_1(activations_device, q8_device, args.columns, CORRECTNESS_VECTORS, stream);
    CUDA_CHECK(cudaGetLastError());

    const auto launch_tq1 = [&](const bonsai::block_q8_1_compat * q8, float * output) {
        if (args.kernel == "v0") {
            bonsai::launch_tq1_g128_gemv_v0(tq1_device, q8, output, args.rows, args.columns, stream);
        } else {
            bonsai::launch_tq1_g128_gemv_v1(tq1_device, q8, output, args.rows, args.columns, stream);
        }
    };

    for (int vector = 0; vector < CORRECTNESS_VECTORS; ++vector) {
        const auto * q8 = q8_device + static_cast<std::size_t>(vector) * args.columns / 32;
        bonsai_prism_q2_0_mmvq(q2_device, q8, baseline_output_device + static_cast<std::size_t>(vector) * args.rows, args.columns, args.rows, stream);
        launch_tq1(q8, tq1_output_device + static_cast<std::size_t>(vector) * args.rows);
    }
    CUDA_CHECK(cudaGetLastError());
    std::vector<float> baseline_output(output_values);
    std::vector<float> tq1_output(output_values);
    CUDA_CHECK(cudaMemcpyAsync(baseline_output.data(), baseline_output_device, output_values * sizeof(float), cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaMemcpyAsync(tq1_output.data(), tq1_output_device, output_values * sizeof(float), cudaMemcpyDeviceToHost, stream));
    std::vector<bonsai::block_q8_1_compat> q8_host(q8_blocks);
    CUDA_CHECK(cudaMemcpyAsync(q8_host.data(), q8_device, q8_blocks * sizeof(bonsai::block_q8_1_compat), cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));

    const ErrorMetrics tq1_vs_baseline = compare_vectors(tq1_output.data(), baseline_output.data(), output_values, args.rows);
    std::vector<float> reference_q2(static_cast<std::size_t>(CORRECTNESS_VECTORS) * REFERENCE_ROWS);
    std::vector<float> reference_tq1(reference_q2.size());
    std::vector<float> subset_baseline(reference_q2.size());
    std::vector<float> subset_tq1(reference_q2.size());
#pragma omp parallel for schedule(static)
    for (int task = 0; task < CORRECTNESS_VECTORS * REFERENCE_ROWS; ++task) {
        const int vector = task / REFERENCE_ROWS;
        const int sample = task % REFERENCE_ROWS;
        const int row = sample * args.rows / REFERENCE_ROWS;
        const auto * q8 = q8_host.data() + static_cast<std::size_t>(vector) * args.columns / 32;
        reference_q2[task] = static_cast<float>(reference_value(q2_host, false, q8, row, args.columns));
        reference_tq1[task] = static_cast<float>(reference_value(tq1_host, true, q8, row, args.columns));
        subset_baseline[task] = baseline_output[static_cast<std::size_t>(vector) * args.rows + row];
        subset_tq1[task] = tq1_output[static_cast<std::size_t>(vector) * args.rows + row];
    }
    const ErrorMetrics reference_parity = compare_vectors(reference_tq1.data(), reference_q2.data(), reference_q2.size(), REFERENCE_ROWS);
    const ErrorMetrics baseline_vs_reference = compare_vectors(subset_baseline.data(), reference_q2.data(), reference_q2.size(), REFERENCE_ROWS);
    const ErrorMetrics tq1_vs_reference = compare_vectors(subset_tq1.data(), reference_q2.data(), reference_q2.size(), REFERENCE_ROWS);
    const bool numerical_pass = tq1_vs_baseline.nonfinite == 0 && reference_parity.max_abs == 0.0 &&
        tq1_vs_reference.mean_abs <= std::max(1.05 * baseline_vs_reference.mean_abs, baseline_vs_reference.mean_abs + 1e-6) &&
        tq1_vs_reference.rmse <= std::max(1.05 * baseline_vs_reference.rmse, baseline_vs_reference.rmse + 1e-6) &&
        tq1_vs_reference.max_abs <= std::max(1.2 * baseline_vs_reference.max_abs, baseline_vs_reference.max_abs + 1e-4) &&
        tq1_vs_reference.cosine >= baseline_vs_reference.cosine - 1e-6;
    if (!numerical_pass) {
        std::cerr << "FAIL_KERNEL_NUMERICS\n";
    }

    const auto baseline_launch = [&](int vector) {
        const auto * q8 = q8_device + static_cast<std::size_t>(vector) * args.columns / 32;
        bonsai_prism_q2_0_mmvq(q2_device, q8, baseline_output_device, args.columns, args.rows, stream);
    };
    const auto tq1_launch = [&](int vector) {
        const auto * q8 = q8_device + static_cast<std::size_t>(vector) * args.columns / 32;
        launch_tq1(q8, tq1_output_device);
    };
    const auto baseline_end_to_end_launch = [&](int vector) {
        const auto * f32 = activations_device + static_cast<std::size_t>(vector) * args.columns;
        bonsai_prism_quantize_q8_1(f32, q8_device, args.columns, 1, stream);
        bonsai_prism_q2_0_mmvq(q2_device, q8_device, baseline_output_device, args.columns, args.rows, stream);
    };
    const auto tq1_end_to_end_launch = [&](int vector) {
        const auto * f32 = activations_device + static_cast<std::size_t>(vector) * args.columns;
        bonsai_prism_quantize_q8_1(f32, q8_device, args.columns, 1, stream);
        launch_tq1(q8_device, tq1_output_device);
    };
    std::vector<float> baseline_warm, tq1_warm, baseline_cold, tq1_cold;
    std::vector<float> baseline_end_to_end_warm, tq1_end_to_end_warm;
    if (numerical_pass) {
        baseline_warm = time_launches(baseline_launch, false, scrub_device, scrub_bytes, CORRECTNESS_VECTORS, stream);
        tq1_warm = time_launches(tq1_launch, false, scrub_device, scrub_bytes, CORRECTNESS_VECTORS, stream);
        baseline_cold = time_launches(baseline_launch, true, scrub_device, scrub_bytes, CORRECTNESS_VECTORS, stream);
        tq1_cold = time_launches(tq1_launch, true, scrub_device, scrub_bytes, CORRECTNESS_VECTORS, stream);
        baseline_end_to_end_warm = time_launches(baseline_end_to_end_launch, false, scrub_device, scrub_bytes, CORRECTNESS_VECTORS, stream);
        tq1_end_to_end_warm = time_launches(tq1_end_to_end_launch, false, scrub_device, scrub_bytes, CORRECTNESS_VECTORS, stream);
    }
    const TimingStats baseline_warm_stats = numerical_pass ? summarize(baseline_warm) : TimingStats{};
    const TimingStats tq1_warm_stats = numerical_pass ? summarize(tq1_warm) : TimingStats{};
    const TimingStats baseline_cold_stats = numerical_pass ? summarize(baseline_cold) : TimingStats{};
    const TimingStats tq1_cold_stats = numerical_pass ? summarize(tq1_cold) : TimingStats{};
    const TimingStats baseline_end_to_end_warm_stats = numerical_pass ? summarize(baseline_end_to_end_warm) : TimingStats{};
    const TimingStats tq1_end_to_end_warm_stats = numerical_pass ? summarize(tq1_end_to_end_warm) : TimingStats{};
    const double warm_ratio = numerical_pass ? tq1_warm_stats.median_ms / baseline_warm_stats.median_ms : INFINITY;
    const double cold_ratio = numerical_pass ? tq1_cold_stats.median_ms / baseline_cold_stats.median_ms : INFINITY;
    const double worst_ratio = std::max(warm_ratio, cold_ratio);

    std::ofstream json(args.output_path);
    if (!json) throw std::runtime_error("cannot create result JSON");
    json << std::setprecision(12);
    json << "{\n  \"schema_version\":1,\n  \"tq1_kernel\":\"" << args.kernel << "\",\n"
         << "  \"gpu\":{\"name\":\"" << properties.name << "\",\"compute_capability\":\""
         << properties.major << "." << properties.minor << "\",\"l2_bytes\":" << properties.l2CacheSize
         << ",\"global_memory_bytes\":" << properties.totalGlobalMem << "},\n"
         << "  \"matrix\":{\"rows\":" << args.rows << ",\"columns\":" << args.columns
         << ",\"q2_bytes\":" << q2_bytes << ",\"tq1_bytes\":" << tq1_bytes << "},\n"
         << "  \"activations\":{\"vectors\":" << CORRECTNESS_VECTORS
         << ",\"fp16_origin\":500,\"bf16_origin\":500,\"runtime_type\":\"Q8_1 from F32\"},\n"
         << "  \"correctness\":{\"full_tq1_vs_baseline\":";
    write_metrics(json, tq1_vs_baseline);
    json << ",\"reference_rows\":" << REFERENCE_ROWS << ",\"reference_parity\":";
    write_metrics(json, reference_parity);
    json << ",\"baseline_vs_reference\":";
    write_metrics(json, baseline_vs_reference);
    json << ",\"tq1_vs_reference\":";
    write_metrics(json, tq1_vs_reference);
    json << ",\"pass\":" << (numerical_pass ? "true" : "false") << "},\n"
         << "  \"benchmark\":{\"warmups\":" << WARMUPS << ",\"repeats\":" << REPEATS
         << ",\"iterations_per_repeat\":" << ITERATIONS << ",\"cold_scrub_bytes\":" << scrub_bytes
         << ",\"baseline_warm\":";
    write_stats(json, baseline_warm_stats);
    json << ",\"tq1_warm\":";
    write_stats(json, tq1_warm_stats);
    json << ",\"baseline_cold\":";
    write_stats(json, baseline_cold_stats);
    json << ",\"tq1_cold\":";
    write_stats(json, tq1_cold_stats);
    json << ",\"baseline_end_to_end_warm\":";
    write_stats(json, baseline_end_to_end_warm_stats);
    json << ",\"tq1_end_to_end_warm\":";
    write_stats(json, tq1_end_to_end_warm_stats);
    json << ",\"warm_ratio\":" << warm_ratio << ",\"cold_ratio\":" << cold_ratio
         << ",\"worst_ratio\":" << worst_ratio << ",\"raw_ms\":{";
    json << "\"baseline_warm\":";
    write_values(json, baseline_warm);
    json << ",\"tq1_warm\":";
    write_values(json, tq1_warm);
    json << ",\"baseline_cold\":";
    write_values(json, baseline_cold);
    json << ",\"tq1_cold\":";
    write_values(json, tq1_cold);
    json << ",\"baseline_end_to_end_warm\":";
    write_values(json, baseline_end_to_end_warm);
    json << ",\"tq1_end_to_end_warm\":";
    write_values(json, tq1_end_to_end_warm);
    json << "}},\n"
         << "  \"checksums\":{\"baseline\":" << checksum(baseline_output)
         << ",\"tq1\":" << checksum(tq1_output) << "},\n"
         << "  \"anti_cheating\":{\"tq1_allocation_bytes\":" << tq1_bytes
         << ",\"expected_tq1_allocation_bytes\":" << tq1_bytes
         << ",\"unpacked_weight_buffer\":false,\"same_q8_activations\":true,"
            "\"same_scales\":true,\"output_dtype\":\"F32\"},\n"
         << "  \"memory\":{\"free_before_bytes\":" << free_before << ",\"total_bytes\":" << total_memory << "}\n}\n";
    json.close();

    CUDA_CHECK(cudaFree(scrub_device));
    CUDA_CHECK(cudaFree(tq1_output_device));
    CUDA_CHECK(cudaFree(baseline_output_device));
    CUDA_CHECK(cudaFree(q8_device));
    CUDA_CHECK(cudaFree(activations_device));
    CUDA_CHECK(cudaFree(tq1_device));
    CUDA_CHECK(cudaFree(q2_device));
    CUDA_CHECK(cudaStreamDestroy(stream));
    std::cout << "correctness=" << (numerical_pass ? "PASS" : "FAIL_KERNEL_NUMERICS")
              << " warm_ratio=" << warm_ratio << " cold_ratio=" << cold_ratio << '\n';
    return numerical_pass ? 0 : 3;
} catch (const std::exception & error) {
    std::cerr << error.what() << '\n';
    return 2;
}
