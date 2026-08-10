#include "tq1_g128.cuh"
#include "tq1_lut23_streamk.cuh"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <array>
#include <cerrno>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <fcntl.h>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <numeric>
#include <random>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include <unistd.h>

extern "C" void bonsai_prism_q2_0_mmvq(
    const void * q2, const void * q8, float * dst,
    int ncols, int nrows, cudaStream_t stream);
extern "C" void bonsai_prism_quantize_q8_1(
    const float * src, void * q8, int ncols, int nrows,
    cudaStream_t stream);

namespace {

constexpr int Q2_BLOCK_BYTES = 34;
constexpr int RANDOM_VECTORS = 128;
constexpr int EDGE_VECTORS = 8;
constexpr int CORRECTNESS_VECTORS = RANDOM_VECTORS + EDGE_VECTORS;
constexpr int REFERENCE_ROWS = 32;
constexpr int DEFAULT_WARMUPS = 200;
constexpr int DEFAULT_PAIRS = 200;
constexpr int RING_TENSORS = 6;
constexpr int RING_ACTIVATIONS = 8;
constexpr double MAX_ABS_LIMIT = 1.5e-6;

#define CUDA_CHECK(call) do { \
    const cudaError_t error_ = (call); \
    if (error_ != cudaSuccess) { \
        throw std::runtime_error(std::string(#call) + ": " + cudaGetErrorString(error_)); \
    } \
} while (false)

struct Arguments {
    std::string manifest_path;
    std::string q2_path;
    std::string tq1_path;
    std::string reordered_path;
    std::string output_path;
    std::string mode = "qualify";
    int k_chunk = 1024;
    int warmups = DEFAULT_WARMUPS;
    int pairs = DEFAULT_PAIRS;
};

struct TensorDescriptor {
    int order = 0;
    int layer = 0;
    std::string kind;
    std::string name;
    int rows = 0;
    int columns = 0;
    std::uint64_t groups = 0;
    std::uint64_t q2_file_offset = 0;
    std::uint64_t q2_bytes = 0;
    std::uint64_t tq1_file_offset = 0;
    std::uint64_t tq1_bytes = 0;
    std::uint64_t codes_file_offset = 0;
    std::uint64_t codes_bytes = 0;
    std::uint64_t scales_file_offset = 0;
    std::uint64_t scales_bytes = 0;
    int tiles = 0;
    int groups_per_row = 0;
};

struct ErrorMetrics {
    double max_abs = 0.0;
    double mean_abs = 0.0;
    double rmse = 0.0;
    double cosine = 0.0;
    std::uint64_t nonfinite = 0;
    std::uint64_t values = 0;
};

struct Summary {
    double median = 0.0;
    double mean = 0.0;
    double p5 = 0.0;
    double p95 = 0.0;
    double stddev = 0.0;
};

struct PairResult {
    std::vector<double> baseline_ms;
    std::vector<double> candidate_ms;
    std::vector<double> ratios;
    std::vector<std::string> order;
    std::vector<double> baseline_checksums;
    std::vector<double> candidate_checksums;
};

Arguments parse_arguments(int argc, char ** argv) {
    Arguments result;
    for (int index = 1; index < argc; ++index) {
        const std::string key = argv[index];
        if (index + 1 >= argc) throw std::invalid_argument("missing value for " + key);
        const std::string value = argv[++index];
        if (key == "--manifest") result.manifest_path = value;
        else if (key == "--q2") result.q2_path = value;
        else if (key == "--tq1") result.tq1_path = value;
        else if (key == "--reordered") result.reordered_path = value;
        else if (key == "--output") result.output_path = value;
        else if (key == "--mode") result.mode = value;
        else if (key == "--k-chunk") result.k_chunk = std::stoi(value);
        else if (key == "--warmups") result.warmups = std::stoi(value);
        else if (key == "--pairs") result.pairs = std::stoi(value);
        else throw std::invalid_argument("unknown argument " + key);
    }
    if (result.manifest_path.empty() || result.q2_path.empty() || result.tq1_path.empty() ||
        result.reordered_path.empty() || result.output_path.empty() ||
        (result.mode != "qualify" && result.mode != "profile" && result.mode != "ring") ||
        (result.k_chunk != 512 && result.k_chunk != 1024 && result.k_chunk != 2048) ||
        result.warmups < 20 || result.pairs < 30) {
        throw std::invalid_argument("invalid or incomplete LUT23 primary arguments");
    }
    return result;
}

std::vector<std::string> split_tsv(const std::string & line) {
    std::vector<std::string> fields;
    std::size_t begin = 0;
    while (true) {
        const std::size_t end = line.find('\t', begin);
        if (end == std::string::npos) {
            fields.push_back(line.substr(begin));
            return fields;
        }
        fields.push_back(line.substr(begin, end - begin));
        begin = end + 1;
    }
}

std::vector<TensorDescriptor> read_manifest(const std::string & path) {
    std::ifstream input(path);
    if (!input) throw std::runtime_error("cannot open " + path);
    std::string line;
    if (!std::getline(input, line)) throw std::runtime_error("empty LUT23 manifest");
    const auto header = split_tsv(line);
    std::unordered_map<std::string, std::size_t> columns;
    for (std::size_t index = 0; index < header.size(); ++index) columns.emplace(header[index], index);
    const auto get = [&](const std::vector<std::string> & fields, const std::string & name) -> const std::string & {
        const auto found = columns.find(name);
        if (found == columns.end() || found->second >= fields.size()) {
            throw std::runtime_error("missing manifest field " + name);
        }
        return fields[found->second];
    };
    std::vector<TensorDescriptor> result;
    while (std::getline(input, line)) {
        if (line.empty()) continue;
        const auto fields = split_tsv(line);
        TensorDescriptor tensor;
        tensor.order = std::stoi(get(fields, "order"));
        tensor.layer = std::stoi(get(fields, "layer"));
        tensor.kind = get(fields, "kind");
        tensor.name = get(fields, "name");
        tensor.rows = std::stoi(get(fields, "m"));
        tensor.columns = std::stoi(get(fields, "k"));
        tensor.groups = std::stoull(get(fields, "groups"));
        tensor.q2_file_offset = std::stoull(get(fields, "q2_file_offset"));
        tensor.q2_bytes = std::stoull(get(fields, "q2_bytes"));
        tensor.tq1_file_offset = std::stoull(get(fields, "tq1_file_offset"));
        tensor.tq1_bytes = std::stoull(get(fields, "tq1_bytes"));
        tensor.codes_file_offset = std::stoull(get(fields, "codes_file_offset"));
        tensor.codes_bytes = std::stoull(get(fields, "codes_bytes"));
        tensor.scales_file_offset = std::stoull(get(fields, "scales_file_offset"));
        tensor.scales_bytes = std::stoull(get(fields, "scales_bytes"));
        tensor.tiles = std::stoi(get(fields, "tiles"));
        tensor.groups_per_row = std::stoi(get(fields, "groups_per_row"));
        if (tensor.columns / 128 != tensor.groups_per_row ||
            tensor.groups != static_cast<std::uint64_t>(tensor.rows) * tensor.groups_per_row ||
            tensor.q2_bytes != tensor.groups * Q2_BLOCK_BYTES ||
            tensor.tq1_bytes != tensor.groups * sizeof(bonsai::block_tq1_g128) ||
            tensor.codes_bytes != static_cast<std::uint64_t>(tensor.tiles) * tensor.groups_per_row * 26 * 256 ||
            tensor.scales_bytes != static_cast<std::uint64_t>(tensor.tiles) * tensor.groups_per_row * 2 * 256 ||
            tensor.codes_file_offset % 256 || tensor.scales_file_offset % 256) {
            throw std::runtime_error("invalid tensor geometry for " + tensor.name);
        }
        result.push_back(std::move(tensor));
    }
    if (result.size() != 497) throw std::runtime_error("benchmark manifest must contain 497 tensors");
    return result;
}

std::vector<std::uint8_t> read_extent(
        const std::string & path, std::uint64_t offset, std::size_t bytes) {
    const int descriptor = open(path.c_str(), O_RDONLY);
    if (descriptor < 0) throw std::runtime_error("cannot open " + path);
    std::vector<std::uint8_t> result(bytes);
    std::size_t done = 0;
    while (done < bytes) {
        const ssize_t count = pread(
            descriptor, result.data() + done, bytes - done,
            static_cast<off_t>(offset + done));
        if (count < 0 && errno == EINTR) continue;
        if (count < 0) {
            close(descriptor);
            throw std::runtime_error("pread failed for " + path + ": " + std::strerror(errno));
        }
        if (count == 0) {
            close(descriptor);
            throw std::runtime_error("truncated extent in " + path);
        }
        done += static_cast<std::size_t>(count);
    }
    close(descriptor);
    return result;
}

float round_bfloat16(float value) {
    std::uint32_t bits;
    std::memcpy(&bits, &value, sizeof(bits));
    bits = (bits + 0x7fffu + ((bits >> 16u) & 1u)) & 0xffff0000u;
    std::memcpy(&value, &bits, sizeof(bits));
    return value;
}

std::vector<float> make_activations(int columns) {
    std::vector<float> result(static_cast<std::size_t>(CORRECTNESS_VECTORS) * columns);
    std::mt19937 generator(20260809u);
    std::normal_distribution<float> normal(0.0f, 1.0f);
    for (int vector = 0; vector < RANDOM_VECTORS; ++vector) {
        for (int column = 0; column < columns; ++column) {
            float value = normal(generator);
            value = vector < RANDOM_VECTORS / 2
                ? __half2float(__float2half_rn(value))
                : round_bfloat16(value);
            result[static_cast<std::size_t>(vector) * columns + column] = value;
        }
    }
    for (int column = 0; column < columns; ++column) {
        result[static_cast<std::size_t>(128) * columns + column] = 0.0f;
        result[static_cast<std::size_t>(129) * columns + column] = 1.0f;
        result[static_cast<std::size_t>(130) * columns + column] = -1.0f;
        result[static_cast<std::size_t>(131) * columns + column] = column & 1 ? -1.0f : 1.0f;
        result[static_cast<std::size_t>(132) * columns + column] =
            static_cast<float>((column % 257) - 128) / 32.0f;
        result[static_cast<std::size_t>(133) * columns + column] =
            column & 1 ? -0.0009765625f : 0.0009765625f;
        result[static_cast<std::size_t>(134) * columns + column] = column == columns / 2 ? 16.0f : 0.0f;
        result[static_cast<std::size_t>(135) * columns + column] =
            std::sin(static_cast<float>(column) * 0.03125f);
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

float reference_value(
        const std::vector<std::uint8_t> & weights,
        const bonsai::block_q8_1_compat * activation,
        int row,
        int columns) {
    const int groups = columns / 128;
    const std::uint8_t * row_data = weights.data() +
        static_cast<std::size_t>(row) * groups * sizeof(bonsai::block_tq1_g128);
    double result = 0.0;
    for (int group = 0; group < groups; ++group) {
        const std::uint8_t * block = row_data +
            static_cast<std::size_t>(group) * sizeof(bonsai::block_tq1_g128);
        const double weight_scale = half_bits_to_float(load_u16(block));
        for (int chunk = 0; chunk < 4; ++chunk) {
            const auto & q8 = activation[group * 4 + chunk];
            int dot = 0;
            for (int local = 0; local < 32; ++local) {
                dot += decode_tq1_symbol(block, chunk * 32 + local) *
                    static_cast<int>(q8.qs[local]);
            }
            result += weight_scale * half_bits_to_float(q8.d) * dot;
        }
    }
    return static_cast<float>(result);
}

ErrorMetrics compare(const std::vector<float> & actual, const std::vector<float> & reference, int width) {
    if (actual.size() != reference.size() || actual.size() % width) {
        throw std::invalid_argument("comparison shape mismatch");
    }
    ErrorMetrics result;
    long double absolute_sum = 0.0;
    long double square_sum = 0.0;
    long double cosine_sum = 0.0;
    int cosine_vectors = 0;
    const int vectors = static_cast<int>(actual.size() / width);
    for (int vector = 0; vector < vectors; ++vector) {
        long double dot = 0.0, norm_a = 0.0, norm_b = 0.0;
        for (int index = 0; index < width; ++index) {
            const std::size_t offset = static_cast<std::size_t>(vector) * width + index;
            const double left = actual[offset];
            const double right = reference[offset];
            if (!std::isfinite(left) || !std::isfinite(right)) {
                ++result.nonfinite;
                continue;
            }
            const double error = std::abs(left - right);
            result.max_abs = std::max(result.max_abs, error);
            absolute_sum += error;
            square_sum += error * error;
            dot += left * right;
            norm_a += left * left;
            norm_b += right * right;
        }
        if (norm_a > 0.0 && norm_b > 0.0) {
            cosine_sum += dot / std::sqrt(norm_a * norm_b);
            ++cosine_vectors;
        }
    }
    result.values = actual.size();
    const std::uint64_t finite = result.values - result.nonfinite;
    result.mean_abs = finite ? static_cast<double>(absolute_sum / finite) : INFINITY;
    result.rmse = finite ? std::sqrt(static_cast<double>(square_sum / finite)) : INFINITY;
    result.cosine = cosine_vectors ? static_cast<double>(cosine_sum / cosine_vectors) : 1.0;
    return result;
}

double quantile_sorted(const std::vector<double> & values, double q) {
    const double position = q * (values.size() - 1);
    const std::size_t low = static_cast<std::size_t>(position);
    const std::size_t high = std::min(low + 1, values.size() - 1);
    const double fraction = position - low;
    return values[low] * (1.0 - fraction) + values[high] * fraction;
}

Summary summarize(const std::vector<double> & values) {
    if (values.empty()) return {};
    std::vector<double> sorted = values;
    std::sort(sorted.begin(), sorted.end());
    const double mean = std::accumulate(values.begin(), values.end(), 0.0) / values.size();
    long double variance = 0.0;
    for (const double value : values) variance += (value - mean) * (value - mean);
    return {
        quantile_sorted(sorted, 0.5), mean,
        quantile_sorted(sorted, 0.05), quantile_sorted(sorted, 0.95),
        std::sqrt(static_cast<double>(variance / values.size())),
    };
}

__global__ void checksum_kernel(const float * values, int count, double * output) {
    __shared__ double partials[256];
    double sum = 0.0;
    for (int index = blockIdx.x * blockDim.x + threadIdx.x;
         index < count; index += blockDim.x * gridDim.x) {
        sum += static_cast<double>(values[index]) * static_cast<double>((index % 251) + 1);
    }
    partials[threadIdx.x] = sum;
    __syncthreads();
    for (int offset = 128; offset; offset >>= 1) {
        if (threadIdx.x < offset) partials[threadIdx.x] += partials[threadIdx.x + offset];
        __syncthreads();
    }
    if (threadIdx.x == 0) atomicAdd(output, partials[0]);
}

template <typename Launch>
std::pair<double, double> time_one(
        Launch launch, const float * output, int rows, double * checksum_device,
        cudaEvent_t begin, cudaEvent_t end, cudaStream_t stream) {
    CUDA_CHECK(cudaEventRecord(begin, stream));
    launch();
    CUDA_CHECK(cudaEventRecord(end, stream));
    CUDA_CHECK(cudaMemsetAsync(checksum_device, 0, sizeof(double), stream));
    checksum_kernel<<<std::min(64, (rows + 255) / 256), 256, 0, stream>>>(
        output, rows, checksum_device);
    double checksum = 0.0;
    CUDA_CHECK(cudaMemcpyAsync(
        &checksum, checksum_device, sizeof(checksum), cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));
    float elapsed = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&elapsed, begin, end));
    if (!std::isfinite(checksum) || std::abs(checksum) <= 1e-20) {
        throw std::runtime_error("timed output checksum is invalid");
    }
    return {elapsed, checksum};
}

template <typename BaselineLaunch, typename CandidateLaunch>
PairResult paired_benchmark(
        BaselineLaunch baseline,
        CandidateLaunch candidate,
        const float * baseline_output,
        const float * candidate_output,
        int rows,
        int activation_count,
        int warmups,
        int pairs,
        double * checksum_device,
        cudaStream_t stream) {
    for (int index = 0; index < warmups; ++index) {
        const int vector = index % activation_count;
        if (index & 1) {
            candidate(vector);
            baseline(vector);
        } else {
            baseline(vector);
            candidate(vector);
        }
    }
    CUDA_CHECK(cudaStreamSynchronize(stream));
    cudaEvent_t begin, end;
    CUDA_CHECK(cudaEventCreate(&begin));
    CUDA_CHECK(cudaEventCreate(&end));
    PairResult result;
    for (int index = 0; index < pairs; ++index) {
        const int vector = index % activation_count;
        std::pair<double, double> baseline_result;
        std::pair<double, double> candidate_result;
        if (index & 1) {
            candidate_result = time_one(
                [&] { candidate(vector); }, candidate_output, rows, checksum_device, begin, end, stream);
            baseline_result = time_one(
                [&] { baseline(vector); }, baseline_output, rows, checksum_device, begin, end, stream);
            result.order.emplace_back("BA");
        } else {
            baseline_result = time_one(
                [&] { baseline(vector); }, baseline_output, rows, checksum_device, begin, end, stream);
            candidate_result = time_one(
                [&] { candidate(vector); }, candidate_output, rows, checksum_device, begin, end, stream);
            result.order.emplace_back("AB");
        }
        result.baseline_ms.push_back(baseline_result.first);
        result.candidate_ms.push_back(candidate_result.first);
        result.ratios.push_back(candidate_result.first / baseline_result.first);
        result.baseline_checksums.push_back(baseline_result.second);
        result.candidate_checksums.push_back(candidate_result.second);
    }
    CUDA_CHECK(cudaEventDestroy(end));
    CUDA_CHECK(cudaEventDestroy(begin));
    return result;
}

void write_metrics(std::ostream & output, const ErrorMetrics & metrics) {
    output << "{\"max_abs\":" << metrics.max_abs
           << ",\"mean_abs\":" << metrics.mean_abs
           << ",\"rmse\":" << metrics.rmse
           << ",\"cosine\":" << metrics.cosine
           << ",\"nonfinite\":" << metrics.nonfinite
           << ",\"values\":" << metrics.values << '}';
}

void write_summary(std::ostream & output, const Summary & value) {
    output << "{\"median_ms\":" << value.median
           << ",\"mean_ms\":" << value.mean
           << ",\"p5_ms\":" << value.p5
           << ",\"p95_ms\":" << value.p95
           << ",\"std_ms\":" << value.stddev << '}';
}

template <typename Value>
void write_values(std::ostream & output, const std::vector<Value> & values) {
    output << '[';
    for (std::size_t index = 0; index < values.size(); ++index) {
        if (index) output << ',';
        output << values[index];
    }
    output << ']';
}

void write_strings(std::ostream & output, const std::vector<std::string> & values) {
    output << '[';
    for (std::size_t index = 0; index < values.size(); ++index) {
        if (index) output << ',';
        output << '\"' << values[index] << '\"';
    }
    output << ']';
}

void write_pair(std::ostream & output, const PairResult & pair) {
    const Summary baseline = summarize(pair.baseline_ms);
    const Summary candidate = summarize(pair.candidate_ms);
    output << "{\"baseline\":";
    write_summary(output, baseline);
    output << ",\"candidate\":";
    write_summary(output, candidate);
    output << ",\"ratio_of_medians\":" << candidate.median / baseline.median
           << ",\"paired_ratio\":";
    write_summary(output, summarize(pair.ratios));
    output << ",\"raw\":{\"baseline_ms\":";
    write_values(output, pair.baseline_ms);
    output << ",\"candidate_ms\":";
    write_values(output, pair.candidate_ms);
    output << ",\"ratios\":";
    write_values(output, pair.ratios);
    output << ",\"order\":";
    write_strings(output, pair.order);
    output << ",\"baseline_checksums\":";
    write_values(output, pair.baseline_checksums);
    output << ",\"candidate_checksums\":";
    write_values(output, pair.candidate_checksums);
    output << "}}";
}

int run_ring(
        const Arguments & args,
        const std::vector<TensorDescriptor> & tensors) {
    std::array<const TensorDescriptor *, RING_TENSORS> ring{};
    int selected = 0;
    for (const auto & tensor : tensors) {
        if (tensor.kind == "mlp_down" && tensor.rows == 5120 && tensor.columns == 17408 &&
            selected < RING_TENSORS) {
            ring[selected++] = &tensor;
        }
    }
    if (selected != RING_TENSORS) {
        throw std::runtime_error("six same-geometry real FFN-down tensors are required");
    }
    const TensorDescriptor & geometry = *ring.front();
    for (const TensorDescriptor * tensor : ring) {
        if (tensor->q2_bytes != geometry.q2_bytes || tensor->tq1_bytes != geometry.tq1_bytes ||
            tensor->codes_bytes != geometry.codes_bytes ||
            tensor->scales_bytes != geometry.scales_bytes) {
            throw std::runtime_error("streaming-ring tensor extents differ");
        }
    }

    const auto collect_extents = [&](const std::string & path, auto offset, auto bytes) {
        const std::size_t stride = static_cast<std::size_t>(bytes(geometry));
        std::vector<std::uint8_t> result(static_cast<std::size_t>(RING_TENSORS) * stride);
        for (int index = 0; index < RING_TENSORS; ++index) {
            const auto extent = read_extent(
                path, static_cast<std::uint64_t>(offset(*ring[index])), stride);
            std::copy(extent.begin(), extent.end(), result.begin() + index * stride);
        }
        return result;
    };
    auto q2_host = collect_extents(
        args.q2_path,
        [](const TensorDescriptor & tensor) { return tensor.q2_file_offset; },
        [](const TensorDescriptor & tensor) { return tensor.q2_bytes; });
    auto tq1_host = collect_extents(
        args.tq1_path,
        [](const TensorDescriptor & tensor) { return tensor.tq1_file_offset; },
        [](const TensorDescriptor & tensor) { return tensor.tq1_bytes; });
    auto codes_host = collect_extents(
        args.reordered_path,
        [](const TensorDescriptor & tensor) { return tensor.codes_file_offset; },
        [](const TensorDescriptor & tensor) { return tensor.codes_bytes; });
    auto scales_host = collect_extents(
        args.reordered_path,
        [](const TensorDescriptor & tensor) { return tensor.scales_file_offset; },
        [](const TensorDescriptor & tensor) { return tensor.scales_bytes; });
    auto activations_host = make_activations(geometry.columns);
    activations_host.resize(static_cast<std::size_t>(RING_ACTIVATIONS) * geometry.columns);

    int device = 0;
    CUDA_CHECK(cudaGetDevice(&device));
    cudaDeviceProp properties{};
    CUDA_CHECK(cudaGetDeviceProperties(&properties, device));
    if (std::string(properties.name) != "NVIDIA RTX 6000 Ada Generation" ||
        properties.major != 8 || properties.minor != 9) {
        throw std::runtime_error("INVALID_EXPERIMENT: target is not RTX 6000 Ada sm_89");
    }

    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    std::uint8_t * q2_device = nullptr;
    std::uint8_t * tq1_device = nullptr;
    std::uint8_t * codes_device = nullptr;
    std::uint16_t * scales_device = nullptr;
    float * activations_device = nullptr;
    bonsai::block_q8_1_compat * q8_device = nullptr;
    float * q2_output_device = nullptr;
    float * v1_output_device = nullptr;
    float * new_output_device = nullptr;
    float * workspace_device = nullptr;
    unsigned int * counters_device = nullptr;
    double * checksum_device = nullptr;

    const std::size_t activation_values =
        static_cast<std::size_t>(RING_ACTIVATIONS) * geometry.columns;
    const std::size_t q8_blocks = activation_values / 32;
    const std::size_t output_values =
        static_cast<std::size_t>(RING_TENSORS) * geometry.rows;
    const std::size_t workspace_bytes = bonsai::lut23_workspace_bytes(
        geometry.rows, geometry.columns, args.k_chunk);
    const std::size_t counter_bytes = bonsai::lut23_counter_bytes(geometry.rows);
    CUDA_CHECK(cudaMalloc(&q2_device, q2_host.size()));
    CUDA_CHECK(cudaMalloc(&tq1_device, tq1_host.size()));
    CUDA_CHECK(cudaMalloc(&codes_device, codes_host.size()));
    CUDA_CHECK(cudaMalloc(&scales_device, scales_host.size()));
    CUDA_CHECK(cudaMalloc(&activations_device, activation_values * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&q8_device, q8_blocks * sizeof(bonsai::block_q8_1_compat)));
    CUDA_CHECK(cudaMalloc(&q2_output_device, output_values * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&v1_output_device, output_values * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&new_output_device, output_values * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&workspace_device, workspace_bytes));
    CUDA_CHECK(cudaMalloc(&counters_device, counter_bytes));
    CUDA_CHECK(cudaMalloc(&checksum_device, sizeof(double)));
    CUDA_CHECK(cudaMemsetAsync(counters_device, 0, counter_bytes, stream));
    CUDA_CHECK(cudaMemcpyAsync(
        q2_device, q2_host.data(), q2_host.size(), cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaMemcpyAsync(
        tq1_device, tq1_host.data(), tq1_host.size(), cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaMemcpyAsync(
        codes_device, codes_host.data(), codes_host.size(), cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaMemcpyAsync(
        scales_device, scales_host.data(), scales_host.size(), cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaMemcpyAsync(
        activations_device, activations_host.data(), activation_values * sizeof(float),
        cudaMemcpyHostToDevice, stream));
    bonsai::initialize_tq1_lut();
    bonsai_prism_quantize_q8_1(
        activations_device, q8_device, geometry.columns, RING_ACTIVATIONS, stream);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaStreamSynchronize(stream));

    const auto q8_for = [&](int vector) {
        return q8_device + static_cast<std::size_t>(vector) * geometry.columns / 32;
    };
    const auto launch_q2_ring = [&](int vector) {
        for (int index = 0; index < RING_TENSORS; ++index) {
            bonsai_prism_q2_0_mmvq(
                q2_device + static_cast<std::size_t>(index) * geometry.q2_bytes,
                q8_for((vector + index) % RING_ACTIVATIONS),
                q2_output_device + static_cast<std::size_t>(index) * geometry.rows,
                geometry.columns, geometry.rows, stream);
        }
    };
    const auto launch_v1_ring = [&](int vector) {
        for (int index = 0; index < RING_TENSORS; ++index) {
            bonsai::launch_tq1_g128_gemv_v1(
                reinterpret_cast<const bonsai::block_tq1_g128 *>(
                    tq1_device + static_cast<std::size_t>(index) * geometry.tq1_bytes),
                q8_for((vector + index) % RING_ACTIVATIONS),
                v1_output_device + static_cast<std::size_t>(index) * geometry.rows,
                geometry.rows, geometry.columns, stream);
        }
    };
    const auto launch_new_ring = [&](int vector) {
        for (int index = 0; index < RING_TENSORS; ++index) {
            bonsai::launch_tq1_lut23_smem_streamk(
                codes_device + static_cast<std::size_t>(index) * geometry.codes_bytes,
                scales_device + static_cast<std::size_t>(index) * geometry.scales_bytes / 2,
                q8_for((vector + index) % RING_ACTIVATIONS),
                new_output_device + static_cast<std::size_t>(index) * geometry.rows,
                workspace_device, counters_device, geometry.rows, geometry.columns,
                args.k_chunk, stream, nullptr);
        }
    };

    launch_q2_ring(0);
    launch_v1_ring(0);
    launch_new_ring(0);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaStreamSynchronize(stream));
    std::vector<float> q2_output(output_values), v1_output(output_values), new_output(output_values);
    CUDA_CHECK(cudaMemcpy(
        q2_output.data(), q2_output_device, output_values * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(
        v1_output.data(), v1_output_device, output_values * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(
        new_output.data(), new_output_device, output_values * sizeof(float), cudaMemcpyDeviceToHost));
    const ErrorMetrics v1_error = compare(v1_output, q2_output, geometry.rows);
    const ErrorMetrics new_error = compare(new_output, q2_output, geometry.rows);
    const bool correctness_pass = v1_error.nonfinite == 0 && new_error.nonfinite == 0;

    PairResult q2_candidate;
    PairResult v1_candidate;
    if (correctness_pass) {
        q2_candidate = paired_benchmark(
            launch_q2_ring, launch_new_ring, q2_output_device, new_output_device,
            static_cast<int>(output_values), RING_ACTIVATIONS, args.warmups, args.pairs,
            checksum_device, stream);
        v1_candidate = paired_benchmark(
            launch_v1_ring, launch_new_ring, v1_output_device, new_output_device,
            static_cast<int>(output_values), RING_ACTIVATIONS, args.warmups, args.pairs,
            checksum_device, stream);
    }

    std::ofstream json(args.output_path);
    if (!json) throw std::runtime_error("cannot create streaming-ring output");
    json << std::setprecision(12)
         << "{\n  \"schema_version\":1,\n  \"experiment\":\"TQ1_LUT23_SMEM_STREAMK\",\n"
         << "  \"mode\":\"ring\",\n  \"gpu\":{\"name\":\"" << properties.name
         << "\",\"compute_capability\":\"" << properties.major << '.' << properties.minor
         << "\",\"l2_bytes\":" << properties.l2CacheSize << "},\n"
         << "  \"k_chunk\":" << args.k_chunk << ",\n  \"tensors\":[";
    for (int index = 0; index < RING_TENSORS; ++index) {
        if (index) json << ',';
        json << "{\"name\":\"" << ring[index]->name << "\",\"layer\":"
             << ring[index]->layer << '}';
    }
    json << "],\n  \"geometry\":{\"m\":" << geometry.rows << ",\"k\":" << geometry.columns
         << ",\"traversal_launches\":" << RING_TENSORS
         << ",\"resident_q2_bytes\":" << q2_host.size()
         << ",\"resident_v1_bytes\":" << tq1_host.size()
         << ",\"resident_reordered_codes_bytes\":" << codes_host.size()
         << ",\"resident_reordered_scales_bytes\":" << scales_host.size() << "},\n"
         << "  \"correctness\":{\"v1_vs_q2\":";
    write_metrics(json, v1_error);
    json << ",\"new_vs_q2\":";
    write_metrics(json, new_error);
    json << ",\"pass\":" << (correctness_pass ? "true" : "false") << "},\n"
         << "  \"benchmark\":{\"warmups\":" << args.warmups
         << ",\"pairs\":" << args.pairs << ",\"alternating_order\":true},\n"
         << "  \"q2_vs_new\":";
    write_pair(json, q2_candidate);
    json << ",\n  \"v1_vs_new\":";
    write_pair(json, v1_candidate);
    json << ",\n  \"anti_cheating\":{\"unpacked_weight_buffer\":false,"
            "\"six_distinct_real_weight_extents\":true,\"same_q8_pointer_per_launch\":true,"
            "\"all_six_launches_inside_each_event\":true,\"output_dtype\":\"F32\","
            "\"lut_and_reduction_inside_timed_kernel\":true}\n}\n";
    json.close();

    CUDA_CHECK(cudaFree(checksum_device));
    CUDA_CHECK(cudaFree(counters_device));
    CUDA_CHECK(cudaFree(workspace_device));
    CUDA_CHECK(cudaFree(new_output_device));
    CUDA_CHECK(cudaFree(v1_output_device));
    CUDA_CHECK(cudaFree(q2_output_device));
    CUDA_CHECK(cudaFree(q8_device));
    CUDA_CHECK(cudaFree(activations_device));
    CUDA_CHECK(cudaFree(scales_device));
    CUDA_CHECK(cudaFree(codes_device));
    CUDA_CHECK(cudaFree(tq1_device));
    CUDA_CHECK(cudaFree(q2_device));
    CUDA_CHECK(cudaStreamDestroy(stream));
    std::cout << "ring_correctness=" << (correctness_pass ? "PASS" : "FAIL")
              << " tensors=" << RING_TENSORS << " k_chunk=" << args.k_chunk << '\n';
    return correctness_pass ? 0 : 3;
}

}  // namespace

int main(int argc, char ** argv) try {
    const Arguments args = parse_arguments(argc, argv);
    const auto tensors = read_manifest(args.manifest_path);
    if (args.mode == "ring") return run_ring(args, tensors);
    const auto found = std::find_if(tensors.begin(), tensors.end(), [](const auto & tensor) {
        return tensor.name == "blk.0.ffn_down.weight";
    });
    if (found == tensors.end() || found->rows != 5120 || found->columns != 17408) {
        throw std::runtime_error("primary tensor missing or has unexpected geometry");
    }
    const TensorDescriptor & tensor = *found;
    auto q2_host = read_extent(args.q2_path, tensor.q2_file_offset, tensor.q2_bytes);
    auto tq1_host = read_extent(args.tq1_path, tensor.tq1_file_offset, tensor.tq1_bytes);
    auto codes_host = read_extent(args.reordered_path, tensor.codes_file_offset, tensor.codes_bytes);
    auto scales_host = read_extent(args.reordered_path, tensor.scales_file_offset, tensor.scales_bytes);
    auto activations_host = make_activations(tensor.columns);

    int device = 0;
    CUDA_CHECK(cudaGetDevice(&device));
    cudaDeviceProp properties{};
    CUDA_CHECK(cudaGetDeviceProperties(&properties, device));
    if (std::string(properties.name) != "NVIDIA RTX 6000 Ada Generation" ||
        properties.major != 8 || properties.minor != 9) {
        throw std::runtime_error("INVALID_EXPERIMENT: target is not RTX 6000 Ada sm_89");
    }

    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    void * q2_device = nullptr;
    bonsai::block_tq1_g128 * tq1_device = nullptr;
    std::uint8_t * codes_device = nullptr;
    std::uint16_t * scales_device = nullptr;
    float * activations_device = nullptr;
    bonsai::block_q8_1_compat * q8_device = nullptr;
    float * q2_output_device = nullptr;
    float * v1_output_device = nullptr;
    float * new_output_device = nullptr;
    float * workspace_device = nullptr;
    unsigned int * counters_device = nullptr;
    double * checksum_device = nullptr;
    bonsai::lut23_phase_counters * phase_device = nullptr;

    const std::size_t activation_values =
        static_cast<std::size_t>(CORRECTNESS_VECTORS) * tensor.columns;
    const std::size_t q8_blocks = activation_values / 32;
    const std::size_t output_values =
        static_cast<std::size_t>(CORRECTNESS_VECTORS) * tensor.rows;
    const std::size_t workspace_bytes = bonsai::lut23_workspace_bytes(
        tensor.rows, tensor.columns, 512);
    const std::size_t counter_bytes = bonsai::lut23_counter_bytes(tensor.rows);
    CUDA_CHECK(cudaMalloc(&q2_device, tensor.q2_bytes));
    CUDA_CHECK(cudaMalloc(&tq1_device, tensor.tq1_bytes));
    CUDA_CHECK(cudaMalloc(&codes_device, tensor.codes_bytes));
    CUDA_CHECK(cudaMalloc(&scales_device, tensor.scales_bytes));
    CUDA_CHECK(cudaMalloc(&activations_device, activation_values * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&q8_device, q8_blocks * sizeof(bonsai::block_q8_1_compat)));
    CUDA_CHECK(cudaMalloc(&q2_output_device, output_values * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&v1_output_device, output_values * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&new_output_device, output_values * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&workspace_device, workspace_bytes));
    CUDA_CHECK(cudaMalloc(&counters_device, counter_bytes));
    CUDA_CHECK(cudaMalloc(&checksum_device, sizeof(double)));
    CUDA_CHECK(cudaMalloc(&phase_device, sizeof(bonsai::lut23_phase_counters)));
    CUDA_CHECK(cudaMemsetAsync(counters_device, 0, counter_bytes, stream));
    CUDA_CHECK(cudaMemsetAsync(phase_device, 0, sizeof(bonsai::lut23_phase_counters), stream));
    CUDA_CHECK(cudaMemcpyAsync(q2_device, q2_host.data(), tensor.q2_bytes, cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaMemcpyAsync(tq1_device, tq1_host.data(), tensor.tq1_bytes, cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaMemcpyAsync(codes_device, codes_host.data(), tensor.codes_bytes, cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaMemcpyAsync(scales_device, scales_host.data(), tensor.scales_bytes, cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaMemcpyAsync(
        activations_device, activations_host.data(), activation_values * sizeof(float),
        cudaMemcpyHostToDevice, stream));
    bonsai::initialize_tq1_lut();
    bonsai_prism_quantize_q8_1(
        activations_device, q8_device, tensor.columns, CORRECTNESS_VECTORS, stream);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaStreamSynchronize(stream));

    const auto q8_for = [&](int vector) {
        return q8_device + static_cast<std::size_t>(vector) * tensor.columns / 32;
    };
    const auto launch_q2 = [&](int vector, float * output) {
        bonsai_prism_q2_0_mmvq(
            q2_device, q8_for(vector), output, tensor.columns, tensor.rows, stream);
    };
    const auto launch_v1 = [&](int vector, float * output) {
        bonsai::launch_tq1_g128_gemv_v1(
            tq1_device, q8_for(vector), output, tensor.rows, tensor.columns, stream);
    };
    const auto launch_new = [&](int vector, float * output, int k_chunk,
                                bonsai::lut23_phase_counters * phase = nullptr) {
        bonsai::launch_tq1_lut23_smem_streamk(
            codes_device, scales_device, q8_for(vector), output,
            workspace_device, counters_device, tensor.rows, tensor.columns,
            k_chunk, stream, phase);
    };

    if (args.mode == "profile") {
        launch_new(0, new_output_device, args.k_chunk, phase_device);
        CUDA_CHECK(cudaGetLastError());
        bonsai::lut23_phase_counters phases{};
        CUDA_CHECK(cudaMemcpyAsync(
            &phases, phase_device, sizeof(phases), cudaMemcpyDeviceToHost, stream));
        CUDA_CHECK(cudaStreamSynchronize(stream));
        const unsigned long long total = phases.lut_build_cycles + phases.code_consume_cycles +
            phases.reduction_cycles;
        std::ofstream json(args.output_path);
        if (!json) throw std::runtime_error("cannot create profile phase output");
        json << std::setprecision(12)
             << "{\n  \"schema_version\":1,\n  \"mode\":\"profile\",\n"
             << "  \"k_chunk\":" << args.k_chunk << ",\n"
             << "  \"phase_cycles\":{\"lut_build\":" << phases.lut_build_cycles
             << ",\"code_consume\":" << phases.code_consume_cycles
             << ",\"streamk_reduction\":" << phases.reduction_cycles
             << ",\"total\":" << total << "},\n"
             << "  \"phase_fractions\":{\"lut_build\":"
             << (total ? static_cast<double>(phases.lut_build_cycles) / total : 0.0)
             << ",\"code_consume\":"
             << (total ? static_cast<double>(phases.code_consume_cycles) / total : 0.0)
             << ",\"streamk_reduction\":"
             << (total ? static_cast<double>(phases.reduction_cycles) / total : 0.0)
             << "}\n}\n";
        json.close();
        std::cout << "profile_k_chunk=" << args.k_chunk << " phase_cycles=" << total << '\n';
    } else {
        for (int vector = 0; vector < CORRECTNESS_VECTORS; ++vector) {
            float * q2_output = q2_output_device + static_cast<std::size_t>(vector) * tensor.rows;
            float * v1_output = v1_output_device + static_cast<std::size_t>(vector) * tensor.rows;
            launch_q2(vector, q2_output);
            launch_v1(vector, v1_output);
            for (const int k_chunk : {512, 1024, 2048}) {
                float * new_output = new_output_device + static_cast<std::size_t>(vector) * tensor.rows;
                launch_new(vector, new_output, k_chunk);
                CUDA_CHECK(cudaGetLastError());
                CUDA_CHECK(cudaStreamSynchronize(stream));
            }
        }
        std::vector<float> q2_output(output_values), v1_output(output_values);
        std::vector<bonsai::block_q8_1_compat> q8_host(q8_blocks);
        CUDA_CHECK(cudaMemcpy(
            q2_output.data(), q2_output_device, output_values * sizeof(float), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(
            v1_output.data(), v1_output_device, output_values * sizeof(float), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(
            q8_host.data(), q8_device, q8_blocks * sizeof(bonsai::block_q8_1_compat),
            cudaMemcpyDeviceToHost));
        const ErrorMetrics v1_error = compare(v1_output, q2_output, tensor.rows);
        std::vector<float> reference(static_cast<std::size_t>(CORRECTNESS_VECTORS) * REFERENCE_ROWS);
        std::vector<float> v1_reference_subset(reference.size());
#pragma omp parallel for schedule(static)
        for (int task = 0; task < CORRECTNESS_VECTORS * REFERENCE_ROWS; ++task) {
            const int vector = task / REFERENCE_ROWS;
            const int sample = task % REFERENCE_ROWS;
            const int row = sample * tensor.rows / REFERENCE_ROWS;
            reference[task] = reference_value(
                tq1_host,
                q8_host.data() + static_cast<std::size_t>(vector) * tensor.columns / 32,
                row,
                tensor.columns);
            v1_reference_subset[task] =
                v1_output[static_cast<std::size_t>(vector) * tensor.rows + row];
        }
        const ErrorMetrics v1_reference_error = compare(
            v1_reference_subset, reference, REFERENCE_ROWS);
        std::array<ErrorMetrics, 3> new_errors{};
        std::array<ErrorMetrics, 3> new_reference_errors{};
        for (int choice = 0; choice < 3; ++choice) {
            const int k_chunk = std::array<int, 3>{512, 1024, 2048}[choice];
            for (int vector = 0; vector < CORRECTNESS_VECTORS; ++vector) {
                float * output = new_output_device + static_cast<std::size_t>(vector) * tensor.rows;
                launch_new(vector, output, k_chunk);
            }
            CUDA_CHECK(cudaStreamSynchronize(stream));
            std::vector<float> new_output(output_values);
            CUDA_CHECK(cudaMemcpy(
                new_output.data(), new_output_device, output_values * sizeof(float), cudaMemcpyDeviceToHost));
            new_errors[choice] = compare(new_output, q2_output, tensor.rows);
            std::vector<float> subset(reference.size());
            for (int vector = 0; vector < CORRECTNESS_VECTORS; ++vector) {
                for (int sample = 0; sample < REFERENCE_ROWS; ++sample) {
                    const int row = sample * tensor.rows / REFERENCE_ROWS;
                    subset[static_cast<std::size_t>(vector) * REFERENCE_ROWS + sample] =
                        new_output[static_cast<std::size_t>(vector) * tensor.rows + row];
                }
            }
            new_reference_errors[choice] = compare(subset, reference, REFERENCE_ROWS);
        }
        bool correctness_pass = v1_error.nonfinite == 0 && v1_reference_error.nonfinite == 0;
        for (int choice = 0; choice < 3; ++choice) {
            const auto & error = new_reference_errors[choice];
            correctness_pass = correctness_pass && error.nonfinite == 0 &&
                new_errors[choice].nonfinite == 0 && error.max_abs <= MAX_ABS_LIMIT;
        }

        struct ChunkTiming {
            int k_chunk;
            PairResult q2;
            PairResult v1;
        };
        std::vector<ChunkTiming> timings;
        if (correctness_pass) {
            for (const int k_chunk : {512, 1024, 2048}) {
                const auto q2_candidate = paired_benchmark(
                    [&](int vector) { launch_q2(vector, q2_output_device); },
                    [&](int vector) { launch_new(vector, new_output_device, k_chunk); },
                    q2_output_device, new_output_device, tensor.rows, 2,
                    args.warmups, args.pairs, checksum_device, stream);
                const auto v1_candidate = paired_benchmark(
                    [&](int vector) { launch_v1(vector, v1_output_device); },
                    [&](int vector) { launch_new(vector, new_output_device, k_chunk); },
                    v1_output_device, new_output_device, tensor.rows, 2,
                    args.warmups, args.pairs, checksum_device, stream);
                timings.push_back({k_chunk, q2_candidate, v1_candidate});
            }
        }

        std::ofstream json(args.output_path);
        if (!json) throw std::runtime_error("cannot create qualification output");
        json << std::setprecision(12)
             << "{\n  \"schema_version\":1,\n  \"experiment\":\"TQ1_LUT23_SMEM_STREAMK\",\n"
             << "  \"mode\":\"qualify\",\n"
             << "  \"gpu\":{\"name\":\"" << properties.name
             << "\",\"compute_capability\":\"" << properties.major << '.' << properties.minor
             << "\",\"l2_bytes\":" << properties.l2CacheSize << "},\n"
             << "  \"matrix\":{\"name\":\"" << tensor.name << "\",\"m\":" << tensor.rows
             << ",\"k\":" << tensor.columns << ",\"q2_bytes\":" << tensor.q2_bytes
             << ",\"v1_tq1_bytes\":" << tensor.tq1_bytes
             << ",\"reordered_codes_bytes\":" << tensor.codes_bytes
             << ",\"reordered_scales_bytes\":" << tensor.scales_bytes << "},\n"
             << "  \"correctness\":{\"random_vectors\":" << RANDOM_VECTORS
             << ",\"edge_vectors\":" << EDGE_VECTORS << ",\"max_abs_limit\":" << MAX_ABS_LIMIT
             << ",\"reference_rows\":" << REFERENCE_ROWS << ",\"gate_reference\":"
                "\"independent FP64 CPU accumulation rounded once to F32\",\"v1_vs_q2\":";
        write_metrics(json, v1_error);
        json << ",\"v1_vs_reference\":";
        write_metrics(json, v1_reference_error);
        json << ",\"chunks\":[";
        for (int choice = 0; choice < 3; ++choice) {
            if (choice) json << ',';
            json << "{\"k_chunk\":" << std::array<int, 3>{512, 1024, 2048}[choice]
                 << ",\"new_vs_q2\":";
            write_metrics(json, new_errors[choice]);
            json << ",\"new_vs_reference\":";
            write_metrics(json, new_reference_errors[choice]);
            json << '}';
        }
        json << "],\"pass\":" << (correctness_pass ? "true" : "false") << "},\n"
             << "  \"warm_benchmark\":{\"warmups\":" << args.warmups
             << ",\"pairs\":" << args.pairs << ",\"chunks\":[";
        for (std::size_t index = 0; index < timings.size(); ++index) {
            if (index) json << ',';
            json << "{\"k_chunk\":" << timings[index].k_chunk << ",\"q2_vs_new\":";
            write_pair(json, timings[index].q2);
            json << ",\"v1_vs_new\":";
            write_pair(json, timings[index].v1);
            json << '}';
        }
        json << "]},\n"
             << "  \"anti_cheating\":{\"unpacked_weight_buffer\":false,"
                "\"new_weight_allocations_equal_reordered_packed_extents\":true,"
                "\"same_q8_pointer\":true,\"output_dtype\":\"F32\","
                "\"lut_and_reduction_inside_timed_kernel\":true,"
                "\"separate_lut_kernel\":false,\"separate_reduction_kernel\":false}\n}\n";
        json.close();
        std::cout << "correctness=" << (correctness_pass ? "PASS" : "FAIL")
                  << " tested_k_chunks=" << timings.size() << '\n';
        if (!correctness_pass) return 3;
    }

    CUDA_CHECK(cudaFree(phase_device));
    CUDA_CHECK(cudaFree(checksum_device));
    CUDA_CHECK(cudaFree(counters_device));
    CUDA_CHECK(cudaFree(workspace_device));
    CUDA_CHECK(cudaFree(new_output_device));
    CUDA_CHECK(cudaFree(v1_output_device));
    CUDA_CHECK(cudaFree(q2_output_device));
    CUDA_CHECK(cudaFree(q8_device));
    CUDA_CHECK(cudaFree(activations_device));
    CUDA_CHECK(cudaFree(scales_device));
    CUDA_CHECK(cudaFree(codes_device));
    CUDA_CHECK(cudaFree(tq1_device));
    CUDA_CHECK(cudaFree(q2_device));
    CUDA_CHECK(cudaStreamDestroy(stream));
    return 0;
} catch (const std::exception & error) {
    std::cerr << error.what() << '\n';
    return 2;
}
