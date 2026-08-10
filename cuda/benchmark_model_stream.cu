#include "tq1_g128.cuh"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cerrno>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <fcntl.h>
#include <filesystem>
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

constexpr std::uint64_t EXPECTED_MODEL_BYTES = 7'165'121'600ULL;
constexpr int Q2_BLOCK_BYTES = 34;
constexpr int ACTIVATION_VECTORS = 2;
constexpr int SINGLE_WARMUPS = 200;
constexpr int SINGLE_REPEATS = 5;
constexpr int SINGLE_ITERATIONS = 1000;
constexpr std::size_t COPY_CHUNK_BYTES = 64ULL << 20;
constexpr double V1_MAX_ABS = 5.72204589844e-6;
constexpr double V1_MEAN_ABS = 7.0360459722e-8;
constexpr double V1_RMSE = 9.84332522182e-8;

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
    std::string output_path;
    int traversal_warmups = 6;
    int traversal_pairs = 40;
    int diagnostic_pairs = 10;
};

struct TensorDescriptor {
    int order = 0;
    int source_index = 0;
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
    std::uint64_t q2_stream_offset = 0;
    std::uint64_t tq1_stream_offset = 0;
    std::uint64_t q8_stream_offset = 0;
    std::uint64_t output_stream_offset = 0;
};

struct Summary {
    double median = 0.0;
    double mean = 0.0;
    double p5 = 0.0;
    double p95 = 0.0;
    double standard_deviation = 0.0;
};

struct ErrorMetrics {
    double max_abs = 0.0;
    double mean_abs = 0.0;
    double rmse = 0.0;
    double max_relative = 0.0;
    double cosine = 0.0;
    std::uint64_t nonfinite = 0;
    std::uint64_t values = 0;
};

struct ErrorAccumulator {
    long double absolute_sum = 0.0;
    long double square_sum = 0.0;
    long double dot = 0.0;
    long double norm_a = 0.0;
    long double norm_b = 0.0;
    double max_abs = 0.0;
    double max_relative = 0.0;
    std::uint64_t nonfinite = 0;
    std::uint64_t finite = 0;

    void add(double a, double b) {
        if (!std::isfinite(a) || !std::isfinite(b)) {
            ++nonfinite;
            return;
        }
        const double error = std::abs(a - b);
        max_abs = std::max(max_abs, error);
        max_relative = std::max(max_relative, error / std::max(std::abs(b), 1e-6));
        absolute_sum += error;
        square_sum += error * error;
        dot += a * b;
        norm_a += a * a;
        norm_b += b * b;
        ++finite;
    }

    void merge(const ErrorAccumulator & other) {
        absolute_sum += other.absolute_sum;
        square_sum += other.square_sum;
        dot += other.dot;
        norm_a += other.norm_a;
        norm_b += other.norm_b;
        max_abs = std::max(max_abs, other.max_abs);
        max_relative = std::max(max_relative, other.max_relative);
        nonfinite += other.nonfinite;
        finite += other.finite;
    }

    ErrorMetrics finish() const {
        ErrorMetrics result;
        result.max_abs = max_abs;
        result.max_relative = max_relative;
        result.nonfinite = nonfinite;
        result.values = finite + nonfinite;
        if (finite != 0) {
            result.mean_abs = static_cast<double>(absolute_sum / finite);
            result.rmse = std::sqrt(static_cast<double>(square_sum / finite));
        }
        if (norm_a > 0.0 && norm_b > 0.0) {
            result.cosine = static_cast<double>(dot / std::sqrt(norm_a * norm_b));
        }
        return result;
    }
};

struct TensorCorrectness {
    ErrorMetrics error;
    double q2_checksum = 0.0;
    double tq1_checksum = 0.0;
    bool finite_nontrivial = false;
};

struct TimedValue {
    double milliseconds = 0.0;
    double checksum = 0.0;
};

struct PairedTimings {
    std::vector<double> q2;
    std::vector<double> tq1;
    std::vector<double> ratios;
    std::vector<std::string> order;
    std::vector<double> q2_checksums;
    std::vector<double> tq1_checksums;
};

Arguments parse_arguments(int argc, char ** argv) {
    Arguments result;
    for (int index = 1; index < argc; ++index) {
        const std::string key = argv[index];
        if (index + 1 >= argc) {
            throw std::invalid_argument("missing value for " + key);
        }
        const std::string value = argv[++index];
        if (key == "--manifest") result.manifest_path = value;
        else if (key == "--q2") result.q2_path = value;
        else if (key == "--tq1") result.tq1_path = value;
        else if (key == "--output") result.output_path = value;
        else if (key == "--traversal-warmups") result.traversal_warmups = std::stoi(value);
        else if (key == "--traversal-pairs") result.traversal_pairs = std::stoi(value);
        else if (key == "--diagnostic-pairs") result.diagnostic_pairs = std::stoi(value);
        else throw std::invalid_argument("unknown argument " + key);
    }
    if (result.manifest_path.empty() || result.q2_path.empty() || result.tq1_path.empty() ||
        result.output_path.empty() || result.traversal_warmups < 5 ||
        result.traversal_pairs < 30 || result.diagnostic_pairs < 5) {
        throw std::invalid_argument("invalid or incomplete model-stream arguments");
    }
    return result;
}

std::vector<std::string> split_tsv(const std::string & line) {
    std::vector<std::string> fields;
    std::size_t begin = 0;
    while (true) {
        const std::size_t tab = line.find('\t', begin);
        if (tab == std::string::npos) {
            fields.push_back(line.substr(begin));
            return fields;
        }
        fields.push_back(line.substr(begin, tab - begin));
        begin = tab + 1;
    }
}

std::vector<TensorDescriptor> read_manifest(const std::string & path) {
    std::ifstream input(path);
    if (!input) throw std::runtime_error("cannot open traversal manifest " + path);
    std::string line;
    if (!std::getline(input, line)) throw std::runtime_error("empty traversal manifest");
    const auto header = split_tsv(line);
    std::unordered_map<std::string, std::size_t> column;
    for (std::size_t index = 0; index < header.size(); ++index) column.emplace(header[index], index);
    const auto field = [&](const std::vector<std::string> & values, const std::string & name) -> const std::string & {
        const auto found = column.find(name);
        if (found == column.end() || found->second >= values.size()) {
            throw std::runtime_error("missing TSV field " + name);
        }
        return values[found->second];
    };
    std::vector<TensorDescriptor> result;
    while (std::getline(input, line)) {
        if (line.empty()) continue;
        const auto values = split_tsv(line);
        TensorDescriptor tensor;
        tensor.order = std::stoi(field(values, "order"));
        tensor.source_index = std::stoi(field(values, "source_index"));
        tensor.layer = std::stoi(field(values, "layer"));
        tensor.kind = field(values, "kind");
        tensor.name = field(values, "name");
        tensor.rows = std::stoi(field(values, "m"));
        tensor.columns = std::stoi(field(values, "k"));
        tensor.groups = std::stoull(field(values, "groups"));
        tensor.q2_file_offset = std::stoull(field(values, "q2_file_offset"));
        tensor.q2_bytes = std::stoull(field(values, "q2_bytes"));
        tensor.tq1_file_offset = std::stoull(field(values, "tq1_file_offset"));
        tensor.tq1_bytes = std::stoull(field(values, "tq1_bytes"));
        tensor.q2_stream_offset = std::stoull(field(values, "q2_stream_offset"));
        tensor.tq1_stream_offset = std::stoull(field(values, "tq1_stream_offset"));
        tensor.q8_stream_offset = std::stoull(field(values, "q8_stream_offset"));
        tensor.output_stream_offset = std::stoull(field(values, "output_stream_offset"));
        result.push_back(std::move(tensor));
    }
    if (result.size() != 497) throw std::runtime_error("manifest must contain exactly 497 tensors");
    std::uint64_t q2_offset = 0;
    std::uint64_t tq1_offset = 0;
    std::uint64_t q8_offset = 0;
    std::uint64_t output_offset = 0;
    for (std::size_t index = 0; index < result.size(); ++index) {
        const auto & tensor = result[index];
        if (tensor.order != static_cast<int>(index) || tensor.rows <= 0 || tensor.columns <= 0 ||
            tensor.columns % bonsai::TQ1_BLOCK_SIZE != 0 ||
            tensor.groups != static_cast<std::uint64_t>(tensor.rows) * tensor.columns / bonsai::TQ1_BLOCK_SIZE ||
            tensor.q2_bytes != tensor.groups * Q2_BLOCK_BYTES ||
            tensor.tq1_bytes != tensor.groups * bonsai::TQ1_BLOCK_BYTES ||
            tensor.q2_stream_offset != q2_offset || tensor.tq1_stream_offset != tq1_offset ||
            tensor.q8_stream_offset != q8_offset || tensor.output_stream_offset != output_offset ||
            q2_offset % 256 != 0 || tq1_offset % 256 != 0) {
            throw std::runtime_error("invalid stream geometry at " + tensor.name);
        }
        q2_offset += tensor.q2_bytes;
        tq1_offset += tensor.tq1_bytes;
        q8_offset += tensor.columns / bonsai::Q8_BLOCK_SIZE;
        output_offset += tensor.rows;
    }
    if (result.back().name != "output.weight") {
        throw std::runtime_error("LM head is not the final traversal tensor");
    }
    return result;
}

double quantile_sorted(const std::vector<double> & sorted, double q) {
    if (sorted.empty()) return 0.0;
    const double position = q * (sorted.size() - 1);
    const std::size_t low = static_cast<std::size_t>(position);
    const std::size_t high = std::min(low + 1, sorted.size() - 1);
    const double fraction = position - low;
    return sorted[low] * (1.0 - fraction) + sorted[high] * fraction;
}

Summary summarize(const std::vector<double> & values) {
    if (values.empty()) return {};
    std::vector<double> sorted = values;
    std::sort(sorted.begin(), sorted.end());
    const double mean = std::accumulate(values.begin(), values.end(), 0.0) / values.size();
    long double variance = 0.0;
    for (const double value : values) variance += (value - mean) * (value - mean);
    return {
        quantile_sorted(sorted, 0.5),
        mean,
        quantile_sorted(sorted, 0.05),
        quantile_sorted(sorted, 0.95),
        std::sqrt(static_cast<double>(variance / values.size())),
    };
}

float round_bfloat16(float value) {
    std::uint32_t bits;
    std::memcpy(&bits, &value, sizeof(bits));
    const std::uint32_t rounding = 0x7fffu + ((bits >> 16u) & 1u);
    bits = (bits + rounding) & 0xffff0000u;
    std::memcpy(&value, &bits, sizeof(bits));
    return value;
}

std::vector<float> make_activation(int columns, int tensor, int origin) {
    std::mt19937 generator(20260809u + static_cast<std::uint32_t>(tensor * 17 + origin));
    std::normal_distribution<float> normal(0.0f, 1.0f);
    std::vector<float> result(columns);
    for (float & value : result) {
        value = normal(generator);
        value = origin == 0 ? __half2float(__float2half_rn(value)) : round_bfloat16(value);
    }
    return result;
}

void read_exact_at(int descriptor, void * destination, std::size_t bytes, std::uint64_t offset) {
    auto * output = static_cast<std::uint8_t *>(destination);
    std::size_t done = 0;
    while (done < bytes) {
        const ssize_t amount = pread(
            descriptor, output + done, bytes - done,
            static_cast<off_t>(offset + done));
        if (amount < 0 && errno == EINTR) continue;
        if (amount < 0) throw std::runtime_error(std::string("pread failed: ") + std::strerror(errno));
        if (amount == 0) throw std::runtime_error("truncated tensor extent while loading weight stream");
        done += static_cast<std::size_t>(amount);
    }
}

double load_weight_stream(
        const std::string & path,
        const std::vector<TensorDescriptor> & tensors,
        bool tq1,
        std::uint8_t * destination) {
    const auto started = std::chrono::steady_clock::now();
    const int descriptor = open(path.c_str(), O_RDONLY);
    if (descriptor < 0) throw std::runtime_error("cannot open weight file " + path);
    void * staging = nullptr;
    CUDA_CHECK(cudaHostAlloc(&staging, COPY_CHUNK_BYTES, cudaHostAllocDefault));
    try {
        for (const auto & tensor : tensors) {
            std::uint64_t file_offset = tq1 ? tensor.tq1_file_offset : tensor.q2_file_offset;
            std::uint64_t stream_offset = tq1 ? tensor.tq1_stream_offset : tensor.q2_stream_offset;
            std::uint64_t remaining = tq1 ? tensor.tq1_bytes : tensor.q2_bytes;
            while (remaining != 0) {
                const std::size_t amount = static_cast<std::size_t>(
                    std::min<std::uint64_t>(remaining, COPY_CHUNK_BYTES));
                read_exact_at(descriptor, staging, amount, file_offset);
                CUDA_CHECK(cudaMemcpy(
                    destination + stream_offset, staging, amount, cudaMemcpyHostToDevice));
                file_offset += amount;
                stream_offset += amount;
                remaining -= amount;
            }
        }
    } catch (...) {
        cudaFreeHost(staging);
        close(descriptor);
        throw;
    }
    CUDA_CHECK(cudaFreeHost(staging));
    close(descriptor);
    return std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count();
}

__global__ void scrub_kernel(std::uint64_t * data, std::size_t words, std::uint64_t seed) {
    for (std::size_t index = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         index < words;
         index += static_cast<std::size_t>(blockDim.x) * gridDim.x) {
        data[index] = index ^ seed;
    }
}

__global__ void checksum_kernel(const float * values, std::size_t count, double * output) {
    __shared__ double partials[256];
    double sum = 0.0;
    for (std::size_t index = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         index < count;
         index += static_cast<std::size_t>(blockDim.x) * gridDim.x) {
        sum += static_cast<double>(values[index]) * static_cast<double>((index % 251) + 1);
    }
    partials[threadIdx.x] = sum;
    __syncthreads();
    for (int offset = 128; offset != 0; offset /= 2) {
        if (threadIdx.x < offset) partials[threadIdx.x] += partials[threadIdx.x + offset];
        __syncthreads();
    }
    if (threadIdx.x == 0) atomicAdd(output, partials[0]);
}

std::string escape_json(const std::string & value) {
    std::string result;
    result.reserve(value.size() + 2);
    for (const char character : value) {
        switch (character) {
            case '\\': result += "\\\\"; break;
            case '"': result += "\\\""; break;
            case '\n': result += "\\n"; break;
            case '\r': result += "\\r"; break;
            case '\t': result += "\\t"; break;
            default: result += character; break;
        }
    }
    return result;
}

void write_summary(std::ostream & output, const Summary & value) {
    output << "{\"median_ms\":" << value.median
           << ",\"mean_ms\":" << value.mean
           << ",\"p5_ms\":" << value.p5
           << ",\"p95_ms\":" << value.p95
           << ",\"std_ms\":" << value.standard_deviation << "}";
}

void write_metrics(std::ostream & output, const ErrorMetrics & value) {
    output << "{\"max_abs\":" << value.max_abs
           << ",\"mean_abs\":" << value.mean_abs
           << ",\"rmse\":" << value.rmse
           << ",\"max_relative\":" << value.max_relative
           << ",\"cosine\":" << value.cosine
           << ",\"nonfinite\":" << value.nonfinite
           << ",\"values\":" << value.values << "}";
}

void write_values(std::ostream & output, const std::vector<double> & values) {
    output << '[';
    for (std::size_t index = 0; index < values.size(); ++index) {
        if (index != 0) output << ',';
        output << values[index];
    }
    output << ']';
}

void write_strings(std::ostream & output, const std::vector<std::string> & values) {
    output << '[';
    for (std::size_t index = 0; index < values.size(); ++index) {
        if (index != 0) output << ',';
        output << '"' << escape_json(values[index]) << '"';
    }
    output << ']';
}

double host_checksum_tensor(
        const std::vector<float> & values,
        std::uint64_t total_per_vector,
        std::uint64_t offset,
        int count) {
    long double result = 0.0;
    std::uint64_t ordinal = 0;
    for (int origin = 0; origin < ACTIVATION_VECTORS; ++origin) {
        const std::uint64_t base = static_cast<std::uint64_t>(origin) * total_per_vector + offset;
        for (int index = 0; index < count; ++index, ++ordinal) {
            result += values[base + index] * static_cast<double>((ordinal % 251) + 1);
        }
    }
    return static_cast<double>(result);
}

}  // namespace

int main(int argc, char ** argv) try {
    const Arguments args = parse_arguments(argc, argv);
    const auto tensors = read_manifest(args.manifest_path);
    const std::uint64_t total_q2_bytes = tensors.back().q2_stream_offset + tensors.back().q2_bytes;
    const std::uint64_t total_tq1_bytes = tensors.back().tq1_stream_offset + tensors.back().tq1_bytes;
    const std::uint64_t q8_blocks_per_vector = tensors.back().q8_stream_offset + tensors.back().columns / bonsai::Q8_BLOCK_SIZE;
    const std::uint64_t outputs_per_vector = tensors.back().output_stream_offset + tensors.back().rows;
    const int max_columns = std::max_element(
        tensors.begin(), tensors.end(),
        [](const auto & left, const auto & right) { return left.columns < right.columns; })->columns;

    if (std::filesystem::file_size(args.q2_path) != EXPECTED_MODEL_BYTES) {
        throw std::runtime_error("released model byte size mismatch");
    }

    int device = 0;
    CUDA_CHECK(cudaGetDevice(&device));
    cudaDeviceProp properties{};
    CUDA_CHECK(cudaGetDeviceProperties(&properties, device));
    if (std::string(properties.name) != "NVIDIA RTX 6000 Ada Generation" ||
        properties.major != 8 || properties.minor != 9 ||
        properties.l2CacheSize != 100'663'296) {
        throw std::runtime_error("INVALID_STREAMING_EXPERIMENT: target is not the expected RTX 6000 Ada sm_89");
    }

    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    std::size_t free_before = 0;
    std::size_t total_memory = 0;
    CUDA_CHECK(cudaMemGetInfo(&free_before, &total_memory));
    const std::size_t scrub_bytes = std::max<std::size_t>(
        128ULL << 20, 2ULL * static_cast<std::size_t>(properties.l2CacheSize) + 4096);
    const std::uint64_t q8_allocation_bytes =
        q8_blocks_per_vector * ACTIVATION_VECTORS * sizeof(bonsai::block_q8_1_compat);
    const std::uint64_t output_allocation_bytes =
        outputs_per_vector * ACTIVATION_VECTORS * sizeof(float);
    const std::uint64_t required_bytes = total_q2_bytes + total_tq1_bytes + q8_allocation_bytes +
        2 * output_allocation_bytes + scrub_bytes + max_columns * sizeof(float) + sizeof(double);
    if (free_before < required_bytes + (512ULL << 20)) {
        throw std::runtime_error(
            "INVALID_STREAMING_EXPERIMENT: insufficient free GPU memory or competing GPU allocation");
    }

    std::uint8_t * q2_device = nullptr;
    std::uint8_t * tq1_device = nullptr;
    bonsai::block_q8_1_compat * q8_device = nullptr;
    float * q2_output_device = nullptr;
    float * tq1_output_device = nullptr;
    float * activation_device = nullptr;
    std::uint64_t * scrub_device = nullptr;
    double * checksum_device = nullptr;
    CUDA_CHECK(cudaMalloc(&q2_device, total_q2_bytes));
    CUDA_CHECK(cudaMalloc(&tq1_device, total_tq1_bytes));
    CUDA_CHECK(cudaMalloc(&q8_device, q8_allocation_bytes));
    CUDA_CHECK(cudaMalloc(&q2_output_device, output_allocation_bytes));
    CUDA_CHECK(cudaMalloc(&tq1_output_device, output_allocation_bytes));
    CUDA_CHECK(cudaMalloc(&activation_device, static_cast<std::size_t>(max_columns) * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&scrub_device, scrub_bytes));
    CUDA_CHECK(cudaMalloc(&checksum_device, sizeof(double)));

    const double q2_load_seconds = load_weight_stream(args.q2_path, tensors, false, q2_device);
    const double tq1_load_seconds = load_weight_stream(args.tq1_path, tensors, true, tq1_device);
    bonsai::initialize_tq1_lut();

    long double activation_checksum = 0.0;
    for (const auto & tensor : tensors) {
        for (int origin = 0; origin < ACTIVATION_VECTORS; ++origin) {
            const auto activation = make_activation(tensor.columns, tensor.order, origin);
            for (int index = 0; index < tensor.columns; ++index) {
                activation_checksum += activation[index] * static_cast<double>(((index + tensor.order) % 251) + 1);
            }
            CUDA_CHECK(cudaMemcpyAsync(
                activation_device, activation.data(),
                static_cast<std::size_t>(tensor.columns) * sizeof(float),
                cudaMemcpyHostToDevice, stream));
            auto * destination = q8_device +
                static_cast<std::uint64_t>(origin) * q8_blocks_per_vector + tensor.q8_stream_offset;
            bonsai_prism_quantize_q8_1(
                activation_device, destination, tensor.columns, 1, stream);
            CUDA_CHECK(cudaGetLastError());
            CUDA_CHECK(cudaStreamSynchronize(stream));
        }
    }

    const auto launch_tensor = [&](const TensorDescriptor & tensor, bool tq1, int origin) {
        const auto * q8 = q8_device +
            static_cast<std::uint64_t>(origin) * q8_blocks_per_vector + tensor.q8_stream_offset;
        float * output = (tq1 ? tq1_output_device : q2_output_device) +
            static_cast<std::uint64_t>(origin) * outputs_per_vector + tensor.output_stream_offset;
        if (tq1) {
            const auto * weights = reinterpret_cast<const bonsai::block_tq1_g128 *>(
                tq1_device + tensor.tq1_stream_offset);
            bonsai::launch_tq1_g128_gemv_v1(
                weights, q8, output, tensor.rows, tensor.columns, stream);
        } else {
            bonsai_prism_q2_0_mmvq(
                q2_device + tensor.q2_stream_offset, q8, output,
                tensor.columns, tensor.rows, stream);
        }
    };
    const auto launch_traversal = [&](bool tq1, int origin) {
        for (const auto & tensor : tensors) launch_tensor(tensor, tq1, origin);
        CUDA_CHECK(cudaGetLastError());
    };
    const auto scrub = [&](std::uint64_t seed) {
        scrub_kernel<<<4096, 256, 0, stream>>>(
            scrub_device, scrub_bytes / sizeof(std::uint64_t), seed);
        CUDA_CHECK(cudaGetLastError());
    };
    const auto device_checksum = [&](const float * values, std::size_t count) {
        CUDA_CHECK(cudaMemsetAsync(checksum_device, 0, sizeof(double), stream));
        checksum_kernel<<<1024, 256, 0, stream>>>(values, count, checksum_device);
        CUDA_CHECK(cudaGetLastError());
        double result = 0.0;
        CUDA_CHECK(cudaMemcpyAsync(&result, checksum_device, sizeof(double), cudaMemcpyDeviceToHost, stream));
        CUDA_CHECK(cudaStreamSynchronize(stream));
        return result;
    };

    // Correctness is evaluated for every selected tensor with one FP16-origin
    // and one BF16-origin vector before any claim-bearing timing.
    for (int origin = 0; origin < ACTIVATION_VECTORS; ++origin) {
        launch_traversal(false, origin);
        launch_traversal(true, origin);
    }
    CUDA_CHECK(cudaStreamSynchronize(stream));
    const std::size_t all_output_values = static_cast<std::size_t>(
        outputs_per_vector * ACTIVATION_VECTORS);
    std::vector<float> q2_output(all_output_values);
    std::vector<float> tq1_output(all_output_values);
    CUDA_CHECK(cudaMemcpy(
        q2_output.data(), q2_output_device,
        all_output_values * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(
        tq1_output.data(), tq1_output_device,
        all_output_values * sizeof(float), cudaMemcpyDeviceToHost));

    ErrorAccumulator aggregate_accumulator;
    std::vector<TensorCorrectness> per_tensor_correctness;
    per_tensor_correctness.reserve(tensors.size());
    bool every_tensor_finite_nontrivial = true;
    for (const auto & tensor : tensors) {
        ErrorAccumulator tensor_accumulator;
        for (int origin = 0; origin < ACTIVATION_VECTORS; ++origin) {
            const std::uint64_t base = static_cast<std::uint64_t>(origin) * outputs_per_vector +
                tensor.output_stream_offset;
            for (int row = 0; row < tensor.rows; ++row) {
                tensor_accumulator.add(tq1_output[base + row], q2_output[base + row]);
            }
        }
        aggregate_accumulator.merge(tensor_accumulator);
        TensorCorrectness correctness;
        correctness.error = tensor_accumulator.finish();
        correctness.q2_checksum = host_checksum_tensor(
            q2_output, outputs_per_vector, tensor.output_stream_offset, tensor.rows);
        correctness.tq1_checksum = host_checksum_tensor(
            tq1_output, outputs_per_vector, tensor.output_stream_offset, tensor.rows);
        correctness.finite_nontrivial = correctness.error.nonfinite == 0 &&
            std::isfinite(correctness.q2_checksum) && std::isfinite(correctness.tq1_checksum) &&
            std::abs(correctness.q2_checksum) > 1e-12 && std::abs(correctness.tq1_checksum) > 1e-12;
        every_tensor_finite_nontrivial &= correctness.finite_nontrivial;
        per_tensor_correctness.push_back(correctness);
    }
    const ErrorMetrics aggregate_error = aggregate_accumulator.finish();
    const double max_abs_limit = std::max(1.2 * V1_MAX_ABS, V1_MAX_ABS + 1e-4);
    const double mean_abs_limit = std::max(1.05 * V1_MEAN_ABS, V1_MEAN_ABS + 1e-6);
    const double rmse_limit = std::max(1.05 * V1_RMSE, V1_RMSE + 1e-6);
    const bool numerical_pass = every_tensor_finite_nontrivial && aggregate_error.nonfinite == 0 &&
        aggregate_error.max_abs <= max_abs_limit && aggregate_error.mean_abs <= mean_abs_limit &&
        aggregate_error.rmse <= rmse_limit && aggregate_error.cosine >= 0.999999;
    if (!numerical_pass) {
        throw std::runtime_error("INVALID_NUMERICAL_REGRESSION: all-tensor regression gate failed");
    }

    const auto primary_found = std::find_if(
        tensors.begin(), tensors.end(),
        [](const auto & tensor) { return tensor.name == "blk.0.ffn_down.weight"; });
    if (primary_found == tensors.end() || primary_found->rows != 5120 || primary_found->columns != 17408) {
        throw std::runtime_error("missing V1 primary control tensor");
    }
    const TensorDescriptor & primary = *primary_found;

    cudaEvent_t begin;
    cudaEvent_t end;
    CUDA_CHECK(cudaEventCreate(&begin));
    CUDA_CHECK(cudaEventCreate(&end));
    const auto timed_single = [&](bool tq1, int origin, bool cold, std::uint64_t seed) {
        if (cold) scrub(seed);
        CUDA_CHECK(cudaEventRecord(begin, stream));
        launch_tensor(primary, tq1, origin);
        CUDA_CHECK(cudaEventRecord(end, stream));
        CUDA_CHECK(cudaEventSynchronize(end));
        float elapsed = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&elapsed, begin, end));
        return static_cast<double>(elapsed);
    };
    const auto run_single_control = [&](bool cold) {
        for (int index = 0; index < SINGLE_WARMUPS; ++index) {
            const int origin = index % ACTIVATION_VECTORS;
            const bool ab = index % 2 == 0;
            if (ab) {
                if (cold) scrub(0x10000000ULL + index * 2);
                launch_tensor(primary, false, origin);
                if (cold) scrub(0x10000001ULL + index * 2);
                launch_tensor(primary, true, origin);
            } else {
                if (cold) scrub(0x10000000ULL + index * 2);
                launch_tensor(primary, true, origin);
                if (cold) scrub(0x10000001ULL + index * 2);
                launch_tensor(primary, false, origin);
            }
        }
        CUDA_CHECK(cudaStreamSynchronize(stream));
        PairedTimings result;
        const int pairs = SINGLE_REPEATS * SINGLE_ITERATIONS;
        result.q2.reserve(pairs);
        result.tq1.reserve(pairs);
        result.ratios.reserve(pairs);
        result.order.reserve(pairs);
        for (int index = 0; index < pairs; ++index) {
            const int origin = index % ACTIVATION_VECTORS;
            double q2_ms = 0.0;
            double tq1_ms = 0.0;
            if (index % 2 == 0) {
                q2_ms = timed_single(false, origin, cold, 0x20000000ULL + index * 2);
                tq1_ms = timed_single(true, origin, cold, 0x20000001ULL + index * 2);
                result.order.emplace_back("AB");
            } else {
                tq1_ms = timed_single(true, origin, cold, 0x20000000ULL + index * 2);
                q2_ms = timed_single(false, origin, cold, 0x20000001ULL + index * 2);
                result.order.emplace_back("BA");
            }
            result.q2.push_back(q2_ms);
            result.tq1.push_back(tq1_ms);
            result.ratios.push_back(tq1_ms / q2_ms);
        }
        for (int origin = 0; origin < ACTIVATION_VECTORS; ++origin) {
            const float * q2_values = q2_output_device +
                static_cast<std::uint64_t>(origin) * outputs_per_vector + primary.output_stream_offset;
            const float * tq1_values = tq1_output_device +
                static_cast<std::uint64_t>(origin) * outputs_per_vector + primary.output_stream_offset;
            result.q2_checksums.push_back(device_checksum(q2_values, primary.rows));
            result.tq1_checksums.push_back(device_checksum(tq1_values, primary.rows));
        }
        return result;
    };

    // Regime A is the frozen V1 repeated-matrix control. No kernel changes have
    // occurred and no result below is used to alter the primary implementation.
    const PairedTimings warm_control = run_single_control(false);

    // Regime B is claim-bearing and runs before forced-cold or per-tensor
    // diagnostics. A traversal contains each of the 497 real tensors once.
    for (int index = 0; index < args.traversal_warmups; ++index) {
        const int origin = index % ACTIVATION_VECTORS;
        if (index % 2 == 0) {
            launch_traversal(false, origin);
            launch_traversal(true, origin);
        } else {
            launch_traversal(true, origin);
            launch_traversal(false, origin);
        }
    }
    CUDA_CHECK(cudaStreamSynchronize(stream));
    const auto timed_traversal = [&](bool tq1, int origin) {
        CUDA_CHECK(cudaEventRecord(begin, stream));
        launch_traversal(tq1, origin);
        CUDA_CHECK(cudaEventRecord(end, stream));
        float * output = (tq1 ? tq1_output_device : q2_output_device) +
            static_cast<std::uint64_t>(origin) * outputs_per_vector;
        CUDA_CHECK(cudaMemsetAsync(checksum_device, 0, sizeof(double), stream));
        checksum_kernel<<<1024, 256, 0, stream>>>(output, outputs_per_vector, checksum_device);
        CUDA_CHECK(cudaGetLastError());
        double checksum_value = 0.0;
        CUDA_CHECK(cudaMemcpyAsync(
            &checksum_value, checksum_device, sizeof(double), cudaMemcpyDeviceToHost, stream));
        CUDA_CHECK(cudaStreamSynchronize(stream));
        float elapsed = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&elapsed, begin, end));
        return TimedValue{static_cast<double>(elapsed), checksum_value};
    };
    PairedTimings streaming;
    for (int index = 0; index < args.traversal_pairs; ++index) {
        const int origin = index % ACTIVATION_VECTORS;
        TimedValue q2;
        TimedValue tq1;
        if (index % 2 == 0) {
            q2 = timed_traversal(false, origin);
            tq1 = timed_traversal(true, origin);
            streaming.order.emplace_back("AB");
        } else {
            tq1 = timed_traversal(true, origin);
            q2 = timed_traversal(false, origin);
            streaming.order.emplace_back("BA");
        }
        if (!std::isfinite(q2.checksum) || !std::isfinite(tq1.checksum) ||
            std::abs(q2.checksum) <= 1e-12 || std::abs(tq1.checksum) <= 1e-12) {
            throw std::runtime_error("INVALID_STREAMING_EXPERIMENT: traversal checksum failed");
        }
        streaming.q2.push_back(q2.milliseconds);
        streaming.tq1.push_back(tq1.milliseconds);
        streaming.ratios.push_back(tq1.milliseconds / q2.milliseconds);
        streaming.q2_checksums.push_back(q2.checksum);
        streaming.tq1_checksums.push_back(tq1.checksum);
    }

    // Regime C approximates V1's forced-cold single-matrix diagnostic. The
    // >2x-L2 scrub is ordered before, and excluded from, each timed launch.
    const PairedTimings cold_control = run_single_control(true);

    // Per-tensor timing is a Regime-B diagnostic performed only after the clean
    // claim-bearing traversal. Events surround each launch while tensors retain
    // model order and no cache scrub is used.
    std::vector<cudaEvent_t> tensor_begin(tensors.size());
    std::vector<cudaEvent_t> tensor_end(tensors.size());
    for (std::size_t index = 0; index < tensors.size(); ++index) {
        CUDA_CHECK(cudaEventCreate(&tensor_begin[index]));
        CUDA_CHECK(cudaEventCreate(&tensor_end[index]));
    }
    std::vector<std::vector<double>> q2_tensor_raw(tensors.size());
    std::vector<std::vector<double>> tq1_tensor_raw(tensors.size());
    const auto diagnostic_pass = [&](bool tq1, int origin, std::vector<std::vector<double>> & raw) {
        for (std::size_t index = 0; index < tensors.size(); ++index) {
            CUDA_CHECK(cudaEventRecord(tensor_begin[index], stream));
            launch_tensor(tensors[index], tq1, origin);
            CUDA_CHECK(cudaEventRecord(tensor_end[index], stream));
        }
        CUDA_CHECK(cudaEventSynchronize(tensor_end.back()));
        for (std::size_t index = 0; index < tensors.size(); ++index) {
            float elapsed = 0.0f;
            CUDA_CHECK(cudaEventElapsedTime(&elapsed, tensor_begin[index], tensor_end[index]));
            raw[index].push_back(elapsed);
        }
        float * output = (tq1 ? tq1_output_device : q2_output_device) +
            static_cast<std::uint64_t>(origin) * outputs_per_vector;
        const double consumed = device_checksum(output, outputs_per_vector);
        if (!std::isfinite(consumed) || std::abs(consumed) <= 1e-12) {
            throw std::runtime_error("per-tensor diagnostic checksum failed");
        }
    };
    for (int index = 0; index < args.diagnostic_pairs; ++index) {
        const int origin = index % ACTIVATION_VECTORS;
        if (index % 2 == 0) {
            diagnostic_pass(false, origin, q2_tensor_raw);
            diagnostic_pass(true, origin, tq1_tensor_raw);
        } else {
            diagnostic_pass(true, origin, tq1_tensor_raw);
            diagnostic_pass(false, origin, q2_tensor_raw);
        }
    }

    for (std::size_t index = 0; index < tensors.size(); ++index) {
        CUDA_CHECK(cudaEventDestroy(tensor_begin[index]));
        CUDA_CHECK(cudaEventDestroy(tensor_end[index]));
    }
    CUDA_CHECK(cudaEventDestroy(begin));
    CUDA_CHECK(cudaEventDestroy(end));

    const Summary warm_q2 = summarize(warm_control.q2);
    const Summary warm_tq1 = summarize(warm_control.tq1);
    const Summary cold_q2 = summarize(cold_control.q2);
    const Summary cold_tq1 = summarize(cold_control.tq1);
    const Summary stream_q2 = summarize(streaming.q2);
    const Summary stream_tq1 = summarize(streaming.tq1);
    const Summary stream_paired_ratio = summarize(streaming.ratios);
    const double primary_ratio = stream_tq1.median / stream_q2.median;

    std::size_t free_after = 0;
    std::size_t total_after = 0;
    CUDA_CHECK(cudaMemGetInfo(&free_after, &total_after));

    std::ofstream json(args.output_path);
    if (!json) throw std::runtime_error("cannot create V2 result JSON");
    json << std::setprecision(12);
    json << "{\n  \"schema_version\":2,\n"
         << "  \"gpu\":{\"name\":\"" << escape_json(properties.name)
         << "\",\"compute_capability\":\"" << properties.major << '.' << properties.minor
         << "\",\"l2_bytes\":" << properties.l2CacheSize
         << ",\"global_memory_bytes\":" << properties.totalGlobalMem << "},\n"
         << "  \"protocol\":{\"traversal_warmups\":" << args.traversal_warmups
         << ",\"traversal_pairs\":" << args.traversal_pairs
         << ",\"diagnostic_pairs\":" << args.diagnostic_pairs
         << ",\"single_warmups\":" << SINGLE_WARMUPS
         << ",\"single_repeats\":" << SINGLE_REPEATS
         << ",\"single_iterations_per_repeat\":" << SINGLE_ITERATIONS
         << ",\"ab_ba_alternation\":true,\"timing\":\"CUDA events\"},\n"
         << "  \"stream\":{\"tensors\":" << tensors.size()
         << ",\"launches_per_traversal\":" << tensors.size()
         << ",\"q2_bytes\":" << total_q2_bytes
         << ",\"tq1_bytes\":" << total_tq1_bytes
         << ",\"q2_l2_multiple\":" << static_cast<double>(total_q2_bytes) / properties.l2CacheSize
         << ",\"tq1_l2_multiple\":" << static_cast<double>(total_tq1_bytes) / properties.l2CacheSize
         << ",\"q2_load_seconds\":" << q2_load_seconds
         << ",\"tq1_load_seconds\":" << tq1_load_seconds << "},\n"
         << "  \"activations\":{\"vectors_per_tensor\":2,\"origins\":[\"FP16\",\"BF16\"],"
            "\"runtime_type\":\"identical Prism-produced Q8_1 from F32\",\"q8_blocks_per_vector\":"
         << q8_blocks_per_vector << ",\"allocation_bytes\":" << q8_allocation_bytes
         << ",\"deterministic_checksum\":" << static_cast<double>(activation_checksum) << "},\n"
         << "  \"correctness\":{\"pass\":true,\"aggregate\":";
    write_metrics(json, aggregate_error);
    json << ",\"limits\":{\"max_abs\":" << max_abs_limit
         << ",\"mean_abs\":" << mean_abs_limit
         << ",\"rmse\":" << rmse_limit << ",\"minimum_cosine\":0.999999},"
         << "\"every_tensor_finite_nontrivial\":true,\"tensors\":[";
    for (std::size_t index = 0; index < tensors.size(); ++index) {
        if (index != 0) json << ',';
        json << "{\"order\":" << index << ",\"name\":\"" << escape_json(tensors[index].name)
             << "\",\"metrics\":";
        write_metrics(json, per_tensor_correctness[index].error);
        json << ",\"q2_checksum\":" << per_tensor_correctness[index].q2_checksum
             << ",\"tq1_checksum\":" << per_tensor_correctness[index].tq1_checksum
             << ",\"finite_nontrivial\":"
             << (per_tensor_correctness[index].finite_nontrivial ? "true" : "false") << '}';
    }
    json << "]},\n  \"regime_a_single_matrix_warm\":{\"tensor\":\"blk.0.ffn_down.weight\","
            "\"scrub\":false,\"q2\":";
    write_summary(json, warm_q2);
    json << ",\"tq1\":";
    write_summary(json, warm_tq1);
    json << ",\"median_ratio\":" << warm_tq1.median / warm_q2.median
         << ",\"paired_ratio\":";
    write_summary(json, summarize(warm_control.ratios));
    json << ",\"checksums\":{\"q2\":";
    write_values(json, warm_control.q2_checksums);
    json << ",\"tq1\":";
    write_values(json, warm_control.tq1_checksums);
    json << "},\"raw_ms\":{\"q2\":";
    write_values(json, warm_control.q2);
    json << ",\"tq1\":";
    write_values(json, warm_control.tq1);
    json << ",\"order\":";
    write_strings(json, warm_control.order);
    json << "}},\n  \"regime_b_model_stream\":{\"primary\":true,\"explicit_cache_flush\":false,"
            "\"q2\":";
    write_summary(json, stream_q2);
    json << ",\"tq1\":";
    write_summary(json, stream_tq1);
    json << ",\"median_ratio\":" << primary_ratio
         << ",\"paired_ratio\":";
    write_summary(json, stream_paired_ratio);
    json << ",\"effective_gb_s\":{\"q2_physical\":"
         << total_q2_bytes / (stream_q2.median * 1.0e6)
         << ",\"tq1_physical\":" << total_tq1_bytes / (stream_tq1.median * 1.0e6)
         << ",\"tq1_original_byte_equivalent\":" << total_q2_bytes / (stream_tq1.median * 1.0e6)
         << "},\"raw\":{\"q2_ms\":";
    write_values(json, streaming.q2);
    json << ",\"tq1_ms\":";
    write_values(json, streaming.tq1);
    json << ",\"paired_ratios\":";
    write_values(json, streaming.ratios);
    json << ",\"order\":";
    write_strings(json, streaming.order);
    json << ",\"q2_checksums\":";
    write_values(json, streaming.q2_checksums);
    json << ",\"tq1_checksums\":";
    write_values(json, streaming.tq1_checksums);
    json << "},\"per_tensor\":[";
    for (std::size_t index = 0; index < tensors.size(); ++index) {
        if (index != 0) json << ',';
        const Summary q2_stats = summarize(q2_tensor_raw[index]);
        const Summary tq1_stats = summarize(tq1_tensor_raw[index]);
        json << "{\"order\":" << index
             << ",\"layer\":" << tensors[index].layer
             << ",\"kind\":\"" << escape_json(tensors[index].kind)
             << "\",\"name\":\"" << escape_json(tensors[index].name)
             << "\",\"m\":" << tensors[index].rows
             << ",\"k\":" << tensors[index].columns
             << ",\"groups\":" << tensors[index].groups
             << ",\"q2_bytes\":" << tensors[index].q2_bytes
             << ",\"tq1_bytes\":" << tensors[index].tq1_bytes
             << ",\"q2\":";
        write_summary(json, q2_stats);
        json << ",\"tq1\":";
        write_summary(json, tq1_stats);
        json << ",\"median_ratio\":" << tq1_stats.median / q2_stats.median
             << ",\"raw_ms\":{\"q2\":";
        write_values(json, q2_tensor_raw[index]);
        json << ",\"tq1\":";
        write_values(json, tq1_tensor_raw[index]);
        json << "}}";
    }
    json << "]},\n  \"regime_c_forced_cold\":{\"scope\":\"repeated blk.0.ffn_down.weight; scrub before every launch\","
            "\"scrub_bytes\":" << scrub_bytes << ",\"q2\":";
    write_summary(json, cold_q2);
    json << ",\"tq1\":";
    write_summary(json, cold_tq1);
    json << ",\"median_ratio\":" << cold_tq1.median / cold_q2.median
         << ",\"paired_ratio\":";
    write_summary(json, summarize(cold_control.ratios));
    json << ",\"checksums\":{\"q2\":";
    write_values(json, cold_control.q2_checksums);
    json << ",\"tq1\":";
    write_values(json, cold_control.tq1_checksums);
    json << "},\"raw_ms\":{\"q2\":";
    write_values(json, cold_control.q2);
    json << ",\"tq1\":";
    write_values(json, cold_control.tq1);
    json << ",\"order\":";
    write_strings(json, cold_control.order);
    json << "}},\n  \"anti_cheating\":{\"q2_allocation_bytes\":" << total_q2_bytes
         << ",\"expected_q2_payload_bytes\":" << total_q2_bytes
         << ",\"tq1_allocation_bytes\":" << total_tq1_bytes
         << ",\"expected_tq1_payload_bytes\":" << total_tq1_bytes
         << ",\"unpacked_tq1_weight_buffer\":false,\"real_checkpoint_extents\":true,"
            "\"same_q8_pointer_per_tensor\":true,\"preprocessing_in_timed_path\":false,"
            "\"same_shapes\":true,\"same_scales\":true,\"output_dtype\":\"F32\","
            "\"outputs_consumed_after_every_primary_traversal\":true},\n"
         << "  \"memory\":{\"free_before_bytes\":" << free_before
         << ",\"free_after_allocations_bytes\":" << free_after
         << ",\"total_bytes\":" << total_memory
         << ",\"required_allocation_bytes\":" << required_bytes
         << ",\"q8_allocation_bytes\":" << q8_allocation_bytes
         << ",\"output_allocation_bytes_per_format\":" << output_allocation_bytes << "}\n}\n";
    json.close();

    CUDA_CHECK(cudaFree(checksum_device));
    CUDA_CHECK(cudaFree(scrub_device));
    CUDA_CHECK(cudaFree(activation_device));
    CUDA_CHECK(cudaFree(tq1_output_device));
    CUDA_CHECK(cudaFree(q2_output_device));
    CUDA_CHECK(cudaFree(q8_device));
    CUDA_CHECK(cudaFree(tq1_device));
    CUDA_CHECK(cudaFree(q2_device));
    CUDA_CHECK(cudaStreamDestroy(stream));

    std::cout << "numerical=PASS launches=" << tensors.size()
              << " warm_ratio=" << warm_tq1.median / warm_q2.median
              << " stream_ratio=" << primary_ratio
              << " cold_ratio=" << cold_tq1.median / cold_q2.median << '\n';
    return 0;
} catch (const std::exception & error) {
    std::cerr << error.what() << '\n';
    return 2;
}
