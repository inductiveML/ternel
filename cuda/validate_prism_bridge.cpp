#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-cuda.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
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

#define CUDA_CHECK(call) do { \
    const cudaError_t error_ = (call); \
    if (error_ != cudaSuccess) throw std::runtime_error(cudaGetErrorString(error_)); \
} while (false)

struct Arguments {
    std::string q2_path;
    std::string output_path;
    std::uint64_t q2_offset = 0;
    int rows = 0;
    int columns = 0;
};

Arguments parse_arguments(int argc, char ** argv) {
    Arguments result;
    for (int index = 1; index < argc; ++index) {
        if (index + 1 >= argc) throw std::invalid_argument("missing argument value");
        const std::string key = argv[index];
        const std::string value = argv[++index];
        if (key == "--q2") result.q2_path = value;
        else if (key == "--q2-offset") result.q2_offset = std::stoull(value);
        else if (key == "--rows") result.rows = std::stoi(value);
        else if (key == "--columns") result.columns = std::stoi(value);
        else if (key == "--output") result.output_path = value;
        else throw std::invalid_argument("unknown argument " + key);
    }
    if (result.q2_path.empty() || result.output_path.empty() || result.rows <= 0 ||
        result.columns <= 0 || result.columns % 128 != 0) {
        throw std::invalid_argument("invalid or incomplete arguments");
    }
    return result;
}

std::vector<std::uint8_t> read_extent(
        const std::string & path, std::uint64_t offset, std::size_t bytes) {
    std::ifstream input(path, std::ios::binary);
    if (!input) throw std::runtime_error("cannot open source tensor");
    input.seekg(static_cast<std::streamoff>(offset));
    std::vector<std::uint8_t> result(bytes);
    input.read(reinterpret_cast<char *>(result.data()), static_cast<std::streamsize>(bytes));
    if (input.gcount() != static_cast<std::streamsize>(bytes)) {
        throw std::runtime_error("truncated source tensor");
    }
    return result;
}

}  // namespace

int main(int argc, char ** argv) try {
    const Arguments args = parse_arguments(argc, argv);
    const std::size_t q2_bytes = static_cast<std::size_t>(args.rows) * (args.columns / 128) * 34;
    const auto q2_host = read_extent(args.q2_path, args.q2_offset, q2_bytes);
    std::mt19937 generator(0xB05A1u);
    std::normal_distribution<float> normal(0.0f, 1.0f);
    std::vector<float> activation(args.columns);
    for (float & value : activation) value = normal(generator);

    ggml_backend_t backend = ggml_backend_cuda_init(0);
    if (backend == nullptr) throw std::runtime_error("ggml_backend_cuda_init failed");
    ggml_init_params params = {
        /* .mem_size = */ 4u << 20,
        /* .mem_buffer = */ nullptr,
        /* .no_alloc = */ true,
    };
    ggml_context * context = ggml_init(params);
    if (context == nullptr) throw std::runtime_error("ggml_init failed");
    ggml_tensor * weights = ggml_new_tensor_2d(context, GGML_TYPE_Q2_0, args.columns, args.rows);
    ggml_tensor * input = ggml_new_tensor_1d(context, GGML_TYPE_F32, args.columns);
    ggml_tensor * output = ggml_mul_mat(context, weights, input);
    ggml_cgraph * graph = ggml_new_graph(context);
    ggml_build_forward_expand(graph, output);
    ggml_backend_buffer_t buffer = ggml_backend_alloc_ctx_tensors(context, backend);
    if (buffer == nullptr) throw std::runtime_error("graph tensor allocation failed");
    ggml_backend_tensor_set(weights, q2_host.data(), 0, q2_host.size());
    ggml_backend_tensor_set(input, activation.data(), 0, activation.size() * sizeof(float));
    const ggml_status status = ggml_backend_graph_compute(backend, graph);
    if (status != GGML_STATUS_SUCCESS) throw std::runtime_error("public graph compute failed");
    std::vector<float> graph_output(args.rows);
    ggml_backend_tensor_get(output, graph_output.data(), 0, graph_output.size() * sizeof(float));

    void * q2_device = nullptr;
    void * q8_device = nullptr;
    float * input_device = nullptr;
    float * bridge_output_device = nullptr;
    CUDA_CHECK(cudaMalloc(&q2_device, q2_bytes));
    CUDA_CHECK(cudaMalloc(&q8_device, static_cast<std::size_t>(args.columns / 32) * 36));
    CUDA_CHECK(cudaMalloc(&input_device, activation.size() * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&bridge_output_device, static_cast<std::size_t>(args.rows) * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(q2_device, q2_host.data(), q2_bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(input_device, activation.data(), activation.size() * sizeof(float), cudaMemcpyHostToDevice));
    bonsai_prism_quantize_q8_1(input_device, q8_device, args.columns, 1, nullptr);
    bonsai_prism_q2_0_mmvq(q2_device, q8_device, bridge_output_device, args.columns, args.rows, nullptr);
    CUDA_CHECK(cudaGetLastError());
    std::vector<float> bridge_output(args.rows);
    CUDA_CHECK(cudaMemcpy(bridge_output.data(), bridge_output_device,
                          bridge_output.size() * sizeof(float), cudaMemcpyDeviceToHost));

    std::uint64_t bit_mismatches = 0;
    double max_abs = 0.0;
    for (int row = 0; row < args.rows; ++row) {
        if (std::memcmp(&graph_output[row], &bridge_output[row], sizeof(float)) != 0) ++bit_mismatches;
        max_abs = std::max(max_abs, std::abs(static_cast<double>(graph_output[row]) - bridge_output[row]));
    }
    const bool pass = bit_mismatches == 0;
    std::ofstream json(args.output_path);
    if (!json) throw std::runtime_error("cannot create validation result");
    json << std::setprecision(12)
         << "{\n  \"schema_version\":1,\n  \"public_path\":\"ggml MUL_MAT graph\",\n"
         << "  \"bridge_path\":\"pinned private Q2_0 MMVQ dispatcher\",\n"
         << "  \"rows\":" << args.rows << ",\n  \"columns\":" << args.columns << ",\n"
         << "  \"bit_mismatches\":" << bit_mismatches << ",\n"
         << "  \"max_abs_error\":" << max_abs << ",\n  \"pass\":" << (pass ? "true" : "false") << "\n}\n";

    CUDA_CHECK(cudaFree(bridge_output_device));
    CUDA_CHECK(cudaFree(input_device));
    CUDA_CHECK(cudaFree(q8_device));
    CUDA_CHECK(cudaFree(q2_device));
    ggml_backend_buffer_free(buffer);
    ggml_free(context);
    ggml_backend_free(backend);
    std::cout << "bridge_validation=" << (pass ? "PASS" : "INVALID_EXPERIMENT")
              << " mismatches=" << bit_mismatches << " max_abs=" << max_abs << '\n';
    return pass ? 0 : 3;
} catch (const std::exception & error) {
    std::cerr << error.what() << '\n';
    return 2;
}
