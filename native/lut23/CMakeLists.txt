cmake_minimum_required(VERSION 3.25)

project(tq1_lut23_final LANGUAGES CXX CUDA)

find_package(CUDAToolkit REQUIRED)
find_package(OpenMP REQUIRED)
find_package(Threads REQUIRED)
find_library(NCCL_LIBRARY nccl REQUIRED)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

get_filename_component(EXPERIMENT_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/../.." ABSOLUTE)
set(PRISM_SOURCE_DIR "${EXPERIMENT_ROOT}/artifacts/src/llama.cpp")
set(FROZEN_BUILD_DIR "${EXPERIMENT_ROOT}/build")

foreach(required
        "${FROZEN_BUILD_DIR}/libtq1-g128.a"
        "${FROZEN_BUILD_DIR}/prism/ggml/src/ggml-cuda/libggml-cuda.a"
        "${FROZEN_BUILD_DIR}/prism/ggml/src/libggml-base.a")
    if(NOT EXISTS "${required}")
        message(FATAL_ERROR "Missing frozen baseline artifact: ${required}")
    endif()
endforeach()

add_library(tq1-lut23-streamk STATIC
    "${EXPERIMENT_ROOT}/cuda/tq1_lut23_streamk.cu")
target_include_directories(tq1-lut23-streamk PUBLIC "${EXPERIMENT_ROOT}/cuda")
target_compile_options(tq1-lut23-streamk PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:-lineinfo;--ptxas-options=-v>)

add_executable(benchmark-lut23-primary
    "${EXPERIMENT_ROOT}/cuda/benchmark_lut23_primary.cu")
target_include_directories(benchmark-lut23-primary PRIVATE
    "${EXPERIMENT_ROOT}/cuda"
    "${PRISM_SOURCE_DIR}/ggml/include"
    "${PRISM_SOURCE_DIR}/ggml/src"
    "${PRISM_SOURCE_DIR}/ggml/src/ggml-cuda")
target_compile_options(benchmark-lut23-primary PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:-lineinfo;--ptxas-options=-v>)
target_link_libraries(benchmark-lut23-primary PRIVATE
    tq1-lut23-streamk
    "${FROZEN_BUILD_DIR}/libtq1-g128.a"
    "${FROZEN_BUILD_DIR}/prism/ggml/src/ggml-cuda/libggml-cuda.a"
    "${FROZEN_BUILD_DIR}/prism/ggml/src/libggml-base.a"
    CUDA::cudart
    CUDA::cublas
    CUDA::cublasLt
    CUDA::culibos
    CUDA::cuda_driver
    "${NCCL_LIBRARY}"
    OpenMP::OpenMP_CXX
    Threads::Threads
    m
    dl)
