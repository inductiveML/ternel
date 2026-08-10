cmake_minimum_required(VERSION 3.25)

project(ternel LANGUAGES C CXX CUDA)

find_package(OpenMP REQUIRED)
find_package(CUDAToolkit REQUIRED)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

if(NOT DEFINED PRISM_SOURCE_DIR)
    set(PRISM_SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/artifacts/src/llama.cpp")
endif()
if(NOT EXISTS "${PRISM_SOURCE_DIR}/CMakeLists.txt")
    message(FATAL_ERROR "Pinned Prism source missing; run uv run python tools/bootstrap_prism.py")
endif()

set(GGML_CUDA ON CACHE BOOL "" FORCE)
set(BUILD_SHARED_LIBS OFF CACHE BOOL "" FORCE)
set(GGML_STATIC OFF CACHE BOOL "" FORCE)
set(GGML_BACKEND_DL OFF CACHE BOOL "" FORCE)
set(GGML_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_COMMON ON CACHE BOOL "" FORCE)
set(LLAMA_BUILD_TOOLS ON CACHE BOOL "" FORCE)
set(LLAMA_BUILD_APP ON CACHE BOOL "" FORCE)
set(LLAMA_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_SERVER ON CACHE BOOL "" FORCE)
set(LLAMA_CURL OFF CACHE BOOL "" FORCE)

add_subdirectory("${PRISM_SOURCE_DIR}" "${CMAKE_BINARY_DIR}/prism" EXCLUDE_FROM_ALL)

add_library(tq1-g128 STATIC cuda/tq1_g128_gemv.cu)
target_include_directories(tq1-g128 PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/cuda")
target_compile_options(tq1-g128 PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-lineinfo;--ptxas-options=-v>)

add_library(tq1-lut23-streamk STATIC cuda/tq1_lut23_streamk.cu)
target_include_directories(tq1-lut23-streamk PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/cuda")
target_link_libraries(tq1-lut23-streamk PUBLIC tq1-g128 CUDA::cudart)
target_compile_options(tq1-lut23-streamk PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-lineinfo;--ptxas-options=-v>)

add_executable(benchmark-gemv cuda/benchmark_gemv.cu)
target_include_directories(benchmark-gemv PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}/cuda"
    "${PRISM_SOURCE_DIR}/ggml/include"
    "${PRISM_SOURCE_DIR}/ggml/src"
    "${PRISM_SOURCE_DIR}/ggml/src/ggml-cuda")
target_link_libraries(benchmark-gemv PRIVATE tq1-g128 ggml-cuda CUDA::cudart OpenMP::OpenMP_CXX)
target_compile_options(benchmark-gemv PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-lineinfo;--ptxas-options=-v>)

add_executable(benchmark-model-stream cuda/benchmark_model_stream.cu)
target_include_directories(benchmark-model-stream PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}/cuda"
    "${PRISM_SOURCE_DIR}/ggml/include"
    "${PRISM_SOURCE_DIR}/ggml/src"
    "${PRISM_SOURCE_DIR}/ggml/src/ggml-cuda")
target_link_libraries(benchmark-model-stream PRIVATE tq1-g128 ggml-cuda CUDA::cudart OpenMP::OpenMP_CXX)
target_compile_options(benchmark-model-stream PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-lineinfo;--ptxas-options=-v>)

add_executable(benchmark-lut23-primary cuda/benchmark_lut23_primary.cu)
target_include_directories(benchmark-lut23-primary PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}/cuda"
    "${PRISM_SOURCE_DIR}/ggml/include"
    "${PRISM_SOURCE_DIR}/ggml/src"
    "${PRISM_SOURCE_DIR}/ggml/src/ggml-cuda")
target_link_libraries(benchmark-lut23-primary PRIVATE
    tq1-lut23-streamk tq1-g128 ggml-cuda CUDA::cudart OpenMP::OpenMP_CXX)
target_compile_options(benchmark-lut23-primary PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-lineinfo;--ptxas-options=-v>)

add_executable(validate-prism-bridge cuda/validate_prism_bridge.cpp)
target_include_directories(validate-prism-bridge PRIVATE
    "${PRISM_SOURCE_DIR}/ggml/include"
    "${PRISM_SOURCE_DIR}/ggml/src"
    "${PRISM_SOURCE_DIR}/ggml/src/ggml-cuda")
target_link_libraries(validate-prism-bridge PRIVATE ggml-base ggml-cuda CUDA::cudart)

add_executable(validate-llama-distribution native/validate_llama_distribution.cpp)
target_include_directories(validate-llama-distribution PRIVATE
    "${PRISM_SOURCE_DIR}/include"
    "${PRISM_SOURCE_DIR}/ggml/include")
target_link_libraries(validate-llama-distribution PRIVATE llama)
