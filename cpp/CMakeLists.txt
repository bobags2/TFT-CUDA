cmake_minimum_required(VERSION 3.18)
project(TFT_CUDA LANGUAGES CXX CUDA)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find required packages
find_package(CUDA REQUIRED)
find_package(Python COMPONENTS Interpreter Development REQUIRED)
find_package(pybind11 REQUIRED)

# PyTorch
execute_process(
    COMMAND ${Python_EXECUTABLE} -c "import torch; print(torch.utils.cmake_prefix_path)"
    OUTPUT_VARIABLE TORCH_CMAKE_PREFIX_PATH
    OUTPUT_STRIP_TRAILING_WHITESPACE
)
set(CMAKE_PREFIX_PATH ${TORCH_CMAKE_PREFIX_PATH})
find_package(Torch REQUIRED)

# CUDA architecture
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES 75 80 86)
endif()

# Compiler flags
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -Wall -Wextra")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -O3 -use_fast_math -Xptxas -v --expt-relaxed-constexpr")

# Include directories
include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/forward_pass
    ${CMAKE_CURRENT_SOURCE_DIR}/backward_pass
    ${CMAKE_CURRENT_SOURCE_DIR}/interpretability
    ${CMAKE_CURRENT_SOURCE_DIR}/wrappers
    ${TORCH_INCLUDE_DIRS}
)

# Source files
set(CUDA_SOURCES
    forward_pass/lstm.cu
    forward_pass/mha.cu
    forward_pass/linear_fwd.cu
    forward_pass/quantile_heads.cu
    forward_pass/static_encoder.cu
    backward_pass/lstm_backward.cu
    backward_pass/mha_backward.cu
    backward_pass/linear_bwd.cu
    backward_pass/quantile_heads_backward.cu
    backward_pass/layer_norm_backward.cu
    interpretability/attention_aggregate.cu
    interpretability/vsn_aggregate.cu
    interpretability/static_embedding_importance.cu
)

set(CPP_SOURCES
    bindings/tft_bindings.cpp
)

# Create Python extension
pybind11_add_module(tft_cuda ${CPP_SOURCES} ${CUDA_SOURCES})

# Link libraries
target_link_libraries(tft_cuda PRIVATE ${TORCH_LIBRARIES})

# Compiler-specific options
target_compile_definitions(tft_cuda PRIVATE VERSION_INFO=${VERSION_INFO})

# Set properties for CUDA compilation
set_property(TARGET tft_cuda PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS ON)

# Installation
install(TARGETS tft_cuda DESTINATION .)

# Optional: Create a library for C++ usage
add_library(tft_cuda_lib STATIC ${CUDA_SOURCES})
target_link_libraries(tft_cuda_lib ${TORCH_LIBRARIES})
set_property(TARGET tft_cuda_lib PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS ON)
