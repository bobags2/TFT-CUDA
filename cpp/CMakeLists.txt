cmake_minimum_required(VERSION 3.18)

# Force specific compiler for PyTorch compatibility
set(CMAKE_C_COMPILER "/usr/bin/gcc-11")
set(CMAKE_CXX_COMPILER "/usr/bin/g++-11")

# Enable CUDA language support
project(TFT_CUDA LANGUAGES CXX CUDA)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find required packages
find_package(Python COMPONENTS Interpreter Development REQUIRED)

# Find threading support first
set(CMAKE_THREAD_PREFER_PTHREAD TRUE)
set(THREADS_PREFER_PTHREAD_FLAG TRUE)
find_package(Threads REQUIRED)

# PyTorch development header compatibility fixes
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DTORCH_EXTENSION_NAME=tft_cuda")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -D_GLIBCXX_USE_CXX11_ABI=1")

# Disable problematic warnings that become errors with PyTorch dev headers
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-deprecated-declarations")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-unused-variable")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-unused-but-set-variable")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-missing-field-initializers")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-type-limits")

# Critical fixes for PyTorch dev header template issues
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-placement-new")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-error=placement-new")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-error=maybe-uninitialized")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-stringop-overflow")

# Template instantiation workarounds for GCC/PyTorch compatibility
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -ftemplate-backtrace-limit=0")
# Remove invalid flag that doesn't exist in GCC-11
# set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-error=template-id-cdtor")

# PyTorch development header specific fixes
add_compile_definitions(
    TORCH_EXTENSION_NAME=tft_cuda
    _DISABLE_CONSTEXPR_MUTEX_CONSTRUCTOR
    TORCH_API_INCLUDE_EXTENSION_H
)

# Find CUDA Toolkit (new method) - make optional to avoid threading issues
if(NOT DISABLE_CUDA_TOOLKIT)
    find_package(CUDAToolkit QUIET)
    if(CUDAToolkit_FOUND)
        message(STATUS "CUDA Toolkit found: ${CUDAToolkit_VERSION}")
        set(CUDA_TOOLKIT_ROOT_DIR ${CUDAToolkit_TARGET_DIR})
    else()
        message(WARNING "CUDAToolkit not found, using fallback")
        # Fallback to explicit path
        set(CUDA_TOOLKIT_ROOT_DIR "/usr/local/cuda")
        set(CUDAToolkit_BIN_DIR "/usr/local/cuda/bin")
    endif()
else()
    message(STATUS "CUDAToolkit disabled, using fallback paths")
    set(CUDA_TOOLKIT_ROOT_DIR "/usr/local/cuda")
    set(CUDAToolkit_BIN_DIR "/usr/local/cuda/bin")
endif()

# PyTorch still requires legacy CUDA package for compatibility
find_package(CUDA QUIET)
if(CUDA_FOUND)
    message(STATUS "CUDA (legacy) found: ${CUDA_VERSION}")
else()
    message(WARNING "CUDA (legacy) not found")
    set(CUDA_TOOLKIT_ROOT_DIR "/usr/local/cuda")
    set(CUDA_INCLUDE_DIRS "/usr/local/cuda/include")
    set(CUDA_LIBRARIES "/usr/local/cuda/lib64")
endif()

# Find PyTorch with error handling for development versions
execute_process(
    COMMAND ${Python_EXECUTABLE} -c "import torch; print(torch.utils.cmake_prefix_path)"
    OUTPUT_VARIABLE TORCH_CMAKE_PREFIX_PATH
    OUTPUT_STRIP_TRAILING_WHITESPACE
    ERROR_QUIET
)

if(TORCH_CMAKE_PREFIX_PATH)
    set(CMAKE_PREFIX_PATH ${TORCH_CMAKE_PREFIX_PATH})
    find_package(Torch QUIET)
endif()

if(NOT Torch_FOUND)
    message(STATUS "PyTorch CMake not found, using Python detection")
    execute_process(
        COMMAND ${Python_EXECUTABLE} -c "import torch; print(torch.__file__)"
        OUTPUT_VARIABLE TORCH_PYTHON_PATH
        OUTPUT_STRIP_TRAILING_WHITESPACE
    )
    
    if(TORCH_PYTHON_PATH)
        get_filename_component(TORCH_ROOT "${TORCH_PYTHON_PATH}" DIRECTORY)
        set(TORCH_INCLUDE_DIRS "${TORCH_ROOT}/include")
        set(TORCH_LIBRARY_DIRS "${TORCH_ROOT}/lib")
        
        # Find PyTorch libraries
        find_library(TORCH_LIBRARY torch PATHS ${TORCH_LIBRARY_DIRS})
        find_library(TORCH_CPU_LIBRARY torch_cpu PATHS ${TORCH_LIBRARY_DIRS})
        find_library(TORCH_CUDA_LIBRARY torch_cuda PATHS ${TORCH_LIBRARY_DIRS})
        
        if(TORCH_LIBRARY)
            set(TORCH_LIBRARIES ${TORCH_LIBRARY})
            if(TORCH_CPU_LIBRARY)
                list(APPEND TORCH_LIBRARIES ${TORCH_CPU_LIBRARY})
            endif()
            if(TORCH_CUDA_LIBRARY)
                list(APPEND TORCH_LIBRARIES ${TORCH_CUDA_LIBRARY})
            endif()
            message(STATUS "PyTorch found via Python: ${TORCH_ROOT}")
        else()
            message(FATAL_ERROR "PyTorch libraries not found")
        endif()
    else()
        message(FATAL_ERROR "PyTorch not found")
    endif()
endif()

# PyTorch provides its own extension system - no need for separate PyBind11
# The torch/extension.h header includes everything needed for PyTorch extensions

# Set CUDA architecture for RTX 3070 Ti
set(CMAKE_CUDA_ARCHITECTURES "86")
set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -arch=sm_86")

# CUDA compilation flags
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --expt-relaxed-constexpr")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --use_fast_math")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xcompiler -fPIC")

# Optimization flags
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG")
set(CMAKE_CUDA_FLAGS_RELEASE "-O3 -DNDEBUG")

# Source files
set(SOURCES
    bindings/tft_bindings.cpp
    tft_cuda_impl.cpp
)

# Collect CUDA kernel files
file(GLOB_RECURSE CUDA_SOURCES
    "forward_pass/*.cu"
    "backward_pass/*.cu"
    "interpretability/*.cu"
)

if(CUDA_SOURCES)
    list(APPEND SOURCES ${CUDA_SOURCES})
    message(STATUS "Found CUDA sources: ${CUDA_SOURCES}")
else()
    message(STATUS "No CUDA source files found")
endif()

# Create the PyTorch extension module using PyTorch's native system
add_library(tft_cuda MODULE ${SOURCES})

# Set Python module properties for PyTorch extensions
set_target_properties(tft_cuda PROPERTIES
    PREFIX ""
    SUFFIX ".so"
    CXX_VISIBILITY_PRESET hidden
)

if(MSVC)
    set_target_properties(tft_cuda PROPERTIES SUFFIX ".pyd")
endif()

# PyTorch extension specific suffix
execute_process(
    COMMAND ${Python_EXECUTABLE} -c "import sysconfig; print(sysconfig.get_config_var('EXT_SUFFIX') or '.so')"
    OUTPUT_VARIABLE PYTHON_MODULE_SUFFIX
    OUTPUT_STRIP_TRAILING_WHITESPACE
)

if(PYTHON_MODULE_SUFFIX)
    set_target_properties(tft_cuda PROPERTIES SUFFIX "${PYTHON_MODULE_SUFFIX}")
endif()

# Include directories
target_include_directories(tft_cuda PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${TORCH_INCLUDE_DIRS}
    ${CUDA_INCLUDE_DIRS}
    ${Python_INCLUDE_DIRS}
)

# Link libraries
target_link_libraries(tft_cuda PRIVATE
    ${TORCH_LIBRARIES}
    ${Python_LIBRARIES}
    Threads::Threads
)

# Add CUDA libraries if available - only essential ones
if(CUDAToolkit_FOUND)
    target_link_libraries(tft_cuda PRIVATE
        CUDA::cudart
        CUDA::cublas
    )
elseif(CUDA_FOUND)
    target_link_libraries(tft_cuda PRIVATE
        ${CUDA_LIBRARIES}
        ${CUDA_CUBLAS_LIBRARIES}
    )
endif()

# Compilation properties
set_target_properties(tft_cuda PROPERTIES
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED ON
    CUDA_STANDARD 17
    CUDA_STANDARD_REQUIRED ON
    POSITION_INDEPENDENT_CODE ON
)

# Platform-specific settings
if(WIN32)
    target_compile_definitions(tft_cuda PRIVATE WIN32_LEAN_AND_MEAN)
endif()

message(STATUS "TFT-CUDA configuration complete")
message(STATUS "Sources: ${SOURCES}")
message(STATUS "PyTorch: ${TORCH_LIBRARIES}")
message(STATUS "CUDA: ${CUDA_TOOLKIT_ROOT_DIR}")