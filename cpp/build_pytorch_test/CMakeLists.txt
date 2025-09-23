cmake_minimum_required(VERSION 3.18)

# Force GCC-11
set(CMAKE_C_COMPILER "/usr/bin/gcc-11")
set(CMAKE_CXX_COMPILER "/usr/bin/g++-11")

project(TEST_PYTORCH_HEADERS LANGUAGES CXX)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find Python and PyTorch
find_package(Python COMPONENTS Interpreter Development REQUIRED)

# Find PyTorch
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
    execute_process(
        COMMAND ${Python_EXECUTABLE} -c "import torch; print(torch.__file__)"
        OUTPUT_VARIABLE TORCH_PYTHON_PATH
        OUTPUT_STRIP_TRAILING_WHITESPACE
    )
    
    if(TORCH_PYTHON_PATH)
        get_filename_component(TORCH_ROOT "${TORCH_PYTHON_PATH}" DIRECTORY)
        set(TORCH_INCLUDE_DIRS "${TORCH_ROOT}/include")
        set(TORCH_LIBRARY_DIRS "${TORCH_ROOT}/lib")
        
        find_library(TORCH_LIBRARY torch PATHS ${TORCH_LIBRARY_DIRS})
        if(TORCH_LIBRARY)
            set(TORCH_LIBRARIES ${TORCH_LIBRARY})
            message(STATUS "PyTorch found via Python: ${TORCH_ROOT}")
        else()
            message(FATAL_ERROR "PyTorch libraries not found")
        endif()
    else()
        message(FATAL_ERROR "PyTorch not found")
    endif()
endif()

# Create test executable
add_executable(test_pytorch_headers test_pytorch_headers.cpp)

# Include directories
target_include_directories(test_pytorch_headers PRIVATE
    ${TORCH_INCLUDE_DIRS}
    ${Python_INCLUDE_DIRS}
)

# Link libraries
target_link_libraries(test_pytorch_headers PRIVATE
    ${TORCH_LIBRARIES}
    ${Python_LIBRARIES}
)

# Compilation properties
set_target_properties(test_pytorch_headers PROPERTIES
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED ON
)