#pragma once

// PyTorch header compatibility wrapper for development versions
// This file contains workarounds for template instantiation issues in PyTorch 2.9.0.dev

// Include standard headers first
#include <new>
#include <memory>

// Ensure ABI compatibility
#define _GLIBCXX_USE_CXX11_ABI 1

// Disable specific warnings that cause issues with PyTorch dev headers
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
#pragma GCC diagnostic ignored "-Wunused-variable"
#pragma GCC diagnostic ignored "-Wunused-but-set-variable"
#pragma GCC diagnostic ignored "-Wmissing-field-initializers"
#pragma GCC diagnostic ignored "-Wtype-limits"
#pragma GCC diagnostic ignored "-Wplacement-new"
#pragma GCC diagnostic ignored "-Wmaybe-uninitialized"
#pragma GCC diagnostic ignored "-Wstringop-overflow"
#pragma GCC diagnostic ignored "-Wunused-parameter"
#pragma GCC diagnostic ignored "-Wignored-qualifiers"
#pragma GCC diagnostic ignored "-Wold-style-cast"
#pragma GCC diagnostic ignored "-Wsign-conversion"
#pragma GCC diagnostic ignored "-Wconversion"

// Define compatibility macros for PyTorch dev headers
#define _DISABLE_CONSTEXPR_MUTEX_CONSTRUCTOR
#define TORCH_API_INCLUDE_EXTENSION_H
#ifndef TORCH_EXTENSION_NAME
#define TORCH_EXTENSION_NAME tft_cuda
#endif

// Template compatibility fixes
#define TORCH_CHECK_ALWAYS_SHOW_CPP_STACKTRACE 0

// Include PyTorch headers with error suppression
#include <torch/extension.h>
#include <torch/torch.h>

// Re-enable warnings after PyTorch includes
#pragma GCC diagnostic pop

// Utility macros for safe tensor operations
#define SAFE_TENSOR_CHECK(tensor, name) \
    do { \
        if (!tensor.defined()) { \
            throw std::runtime_error(name " tensor is not defined"); \
        } \
        if (!tensor.is_contiguous()) { \
            tensor = tensor.contiguous(); \
        } \
    } while(0)

#define CUDA_TENSOR_CHECK(tensor, name) \
    do { \
        SAFE_TENSOR_CHECK(tensor, name); \
        if (!tensor.is_cuda()) { \
            throw std::runtime_error(name " tensor must be on CUDA device"); \
        } \
    } while(0)