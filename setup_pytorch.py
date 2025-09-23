"""
Setup script for TFT-CUDA: uses PyTorch's native extension system.
Eliminates PyBind11 dependency for better compatibility with PyTorch dev versions.
"""

from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension
import torch
from pathlib import Path


def cuda_available() -> bool:
    try:
        return torch.cuda.is_available()
    except Exception:
        return False


def get_cuda_arch() -> str:
    try:
        if torch.cuda.is_available():
            major, minor = torch.cuda.get_device_capability(0)
            return f"{major}{minor}"
    except Exception:
        pass
    return "86"


ext_modules = []

if cuda_available():
    cuda_include_dirs = [
        "/usr/local/cuda/include",
        "cpp",
        "cpp/forward_pass",
        "cpp/backward_pass",
        "cpp/interpretability",
    ]

    cuda_sources = [
        "cpp/bindings/tft_bindings.cpp",
        "cpp/tft_cuda_impl.cu",
        "cpp/forward_pass/lstm.cu",
        "cpp/forward_pass/mha.cu",
        "cpp/forward_pass/linear_fwd.cu",
        "cpp/forward_pass/quantile_heads.cu",
        "cpp/forward_pass/static_encoder.cu",
        "cpp/backward_pass/lstm_backward.cu",
        "cpp/backward_pass/mha_backward.cu",
        "cpp/backward_pass/linear_bwd.cu",
        "cpp/backward_pass/quantile_heads_backward.cu",
        "cpp/backward_pass/layer_norm_backward.cu",
        "cpp/interpretability/attention_aggregate.cu",
        "cpp/interpretability/vsn_aggregate.cu",
        "cpp/interpretability/static_embedding_importance.cu",
    ]

    existing_sources = [src for src in cuda_sources if Path(src).exists()]
    if existing_sources:
        ext_modules.append(
            CUDAExtension(
                name="tft_cuda",
                sources=existing_sources,
                include_dirs=cuda_include_dirs,
                extra_compile_args={
                    "cxx": ["-O3", "-Wall", "-std=c++17", "-fPIC"],
                    "nvcc": [
                        "-O3",
                        "--use_fast_math",
                        "--expt-relaxed-constexpr",
                        f"-arch=sm_{get_cuda_arch()}",
                        "-Xcompiler",
                        "-fPIC",
                    ],
                },
                # Rely on PyTorch/conda for CUDA/Torch linkage
            )
        )
    else:
        print("Warning: CUDA source files not found. Building without CUDA support.")
else:
    print("Warning: CUDA not available. Building without CUDA support.")

setup(
    name="tft_cuda",
    ext_modules=ext_modules,
    cmdclass={"build_ext": BuildExtension},
)