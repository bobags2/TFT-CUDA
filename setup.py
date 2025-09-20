"""
Setup script for TFT-CUDA: relies on pyproject.toml for metadata.
This file only wires up the optional CUDA extension build via pybind11.
"""

from setuptools import setup
from pybind11.setup_helpers import Pybind11Extension, build_ext
import pybind11
from pathlib import Path


def cuda_available() -> bool:
    try:
        import torch  # noqa: F401
        return torch.cuda.is_available()
    except Exception:
        return False


def get_cuda_arch() -> str:
    try:
        import torch

        if torch.cuda.is_available():
            major, minor = torch.cuda.get_device_capability(0)
            return f"{major}{minor}"
    except Exception:
        pass
    return "86"


ext_modules = []
if cuda_available():
    print("CUDA detected. Building with CUDA support.")

    cuda_arch = get_cuda_arch()
    cuda_include_dirs = [
        "cpp/",
        "cpp/forward_pass/",
        "cpp/backward_pass/",
        "cpp/interpretability/",
        "/usr/local/cuda/include",
    ]

    cuda_sources = [
        "cpp/bindings/tft_bindings.cpp",
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
            Pybind11Extension(
                "tft_cuda",
                sources=existing_sources,
                include_dirs=cuda_include_dirs + [pybind11.get_include()],
                language="c++",
                cxx_std=17,
                extra_compile_args={
                    "cxx": ["-O3", "-Wall", "-std=c++17", "-fPIC"],
                    "nvcc": [
                        "-O3",
                        "-std=c++17",
                        f"-gencode=arch=compute_{cuda_arch},code=sm_{cuda_arch}",
                        "-use_fast_math",
                        "-Xptxas",
                        "-v",
                        "--expt-relaxed-constexpr",
                        "-Xcompiler",
                        "-fPIC",
                    ],
                },
                libraries=["cuda", "cudart", "cublas", "curand"],
                library_dirs=["/usr/local/cuda/lib64"],
            )
        )
    else:
        print("Warning: CUDA source files not found. Building without CUDA support.")
else:
    print("CUDA not available. Building without CUDA support.")


setup(
    ext_modules=ext_modules,
    cmdclass={"build_ext": build_ext},
)