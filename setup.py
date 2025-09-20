"""
Setup script for TFT-CUDA: Temporal Fusion Transformer with CUDA acceleration.
"""

from setuptools import setup, find_packages, Extension
from pybind11.setup_helpers import Pybind11Extension, build_ext
from pybind11 import get_cmake_dir
import pybind11
from pathlib import Path
import os
import sys

# Read README
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text() if (this_directory / "README.md").exists() else ""

# Version
VERSION = "0.1.0"

# CUDA availability check
def cuda_available():
    """Check if CUDA is available for compilation."""
    try:
        import torch
        return torch.cuda.is_available()
    except ImportError:
        return False

def get_cuda_arch():
    """Get CUDA architecture for compilation."""
    try:
        import torch
        if torch.cuda.is_available():
            # Get the compute capability of the first GPU
            major, minor = torch.cuda.get_device_capability(0)
            return f"{major}{minor}"
        return "86"  # Default to compute_86 (RTX 30xx)
    except:
        return "86"

# CUDA extension setup
cuda_extensions = []
if cuda_available():
    print("CUDA detected. Building with CUDA support.")
    
    cuda_arch = get_cuda_arch()
    cuda_include_dirs = [
        "cpp/",
        "cpp/forward_pass/",
        "cpp/backward_pass/", 
        "cpp/interpretability/",
        "cpp/wrappers/",
        "/usr/local/cuda/include",
    ]
    
    # CUDA source files
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
    
    # Check which source files exist
    existing_sources = [src for src in cuda_sources if Path(src).exists()]
    
    if existing_sources:
        cuda_extension = Pybind11Extension(
            "tft_cuda",
            sources=existing_sources,
            include_dirs=cuda_include_dirs + [pybind11.get_include()],
            language="c++",
            cxx_std=17,
            define_macros=[("VERSION_INFO", f'"{VERSION}"')],
            extra_compile_args={
                "cxx": ["-O3", "-Wall", "-shared", "-std=c++17", "-fPIC"],
                "nvcc": [
                    "-O3",
                    "-std=c++17",
                    f"-gencode=arch=compute_{cuda_arch},code=sm_{cuda_arch}",
                    "-use_fast_math",
                    "-Xptxas", "-v",
                    "--expt-relaxed-constexpr",
                    "-Xcompiler", "-fPIC"
                ],
            },
            libraries=["cuda", "cudart", "cublas", "curand"],
            library_dirs=["/usr/local/cuda/lib64"],
        )
        cuda_extensions.append(cuda_extension)
    else:
        print("Warning: CUDA source files not found. Building without CUDA support.")
else:
    print("CUDA not available. Building without CUDA support.")

# Python dependencies
install_requires = [
    "torch>=1.13.0",
    "numpy>=1.21.0",
    "pandas>=1.3.0",
    "scipy>=1.7.0",
    "matplotlib>=3.5.0",
    "seaborn>=0.11.0",
    "scikit-learn>=1.0.0",
    "tqdm>=4.62.0",
    "tensorboard>=2.8.0",
]

# Optional dependencies
extras_require = {
    "dev": [
        "pytest>=6.0.0",
        "pytest-cov>=2.12.0",
        "black>=21.0.0",
        "flake8>=3.9.0",
        "mypy>=0.910",
        "pre-commit>=2.15.0",
    ],
    "viz": [
        "plotly>=5.0.0",
        "dash>=2.0.0",
        "ipywidgets>=7.6.0",
    ],
    "wandb": [
        "wandb>=0.12.0",
    ],
    "all": [
        "plotly>=5.0.0",
        "dash>=2.0.0", 
        "ipywidgets>=7.6.0",
        "wandb>=0.12.0",
        "pytest>=6.0.0",
        "pytest-cov>=2.12.0",
        "black>=21.0.0",
        "flake8>=3.9.0",
        "mypy>=0.910",
        "pre-commit>=2.15.0",
    ]
}

setup(
    name="tft-cuda",
    version=VERSION,
    author="TFT-CUDA Team",
    author_email="contact@tft-cuda.com",
    description="Temporal Fusion Transformer with CUDA acceleration for financial forecasting",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/bobags2/TFT-CUDA",
    packages=find_packages(where="python"),
    package_dir={"": "python"},
    ext_modules=cuda_extensions,
    cmdclass={"build_ext": build_ext},
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: POSIX :: Linux",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: C++",
        "Programming Language :: CUDA",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Office/Business :: Financial",
    ],
    python_requires=">=3.8",
    install_requires=install_requires,
    extras_require=extras_require,
    include_package_data=True,
    package_data={
        "": ["*.json", "*.yaml", "*.yml", "*.md"],
    },
    entry_points={
        "console_scripts": [
            "tft-train=tft_cuda.scripts.train:main",
            "tft-predict=tft_cuda.scripts.predict:main",
            "tft-backtest=tft_cuda.scripts.backtest:main",
        ],
    },
    project_urls={
        "Bug Reports": "https://github.com/bobags2/TFT-CUDA/issues",
        "Source": "https://github.com/bobags2/TFT-CUDA",
        "Documentation": "https://tft-cuda.readthedocs.io/",
    },
)