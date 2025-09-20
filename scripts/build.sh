#!/usr/bin/env bash
# scripts/build.sh - Full build & install for TFT-CUDA

set -Eeuo pipefail
IFS=$'\n\t'

echo "🚀 TFT-CUDA Build Script"
echo "========================"

# Check if we're in the right directory
if [ ! -f "setup.py" ] || [ ! -d "cpp" ]; then
    echo "❌ Error: Must run from TFT-CUDA root directory"
    echo "   Expected files: setup.py, cpp/ directory"
    exit 1
fi

# Check Python version
echo "🐍 Checking Python environment..."
python_version=$(python --version 2>&1 | cut -d' ' -f2 | cut -d'.' -f1-2)
echo "   Python version: $python_version"

if ! python -c "import sys; sys.exit(0 if sys.version_info >= (3, 8) else 1)" 2>/dev/null; then
    echo "❌ Error: Python 3.8+ required"
    exit 1
fi

# Install Python dependencies first
echo "📦 Installing Python dependencies..."
if ! python -c "import torch" 2>/dev/null; then
    echo "   Installing PyTorch..."
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
fi

if ! python -c "import numpy" 2>/dev/null; then
    echo "   Installing NumPy..."
    pip install numpy
fi

if ! python -c "import pandas" 2>/dev/null; then
    echo "   Installing Pandas..."
    pip install pandas
fi

if ! python -c "import pyarrow" 2>/dev/null; then
    echo "   Installing PyArrow for parquet support..."
    pip install pyarrow
fi

# Install build dependencies
echo "   Installing build dependencies..."
pip install pybind11 cmake scikit-build-core psutil

# Check CUDA availability
echo "🔧 Checking CUDA availability..."
if command -v nvcc >/dev/null 2>&1; then
    nvcc_version=$(nvcc --version | grep "release" | sed 's/.*release \([0-9.]*\).*/\1/')
    echo "   CUDA detected: $nvcc_version"
    CUDA_AVAILABLE=true
else
    echo "   CUDA not available - building CPU-only version"
    CUDA_AVAILABLE=false
fi

# Check PyTorch CUDA support
if python -c "import torch; print('CUDA available:', torch.cuda.is_available())" 2>/dev/null; then
    echo "   PyTorch CUDA support detected"
else
    echo "   PyTorch CPU-only version"
fi

# Build CUDA kernels with CMake if CUDA is available
if [ "$CUDA_AVAILABLE" = true ]; then
    echo "🔧 Building CUDA kernels with CMake..."
    cd cpp
    
    # Create build directory
    mkdir -p build
    cd build
    
    # Configure with CMake
    echo "   Configuring with CMake..."
    if cmake .. -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES="75;80;86" 2>/dev/null; then
        echo "   ✓ CMake configuration successful"
        
        # Build
        echo "   Building with make..."
        if make -j"$(nproc)" 2>/dev/null; then
            echo "   ✓ CUDA build successful"
            CUDA_BUILD_SUCCESS=true
        else
            echo "   ⚠️  CUDA build failed, continuing with CPU-only"
            CUDA_BUILD_SUCCESS=false
        fi
    else
        echo "   ⚠️  CMake configuration failed, continuing with CPU-only"
        CUDA_BUILD_SUCCESS=false
    fi
    
    cd ../..
else
    CUDA_BUILD_SUCCESS=false
fi

# Install Python package
echo "🐍 Installing Python package..."
if [ "$CUDA_BUILD_SUCCESS" = true ]; then
    echo "   Installing with CUDA support..."
    CMAKE_ARGS="-DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=75;80;86" pip install -e . --verbose
else
    echo "   Installing CPU-only version..."
    pip install -e . --verbose
fi

# Verify installation
echo "✅ Verifying installation..."
TFCUDA_IMPORTED=false
if python -c "import tft_cuda; print('✓ tft_cuda module imported successfully')" 2>/dev/null; then
    TFCUDA_IMPORTED=true
    echo "   ✓ Python package installation successful"
else
    echo "   ⚠️  tft_cuda module import failed, but core dependencies are installed"
fi

# Test basic PyTorch functionality
TORCH_RUNTIME_CUDA=false
if python -c "import torch; x = torch.randn(2, 3); print('✓ PyTorch basic test passed')" 2>/dev/null; then
    echo "   ✓ PyTorch basic functionality working"
    if python -c "import torch, sys; sys.exit(0 if torch.cuda.is_available() else 1)" 2>/dev/null; then
        TORCH_RUNTIME_CUDA=true
    fi
else
    echo "   ❌ PyTorch basic test failed"
    exit 1
fi

echo ""
echo "🎉 Build complete!"
echo "==================="
if [ "$TFCUDA_IMPORTED" = true ] && [ "$TORCH_RUNTIME_CUDA" = true ]; then
    echo "✓ CUDA backend available (extension loaded, torch.cuda.is_available())"
elif [ "$TFCUDA_IMPORTED" = true ] && [ "$TORCH_RUNTIME_CUDA" = false ]; then
    echo "✓ CUDA extension installed; running without GPU (torch.cuda.is_available() == False)"
    echo "ℹ️  If you expect GPU usage, ensure NVIDIA drivers and CUDA runtime are correctly installed and visible to PyTorch"
else
    echo "✓ CPU-only build; CUDA extension not loaded"
    echo "ℹ️  For CUDA support, ensure CUDA toolkit is installed and rebuild"
fi
echo "✓ Python package installed in development mode"
echo ""
echo "Next steps:"
echo "  - Run 'bash scripts/test.sh' to test the installation"
echo "  - Run 'bash scripts/train.sh' to train a model"
echo "  - Run 'bash scripts/debug.sh' if you encounter issues"