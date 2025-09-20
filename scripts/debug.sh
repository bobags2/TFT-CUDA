#!/usr/bin/env bash
# scripts/debug.sh - Debug common issues

set +e  # Don't exit on errors - we want to diagnose them
IFS=$'\n\t'

echo "🔍 TFT-CUDA Debug Script"
echo "========================"

# Function to check status
check_status() {
    if [ $1 -eq 0 ]; then
        echo "   ✅ $2"
    else
        echo "   ❌ $2"
    fi
}

# System information
echo "🖥️  System Information"
echo "----------------------"
echo "   OS: $(uname -s) $(uname -r)"
echo "   Architecture: $(uname -m)"
echo "   Python: $(python --version 2>&1)"
echo "   Working directory: $(pwd)"

# Check if we're in the right directory
echo ""
echo "📁 Directory Structure Check"
echo "----------------------------"
[ -f "setup.py" ] && echo "   ✅ setup.py found" || echo "   ❌ setup.py missing"
[ -d "cpp" ] && echo "   ✅ cpp/ directory found" || echo "   ❌ cpp/ directory missing"
[ -d "python" ] && echo "   ✅ python/ directory found" || echo "   ❌ python/ directory missing"
[ -d "tests" ] && echo "   ✅ tests/ directory found" || echo "   ❌ tests/ directory missing"
[ -d "config" ] && echo "   ✅ config/ directory found" || echo "   ❌ config/ directory missing"
[ -d "data" ] && echo "   ✅ data/ directory found" || echo "   ❌ data/ directory missing"

# CUDA availability check
echo ""
echo "🔧 CUDA Availability Check"
echo "--------------------------"
if command -v nvcc >/dev/null 2>&1; then
    nvcc_version=$(nvcc --version | grep "release" | sed 's/.*release \([0-9.]*\).*/\1/' 2>/dev/null || echo "unknown")
    echo "   ✅ CUDA toolkit found: $nvcc_version"
    
    echo "   CUDA devices:"
    if command -v nvidia-smi >/dev/null 2>&1; then
        nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader,nounits 2>/dev/null | \
        while IFS=, read -r idx name memory; do
            echo "     GPU $idx: $name (${memory}MB)"
        done
    else
        echo "     nvidia-smi not available"
    fi
else
    echo "   ❌ CUDA toolkit not found"
    echo "     Install CUDA toolkit for GPU acceleration"
fi

# Python environment check
echo ""
echo "🐍 Python Environment Check"
echo "---------------------------"
python -c "
import sys
print(f'   Python version: {sys.version}')
print(f'   Python executable: {sys.executable}')
print(f'   Python path: {sys.path[0:3]}...')

# Check critical packages
packages = ['torch', 'numpy', 'pandas', 'json', 'pathlib']
for pkg in packages:
    try:
        __import__(pkg)
        print(f'   ✅ {pkg} available')
    except ImportError:
        print(f'   ❌ {pkg} missing')
"

# PyTorch specific checks
echo ""
echo "🔥 PyTorch Check"
echo "---------------"
python -c "
try:
    import torch
    print(f'   PyTorch version: {torch.__version__}')
    print(f'   CUDA available: {torch.cuda.is_available()}')
    if torch.cuda.is_available():
        print(f'   CUDA devices: {torch.cuda.device_count()}')
        print(f'   Current device: {torch.cuda.current_device()}')
        print(f'   Device name: {torch.cuda.get_device_name()}')
    else:
        print('   Running in CPU mode')
    
    # Basic tensor test
    x = torch.randn(2, 3)
    print(f'   ✅ Basic tensor creation works: {x.shape}')
    
except Exception as e:
    print(f'   ❌ PyTorch error: {e}')
"

# Package import check
echo ""
echo "📦 TFT-CUDA Package Check"
echo "-------------------------"
python -c "
import sys
sys.path.insert(0, 'python')

modules = ['data', 'tft_model', 'trainer', 'loss', 'interpretability']
for module in modules:
    try:
        __import__(module)
        print(f'   ✅ {module} imported successfully')
    except Exception as e:
        print(f'   ❌ {module} import failed: {e}')

# Try importing the compiled extension
try:
    import tft_cuda
    print('   ✅ tft_cuda extension imported successfully')
except Exception as e:
    print(f'   ⚠️  tft_cuda extension import failed: {e}')
    print('      This is expected if not built with CUDA support')
"

# Configuration check
echo ""
echo "⚙️  Configuration Check"
echo "-----------------------"
python -c "
import json
from pathlib import Path

config_files = ['config/default_config.json', 'config/ensemble.json']
for config_file in config_files:
    path = Path(config_file)
    if path.exists():
        try:
            with open(path, 'r') as f:
                config = json.load(f)
            print(f'   ✅ {config_file} loaded successfully')
        except Exception as e:
            print(f'   ❌ {config_file} invalid JSON: {e}')
    else:
        print(f'   ⚠️  {config_file} not found')
"

# Data files check
echo ""
echo "📊 Data Files Check"
echo "------------------"
if [ -d "data" ]; then
    echo "   Data directory contents:"
    ls -la data/ | while read -r line; do
        echo "     $line"
    done
    
    # Check for expected data files
    data_patterns=("*es10m*.csv" "*vx10m*.csv" "*zn10m*.csv" "*.parquet" "*.npy")
    for pattern in "${data_patterns[@]}"; do
        if ls data/$pattern >/dev/null 2>&1; then
            files=$(ls data/$pattern 2>/dev/null | wc -l)
            echo "   ✅ Found $files file(s) matching $pattern"
        else
            echo "   ⚠️  No files matching $pattern"
        fi
    done
else
    echo "   ❌ Data directory not found"
fi

# Build check
echo ""
echo "🏗️  Build Check"
echo "---------------"
if [ -d "cpp/build" ]; then
    echo "   ✅ Build directory exists"
    if [ -f "cpp/build/Makefile" ]; then
        echo "   ✅ CMake build files found"
    else
        echo "   ⚠️  No CMake build files found"
    fi
else
    echo "   ⚠️  No build directory found"
    echo "      Run 'bash scripts/build.sh' to build the project"
fi

# Test minimal forward pass
echo ""
echo "🧪 Minimal Forward Pass Test"
echo "----------------------------"
python -c "
import sys
import warnings
warnings.filterwarnings('ignore')
sys.path.insert(0, 'python')

try:
    import torch
    import numpy as np
    
    print('   Creating test tensors...')
    x = torch.randn(1, 100, 64)
    print(f'   ✅ Test input created: {x.shape}')
    
    # Try TFT model
    try:
        from tft_model import TemporalFusionTransformer, create_tft_config
        
        config = create_tft_config(input_size=64, hidden_size=32)
        model = TemporalFusionTransformer(config)
        
        sample_input = {
            'historical_features': x,
            'static_features': torch.randn(1, config['static_input_size'])
        }
        
        with torch.no_grad():
            output = model(sample_input)
        
        print('   ✅ TFT forward pass succeeded')
        print(f'      Output keys: {list(output.keys())}')
        
    except Exception as e:
        print(f'   ⚠️  TFT forward pass failed: {e}')
        
        # Try simple model
        try:
            import torch.nn as nn
            simple_model = nn.Linear(64, 1)
            output = simple_model(x[:, -1, :])
            print(f'   ✅ Simple model forward pass succeeded: {output.shape}')
        except Exception as e2:
            print(f'   ❌ Simple model forward pass failed: {e2}')

except Exception as e:
    print(f'   ❌ Forward pass test completely failed: {e}')
"

# Memory and performance check
echo ""
echo "💾 Memory and Performance Check"
echo "------------------------------"
python -c "
import sys
import psutil
import torch

# System memory
mem = psutil.virtual_memory()
print(f'   System memory: {mem.total / (1024**3):.1f} GB total, {mem.available / (1024**3):.1f} GB available')

# GPU memory (if available)
if torch.cuda.is_available():
    for i in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(i)
        total_mem = props.total_memory / (1024**3)
        reserved = torch.cuda.memory_reserved(i) / (1024**3)
        allocated = torch.cuda.memory_allocated(i) / (1024**3)
        print(f'   GPU {i} memory: {total_mem:.1f} GB total, {allocated:.2f} GB allocated, {reserved:.2f} GB reserved')

# CPU info
print(f'   CPU cores: {psutil.cpu_count()} logical, {psutil.cpu_count(logical=False)} physical')
print(f'   CPU usage: {psutil.cpu_percent(interval=1):.1f}%')
"

# Common issues and solutions
echo ""
echo "🔧 Common Issues and Solutions"
echo "-----------------------------"
echo "   Common problems and fixes:"
echo ""
echo "   📋 Import Errors:"
echo "      → Run 'pip install torch numpy pandas' for missing packages"
echo "      → Run 'bash scripts/build.sh' to install the package"
echo ""
echo "   🔧 CUDA Issues:"
echo "      → Install CUDA toolkit: https://developer.nvidia.com/cuda-downloads"
echo "      → Check CUDA compatibility with PyTorch version"
echo "      → Use CPU-only mode if CUDA not needed"
echo ""
echo "   💾 Memory Issues:"
echo "      → Reduce batch_size in training config"
echo "      → Use gradient accumulation for large batches"
echo "      → Close other GPU applications"
echo ""
echo "   📊 Data Issues:"
echo "      → Place CSV files in data/ directory"
echo "      → Check file format matches expected columns"
echo "      → Use synthetic data for testing (automatic fallback)"
echo ""
echo "   🏗️  Build Issues:"
echo "      → Install CMake: 'sudo apt-get install cmake' (Linux)"
echo "      → Install pybind11: 'pip install pybind11'"
echo "      → Check C++ compiler is available"

echo ""
echo "🎯 Debug Summary"
echo "==============="
echo "✓ System information collected"
echo "✓ Environment checked"
echo "✓ Common issues diagnosed"
echo ""
echo "Next steps:"
echo "  - Address any ❌ issues shown above"
echo "  - Run 'bash scripts/build.sh' if build issues found"
echo "  - Run 'bash scripts/test.sh' to verify fixes"
echo "  - Check documentation for detailed troubleshooting"