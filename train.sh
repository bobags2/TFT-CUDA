#!/usr/bin/env bash
# THE DEFINITIVE TFT TRAINING SCRIPT
# ==================================
# One script to rule them all!

set -Eeuo pipefail
IFS=$'\n\t'

echo "🚀 THE DEFINITIVE TFT-CUDA TRAINING SCRIPT"
echo "=========================================="
echo "🔧 4 Multi-Head Attention + FP32 Precision"
echo "🔧 Gradient Accumulation + Stability Monitoring"
echo "🔧 Proper VSN Configuration (134 features)"
echo ""

# Check environment
if ! command -v python &> /dev/null; then
    echo "❌ Python not found in PATH"
    exit 1
fi

# Check conda environment
if [[ "${CONDA_DEFAULT_ENV:-}" != "rapids-25.08" ]]; then
    echo "⚠️  Not in rapids-25.08 environment"
    echo "💡 Run: conda activate rapids-25.08"
    # Don't exit, let user decide
fi

# Check CUDA
if command -v nvidia-smi &> /dev/null; then
    echo "✅ NVIDIA GPU detected:"
    nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader,nounits | head -1
    echo ""
else
    echo "⚠️  No NVIDIA GPU detected"
fi

# Check data files
echo "📊 Checking data files..."
required_files=(
    "data/X_train_temporal.npy"
    "data/y_train_temporal.npy" 
    "data/X_val_temporal.npy"
    "data/y_val_temporal.npy"
)

for file in "${required_files[@]}"; do
    if [[ -f "$file" ]]; then
        echo "✅ $file"
    else
        echo "❌ Missing: $file"
        exit 1
    fi
done

# Check checkpoints directory
if [[ ! -d "checkpoints" ]]; then
    echo "📁 Creating checkpoints directory..."
    mkdir -p checkpoints
fi

echo ""
echo "🏋️ Starting THE definitive TFT training..."
echo "=========================================="

# Run the training with proper error handling
python train.py || {
    echo ""
    echo "❌ Training failed!"
    echo "💡 Check the error messages above"
    echo "💡 Common issues:"
    echo "   - Out of GPU memory (reduce batch size)"
    echo "   - Missing dependencies (pip install requirements)"
    echo "   - Data file corruption (re-download data)"
    exit 1
}

echo ""
echo "✅ Training completed successfully!"
echo "🏆 Check checkpoints/tft_best.pth for the best model"