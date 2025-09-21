#!/usr/bin/env bash
# Run TFT Model Evaluation

set -e

echo "🚀 TFT MODEL EVALUATION SUITE"
echo "=============================="
echo ""

# Check for required files
if [ ! -f "checkpoints/tft_best.pth" ]; then
    echo "❌ Model checkpoint not found: checkpoints/tft_best.pth"
    echo "   Please train the model first with: bash train.sh"
    exit 1
fi

# Check for data files
if [ ! -f "data/X_val.npy" ] && [ ! -f "data/X_val_temporal.npy" ]; then
    echo "❌ Validation data not found"
    echo "   Please prepare data first"
    exit 1
fi

# Create results directory
mkdir -p evaluation_results

# Run evaluation
echo "📊 Running comprehensive evaluation..."
echo ""
python evaluate_tft.py

echo ""
echo "✅ Evaluation complete!"
echo "📁 Check evaluation_results/ for detailed reports and visualizations"
