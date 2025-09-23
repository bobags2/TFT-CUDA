"""
TFT-CUDA: Temporal Fusion Transformer with CUDA acceleration for financial forecasting.

This package provides a high-performance implementation of the Temporal Fusion Transformer
optimized for financial time-series forecasting with CUDA acceleration.
"""

__version__ = "0.1.0"
__author__ = "TFT-CUDA Team"
__email__ = "contact@tft-cuda.com"

# Core imports
from .tft_model import (
    TemporalFusionTransformer,
    TFTLoss,
    GatedResidualNetwork,
    VariableSelectionNetwork,
    MultiHeadAttention,
    create_tft_config
)

from .data import (
    FinancialDataset,
    TFTDataset
)

from .loss import (
    QuantileLoss,
    HuberQuantileLoss,
    QuantileSmoothingLoss,
    MultiHorizonLoss,
    FinancialMetricLoss,
    AdaptiveQuantileLoss,
    create_tft_loss
)

# Trainer utilities are not packaged here; import from your training script if needed.

from .interpretability import (
    TFTInterpretability
)

# Try to import CUDA backend
try:
    import tft_cuda_ext as tft_cuda
    CUDA_AVAILABLE = True
    print("TFT-CUDA: CUDA backend loaded successfully")
except ImportError:
    CUDA_AVAILABLE = False
    print("TFT-CUDA: Running in CPU-only mode. Install CUDA backend for acceleration.")

# Package metadata
__all__ = [
    # Core model components
    "TemporalFusionTransformer",
    "TFTLoss", 
    "GatedResidualNetwork",
    "VariableSelectionNetwork",
    "MultiHeadAttention",
    "create_tft_config",
    
    # Data handling
    "FinancialDataset",
    "TFTDataset",
    
    # Loss functions
    "QuantileLoss",
    "HuberQuantileLoss", 
    "QuantileSmoothingLoss",
    "MultiHorizonLoss",
    "FinancialMetricLoss",
    "AdaptiveQuantileLoss",
    "create_tft_loss",
    
    # Training infrastructure (provided by user code)
    
    # Interpretability
    "TFTInterpretability",
    
    # Package info
    "CUDA_AVAILABLE",
]

# Version info
def get_version():
    """Get package version."""
    return __version__

def get_cuda_info():
    """Get CUDA availability and version info."""
    info = {
        "cuda_available": CUDA_AVAILABLE,
        "package_version": __version__
    }
    
    if CUDA_AVAILABLE:
        try:
            import torch
            info.update({
                "torch_cuda_available": torch.cuda.is_available(),
                "torch_version": torch.__version__,
                "cuda_device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0
            })
            
            if torch.cuda.is_available():
                info["cuda_device_name"] = torch.cuda.get_device_name(0)
                info["cuda_capability"] = torch.cuda.get_device_capability(0)
        except Exception as e:
            info["error"] = str(e)
    
    return info

def print_system_info():
    """Print system and package information."""
    info = get_cuda_info()
    
    print("TFT-CUDA System Information")
    print("=" * 30)
    print(f"Package Version: {info['package_version']}")
    print(f"CUDA Backend: {'Available' if info['cuda_available'] else 'Not Available'}")
    
    if 'torch_version' in info:
        print(f"PyTorch Version: {info['torch_version']}")
        print(f"PyTorch CUDA: {'Available' if info.get('torch_cuda_available', False) else 'Not Available'}")
        
        if info.get('cuda_device_count', 0) > 0:
            print(f"CUDA Devices: {info['cuda_device_count']}")
            print(f"Primary Device: {info.get('cuda_device_name', 'Unknown')}")
            print(f"Compute Capability: {info.get('cuda_capability', 'Unknown')}")
    
    if 'error' in info:
        print(f"Error: {info['error']}")

# Configuration defaults
DEFAULT_CONFIG = {
    "model": {
        "hidden_size": 256,
        "num_heads": 8,
        "quantile_levels": [0.1, 0.5, 0.9],
        "prediction_horizon": [1, 5, 10],
        "sequence_length": 100,
        "dropout_rate": 0.1
    },
    "training": {
        "optimizer": "adamw",
    "learning_rate": 9.38e-06,
        "weight_decay": 9.38e-08,
        "batch_size": 64,
        "epochs": 100,
        "early_stopping": True,
        "patience": 15
    },
    "data": {
        "assets": ["es", "vx", "zn"],
        "lookback_window": 100,
        "robust_scaling": True,
        "handle_missing": "forward_fill"
    }
}

def get_default_config():
    """Get default configuration for TFT models."""
    return DEFAULT_CONFIG.copy()

# Quick start function
def quick_start_example():
    """Print a quick start example."""
    example_code = '''
# TFT-CUDA Quick Start Example

import tft_cuda

# 1. Load and process financial data
dataset = tft_cuda.FinancialDataset(data_dir="data/")
processed_data = dataset.process_pipeline()

# 2. Create sequences for training
X, y = dataset.create_sequences(processed_data, sequence_length=100)
X_train, X_val, X_test, y_train, y_val, y_test = dataset.train_val_test_split(X, y)

# 3. Create TFT model
config = tft_cuda.create_tft_config(input_size=X.shape[-1])
model = tft_cuda.TemporalFusionTransformer(config)

# 4. Setup training
training_config = tft_cuda.create_training_config()
trainer = tft_cuda.TFTTrainer(model, training_config)

# 5. Train the model
from torch.utils.data import DataLoader
train_dataset = tft_cuda.TFTDataset(X_train, y_train)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

history = trainer.train(train_loader, epochs=50)

# 6. Generate predictions
sample_inputs = {
    'historical_features': torch.tensor(X_test[:1]),
    'static_features': torch.zeros(1, 10)
}
predictions = model.predict(sample_inputs)

print("Training complete! Check predictions:", predictions)
'''
    print(example_code)

if __name__ == "__main__":
    print_system_info()
    print("\n")
    quick_start_example()