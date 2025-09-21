#!/usr/bin/env python3
"""
Debug TFT tensor shape issues
"""

import sys
sys.path.insert(0, "python")

import torch
import numpy as np
from tft_model import TemporalFusionTransformer

def debug_tft_shapes():
    """Debug TFT model tensor shapes to identify mismatch."""
    
    print("🔍 Debugging TFT tensor shapes...")
    
    # Sample data dimensions (matching your actual data)
    batch_size = 64
    seq_len = 128
    input_size = 134
    
    # Create sample input data
    X_sample = torch.randn(batch_size, seq_len, input_size)
    
    # Fixed model configuration
    model_config = {
        'input_size': input_size,
        'output_size': 1,
        'hidden_size': 512,
        'num_heads': 4,
        'num_encoder_layers': 3,
        'num_decoder_layers': 3,
        'dropout_rate': 0.2,
        'sequence_length': seq_len,
        'quantile_levels': [0.5],    # Single quantile
        'prediction_horizon': [1],   # Single horizon
        'num_historical_features': input_size,
        'num_future_features': 10,
        'static_input_size': 10
    }
    
    print(f"📊 Input shape: {X_sample.shape}")
    print(f"🔧 Model config: {model_config}")
    
    try:
        # Initialize model
        model = TemporalFusionTransformer(model_config)
        print("✅ Model initialized successfully")
        
        # Test forward pass
        print("🧪 Testing forward pass...")
        
        # Create TFT-style input dict
        inputs = {
            'historical_features': X_sample,
            'future_features': torch.zeros(batch_size, 1, input_size),  # Single prediction horizon
            'static_features': torch.zeros(batch_size, 10)
        }
        
        with torch.no_grad():
            outputs = model(inputs)
            print(f"✅ Forward pass successful!")
            print(f"📤 Output type: {type(outputs)}")
            if isinstance(outputs, dict):
                for key, value in outputs.items():
                    if isinstance(value, torch.Tensor):
                        print(f"   {key}: {value.shape}")
                    elif isinstance(value, dict):
                        print(f"   {key}: dict with keys {list(value.keys())}")
                        for sub_key, sub_value in value.items():
                            if isinstance(sub_value, torch.Tensor):
                                print(f"     {sub_key}: {sub_value.shape}")
    
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True

if __name__ == "__main__":
    debug_tft_shapes()