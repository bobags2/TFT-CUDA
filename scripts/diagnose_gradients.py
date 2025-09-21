#!/usr/bin/env python3
"""
Comprehensive gradient explosion diagnostic tool for TFT model.
Investigates data, model architecture, and training dynamics.
"""

import sys
sys.path.insert(0, 'python')

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
import warnings
warnings.filterwarnings('ignore')

def check_data_statistics():
    """Check data quality and normalization results."""
    print("🔍 Data Statistics Investigation")
    print("=" * 50)
    
    # Load processed data
    try:
        X_train = np.load('data/X_train.npy')
        y_train = np.load('data/y_train.npy')
        X_val = np.load('data/X_val.npy')
        y_val = np.load('data/y_val.npy')
        
        print(f"Data shapes: X_train{X_train.shape}, y_train{y_train.shape}")
        
        # Check for problematic values
        def analyze_array(arr, name):
            print(f"\n{name} Analysis:")
            print(f"  Shape: {arr.shape}")
            print(f"  Min: {np.nanmin(arr):.6f}")
            print(f"  Max: {np.nanmax(arr):.6f}")
            print(f"  Mean: {np.nanmean(arr):.6f}")
            print(f"  Std: {np.nanstd(arr):.6f}")
            print(f"  NaN count: {np.isnan(arr).sum()}")
            print(f"  Inf count: {np.isinf(arr).sum()}")
            print(f"  Zero count: {(arr == 0).sum()}")
            
            # Check for extreme values
            extreme_threshold = 1000
            extreme_count = np.sum(np.abs(arr) > extreme_threshold)
            print(f"  Values > {extreme_threshold}: {extreme_count}")
            
            # Check distribution
            finite_vals = arr[np.isfinite(arr)]
            if len(finite_vals) > 0:
                percentiles = np.percentile(finite_vals, [1, 5, 95, 99])
                print(f"  Percentiles [1%, 5%, 95%, 99%]: {percentiles}")
        
        analyze_array(X_train, "X_train")
        analyze_array(y_train, "y_train")
        
        # Check for correlated features that could cause issues
        print(f"\nFeature Correlation Analysis:")
        X_flat = X_train.reshape(-1, X_train.shape[-1])
        corr_matrix = np.corrcoef(X_flat.T)
        high_corr = np.where(np.abs(corr_matrix) > 0.99)
        high_corr_pairs = [(i, j) for i, j in zip(high_corr[0], high_corr[1]) if i != j]
        print(f"  High correlation pairs (>0.99): {len(high_corr_pairs)}")
        
        return True
    except Exception as e:
        print(f"❌ Data analysis failed: {e}")
        return False

def test_minimal_model():
    """Test with extremely simple model to isolate the issue."""
    print("\n🧪 Minimal Model Test")
    print("=" * 50)
    
    try:
        # Load data
        X_train = np.load('data/X_train.npy')[:1000]  # Small subset
        y_train = np.load('data/y_train.npy')[:1000, 0:1]  # Single target
        
        # Create minimal linear model
        class MinimalModel(nn.Module):
            def __init__(self, input_size):
                super().__init__()
                self.linear = nn.Linear(input_size, 1)
                # Ultra-conservative initialization
                nn.init.normal_(self.linear.weight, mean=0.0, std=0.001)
                nn.init.zeros_(self.linear.bias)
            
            def forward(self, x):
                # Simple mean pooling over sequence
                x_pooled = x.mean(dim=1)  # [batch, features]
                return self.linear(x_pooled)
        
        model = MinimalModel(X_train.shape[-1])
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = model.to(device)
        
        # Convert to tensors
        X_tensor = torch.FloatTensor(X_train).to(device)
        y_tensor = torch.FloatTensor(y_train).to(device)
        
        # Test forward pass
        with torch.no_grad():
            output = model(X_tensor[:10])
            print(f"✓ Forward pass successful: output shape {output.shape}")
            print(f"  Output range: [{output.min().item():.6f}, {output.max().item():.6f}]")
        
        # Test single training step
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        criterion = nn.MSELoss()
        
        optimizer.zero_grad()
        pred = model(X_tensor[:32])
        loss = criterion(pred.squeeze(), y_tensor[:32].squeeze())
        loss.backward()
        
        # Check gradients
        total_norm = 0
        for p in model.parameters():
            if p.grad is not None:
                param_norm = p.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
                print(f"  Param gradient norm: {param_norm.item():.6f}")
        total_norm = total_norm ** (1. / 2)
        
        print(f"✓ Minimal model gradient norm: {total_norm:.6f}")
        print(f"✓ Minimal model loss: {loss.item():.6f}")
        
        return total_norm < 1.0  # Should be very stable
        
    except Exception as e:
        print(f"❌ Minimal model test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def analyze_tft_model():
    """Analyze TFT model components for gradient explosion sources."""
    print("\n🏗️ TFT Model Architecture Analysis")
    print("=" * 50)
    
    try:
        from tft_model import TemporalFusionTransformer, create_tft_config
        
        # Create model with debugging
        config = create_tft_config(
            input_size=134,  # From training
            hidden_size=512,  # Smaller for debugging
            num_heads=4,      # Fewer heads
            sequence_length=128,
            quantile_levels=[0.5],
            prediction_horizon=[1],
            dropout_rate=0.1
        )
        
        model = TemporalFusionTransformer(config)
        
        # Print model architecture
        print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
        
        # Check layer-wise parameter initialization
        for name, param in model.named_parameters():
            if param.requires_grad:
                print(f"  {name}: shape={param.shape}, "
                      f"mean={param.data.mean().item():.6f}, "
                      f"std={param.data.std().item():.6f}, "
                      f"max_abs={param.data.abs().max().item():.6f}")
        
        # Test with synthetic input
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = model.to(device)
        
        # Create synthetic normalized input
        batch_size = 4
        seq_len = 128
        features = 134
        
        # Properly normalized synthetic data
        synthetic_input = torch.randn(batch_size, seq_len, features).to(device) * 0.1
        
        print(f"\nSynthetic input stats:")
        print(f"  Mean: {synthetic_input.mean().item():.6f}")
        print(f"  Std: {synthetic_input.std().item():.6f}")
        print(f"  Max abs: {synthetic_input.abs().max().item():.6f}")
        
        # Forward pass
        model.eval()
        with torch.no_grad():
            try:
                output = model({'historical_features': synthetic_input})
                print(f"✓ Forward pass successful")
                
                if isinstance(output, dict) and 'predictions' in output:
                    pred = output['predictions']['horizon_1']
                    print(f"  Output shape: {pred.shape}")
                    print(f"  Output range: [{pred.min().item():.6f}, {pred.max().item():.6f}]")
                    print(f"  Output mean: {pred.mean().item():.6f}")
                    print(f"  Output std: {pred.std().item():.6f}")
                
            except Exception as e:
                print(f"❌ Forward pass failed: {e}")
                return False
        
        return True
        
    except Exception as e:
        print(f"❌ TFT model analysis failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def gradient_debugging_training():
    """Debug training with comprehensive gradient monitoring."""
    print("\n🐛 Gradient Debugging Training")
    print("=" * 50)
    
    try:
        # Load small subset for debugging
        X_train = np.load('data/X_train.npy')[:100]
        y_train = np.load('data/y_train.npy')[:100, 0:1]
        
        print(f"Debug data: X{X_train.shape}, y{y_train.shape}")
        
        # Create simple LSTM for debugging
        class DebugLSTM(nn.Module):
            def __init__(self, input_size, hidden_size=32):
                super().__init__()
                self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
                self.fc = nn.Linear(hidden_size, 1)
                
                # Ultra-conservative initialization
                for name, param in self.named_parameters():
                    if 'weight' in name:
                        nn.init.xavier_uniform_(param, gain=0.01)
                    elif 'bias' in name:
                        nn.init.zeros_(param)
            
            def forward(self, x):
                lstm_out, _ = self.lstm(x)
                return self.fc(lstm_out[:, -1, :])
        
        model = DebugLSTM(X_train.shape[-1])
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = model.to(device)
        
        # Convert data
        X_tensor = torch.FloatTensor(X_train).to(device)
        y_tensor = torch.FloatTensor(y_train).to(device)
        
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)  # Very small LR
        criterion = nn.MSELoss()
        
        print("Starting debug training...")
        
        for step in range(10):
            optimizer.zero_grad()
            
            # Forward pass
            pred = model(X_tensor)
            loss = criterion(pred.squeeze(), y_tensor.squeeze())
            
            print(f"Step {step}: Loss = {loss.item():.6f}")
            
            # Backward pass
            loss.backward()
            
            # Detailed gradient analysis
            total_norm = 0
            max_grad = 0
            for name, param in model.named_parameters():
                if param.grad is not None:
                    param_norm = param.grad.data.norm(2)
                    max_param_grad = param.grad.data.abs().max()
                    total_norm += param_norm.item() ** 2
                    max_grad = max(max_grad, max_param_grad.item())
                    print(f"  {name}: grad_norm={param_norm.item():.6f}, max_grad={max_param_grad.item():.6f}")
            
            total_norm = total_norm ** (1. / 2)
            print(f"  Total gradient norm: {total_norm:.6f}")
            print(f"  Max gradient value: {max_grad:.6f}")
            
            # Check for explosion
            if total_norm > 1.0:
                print(f"  ⚠️ Gradient explosion detected at step {step}!")
                break
            
            optimizer.step()
            
        return True
        
    except Exception as e:
        print(f"❌ Debug training failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run comprehensive diagnostic."""
    print("🚀 TFT Gradient Explosion Diagnostic")
    print("=" * 60)
    
    # Check if data exists
    data_files = ['data/X_train.npy', 'data/y_train.npy', 'data/X_val.npy', 'data/y_val.npy']
    if not all(Path(f).exists() for f in data_files):
        print("❌ Training data not found. Run training script first.")
        return
    
    results = {}
    
    # Run investigations
    results['data_ok'] = check_data_statistics()
    results['minimal_ok'] = test_minimal_model()
    results['tft_ok'] = analyze_tft_model()
    results['debug_ok'] = gradient_debugging_training()
    
    # Summary
    print("\n" + "=" * 60)
    print("🎯 DIAGNOSTIC SUMMARY")
    print("=" * 60)
    
    for test, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{test:15s}: {status}")
    
    # Recommendations
    print("\n💡 RECOMMENDATIONS:")
    if not results.get('data_ok', False):
        print("🔴 DATA ISSUE: Fix data normalization and outlier handling")
    if not results.get('minimal_ok', False):
        print("🔴 FUNDAMENTAL ISSUE: Problem with basic optimization setup")
    if not results.get('tft_ok', False):
        print("🔴 MODEL ISSUE: TFT architecture causing instability")
    if not results.get('debug_ok', False):
        print("🔴 TRAINING ISSUE: Gradient accumulation or computational problem")
    
    if all(results.values()):
        print("🟢 All tests passed - issue may be in training configuration")
    else:
        print("🔴 Issues detected - see analysis above for fixes")

if __name__ == "__main__":
    main()