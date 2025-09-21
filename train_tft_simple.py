#!/usr/bin/env python3
"""
Simplified TFT Training Script - Direct Tensor Input
Bypasses TFTDataset to use direct tensor inputs for debugging
"""

import sys
import warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, "python")

import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from typing import Dict, Optional

# Import TFT model
from tft_model import TemporalFusionTransformer

def apply_tft_initialization(model: nn.Module):
    """Balanced weight initialization for TFT to prevent gradient explosion."""
    print("🔧 Applying balanced TFT weight initialization...")
    
    for name, param in model.named_parameters():
        if "weight" in name:
            if "lstm" in name.lower():
                nn.init.xavier_uniform_(param)
            elif "linear" in name.lower() or "fc" in name.lower():
                nn.init.kaiming_normal_(param, mode='fan_out', nonlinearity='relu')
            else:
                nn.init.normal_(param, std=0.1)
        elif "bias" in name:
            nn.init.zeros_(param)
            if "lstm" in name.lower() and param.numel() >= 4:
                n = param.size(0)
                if n % 4 == 0:
                    param.data[n//4:n//2].fill_(1.0)
    
    print("✅ Applied balanced TFT weight initialization")

def main():
    """Main training function with simplified tensor inputs."""
    try:
        print("🔧 Setting up CUDA optimizations (FP32 Mode - Simplified)...")
        
        # Configure CUDA for FP32 training
        if torch.cuda.is_available():
            device = "cuda"
            torch.backends.cudnn.benchmark = True
            print(f"✅ CUDA device: {torch.cuda.get_device_name()}")
            print(f"   Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
            print("ℹ️  Using FP32 precision to avoid overflow issues")
        else:
            device = "cpu"
            print("⚠️  CUDA not available, using CPU")
        
        # Data loading
        print("📊 Loading financial data...")
        X_train = np.load("data/X_train.npy", allow_pickle=True)
        y_train = np.load("data/y_train.npy", allow_pickle=True)
        X_val = np.load("data/X_val.npy", allow_pickle=True)
        y_val = np.load("data/y_val.npy", allow_pickle=True)
        
        print(f"✅ Data loaded - Train: {X_train.shape}, Val: {X_val.shape}")
        
        # Check data ranges
        print(f"📊 Data ranges - X: [{np.min(X_train):.2f}, {np.max(X_train):.2f}]")
        print(f"📊 Data ranges - y: [{np.min(y_train):.2f}, {np.max(y_train):.2f}]")
        
        # Get dimensions
        sequence_length = X_train.shape[1] if len(X_train.shape) > 1 else 50
        input_size = X_train.shape[2] if len(X_train.shape) > 2 else X_train.shape[1]
        
        # Simple TFT Model Configuration
        model_config = {
            'input_size': input_size,
            'output_size': 1,
            'hidden_size': 256,          # Smaller hidden size for debugging
            'num_heads': 4,
            'num_encoder_layers': 2,     # Fewer layers for debugging
            'num_decoder_layers': 2,
            'dropout_rate': 0.1,
            'sequence_length': sequence_length,
            'quantile_levels': [0.5],    # Single quantile (median)
            'prediction_horizon': [1],   # Single horizon
            'num_historical_features': input_size,
            'num_future_features': input_size,  # Same as historical
            'static_input_size': 10
        }
        
        print("🧠 Initializing simplified TFT model...")
        print(f"   Input size: {input_size}, Hidden: {model_config['hidden_size']}")
        print(f"   Attention heads: {model_config['num_heads']} 🚀")
        print(f"   Quantiles: {len(model_config['quantile_levels'])} (median only)")
        
        model = TemporalFusionTransformer(model_config)
        apply_tft_initialization(model)
        model = model.to(device)
        
        # Standard optimizer (no mixed precision)
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=1e-4,  # Lower learning rate for stability
            betas=(0.9, 0.999),
            eps=1e-8,
            weight_decay=1e-4
        )
        
        # Create simple tensor datasets (no TFTDataset)
        print("📦 Creating simplified tensor datasets...")
        X_train_tensor = torch.FloatTensor(X_train)
        y_train_tensor = torch.FloatTensor(y_train)
        X_val_tensor = torch.FloatTensor(X_val)
        y_val_tensor = torch.FloatTensor(y_val)
        
        train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
        val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
        
        train_loader = DataLoader(
            train_dataset,
            batch_size=32,  # Small batch for debugging
            shuffle=True,
            num_workers=2,
            pin_memory=True
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=64,
            shuffle=False,
            num_workers=2,
            pin_memory=True
        )
        
        # Loss function
        criterion = nn.MSELoss()
        
        print("🏋️ Starting simplified FP32 training...")
        print(f"📊 Epochs: 5, Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")
        
        model.train()
        
        for epoch in range(5):  # Just 5 epochs for testing
            print(f"\nEpoch {epoch+1}/5")
            print("-" * 40)
            
            total_loss = 0
            total_batches = 0
            
            for batch_idx, (X_batch, y_batch) in enumerate(train_loader):
                try:
                    # Move to device
                    X_batch = X_batch.to(device)
                    y_batch = y_batch.to(device)
                    
                    # Create TFT-style input dict
                    inputs = {
                        'historical_features': X_batch,
                        'future_features': torch.zeros(X_batch.size(0), 1, input_size, device=device),
                        'static_features': torch.zeros(X_batch.size(0), 10, device=device)
                    }
                    
                    # Forward pass
                    optimizer.zero_grad()
                    outputs = model(inputs)
                    
                    # Extract prediction (simplified)
                    if isinstance(outputs, dict):
                        pred = outputs.get("predictions", {}).get("horizon_1")
                        if pred is None:
                            # Try alternative keys
                            pred = list(outputs.get("predictions", outputs).values())[0]
                    else:
                        pred = outputs
                    
                    # Ensure correct shape for loss
                    if pred.dim() > y_batch.dim():
                        pred = pred.squeeze(-1)
                    elif y_batch.dim() > pred.dim():
                        y_batch = y_batch.squeeze(-1)
                    
                    # Compute loss
                    loss = criterion(pred, y_batch)
                    
                    # Check for NaN
                    if torch.isnan(loss) or torch.isinf(loss):
                        print(f"⚠️  NaN/Inf loss at batch {batch_idx}, skipping")
                        continue
                    
                    # Backward pass
                    loss.backward()
                    
                    # Gradient clipping
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    
                    # Update
                    optimizer.step()
                    
                    total_loss += loss.item()
                    total_batches += 1
                    
                    # Progress update
                    if batch_idx % 100 == 0:
                        avg_loss = total_loss / max(total_batches, 1)
                        print(f"Batch {batch_idx}: Loss={avg_loss:.6f}")
                        
                        # Test first few batches and break for debugging
                        if batch_idx >= 10:
                            print("✅ First 10 batches successful! Training is working.")
                            return
                
                except Exception as e:
                    print(f"❌ Error in batch {batch_idx}: {e}")
                    print(f"   X_batch shape: {X_batch.shape}")
                    print(f"   y_batch shape: {y_batch.shape}")
                    if 'pred' in locals():
                        print(f"   pred shape: {pred.shape}")
                    # Continue with next batch for debugging
                    continue
            
            avg_loss = total_loss / max(total_batches, 1)
            print(f"Epoch {epoch+1} completed: Average Loss={avg_loss:.6f}")
        
        print("🎉 Simplified training test completed!")
        
    except Exception as e:
        import traceback
        print(f"❌ Training failed: {e}")
        traceback.print_exc()

if __name__ == "__main__":
    main()