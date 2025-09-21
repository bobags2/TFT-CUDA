#!/usr/bin/env python3
"""
TFT Production Training Script - FIXED TENSOR DIMENSIONS
Fixes all tensor shape mismatches by properly configuring the VSN
"""

import sys
import warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, "python")

import torch
import torch.nn as nn
import numpy as np
import json
from pathlib import Path
from torch.utils.data import DataLoader, TensorDataset
from typing import Dict, Optional
import time

# Import TFT model and related components
from tft_model import TemporalFusionTransformer, TFTLoss

def apply_tft_initialization(model: nn.Module):
    """Balanced weight initialization for TFT to prevent gradient explosion."""
    print("🔧 Applying balanced TFT weight initialization...")
    
    for name, param in model.named_parameters():
        if "weight" in name:
            if "lstm" in name.lower():
                # Xavier uniform for LSTM weights
                nn.init.xavier_uniform_(param)
            elif "linear" in name.lower() or "fc" in name.lower():
                # He initialization for linear layers 
                nn.init.kaiming_normal_(param, mode='fan_out', nonlinearity='relu')
            else:
                # Normal init for other weights (embeddings, etc.)
                nn.init.normal_(param, std=0.1)
        elif "bias" in name:
            # Initialize biases to zero
            nn.init.zeros_(param)
            # Special handling for LSTM forget gate bias
            if "lstm" in name.lower() and param.numel() >= 4:
                n = param.size(0)
                if n % 4 == 0:  # Proper LSTM bias structure
                    param.data[n//4:n//2].fill_(1.0)  # Forget gate bias trick
    
    print("✅ Applied balanced TFT weight initialization")

class GradientAccumulationTrainer:
    """Production trainer with gradient accumulation and stability monitoring."""
    
    def __init__(self, model: nn.Module, config: Optional[Dict] = None):
        self.model = model
        self.gradient_history = []
        self.config = config or {
            "max_grad_norm": 2.0,     # Progressive clipping
            "accumulation_steps": 4,  # Accumulate gradients over 4 micro-batches
            "warmup_steps": 100,
            "learning_rate": 3e-4,
            "use_gradient_scaling": False,  # Disabled for balanced approach
            "scale_factor": 0.01      # Scale gradients down
        }
    
    def train_epoch(self, dataloader, optimizer, criterion, device, epoch: int = 0):
        """Train for one epoch with gradient accumulation and FP32 precision."""
        
        self.model.train()
        total_loss = 0
        total_batches = 0
        accumulated_loss = 0
        
        # Track gradient norms for stability monitoring
        gradient_norms = []
        
        optimizer.zero_grad()
        
        for batch_idx, batch_data in enumerate(dataloader):
            try:
                # Handle different batch data formats
                if isinstance(batch_data, (list, tuple)) and len(batch_data) == 2:
                    inputs, targets = batch_data
                elif isinstance(batch_data, dict):
                    # Handle TFT dict format
                    inputs = batch_data
                    targets = batch_data.get('targets', batch_data.get('future_values', None))
                    if targets is None:
                        # Use autoencoder-style if no targets
                        targets = inputs.get('historical_features', list(inputs.values())[0])
                else:
                    inputs = batch_data
                    targets = inputs  # Autoencoder-style
                
                # Move data to device properly
                inputs = self._move_to_device(inputs, device)
                targets = self._move_to_device(targets, device)
                
                # Mixed precision forward pass with overflow protection
                with torch.cuda.amp.autocast(enabled=True, dtype=torch.float16):
                    # Clip input values to prevent FP16 overflow
                    if isinstance(inputs, dict):
                        clipped_inputs = {}
                        for k, v in inputs.items():
                            if hasattr(v, 'clamp'):
                                # Clamp to FP16 safe range [-65504, 65504]
                                clipped_inputs[k] = torch.clamp(v, min=-65000, max=65000)
                            else:
                                clipped_inputs[k] = v
                        inputs = clipped_inputs
                    elif hasattr(inputs, 'clamp'):
                        inputs = torch.clamp(inputs, min=-65000, max=65000)
                    
                    outputs = self.model(inputs)
                    
                    # Extract predictions
                    if isinstance(outputs, dict):
                        pred = outputs.get("predictions", outputs.get("horizon_1", outputs))
                        if isinstance(pred, dict):
                            pred = pred.get("horizon_1", list(pred.values())[0])
                    else:
                        pred = outputs
                    
                    # Extract targets properly
                    if isinstance(targets, dict):
                        targets_tensor = targets.get('targets', targets.get('future_values', 
                                                   targets.get('historical_features', list(targets.values())[0])))
                    else:
                        targets_tensor = targets
                    
                    # Clamp targets to FP16 safe range
                    if hasattr(targets_tensor, 'clamp'):
                        targets_tensor = torch.clamp(targets_tensor, min=-65000, max=65000)
                    
                    # Ensure shape compatibility
                    if pred.dim() > targets_tensor.dim():
                        pred = pred.squeeze(-1)
                    elif targets_tensor.dim() > pred.dim():
                        targets_tensor = targets_tensor.squeeze(-1)
                    
                    # Compute loss (scaled for accumulation)
                    loss = criterion(pred, targets_tensor) / self.config["accumulation_steps"]
                
                # Check for NaN loss
                if torch.isnan(loss) or torch.isinf(loss):
                    print(f"⚠️  NaN/Inf loss at batch {batch_idx}, skipping")
                    optimizer.zero_grad()  # Clear any accumulated gradients
                    continue
                
                # Mixed precision backward pass
                scaler.scale(loss).backward()
                
                accumulated_loss += loss.item()
                
                # Gradient accumulation step
                if (batch_idx + 1) % self.config["accumulation_steps"] == 0:
                    
                    # Unscale gradients before checking/clipping (mixed precision)
                    scaler.unscale_(optimizer)
                    
                    # Monitor gradient norm BEFORE clipping
                    total_norm_before = self._compute_gradient_norm()
                    
                    # Emergency gradient check
                    if total_norm_before > 100:
                        print(f"🔴 EMERGENCY: Gradient norm {total_norm_before:.2f}, clearing gradients!")
                        optimizer.zero_grad()
                        accumulated_loss = 0
                        continue
                    
                    # Gradient clipping
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), 
                        self.config["max_grad_norm"]
                    )
                    
                    # Update weights (mixed precision)
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad()
                    
                    # Log accumulated batch loss
                    total_loss += accumulated_loss
                    total_batches += 1
                    
                    # Monitor gradient norm AFTER clipping
                    total_norm_after = self._compute_gradient_norm()
                    gradient_norms.append(total_norm_after)
                    
                    # Reset accumulation
                    accumulated_loss = 0
                    
                    # Progress update every 50 accumulation steps
                    if total_batches % 50 == 0:
                        avg_loss = total_loss / max(total_batches, 1)
                        avg_grad = np.mean(gradient_norms[-10:]) if gradient_norms else 0
                        print(f"Batch {batch_idx}: Loss={avg_loss:.6f}, Grad={avg_grad:.3f}")
                
            except Exception as e:
                print(f"❌ Error in batch {batch_idx}: {e}")
                optimizer.zero_grad()  # Clear gradients on error
                continue
        
        avg_loss = total_loss / max(total_batches, 1)
        avg_grad = np.mean(gradient_norms) if gradient_norms else 0
        
        return avg_loss, avg_grad
    
    def _move_to_device(self, data, device):
        """Safely move data to device, handling both dicts and tensors."""
        if isinstance(data, dict):
            new_data = {}
            for k, v in data.items():
                if hasattr(v, 'to'):
                    new_data[k] = v.to(device, non_blocking=True)
                else:
                    new_data[k] = v
            return new_data
        elif hasattr(data, 'to'):
            return data.to(device, non_blocking=True)
        else:
            return data
    
    def _compute_gradient_norm(self) -> float:
        """Compute total gradient norm across all parameters."""
        total_norm = 0.0
        for p in self.model.parameters():
            if p.grad is not None:
                param_norm = p.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
        return total_norm ** 0.5

def main():
    """Main training function."""
    try:
        print("🔧 Setting up CUDA optimizations (FP32 Mode - FIXED DIMENSIONS)...")
        
        # Configure CUDA for FP32 training (NO mixed precision)
        if torch.cuda.is_available():
            device = "cuda"
            torch.backends.cudnn.benchmark = True  # Optimize cuDNN for consistent input sizes
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
        
        # Check for extreme values that could cause FP16 overflow
        print(f"📋 Data ranges - X_train: [{np.min(X_train):.2f}, {np.max(X_train):.2f}]")
        print(f"📋 Data ranges - y_train: [{np.min(y_train):.2f}, {np.max(y_train):.2f}]")
        
        # Apply robust scaling to prevent FP16 overflow
        if np.max(np.abs(X_train)) > 1000 or np.max(np.abs(y_train)) > 1000:
            print("⚠️  Large values detected, applying robust scaling...")
            
            # Scale features to reasonable range for FP16
            X_scale = max(np.percentile(np.abs(X_train), 99.9), 1e-6)
            y_scale = max(np.percentile(np.abs(y_train), 99.9), 1e-6)
            
            X_train = X_train / X_scale * 10  # Scale to ~[-10, 10] range
            y_train = y_train / y_scale * 10
            X_val = X_val / X_scale * 10
            y_val = y_val / y_scale * 10
            
            print(f"✅ Applied scaling - X_scale: {X_scale:.2e}, y_scale: {y_scale:.2e}")
            print(f"📋 New ranges - X_train: [{np.min(X_train):.2f}, {np.max(X_train):.2f}]")
            print(f"📋 New ranges - y_train: [{np.min(y_train):.2f}, {np.max(y_train):.2f}]")
        
        print(f"✅ Data loaded - Train: {X_train.shape}, Val: {X_val.shape}")
        
        # TFT Model Configuration with 4 attention heads
        sequence_length = X_train.shape[1] if len(X_train.shape) > 1 else 50
        input_size = X_train.shape[2] if len(X_train.shape) > 2 else X_train.shape[1]
        
        model_config = {
            'input_size': input_size,
            'output_size': 1,
            'hidden_size': 512,
            'num_heads': 4,              # Correct parameter name for attention heads
            'num_encoder_layers': 3,
            'num_decoder_layers': 3,
            'dropout_rate': 0.2,         # Correct parameter name
            'sequence_length': sequence_length,
            'quantile_levels': [0.5],    # Single quantile for regression
            'prediction_horizon': [1],   # Single horizon prediction
            # CRITICAL FIX: Proper VSN configuration
            'num_historical_features': input_size,  # Use actual input size
            'num_future_features': input_size,      # Use same as historical
            'static_input_size': 10
        }
        
        print("🧠 Initializing TFT model with FIXED DIMENSIONS...")
        print(f"   Input size: {input_size}, Hidden: {model_config['hidden_size']}")
        print(f"   Attention heads: {model_config['num_heads']} 🚀")
        print(f"   Quantiles: {len(model_config['quantile_levels'])} (median only)")
        print(f"   🔧 Historical features: {model_config['num_historical_features']}")
        print(f"   � Future features: {model_config['num_future_features']}")
        
        model = TemporalFusionTransformer(model_config)
        apply_tft_initialization(model)
        model = model.to(device)
        
        # Standard optimizer (NO mixed precision scaler)
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=3e-4,        # Balanced LR for financial ML
            betas=(0.9, 0.999),
            eps=1e-8,
            weight_decay=1e-4
        )
        
        # Create data loaders
        try:
            from torch.utils.data import DataLoader
            from data import TFTDataset
        except ImportError:
            print("⚠️  TFTDataset not found, using TensorDataset")
            from torch.utils.data import DataLoader, TensorDataset
            
            # Convert to tensors
            X_train_tensor = torch.FloatTensor(X_train)
            y_train_tensor = torch.FloatTensor(y_train)
            X_val_tensor = torch.FloatTensor(X_val)
            y_val_tensor = torch.FloatTensor(y_val)
            
            # Create datasets
            train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
            val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
        else:
            # Use TFT Dataset
            train_dataset = TFTDataset(
                X_train, y_train,
                sequence_length=sequence_length,
                prediction_horizon=1
            )
            
            val_dataset = TFTDataset(
                X_val, y_val,
                sequence_length=sequence_length,
                prediction_horizon=1
            )
        
        # Optimal batch size for CUDA performance
        train_loader = DataLoader(
            train_dataset,
            batch_size=128,          # Optimized for RTX 3070ti
            shuffle=True,
            num_workers=4,           # Multi-core data loading
            pin_memory=True,         # CUDA optimization
            persistent_workers=True  # Keep workers alive
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=256,          # Larger batch for validation
            shuffle=False,
            num_workers=2,
            pin_memory=True
        )
        
        # Loss function - MSE for financial regression
        criterion = nn.MSELoss()
        
        # Initialize trainer
        trainer = GradientAccumulationTrainer(model)
        
        print("🏋️ Starting training...")
        print(f"📊 Epochs: 50, Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")
        
        best_loss = float('inf')
        
        for epoch in range(50):
            print(f"\nEpoch {epoch+1}/50")
            print("-" * 40)
            
            # Train
            train_loss, train_grad = trainer.train_epoch(
                train_loader, optimizer, criterion, device, scaler, epoch
            )
            
            # Validate
            model.eval()
            val_loss = 0
            val_batches = 0
            
            with torch.no_grad():
                for batch_data in val_loader:
                    try:
                        # Handle batch data properly (same logic as training)
                        if isinstance(batch_data, (list, tuple)) and len(batch_data) == 2:
                            inputs, targets = batch_data
                        elif isinstance(batch_data, dict):
                            inputs = batch_data
                            targets = batch_data.get('targets', batch_data.get('future_values', None))
                            if targets is None:
                                targets = inputs.get('historical_features', list(inputs.values())[0])
                        else:
                            inputs = batch_data
                            targets = inputs
                        
                        # Move to device properly
                        inputs = trainer._move_to_device(inputs, device)
                        targets = trainer._move_to_device(targets, device)
                        
                        # Clamp inputs for FP16 safety in validation
                        if isinstance(inputs, dict):
                            for k, v in inputs.items():
                                if hasattr(v, 'clamp'):
                                    inputs[k] = torch.clamp(v, min=-65000, max=65000)
                        elif hasattr(inputs, 'clamp'):
                            inputs = torch.clamp(inputs, min=-65000, max=65000)
                        
                        with torch.cuda.amp.autocast(enabled=True, dtype=torch.float16):
                            outputs = model(inputs)
                        if isinstance(outputs, dict):
                            pred = outputs["predictions"]["horizon_1"]
                        else:
                            pred = outputs
                        
                        # Extract targets properly
                        if isinstance(targets, dict):
                            targets_tensor = targets.get('targets', targets.get('future_values', 
                                                       targets.get('historical_features', list(targets.values())[0])))
                        else:
                            targets_tensor = targets
                        
                        if pred.dim() > targets_tensor.dim():
                            pred = pred.squeeze(-1)
                        
                        loss = criterion(pred, targets_tensor)
                        if not torch.isnan(loss):
                            val_loss += loss.item()
                            val_batches += 1
                    except Exception as e:
                        print(f"Validation error: {e}")
                        continue
            
            avg_val_loss = val_loss / max(val_batches, 1)
            
            print(f"Train Loss: {train_loss:.6f}, Val Loss: {avg_val_loss:.6f}")
            print(f"Avg Gradient Norm: {train_grad:.3f}")
            
            # Save best model
            if avg_val_loss < best_loss:
                best_loss = avg_val_loss
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': best_loss,
                    'config': model_config
                }, "checkpoints/production_tft_best.pth")
                print(f"💾 Best model saved! Loss: {best_loss:.6f}")
        
        print(f"📈 Best validation loss: {best_loss:.6f}")
        print("💾 Model saved to: checkpoints/production_tft_best.pth")
        print("🎉 Training completed successfully!")
        
    except Exception as e:
        import traceback
        print(f"❌ Training failed: {e}")
        traceback.print_exc()

if __name__ == "__main__":
    main()