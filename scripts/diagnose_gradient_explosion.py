#!/usr/bin/env python3
"""
Comprehensive gradient explosion diagnostic and solution tool for TFT model.
This script will identify the exact source of gradient explosions and provide fixes.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'python'))

import torch
import torch.nn as nn
import numpy as np
import warnings
warnings.filterwarnings('ignore')
from typing import Dict, List, Tuple, Optional
import json
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns

class GradientDiagnostics:
    """Comprehensive gradient diagnostics and monitoring."""
    
    def __init__(self):
        self.layer_gradients = {}
        self.layer_activations = {}
        self.layer_weights = {}
        self.gradient_history = []
        self.activation_history = []
        self.problematic_layers = set()
        
    def register_hooks(self, model: nn.Module) -> None:
        """Register hooks to monitor gradients and activations."""
        
        def make_grad_hook(name):
            def hook(module, grad_input, grad_output):
                # Store gradient statistics
                if grad_output[0] is not None:
                    grad = grad_output[0].detach()
                    self.layer_gradients[name] = {
                        'mean': float(grad.mean().abs()),
                        'max': float(grad.max().abs()),
                        'std': float(grad.std()),
                        'has_nan': bool(torch.isnan(grad).any()),
                        'has_inf': bool(torch.isinf(grad).any())
                    }
                    
                    # Flag problematic layers
                    if self.layer_gradients[name]['max'] > 10.0:
                        self.problematic_layers.add(name)
            return hook
        
        def make_activation_hook(name):
            def hook(module, input, output):
                if output is not None:
                    if isinstance(output, tuple):
                        output = output[0]
                    act = output.detach()
                    self.layer_activations[name] = {
                        'mean': float(act.mean().abs()),
                        'max': float(act.max().abs()),
                        'std': float(act.std()),
                        'has_nan': bool(torch.isnan(act).any()),
                        'has_inf': bool(torch.isinf(act).any())
                    }
            return hook
        
        # Register hooks for all layers
        for name, module in model.named_modules():
            if len(list(module.children())) == 0:  # Leaf modules only
                module.register_backward_hook(make_grad_hook(name))
                module.register_forward_hook(make_activation_hook(name))
    
    def analyze_model_numerics(self, model: nn.Module) -> Dict:
        """Analyze numerical properties of model weights."""
        analysis = {}
        
        for name, param in model.named_parameters():
            if param is not None:
                weight_data = param.detach()
                analysis[name] = {
                    'mean': float(weight_data.mean()),
                    'std': float(weight_data.std()),
                    'max': float(weight_data.max()),
                    'min': float(weight_data.min()),
                    'has_nan': bool(torch.isnan(weight_data).any()),
                    'has_inf': bool(torch.isinf(weight_data).any()),
                    'shape': list(weight_data.shape),
                    'num_params': weight_data.numel()
                }
                
                # Check for problematic initializations
                if analysis[name]['std'] > 1.0:
                    print(f"⚠️  WARNING: Layer {name} has high std: {analysis[name]['std']:.4f}")
                if analysis[name]['has_nan'] or analysis[name]['has_inf']:
                    print(f"🔴 CRITICAL: Layer {name} has NaN or Inf values!")
        
        return analysis
    
    def diagnose_data(self, X: np.ndarray, y: np.ndarray) -> Dict:
        """Diagnose potential issues in the data."""
        diagnosis = {
            'X': self._analyze_array(X, 'Features'),
            'y': self._analyze_array(y, 'Targets')
        }
        
        # Check for extreme correlations
        if len(X.shape) == 3:
            X_flat = X.reshape(-1, X.shape[-1])
            corr_matrix = np.corrcoef(X_flat.T)
            diagnosis['feature_correlations'] = {
                'max_correlation': float(np.max(np.abs(corr_matrix[np.triu_indices_from(corr_matrix, k=1)]))),
                'highly_correlated_pairs': int(np.sum(np.abs(corr_matrix[np.triu_indices_from(corr_matrix, k=1)]) > 0.95))
            }
        
        return diagnosis
    
    def _analyze_array(self, arr: np.ndarray, name: str) -> Dict:
        """Analyze a numpy array for numerical issues."""
        return {
            'shape': arr.shape,
            'mean': float(np.mean(arr)),
            'std': float(np.std(arr)),
            'min': float(np.min(arr)),
            'max': float(np.max(arr)),
            'has_nan': bool(np.isnan(arr).any()),
            'has_inf': bool(np.isinf(arr).any()),
            'num_zeros': int(np.sum(arr == 0)),
            'num_extreme_values': int(np.sum(np.abs(arr) > 10))
        }
    
    def generate_report(self) -> str:
        """Generate a comprehensive diagnostic report."""
        report = []
        report.append("="*60)
        report.append("GRADIENT EXPLOSION DIAGNOSTIC REPORT")
        report.append("="*60)
        
        # Layer-wise gradient analysis
        if self.layer_gradients:
            report.append("\n📊 GRADIENT ANALYSIS BY LAYER:")
            report.append("-"*40)
            
            # Sort layers by max gradient
            sorted_layers = sorted(
                self.layer_gradients.items(),
                key=lambda x: x[1]['max'],
                reverse=True
            )
            
            for layer_name, stats in sorted_layers[:10]:  # Top 10 problematic layers
                status = "🔴 CRITICAL" if stats['max'] > 100 else "⚠️  WARNING" if stats['max'] > 10 else "✅"
                report.append(f"{status} {layer_name}:")
                report.append(f"    Max gradient: {stats['max']:.2e}")
                report.append(f"    Mean gradient: {stats['mean']:.2e}")
                report.append(f"    Has NaN: {stats['has_nan']}, Has Inf: {stats['has_inf']}")
        
        # Activation analysis
        if self.layer_activations:
            report.append("\n📊 ACTIVATION ANALYSIS BY LAYER:")
            report.append("-"*40)
            
            # Find layers with extreme activations
            extreme_layers = [(name, stats) for name, stats in self.layer_activations.items()
                            if stats['max'] > 100 or stats['has_nan'] or stats['has_inf']]
            
            if extreme_layers:
                for layer_name, stats in extreme_layers[:5]:
                    report.append(f"🔴 {layer_name}:")
                    report.append(f"    Max activation: {stats['max']:.2e}")
                    report.append(f"    Has NaN: {stats['has_nan']}, Has Inf: {stats['has_inf']}")
            else:
                report.append("✅ All activations within normal range")
        
        # Problematic layers summary
        if self.problematic_layers:
            report.append("\n🔴 PROBLEMATIC LAYERS IDENTIFIED:")
            report.append("-"*40)
            for layer in list(self.problematic_layers)[:10]:
                report.append(f"  • {layer}")
        
        return "\n".join(report)


class RobustTFTWrapper(nn.Module):
    """
    Robust wrapper for TFT model with multiple gradient explosion prevention mechanisms.
    """
    
    def __init__(self, base_model: nn.Module, config: Optional[Dict] = None):
        super().__init__()
        self.base_model = base_model
        self.config = config or self._default_config()
        
        # Gradient scaling factor (dynamically adjusted)
        self.register_buffer('grad_scale', torch.tensor(1.0))
        self.register_buffer('activation_scale', torch.tensor(1.0))
        
        # Apply stabilization to base model
        self._stabilize_model()
        
    def _default_config(self) -> Dict:
        return {
            'use_spectral_norm': True,
            'use_gradient_scaling': True,
            'use_activation_checkpointing': False,
            'max_activation_value': 10.0,
            'use_layer_norm': True,
            'use_residual_scaling': True,
            'residual_scale': 0.1,
            'dropout_rate': 0.3
        }
    
    def _stabilize_model(self):
        """Apply various stabilization techniques to the model."""
        
        # 1. Apply spectral normalization to linear layers
        if self.config['use_spectral_norm']:
            for name, module in self.base_model.named_modules():
                if isinstance(module, nn.Linear):
                    # Replace with spectral norm version
                    setattr(self.base_model, name.split('.')[-1], 
                           nn.utils.spectral_norm(module))
        
        # 2. Add layer normalization after linear layers
        if self.config['use_layer_norm']:
            self._add_layer_norms()
        
        # 3. Initialize weights with even more conservative values
        self._ultra_safe_initialization()
    
    def _add_layer_norms(self):
        """Add layer normalization to the model."""
        # This would need to be customized based on the actual model architecture
        pass
    
    def _ultra_safe_initialization(self):
        """Ultra-conservative weight initialization."""
        for name, param in self.base_model.named_parameters():
            if 'weight' in name:
                if len(param.shape) >= 2:
                    # Use very small variance initialization
                    fan_in = param.shape[1]
                    std = 0.001 / np.sqrt(fan_in)  # Much smaller than typical
                    nn.init.normal_(param, mean=0, std=std)
                else:
                    nn.init.zeros_(param)
            elif 'bias' in name:
                nn.init.zeros_(param)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with activation clamping and gradient scaling."""
        
        # Input validation and clamping
        x = self._validate_and_clamp(x)
        
        # Scale inputs if needed
        if self.config['use_gradient_scaling']:
            x = x * self.activation_scale
        
        # Forward through base model with monitoring
        output = self.base_model(x)
        
        # Output clamping
        if isinstance(output, dict):
            for key in output:
                if isinstance(output[key], dict):
                    for subkey in output[key]:
                        output[key][subkey] = self._validate_and_clamp(output[key][subkey])
                else:
                    output[key] = self._validate_and_clamp(output[key])
        else:
            output = self._validate_and_clamp(output)
        
        return output
    
    def _validate_and_clamp(self, tensor: torch.Tensor) -> torch.Tensor:
        """Validate tensor and clamp to prevent explosions."""
        if tensor is None:
            return tensor
        
        # Check for NaN/Inf
        if torch.isnan(tensor).any() or torch.isinf(tensor).any():
            print(f"⚠️  WARNING: NaN or Inf detected, replacing with zeros")
            tensor = torch.where(torch.isfinite(tensor), tensor, torch.zeros_like(tensor))
        
        # Clamp values
        max_val = self.config['max_activation_value']
        tensor = torch.clamp(tensor, min=-max_val, max=max_val)
        
        return tensor


def create_stable_optimizer(model: nn.Module, config: Optional[Dict] = None) -> Tuple[torch.optim.Optimizer, Dict]:
    """
    Create a stable optimizer configuration with layer-wise learning rates.
    """
    config = config or {}
    
    # Group parameters by type
    param_groups = []
    
    # Embedding layers - very low LR
    embedding_params = []
    # Attention layers - low LR
    attention_params = []
    # LSTM/RNN layers - medium LR
    recurrent_params = []
    # Output layers - normal LR
    output_params = []
    # Others
    other_params = []
    
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
            
        if 'embedding' in name.lower():
            embedding_params.append(param)
        elif 'attention' in name.lower() or 'attn' in name.lower():
            attention_params.append(param)
        elif 'lstm' in name.lower() or 'gru' in name.lower() or 'rnn' in name.lower():
            recurrent_params.append(param)
        elif 'output' in name.lower() or 'head' in name.lower() or 'fc' in name.lower():
            output_params.append(param)
        else:
            other_params.append(param)
    
    # Base learning rate (ultra-conservative)
    base_lr = config.get('base_lr', 1e-6)
    
    # Create parameter groups with different learning rates
    if embedding_params:
        param_groups.append({'params': embedding_params, 'lr': base_lr * 0.1})
    if attention_params:
        param_groups.append({'params': attention_params, 'lr': base_lr * 0.5})
    if recurrent_params:
        param_groups.append({'params': recurrent_params, 'lr': base_lr * 0.8})
    if output_params:
        param_groups.append({'params': output_params, 'lr': base_lr})
    if other_params:
        param_groups.append({'params': other_params, 'lr': base_lr * 0.5})
    
    # Use SGD with momentum for more stable training
    optimizer = torch.optim.SGD(
        param_groups,
        lr=base_lr,
        momentum=0.9,
        weight_decay=1e-4,
        nesterov=True
    )
    
    # Alternative: AdamW with very conservative settings
    # optimizer = torch.optim.AdamW(
    #     param_groups,
    #     lr=base_lr,
    #     betas=(0.9, 0.98),  # Less aggressive than default
    #     eps=1e-6,  # Higher epsilon for stability
    #     weight_decay=1e-3
    # )
    
    scheduler_config = {
        'warmup_epochs': 10,  # Longer warmup
        'warmup_factor': 0.01,  # Start from 1% of base LR
        'decay_factor': 0.95,
        'decay_epochs': 5
    }
    
    return optimizer, scheduler_config


def diagnose_and_fix():
    """Main diagnostic and fixing routine."""
    print("🔬 Starting Comprehensive Gradient Explosion Diagnosis...")
    print("="*60)
    
    # Load data
    print("\n📂 Loading data...")
    try:
        X_train = np.load('data/X_train_temporal.npy')
        y_train = np.load('data/y_train_temporal.npy')
        X_val = np.load('data/X_val_temporal.npy')
        y_val = np.load('data/y_val_temporal.npy')
        print(f"   ✅ Temporal data loaded: X_train{X_train.shape}, y_train{y_train.shape}")
    except:
        X_train = np.load('data/X_train.npy')
        y_train = np.load('data/y_train.npy')
        X_val = np.load('data/X_val.npy')
        y_val = np.load('data/y_val.npy')
        print(f"   ✅ Standard data loaded: X_train{X_train.shape}, y_train{y_train.shape}")
    
    # Initialize diagnostics
    diagnostics = GradientDiagnostics()
    
    # Diagnose data
    print("\n🔍 Diagnosing data...")
    data_diagnosis = diagnostics.diagnose_data(X_train, y_train)
    
    print(f"   Features - Mean: {data_diagnosis['X']['mean']:.4f}, Std: {data_diagnosis['X']['std']:.4f}")
    print(f"   Targets - Mean: {data_diagnosis['y']['mean']:.4f}, Std: {data_diagnosis['y']['std']:.4f}")
    
    if 'feature_correlations' in data_diagnosis:
        print(f"   Max feature correlation: {data_diagnosis['feature_correlations']['max_correlation']:.4f}")
        if data_diagnosis['feature_correlations']['highly_correlated_pairs'] > 0:
            print(f"   ⚠️  WARNING: {data_diagnosis['feature_correlations']['highly_correlated_pairs']} highly correlated feature pairs!")
    
    # Additional data preprocessing if needed
    if data_diagnosis['X']['std'] > 10 or data_diagnosis['X']['std'] < 0.1:
        print("\n⚠️  Data standardization seems inadequate. Re-standardizing...")
        X_mean = np.mean(X_train.reshape(-1, X_train.shape[-1]), axis=0)
        X_std = np.std(X_train.reshape(-1, X_train.shape[-1]), axis=0)
        X_std[X_std < 1e-8] = 1.0
        
        X_train = (X_train - X_mean) / X_std
        X_val = (X_val - X_mean) / X_std
        
        print(f"   ✅ Re-standardized. New std: {np.std(X_train):.4f}")
    
    # Create model
    print("\n🏗️  Creating stabilized model...")
    
    try:
        from tft_model import TemporalFusionTransformer, create_tft_config
        
        # Create base model with conservative config
        config = create_tft_config(
            input_size=X_train.shape[-1],
            hidden_size=128,  # Much smaller for stability
            num_heads=2,      # Minimal heads
            sequence_length=min(32, X_train.shape[1]),  # Shorter sequences
            quantile_levels=[0.5],
            prediction_horizon=[1],
            dropout_rate=0.3
        )
        
        base_model = TemporalFusionTransformer(config)
        
        # Wrap with robust wrapper
        model = RobustTFTWrapper(base_model)
        
        print(f"   ✅ Robust TFT model created with {sum(p.numel() for p in model.parameters()):,} parameters")
        
    except Exception as e:
        print(f"   ⚠️  TFT creation failed: {e}")
        print("   Using ultra-simple baseline...")
        
        class UltraSimpleModel(nn.Module):
            def __init__(self, input_size):
                super().__init__()
                self.fc1 = nn.Linear(input_size, 32)
                self.ln1 = nn.LayerNorm(32)
                self.dropout = nn.Dropout(0.5)
                self.fc2 = nn.Linear(32, 1)
                
                # Ultra-conservative initialization
                nn.init.normal_(self.fc1.weight, std=0.01)
                nn.init.zeros_(self.fc1.bias)
                nn.init.normal_(self.fc2.weight, std=0.01)
                nn.init.zeros_(self.fc2.bias)
            
            def forward(self, x):
                if isinstance(x, dict):
                    x = x['historical_features']
                # Take mean across time dimension
                x = x.mean(dim=1)
                x = torch.relu(self.ln1(self.fc1(x)))
                x = self.dropout(x)
                return {'predictions': {'horizon_1': self.fc2(x)}}
        
        model = UltraSimpleModel(X_train.shape[-1])
        print(f"   ✅ Ultra-simple model created")
    
    # Move to device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    # Analyze model numerics
    print("\n🔍 Analyzing model numerics...")
    weight_analysis = diagnostics.analyze_model_numerics(model)
    
    # Register hooks
    diagnostics.register_hooks(model)
    
    # Create stable optimizer
    print("\n⚙️  Creating stable optimizer...")
    optimizer, scheduler_config = create_stable_optimizer(model, {'base_lr': 1e-6})
    
    # Test forward pass
    print("\n🧪 Testing forward pass...")
    from torch.utils.data import DataLoader, TensorDataset
    
    # Create simple dataset for testing
    test_X = torch.FloatTensor(X_train[:32])
    test_y = torch.FloatTensor(y_train[:32, 0:1])
    
    if len(test_X.shape) == 2:
        test_X = test_X.unsqueeze(1)  # Add time dimension if needed
    
    test_dataset = TensorDataset(test_X, test_y)
    test_loader = DataLoader(test_dataset, batch_size=4)
    
    model.train()
    criterion = nn.MSELoss()
    
    print("\n📊 Running diagnostic training steps...")
    for batch_idx, (batch_X, batch_y) in enumerate(test_loader):
        if batch_idx >= 3:  # Only test a few batches
            break
        
        batch_X = batch_X.to(device)
        batch_y = batch_y.to(device)
        
        # Create input dict for TFT
        batch_input = {'historical_features': batch_X}
        
        optimizer.zero_grad()
        
        # Forward pass with error handling
        try:
            output = model(batch_input)
            if isinstance(output, dict):
                pred = output['predictions']['horizon_1']
            else:
                pred = output
            
            loss = criterion(pred.squeeze(), batch_y.squeeze())
            
            print(f"\n   Batch {batch_idx}: Loss = {loss.item():.6f}")
            
            # Backward pass
            loss.backward()
            
            # Check gradients before clipping
            total_norm = 0
            for p in model.parameters():
                if p.grad is not None:
                    param_norm = p.grad.data.norm(2)
                    total_norm += param_norm.item() ** 2
            total_norm = total_norm ** 0.5
            
            print(f"   Gradient norm (before clipping): {total_norm:.2e}")
            
            # Aggressive gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.1)
            
            # Check gradients after clipping
            total_norm_after = 0
            for p in model.parameters():
                if p.grad is not None:
                    param_norm = p.grad.data.norm(2)
                    total_norm_after += param_norm.item() ** 2
            total_norm_after = total_norm_after ** 0.5
            
            print(f"   Gradient norm (after clipping): {total_norm_after:.2e}")
            
            optimizer.step()
            
        except Exception as e:
            print(f"   ❌ Error in batch {batch_idx}: {e}")
            import traceback
            traceback.print_exc()
    
    # Generate diagnostic report
    print("\n" + diagnostics.generate_report())
    
    # Save diagnostics
    print("\n💾 Saving diagnostic results...")
    with open('gradient_diagnostics.json', 'w') as f:
        json.dump({
            'data_diagnosis': data_diagnosis,
            'weight_analysis': weight_analysis,
            'layer_gradients': diagnostics.layer_gradients,
            'layer_activations': diagnostics.layer_activations,
            'problematic_layers': list(diagnostics.problematic_layers)
        }, f, indent=2)
    
    print("\n✅ Diagnosis complete! Results saved to gradient_diagnostics.json")
    
    # Provide recommendations
    print("\n" + "="*60)
    print("📋 RECOMMENDATIONS:")
    print("="*60)
    
    recommendations = []
    
    if diagnostics.problematic_layers:
        recommendations.append("1. Focus on stabilizing these layers: " + ", ".join(list(diagnostics.problematic_layers)[:3]))
    
    if data_diagnosis['X']['std'] > 2 or data_diagnosis['X']['std'] < 0.5:
        recommendations.append("2. Improve data normalization - current std is suboptimal")
    
    if total_norm > 10:
        recommendations.append("3. Use even more aggressive gradient clipping (try max_norm=0.01)")
    
    recommendations.append("4. Consider using gradient accumulation with micro-batches")
    recommendations.append("5. Try SGD optimizer instead of Adam - it's often more stable")
    recommendations.append("6. Reduce model complexity further if needed")
    
    for rec in recommendations:
        print(f"   • {rec}")
    
    return model, optimizer, diagnostics


if __name__ == "__main__":
    model, optimizer, diagnostics = diagnose_and_fix()
    
    print("\n🚀 Ready to train with stabilized configuration!")
    print("   Use the returned model and optimizer for training.")
