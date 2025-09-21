#!/usr/bin/env python3
"""Quick test to verify gradient stability before full training."""

import sys
sys.path.insert(0, "python")

import torch
import torch.nn as nn
import numpy as np
from tft_model import TemporalFusionTransformer

# Load sample data
X = np.load('data/X_train.npy')[:32]  # Just 32 samples
y = np.load('data/y_train.npy')[:32]

# Create model with conservative init
config = {
    'input_size': X.shape[-1],
    'output_size': 3,
    'hidden_size': 256,
    'num_heads': 4,
    'dropout_rate': 0.3,
    'sequence_length': X.shape[1],
    'prediction_horizon': [1, 5, 10],
    'num_historical_features': X.shape[-1],
    'num_future_features': X.shape[-1],
    'static_input_size': 10
}

model = TemporalFusionTransformer(config)

# Ultra-conservative init
for name, param in model.named_parameters():
    if 'weight' in name:
        if 'lstm' in name.lower():
            nn.init.normal_(param, mean=0, std=0.001)
        elif len(param.shape) >= 2:
            nn.init.normal_(param, mean=0, std=0.01)
        else:
            nn.init.zeros_(param)
    elif 'bias' in name:
        nn.init.zeros_(param)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
model.to(device)

# Test forward/backward
X_tensor = torch.FloatTensor(X).to(device)
y_tensor = torch.FloatTensor(y).to(device)

tft_inputs = {
    'historical_features': torch.clamp(X_tensor, -10, 10),
    'future_features': X_tensor[:, -1:, :],
    'static_features': torch.zeros(32, 10, device=device)
}

# Forward pass
outputs = model(tft_inputs)
pred = outputs['predictions']['horizon_1']

# Compute loss
loss = nn.MSELoss()(pred, y_tensor[:, :3])
loss.backward()

# Check gradient magnitudes
grad_norms = []
for name, param in model.named_parameters():
    if param.grad is not None:
        grad_norm = param.grad.norm().item()
        grad_norms.append(grad_norm)
        if grad_norm > 10:
            print(f"⚠️ Large gradient in {name}: {grad_norm:.2f}")

print(f"Gradient statistics:")
print(f"  Max: {max(grad_norms):.4f}")
print(f"  Mean: {np.mean(grad_norms):.4f}")
print(f"  Median: {np.median(grad_norms):.4f}")

if max(grad_norms) < 10:
    print("✅ Gradients are stable!")
else:
    print("❌ Gradients are still exploding")
