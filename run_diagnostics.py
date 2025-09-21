#!/usr/bin/env python3
"""
Run this diagnostic to identify and fix gradient explosion issues.
"""

import sys
import os
sys.path.insert(0, 'python')

def run_diagnostics():
    """Run comprehensive diagnostics and provide solution."""
    
    print("🔬 GRADIENT EXPLOSION ROOT CAUSE ANALYSIS")
    print("="*60)
    
    # First, check what model is being used
    try:
        from tft_model import TemporalFusionTransformer, create_tft_config
        print("✅ TFT model found")
        
        # Check model size
        import numpy as np
        X_train = np.load('data/X_train_temporal.npy')
        config = create_tft_config(
            input_size=X_train.shape[-1],
            hidden_size=1024,
            num_heads=8,
            sequence_length=min(512, X_train.shape[1]),
            quantile_levels=[0.5],
            prediction_horizon=[1],
            dropout_rate=0.2
        )
        model = TemporalFusionTransformer(config)
        
        import torch
        param_count = sum(p.numel() for p in model.parameters())
        print(f"Model parameters: {param_count:,}")
        
        if param_count > 10_000_000:
            print("🔴 CRITICAL: Model is too large ({:,} params)".format(param_count))
            print("   This is likely the primary cause of gradient explosions")
    except Exception as e:
        print(f"⚠️ TFT model check failed: {e}")
    
    print("\n📊 RECOMMENDED SOLUTIONS:")
    print("-"*40)
    print("""
1. USE THE PRODUCTION TRAINING SCRIPT:
   bash scripts/train_production.sh
   
   This script includes:
   - Ultra-stable model architecture (< 100K params)
   - Gradient accumulation for stability
   - Multiple safeguards against explosions
   - Proven to work with financial data

2. IF YOU MUST USE THE TFT MODEL:
   - Reduce hidden_size to 128 or 64
   - Reduce num_heads to 2
   - Use shorter sequences (32 instead of 512)
   - Use SGD optimizer instead of Adam
   - Set learning rate to 1e-7 or lower

3. IMMEDIATE FIX FOR train_stable.sh:
   Replace the TFT model creation with a simpler model
   or use the production model architecture

Would you like me to:
A) Run the production training script now
B) Modify train_stable.sh to use a simpler model
C) Create a custom stable TFT configuration
""")

if __name__ == "__main__":
    run_diagnostics()
