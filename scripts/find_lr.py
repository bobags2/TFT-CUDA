#!/usr/bin/env python3
"""
Find optimal learning rate for TFT model using LR range test.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'python'))

import torch
import numpy as np
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

# Import modules
from data import FinancialDataset, TFTDataset
from tft_model import TemporalFusionTransformer, create_tft_config
from trainer import TFTTrainer, create_training_config
from lr_scheduler import find_optimal_lr
from loss import create_tft_loss

def main():
    print("=" * 50)
    print("TFT Learning Rate Finder")
    print("=" * 50)
    
    # 1. Load data
    print("\n📊 Loading data...")
    dataset = FinancialDataset(data_dir="data/")
    
    # Check if processed data exists
    data_dir = "data/"
    if os.path.exists(os.path.join(data_dir, "X_train.npy")):
        print("   Loading cached data...")
        X_train = np.load(os.path.join(data_dir, "X_train.npy"))
        y_train = np.load(os.path.join(data_dir, "y_train.npy"))
        print(f"   Data loaded: X{X_train.shape}, y{y_train.shape}")
    else:
        print("   Processing data pipeline...")
        processed_data = dataset.process_pipeline()
        X, y = dataset.create_sequences(processed_data, sequence_length=50)
        X_train = X[:int(len(X)*0.8)]
        y_train = y[:int(len(y)*0.8)]
    
    # 2. Create model
    print("\n🏗️  Creating model...")
    config = create_tft_config(
        input_size=X_train.shape[-1],
        hidden_size=256,
        num_heads=8,
        sequence_length=X_train.shape[1],
        quantile_levels=[0.1, 0.5, 0.9],
        prediction_horizon=[1, 5, 10]
    )
    model = TemporalFusionTransformer(config)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.to(device)
    print(f"   Model created: {sum(p.numel() for p in model.parameters()):,} parameters")
    print(f"   Device: {device}")
    
    # 3. Prepare data loader
    train_dataset = TFTDataset(X_train, y_train)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    
    # 4. Create loss function
    loss_fn = create_tft_loss(
        quantile_levels=config['quantile_levels'],
        loss_type='quantile'
    )
    
    # 5. Run LR finder
    print("\n🔍 Running LR Range Test...")
    print("   Testing learning rates from 1e-7 to 10...")
    
    best_lr, lr_finder = find_optimal_lr(
        model=model,
        train_loader=train_loader,
        loss_fn=loss_fn,
        device=device,
        start_lr=1e-7,
        end_lr=10,
        num_iter=100
    )
    
    # 6. Plot results
    print("\n📈 Plotting results...")
    fig = lr_finder.plot(suggest=True, save_path='lr_finder_plot.png')
    plt.show()
    
    # 7. Display recommendations
    print("\n✨ Recommendations:")
    print(f"   Suggested learning rate: {best_lr:.2e}")
    print(f"   Conservative (÷10): {best_lr/10:.2e}")
    print(f"   Aggressive (×2): {best_lr*2:.2e}")
    
    # 8. Create training config with optimal LR
    print("\n⚙️  Creating optimized training config...")
    
    training_config = create_training_config(
        epochs=100,
        batch_size=64,
        learning_rate=best_lr,
        scheduler='onecycle',
        max_lr=best_lr * 10,  # OneCycle max LR
        device=device,
        checkpoint_dir='checkpoints'
    )
    
    print("   Training config created with:")
    print(f"   - Base LR: {training_config['learning_rate']:.2e}")
    print(f"   - Max LR: {training_config['max_lr']:.2e}")
    print(f"   - Scheduler: {training_config['scheduler']}")
    
    # 9. Optional: Test different schedulers
    print("\n🔬 Testing different schedulers...")
    
    # Test OneCycle
    trainer = TFTTrainer(model, training_config)
    print(f"   OneCycle LR schedule ready")
    
    # Test Cosine Annealing with Warm Restarts
    config_cosine = training_config.copy()
    config_cosine['scheduler'] = 'cosine_restarts'
    config_cosine['T_0'] = 10
    config_cosine['T_mult'] = 2
    
    # Reinitialize model for fair comparison
    model2 = TemporalFusionTransformer(config).to(device)
    trainer2 = TFTTrainer(model2, config_cosine)
    print(f"   Cosine Restarts schedule ready")
    
    print("\n🎯 LR Finder Complete!")
    print("=" * 50)
    print("\nNext steps:")
    print("1. Review the lr_finder_plot.png")
    print("2. Use suggested LR in training: bash scripts/train.sh")
    print("3. Monitor training with different schedulers")
    
    # Save optimal LR to config
    import json
    lr_config = {
        'optimal_lr': float(best_lr),
        'conservative_lr': float(best_lr / 10),
        'aggressive_lr': float(best_lr * 2),
        'scheduler_recommendations': {
            'onecycle': {
                'max_lr': float(best_lr * 10),
                'base_lr': float(best_lr / 25)
            },
            'cosine_restarts': {
                'base_lr': float(best_lr),
                'eta_min': float(best_lr / 100)
            },
            'plateau': {
                'initial_lr': float(best_lr),
                'min_lr': float(best_lr / 1000)
            }
        }
    }
    
    with open('config/optimal_lr.json', 'w') as f:
        json.dump(lr_config, f, indent=2)
    print(f"\n💾 Optimal LR config saved to config/optimal_lr.json")

if __name__ == "__main__":
    main()
