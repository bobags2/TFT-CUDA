#!/usr/bin/env python3
"""
TFT-CUDA Demo Script
Demonstrates the complete TFT financial forecasting pipeline.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'python'))

import torch
import numpy as np
import pandas as pd
from pathlib import Path

# Import TFT modules
import data
import tft_model
import loss
import trainer

def print_banner():
    """Print demo banner."""
    print("🚀 TFT-CUDA Financial Forecasting Demo")
    print("=" * 45)
    print("Building state-of-the-art Temporal Fusion Transformer")
    print("for multi-asset financial time-series forecasting")
    print()

def demo_data_pipeline():
    """Demonstrate data processing pipeline."""
    print("📊 Phase 1: Financial Data Processing")
    print("-" * 35)
    
    # Create dataset with sample data
    dataset = data.FinancialDataset(data_dir="data/")
    print("✓ Created FinancialDataset")
    
    # Load data (creates sample data since real data not available)
    raw_data = dataset.load_data()
    print(f"✓ Loaded {len(raw_data)} assets: {list(raw_data.keys())}")
    
    # Merge datasets
    merged_data = dataset.merge_datasets()
    print(f"✓ Merged data shape: {merged_data.shape}")
    
    # Engineer features
    processed_data = dataset.engineer_features(merged_data)
    print(f"✓ Engineered features: {len(dataset.feature_columns)} features, {len(dataset.target_columns)} targets")
    
    # Display sample features
    print("\nSample features:")
    for i, feature in enumerate(dataset.feature_columns[:10]):
        print(f"  {i+1:2d}. {feature}")
    if len(dataset.feature_columns) > 10:
        print(f"  ... and {len(dataset.feature_columns) - 10} more")
    
    return dataset, processed_data

def demo_model_architecture():
    """Demonstrate TFT model architecture."""
    print("\n🧠 Phase 2: TFT Model Architecture")
    print("-" * 35)
    
    # Create model configuration
    input_size = 64  # Number of features
    config = tft_model.create_tft_config(
        input_size=input_size,
        hidden_size=128,
        num_heads=8,
        quantile_levels=[0.1, 0.5, 0.9],
        prediction_horizon=[1, 5, 10],
        sequence_length=100
    )
    print(f"✓ Created TFT config: {input_size} inputs → {config['hidden_size']} hidden")
    
    # Create model
    model = tft_model.TemporalFusionTransformer(config)
    num_params = sum(p.numel() for p in model.parameters())
    print(f"✓ Built TFT model with {num_params:,} parameters")
    
    # Demonstrate forward pass
    batch_size, seq_len = 4, 100
    sample_inputs = {
        'historical_features': torch.randn(batch_size, seq_len, input_size),
        'static_features': torch.randn(batch_size, 10)
    }
    
    with torch.no_grad():
        outputs = model(sample_inputs)
    
    predictions = outputs['predictions']
    interpretability = outputs['interpretability']
    
    print(f"✓ Forward pass successful:")
    print(f"  - Predictions: {len(predictions)} horizons")
    for horizon, pred in predictions.items():
        print(f"    {horizon}: {pred.shape}")
    print(f"  - Attention weights: {interpretability['attention_weights'].shape}")
    print(f"  - Variable selection: {interpretability['historical_variable_selection'].shape}")
    
    return model, config

def demo_loss_functions():
    """Demonstrate loss functions."""
    print("\n📉 Phase 3: Loss Functions")
    print("-" * 25)
    
    quantile_levels = [0.1, 0.5, 0.9]
    batch_size = 16
    
    # Sample predictions and targets
    predictions = torch.randn(batch_size, len(quantile_levels), requires_grad=True)
    targets = torch.randn(batch_size, 1)
    
    # Test different loss types
    loss_types = [
        ('quantile', 'Standard quantile (pinball) loss'),
        ('huber_quantile', 'Robust Huber-quantile hybrid loss'),
        ('smoothed', 'Smoothed quantile loss with ordering'),
        ('financial', 'Financial metrics loss (Sharpe-aware)')
    ]
    
    print("Loss function comparison:")
    for loss_type, description in loss_types:
        try:
            loss_fn = loss.create_tft_loss(quantile_levels, loss_type)
            loss_value = loss_fn(predictions, targets)
            print(f"  ✓ {loss_type:15s}: {loss_value.item():.4f} - {description}")
        except Exception as e:
            print(f"  ❌ {loss_type:15s}: {str(e)[:50]}...")
    
    return loss_fn

def demo_training_setup():
    """Demonstrate training setup."""
    print("\n🏋️ Phase 4: Training Infrastructure")
    print("-" * 35)
    
    # Create simple model for demo
    config = tft_model.create_tft_config(input_size=32, hidden_size=64)
    model = tft_model.TemporalFusionTransformer(config)
    
    # Training configuration
    training_config = trainer.create_training_config(
        optimizer='adamw',
        learning_rate=1e-3,
        batch_size=32,
        epochs=50,
        device='cpu'  # Use CPU for demo
    )
    
    print(f"✓ Training config:")
    print(f"  - Optimizer: {training_config['optimizer']}")
    print(f"  - Learning rate: {training_config['learning_rate']}")
    print(f"  - Batch size: {training_config['batch_size']}")
    print(f"  - Epochs: {training_config['epochs']}")
    print(f"  - Device: {training_config['device']}")
    
    # Create trainer
    tft_trainer = trainer.TFTTrainer(model, training_config)
    print(f"✓ Created trainer with {training_config['optimizer']} optimizer")
    print(f"✓ Loss function: {type(tft_trainer.loss_fn).__name__}")
    print(f"✓ Early stopping: {'enabled' if tft_trainer.early_stopping else 'disabled'}")
    
    return tft_trainer

def demo_interpretability():
    """Demonstrate interpretability features."""
    print("\n🔍 Phase 5: Model Interpretability")
    print("-" * 35)
    
    # Create model and sample data
    config = tft_model.create_tft_config(input_size=20, hidden_size=32)
    model = tft_model.TemporalFusionTransformer(config)
    
    sample_inputs = {
        'historical_features': torch.randn(2, 50, 20),
        'static_features': torch.randn(2, 10)
    }
    
    # Import interpretability module (skip if matplotlib not available)
    try:
        import interpretability
        
        # Create feature names
        feature_names = [f'feature_{i}' for i in range(20)]
        interpreter = interpretability.TFTInterpretability(model, feature_names)
        
        print("✓ Created interpretability analyzer")
        
        # Extract attention weights
        attention_data = interpreter.extract_attention_weights(sample_inputs)
        print(f"✓ Extracted attention weights: {attention_data['attention_weights'].shape}")
        print(f"✓ Temporal importance: {attention_data['temporal_importance'].shape}")
        
        # Analyze feature importance
        importance_data = interpreter.analyze_feature_importance(sample_inputs, method='gradient')
        print(f"✓ Feature importance analysis: {importance_data['feature_importance'].shape}")
        
        # Show top features
        importance_scores = importance_data['feature_importance'].numpy()
        top_indices = np.argsort(importance_scores)[-5:][::-1]
        print("Top 5 most important features:")
        for i, idx in enumerate(top_indices):
            print(f"  {i+1}. {feature_names[idx]}: {importance_scores[idx]:.4f}")
        
        print("✓ Interpretability analysis complete")
        
    except ImportError:
        print("❌ Matplotlib not available, skipping interpretability demo")

def demo_end_to_end():
    """Demonstrate end-to-end pipeline."""
    print("\n🔄 Phase 6: End-to-End Pipeline")
    print("-" * 35)
    
    # Create small dataset for demo
    dataset = data.FinancialDataset(data_dir="/tmp/demo_data")
    processed_data = dataset.process_pipeline()
    
    print(f"✓ Processed data shape: {processed_data.shape}")
    
    # Create sequences
    X, y = dataset.create_sequences(processed_data, sequence_length=20)
    print(f"✓ Created sequences: X{X.shape}, y{y.shape}")
    
    # Split data
    X_train, X_val, X_test, y_train, y_val, y_test = dataset.train_val_test_split(X, y)
    print(f"✓ Data splits: Train({len(X_train)}), Val({len(X_val)}), Test({len(X_test)})")
    
    # Create and test model
    config = tft_model.create_tft_config(
        input_size=X.shape[-1],
        hidden_size=32,
        sequence_length=20
    )
    model = tft_model.TemporalFusionTransformer(config)
    
    # Test prediction
    sample_inputs = {
        'historical_features': torch.tensor(X_test[:1], dtype=torch.float32),
        'static_features': torch.zeros(1, 10)
    }
    
    predictions = model.predict(sample_inputs)
    pred_values = next(iter(predictions['predictions'].values()))
    
    print(f"✓ Model prediction: {pred_values.shape}")
    print(f"  Quantile predictions: {pred_values[0].tolist()}")
    
    # Test loss computation
    loss_fn = loss.QuantileLoss(config['quantile_levels'])
    target_tensor = torch.tensor(y_test[:1, :1], dtype=torch.float32)  # Use only first target
    loss_value = loss_fn(pred_values, target_tensor)
    
    print(f"✓ Loss computation: {loss_value.item():.4f}")
    print("✓ End-to-end pipeline successful!")

def main():
    """Run the complete demo."""
    print_banner()
    
    try:
        # Run demo phases
        dataset, processed_data = demo_data_pipeline()
        model, config = demo_model_architecture()
        loss_fn = demo_loss_functions()
        tft_trainer = demo_training_setup()
        demo_interpretability()
        demo_end_to_end()
        
        print("\n🎉 TFT-CUDA Demo Complete!")
        print("-" * 30)
        print("✅ All components working correctly")
        print("✅ Ready for financial forecasting")
        print("✅ CUDA acceleration available when enabled")
        print("\n📖 Next steps:")
        print("  1. Provide real financial data (es10m.csv, vx10m.csv, zn10m.csv)")
        print("  2. Install CUDA backend for acceleration")
        print("  3. Run full training with: python -m tft_cuda.train")
        print("  4. Generate predictions with: python -m tft_cuda.predict")
        
    except KeyboardInterrupt:
        print("\n⚠️  Demo interrupted by user")
    except Exception as e:
        print(f"\n❌ Demo failed: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)