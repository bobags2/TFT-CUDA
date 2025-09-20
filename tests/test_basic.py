"""
Basic test for TFT-CUDA implementation.
Tests core functionality without requiring CUDA.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'python'))

import torch
import numpy as np
# import pytest  # Not needed for basic test

# Fix relative import issues by directly importing modules
import data
import tft_model
import loss
import trainer

def test_financial_dataset():
    """Test financial dataset creation and processing."""
    print("Testing FinancialDataset...")
    
    # Create dataset with sample data (since real data may not be available)
    dataset = data.FinancialDataset(data_dir="/tmp/test_data")
    
    # Load data (will create sample data)
    raw_data = dataset.load_data()
    
    assert len(raw_data) == 3, f"Expected 3 assets, got {len(raw_data)}"
    assert 'es' in data, "ES data not found"
    assert 'vx' in data, "VX data not found" 
    assert 'zn' in data, "ZN data not found"
    
    # Test merging
    merged_data = dataset.merge_datasets()
    assert len(merged_data) > 0, "Merged data is empty"
    
    # Test feature engineering
    processed_data = dataset.engineer_features(merged_data)
    assert len(processed_data.columns) > len(merged_data.columns), "No features were engineered"
    
    print(f"✓ Dataset test passed. Processed shape: {processed_data.shape}")
    # Don't return values from test functions

def test_tft_model():
    """Test TFT model creation and forward pass."""
    print("Testing TFT model...")
    
    # Create model config
    input_size = 32
    config = tft_model.create_tft_config(input_size=input_size)
    
    # Create model
    model = tft_model.TemporalFusionTransformer(config)
    
    # Test forward pass
    batch_size, seq_len = 4, 100
    sample_inputs = {
        'historical_features': torch.randn(batch_size, seq_len, input_size),
        'static_features': torch.randn(batch_size, 10)
    }
    
    # Forward pass
    outputs = model(sample_inputs)
    
    # Check outputs
    assert 'predictions' in outputs, "No predictions in output"
    assert 'interpretability' in outputs, "No interpretability in output"
    
    predictions = outputs['predictions']
    assert isinstance(predictions, dict), "Predictions should be a dict"
    
    # Check prediction shapes
    for horizon_key, pred in predictions.items():
        expected_shape = (batch_size, len(config['quantile_levels']))
        assert pred.shape == expected_shape, f"Wrong shape for {horizon_key}: {pred.shape} vs {expected_shape}"
    
    print(f"✓ Model test passed. Predictions shape: {next(iter(predictions.values())).shape}")
    # Test completed successfully

def test_loss_functions():
    """Test loss function implementations."""
    print("Testing loss functions...")
    
    quantile_levels = [0.1, 0.5, 0.9]
    batch_size, num_quantiles = 16, len(quantile_levels)
    
    # Sample data with gradient enabled
    predictions = torch.randn(batch_size, num_quantiles, requires_grad=True)
    targets = torch.randn(batch_size, 1)
    
    # Test different loss types
    loss_types = ['quantile', 'huber_quantile', 'smoothed']
    
    for loss_type in loss_types:
        loss_fn = loss.create_tft_loss(quantile_levels, loss_type)
        loss_value = loss_fn(predictions, targets)
        
        assert loss_value.requires_grad, f"Loss should require gradients for {loss_type}"
        assert loss_value.item() > 0, f"Loss should be positive for {loss_type}"
        
        print(f"  ✓ {loss_type} loss: {loss_value.item():.4f}")
    
    print("✓ Loss function tests passed")

def test_trainer():
    """Test trainer functionality.""" 
    print("Testing trainer...")
    
    # Create simple model and data
    input_size = 16
    config = tft_model.create_tft_config(input_size=input_size, hidden_size=64)
    model = tft_model.TemporalFusionTransformer(config)
    
    # Training config
    training_config = trainer.create_training_config(
        epochs=2,  # Very short for testing
        batch_size=8,
        device='cpu'  # Force CPU for testing
    )
    
    # Create trainer
    tft_trainer = trainer.TFTTrainer(model, training_config)
    
    # Check trainer components
    assert tft_trainer.optimizer is not None, "Optimizer not created"
    assert tft_trainer.loss_fn is not None, "Loss function not created"
    
    print("✓ Trainer test passed")
    # Trainer test completed

def test_integration():
    """Test full integration pipeline."""
    print("Testing integration pipeline...")
    
    # 1. Create dataset
    dataset = data.FinancialDataset(data_dir="/tmp/test_data")
    raw_data = dataset.load_data()
    merged_data = dataset.merge_datasets()
    processed_data = dataset.engineer_features(merged_data)
    
    # 2. Create sequences (small for testing)
    X, y = dataset.create_sequences(processed_data, sequence_length=20)
    
    # Ensure we have enough data
    if len(X) < 10:
        print("Warning: Very small dataset for testing")
        return
    
    # 3. Create model
    input_size = X.shape[-1]
    config = tft_model.create_tft_config(
        input_size=input_size,
        hidden_size=32,  # Small for testing
        sequence_length=20
    )
    model = tft_model.TemporalFusionTransformer(config)
    
    # 4. Test prediction
    sample_X = X[:2]  # Take 2 samples
    sample_inputs = {
        'historical_features': torch.tensor(sample_X, dtype=torch.float32),
        'static_features': torch.zeros(2, 10)
    }
    
    predictions = model.predict(sample_inputs)
    assert 'predictions' in predictions, "No predictions from model"
    
    print(f"✓ Integration test passed. Feature shape: {X.shape}")

def run_all_tests():
    """Run all tests."""
    print("Running TFT-CUDA Tests")
    print("=" * 30)
    
    try:
        # Individual component tests
        test_loss_functions()
        test_tft_model()
        test_trainer()
        
        # Integration test
        test_integration()
        
        print("\n🎉 All tests passed!")
        return True
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)