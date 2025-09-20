"""
Simple test for TFT-CUDA core functionality.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'python'))

import torch
import numpy as np

def test_data_module():
    """Test data module functionality."""
    print("Testing data module...")
    
    try:
        import data
        
        # Create dataset with actual data directory
        dataset = data.FinancialDataset(data_dir="data/")
        
        # Test sample data creation
        sample_data = dataset._create_sample_data('es')
        assert len(sample_data) > 0, "Sample data creation failed"
        
        print(f"✓ Data module test passed. Sample data shape: {sample_data.shape}")
        
    except Exception as e:
        print(f"❌ Data module test failed: {e}")
        raise

def test_model_module():
    """Test TFT model module."""
    print("Testing TFT model module...")
    
    try:
        import tft_model
        
        # Create simple config
        config = {
            'input_size': 32,
            'hidden_size': 64,
            'num_heads': 4,
            'quantile_levels': [0.1, 0.5, 0.9],
            'prediction_horizon': [1],
            'sequence_length': 50,
            'dropout_rate': 0.1
        }
        
        # Create model
        model = tft_model.TemporalFusionTransformer(config)
        
        # Test forward pass
        batch_size, seq_len, input_size = 2, 50, 32
        sample_inputs = {
            'historical_features': torch.randn(batch_size, seq_len, input_size),
            'static_features': torch.randn(batch_size, 10)
        }
        
        outputs = model(sample_inputs)
        
        assert 'predictions' in outputs, "No predictions in output"
        predictions = outputs['predictions']
        
        # Check shapes
        for horizon_key, pred in predictions.items():
            expected_shape = (batch_size, len(config['quantile_levels']))
            assert pred.shape == expected_shape, f"Wrong prediction shape: {pred.shape} vs {expected_shape}"
        
        print(f"✓ Model module test passed. Output shape: {next(iter(predictions.values())).shape}")
        
    except Exception as e:
        print(f"❌ Model module test failed: {e}")
        import traceback
        traceback.print_exc()
        raise

def test_loss_module():
    """Test loss module."""
    print("Testing loss module...")
    
    try:
        import loss
        
        quantile_levels = [0.1, 0.5, 0.9]
        batch_size = 8
        
        # Sample data
        predictions = torch.randn(batch_size, len(quantile_levels))
        targets = torch.randn(batch_size, 1)
        
        # Test quantile loss
        predictions.requires_grad_(True)  # Enable gradients
        loss_fn = loss.QuantileLoss(quantile_levels)
        loss_value = loss_fn(predictions, targets)
        
        assert loss_value.requires_grad, "Loss should require gradients"
        assert loss_value.item() > 0, "Loss should be positive"
        
        print(f"✓ Loss module test passed. Loss value: {loss_value.item():.4f}")
        
    except Exception as e:
        print(f"❌ Loss module test failed: {e}")
        import traceback
        traceback.print_exc()
        raise

def test_end_to_end():
    """Test end-to-end functionality."""
    print("Testing end-to-end pipeline...")
    
    try:
        import data
        import tft_model
        import loss
        
        # 1. Create sample dataset with actual data directory
        dataset = data.FinancialDataset(data_dir="data/")
        
        # Create minimal data for testing
        sample_data = {}
        for asset in ['es', 'vx', 'zn']:
            sample_data[asset] = dataset._create_sample_data(asset)
        
        dataset.raw_data = sample_data
        merged_data = dataset.merge_datasets()
        
        # Simple feature engineering (subset)
        processed_data = merged_data.copy()
        for asset in ['es', 'vx', 'zn']:
            close_col = f'{asset}_close'
            if close_col in processed_data.columns:
                processed_data[f'{asset}_log_return'] = np.log(processed_data[close_col] / processed_data[close_col].shift(1))
        
        # Add target
        processed_data['target_return_1'] = processed_data['es_log_return'].shift(-1)
        processed_data = processed_data.dropna()
        
        # 2. Create sequences
        feature_cols = [col for col in processed_data.columns if not col.startswith('target_')]
        target_cols = [col for col in processed_data.columns if col.startswith('target_')]
        
        X_data = processed_data[feature_cols].values
        y_data = processed_data[target_cols].values
        
        seq_len = 20
        if len(X_data) > seq_len:
            X_seq = X_data[:seq_len].reshape(1, seq_len, -1)
            y_seq = y_data[seq_len:seq_len+1]
        else:
            print("Not enough data for sequence test")
            return True
        
        # 3. Create model
        config = {
            'input_size': X_seq.shape[-1],
            'hidden_size': 32,
            'num_heads': 2,
            'quantile_levels': [0.1, 0.5, 0.9],
            'prediction_horizon': [1],
            'sequence_length': seq_len,
            'dropout_rate': 0.1
        }
        
        model = tft_model.TemporalFusionTransformer(config)
        
        # 4. Test prediction
        sample_inputs = {
            'historical_features': torch.tensor(X_seq, dtype=torch.float32),
            'static_features': torch.zeros(1, 10)
        }
        
        predictions = model.predict(sample_inputs)
        
        # 5. Test loss
        loss_fn = loss.QuantileLoss(config['quantile_levels'])
        pred_tensor = next(iter(predictions['predictions'].values()))
        target_tensor = torch.tensor(y_seq, dtype=torch.float32)
        
        loss_value = loss_fn(pred_tensor, target_tensor)
        
        print(f"✓ End-to-end test passed. Loss: {loss_value.item():.4f}")
        
    except Exception as e:
        print(f"❌ End-to-end test failed: {e}")
        import traceback
        traceback.print_exc()
        raise

def run_tests():
    """Run all tests."""
    print("Running TFT-CUDA Core Tests")
    print("=" * 35)
    
    tests = [
        test_data_module,
        test_model_module,
        test_loss_module,
        test_end_to_end
    ]
    
    passed = 0
    total = len(tests)
    
    for test_func in tests:
        try:
            test_func()
            passed += 1
        except:
            pass
        print()
    
    print(f"Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed!")
        return True
    else:
        print(f"❌ {total - passed} tests failed")
        return False

if __name__ == "__main__":
    success = run_tests()
    sys.exit(0 if success else 1)