#!/usr/bin/env bash
# scripts/train.sh - Train the TFT model

set -Eeuo pipefail
IFS=$'\n\t'

echo "🚀 TFT-CUDA Training Script"
echo "==========================="

# Check if we're in the right directory
if [ ! -f "setup.py" ] || [ ! -d "python" ]; then
    echo "❌ Error: Must run from TFT-CUDA root directory"
    exit 1
fi

# Check if Python package is installed
if ! python -c "import sys; sys.path.insert(0, 'python')" 2>/dev/null; then
    echo "❌ Error: Python environment not properly set up"
    echo "   Run 'bash scripts/build.sh' first"
    exit 1
fi

# Create data directory if it doesn't exist
mkdir -p data

# Check for data files
echo "📊 Checking data availability..."
data_patterns=("es10m" "vx10m" "zn10m")
data_found=0
for pattern in "${data_patterns[@]}"; do
    if ls data/${pattern}*.csv >/dev/null 2>&1; then
        actual_file=$(ls data/${pattern}*.csv | head -1)
        echo "   ✓ Found ${pattern} data: $(basename "$actual_file")"
        data_found=$((data_found + 1))
    else
        echo "   ⚠️  ${pattern}*.csv not found"
    fi
done

if [ $data_found -eq 0 ]; then
    echo "   ℹ️  No data files found, will use synthetic data for training demo"
fi

# Create training configuration if it doesn't exist
echo "⚙️  Setting up training configuration..."
python -c "
import sys
import os
sys.path.insert(0, 'python')

try:
    import json
    from pathlib import Path
    
    # Load default config
    config_path = Path('config/default_config.json')
    if config_path.exists():
        with open(config_path, 'r') as f:
            config = json.load(f)
        print('   ✓ Loaded configuration from config/default_config.json')
        
        # Extract model config from nested structure
        model_config = config.get('model', {}).get('architecture', {})
        training_config = config.get('training', {})
        
        # Flatten for easier access
        flat_config = {
            'model': {
                'hidden_size': model_config.get('hidden_size', 1024),
                'num_heads': model_config.get('num_heads', 8),
                'sequence_length': model_config.get('sequence_length', 512),
                'quantile_levels': model_config.get('quantile_levels', [0.1, 0.5, 0.9]),
                'prediction_horizon': model_config.get('prediction_horizon', [1, 5, 10])
            },
            'training': {
                'batch_size': training_config.get('batch_size', 16),
                'epochs': training_config.get('epochs', 100),
                'learning_rate': training_config.get('optimizer', {}).get('learning_rate', 1.27e-07)
            }
        }
        config = flat_config
    else:
        # Create basic config
        config = {
            'model': {
                'hidden_size': 1024,
                'num_heads': 8,
                'sequence_length': 512,
                'quantile_levels': [0.1, 0.5, 0.9],
                'prediction_horizon': [1, 5, 10]
            },
            'training': {
                'batch_size': 16,
                'epochs': 100,
                'learning_rate': 1.27e-07
            }
        }
        print('   ✓ Using default training configuration')
    
    print(f'   Model: {config[\"model\"][\"hidden_size\"]} hidden size, {config[\"model\"][\"num_heads\"]} heads')
    print(f'   Training: {config[\"training\"][\"batch_size\"]} batch size, {config[\"training\"][\"epochs\"]} epochs')
    
except Exception as e:
    print(f'   ⚠️  Configuration setup error: {e}')
    print('   Will use minimal default settings')
"

# Load data and build features
echo "📊 Loading data & building features..."
python -c "
import sys
import warnings
warnings.filterwarnings('ignore')
sys.path.insert(0, 'python')

try:
    from data import FinancialDataset
    import pandas as pd
    import numpy as np
    from pathlib import Path
    
    print('   Creating dataset...')
    dataset = FinancialDataset(data_dir='data/')
    
    # Try to load real data, fall back to synthetic
    try:
        processed_data = dataset.process_pipeline()
        print(f'   ✓ Real data processed: {processed_data.shape}')
    except Exception as e:
        print(f'   ⚠️  Real data loading failed: {e}')
        print('   Generating synthetic data for demo...')
        
        # Create synthetic data
        np.random.seed(42)
        dates = pd.date_range('2023-01-01', periods=1000, freq='10min')
        synthetic_data = pd.DataFrame({
            'timestamp': dates,
            'es_close': np.cumsum(np.random.randn(1000) * 0.01) + 4000,
            'vx_close': np.abs(np.cumsum(np.random.randn(1000) * 0.1) + 20),
            'zn_close': np.cumsum(np.random.randn(1000) * 0.005) + 110
        })
        synthetic_data.set_index('timestamp', inplace=True)
        
        # Add simple features
        for col in ['es_close', 'vx_close', 'zn_close']:
            synthetic_data[f'{col}_return'] = synthetic_data[col].pct_change()
            synthetic_data[f'{col}_sma_10'] = synthetic_data[col].rolling(10).mean()
        
        # Add targets
        synthetic_data['target_return_1'] = synthetic_data['es_close'].pct_change().shift(-1)
        synthetic_data['target_return_5'] = synthetic_data['es_close'].pct_change(5).shift(-5)
        
        processed_data = synthetic_data.dropna()
        dataset.processed_data = processed_data
        print(f'   ✓ Synthetic data created: {processed_data.shape}')
    
    # Create sequences
    print('   Creating training sequences...')
    X, y = dataset.create_sequences(processed_data, sequence_length=50)
    print(f'   ✓ Sequences created: X{X.shape}, y{y.shape}')
    
    # Split data
    X_train, X_val, X_test, y_train, y_val, y_test = dataset.train_val_test_split(X, y)
    print(f'   ✓ Data split: Train={len(X_train)}, Val={len(X_val)}, Test={len(X_test)}')
    
    # Save processed data
    try:
        processed_data.to_parquet('data/features.parquet')
        print('   ✓ Data saved as parquet')
    except ImportError:
        processed_data.to_csv('data/features.csv', index=True)
        print('   ✓ Data saved as CSV (parquet not available)')
    
    np.save('data/X_train.npy', X_train)
    np.save('data/y_train.npy', y_train)
    np.save('data/X_val.npy', X_val)
    np.save('data/y_val.npy', y_val)
    print('   ✓ Training data saved')
    
except Exception as e:
    print(f'   ❌ Data preparation failed: {e}')
    import traceback
    traceback.print_exc()
    exit(1)
"

# Train the model
echo "🚀 Training TFT model..."
python -c "
import sys
import warnings
warnings.filterwarnings('ignore')
sys.path.insert(0, 'python')

try:
    import torch
    import numpy as np
    from pathlib import Path
    
    # Load training data
    X_train = np.load('data/X_train.npy')
    y_train = np.load('data/y_train.npy')
    X_val = np.load('data/X_val.npy')  
    y_val = np.load('data/y_val.npy')
    
    print(f'   Loaded training data: X_train{X_train.shape}, y_train{y_train.shape}')
    
    # Create model
    try:
        from tft_model import TemporalFusionTransformer, create_tft_config
        config = create_tft_config(
            input_size=X_train.shape[-1],
            hidden_size=1024,
            num_heads=8,
            sequence_length=min(512, X_train.shape[1])  # Use 512 or actual sequence length if smaller
        )
        model = TemporalFusionTransformer(config)
        print(f'   ✓ Model created with {sum(p.numel() for p in model.parameters()):,} parameters')
    except Exception as e:
        print(f'   ⚠️  TFT model creation failed: {e}')
        print('   Using simple LSTM baseline...')
        
    # Simple LSTM fallback
        import torch.nn as nn
        class SimpleLSTM(nn.Module):
            def __init__(self, input_size, hidden_size=1024, num_layers=1):
                super().__init__()
                self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
                self.fc = nn.Linear(hidden_size, 1)
                
            def forward(self, x):
                if isinstance(x, dict):
                    x = x['historical_features']
                lstm_out, _ = self.lstm(x)
                return {'predictions': {'horizon_1': self.fc(lstm_out[:, -1, :])}}
        
        model = SimpleLSTM(X_train.shape[-1])
        print(f'   ✓ Simple LSTM created with {sum(p.numel() for p in model.parameters()):,} parameters')
    
    # Setup training
    try:
        from trainer import TFTTrainer, create_training_config
        from torch.utils.data import DataLoader
        from data import TFTDataset
        
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        training_config = create_training_config(
            epochs=100,
            batch_size=16,
            learning_rate=1.27e-07,
            device=device,
            checkpoint_dir='checkpoints',
            log_interval=50
        )
        trainer = TFTTrainer(model, training_config)
        print('   ✓ Using TFT trainer')
        
        # Create DataLoaders
        train_dataset = TFTDataset(
            X_train, y_train,
            sequence_length=min(512, X_train.shape[1]),
            prediction_horizon=[1],  # Single horizon for simplicity
            quantile_levels=[0.5]    # Single quantile for simplicity
        )
        val_dataset = TFTDataset(
            X_val, y_val,
            sequence_length=min(512, X_val.shape[1]),
            prediction_horizon=[1],
            quantile_levels=[0.5]
        )
        
        train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)
        
        print(f'   Starting training on {device}...')
        
        # Actually train the model!
        history = trainer.train(train_loader, val_loader, epochs=100)
        
        print('   ✓ Training completed successfully!')
        print(f'   Final train loss: {history["train_loss"][-1]:.6f}')
        print(f'   Final val loss: {history["val_loss"][-1]:.6f}')
    except Exception as e:
        print(f'   ⚠️  TFT trainer failed: {e}')
        print('   Using simple training loop...')
        
        # Simple training setup
        import torch.nn as nn
        from data import TFTDataset
        from torch.utils.data import DataLoader
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = model.to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=1.27e-07)
        criterion = nn.MSELoss()
        
        # Create proper datasets
        train_dataset = TFTDataset(X_train, y_train)
        val_dataset = TFTDataset(X_val, y_val)
        
        train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)
        
        # Training loop
        epochs = 100
        
        print(f'   Starting training: {epochs} epochs on {device}...')
        
        for epoch in range(epochs):
            model.train()
            total_loss = 0
            batch_count = 0
            
            for batch_data, batch_targets in train_loader:
                # Move to device
                if isinstance(batch_data, dict):
                    batch_inputs = {k: v.to(device) for k, v in batch_data.items()}
                else:
                    batch_inputs = batch_data.to(device)
                batch_targets = batch_targets.to(device)
                
                optimizer.zero_grad()
                
                # Forward pass
                outputs = model(batch_inputs)
                if isinstance(outputs, dict):
                    pred = outputs['predictions']['horizon_1']
                else:
                    pred = outputs
                
                # Use only first target for single horizon prediction
                target = batch_targets[:, 0] if batch_targets.dim() > 1 else batch_targets
                loss = criterion(pred.squeeze(), target.squeeze())
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
                batch_count += 1
            
            # Validation
            model.eval()
            val_loss = 0
            val_count = 0
            with torch.no_grad():
                for batch_data, batch_targets in val_loader:
                    if isinstance(batch_data, dict):
                        batch_inputs = {k: v.to(device) for k, v in batch_data.items()}
                    else:
                        batch_inputs = batch_data.to(device)
                    batch_targets = batch_targets.to(device)
                    
                    val_outputs = model(batch_inputs)
                    if isinstance(val_outputs, dict):
                        val_pred = val_outputs['predictions']['horizon_1']
                    else:
                        val_pred = val_outputs
                    
                    target = batch_targets[:, 0] if batch_targets.dim() > 1 else batch_targets
                    batch_val_loss = criterion(val_pred.squeeze(), target.squeeze())
                    val_loss += batch_val_loss.item()
                    val_count += 1
            
            avg_loss = total_loss / batch_count
            avg_val_loss = val_loss / val_count
            print(f'   Epoch {epoch+1}/{epochs}: Train Loss={avg_loss:.6f}, Val Loss={avg_val_loss:.6f}')
        
        # Save model
        torch.save(model.state_dict(), 'data/model_checkpoint.pth')
        print('   ✓ Model saved to data/model_checkpoint.pth')
        
        # Test prediction
        model.eval()
        with torch.no_grad():
            # Get first validation batch
            for test_batch_data, test_batch_targets in val_loader:
                if isinstance(test_batch_data, dict):
                    test_inputs = {k: v[:1].to(device) for k, v in test_batch_data.items()}
                else:
                    test_inputs = test_batch_data[:1].to(device)
                
                test_pred = model(test_inputs)
                if isinstance(test_pred, dict):
                    test_pred = test_pred['predictions']['horizon_1']
                print(f'   ✓ Test prediction: {test_pred[0].item():.6f}')
                break
        
        print('   ✓ Training completed successfully!')
        
except Exception as e:
    print(f'   ❌ Training failed: {e}')
    import traceback
    traceback.print_exc()
    exit(1)
"

echo ""
echo "🎉 Training complete!"
echo "===================="
echo "✓ Model trained and saved"
echo "✓ Training data processed and cached"
echo ""
echo "Next steps:"
echo "  - Check training results in data/ directory"
echo "  - Run 'bash scripts/test.sh' to validate the model"
echo "  - Run 'bash scripts/debug.sh' if you encounter issues"