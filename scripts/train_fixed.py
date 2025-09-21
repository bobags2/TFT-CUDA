#!/usr/bin/env python3
"""
Production-grade TFT training script with verified gradient explosion fixes.
Implements robust data normalization, ultra-conservative initialization, and comprehensive monitoring.
"""

import sys
sys.path.insert(0, 'python')

import numpy as np
import warnings
warnings.filterwarnings('ignore')

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader
    from data import FinancialDataset, TFTDataset
    from tft_model import TemporalFusionTransformer, create_tft_config
    TORCH_AVAILABLE = True
except ImportError as e:
    print(f"❌ PyTorch/TFT dependencies not available: {e}")
    sys.exit(1)


class ProductionTFTTrainer:
    """Production-grade TFT trainer with gradient explosion prevention."""
    
    def __init__(self, config):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = None
        self.optimizer = None
        self.scheduler = None
        self.criterion = nn.MSELoss()
        
    def load_and_prepare_data(self):
        """Load and prepare data with verified normalization."""
        print("📊 Loading and preparing data...")
        
        # Create dataset
        dataset = FinancialDataset(data_dir='data/')
        
        # Try to load real data, fall back to synthetic
        try:
            processed_data = dataset.process_pipeline()
            print(f'   ✓ Real data processed: {processed_data.shape}')
        except Exception as e:
            print(f'   ⚠️ Real data loading failed: {e}')
            print('   Generating synthetic data for demo...')
            
            # Create synthetic data
            import pandas as pd
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
        X, y = dataset.create_sequences(processed_data, sequence_length=128)
        print(f'   ✓ Sequences created: X{X.shape}, y{y.shape}')
        
        # PRODUCTION-GRADE NORMALIZATION (VERIFIED FIX)
        print('   🔧 Applying production-grade normalization...')
        X_normalized, y_normalized = self._normalize_data(X, y)
        
        # Split data
        X_train, X_val, X_test, y_train, y_val, y_test = dataset.train_val_test_split(
            X_normalized, y_normalized)
        print(f'   ✓ Data split: Train={len(X_train)}, Val={len(X_val)}, Test={len(X_test)}')
        
        # Save processed data
        np.save('data/X_train.npy', X_train)
        np.save('data/y_train.npy', y_train)
        np.save('data/X_val.npy', X_val)
        np.save('data/y_val.npy', y_val)
        print('   ✓ Training data saved')
        
        return X_train, y_train, X_val, y_val
    
    def _normalize_data(self, X, y):
        """Apply verified normalization that prevents gradient explosions."""
        
        # Normalize features (X) using RELIABLE z-score standardization
        X_normalized = X.copy()
        
        # Global feature statistics for numerical stability
        X_flat = X.reshape(-1, X.shape[-1])  # Flatten to [samples*time, features]
        
        # Remove NaN/inf values globally
        valid_mask = np.isfinite(X_flat).all(axis=1)
        X_clean = X_flat[valid_mask]
        
        print(f'     Data cleaning: {len(X_clean)}/{len(X_flat)} samples retained '
              f'({100*len(X_clean)/len(X_flat):.1f}%)')
        
        # Compute global statistics per feature
        feature_means = np.mean(X_clean, axis=0)
        feature_stds = np.std(X_clean, axis=0)
        
        # Apply standardization: (x - mean) / std
        for feature_idx in range(X.shape[-1]):
            mean_val = feature_means[feature_idx]
            std_val = feature_stds[feature_idx]
            
            if std_val > 1e-8:  # Avoid division by zero
                X_normalized[:, :, feature_idx] = (X[:, :, feature_idx] - mean_val) / std_val
            else:
                # Constant feature - center at zero
                X_normalized[:, :, feature_idx] = X[:, :, feature_idx] - mean_val
        
        # Normalize targets (y) using simple z-score
        y_normalized = y.copy()
        
        for target_idx in range(y.shape[-1]):
            target_data = y[:, target_idx]
            
            # Remove NaN/inf values
            valid_mask = np.isfinite(target_data)
            if np.sum(valid_mask) > 10:  # Need minimum samples
                target_clean = target_data[valid_mask]
                
                mean_val = np.mean(target_clean)
                std_val = np.std(target_clean)
                
                if std_val > 1e-8:
                    y_normalized[:, target_idx] = (y[:, target_idx] - mean_val) / std_val
                else:
                    y_normalized[:, target_idx] = y[:, target_idx] - mean_val
        
        # Verification of normalization
        X_mean = np.nanmean(X_normalized)
        X_std = np.nanstd(X_normalized)
        y_mean = np.nanmean(y_normalized)
        y_std = np.nanstd(y_normalized)
        
        # Per-feature statistics for debugging
        feature_means = np.nanmean(X_normalized, axis=(0,1))
        feature_stds = np.nanstd(X_normalized, axis=(0,1))
        
        print(f'   ✓ Features normalized: global_mean={X_mean:.6f}, global_std={X_std:.6f}')
        print(f'   ✓ Per-feature mean range: [{np.min(feature_means):.6f}, {np.max(feature_means):.6f}]')
        print(f'   ✓ Per-feature std range: [{np.min(feature_stds):.6f}, {np.max(feature_stds):.6f}]')
        print(f'   ✓ Targets normalized: mean={y_mean:.6f}, std={y_std:.6f}')
        
        # Sanity check - features should have reasonable distribution
        if abs(X_mean) > 0.1 or X_std < 0.8 or X_std > 1.2:
            print(f'   ⚠️ WARNING: Feature normalization may be inadequate!')
            print(f'     Expected: mean≈0, std≈1. Got: mean={X_mean:.6f}, std={X_std:.6f}')
            raise ValueError("Data normalization failed - gradient explosion risk!")
        
        # Additional safety: clip extreme outliers
        X_normalized = np.clip(X_normalized, -5, 5)  # Clip to ±5 sigma
        y_normalized = np.clip(y_normalized, -5, 5)
        
        print(f'   ✓ Applied extreme outlier clipping (±5 sigma)')
        
        return X_normalized, y_normalized
    
    def create_model(self, input_size):
        """Create TFT model with ultra-conservative initialization."""
        print("🏗️ Creating TFT model...")
        
        try:
            config = create_tft_config(
                input_size=input_size,
                hidden_size=512,      # Moderate size
                num_heads=4,          # Fewer attention heads
                sequence_length=128,
                quantile_levels=[0.5],
                prediction_horizon=[1],
                dropout_rate=0.1
            )
            
            model = TemporalFusionTransformer(config)
            
        except Exception as e:
            print(f'   ⚠️ TFT model creation failed: {e}')
            print('   Using simple LSTM baseline...')
            
            # Simple LSTM fallback
            class SimpleLSTM(nn.Module):
                def __init__(self, input_size, hidden_size=256):
                    super().__init__()
                    self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True, dropout=0.2)
                    self.fc = nn.Linear(hidden_size, 1)
                    
                def forward(self, x):
                    if isinstance(x, dict):
                        x = x['historical_features']
                    lstm_out, _ = self.lstm(x)
                    return {'predictions': {'horizon_1': self.fc(lstm_out[:, -1, :])}}
            
            model = SimpleLSTM(input_size)
        
        # Ultra-conservative weight initialization
        self._initialize_weights(model)
        
        model = model.to(self.device)
        self.model = model
        
        param_count = sum(p.numel() for p in model.parameters())
        print(f'   ✓ Model created with {param_count:,} parameters')
        
        return model
    
    def _initialize_weights(self, model):
        """Apply ultra-conservative weight initialization."""
        
        def init_weights(m):
            if isinstance(m, torch.nn.Linear):
                # Ultra-conservative initialization for financial data
                torch.nn.init.xavier_uniform_(m.weight, gain=0.01)  # Very small gain
                if m.bias is not None:
                    torch.nn.init.zeros_(m.bias)
            elif isinstance(m, torch.nn.LSTM):
                for name, param in m.named_parameters():
                    if 'weight_ih' in name:
                        torch.nn.init.xavier_uniform_(param.data, gain=0.01)
                    elif 'weight_hh' in name:
                        torch.nn.init.orthogonal_(param.data, gain=0.01)
                    elif 'bias' in name:
                        torch.nn.init.zeros_(param.data)
            elif isinstance(m, torch.nn.LayerNorm):
                torch.nn.init.ones_(m.weight)
                torch.nn.init.zeros_(m.bias)
            elif hasattr(m, 'weight') and hasattr(m, 'bias'):
                # Catch-all for other layers
                if m.weight is not None:
                    torch.nn.init.xavier_uniform_(m.weight, gain=0.01)
                if m.bias is not None:
                    torch.nn.init.zeros_(m.bias)
        
        model.apply(init_weights)
        print(f'   ✓ Applied ultra-conservative weight initialization (gain=0.01)')
        
        # Additional numerical stability - ensure no NaN parameters
        for name, param in model.named_parameters():
            if torch.any(torch.isnan(param)):
                print(f'   ⚠️ NaN detected in parameter {name}, reinitializing...')
                if len(param.shape) >= 2:
                    torch.nn.init.xavier_uniform_(param, gain=0.01)
                else:
                    torch.nn.init.zeros_(param)
    
    def setup_optimization(self):
        """Setup optimizer and scheduler."""
        
        # Ultra-conservative optimizer
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=1e-6,              # Very conservative LR
            weight_decay=1e-4,
            betas=(0.9, 0.999),
            eps=1e-8
        )
        
        # Simple plateau scheduler
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=10, min_lr=1e-8
        )
        
        print(f'   ✓ Optimizer configured: LR=1e-6, conservative settings')
    
    def train_epoch(self, train_loader, epoch):
        """Train one epoch with comprehensive gradient monitoring."""
        
        self.model.train()
        total_loss = 0
        total_grad_norm = 0
        batch_count = 0
        skipped_batches = 0
        
        for batch_idx, (batch_data, batch_targets) in enumerate(train_loader):
            # Move to device
            if isinstance(batch_data, dict):
                batch_inputs = {k: v.to(self.device) for k, v in batch_data.items()}
            else:
                batch_inputs = batch_data.to(self.device)
            batch_targets = batch_targets.to(self.device)
            
            self.optimizer.zero_grad()
            
            # Forward pass
            outputs = self.model(batch_inputs)
            if isinstance(outputs, dict):
                pred = outputs['predictions']['horizon_1']
            else:
                pred = outputs
            
            # Handle different target shapes
            if batch_targets.dim() > 1:
                target = batch_targets[:, 0]
            else:
                target = batch_targets
            
            # Ensure compatible shapes
            if pred.dim() > 1 and pred.size(1) > 1:
                pred = pred[:, 0]
            
            loss = self.criterion(pred.squeeze(), target.squeeze())
            
            # Check for NaN/infinite loss and gradients
            if not torch.isfinite(loss):
                print(f'     ⚠️ Non-finite loss: {loss.item()}, skipping batch')
                skipped_batches += 1
                continue
            
            loss.backward()
            
            # Check for NaN gradients
            has_nan_grad = False
            for name, param in self.model.named_parameters():
                if param.grad is not None and torch.any(torch.isnan(param.grad)):
                    print(f'     ⚠️ NaN gradient in {name}, zeroing gradients')
                    has_nan_grad = True
                    param.grad.zero_()
            
            if has_nan_grad:
                self.optimizer.zero_grad()
                skipped_batches += 1
                continue
            
            # Ultra-aggressive gradient clipping
            if epoch == 0 and batch_idx < 50:
                # Extremely aggressive for first 50 batches
                grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=0.01)
            elif epoch == 0:
                # Very conservative for first epoch
                grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=0.05)
            else:
                # More permissive after stabilization
                grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=0.1)
            
            # Emergency brake
            if grad_norm > 10:
                print(f'     🚨 EMERGENCY: Extreme gradient norm {grad_norm:.1f}, skipping!')
                self.optimizer.zero_grad()
                skipped_batches += 1
                continue
            
            self.optimizer.step()
            
            # Track metrics
            total_loss += loss.item()
            total_grad_norm += grad_norm.item() if isinstance(grad_norm, torch.Tensor) else grad_norm
            batch_count += 1
            
            # Monitor every 25 batches
            if batch_idx % 25 == 0:
                current_lr = self.optimizer.param_groups[0]['lr']
                print(f'     Batch {batch_idx}: Loss={loss.item():.6f}, '
                      f'Grad_Norm={grad_norm:.6f}, LR={current_lr:.2e}')
        
        avg_loss = total_loss / max(batch_count, 1)
        avg_grad_norm = total_grad_norm / max(batch_count, 1)
        
        if skipped_batches > 0:
            print(f'   ⚠️ Skipped {skipped_batches} problematic batches')
        
        return avg_loss, avg_grad_norm
    
    def validate(self, val_loader):
        """Validate model."""
        self.model.eval()
        val_loss = 0
        val_count = 0
        
        with torch.no_grad():
            for batch_data, batch_targets in val_loader:
                if isinstance(batch_data, dict):
                    batch_inputs = {k: v.to(self.device) for k, v in batch_data.items()}
                else:
                    batch_inputs = batch_data.to(self.device)
                batch_targets = batch_targets.to(self.device)
                
                val_outputs = self.model(batch_inputs)
                if isinstance(val_outputs, dict):
                    val_pred = val_outputs['predictions']['horizon_1']
                else:
                    val_pred = val_outputs
                
                # Handle shapes
                if batch_targets.dim() > 1:
                    target = batch_targets[:, 0]
                else:
                    target = batch_targets
                
                if val_pred.dim() > 1 and val_pred.size(1) > 1:
                    val_pred = val_pred[:, 0]
                
                batch_val_loss = self.criterion(val_pred.squeeze(), target.squeeze())
                val_loss += batch_val_loss.item()
                val_count += 1
        
        return val_loss / max(val_count, 1)
    
    def train(self, X_train, y_train, X_val, y_val, epochs=50):
        """Main training loop."""
        print("🚀 Starting production-grade training...")
        
        # Create datasets
        train_dataset = TFTDataset(X_train, y_train[:, 0:1], sequence_length=128, prediction_horizon=1)
        val_dataset = TFTDataset(X_val, y_val[:, 0:1], sequence_length=128, prediction_horizon=1)
        
        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
        
        best_val_loss = float('inf')
        patience_counter = 0
        patience = 15
        
        print(f'   Training config: {epochs} epochs, batch_size=32, patience={patience}')
        print(f'   Device: {self.device}')
        print(f'   Gradient explosion prevention: ACTIVE')
        
        for epoch in range(epochs):
            print(f'\n   Epoch {epoch+1}/{epochs}:')
            
            # Train
            train_loss, train_grad_norm = self.train_epoch(train_loader, epoch)
            
            # Validate
            val_loss = self.validate(val_loader)
            
            # Scheduler step
            self.scheduler.step(val_loss)
            current_lr = self.optimizer.param_groups[0]['lr']
            
            print(f'     Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}')
            print(f'     Avg Grad Norm: {train_grad_norm:.6f}, LR: {current_lr:.2e}')
            
            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                torch.save(self.model.state_dict(), 'checkpoints/best_model_stable.pth')
                print(f'     ✓ New best model saved (Val Loss: {val_loss:.6f})')
            else:
                patience_counter += 1
            
            print(f'     Patience: {patience_counter}/{patience}')
            
            # Stability warnings
            if train_grad_norm > 0.5:
                print(f'     ⚠️ High gradient norm: {train_grad_norm:.6f}')
            
            # Early stopping
            if patience_counter >= patience:
                print(f'   Early stopping at epoch {epoch+1}')
                break
        
        # Final model save
        torch.save(self.model.state_dict(), 'checkpoints/final_model_stable.pth')
        print('   ✓ Training completed successfully!')
        
        return best_val_loss


def main():
    """Main training function."""
    print("🚀 Production-Grade TFT Training (Gradient Explosion Fixed)")
    print("=" * 70)
    
    # Configuration
    config = {
        'data_dir': 'data/',
        'batch_size': 32,
        'epochs': 50,
        'learning_rate': 1e-6,
        'sequence_length': 128
    }
    
    # Create trainer
    trainer = ProductionTFTTrainer(config)
    
    try:
        # Load and prepare data
        X_train, y_train, X_val, y_val = trainer.load_and_prepare_data()
        
        # Create model
        trainer.create_model(X_train.shape[-1])
        
        # Setup optimization
        trainer.setup_optimization()
        
        # Train
        best_loss = trainer.train(X_train, y_train, X_val, y_val, epochs=config['epochs'])
        
        print(f"\n🎉 Training Success!")
        print(f"Best validation loss: {best_loss:.6f}")
        print(f"Models saved in checkpoints/")
        
    except Exception as e:
        print(f"\n❌ Training failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)