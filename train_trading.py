#!/usr/bin/env python3
"""
Production-Grade TFT Training with Trading Metrics
===================================================
Enhanced training pipeline with financial loss functions and overfitting prevention.
"""

import sys
import warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, "python")

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from typing import Dict, Optional, Tuple
import json
from pathlib import Path
from collections import deque

from tft_model import TemporalFusionTransformer


class TradingMetrics:
    """Calculate trading performance metrics during training."""
    
    def __init__(self, initial_capital: float = 100000):
        self.initial_capital = initial_capital
        self.reset()
    
    def reset(self):
        """Reset metrics for new epoch."""
        self.positions = []
        self.returns = []
        self.pnl = []
        self.capital = self.initial_capital
        self.trades = 0
        self.winning_trades = 0
        self.losing_trades = 0
        
    def update(self, predictions: torch.Tensor, targets: torch.Tensor, 
               transaction_cost: float = 0.001):
        """Update metrics with batch results."""
        with torch.no_grad():
            # Convert to numpy
            preds = predictions.detach().cpu().numpy()
            targs = targets.detach().cpu().numpy()
            
            # Calculate positions based on predictions (sigmoid for position sizing)
            positions = np.tanh(preds * 5)  # Scale predictions to [-1, 1] range
            
            # Calculate returns
            returns = positions * targs
            
            # Apply transaction costs for position changes
            if len(self.positions) > 0:
                position_changes = np.abs(positions - self.positions[-1])
                costs = position_changes * transaction_cost
                returns = returns - costs
            
            # Update tracking
            self.positions.append(positions)
            self.returns.extend(returns.flatten())
            
            # Track winning/losing trades
            self.trades += len(returns)
            self.winning_trades += np.sum(returns > 0)
            self.losing_trades += np.sum(returns < 0)
            
            # Update capital
            batch_pnl = self.capital * returns.mean()
            self.capital += batch_pnl
            self.pnl.append(batch_pnl)
    
    def get_metrics(self) -> Dict:
        """Calculate final metrics."""
        if len(self.returns) == 0:
            return {}
        
        returns = np.array(self.returns)
        
        # Calculate Sharpe ratio (annualized for 10-min bars)
        sharpe = 0.0
        if np.std(returns) > 1e-8:
            sharpe = np.mean(returns) / np.std(returns) * np.sqrt(39000)
        
        # Win rate
        win_rate = self.winning_trades / max(self.trades, 1)
        
        # Total return
        total_return = (self.capital - self.initial_capital) / self.initial_capital
        
        # Max drawdown
        cumulative = np.cumsum(returns)
        running_max = np.maximum.accumulate(cumulative)
        drawdown = (cumulative - running_max) / (running_max + 1e-8)
        max_drawdown = np.min(drawdown) if len(drawdown) > 0 else 0
        
        return {
            'sharpe': sharpe,
            'win_rate': win_rate,
            'total_return': total_return,
            'max_drawdown': max_drawdown,
            'final_capital': self.capital,
            'trades': self.trades
        }


class TradingAwareLoss(nn.Module):
    """
    Multi-objective loss function combining prediction accuracy with trading metrics.
    """
    
    def __init__(self, alpha: float = 0.5, beta: float = 0.3, gamma: float = 0.2):
        """
        Args:
            alpha: Weight for MSE loss
            beta: Weight for directional loss
            gamma: Weight for profit loss
        """
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.mse = nn.MSELoss()
        
    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> Tuple[torch.Tensor, Dict]:
        """
        Calculate combined loss with component tracking.
        
        Returns:
            total_loss: Combined loss for backprop
            components: Dict of individual loss components
        """
        # MSE Loss
        mse_loss = self.mse(predictions, targets)
        
        # Directional Loss (penalize wrong direction predictions)
        pred_sign = torch.sign(predictions)
        target_sign = torch.sign(targets)
        direction_accuracy = (pred_sign == target_sign).float().mean()
        direction_loss = 1.0 - direction_accuracy
        
        # Profit Loss (simulate trading PnL)
        positions = torch.tanh(predictions * 5)  # Convert to position sizes [-1, 1]
        returns = positions * targets  # Simple PnL
        profit_loss = -returns.mean()  # Negative because we want to maximize profit
        
        # Sharpe Loss (maximize risk-adjusted returns)
        returns_std = returns.std() + 1e-8
        sharpe_loss = -returns.mean() / returns_std if returns_std > 1e-8 else torch.tensor(0.0)
        
        # Combined loss with stabilization
        total_loss = (
            self.alpha * mse_loss + 
            self.beta * direction_loss + 
            self.gamma * (profit_loss + 0.1 * sharpe_loss)
        )
        
        components = {
            'mse': mse_loss.item(),
            'direction': direction_loss.item(),
            'profit': profit_loss.item(),
            'sharpe': sharpe_loss.item(),
            'direction_acc': direction_accuracy.item()
        }
        
        return total_loss, components


class StableTrainer:
    """Enhanced trainer with overfitting prevention and trading metrics."""
    
    def __init__(self, model: nn.Module, config: Optional[Dict] = None):
        self.model = model
        self.config = config or self._default_config()
        self.gradient_history = deque(maxlen=100)
        self.loss_history = deque(maxlen=100)
        self.trading_metrics = TradingMetrics()
        
    def _default_config(self) -> Dict:
        return {
            'max_grad_norm': 0.5,
            'accumulation_steps': 2,
            'warmup_epochs': 3,
            'label_smoothing': 0.1,
            'dropout_increase': 0.1  # Add extra dropout during training
        }
    
    def add_training_dropout(self):
        """Temporarily increase dropout for training."""
        for module in self.model.modules():
            if isinstance(module, nn.Dropout):
                module.p = min(module.p + self.config['dropout_increase'], 0.5)
    
    def restore_dropout(self):
        """Restore original dropout rates."""
        for module in self.model.modules():
            if isinstance(module, nn.Dropout):
                module.p = max(module.p - self.config['dropout_increase'], 0.1)
    
    def train_epoch(self, dataloader, optimizer, criterion, device, epoch: int = 0):
        """Train with trading metrics and stability monitoring."""
        
        self.model.train()
        self.add_training_dropout()  # Increase dropout for training
        
        total_loss = 0
        total_batches = 0
        accumulated_loss = 0
        
        # Component tracking
        loss_components = {'mse': 0, 'direction': 0, 'profit': 0, 'sharpe': 0, 'direction_acc': 0}
        
        # Reset trading metrics
        self.trading_metrics.reset()
        
        optimizer.zero_grad()
        
        for batch_idx, (X_batch, y_batch) in enumerate(dataloader):
            try:
                X_batch = X_batch.to(device, non_blocking=True)
                y_batch = y_batch.to(device, non_blocking=True)
                
                # Add noise for regularization (only during training)
                if epoch > 0:  # Skip first epoch to establish baseline
                    noise = torch.randn_like(X_batch) * 0.01
                    X_batch = X_batch + noise
                
                # Convert to TFT format
                batch_size, seq_len, num_features = X_batch.shape
                tft_inputs = {
                    'historical_features': torch.clamp(X_batch, -10, 10),
                    'future_features': X_batch[:, -1:, :],
                    'static_features': torch.zeros(batch_size, 10, device=device)
                }
                
                # Forward pass
                outputs = self.model(tft_inputs)
                
                # Extract predictions
                if isinstance(outputs, dict) and 'predictions' in outputs:
                    preds = []
                    for horizon in [1, 5, 10]:
                        if f'horizon_{horizon}' in outputs['predictions']:
                            preds.append(outputs['predictions'][f'horizon_{horizon}'])
                    pred = torch.cat(preds, dim=-1) if preds else outputs['predictions'][list(outputs['predictions'].keys())[0]]
                else:
                    pred = outputs
                
                # Ensure shape compatibility
                if pred.shape != y_batch.shape:
                    if pred.shape[1] == 1 and y_batch.shape[1] == 9:
                        pred = pred.repeat(1, 9)
                    elif pred.shape[1] == 9 and y_batch.shape[1] == 1:
                        y_batch = y_batch.repeat(1, 9)
                
                # Calculate loss with components
                loss, components = criterion(pred, y_batch)
                loss = loss / self.config['accumulation_steps']
                
                # Check for NaN
                if torch.isnan(loss) or torch.isinf(loss):
                    print(f"⚠️  NaN/Inf loss at batch {batch_idx}, skipping")
                    optimizer.zero_grad()
                    continue
                
                loss.backward()
                accumulated_loss += loss.item()
                
                # Update component tracking
                for key, value in components.items():
                    loss_components[key] += value
                
                # Update trading metrics
                self.trading_metrics.update(pred, y_batch)
                
                # Gradient accumulation
                if (batch_idx + 1) % self.config['accumulation_steps'] == 0:
                    # Gradient clipping
                    grad_norm = torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), 
                        max_norm=self.config['max_grad_norm']
                    )
                    
                    self.gradient_history.append(grad_norm.item() if hasattr(grad_norm, 'item') else grad_norm)
                    
                    # Skip update if gradient is too large (emergency brake)
                    if grad_norm > 10:
                        print(f"  🛑 Gradient explosion detected: {grad_norm:.2f}, skipping update")
                        optimizer.zero_grad()
                        accumulated_loss = 0
                        continue
                    
                    optimizer.step()
                    optimizer.zero_grad()
                    
                    total_loss += accumulated_loss * self.config['accumulation_steps']
                    total_batches += 1
                    accumulated_loss = 0
                    
                    # Progress update with trading metrics
                    if batch_idx % 80 == 79:
                        current_metrics = self.trading_metrics.get_metrics()
                        avg_grad = np.mean(list(self.gradient_history))
                        
                        print(f"  Batch {batch_idx}: Loss={total_loss/total_batches:.6f}, "
                              f"Grad={avg_grad:.3f}, "
                              f"Dir={components['direction_acc']:.1%}, "
                              f"PnL=${current_metrics.get('final_capital', 100000)-100000:+.0f}")
                
            except Exception as e:
                print(f"❌ Batch {batch_idx} error: {e}")
                optimizer.zero_grad()
                continue
        
        self.restore_dropout()  # Restore original dropout
        
        # Calculate averages
        avg_loss = total_loss / max(total_batches, 1)
        avg_grad = np.mean(list(self.gradient_history))
        
        for key in loss_components:
            loss_components[key] /= max(total_batches, 1)
        
        # Get final trading metrics
        trading_metrics = self.trading_metrics.get_metrics()
        
        return avg_loss, avg_grad, loss_components, trading_metrics
    
    def validate(self, dataloader, criterion, device):
        """Validation with trading metrics."""
        self.model.eval()
        val_loss = 0
        val_batches = 0
        
        val_trading = TradingMetrics()
        loss_components_sum = {'mse': 0, 'direction': 0, 'profit': 0, 'sharpe': 0, 'direction_acc': 0}
        
        with torch.no_grad():
            for X_batch, y_batch in dataloader:
                try:
                    X_batch = X_batch.to(device, non_blocking=True)
                    y_batch = y_batch.to(device, non_blocking=True)
                    
                    # TFT inputs (no noise during validation)
                    batch_size, seq_len, num_features = X_batch.shape
                    tft_inputs = {
                        'historical_features': torch.clamp(X_batch, -10, 10),
                        'future_features': X_batch[:, -1:, :],
                        'static_features': torch.zeros(batch_size, 10, device=device)
                    }
                    
                    outputs = self.model(tft_inputs)
                    
                    # Extract predictions
                    if isinstance(outputs, dict) and 'predictions' in outputs:
                        preds = []
                        for horizon in [1, 5, 10]:
                            if f'horizon_{horizon}' in outputs['predictions']:
                                preds.append(outputs['predictions'][f'horizon_{horizon}'])
                        pred = torch.cat(preds, dim=-1) if preds else outputs['predictions'][list(outputs['predictions'].keys())[0]]
                    else:
                        pred = outputs
                    
                    # Shape matching
                    if pred.shape != y_batch.shape:
                        if pred.shape[1] == 1 and y_batch.shape[1] == 9:
                            pred = pred.repeat(1, 9)
                        elif pred.shape[1] == 9 and y_batch.shape[1] == 1:
                            y_batch = y_batch.repeat(1, 9)
                    
                    loss, components = criterion(pred, y_batch)
                    val_loss += loss.item()
                    
                    for key, value in components.items():
                        loss_components_sum[key] += value
                    
                    val_trading.update(pred, y_batch)
                    val_batches += 1
                    
                except Exception as e:
                    print(f"  Validation error: {e}")
                    continue
        
        avg_val_loss = val_loss / max(val_batches, 1)
        
        for key in loss_components_sum:
            loss_components_sum[key] /= max(val_batches, 1)
        
        val_metrics = val_trading.get_metrics()
        
        return avg_val_loss, loss_components_sum, val_metrics


def main():
    """Enhanced training with trading metrics and stability."""
    
    print("🎯 TFT TRAINING WITH TRADING METRICS")
    print("="*60)
    
    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    # Load data
    X_train = np.load('data/X_train.npy')
    y_train = np.load('data/y_train.npy')
    X_val = np.load('data/X_val.npy')
    y_val = np.load('data/y_val.npy')
    
    print(f"Data: X_train{X_train.shape}, y_train{y_train.shape}")
    
    # Ensure proper target shape
    if len(y_train.shape) == 1:
        y_train = y_train.reshape(-1, 1)
    if len(y_val.shape) == 1:
        y_val = y_val.reshape(-1, 1)
    
    # Data normalization check
    print(f"X range: [{X_train.min():.2f}, {X_train.max():.2f}]")
    print(f"y range: [{y_train.min():.2f}, {y_train.max():.2f}]")
    
    # Normalize targets if needed
    y_mean = y_train.mean()
    y_std = y_train.std()
    if y_std > 2.0:  # Targets not normalized
        print(f"Normalizing targets: mean={y_mean:.4f}, std={y_std:.4f}")
        y_train = (y_train - y_mean) / y_std
        y_val = (y_val - y_mean) / y_std
    
    # Model configuration
    config = {
        'input_size': X_train.shape[-1],
        'output_size': 3,
        'hidden_size': 256,
        'num_heads': 4,
        'num_encoder_layers': 2,
        'num_decoder_layers': 2,
        'dropout_rate': 0.3,  # Base dropout
        'sequence_length': X_train.shape[1],
        'quantile_levels': [0.5],
        'prediction_horizon': [1, 5, 10],
        'num_historical_features': X_train.shape[-1],
        'num_future_features': X_train.shape[-1],
        'static_input_size': 10
    }
    
    # Initialize model
    model = TemporalFusionTransformer(config)
    model.to(device)
    
    # Initialize with smaller weights
    for name, param in model.named_parameters():
        if 'weight' in name:
            if len(param.shape) >= 2:
                nn.init.xavier_uniform_(param, gain=0.5)  # Smaller initialization
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Optimizer with lower learning rate to prevent overfitting
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=1e-5,  # Lower than before
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=1e-3  # Higher weight decay
    )
    
    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100, eta_min=1e-7)
    
    # Loss function
    criterion = TradingAwareLoss(alpha=0.5, beta=0.3, gamma=0.2)
    
    # Data loaders
    train_dataset = TensorDataset(torch.FloatTensor(X_train), torch.FloatTensor(y_train))
    val_dataset = TensorDataset(torch.FloatTensor(X_val), torch.FloatTensor(y_val))
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, pin_memory=True)  # Smaller batch
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False, pin_memory=True)
    
    # Trainer
    trainer = StableTrainer(model)
    
    # Training loop
    best_val_loss = float('inf')
    patience = 0
    max_patience = 7
    
    print("\n" + "="*80)
    print("Starting Training with Trading Metrics")
    print("="*80)
    
    for epoch in range(100):
        print(f"\nEpoch {epoch+1}/100")
        print("-"*40)
        
        # Train
        train_loss, train_grad, train_components, train_metrics = trainer.train_epoch(
            train_loader, optimizer, criterion, device, epoch
        )
        
        # Validate
        val_loss, val_components, val_metrics = trainer.validate(
            val_loader, criterion, device
        )
        
        # Update learning rate
        scheduler.step()
        
        # Display metrics
        print(f"📊 Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}")
        print(f"📈 Gradient Norm: {train_grad:.3f}")
        print(f"🎯 Direction Acc - Train: {train_components['direction_acc']:.1%}, "
              f"Val: {val_components['direction_acc']:.1%}")
        print(f"💰 PnL - Train: ${train_metrics['final_capital']-100000:+,.0f}, "
              f"Val: ${val_metrics['final_capital']-100000:+,.0f}")
        print(f"📊 Sharpe - Train: {train_metrics['sharpe']:.2f}, "
              f"Val: {val_metrics['sharpe']:.2f}")
        print(f"📉 Max DD - Train: {train_metrics['max_drawdown']:.2%}, "
              f"Val: {val_metrics['max_drawdown']:.2%}")
        
        # Save best model based on validation Sharpe ratio
        if val_metrics['sharpe'] > 0 and val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'config': config,
                'metrics': {
                    'train': train_metrics,
                    'val': val_metrics,
                    'loss': val_loss
                }
            }, 'checkpoints/tft_trading_best.pth')
            print("💾 Saved new best model!")
            patience = 0
        else:
            patience += 1
            print(f"⏳ No improvement for {patience}/{max_patience} epochs")
        
        # Early stopping
        if patience >= max_patience:
            print("\n🛑 Early stopping triggered")
            break
    
    print("\n✅ Training complete!")


if __name__ == "__main__":
    main()
