#!/usr/bin/env python3
"""
TFT Training Script with Trading Metrics
========================================
Production-grade training with financial loss functions and stability monitoring.
"""

import warnings
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from typing import Dict, Optional, Tuple
import json
from pathlib import Path
from collections import deque

from python.tft_model import TemporalFusionTransformer

warnings.filterwarnings("ignore")


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
            # Ensure predictions and targets have compatible shapes
            if predictions.shape != targets.shape:
                # Skip this batch if shapes don't match (edge case with partial batches)
                return
                
            preds = predictions.detach().cpu().numpy()
            targs = targets.detach().cpu().numpy()
            
            # Take only first 3 outputs for position calculation
            if preds.shape[-1] >= 3:
                positions = np.tanh(preds[:, :3].mean(axis=1))  # Average across horizons
                target_returns = targs[:, :3].mean(axis=1)     # Average across horizons
            else:
                positions = np.tanh(preds.mean(axis=1))
                target_returns = targs.mean(axis=1)
            
            returns = positions * target_returns
            
            if len(self.positions) > 0 and len(positions) == len(self.positions[-1]):
                position_changes = np.abs(positions - self.positions[-1])
                costs = position_changes * transaction_cost
                returns = returns - costs
            
            self.positions.append(positions)
            self.returns.extend(returns.flatten())
            
            self.trades += len(returns)
            self.winning_trades += np.sum(returns > 0)
            self.losing_trades += np.sum(returns < 0)
            
            batch_pnl = self.capital * returns.mean()
            self.capital += batch_pnl
            self.pnl.append(batch_pnl)
    
    def get_metrics(self) -> Dict:
        """Calculate final metrics."""
        if len(self.returns) == 0:
            return {'sharpe': 0, 'win_rate': 0, 'total_return': 0, 
                   'max_drawdown': 0, 'final_capital': self.initial_capital, 'trades': 0}
        
        returns = np.array(self.returns)
        
        sharpe = 0.0
        if np.std(returns) > 1e-8:
            sharpe = np.mean(returns) / np.std(returns) * np.sqrt(39000)
        
        win_rate = self.winning_trades / max(self.trades, 1)
        total_return = (self.capital - self.initial_capital) / self.initial_capital
        
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
    """Multi-objective loss combining prediction accuracy with trading metrics."""
    
    def __init__(self, alpha: float = 0.5, beta: float = 0.3, gamma: float = 0.2):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.mse = nn.MSELoss()
        
    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> Tuple[torch.Tensor, Dict]:
        mse_loss = self.mse(predictions, targets)
        
        pred_sign = torch.sign(predictions)
        target_sign = torch.sign(targets)
        direction_accuracy = (pred_sign == target_sign).float().mean()
        direction_loss = 1.0 - direction_accuracy
        
        positions = torch.tanh(predictions * 5)
        returns = positions * targets
        profit_loss = -returns.mean()
        
        returns_std = returns.std() + 1e-8
        sharpe_loss = -returns.mean() / returns_std if returns_std > 1e-8 else torch.tensor(0.0)
        
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


# Global flag for CUDA kernels usage within this module
USE_CUDA_KERNELS = True

class TFTTrainer:
    """Enhanced TFT trainer with trading metrics and stability."""
    
    def __init__(self, model: nn.Module, config: Optional[Dict] = None):
        self.model = model
        self.gradient_history = deque(maxlen=100)
        self.trading_metrics = TradingMetrics()
        self.config = config or {
            "max_grad_norm": 1.0,  # More reasonable clipping
            "accumulation_steps": 1,
            "dropout_increase": 0.1,
            "gradient_scale": 0.1  # Less aggressive scaling
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
        self.add_training_dropout()
        
        total_loss = 0
        total_batches = 0
        accumulated_loss = 0
        
        loss_components = {'mse': 0.0, 'direction': 0.0, 'profit': 0.0, 'sharpe': 0.0, 'direction_acc': 0.0}
        self.trading_metrics.reset()
        
        optimizer.zero_grad()
        
        for batch_idx, (X_batch, y_batch) in enumerate(dataloader):
            try:
                X_batch = X_batch.to(device, non_blocking=True)
                y_batch = y_batch.to(device, non_blocking=True)
                
                # Add noise for regularization (skip first epoch)
                if epoch > 0:
                    noise = torch.randn_like(X_batch) * 0.01
                    X_batch = X_batch + noise
                
                batch_size, seq_len, num_features = X_batch.shape
                tft_inputs = {
                    'historical_features': torch.clamp(X_batch, -10, 10),
                    'future_features': X_batch[:, -1:, :],
                    'static_features': torch.zeros(batch_size, 10, device=device)
                }
                
                outputs = self.model(tft_inputs, use_cuda=USE_CUDA_KERNELS)
                
                # Robustly extract tensor predictions from model outputs
                if isinstance(outputs, dict):
                    if 'predictions' in outputs:
                        pred_container = outputs['predictions']
                        if isinstance(pred_container, dict):
                            preds = []
                            for horizon in [1, 5, 10]:
                                key = f'horizon_{horizon}'
                                if key in pred_container and isinstance(pred_container[key], torch.Tensor):
                                    preds.append(pred_container[key])
                            if preds:
                                pred = torch.cat(preds, dim=-1)
                            else:
                                # Fallback: first tensor in predictions dict
                                pred = next((v for v in pred_container.values() if isinstance(v, torch.Tensor)), None)
                        elif isinstance(pred_container, torch.Tensor):
                            pred = pred_container
                        else:
                            pred = None
                    else:
                        # Fallback: first tensor in outputs dict
                        pred = next((v for v in outputs.values() if isinstance(v, torch.Tensor)), None)
                else:
                    pred = outputs if isinstance(outputs, torch.Tensor) else None

                if pred is None or not isinstance(pred, torch.Tensor):
                    raise TypeError("Model output does not contain a tensor prediction")

                # Ensure target is a tensor (handle potential dict batches)
                if isinstance(y_batch, dict):
                    y_batch = next((v for v in y_batch.values() if isinstance(v, torch.Tensor)), None)
                    if y_batch is None:
                        raise TypeError("Target batch does not contain a tensor")

                if pred.shape != y_batch.shape:
                    if pred.dim() >= 2 and y_batch.dim() >= 2:
                        if pred.shape[1] == 1 and y_batch.shape[1] == 9:
                            pred = pred.repeat(1, 9)
                        elif pred.shape[1] == 9 and y_batch.shape[1] == 1:
                            y_batch = y_batch.repeat(1, 9)
                
                loss, components = criterion(pred, y_batch)
                loss = loss / self.config['accumulation_steps']
                
                # Scale loss further to prevent gradient accumulation explosion
                loss = loss * 0.1
                
                if torch.isnan(loss) or torch.isinf(loss):
                    print(f"⚠️  NaN/Inf loss at batch {batch_idx}, skipping")
                    optimizer.zero_grad()
                    continue
                
                loss.backward()
                accumulated_loss += loss.item() * 10  # Compensate for scaling in display
                
                for key, value in components.items():
                    loss_components[key] += value
                
                self.trading_metrics.update(pred, y_batch)
                
                if (batch_idx + 1) % self.config['accumulation_steps'] == 0:
                    # Check gradient norm BEFORE any processing
                    grad_norm_before = 0
                    for p in self.model.parameters():
                        if p.grad is not None:
                            grad_norm_before += p.grad.data.norm(2).item() ** 2
                    grad_norm_before = grad_norm_before ** 0.5
                    
                    # Scale down if too large BEFORE clipping
                    if grad_norm_before > 15:  # Higher threshold to reduce spam
                        scale_factor = 10 / grad_norm_before
                        for param in self.model.parameters():
                            if param.grad is not None:
                                param.grad.data = param.grad.data * scale_factor
                        # Only print occasionally to reduce spam
                        if not hasattr(self, '_scale_counter'):
                            self._scale_counter = 0
                        self._scale_counter += 1
                        if self._scale_counter % 20 == 0:
                            print(f"  ⚠️  Gradient scaling applied {self._scale_counter} times")
                    
                    # Now clip normally
                    grad_norm = torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), 
                        max_norm=self.config['max_grad_norm']
                    )
                    
                    self.gradient_history.append(grad_norm.item() if hasattr(grad_norm, 'item') else grad_norm)
                    
                    optimizer.step()
                    optimizer.zero_grad()
                    
                    total_loss += accumulated_loss * self.config['accumulation_steps']
                    total_batches += 1
                    accumulated_loss = 0
                    
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
        
        self.restore_dropout()
        
        avg_loss = total_loss / max(total_batches, 1)
        avg_grad = np.mean(list(self.gradient_history))
        
        for key in loss_components:
            loss_components[key] /= float(max(total_batches, 1))
        
        trading_metrics = self.trading_metrics.get_metrics()
        
        return avg_loss, avg_grad, loss_components, trading_metrics
    
    def validate(self, dataloader, criterion, device):
        """Validation with trading metrics."""
        self.model.eval()
        val_loss = 0
        val_batches = 0
        
        val_trading = TradingMetrics()
        loss_components_sum = {'mse': 0.0, 'direction': 0.0, 'profit': 0.0, 'sharpe': 0.0, 'direction_acc': 0.0}
        
        with torch.no_grad():
            for X_batch, y_batch in dataloader:
                try:
                    X_batch = X_batch.to(device, non_blocking=True)
                    y_batch = y_batch.to(device, non_blocking=True)
                    
                    batch_size, seq_len, num_features = X_batch.shape
                    tft_inputs = {
                        'historical_features': torch.clamp(X_batch, -10, 10),
                        'future_features': X_batch[:, -1:, :],
                        'static_features': torch.zeros(batch_size, 10, device=device)
                    }
                    
                    outputs = self.model(tft_inputs, use_cuda=USE_CUDA_KERNELS)
                    
                    # Robustly extract tensor predictions from model outputs (validation)
                    if isinstance(outputs, dict):
                        if 'predictions' in outputs:
                            pred_container = outputs['predictions']
                            if isinstance(pred_container, dict):
                                preds = []
                                for horizon in [1, 5, 10]:
                                    key = f'horizon_{horizon}'
                                    if key in pred_container and isinstance(pred_container[key], torch.Tensor):
                                        preds.append(pred_container[key])
                                if preds:
                                    pred = torch.cat(preds, dim=-1)
                                else:
                                    pred = next((v for v in pred_container.values() if isinstance(v, torch.Tensor)), None)
                            elif isinstance(pred_container, torch.Tensor):
                                pred = pred_container
                            else:
                                pred = None
                        else:
                            pred = next((v for v in outputs.values() if isinstance(v, torch.Tensor)), None)
                    else:
                        pred = outputs if isinstance(outputs, torch.Tensor) else None

                    if pred is None or not isinstance(pred, torch.Tensor):
                        raise TypeError("Model output does not contain a tensor prediction")

                    # Ensure target is a tensor (handle potential dict batches)
                    if isinstance(y_batch, dict):
                        y_batch = next((v for v in y_batch.values() if isinstance(v, torch.Tensor)), None)
                        if y_batch is None:
                            raise TypeError("Target batch does not contain a tensor")
                    
                    if pred.shape != y_batch.shape:
                        if pred.dim() >= 2 and y_batch.dim() >= 2:
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
            loss_components_sum[key] /= float(max(val_batches, 1))
        
        val_metrics = val_trading.get_metrics()
        
        return avg_val_loss, loss_components_sum, val_metrics


def main():
    """Main training function with trading metrics."""
    import os
    # Detect CUDA extension and device availability
    backend_ok = False
    try:
        # Prefer explicit import name used by package; optional presence
        __import__('tft_cuda_ext')
        backend_ok = True
    except Exception:
        backend_ok = False
    torch_cuda_ok = torch.cuda.is_available()
    print(
        "TFT-CUDA: CUDA backend loaded successfully" if backend_ok else
        "⚠️  CUDA backend not available, using PyTorch only"
    )
    
    print("🎯 TFT TRAINING WITH TRADING METRICS")
    print("="*60)
    
    device = torch.device('cuda' if torch_cuda_ok else 'cpu')
    print(f"Device: {device}")
    # Toggle CUDA custom kernels usage in model forward
    global USE_CUDA_KERNELS
    # Default: enable when both CUDA device and backend are available
    USE_CUDA_KERNELS = bool(torch_cuda_ok and backend_ok)
    # Optional override via env var
    env_override = os.environ.get("TFT_USE_CUDA_KERNELS")
    if env_override is not None:
        USE_CUDA_KERNELS = env_override.strip().lower() in ("1", "true", "yes", "on")
        print(f"TFT_USE_CUDA_KERNELS override -> {USE_CUDA_KERNELS}")
    
    # Load data
    X_train = np.load('data/X_train.npy')
    y_train = np.load('data/y_train.npy')
    X_val = np.load('data/X_val.npy')
    y_val = np.load('data/y_val.npy')
    
    print(f"Data: X_train{X_train.shape}, y_train{y_train.shape}")
    
    # Ensure proper shape
    if len(y_train.shape) == 1:
        y_train = y_train.reshape(-1, 1)
    if len(y_val.shape) == 1:
        y_val = y_val.reshape(-1, 1)
    
    # Check normalization
    print(f"X range: [{X_train.min():.2f}, {X_train.max():.2f}]")
    print(f"y range: [{y_train.min():.2f}, {y_train.max():.2f}]")
    
    # Normalize targets if needed
    y_mean = y_train.mean()
    y_std = y_train.std()
    if y_std > 2.0:
        print(f"Normalizing targets: mean={y_mean:.4f}, std={y_std:.4f}")
        y_train = (y_train - y_mean) / y_std
        y_val = (y_val - y_mean) / y_std
    
    # Model config
    config = {
        'input_size': X_train.shape[-1],
        'output_size': 3,
        'hidden_size': 512,
        'num_layers': 2, 
        'num_heads': 4,
        'num_encoder_layers': 2,
        'num_decoder_layers': 2,
        'dropout_rate': 0.3,
        'sequence_length': X_train.shape[1],
        'quantile_levels': [0.5],
        'prediction_horizon': [1, 5, 10],
        'num_historical_features': X_train.shape[-1],
        'num_future_features': X_train.shape[-1],
        'static_input_size': 10
    }
    
    # Initialize model
    model = TemporalFusionTransformer(config)
    
    # Enable CUDA autograd functions for optimized backward pass
    model.use_cuda_autograd = True
    print("✅ CUDA paths registered (with safe fallbacks)")
    print(f"   - Attention: {'ENABLED' if USE_CUDA_KERNELS else 'PyTorch (CUDA disabled)'}")
    print("   - LSTM: PyTorch implementation (CUDA path currently disabled)")  
    print(f"   - Quantile Heads: {'ENABLED' if USE_CUDA_KERNELS else 'PyTorch (CUDA disabled)'}")
    
    # ULTRA-conservative initialization to prevent explosions
    for name, param in model.named_parameters():
        if 'weight' in name:
            if 'lstm' in name.lower():
                nn.init.normal_(param, mean=0, std=0.001)  # Tiny LSTM weights
            elif len(param.shape) >= 2:
                nn.init.normal_(param, mean=0, std=0.01)  # Very small weights
            else:
                nn.init.zeros_(param)
        elif 'bias' in name:
            nn.init.zeros_(param)
            # LSTM forget gate bias trick
            if 'lstm' in name.lower() and param.numel() >= 4:
                n = param.size(0)
                if n % 4 == 0:
                    param.data[n//4:n//2].fill_(1.0)
    
    model.to(device)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Learning rate and scheduler from LR Finder recommendations
    lr_cfg_path = Path('config/optimal_lr.json')
    base_lr = 9.38e-06
    # Default ratios
    wd_ratio = 1e-2              # weight_decay = base_lr * 0.01
    onecycle_max_lr_ratio = 10.0 # max_lr = base_lr * 10
    cosine_eta_min_ratio = 1e-2  # eta_min = base_lr * 0.01
    onecycle_pct_start = 0.1
    max_lr = base_lr * onecycle_max_lr_ratio
    if lr_cfg_path.exists():
        try:
            with open(lr_cfg_path, 'r') as f:
                lr_cfg = json.load(f)
            # Use suggested learning rate as base_lr
            base_lr = float(lr_cfg.get('optimal_lr', base_lr))
            # Optional ratios block for consistent scaling
            ratios = lr_cfg.get('ratios', {})
            wd_ratio = float(ratios.get('weight_decay_ratio', wd_ratio))
            onecycle_max_lr_ratio = float(ratios.get('onecycle_max_lr_ratio', onecycle_max_lr_ratio))
            cosine_eta_min_ratio = float(ratios.get('cosine_eta_min_ratio', cosine_eta_min_ratio))
            onecycle = lr_cfg.get('scheduler_recommendations', {}).get('onecycle', {})
            # Prefer provided max_lr, fallback to ratio of base
            max_lr = float(onecycle.get('max_lr', base_lr * onecycle_max_lr_ratio))
            onecycle_pct_start = float(onecycle.get('pct_start', onecycle_pct_start))
            print(f"Using LR Finder config: base_lr={base_lr:.3e}, max_lr={max_lr:.3e}, wd={base_lr*wd_ratio:.3e} (ratios)")
        except Exception as e:
            print(f"⚠️  Failed to read LR config, using defaults: {e}")
    else:
        print(f"⚠️  LR config not found at {lr_cfg_path}, using defaults base_lr={base_lr:.3e}, max_lr={max_lr:.3e}, wd={base_lr*wd_ratio:.3e}")

    # Optimizer (weight decay scaled to LR)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=base_lr,
        betas=(0.9, 0.999),
        eps=1e-6,
        weight_decay=base_lr * wd_ratio
    )
    
    # Loss
    criterion = TradingAwareLoss(alpha=0.5, beta=0.3, gamma=0.2)
    
    # Data loaders
    train_dataset = TensorDataset(torch.FloatTensor(X_train), torch.FloatTensor(y_train))
    val_dataset = TensorDataset(torch.FloatTensor(X_val), torch.FloatTensor(y_val))
    
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False, pin_memory=True)

    # Create trainer
    trainer = TFTTrainer(model)

    # Scheduler: OneCycle using LR Finder recommendation
    try:
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=max_lr,
            epochs=100,
            steps_per_epoch=len(train_loader),
            pct_start=onecycle_pct_start,
            anneal_strategy='cos',
        )
    except Exception as e:
        print(f"⚠️  OneCycleLR setup failed ({e}), falling back to CosineAnnealingLR")
        # Use configured eta_min if present in config; otherwise apply ratio to base_lr
        eta_min = base_lr * cosine_eta_min_ratio
        try:
            with open(lr_cfg_path, 'r') as f:
                lr_cfg = json.load(f)
            cosine_rec = lr_cfg.get('scheduler_recommendations', {}).get('cosine_restarts', {})
            eta_min = float(cosine_rec.get('eta_min', eta_min))
        except Exception:
            pass
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=100, eta_min=eta_min
        )
    
    # Trainer
    # trainer already created above
    
    # Training loop
    best_val_loss = float('inf')
    best_val_sharpe = -float('inf')
    patience = 0
    max_patience = 7
    
    print("\n" + "="*80)
    print("Starting Training")
    print("="*80)
    
    for epoch in range(100):
        print(f"\nEpoch {epoch+1}/100")
        print("-"*40)
        
        train_loss, train_grad, train_components, train_metrics = trainer.train_epoch(
            train_loader, optimizer, criterion, device, epoch
        )
        
        val_loss, val_components, val_metrics = trainer.validate(
            val_loader, criterion, device
        )
        
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
        
        # Save best model (prioritize Sharpe ratio)
        if val_metrics['sharpe'] > best_val_sharpe and val_metrics['sharpe'] > 0:
            best_val_sharpe = val_metrics['sharpe']
            best_val_loss = val_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'config': config,
                'metrics': {'train': train_metrics, 'val': val_metrics, 'loss': val_loss}
            }, 'checkpoints/tft_best.pth')
            print("💾 Saved new best model!")
            patience = 0
        elif val_loss < best_val_loss:
            # Also save if loss improves (even if Sharpe doesn't)
            best_val_loss = val_loss
            patience = 0
        else:
            patience += 1
            print(f"⏳ No improvement for {patience}/{max_patience} epochs")
        
        if patience >= max_patience:
            print("\n🛑 Early stopping triggered")
            break
    
    print("\n✅ Training complete!")
    print(f"Best Val Loss: {best_val_loss:.6f}, Best Sharpe: {best_val_sharpe:.2f}")


if __name__ == "__main__":
    main()
