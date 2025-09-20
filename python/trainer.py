"""
Training infrastructure for TFT financial forecasting model.
Includes trainer, optimization, scheduling, and logging capabilities.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd
from pathlib import Path
import json
import time
from typing import Dict, List, Tuple, Optional, Union, Any
from collections import defaultdict
import warnings

try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_AVAILABLE = True
except ImportError:
    TENSORBOARD_AVAILABLE = False
    print("Warning: TensorBoard not available. Logging will be limited.")

# Silent wandb import
import sys
WANDB_AVAILABLE = False
try:
    import io
    old_stderr = sys.stderr
    sys.stderr = io.StringIO()
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    pass
finally:
    if 'old_stderr' in locals():
        sys.stderr = old_stderr

try:
    from .tft_model import TemporalFusionTransformer
    from .loss import create_tft_loss, MultiHorizonLoss
    from .data import TFTDataset
    from .lr_scheduler import FinancialLRScheduler, LRFinder, find_optimal_lr
except ImportError:
    # Handle relative import issues in standalone execution
    import tft_model
    import loss
    import data
    try:
        import lr_scheduler
        FinancialLRScheduler = lr_scheduler.FinancialLRScheduler
        LRFinder = lr_scheduler.LRFinder
        find_optimal_lr = lr_scheduler.find_optimal_lr
    except ImportError:
        FinancialLRScheduler = None
        LRFinder = None
        find_optimal_lr = None
    TemporalFusionTransformer = tft_model.TemporalFusionTransformer
    create_tft_loss = loss.create_tft_loss
    MultiHorizonLoss = loss.MultiHorizonLoss
    TFTDataset = data.TFTDataset


class EarlyStopping:
    """Early stopping callback to prevent overfitting."""
    
    def __init__(self, patience: int = 10, min_delta: float = 1e-6, 
                 restore_best_weights: bool = True):
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weights
        self.best_loss = float('inf')
        self.counter = 0
        self.best_state = None
    
    def __call__(self, val_loss: float, model: nn.Module) -> bool:
        """
        Check if training should stop.
        
        Args:
            val_loss: Current validation loss
            model: Model to potentially save state
            
        Returns:
            True if training should stop
        """
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            if self.restore_best_weights:
                self.best_state = model.state_dict().copy()
        else:
            self.counter += 1
        
        if self.counter >= self.patience:
            if self.restore_best_weights and self.best_state is not None:
                model.load_state_dict(self.best_state)
                print("Restored best model weights")
            return True
        
        return False


class GradientClipping:
    """Gradient clipping utility."""
    
    def __init__(self, max_norm: float = 1.0, norm_type: float = 2.0):
        self.max_norm = max_norm
        self.norm_type = norm_type
    
    def clip_gradients(self, model: nn.Module) -> float:
        """Clip gradients and return the gradient norm."""
        return torch.nn.utils.clip_grad_norm_(
            model.parameters(), self.max_norm, self.norm_type
        ).item()


class MetricsTracker:
    """Track and compute training metrics."""
    
    def __init__(self, quantile_levels: List[float]):
        self.quantile_levels = quantile_levels
        self.reset()
    
    def reset(self):
        """Reset all metrics."""
        self.losses = []
        self.predictions = []
        self.targets = []
        self.batch_times = []
    
    def update(self, loss: float, predictions: torch.Tensor, targets: torch.Tensor, batch_time: float):
        """Update metrics with batch results."""
        self.losses.append(loss)
        self.predictions.append(predictions.detach().cpu())
        self.targets.append(targets.detach().cpu())
        self.batch_times.append(batch_time)
    
    def compute_metrics(self) -> Dict[str, float]:
        """Compute aggregated metrics."""
        if not self.losses:
            return {}
        
        metrics = {
            'loss': np.mean(self.losses),
            'loss_std': np.std(self.losses),
            'batch_time': np.mean(self.batch_times),
            'throughput': 1.0 / np.mean(self.batch_times)
        }
        
        # Combine predictions and targets
        if self.predictions:
            all_preds = torch.cat(self.predictions, dim=0)
            all_targets = torch.cat(self.targets, dim=0)
            
            # Quantile-specific metrics
            if all_preds.dim() > 1 and all_preds.size(-1) == len(self.quantile_levels):
                for i, q in enumerate(self.quantile_levels):
                    pred_q = all_preds[:, i]
                    
                    # Mean Absolute Error for this quantile
                    mae = torch.mean(torch.abs(pred_q - all_targets.squeeze())).item()
                    metrics[f'mae_q{q:.1f}'] = mae
                    
                    # Coverage (for extreme quantiles)
                    if q <= 0.2:  # Lower quantile
                        coverage = (all_targets.squeeze() <= pred_q).float().mean().item()
                        metrics[f'coverage_q{q:.1f}'] = coverage
                    elif q >= 0.8:  # Upper quantile
                        coverage = (all_targets.squeeze() >= pred_q).float().mean().item()
                        metrics[f'coverage_q{q:.1f}'] = coverage
                
                # Median prediction metrics
                median_idx = len(self.quantile_levels) // 2
                median_pred = all_preds[:, median_idx]
                metrics['mae_median'] = torch.mean(torch.abs(median_pred - all_targets.squeeze())).item()
                metrics['mse_median'] = torch.mean((median_pred - all_targets.squeeze())**2).item()
                
                # Prediction interval width
                if len(self.quantile_levels) >= 3:
                    upper_q = all_preds[:, -1]  # Highest quantile
                    lower_q = all_preds[:, 0]   # Lowest quantile
                    interval_width = (upper_q - lower_q).mean().item()
                    metrics['interval_width'] = interval_width
        
        return metrics


class TFTTrainer:
    """
    Main trainer class for TFT models.
    Handles training loop, validation, checkpointing, and logging.
    """
    
    def __init__(self, model: TemporalFusionTransformer, config: Dict[str, Any]):
        self.model = model
        self.config = config
        self.device = config.get('device', 'cuda' if torch.cuda.is_available() else 'cpu')
        
        # Move model to device
        self.model.to(self.device)
        
        # Training components
        self.optimizer = self._create_optimizer()
        self.scheduler = self._create_scheduler()
        self.loss_fn = self._create_loss_function()
        self.early_stopping = self._create_early_stopping()
        self.grad_clipper = GradientClipping(
            max_norm=config.get('grad_clip_norm', 1.0)
        )
        
        # Metrics tracking
        self.metrics_tracker = MetricsTracker(model.config['quantile_levels'])
        
        # Logging setup
        self.checkpoint_dir = Path(config.get('checkpoint_dir', 'checkpoints'))
        self.checkpoint_dir.mkdir(exist_ok=True)
        
        self.logger = self._setup_logging()
        
        # Training state
        self.epoch = 0
        self.global_step = 0
        self.best_val_loss = float('inf')
        self.training_history = {
            'train_loss': [],
            'val_loss': [],
            'learning_rate': [],
            'metrics': defaultdict(list)
        }
    
    def _create_optimizer(self) -> optim.Optimizer:
        """Create optimizer based on config."""
        optimizer_type = self.config.get('optimizer', 'adamw')
        lr = self.config.get('learning_rate', 1e-3)
        weight_decay = self.config.get('weight_decay', 1e-4)
        
        if optimizer_type.lower() == 'adamw':
            return optim.AdamW(
                self.model.parameters(),
                lr=lr,
                weight_decay=weight_decay,
                betas=self.config.get('betas', (0.9, 0.999))
            )
        elif optimizer_type.lower() == 'adam':
            return optim.Adam(
                self.model.parameters(),
                lr=lr,
                weight_decay=weight_decay
            )
        else:
            raise ValueError(f"Unknown optimizer: {optimizer_type}")
    
    def _create_scheduler(self) -> Optional[Any]:
        """Create learning rate scheduler."""
        scheduler_type = self.config.get('scheduler', 'onecycle')
        
        if scheduler_type is None or scheduler_type == 'none':
            return None
        
        # Try to use FinancialLRScheduler if available
        if 'FinancialLRScheduler' in globals() and FinancialLRScheduler is not None:
            # Calculate total steps for OneCycle
            total_steps = self.config.get('total_steps')
            if total_steps is None and scheduler_type == 'onecycle':
                epochs = self.config.get('epochs', 100)
                batch_size = self.config.get('batch_size', 64)
                estimated_batches = 1000
                total_steps = epochs * estimated_batches
            
            scheduler_configs = {
                'onecycle': {
                    'strategy': 'onecycle',
                    'max_lr': self.config.get('max_lr', 1e-2),
                    'total_steps': total_steps,
                    'pct_start': self.config.get('pct_start', 0.3)
                },
                'cosine_restarts': {
                    'strategy': 'cosine_restarts',
                    'T_0': self.config.get('T_0', 10),
                    'T_mult': self.config.get('T_mult', 2)
                },
                'plateau': {
                    'strategy': 'plateau',
                    'mode': 'min',
                    'factor': 0.5,
                    'patience': 10
                }
            }
            
            if scheduler_type in scheduler_configs:
                return FinancialLRScheduler(self.optimizer, **scheduler_configs[scheduler_type])
        
        # Fallback to PyTorch schedulers
        # Fallback to PyTorch schedulers
        if scheduler_type.lower() == 'onecycle':
            return optim.lr_scheduler.OneCycleLR(
                self.optimizer,
                max_lr=self.config.get('max_lr', 1e-2),
                total_steps=self.config.get('total_steps', 1000),
                pct_start=self.config.get('pct_start', 0.3)
            )
        elif scheduler_type.lower() == 'cosine':
            return optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=self.config.get('cosine_t_max', 100)
            )
        elif scheduler_type.lower() == 'reduce_on_plateau':
            return optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode='min',
                factor=0.5,
                patience=10,
                verbose=True
            )
        else:
            return None
    
    def find_lr(self, train_loader, num_iter: int = 100,
                start_lr: float = 1e-7, end_lr: float = 10) -> Tuple[float, Any]:
        """
        Find optimal learning rate using LR range test.
        
        Returns:
            Tuple of (suggested_lr, lr_finder_object)
        """
        if 'LRFinder' not in globals() or LRFinder is None:
            print("LRFinder not available. Using default learning rate.")
            return self.config.get('learning_rate', 1e-3), None
        
        print("Running LR Range Test...")
        lr_finder = LRFinder(self.model, self.optimizer, self.loss_fn, self.device)
        lr_finder.range_test(
            train_loader,
            start_lr=start_lr,
            end_lr=end_lr,
            num_iter=num_iter
        )
        
        suggested_lr = lr_finder.suggestion()
        print(f"\nSuggested learning rate: {suggested_lr:.2e}")
        
        # Update optimizer with new LR
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = suggested_lr
        
        # Update config
        self.config['learning_rate'] = suggested_lr
        
        return suggested_lr, lr_finder
    
    def _create_loss_function(self) -> nn.Module:
        """Create loss function based on config."""
        return create_tft_loss(
            quantile_levels=self.model.config['quantile_levels'],
            loss_type=self.config.get('loss_type', 'quantile'),
            **self.config.get('loss_kwargs', {})
        )
    
    def _create_early_stopping(self) -> Optional[EarlyStopping]:
        """Create early stopping callback."""
        if self.config.get('early_stopping', True):
            return EarlyStopping(
                patience=self.config.get('patience', 10),
                min_delta=self.config.get('min_delta', 1e-6)
            )
        return None
    
    def _setup_logging(self) -> Optional[Any]:
        """Setup logging (TensorBoard or W&B)."""
        log_type = self.config.get('logger', 'tensorboard')
        
        if log_type == 'tensorboard' and TENSORBOARD_AVAILABLE:
            log_dir = self.config.get('log_dir', 'logs')
            return SummaryWriter(log_dir=log_dir)
        elif log_type == 'wandb' and WANDB_AVAILABLE:
            wandb.init(
                project=self.config.get('wandb_project', 'tft-financial'),
                config=self.config
            )
            return wandb
        else:
            print(f"Logger {log_type} not available. Using console logging only.")
            return None
    
    def train_epoch(self, train_loader: DataLoader, use_cuda: bool = False) -> Dict[str, float]:
        """Train for one epoch."""
        self.model.train()
        self.metrics_tracker.reset()
        
        for batch_idx, batch in enumerate(train_loader):
            start_time = time.time()
            
            # Move batch to device
            if isinstance(batch, (tuple, list)) and len(batch) == 2:
                inputs, targets = batch
                inputs = self._move_to_device(inputs)
                targets = targets.to(self.device)
            else:
                inputs = self._move_to_device(batch)
                targets = None
            
            # Forward pass
            self.optimizer.zero_grad()
            
            outputs = self.model(inputs, use_cuda=use_cuda)
            predictions = outputs['predictions']
            
            # Compute loss
            if isinstance(self.loss_fn, MultiHorizonLoss) and isinstance(predictions, dict):
                # Multi-horizon targets (create target dict if needed)
                if not isinstance(targets, dict):
                    target_dict = {f'horizon_{h}': targets for h in self.model.config['prediction_horizon']}
                else:
                    target_dict = targets
                loss = self.loss_fn(predictions, target_dict)
            else:
                # Single-horizon or aggregated prediction
                if isinstance(predictions, dict):
                    # Use first horizon for single target
                    pred_tensor = next(iter(predictions.values()))
                else:
                    pred_tensor = predictions
                loss = self.loss_fn(pred_tensor, targets)
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping
            grad_norm = self.grad_clipper.clip_gradients(self.model)
            
            self.optimizer.step()
            
            # Update metrics
            batch_time = time.time() - start_time
            pred_for_metrics = pred_tensor if not isinstance(predictions, dict) else next(iter(predictions.values()))
            self.metrics_tracker.update(
                loss.item(), pred_for_metrics, targets, batch_time
            )
            
            # Update scheduler (if step-based)
            if self.scheduler is not None and hasattr(self.scheduler, 'step') and \
               not isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                self.scheduler.step()
            
            self.global_step += 1
            
            # Log batch metrics
            if batch_idx % self.config.get('log_interval', 100) == 0:
                self._log_batch_metrics(batch_idx, len(train_loader), loss.item(), grad_norm)
        
        return self.metrics_tracker.compute_metrics()
    
    def validate_epoch(self, val_loader: DataLoader, use_cuda: bool = False) -> Dict[str, float]:
        """Validate for one epoch."""
        self.model.eval()
        self.metrics_tracker.reset()
        
        with torch.no_grad():
            for batch in val_loader:
                start_time = time.time()
                
                # Move batch to device
                if isinstance(batch, (tuple, list)) and len(batch) == 2:
                    inputs, targets = batch
                    inputs = self._move_to_device(inputs)
                    targets = targets.to(self.device)
                else:
                    inputs = self._move_to_device(batch)
                    targets = None
                
                # Forward pass
                outputs = self.model(inputs, use_cuda=use_cuda)
                predictions = outputs['predictions']
                
                # Compute loss
                if isinstance(self.loss_fn, MultiHorizonLoss) and isinstance(predictions, dict):
                    if not isinstance(targets, dict):
                        target_dict = {f'horizon_{h}': targets for h in self.model.config['prediction_horizon']}
                    else:
                        target_dict = targets
                    loss = self.loss_fn(predictions, target_dict)
                else:
                    pred_tensor = next(iter(predictions.values())) if isinstance(predictions, dict) else predictions
                    loss = self.loss_fn(pred_tensor, targets)
                
                # Update metrics
                batch_time = time.time() - start_time
                pred_for_metrics = pred_tensor if not isinstance(predictions, dict) else next(iter(predictions.values()))
                self.metrics_tracker.update(
                    loss.item(), pred_for_metrics, targets, batch_time
                )
        
        return self.metrics_tracker.compute_metrics()
    
    def train(self, train_loader: DataLoader, val_loader: Optional[DataLoader] = None,
              epochs: int = 100, use_cuda: bool = False) -> Dict[str, List]:
        """
        Main training loop.
        
        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            epochs: Number of epochs to train
            use_cuda: Whether to use CUDA acceleration
            
        Returns:
            Training history dictionary
        """
        print(f"Starting training for {epochs} epochs on {self.device}")
        print(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        
        for epoch in range(epochs):
            self.epoch = epoch
            epoch_start_time = time.time()
            
            # Training
            train_metrics = self.train_epoch(train_loader, use_cuda)
            train_loss = train_metrics['loss']
            
            # Validation
            val_metrics = {}
            val_loss = train_loss  # Default to train loss if no validation
            
            if val_loader is not None:
                val_metrics = self.validate_epoch(val_loader, use_cuda)
                val_loss = val_metrics['loss']
            
            # Update scheduler (if epoch-based)
            if self.scheduler is not None and isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                self.scheduler.step(val_loss)
            
            # Log epoch metrics
            epoch_time = time.time() - epoch_start_time
            self._log_epoch_metrics(epoch, train_metrics, val_metrics, epoch_time)
            
            # Update training history
            self.training_history['train_loss'].append(train_loss)
            self.training_history['val_loss'].append(val_loss)
            self.training_history['learning_rate'].append(self.optimizer.param_groups[0]['lr'])
            
            for key, value in {**train_metrics, **val_metrics}.items():
                self.training_history['metrics'][key].append(value)
            
            # Checkpointing
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.save_checkpoint('best_model.pth', is_best=True)
            
            if epoch % self.config.get('checkpoint_interval', 10) == 0:
                self.save_checkpoint(f'checkpoint_epoch_{epoch}.pth')
            
            # Early stopping
            if self.early_stopping is not None:
                if self.early_stopping(val_loss, self.model):
                    print(f"Early stopping triggered at epoch {epoch}")
                    break
        
        # Final checkpoint
        self.save_checkpoint('final_model.pth')
        
        print("Training completed!")
        return self.training_history
    
    def _move_to_device(self, inputs: Union[Dict, torch.Tensor]) -> Union[Dict, torch.Tensor]:
        """Move inputs to the correct device."""
        if isinstance(inputs, dict):
            return {key: value.to(self.device) for key, value in inputs.items()}
        else:
            return inputs.to(self.device)
    
    def _log_batch_metrics(self, batch_idx: int, total_batches: int, loss: float, grad_norm: float):
        """Log metrics for a batch."""
        lr = self.optimizer.param_groups[0]['lr']
        print(f"Epoch {self.epoch}, Batch {batch_idx}/{total_batches}, "
              f"Loss: {loss:.4f}, LR: {lr:.6f}, Grad Norm: {grad_norm:.4f}")
        
        if self.logger and hasattr(self.logger, 'add_scalar'):
            self.logger.add_scalar('batch/loss', loss, self.global_step)
            self.logger.add_scalar('batch/learning_rate', lr, self.global_step)
            self.logger.add_scalar('batch/grad_norm', grad_norm, self.global_step)
    
    def _log_epoch_metrics(self, epoch: int, train_metrics: Dict, val_metrics: Dict, epoch_time: float):
        """Log metrics for an epoch."""
        print(f"\nEpoch {epoch} Summary:")
        print(f"  Train Loss: {train_metrics['loss']:.4f}")
        if val_metrics:
            print(f"  Val Loss: {val_metrics['loss']:.4f}")
        print(f"  Epoch Time: {epoch_time:.2f}s")
        print(f"  LR: {self.optimizer.param_groups[0]['lr']:.6f}")
        
        # Log additional metrics
        for prefix, metrics in [('train', train_metrics), ('val', val_metrics)]:
            for key, value in metrics.items():
                if key != 'loss':
                    print(f"  {prefix.capitalize()} {key}: {value:.4f}")
        
        if self.logger:
            if hasattr(self.logger, 'add_scalar'):  # TensorBoard
                self.logger.add_scalar('epoch/train_loss', train_metrics['loss'], epoch)
                if val_metrics:
                    self.logger.add_scalar('epoch/val_loss', val_metrics['loss'], epoch)
                self.logger.add_scalar('epoch/learning_rate', self.optimizer.param_groups[0]['lr'], epoch)
            elif hasattr(self.logger, 'log'):  # W&B
                log_dict = {
                    'epoch': epoch,
                    'train_loss': train_metrics['loss'],
                    'learning_rate': self.optimizer.param_groups[0]['lr']
                }
                if val_metrics:
                    log_dict['val_loss'] = val_metrics['loss']
                self.logger.log(log_dict)
    
    def save_checkpoint(self, filename: str, is_best: bool = False):
        """Save model checkpoint."""
        checkpoint = {
            'epoch': self.epoch,
            'global_step': self.global_step,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'best_val_loss': self.best_val_loss,
            'config': self.config,
            'model_config': self.model.config,
            'training_history': self.training_history
        }
        
        filepath = self.checkpoint_dir / filename
        torch.save(checkpoint, filepath)
        
        if is_best:
            print(f"New best model saved: {filepath}")
        else:
            print(f"Checkpoint saved: {filepath}")
    
    def load_checkpoint(self, filename: str) -> Dict:
        """Load model checkpoint."""
        filepath = self.checkpoint_dir / filename
        checkpoint = torch.load(filepath, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        if self.scheduler and checkpoint['scheduler_state_dict']:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        self.epoch = checkpoint['epoch']
        self.global_step = checkpoint['global_step']
        self.best_val_loss = checkpoint['best_val_loss']
        self.training_history = checkpoint.get('training_history', {})
        
        print(f"Checkpoint loaded: {filepath}")
        return checkpoint


def create_training_config(**kwargs) -> Dict[str, Any]:
    """Create default training configuration."""
    config = {
        # Optimization
        'optimizer': 'adamw',
        'learning_rate': 1e-3,
        'weight_decay': 1e-4,
        'betas': (0.9, 0.999),
        
        # Scheduling
        'scheduler': 'onecycle',
        'max_lr': 1e-2,
        'total_steps': 1000,
        'pct_start': 0.3,
        
        # Regularization
        'grad_clip_norm': 1.0,
        'dropout_rate': 0.1,
        
        # Training
        'epochs': 100,
        'batch_size': 64,
        'early_stopping': True,
        'patience': 15,
        'min_delta': 1e-6,
        
        # Loss
        'loss_type': 'quantile',
        'loss_kwargs': {},
        
        # Logging
        'logger': 'tensorboard',
        'log_dir': 'logs',
        'log_interval': 100,
        'checkpoint_dir': 'checkpoints',
        'checkpoint_interval': 10,
        
        # Device
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
        'use_cuda_kernels': False  # Set to True when CUDA backend is available
    }
    
    config.update(kwargs)
    return config


if __name__ == "__main__":
    try:
        from .tft_model import create_tft_config
    except ImportError:
        import tft_model
        create_tft_config = tft_model.create_tft_config
    
    print("TFT Trainer Test")
    print("=" * 20)
    
    # Create sample model and data
    model_config = create_tft_config(input_size=32)
    model = TemporalFusionTransformer(model_config)
    
    # Create sample dataset
    batch_size, seq_len, input_size = 16, 100, 32
    num_samples = 1000
    
    X = torch.randn(num_samples, seq_len, input_size)
    y = torch.randn(num_samples, 1)
    
    # Create data loaders
    dataset = TFTDataset(X, y)
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    # Create trainer
    training_config = create_training_config(epochs=5, log_interval=10)
    tft_trainer = TFTTrainer(model, training_config)
    
    # Train for a few epochs
    print(f"Training on {training_config['device']}")
    history = tft_trainer.train(train_loader, epochs=5)
    
    print("Training completed successfully!")
