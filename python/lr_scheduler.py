"""
Learning Rate Scheduling and Finding for Financial Time Series Models.
Implements production-grade LR scheduling optimized for financial data characteristics.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import _LRScheduler
import numpy as np
import matplotlib.pyplot as plt
from typing import Optional, Tuple, List, Dict, Union, Sequence
from collections import defaultdict
import warnings
from pathlib import Path
from matplotlib.figure import Figure  # use correct Figure type for annotations


class FinancialCosineAnnealingWarmRestarts(_LRScheduler):
    """
    Cosine Annealing with Warm Restarts optimized for financial data.
    Includes volatility-adaptive adjustments.
    """
    
    def __init__(self, optimizer, T_0: int = 10, T_mult: int = 2, 
                 eta_min: float = 1e-6, last_epoch: int = -1,
                 volatility_factor: float = 1.0):
        self.T_0 = T_0
        self.T_mult = T_mult
        self.eta_min = eta_min
        self.T_cur = 0
        self.T_i = T_0
        self.volatility_factor = volatility_factor
        self.restart_count = 0
        super().__init__(optimizer, last_epoch)
    
    def get_lr(self):
        if self.T_cur == self.T_i:
            self.restart_count += 1
            self.T_cur = 0
            self.T_i = self.T_0 * (self.T_mult ** self.restart_count)
        
        # Cosine annealing with volatility adjustment
        lrs = []
        for base_lr in self.base_lrs:
            # Add volatility-based noise for exploration
            noise = np.random.normal(0, 0.01 * self.volatility_factor) if self.volatility_factor > 0 else 0
            lr = self.eta_min + (base_lr - self.eta_min) * \
                 (1 + np.cos(np.pi * self.T_cur / self.T_i)) / 2
            lr = max(self.eta_min, lr * (1 + noise))
            lrs.append(lr)
        
        self.T_cur += 1
        return lrs


class AdaptiveOneCycleLR(_LRScheduler):
    """
    OneCycle learning rate policy with adaptive adjustments for financial data.
    Handles market regime changes effectively.
    """
    
    def __init__(self, optimizer, max_lr: Union[float, int, Sequence[float]], 
                 total_steps: Optional[int] = None, epochs: Optional[int] = None, steps_per_epoch: Optional[int] = None,
                 pct_start: float = 0.3, anneal_strategy: str = 'cos',
                 cycle_momentum: bool = True, base_momentum: float = 0.85,
                 max_momentum: float = 0.95, div_factor: float = 25.0,
                 final_div_factor: float = 1e4, last_epoch: int = -1):
        
        if total_steps is None and epochs is None:
            raise ValueError("Either total_steps or epochs must be specified")
        
        if total_steps is None:
            if epochs is None or steps_per_epoch is None:
                raise ValueError("When total_steps is None, both epochs and steps_per_epoch must be specified")
            total_steps = int(epochs) * int(steps_per_epoch)
        else:
            total_steps = int(total_steps)
        
        self.total_steps = total_steps
        self.step_ratio = None
        self.cycle_momentum = cycle_momentum
        self.base_momentum = base_momentum
        self.max_momentum = max_momentum
        self.anneal_strategy = anneal_strategy
        
        if isinstance(max_lr, (int, float)):
            self.max_lrs = [float(max_lr)] * len(optimizer.param_groups)
        else:
            seq = list(max_lr)
            if len(seq) == 1:
                self.max_lrs = [float(seq[0])] * len(optimizer.param_groups)
            elif len(seq) == len(optimizer.param_groups):
                self.max_lrs = [float(v) for v in seq]
            else:
                raise ValueError(
                    f"max_lr length ({len(seq)}) must be 1 or equal to number of param groups "
                    f"({len(optimizer.param_groups)})"
                )
        
        self.initial_lrs = [max_lr / div_factor for max_lr in self.max_lrs]
        self.min_lrs = [lr / final_div_factor for lr in self.max_lrs]
        
        phases = [
            {
                'end_step': float(pct_start * total_steps) - 1,
                'start_lr': 'initial_lr',
                'end_lr': 'max_lr',
                'start_momentum': 'max_momentum' if cycle_momentum else 'base_momentum',
                'end_momentum': 'base_momentum' if cycle_momentum else None,
            },
            {
                'end_step': float(total_steps) - 1,
                'start_lr': 'max_lr',
                'end_lr': 'min_lr',
                'start_momentum': 'base_momentum' if cycle_momentum else None,
                'end_momentum': 'max_momentum' if cycle_momentum else None,
            },
        ]
        self.phases = phases
        
        super().__init__(optimizer, last_epoch)
    
    def get_lr(self):
        step = self.last_epoch
        
        for i, phase in enumerate(self.phases):
            end_step = phase['end_step']
            if step <= end_step or i == len(self.phases) - 1:
                pct = (step - (self.phases[i-1]['end_step'] if i > 0 else -1)) / \
                      (end_step - (self.phases[i-1]['end_step'] if i > 0 else -1))
                
                if self.anneal_strategy == 'cos':
                    pct = (1 + np.cos(np.pi * pct)) / 2
                
                computed_lrs = []
                for j, base_lr in enumerate(self.base_lrs):
                    start_lr = self.initial_lrs[j] if phase['start_lr'] == 'initial_lr' else self.max_lrs[j]
                    end_lr = self.max_lrs[j] if phase['end_lr'] == 'max_lr' else self.min_lrs[j]
                    lr = start_lr + pct * (end_lr - start_lr)
                    computed_lrs.append(lr)
                
                # Update momentum if cycle_momentum is True
                if self.cycle_momentum and phase['start_momentum'] is not None:
                    for group in self.optimizer.param_groups:
                        if 'momentum' in group:
                            start_mom = self.max_momentum if phase['start_momentum'] == 'max_momentum' else self.base_momentum
                            end_mom = self.max_momentum if phase['end_momentum'] == 'max_momentum' else self.base_momentum
                            group['momentum'] = start_mom + pct * (end_mom - start_mom)
                        elif 'betas' in group:
                            start_mom = self.max_momentum if phase['start_momentum'] == 'max_momentum' else self.base_momentum
                            end_mom = self.max_momentum if phase['end_momentum'] == 'max_momentum' else self.base_momentum
                            group['betas'] = (start_mom + pct * (end_mom - start_mom), group['betas'][1])
                
                return computed_lrs
        
        return [group['lr'] for group in self.optimizer.param_groups]


class LRFinder:
    """Learning Rate Finder for financial time series models."""
    
    def __init__(self, model: nn.Module, optimizer: optim.Optimizer, 
                 loss_fn: nn.Module, device: str = 'cuda'):
        self.model = model
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.device = device
        
        # Store original state
        self.model_state = model.state_dict()
        self.optimizer_state = optimizer.state_dict()
        
        # Results storage
        self.history = {'lr': [], 'loss': [], 'smoothed_loss': []}
        self.best_lr = None
        
    def range_test(self, train_loader, start_lr: float = 1e-7, end_lr: float = 10,
                   num_iter: int = 100, smooth_f: float = 0.98,
                   divergence_th: float = 5.0) -> Dict[str, List[float]]:
        """Run the LR range test."""
        # Reset history
        self.history = {'lr': [], 'loss': [], 'smoothed_loss': []}
        
        # Set model to training mode
        self.model.train()
        
        # Calculate LR schedule
        lr_schedule = np.exp(np.linspace(np.log(start_lr), np.log(end_lr), num_iter))
        
        # Iterator for data
        iterator = iter(train_loader)
        best_loss = float('inf')
        avg_loss = 0
        
        for iteration, lr in enumerate(lr_schedule):
            # Get batch
            try:
                batch = next(iterator)
            except StopIteration:
                iterator = iter(train_loader)
                batch = next(iterator)
            
            # Prepare batch
            if isinstance(batch, (tuple, list)) and len(batch) == 2:
                inputs, targets = batch
                inputs = self._move_to_device(inputs)
                targets = targets.to(self.device)
            else:
                inputs = self._move_to_device(batch)
                targets = None
            
            # Update LR
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = lr
            
            # Forward pass
            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            
            # Handle different output formats
            if isinstance(outputs, dict):
                if 'predictions' in outputs:
                    predictions = outputs['predictions']
                    if isinstance(predictions, dict):
                        predictions = next(iter(predictions.values()))
                else:
                    predictions = outputs
            else:
                predictions = outputs
            
            # Compute loss
            if targets is not None:
                loss = self.loss_fn(predictions, targets)
            else:
                # Safely compute dummy loss from tensor-like predictions
                pred_tensor = self._to_tensor(predictions)
                if pred_tensor is None:
                    raise TypeError("LRFinder.range_test: model output is not tensor-like; cannot compute dummy loss.")
                loss = pred_tensor.float().mean()
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping for stability
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            
            # Optimizer step
            self.optimizer.step()
            
            # Record results
            self.history['lr'].append(lr)
            self.history['loss'].append(loss.item())
            
            # Calculate smoothed loss
            if iteration == 0:
                avg_loss = loss.item()
            else:
                avg_loss = smooth_f * avg_loss + (1 - smooth_f) * loss.item()
            
            smoothed_loss = avg_loss / (1 - smooth_f**(iteration + 1))
            self.history['smoothed_loss'].append(smoothed_loss)
            
            # Check for divergence
            if loss.item() > divergence_th * best_loss:
                print(f"Stopping early due to divergence at LR={lr:.2e}")
                break
            
            if loss.item() < best_loss and iteration > 10:
                best_loss = loss.item()
            
            # Progress update
            if (iteration + 1) % 20 == 0:
                print(f"Iteration {iteration+1}/{num_iter}: LR={lr:.2e}, Loss={loss.item():.4f}")
        
        # Find suggested LR
        self._find_lr()
        
        # Restore original state
        self.model.load_state_dict(self.model_state)
        self.optimizer.load_state_dict(self.optimizer_state)
        
        return self.history
    
    def _find_lr(self):
        """Find the suggested learning rate using multiple methods."""
        if len(self.history['smoothed_loss']) < 10:
            self.best_lr = self.history['lr'][len(self.history['lr'])//2]
            return
        
        losses = np.array(self.history['smoothed_loss'])
        lrs = np.array(self.history['lr'])
        
        # Method 1: Find steepest descent (most negative gradient)
        gradients = np.gradient(losses)
        min_gradient_idx = np.argmin(gradients[5:-5]) + 5
        
        # Method 2: Find minimum loss point
        min_loss_idx = np.argmin(losses)
        
        # Method 3: Find point where loss starts increasing significantly
        loss_increase_threshold = 1.1  # 10% increase
        min_loss = np.min(losses)
        divergence_idx = len(losses) - 1
        for i in range(len(losses) - 1, 0, -1):
            if losses[i] <= min_loss * loss_increase_threshold:
                divergence_idx = i
                break
        
        # Method 4: Find "knee" point using loss rate of change
        second_derivatives = np.gradient(gradients)
        max_curvature_idx = np.argmax(np.abs(second_derivatives[5:-5])) + 5
        
        # Choose the most conservative (earliest) of these methods
        candidate_indices = [
            max(0, min_gradient_idx - 3),  # Steepest descent with buffer
            max(0, min_loss_idx - 2),      # Near minimum loss
            max(0, divergence_idx - 2),    # Before divergence
            max(0, max_curvature_idx - 1)  # Near inflection point
        ]
        
        # Take the median of candidates for robustness
        best_idx = int(np.median(candidate_indices))
        best_idx = max(0, min(best_idx, len(lrs) - 1))
        
        # Apply moderate safety factor instead of aggressive 0.5
        safety_factor = 0.7  # Less aggressive than 0.5
        self.best_lr = lrs[best_idx] * safety_factor
        
        print(f"LR finding methods:")
        print(f"  Steepest descent: {lrs[min_gradient_idx]:.2e}")
        print(f"  Minimum loss: {lrs[min_loss_idx]:.2e}")
        print(f"  Before divergence: {lrs[divergence_idx]:.2e}")
        print(f"  Max curvature: {lrs[max_curvature_idx]:.2e}")
        print(f"  Selected (median): {lrs[best_idx]:.2e}")
        print(f"  Final (with safety): {self.best_lr:.2e}")
        
    def plot(self, suggest: bool = True, save_path: Optional[str] = None) -> Figure:
        """Plot the LR range test results."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        
        # Plot loss vs LR
        ax1.plot(self.history['lr'], self.history['loss'], alpha=0.3, label='Loss')
        ax1.plot(self.history['lr'], self.history['smoothed_loss'], label='Smoothed Loss')
        ax1.set_xscale('log')
        ax1.set_xlabel('Learning Rate')
        ax1.set_ylabel('Loss')
        ax1.set_title('Learning Rate Range Test')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        if suggest and self.best_lr is not None:
            ax1.axvline(float(self.best_lr), color='red', linestyle='--', 
                       label=f'Suggested LR: {self.best_lr:.2e}')
            ax1.legend()
        
        # Plot gradient
        if len(self.history['smoothed_loss']) > 2:
            gradients = np.gradient(self.history['smoothed_loss'])
            ax2.plot(self.history['lr'][1:], gradients[1:])
            ax2.set_xscale('log')
            ax2.set_xlabel('Learning Rate')
            ax2.set_ylabel('Loss Gradient')
            ax2.set_title('Loss Gradient (for finding optimal LR)')
            ax2.grid(True, alpha=0.3)
            ax2.axhline(0, color='gray', linestyle='-', alpha=0.5)
            
            if suggest and self.best_lr is not None:
                ax2.axvline(float(self.best_lr), color='red', linestyle='--')
        
        plt.tight_layout()
        
        if save_path:
            fig.savefig(save_path, dpi=100, bbox_inches='tight')
            print(f"Figure saved to {save_path}")
        
        return fig
    
    def suggestion(self) -> float:
        """Get the suggested learning rate."""
        if self.best_lr is None:
            raise RuntimeError("LRFinder.suggestion() called before range_test; no best_lr computed.")
        return float(self.best_lr)
    
    def _move_to_device(self, data):
        """Move data to device."""
        if isinstance(data, dict):
            return {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                   for k, v in data.items()}
        return data.to(self.device)

    def _to_tensor(self, x) -> Optional[torch.Tensor]:
        """
        Extract the first tensor from nested structures or coerce numeric types to a tensor.
        """
        if isinstance(x, torch.Tensor):
            return x
        if isinstance(x, dict):
            for v in x.values():
                t = self._to_tensor(v)
                if t is not None:
                    return t
            return None
        if isinstance(x, (list, tuple)):
            for v in x:
                t = self._to_tensor(v)
                if t is not None:
                    return t
            return None
        try:
            return torch.as_tensor(x, device=self.device)
        except Exception:
            return None


class FinancialLRScheduler:
    """Master scheduler that combines multiple strategies for financial data."""
    
    def __init__(self, optimizer, strategy: str = 'onecycle', **kwargs):
        self.optimizer = optimizer
        self.strategy = strategy
        self.step_count = 0
        
        if strategy == 'onecycle':
            self.scheduler = AdaptiveOneCycleLR(optimizer, **kwargs)
        elif strategy == 'cosine_restarts':
            self.scheduler = FinancialCosineAnnealingWarmRestarts(optimizer, **kwargs)
        elif strategy == 'plateau':
            self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode=kwargs.get('mode', 'min'),
                factor=kwargs.get('factor', 0.5),
                patience=kwargs.get('patience', 10),
                min_lr=kwargs.get('min_lr', 1e-8)
            )
        elif strategy == 'exponential':
            self.scheduler = optim.lr_scheduler.ExponentialLR(
                optimizer, gamma=kwargs.get('gamma', 0.95)
            )
        elif strategy == 'cyclic':
            self.scheduler = optim.lr_scheduler.CyclicLR(
                optimizer,
                base_lr=kwargs.get('base_lr', 9.38e-06),
                max_lr=kwargs.get('max_lr', 1e-2),
                step_size_up=kwargs.get('step_size_up', 2000),
                mode=kwargs.get('mode', 'triangular2')
            )
        else:
            raise ValueError(f"Unknown strategy: {strategy}")
    
    def step(self, metrics=None, epoch=None):
        """Step the scheduler."""
        if self.strategy == 'plateau':
            if metrics is not None:
                self.scheduler.step(metrics)
        else:
            self.scheduler.step(epoch)
        self.step_count += 1
    
    def get_last_lr(self):
        """Get last learning rate."""
        if hasattr(self.scheduler, 'get_last_lr'):
            return self.scheduler.get_last_lr()
        else:
            return [group['lr'] for group in self.optimizer.param_groups]
    
    def state_dict(self):
        """Get state dict."""
        return self.scheduler.state_dict()
    
    def load_state_dict(self, state_dict):
        """Load state dict."""
        self.scheduler.load_state_dict(state_dict)


def find_optimal_lr(model, train_loader, loss_fn, device='cuda', 
                    start_lr=1e-7, end_lr=10, num_iter=100):
    """
    Convenience function to find optimal learning rate.
    
    Returns:
        Tuple of (best_lr, lr_finder_object)
    """
    # Add some randomization to avoid identical results
    import random
    start_lr_adjusted = start_lr * random.uniform(0.5, 2.0)
    end_lr_adjusted = end_lr * random.uniform(0.5, 2.0)
    
    print(f"LR Range Test: {start_lr_adjusted:.2e} to {end_lr_adjusted:.2e}")
    
    # Create temporary optimizer with random initialization
    temp_optimizer = optim.Adam(model.parameters(), lr=random.uniform(1e-4, 1e-2))
    
    # Create LR finder
    lr_finder = LRFinder(model, temp_optimizer, loss_fn, device)
    
    # Run range test
    lr_finder.range_test(
        train_loader,
        start_lr=start_lr_adjusted,
        end_lr=end_lr_adjusted,
        num_iter=num_iter
    )
    
    # Get suggestion
    best_lr = lr_finder.suggestion()
    
    print(f"\nSuggested learning rate: {best_lr:.2e}")
    
    return best_lr, lr_finder
