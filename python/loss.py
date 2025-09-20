"""
Loss functions for TFT financial forecasting model.
Includes quantile loss, Huber loss, and financial-specific metrics.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional, Union


class QuantileLoss(nn.Module):
    """
    Pinball loss for quantile regression.
    Essential for TFT's probabilistic predictions.
    """
    
    def __init__(self, quantile_levels: List[float], reduction: str = 'mean'):
        super().__init__()
        self.quantile_levels = torch.tensor(quantile_levels, dtype=torch.float32)
        self.reduction = reduction
        self.num_quantiles = len(quantile_levels)
    
    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Compute quantile (pinball) loss.
        
        Args:
            predictions: (batch_size, num_quantiles) quantile predictions
            targets: (batch_size,) or (batch_size, 1) target values
            
        Returns:
            Quantile loss tensor
        """
        if targets.dim() == 1:
            targets = targets.unsqueeze(-1)
        
        # Move quantile levels to same device as predictions
        if self.quantile_levels.device != predictions.device:
            self.quantile_levels = self.quantile_levels.to(predictions.device)
        
        # Expand targets to match quantile predictions
        targets_expanded = targets.expand(-1, self.num_quantiles)
        
        # Compute prediction errors
        errors = targets_expanded - predictions  # (batch_size, num_quantiles)
        
        # Compute quantile loss for each quantile level
        quantile_losses = torch.where(
            errors >= 0,
            self.quantile_levels.unsqueeze(0) * errors,
            (self.quantile_levels.unsqueeze(0) - 1) * errors
        )
        
        if self.reduction == 'mean':
            return quantile_losses.mean()
        elif self.reduction == 'sum':
            return quantile_losses.sum()
        elif self.reduction == 'none':
            return quantile_losses
        else:
            raise ValueError(f"Invalid reduction: {self.reduction}")


class HuberQuantileLoss(nn.Module):
    """
    Huber-quantile hybrid loss for robust training.
    Combines quantile loss with Huber loss for outlier robustness.
    """
    
    def __init__(self, quantile_levels: List[float], delta: float = 1.0, alpha: float = 0.5):
        super().__init__()
        self.quantile_loss = QuantileLoss(quantile_levels)
        self.huber_loss = nn.HuberLoss(delta=delta)
        self.alpha = alpha  # Weight between quantile and Huber loss
        
        # Use median quantile for Huber loss
        self.median_idx = len(quantile_levels) // 2
    
    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Compute combined Huber-quantile loss."""
        quantile_loss = self.quantile_loss(predictions, targets)
        
        # Use median prediction for Huber loss
        median_pred = predictions[:, self.median_idx]
        targets_flat = targets.squeeze(-1) if targets.dim() > 1 else targets
        huber_loss = self.huber_loss(median_pred, targets_flat)
        
        return self.alpha * quantile_loss + (1 - self.alpha) * huber_loss


class QuantileSmoothingLoss(nn.Module):
    """
    Quantile loss with smoothing to prevent flat regions.
    Adds regularization to encourage proper quantile ordering.
    """
    
    def __init__(self, quantile_levels: List[float], smoothing_weight: float = 0.1):
        super().__init__()
        self.quantile_loss = QuantileLoss(quantile_levels)
        self.quantile_levels = torch.tensor(quantile_levels)
        self.smoothing_weight = smoothing_weight
    
    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Compute quantile loss with smoothing regularization."""
        quantile_loss = self.quantile_loss(predictions, targets)
        
        # Quantile ordering regularization
        # Penalize when higher quantiles are below lower quantiles
        if predictions.size(-1) > 1:
            ordering_loss = 0
            for i in range(predictions.size(-1) - 1):
                # Higher quantile should be >= lower quantile
                violation = F.relu(predictions[:, i] - predictions[:, i + 1])
                ordering_loss += violation.mean()
            
            smoothing_loss = self.smoothing_weight * ordering_loss
            return quantile_loss + smoothing_loss
        
        return quantile_loss


class MultiHorizonLoss(nn.Module):
    """
    Loss function for multi-horizon predictions.
    Weights different prediction horizons appropriately.
    """
    
    def __init__(self, quantile_levels: List[float], horizon_weights: Optional[List[float]] = None,
                 loss_type: str = 'quantile'):
        super().__init__()
        self.quantile_levels = quantile_levels
        self.horizon_weights = horizon_weights
        self.loss_type = loss_type
        
        if loss_type == 'quantile':
            self.base_loss = QuantileLoss(quantile_levels)
        elif loss_type == 'huber_quantile':
            self.base_loss = HuberQuantileLoss(quantile_levels)
        elif loss_type == 'smoothed_quantile':
            self.base_loss = QuantileSmoothingLoss(quantile_levels)
        else:
            raise ValueError(f"Unknown loss type: {loss_type}")
    
    def forward(self, predictions: Dict[str, torch.Tensor], 
                targets: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Compute multi-horizon loss.
        
        Args:
            predictions: Dict with keys like 'horizon_1', 'horizon_5', etc.
            targets: Dict with corresponding target values
            
        Returns:
            Weighted multi-horizon loss
        """
        total_loss = 0
        num_horizons = len(predictions)
        
        # Default equal weighting if not specified
        if self.horizon_weights is None:
            weights = [1.0] * num_horizons
        else:
            weights = self.horizon_weights
        
        horizon_keys = sorted(predictions.keys())
        
        for i, horizon_key in enumerate(horizon_keys):
            if horizon_key in targets:
                horizon_loss = self.base_loss(predictions[horizon_key], targets[horizon_key])
                weight = weights[i] if i < len(weights) else 1.0
                total_loss += weight * horizon_loss
        
        return total_loss / sum(weights[:len(horizon_keys)])


class FinancialMetricLoss(nn.Module):
    """
    Financial-specific loss incorporating Sharpe ratio and drawdown penalties.
    Useful for training models that optimize trading performance.
    """
    
    def __init__(self, quantile_levels: List[float], sharpe_weight: float = 0.1, 
                 drawdown_weight: float = 0.05, risk_free_rate: float = 0.02):
        super().__init__()
        self.quantile_loss = QuantileLoss(quantile_levels)
        self.sharpe_weight = sharpe_weight
        self.drawdown_weight = drawdown_weight
        self.risk_free_rate = risk_free_rate / 252  # Daily risk-free rate
        
        # Use median quantile for return prediction
        self.median_idx = len(quantile_levels) // 2
    
    def forward(self, predictions: torch.Tensor, targets: torch.Tensor, 
                returns_sequence: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Compute financial metric loss.
        
        Args:
            predictions: Quantile predictions
            targets: Target returns
            returns_sequence: Historical returns for Sharpe/drawdown calculation
            
        Returns:
            Combined loss with financial penalties
        """
        # Base quantile loss
        base_loss = self.quantile_loss(predictions, targets)
        
        if returns_sequence is None:
            return base_loss
        
        # Use median predictions as return forecasts
        predicted_returns = predictions[:, self.median_idx]
        
        # Sharpe ratio penalty (negative Sharpe should increase loss)
        if len(predicted_returns) > 1:
            mean_return = predicted_returns.mean()
            std_return = predicted_returns.std() + 1e-8
            sharpe_ratio = (mean_return - self.risk_free_rate) / std_return
            sharpe_penalty = -sharpe_ratio  # Negative because we want to maximize Sharpe
        else:
            sharpe_penalty = 0
        
        # Maximum drawdown penalty
        cumulative_returns = torch.cumsum(predicted_returns, dim=0)
        running_max = torch.cummax(cumulative_returns, dim=0)[0]
        drawdowns = running_max - cumulative_returns
        max_drawdown = torch.max(drawdowns)
        
        # Combine losses
        total_loss = (base_loss + 
                     self.sharpe_weight * sharpe_penalty + 
                     self.drawdown_weight * max_drawdown)
        
        return total_loss


class AdaptiveQuantileLoss(nn.Module):
    """
    Adaptive quantile loss that adjusts quantile levels during training.
    Useful for learning optimal quantile levels for the specific task.
    """
    
    def __init__(self, initial_quantiles: List[float], learnable: bool = True):
        super().__init__()
        self.num_quantiles = len(initial_quantiles)
        
        if learnable:
            # Make quantile levels learnable parameters
            self.quantile_levels = nn.Parameter(
                torch.tensor(initial_quantiles, dtype=torch.float32)
            )
        else:
            self.register_buffer('quantile_levels', 
                               torch.tensor(initial_quantiles, dtype=torch.float32))
    
    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Compute adaptive quantile loss."""
        if targets.dim() == 1:
            targets = targets.unsqueeze(-1)
        
        # Ensure quantiles are properly ordered (0 < q1 < q2 < ... < 1)
        sorted_quantiles = torch.sort(torch.clamp(self.quantile_levels, 0.01, 0.99))[0]
        
        # Expand targets
        targets_expanded = targets.expand(-1, self.num_quantiles)
        errors = targets_expanded - predictions
        
        # Compute adaptive quantile loss
        quantile_losses = torch.where(
            errors >= 0,
            sorted_quantiles.unsqueeze(0) * errors,
            (sorted_quantiles.unsqueeze(0) - 1) * errors
        )
        
        return quantile_losses.mean()


class LossScheduler:
    """
    Loss scheduler that adjusts loss weights during training.
    Useful for curriculum learning or focusing on different aspects over time.
    """
    
    def __init__(self, loss_components: Dict[str, nn.Module], 
                 initial_weights: Dict[str, float],
                 schedule_type: str = 'linear'):
        self.loss_components = loss_components
        self.initial_weights = initial_weights
        self.current_weights = initial_weights.copy()
        self.schedule_type = schedule_type
        self.step_count = 0
    
    def step(self, epoch: int = None):
        """Update loss weights based on training progress."""
        self.step_count += 1
        
        if self.schedule_type == 'linear':
            # Example: gradually increase quantile loss weight, decrease others
            for key in self.current_weights:
                if 'quantile' in key:
                    self.current_weights[key] = min(1.0, 
                        self.initial_weights[key] * (1 + 0.01 * self.step_count))
    
    def compute_loss(self, predictions, targets, **kwargs) -> torch.Tensor:
        """Compute weighted combination of losses."""
        total_loss = 0
        
        for loss_name, loss_fn in self.loss_components.items():
            if loss_name in self.current_weights:
                loss_value = loss_fn(predictions, targets, **kwargs)
                weighted_loss = self.current_weights[loss_name] * loss_value
                total_loss += weighted_loss
        
        return total_loss


def create_tft_loss(quantile_levels: List[float], loss_type: str = 'quantile', **kwargs) -> nn.Module:
    """
    Factory function to create appropriate loss for TFT training.
    
    Args:
        quantile_levels: List of quantile levels (e.g., [0.1, 0.5, 0.9])
        loss_type: Type of loss ('quantile', 'huber_quantile', 'smoothed', 'financial', 'adaptive')
        **kwargs: Additional arguments for specific loss types
        
    Returns:
        Configured loss function
    """
    if loss_type == 'quantile':
        return QuantileLoss(quantile_levels, **kwargs)
    elif loss_type == 'huber_quantile':
        return HuberQuantileLoss(quantile_levels, **kwargs)
    elif loss_type == 'smoothed':
        return QuantileSmoothingLoss(quantile_levels, **kwargs)
    elif loss_type == 'financial':
        return FinancialMetricLoss(quantile_levels, **kwargs)
    elif loss_type == 'adaptive':
        return AdaptiveQuantileLoss(quantile_levels, **kwargs)
    elif loss_type == 'multi_horizon':
        return MultiHorizonLoss(quantile_levels, **kwargs)
    else:
        raise ValueError(f"Unknown loss type: {loss_type}")


def test_losses():
    """Test loss functions with sample data."""
    print("Testing TFT Loss Functions")
    print("=" * 30)
    
    # Sample data
    batch_size, num_quantiles = 32, 3
    quantile_levels = [0.1, 0.5, 0.9]
    
    predictions = torch.randn(batch_size, num_quantiles)
    targets = torch.randn(batch_size, 1)
    
    # Test different loss types
    loss_types = ['quantile', 'huber_quantile', 'smoothed', 'adaptive']
    
    for loss_type in loss_types:
        try:
            loss_fn = create_tft_loss(quantile_levels, loss_type)
            loss_value = loss_fn(predictions, targets)
            print(f"{loss_type.capitalize()} Loss: {loss_value.item():.4f}")
        except Exception as e:
            print(f"Error with {loss_type}: {e}")
    
    # Test multi-horizon loss
    horizon_predictions = {
        'horizon_1': torch.randn(batch_size, num_quantiles),
        'horizon_5': torch.randn(batch_size, num_quantiles)
    }
    horizon_targets = {
        'horizon_1': torch.randn(batch_size, 1),
        'horizon_5': torch.randn(batch_size, 1)
    }
    
    multi_loss = MultiHorizonLoss(quantile_levels)
    multi_loss_value = multi_loss(horizon_predictions, horizon_targets)
    print(f"Multi-horizon Loss: {multi_loss_value.item():.4f}")


if __name__ == "__main__":
    test_losses()