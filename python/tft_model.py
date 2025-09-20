"""
Temporal Fusion Transformer (TFT) model implementation for financial forecasting.
Integrates with CUDA kernels for high-performance training and inference.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional, Union
import json
from pathlib import Path

# Silent CUDA backend import
import sys
import io
CUDA_AVAILABLE = False
try:
    old_stdout = sys.stdout
    sys.stdout = io.StringIO()
    import tft_cuda
    CUDA_AVAILABLE = True
except ImportError:
    pass
finally:
    sys.stdout = old_stdout
    
# Print message only if not testing and CUDA is available
if CUDA_AVAILABLE and not any('test' in arg.lower() for arg in sys.argv):
    print("TFT-CUDA: CUDA backend loaded successfully")


class GLU(nn.Module):
    """Gated Linear Unit activation."""
    
    def __init__(self, input_size: int):
        super().__init__()
        self.linear = nn.Linear(input_size, input_size * 2)
    
    def forward(self, x):
        x = self.linear(x)
        return F.glu(x, dim=-1)


class GatedResidualNetwork(nn.Module):
    """Gated Residual Network (GRN) component."""
    
    def __init__(self, input_size: int, hidden_size: int, output_size: int, 
                 dropout_rate: float = 0.1, use_time_distributed: bool = True):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.use_time_distributed = use_time_distributed
        
        self.skip_connection = nn.Linear(input_size, output_size) if input_size != output_size else None
        
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.gate = nn.Linear(hidden_size, output_size)
        self.add_and_norm = nn.LayerNorm(output_size)
        self.dropout = nn.Dropout(dropout_rate)
    
    def forward(self, x, context=None):
        # Main processing
        if context is not None:
            x = torch.cat([x, context], dim=-1)
        
        # Two-layer ELU network
        eta_1 = F.elu(self.fc1(x))
        eta_2 = self.fc2(eta_1)
        
        # Gating mechanism  
        gate = torch.sigmoid(self.gate(eta_2))
        a = gate * eta_2
        
        # Skip connection
        if self.skip_connection is not None:
            x = self.skip_connection(x)
        else:
            x = x[..., :self.output_size]  # Trim if needed
        
        # Add & norm
        return self.add_and_norm(x + self.dropout(a))


class VariableSelectionNetwork(nn.Module):
    """Variable Selection Network (VSN) for feature selection."""
    
    def __init__(self, input_size: int, num_inputs: int, hidden_size: int, dropout_rate: float = 0.1):
        super().__init__()
        self.input_size = input_size
        self.num_inputs = num_inputs
        self.hidden_size = hidden_size
        
        # Simplified approach - use adaptive layers
        self.selection_layer = nn.Sequential(
            nn.Linear(input_size * num_inputs, hidden_size),
            nn.ELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_size, num_inputs),
            nn.Softmax(dim=-1)
        )
        
        # Single transformation for all variables
        self.variable_transform = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ELU(),
            nn.Dropout(dropout_rate)
        )
    
    def forward(self, flattened_inputs, processed_inputs):
        batch_size, seq_len, total_features = processed_inputs.shape
        
        # Calculate actual dimensions
        if total_features % self.input_size == 0:
            actual_num_inputs = total_features // self.input_size
        else:
            # If dimensions don't divide evenly, use the total as single input
            actual_num_inputs = 1
            self.input_size = total_features
        
        # Get selection weights - handle dynamic input size
        if flattened_inputs.size(-1) != actual_num_inputs * self.input_size:
            # Create adaptive layer
            adaptive_layer = nn.Linear(flattened_inputs.size(-1), actual_num_inputs, device=flattened_inputs.device)
            sparse_weights = F.softmax(adaptive_layer(flattened_inputs), dim=-1)
        else:
            sparse_weights = self.selection_layer(flattened_inputs)
        
        # Process variables
        if actual_num_inputs == 1:
            # Single variable case
            transformed = self.variable_transform(processed_inputs)
            outputs = transformed * sparse_weights.unsqueeze(1).unsqueeze(-1)
        else:
            # Multiple variables
            processed_inputs_reshaped = processed_inputs.view(batch_size, seq_len, actual_num_inputs, self.input_size)
            
            # Transform each variable
            transformed_features = []
            for i in range(actual_num_inputs):
                var_input = processed_inputs_reshaped[:, :, i, :]
                transformed = self.variable_transform(var_input)
                transformed_features.append(transformed)
            
            transformed_features = torch.stack(transformed_features, dim=-2)  # (batch, seq, num_vars, hidden)
            
            # Apply selection weights
            weights_expanded = sparse_weights.unsqueeze(1).unsqueeze(-1)  # (batch, 1, num_vars, 1)
            outputs = torch.sum(weights_expanded * transformed_features, dim=-2)  # (batch, seq, hidden)
        
        return outputs, sparse_weights


class MultiHeadAttention(nn.Module):
    """Multi-Head Attention with optional CUDA acceleration."""
    
    def __init__(self, d_model: int, num_heads: int, dropout_rate: float = 0.1):
        super().__init__()
        assert d_model % num_heads == 0
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        self.W_q = nn.Linear(d_model, d_model, bias=False)
        self.W_k = nn.Linear(d_model, d_model, bias=False) 
        self.W_v = nn.Linear(d_model, d_model, bias=False)
        self.W_o = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout_rate)
        self.layer_norm = nn.LayerNorm(d_model)
        
    def forward(self, query, key, value, mask=None, use_cuda=False):
        batch_size, seq_len = query.size(0), query.size(1)
        
        # Linear projections
        Q = self.W_q(query).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        K = self.W_k(key).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        V = self.W_v(value).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        
        # Use CUDA kernel if available
        if use_cuda and CUDA_AVAILABLE:
            try:
                # Convert to expected format: (B, T, H, D)
                Q_cuda = Q.transpose(1, 2).contiguous().half()
                K_cuda = K.transpose(1, 2).contiguous().half()
                V_cuda = V.transpose(1, 2).contiguous().half()
                
                output = torch.zeros_like(Q_cuda)
                attn_weights = torch.zeros(batch_size, seq_len, self.num_heads, seq_len, 
                                         dtype=torch.float32, device=query.device)
                
                tft_cuda.multi_head_attention_mp(
                    Q_cuda, K_cuda, V_cuda, output, attn_weights, 
                    10000.0, batch_size, seq_len, self.num_heads, self.d_k
                )
                
                # Convert back to (B, H, T, D) then (B, T, H, D)
                attention_output = output.transpose(1, 2).float()
                
            except Exception as e:
                print(f"CUDA attention failed, falling back to PyTorch: {e}")
                attention_output, attn_weights = self._pytorch_attention(Q, K, V, mask)
        else:
            attention_output, attn_weights = self._pytorch_attention(Q, K, V, mask)
        
        # Concatenate heads and project
        attention_output = attention_output.transpose(1, 2).contiguous().view(
            batch_size, seq_len, self.d_model
        )
        
        output = self.W_o(attention_output)
        output = self.dropout(output)
        
        # Residual connection and layer norm
        return self.layer_norm(query + output), attn_weights
    
    def _pytorch_attention(self, Q, K, V, mask=None):
        """PyTorch fallback implementation."""
        scores = torch.matmul(Q, K.transpose(-2, -1)) / np.sqrt(self.d_k)
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        attention_output = torch.matmul(attn_weights, V)
        return attention_output, attn_weights


class TemporalFusionTransformer(nn.Module):
    """
    Complete Temporal Fusion Transformer implementation for financial forecasting.
    """
    
    def __init__(self, config: Dict):
        super().__init__()
        self.config = config
        
        # Model dimensions
        self.input_size = config['input_size']
        self.hidden_size = config.get('hidden_size', 256)
        self.num_heads = config.get('num_heads', 8)
        self.num_quantiles = len(config.get('quantile_levels', [0.1, 0.5, 0.9]))
        self.prediction_horizon = config.get('prediction_horizon', [1])
        self.sequence_length = config.get('sequence_length', 100)
        self.dropout_rate = config.get('dropout_rate', 0.1)
        
        # Static feature processing
        self.static_covariate_encoder = GatedResidualNetwork(
            config.get('static_input_size', 10), 
            self.hidden_size, 
            self.hidden_size,
            self.dropout_rate
        )
        
        # Variable selection networks
        self.historical_vsn = VariableSelectionNetwork(
            1,  # Use 1 for input_size since we'll handle reshaping internally
            config.get('num_historical_features', self.input_size),
            self.hidden_size,
            self.dropout_rate
        )
        
        self.future_vsn = VariableSelectionNetwork(
            1,  # Use 1 for input_size since we'll handle reshaping internally
            config.get('num_future_features', 10), 
            self.hidden_size,
            self.dropout_rate
        )
        
        # Sequence processing
        self.lstm = nn.LSTM(
            self.hidden_size, 
            self.hidden_size, 
            batch_first=True,
            dropout=self.dropout_rate if config.get('num_lstm_layers', 1) > 1 else 0,
            num_layers=config.get('num_lstm_layers', 1)
        )
        
        # Temporal attention
        self.multihead_attn = MultiHeadAttention(
            self.hidden_size, 
            self.num_heads, 
            self.dropout_rate
        )
        
        # Output processing
        self.positionwise_grn = GatedResidualNetwork(
            self.hidden_size, 
            self.hidden_size, 
            self.hidden_size,
            self.dropout_rate
        )
        
        # Quantile heads for each prediction horizon
        self.quantile_heads = nn.ModuleList([
            nn.Linear(self.hidden_size, self.num_quantiles) 
            for _ in self.prediction_horizon
        ])
        
        # Interpretability components
        self.static_enrichment = GatedResidualNetwork(
            self.hidden_size * 2, 
            self.hidden_size, 
            self.hidden_size,
            self.dropout_rate
        )
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        """Initialize weights with Xavier/Glorot initialization."""
        if isinstance(module, nn.Linear):
            torch.nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LSTM):
            for name, param in module.named_parameters():
                if 'weight' in name:
                    torch.nn.init.xavier_uniform_(param)
                elif 'bias' in name:
                    torch.nn.init.zeros_(param)
    
    def create_causal_mask(self, seq_len: int, device: torch.device) -> torch.Tensor:
        """Create causal mask for temporal attention."""
        mask = torch.tril(torch.ones(seq_len, seq_len, device=device))
        return mask.unsqueeze(0).unsqueeze(0)  # (1, 1, seq_len, seq_len)
    
    def forward(self, inputs: Dict[str, torch.Tensor], use_cuda: bool = False) -> Dict[str, torch.Tensor]:
        """
        Forward pass of TFT model.
        
        Args:
            inputs: Dictionary containing:
                - 'historical_features': (batch_size, seq_len, num_features)
                - 'future_features': (batch_size, horizon, num_features) 
                - 'static_features': (batch_size, static_features)
            use_cuda: Whether to use CUDA kernels
            
        Returns:
            Dictionary containing predictions and interpretability outputs
        """
        batch_size = inputs['historical_features'].size(0)
        seq_len = inputs['historical_features'].size(1)
        device = inputs['historical_features'].device
        
        # Process static features
        static_context = self.static_covariate_encoder(inputs.get('static_features', 
            torch.zeros(batch_size, 10, device=device)))
        
        # Variable selection for historical features
        historical_flattened = inputs['historical_features'].reshape(batch_size, -1)
        historical_selected, historical_weights = self.historical_vsn(
            historical_flattened, inputs['historical_features']
        )
        
        # LSTM processing with optional CUDA acceleration
        if use_cuda and CUDA_AVAILABLE:
            try:
                # Use CUDA LSTM kernel
                lstm_output = self._cuda_lstm_forward(historical_selected)
            except Exception as e:
                print(f"CUDA LSTM failed, falling back to PyTorch: {e}")
                lstm_output, _ = self.lstm(historical_selected)
        else:
            lstm_output, _ = self.lstm(historical_selected)
        
        # Static enrichment
        static_expanded = static_context.unsqueeze(1).expand(-1, seq_len, -1)
        enriched_sequence = self.static_enrichment(
            torch.cat([lstm_output, static_expanded], dim=-1)
        )
        
        # Multi-head temporal attention with causal mask
        causal_mask = self.create_causal_mask(seq_len, device)
        attended_output, attention_weights = self.multihead_attn(
            enriched_sequence, enriched_sequence, enriched_sequence, 
            mask=causal_mask, use_cuda=use_cuda
        )
        
        # Position-wise processing
        processed_output = self.positionwise_grn(attended_output)
        
        # Generate predictions for each horizon
        predictions = {}
        for i, horizon in enumerate(self.prediction_horizon):
            # Use last time step for prediction
            last_hidden = processed_output[:, -1, :]  # (batch_size, hidden_size)
            
            if use_cuda and CUDA_AVAILABLE:
                try:
                    # Use CUDA quantile heads
                    quantile_output = self._cuda_quantile_forward(last_hidden, i)
                except Exception as e:
                    print(f"CUDA quantile heads failed, falling back to PyTorch: {e}")
                    quantile_output = self.quantile_heads[i](last_hidden)
            else:
                quantile_output = self.quantile_heads[i](last_hidden)
            
            predictions[f'horizon_{horizon}'] = quantile_output
        
        # Interpretability outputs
        interpretability = {
            'historical_variable_selection': historical_weights,
            'attention_weights': attention_weights,
            'static_weights': torch.norm(static_context, dim=-1)  # Simple static importance
        }
        
        return {
            'predictions': predictions,
            'interpretability': interpretability,
            'hidden_states': processed_output
        }
    
    def _cuda_lstm_forward(self, x: torch.Tensor) -> torch.Tensor:
        """CUDA-accelerated LSTM forward pass."""
        # This would use the CUDA LSTM kernel
        # For now, fallback to PyTorch
        return self.lstm(x)[0]
    
    def _cuda_quantile_forward(self, x: torch.Tensor, head_idx: int) -> torch.Tensor:
        """CUDA-accelerated quantile heads forward pass."""
        # This would use the CUDA quantile kernel
        # For now, fallback to PyTorch
        return self.quantile_heads[head_idx](x)
    
    def predict(self, inputs: Dict[str, torch.Tensor], use_cuda: bool = False) -> Dict[str, torch.Tensor]:
        """Generate predictions without gradients."""
        self.eval()
        with torch.no_grad():
            return self.forward(inputs, use_cuda)
    
    def get_feature_importance(self, inputs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Extract feature importance from variable selection networks."""
        with torch.no_grad():
            outputs = self.forward(inputs)
            return {
                'historical_importance': outputs['interpretability']['historical_variable_selection'],
                'attention_importance': outputs['interpretability']['attention_weights'].mean(dim=(1, 2)),
                'static_importance': outputs['interpretability']['static_weights']
            }
    
    def save_model(self, filepath: Union[str, Path]):
        """Save model state and configuration."""
        filepath = Path(filepath)
        
        # Save model state
        torch.save({
            'model_state_dict': self.state_dict(),
            'config': self.config,
            'model_class': self.__class__.__name__
        }, filepath)
        
        print(f"Model saved to {filepath}")
    
    @classmethod
    def load_model(cls, filepath: Union[str, Path], device: str = 'cuda'):
        """Load model from saved state."""
        filepath = Path(filepath)
        
        checkpoint = torch.load(filepath, map_location=device)
        config = checkpoint['config']
        
        model = cls(config)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(device)
        
        print(f"Model loaded from {filepath}")
        return model


class TFTLoss(nn.Module):
    """Quantile loss for TFT training."""
    
    def __init__(self, quantile_levels: List[float], loss_type: str = 'pinball'):
        super().__init__()
        self.quantile_levels = quantile_levels
        self.loss_type = loss_type
        self.num_quantiles = len(quantile_levels)
    
    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Compute quantile loss.
        
        Args:
            predictions: (batch_size, num_quantiles) or dict of predictions
            targets: (batch_size, 1) target values
            
        Returns:
            Loss value
        """
        if isinstance(predictions, dict):
            # Handle multiple horizons
            total_loss = 0
            for horizon_key, pred in predictions.items():
                loss = self._compute_quantile_loss(pred, targets)
                total_loss += loss
            return total_loss / len(predictions)
        else:
            return self._compute_quantile_loss(predictions, targets)
    
    def _compute_quantile_loss(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Compute pinball loss for quantile predictions."""
        batch_size = predictions.size(0)
        
        if targets.dim() == 1:
            targets = targets.unsqueeze(-1)
        
        # Expand targets to match quantile predictions
        targets_expanded = targets.expand(-1, self.num_quantiles)  # (batch_size, num_quantiles)
        
        # Compute quantile loss
        errors = targets_expanded - predictions  # (batch_size, num_quantiles)
        
        quantile_loss = 0
        for i, q in enumerate(self.quantile_levels):
            loss_i = torch.where(
                errors[:, i] >= 0,
                q * errors[:, i],
                (q - 1) * errors[:, i]
            )
            quantile_loss += loss_i.mean()
        
        return quantile_loss / self.num_quantiles


def create_tft_config(input_size: int, **kwargs) -> Dict:
    """Create default TFT configuration."""
    config = {
        'input_size': input_size,
        'hidden_size': kwargs.get('hidden_size', 256),
        'num_heads': kwargs.get('num_heads', 8),
        'quantile_levels': kwargs.get('quantile_levels', [0.1, 0.5, 0.9]),
        'prediction_horizon': kwargs.get('prediction_horizon', [1, 5, 10]),
        'sequence_length': kwargs.get('sequence_length', 100),
        'dropout_rate': kwargs.get('dropout_rate', 0.1),
        'num_lstm_layers': kwargs.get('num_lstm_layers', 1),
        'static_input_size': kwargs.get('static_input_size', 10),
        'num_historical_features': kwargs.get('num_historical_features', input_size),
        'num_future_features': kwargs.get('num_future_features', 10)
    }
    return config


def main():
    """Example usage of TFT model."""
    print("Temporal Fusion Transformer Example")
    print("=" * 40)
    
    # Create sample data
    batch_size, seq_len, input_size = 32, 100, 64
    
    sample_inputs = {
        'historical_features': torch.randn(batch_size, seq_len, input_size),
        'static_features': torch.randn(batch_size, 10)
    }
    
    sample_targets = torch.randn(batch_size, 1)
    
    # Create model
    config = create_tft_config(input_size=input_size)
    model = TemporalFusionTransformer(config)
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Forward pass
    outputs = model(sample_inputs)
    
    print(f"Predictions shape: {outputs['predictions']['horizon_1'].shape}")
    print(f"Attention weights shape: {outputs['interpretability']['attention_weights'].shape}")
    
    # Loss computation
    loss_fn = TFTLoss(config['quantile_levels'])
    loss = loss_fn(outputs['predictions'], sample_targets)
    
    print(f"Loss: {loss.item():.4f}")
    
    # Feature importance
    importance = model.get_feature_importance(sample_inputs)
    print(f"Historical importance shape: {importance['historical_importance'].shape}")


if __name__ == "__main__":
    main()