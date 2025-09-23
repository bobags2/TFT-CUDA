"""
Temporal Fusion Transformer (TFT) model implementation for financial forecasting.
Integrates with CUDA kernels for high-performance training and inference.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Union, Any
from pathlib import Path

# Silent CUDA backend import
import sys
import io
from contextlib import redirect_stdout
CUDA_AVAILABLE = False
tft_cuda: Optional[Any] = None  # ensure name is always defined
try:
    with redirect_stdout(io.StringIO()):
        import tft_cuda_ext as tft_cuda  # Optional CUDA backend
    CUDA_AVAILABLE = True
except ImportError:
	pass
except Exception:
	# Safety: any unexpected error during optional backend load should not break CPU path
	pass

# Print message only if not testing and CUDA is available
if CUDA_AVAILABLE and not any('test' in arg.lower() for arg in sys.argv):
    print("TFT-CUDA: CUDA backend loaded successfully")


# ============================================================================
# Custom Autograd Functions for CUDA Backward Pass Optimization
# ============================================================================

class CUDAMultiHeadAttentionFunction(torch.autograd.Function):
    """Custom autograd for CUDA multi-head attention with optimized backward pass."""

    @staticmethod
    def forward(ctx, Q, K, V, num_heads, scale):
        """Wrapper forward to provide compatibility with multiple compiled signatures."""
        batch_size, seq_len, d_model = Q.shape
        d_k = d_model // num_heads

        Q_reshaped = Q.view(batch_size, seq_len, num_heads, d_k).transpose(1, 2)
        K_reshaped = K.view(batch_size, seq_len, num_heads, d_k).transpose(1, 2)
        V_reshaped = V.view(batch_size, seq_len, num_heads, d_k).transpose(1, 2)

        # Use FP32 to improve numerical stability; kernels accumulate in FP32
        Q_cuda = Q_reshaped.transpose(1, 2).contiguous().float()
        K_cuda = K_reshaped.transpose(1, 2).contiguous().float()
        V_cuda = V_reshaped.transpose(1, 2).contiguous().float()

        # Preferred new API with explicit output + attn buffers
        try:
            cuda = tft_cuda
            if cuda is None:
                raise RuntimeError("CUDA backend unavailable")
            output = torch.zeros_like(Q_cuda)
            attn_weights = torch.zeros(batch_size, seq_len, num_heads, seq_len,
                                     dtype=torch.float32, device=Q.device)
            cuda.multi_head_attention_forward(
                Q_cuda, K_cuda, V_cuda, output, attn_weights,
                scale, batch_size, seq_len, num_heads, d_k
            )
            output = output.transpose(1, 2).float()
        except Exception:
            try:
                # Fallback v1: older API requiring an explicit output tensor as 4th arg, returns output tensor
                output_buf = torch.zeros_like(Q_cuda)
                cuda = tft_cuda
                if cuda is None:
                    raise RuntimeError("CUDA backend unavailable")
                output_only = cuda.multi_head_attention_forward(
                    Q_cuda, K_cuda, V_cuda, output_buf, scale, batch_size, seq_len, num_heads, d_k
                )
                # Some builds might ignore output_buf and return a tensor; handle both
                out = output_only if isinstance(output_only, torch.Tensor) else output_buf
                output = out.transpose(1, 2).float()
                # Compute attention weights for interpretability and backward using PyTorch
                Qh = Q_cuda.transpose(1, 2).float()  # (B, H, T, D)
                Kh = K_cuda.transpose(1, 2).float()
                scores = torch.matmul(Qh, Kh.transpose(-2, -1)) / np.sqrt(d_k)
                attn_head = torch.softmax(scores, dim=-1)  # (B, H, T, T)
                attn_weights = attn_head.transpose(1, 2).contiguous()  # (B, T, H, T)
            except Exception:
                # Fallback v2: older API returning only output without explicit output buffer
                cuda = tft_cuda
                if cuda is None:
                    raise RuntimeError("CUDA backend unavailable")
                output_only = cuda.multi_head_attention_forward(
                    Q_cuda, K_cuda, V_cuda, scale, batch_size, seq_len, num_heads, d_k
                )
                output = output_only.transpose(1, 2).float()
                Qh = Q_cuda.transpose(1, 2).float()
                Kh = K_cuda.transpose(1, 2).float()
                scores = torch.matmul(Qh, Kh.transpose(-2, -1)) / np.sqrt(d_k)
                attn_head = torch.softmax(scores, dim=-1)
                attn_weights = attn_head.transpose(1, 2).contiguous()

        ctx.save_for_backward(Q_cuda, K_cuda, V_cuda, attn_weights)
        ctx.num_heads = num_heads
        ctx.scale = scale
        ctx.batch_size = batch_size
        ctx.seq_len = seq_len
        ctx.d_k = d_k

        return output.transpose(1, 2).contiguous().view(batch_size, seq_len, d_model), attn_weights
    
    @staticmethod
    def backward(ctx, grad_output, grad_attn_weights):
        Q_cuda, K_cuda, V_cuda, attn_weights = ctx.saved_tensors
        
        batch_size, seq_len = ctx.batch_size, ctx.seq_len
        d_model = ctx.num_heads * ctx.d_k
        
        grad_output = grad_output.view(batch_size, seq_len, ctx.num_heads, ctx.d_k).contiguous().float()
        
        # Gradients are accumulated in FP32 per kernel contract
        grad_Q = torch.zeros_like(Q_cuda, dtype=torch.float32)
        grad_K = torch.zeros_like(K_cuda, dtype=torch.float32)
        grad_V = torch.zeros_like(V_cuda, dtype=torch.float32)
        
        cuda = tft_cuda
        if cuda is None or CUDA_AVAILABLE is False:
            raise RuntimeError("CUDA backend unavailable")
        cuda.multi_head_attention_backward(
            grad_output, attn_weights, Q_cuda, K_cuda, V_cuda,
            grad_Q, grad_K, grad_V,
            ctx.scale, batch_size, seq_len, ctx.num_heads, ctx.d_k
        )
        
        grad_Q = grad_Q.transpose(1, 2).reshape(batch_size, seq_len, d_model).float()
        grad_K = grad_K.transpose(1, 2).reshape(batch_size, seq_len, d_model).float()
        grad_V = grad_V.transpose(1, 2).reshape(batch_size, seq_len, d_model).float()
        
        return grad_Q, grad_K, grad_V, None, None


class CUDALSTMVariableSelectionFunction(torch.autograd.Function):
    """Custom autograd for CUDA LSTM with variable selection backward pass."""
    
    @staticmethod
    def forward(ctx, x, h0, c0, weight_ih, weight_hh, bias_ih, bias_hh):
        batch_size, seq_len, hidden_size = x.shape
        
        output = torch.zeros_like(x)
        hn = torch.zeros_like(h0)
        cn = torch.zeros_like(c0)
        
        hidden_states = torch.zeros(batch_size, seq_len, hidden_size, device=x.device)
        cell_states = torch.zeros(batch_size, seq_len, hidden_size, device=x.device)
        
        cuda = tft_cuda
        if cuda is None:
            raise RuntimeError("CUDA backend unavailable")
        cuda.lstm_variable_selection_forward(
            x, h0, c0, weight_ih, weight_hh, bias_ih, bias_hh,
            output, hn, cn,
            batch_size, seq_len, hidden_size
        )
        
        ctx.save_for_backward(x, h0, c0, weight_ih, weight_hh, bias_ih, bias_hh, 
                            output, hn, cn, hidden_states, cell_states)
        ctx.batch_size = batch_size
        ctx.seq_len = seq_len
        ctx.hidden_size = hidden_size
        
        return output, hn, cn
    
    @staticmethod
    def backward(ctx, grad_output, grad_hn, grad_cn):
        x, h0, c0, weight_ih, weight_hh, bias_ih, bias_hh, output, hn, cn, hidden_states, cell_states = ctx.saved_tensors
        
        grad_x = torch.zeros_like(x)
        grad_h0 = torch.zeros_like(h0)
        grad_c0 = torch.zeros_like(c0)
        grad_weight_ih = torch.zeros_like(weight_ih)
        grad_weight_hh = torch.zeros_like(weight_hh)
        grad_bias_ih = torch.zeros_like(bias_ih) if bias_ih is not None else None
        grad_bias_hh = torch.zeros_like(bias_hh) if bias_hh is not None else None
        
        cuda = tft_cuda
        if cuda is None:
            raise RuntimeError("CUDA backend unavailable")
        cuda.lstm_variable_selection_backward(
            grad_output, grad_hn, grad_cn,
            x, hidden_states, cell_states, weight_ih, weight_hh,
            grad_x, grad_h0, grad_c0, grad_weight_ih, grad_weight_hh,
            grad_bias_ih, grad_bias_hh,
            ctx.batch_size, ctx.seq_len, ctx.hidden_size
        )
        
        return grad_x, grad_h0, grad_c0, grad_weight_ih, grad_weight_hh, grad_bias_ih, grad_bias_hh


class CUDAQuantileHeadsFunction(torch.autograd.Function):
    """Custom autograd for CUDA quantile heads with optimized backward pass."""
    
    @staticmethod
    def forward(ctx, x, weight, bias):
        batch_size, hidden_size = x.shape
        output_size = weight.shape[0]
        # Use T=1 logical time dimension to match CUDA kernel signature
        B, T, D, Q = batch_size, 1, hidden_size, output_size
        # Kernel runs in FP32 internally; allocate FP32 buffer then cast
        output_3d = torch.empty(B, T, Q, device=x.device, dtype=torch.float32)
        # Note: kernel expects weight as (D, Q); PyTorch stores as (Q, D)
        w_t = weight.t().contiguous().float()
        cuda = tft_cuda
        if cuda is None:
            raise RuntimeError("CUDA backend unavailable")
        b = bias.float() if bias is not None else torch.zeros(Q, device=x.device, dtype=torch.float32)
        cuda.quantile_heads_forward(
            x.contiguous().float(), w_t, b,
            output_3d, B, T, D, Q
        )
        output = output_3d.view(B, Q).to(x.dtype)
        
        ctx.save_for_backward(x, weight, bias)
        ctx.batch_size = batch_size
        ctx.hidden_size = hidden_size
        ctx.output_size = output_size
        
        return output
    
    @staticmethod
    def backward(ctx, grad_output):
        x, weight, bias = ctx.saved_tensors
        B, T, D, Q = ctx.batch_size, 1, ctx.hidden_size, ctx.output_size
        # Allocate FP32 grads for accumulation
        grad_x_3d = torch.zeros(B, T, D, device=x.device, dtype=torch.float32)
        # Kernel accumulates grad_weight in (D, Q) layout; allocate accordingly then transpose back
        grad_weight_dq = torch.zeros(D, Q, device=weight.device, dtype=torch.float32)
        grad_bias = torch.zeros_like(bias, dtype=torch.float32) if bias is not None else torch.tensor([], device=x.device, dtype=torch.float32)

        # Expand grad_output to (B,T,Q) in FP32 for kernel math
        go = grad_output.view(B, T, Q).contiguous().float()
        # Pass weight in (D, Q) as expected by kernel
        w_t = weight.t().contiguous().float()
        cuda = tft_cuda
        if cuda is None:
            raise RuntimeError("CUDA backend unavailable")
        cuda.quantile_heads_backward(
            go, x.contiguous(), w_t,
            grad_x_3d, grad_weight_dq, grad_bias,
            B, T, D, Q
        )

        # Cast back to original dtypes
        grad_x = grad_x_3d.view(B, D).to(x.dtype)
        grad_weight = grad_weight_dq.t().to(weight.dtype)
        grad_bias = grad_bias.to(bias.dtype) if bias is not None else None
        return grad_x, grad_weight, grad_bias


# ============================================================================


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
    """Variable Selection Network (VSN) for feature selection with CUDA acceleration."""
    
    def __init__(self, input_size: int, num_inputs: int, hidden_size: int, dropout_rate: float = 0.1):
        super().__init__()
        self.input_size = input_size
        self.num_inputs = num_inputs
        self.hidden_size = hidden_size
        
        self.selection_layer = nn.Sequential(
            nn.Linear(input_size * num_inputs, hidden_size),
            nn.ELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_size, num_inputs),
            nn.Softmax(dim=-1)
        )
        
        self.variable_transform = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ELU(),
            nn.Dropout(dropout_rate)
        )
    
    def forward(self, flattened_inputs, processed_inputs):
        batch_size, seq_len, total_features = processed_inputs.shape
        
        if total_features % self.input_size == 0:
            actual_num_inputs = total_features // self.input_size
        else:
            actual_num_inputs = 1
            self.input_size = total_features
        
        # Get selection weights
        if flattened_inputs.size(-1) != actual_num_inputs * self.input_size:
            adaptive_layer = nn.Linear(flattened_inputs.size(-1), actual_num_inputs, device=flattened_inputs.device)
            sparse_weights = F.softmax(adaptive_layer(flattened_inputs), dim=-1)
        else:
            sparse_weights = self.selection_layer(flattened_inputs)
        
        # Use CUDA VSN aggregate if available
        if CUDA_AVAILABLE and actual_num_inputs > 1:
            try:
                processed_inputs_reshaped = processed_inputs.view(batch_size, seq_len, actual_num_inputs, self.input_size)
                transformed_features = []
                for i in range(actual_num_inputs):
                    var_input = processed_inputs_reshaped[:, :, i, :]
                    transformed = self.variable_transform(var_input)
                    transformed_features.append(transformed)
                
                transformed_features = torch.stack(transformed_features, dim=-2)
                outputs = torch.zeros(batch_size, seq_len, self.hidden_size, device=processed_inputs.device)
                
                # CUDA VSN aggregation
                cuda = tft_cuda
                if cuda is None:
                    raise RuntimeError("CUDA backend unavailable")
                cuda.vsn_aggregate(
                    transformed_features, sparse_weights.unsqueeze(1).unsqueeze(-1),
                    outputs, batch_size, seq_len, actual_num_inputs, self.hidden_size
                )
            except Exception:
                # Fallback to PyTorch
                outputs = self._pytorch_vsn(processed_inputs, sparse_weights, batch_size, seq_len, actual_num_inputs)
        else:
            outputs = self._pytorch_vsn(processed_inputs, sparse_weights, batch_size, seq_len, actual_num_inputs)
        
        return outputs, sparse_weights
    
    def _pytorch_vsn(self, processed_inputs, sparse_weights, batch_size, seq_len, actual_num_inputs):
        """PyTorch fallback for VSN."""
        if actual_num_inputs == 1:
            transformed = self.variable_transform(processed_inputs)
            outputs = transformed * sparse_weights.unsqueeze(1).unsqueeze(-1)
        else:
            processed_inputs_reshaped = processed_inputs.view(batch_size, seq_len, actual_num_inputs, self.input_size)
            transformed_features = []
            for i in range(actual_num_inputs):
                var_input = processed_inputs_reshaped[:, :, i, :]
                transformed = self.variable_transform(var_input)
                transformed_features.append(transformed)
            
            transformed_features = torch.stack(transformed_features, dim=-2)
            weights_expanded = sparse_weights.unsqueeze(1).unsqueeze(-1)
            outputs = torch.sum(weights_expanded * transformed_features, dim=-2)
        
        return outputs


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
        
    def forward(self, query, key, value, mask=None, use_cuda=True):
        batch_size, seq_len = query.size(0), query.size(1)

        # Linear projections in model space (B, T, d_model)
        Q_lin = self.W_q(query)
        K_lin = self.W_k(key)
        V_lin = self.W_v(value)

        # Use custom CUDA autograd if available
        if use_cuda and CUDA_AVAILABLE and tft_cuda is not None:
            try:
                # Gate: disable CUDA attention after first failure
                if getattr(self, "_cuda_attn_disabled", False):
                    raise RuntimeError("CUDA attention disabled due to earlier failure")
                attention_output, attn_weights = CUDAMultiHeadAttentionFunction.apply(
                    Q_lin, K_lin, V_lin, self.num_heads, float(1.0/np.sqrt(self.d_k))
                )
                # Health check: if CUDA returns near-zero tensors, fallback
                if (attention_output.abs().sum() == 0) or (attn_weights.abs().sum() == 0):
                    raise RuntimeError("CUDA attention produced all zeros; using PyTorch fallback")
            except Exception as e:
                # Avoid dumping giant tensors from the extension's signature error
                msg = str(e).splitlines()[0]
                print(f"CUDA attention failed, falling back to PyTorch: {msg}")
                self._cuda_attn_disabled = True
                # Fallback path uses PyTorch implementation
                Q = Q_lin.view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
                K = K_lin.view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
                V = V_lin.view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
                attention_output, attn_weights = self._pytorch_attention(Q, K, V, mask)
        else:
            Q = Q_lin.view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
            K = K_lin.view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
            V = V_lin.view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
            attention_output, attn_weights = self._pytorch_attention(Q, K, V, mask)
        
        # Concatenate heads and ensure (B, T, d_model) layout before projection
        if attention_output.dim() == 4:
            # (B, H, T, D) or (B, T, H, D) -> (B, T, H*D)
            # Try to detect which layout and normalize
            if attention_output.size(1) == self.num_heads:
                # (B, H, T, D)
                attention_output = attention_output.transpose(1, 2).contiguous()
            elif attention_output.size(2) == self.num_heads:
                # (B, T, H, D)
                pass  # already (B, T, H, D)
            else:
                # Unexpected layout; fall back to moving head dim to middle conservatively
                attention_output = attention_output.transpose(1, 2).contiguous()
            attention_output = attention_output.view(batch_size, seq_len, self.d_model)
        elif attention_output.dim() == 3 and attention_output.size(-1) == self.d_k and attention_output.size(1) == self.num_heads:
            # (B, H, T, D) missing a dim? normalize via unsqueeze
            attention_output = attention_output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        
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
        self.output_size = config.get('output_size', 3)  # 3 outputs per horizon: return, price, direction
        self.hidden_size = config.get('hidden_size', 512)
        self.num_heads = config.get('num_heads', 4)
        self.num_quantiles = len(config.get('quantile_levels', [0.1, 0.5, 0.9]))
        self.prediction_horizon = config.get('prediction_horizon', [1, 5, 10])
        self.sequence_length = config.get('sequence_length', 512)
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
            dropout=self.dropout_rate if config.get('num_lstm_layers', 2) > 1 else 0,
            num_layers=config.get('num_lstm_layers', 2)
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
        
        # Output heads for each prediction horizon (return, price, direction)
        self.quantile_heads = nn.ModuleList([
            nn.Linear(self.hidden_size, self.output_size) 
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

        # Prepare static features and context first (fix: ensure static_context is bound)
        static_features = inputs.get(
            'static_features',
            torch.zeros(batch_size, self.config.get('static_input_size', 10), device=device)
        )
        static_context = self.static_covariate_encoder(static_features)

        # Static feature importance (fallback to norm; CUDA path requires gradient/weights not available here)
        static_importance = torch.norm(static_context, dim=-1)
        
        # Variable selection for historical features
        historical_flattened = inputs['historical_features'].reshape(batch_size, -1)
        historical_selected, historical_weights = self.historical_vsn(
            historical_flattened, inputs['historical_features']
        )
        
        # LSTM processing with CUDA acceleration
        if use_cuda and CUDA_AVAILABLE:
            try:
                lstm_output = self._cuda_lstm_forward(historical_selected)
            except Exception as e:
                # Avoid printing full pybind "Invoked with:" dumps
                msg = str(e).splitlines()[0]
                print(f"CUDA LSTM failed: {msg}")
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

        # Optional CUDA attention aggregation to get temporal importance per time step
        temporal_importance = None
        if use_cuda and CUDA_AVAILABLE and tft_cuda is not None:
            try:
                temporal_importance = torch.zeros(batch_size, seq_len, device=device, dtype=torch.float32)
                cuda = tft_cuda
                if cuda is None:
                    raise RuntimeError("CUDA backend unavailable")
                cuda.attention_aggregate(
                    attention_weights.contiguous().float(), temporal_importance, batch_size, seq_len, self.num_heads
                )
            except Exception:
                temporal_importance = attention_weights.mean(dim=2).mean(dim=-1)  # (B, T)
        else:
            temporal_importance = attention_weights.mean(dim=2).mean(dim=-1)  # (B, T)
        
        # Position-wise processing
        processed_output = self.positionwise_grn(attended_output)
        
        # Generate predictions for each horizon with CUDA acceleration
        predictions = {}
        for i, horizon in enumerate(self.prediction_horizon):
            last_hidden = processed_output[:, -1, :]
            
            if use_cuda and CUDA_AVAILABLE:
                try:
                    quantile_output = self._cuda_quantile_forward(last_hidden, i)
                except Exception as e:
                    print(f"CUDA quantile failed: {e}")
                    quantile_output = self.quantile_heads[i](last_hidden)
            else:
                quantile_output = self.quantile_heads[i](last_hidden)
            
            predictions[f'horizon_{horizon}'] = quantile_output
        
        # Interpretability outputs with CUDA-calculated importance
        interpretability = {
            'historical_variable_selection': historical_weights,
            'attention_weights': attention_weights,
            'temporal_importance': temporal_importance,
            # fix: always use computed static_importance
            'static_weights': static_importance
        }

        return {
            'predictions': predictions,
            'interpretability': interpretability,
            'hidden_states': processed_output
        }
    
    def _cuda_lstm_forward(self, x: torch.Tensor) -> torch.Tensor:
        """CUDA-accelerated LSTM forward pass with variable selection."""
        # Extension LSTM path is not implemented in current build; use PyTorch LSTM
        return self.lstm(x)[0]
    
    def _cuda_quantile_forward(self, x: torch.Tensor, head_idx: int) -> torch.Tensor:
        """CUDA-accelerated quantile heads forward pass."""
        if CUDA_AVAILABLE:
            try:
                if getattr(self, "_cuda_quantile_disabled", False):
                    raise RuntimeError("CUDA quantile disabled due to earlier failure")
                # Use custom autograd function for forward and backward
                output = CUDAQuantileHeadsFunction.apply(
                    x, 
                    self.quantile_heads[head_idx].weight,
                    self.quantile_heads[head_idx].bias
                )
                if output.abs().sum() == 0:
                    raise RuntimeError("CUDA quantile produced all zeros; using PyTorch fallback")
                return output
            except Exception as e:
                print(f"CUDA quantile failed: {e}")
                self._cuda_quantile_disabled = True
                return self.quantile_heads[head_idx](x)
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
        'hidden_size': kwargs.get('hidden_size', 512),
        'num_heads': kwargs.get('num_heads', 4),
        'quantile_levels': kwargs.get('quantile_levels', [0.1, 0.5, 0.9]),
        'prediction_horizon': kwargs.get('prediction_horizon', [1, 5, 10]),
        'sequence_length': kwargs.get('sequence_length', 512),
        'dropout_rate': kwargs.get('dropout_rate', 0.1),
        'num_lstm_layers': kwargs.get('num_lstm_layers', 2),
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