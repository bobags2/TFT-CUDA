"""
C++ header for TFT CUDA operations.
Defines the interface between Python and CUDA kernels.
"""

#pragma once

#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>

// Forward declarations for CUDA kernels
namespace tft_cuda {

// LSTM operations
torch::Tensor lstm_variable_selection_forward(
    torch::Tensor input,        // (B, T, N)
    torch::Tensor W_i,          // (N, 4*N)
    torch::Tensor W_h,          // (N, 4*N)
    torch::Tensor bias,         // (4*N)
    torch::Tensor V_s,          // (N, N)
    int batch_size,
    int seq_len,
    int hidden_size
);

std::vector<torch::Tensor> lstm_variable_selection_backward(
    torch::Tensor grad_output,  // (B, T, N)
    torch::Tensor input,
    torch::Tensor hidden_states,
    torch::Tensor cell_states,
    torch::Tensor selection_gates,
    torch::Tensor W_i,
    torch::Tensor W_h,
    torch::Tensor V_s,
    int batch_size,
    int seq_len,
    int hidden_size
);

// Multi-Head Attention operations
torch::Tensor multi_head_attention_forward(
    torch::Tensor Q,            // (B, T, H, D)
    torch::Tensor K,            // (B, T, H, D)
    torch::Tensor V,            // (B, T, H, D)
    torch::Tensor attn_weights, // (B, T, H, T) - output
    float theta,                // RoPE parameter
    int batch_size,
    int seq_len,
    int num_heads,
    int head_dim
);

std::vector<torch::Tensor> multi_head_attention_backward(
    torch::Tensor grad_output,  // (B, T, H, D)
    torch::Tensor Q,
    torch::Tensor K,
    torch::Tensor V,
    torch::Tensor attn_weights, // from forward pass
    float theta,
    int batch_size,
    int seq_len,
    int num_heads,
    int head_dim
);

// Linear layer operations
torch::Tensor linear_forward(
    torch::Tensor input,        // (M, K)
    torch::Tensor weight,       // (N, K)
    torch::Tensor bias,         // (N)
    int M, int K, int N
);

std::vector<torch::Tensor> linear_backward(
    torch::Tensor grad_output,  // (M, N)
    torch::Tensor input,        // (M, K)
    torch::Tensor weight,       // (N, K)
    int M, int K, int N
);

// Quantile head operations
torch::Tensor quantile_heads_forward(
    torch::Tensor input,        // (B, T, D)
    torch::Tensor weight,       // (D, Q)
    torch::Tensor bias,         // (Q)
    torch::Tensor quantiles,    // (Q)
    int batch_size,
    int seq_len,
    int input_dim,
    int num_quantiles
);

torch::Tensor quantile_loss(
    torch::Tensor predictions,  // (B, T, Q)
    torch::Tensor targets,      // (B, T)
    torch::Tensor quantiles,    // (Q)
    int batch_size,
    int seq_len,
    int num_quantiles
);

std::vector<torch::Tensor> quantile_heads_backward(
    torch::Tensor grad_output,  // (B, T, Q)
    torch::Tensor input,        // (B, T, D)
    torch::Tensor weight,       // (D, Q)
    torch::Tensor quantiles,    // (Q)
    torch::Tensor predictions,  // (B, T, Q)
    torch::Tensor targets,      // (B, T)
    int batch_size,
    int seq_len,
    int input_dim,
    int num_quantiles
);

// Static encoder operations
torch::Tensor static_encoder_forward(
    torch::Tensor input,        // (B, S)
    torch::Tensor W1,           // (S, H)
    torch::Tensor W2,           // (H, H)
    torch::Tensor gamma,        // (H)
    torch::Tensor beta,         // (H)
    int batch_size,
    int static_size,
    int hidden_size
);

// Layer normalization
std::vector<torch::Tensor> layer_norm_backward(
    torch::Tensor grad_output,  // (B, H)
    torch::Tensor input,        // (B, H)
    torch::Tensor gamma,        // (H)
    torch::Tensor mean,         // (B)
    torch::Tensor inv_std,      // (B)
    int batch_size,
    int hidden_size
);

// Interpretability operations
torch::Tensor attention_aggregate(
    torch::Tensor attn_weights, // (B, T, H, T)
    int batch_size,
    int seq_len,
    int num_heads
);

torch::Tensor vsn_aggregate(
    torch::Tensor selection_weights, // (B, T, N)
    int batch_size,
    int seq_len,
    int num_features
);

torch::Tensor static_embedding_importance(
    torch::Tensor embeddings,   // (B, H)
    torch::Tensor weights,      // (H, O)
    int batch_size,
    int hidden_size,
    int output_size
);

} // namespace tft_cuda