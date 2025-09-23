/*
TFT-CUDA public C++ interface
Defines the interface between Python and CUDA kernels.
Note: We use preallocated output tensors to avoid extra allocations and
to align with Python call sites in `python/tft_model.py`.
*/

#pragma once

#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>

// Forward declarations for CUDA kernels
namespace tft_cuda {

// LSTM operations (CUDA path is optional; Python falls back to PyTorch if unavailable)
void lstm_variable_selection_forward(
    const torch::Tensor& x,           // (B, T, N) FP16/FP32
    const torch::Tensor& h0,          // (B, N)
    const torch::Tensor& c0,          // (B, N)
    const c10::optional<torch::Tensor>& weight_ih, // (4N, N) or (N,4N) depending on layout
    const c10::optional<torch::Tensor>& weight_hh, // (4N, N)
    const c10::optional<torch::Tensor>& bias_ih,   // (4N)
    const c10::optional<torch::Tensor>& bias_hh,   // (4N)
    torch::Tensor& output,             // (B, T, N)
    torch::Tensor& hn,                 // (B, N)
    torch::Tensor& cn,                 // (B, N)
    int batch_size,
    int seq_len,
    int hidden_size
);

void lstm_variable_selection_backward(
    const torch::Tensor& grad_output, // (B, T, N)
    const torch::Tensor& grad_hn,     // (B, N)
    const torch::Tensor& grad_cn,     // (B, N)
    const torch::Tensor& x,           // (B, T, N)
    const torch::Tensor& hidden_states, // (B, T, N)
    const torch::Tensor& cell_states,   // (B, T, N)
    const torch::Tensor& weight_ih,
    const torch::Tensor& weight_hh,
    torch::Tensor& grad_x,
    torch::Tensor& grad_h0,
    torch::Tensor& grad_c0,
    torch::Tensor& grad_weight_ih,
    torch::Tensor& grad_weight_hh,
    c10::optional<torch::Tensor>& grad_bias_ih,
    c10::optional<torch::Tensor>& grad_bias_hh,
    int batch_size,
    int seq_len,
    int hidden_size
);

// Multi-Head Attention operations
void multi_head_attention_forward(
    const torch::Tensor& Q,            // (B, T, H, D) FP16
    const torch::Tensor& K,            // (B, T, H, D) FP16
    const torch::Tensor& V,            // (B, T, H, D) FP16
    torch::Tensor& output,             // (B, T, H, D) FP16
    torch::Tensor& attn_weights,       // (B, T, H, T) FP32
    float theta,                       // RoPE parameter
    int batch_size,
    int seq_len,
    int num_heads,
    int head_dim
);

void multi_head_attention_backward(
    const torch::Tensor& grad_output,  // (B, T, H, D) FP16
    const torch::Tensor& attn_weights, // (B, T, H, T) FP32
    const torch::Tensor& Q,            // (B, T, H, D) FP16
    const torch::Tensor& K,            // (B, T, H, D) FP16
    const torch::Tensor& V,            // (B, T, H, D) FP16
    torch::Tensor& grad_Q,             // (B, T, H, D) FP32
    torch::Tensor& grad_K,             // (B, T, H, D) FP32
    torch::Tensor& grad_V,             // (B, T, H, D) FP32
    float theta,
    int batch_size,
    int seq_len,
    int num_heads,
    int head_dim
);

// Linear layer operations
void linear_forward(
    const torch::Tensor& input,        // (M, K) FP16
    const torch::Tensor& weight,       // (K, N) FP16
    const c10::optional<torch::Tensor>& bias, // (N) FP16
    torch::Tensor& output,             // (M, N) FP16
    int M, int K, int N
);

void linear_backward(
    const torch::Tensor& grad_output,  // (M, N) FP16
    const torch::Tensor& input,        // (M, K) FP16
    torch::Tensor& dL_dW,              // (K, N) FP32
    torch::Tensor& dL_db,              // (N) FP32
    int M, int K, int N
);

// Quantile head operations
void quantile_heads_forward(
    const torch::Tensor& input,        // (B, T, D) FP16/FP32
    const torch::Tensor& weight,       // (D, Q)   FP16/FP32
    const c10::optional<torch::Tensor>& bias,         // (Q)
    torch::Tensor& output,             // (B, T, Q)
    int batch_size,
    int seq_len,
    int input_dim,
    int num_quantiles
);

void quantile_loss(
    const torch::Tensor& predictions,  // (B, T, Q) FP16/FP32
    const torch::Tensor& targets,      // (B, T)    FP16/FP32
    const torch::Tensor& quantiles,    // (Q)       FP16/FP32
    torch::Tensor& losses,             // (B, T, Q) FP32
    int batch_size,
    int seq_len,
    int num_quantiles
);

void quantile_heads_backward(
    const torch::Tensor& grad_output,  // (B, T, Q) FP16/FP32
    const torch::Tensor& input,        // (B, T, D) FP16/FP32
    const torch::Tensor& weight,       // (D, Q)    FP16/FP32
    torch::Tensor& dL_dinput,          // (B, T, D) FP32
    torch::Tensor& dL_dW,              // (D, Q)    FP32
    torch::Tensor& dL_db,              // (Q)       FP32
    int batch_size,
    int seq_len,
    int input_dim,
    int num_quantiles
);

// Static encoder operations (optional)
void static_encoder_forward(
    const torch::Tensor& input,        // (B, S) FP16
    const torch::Tensor& W1,           // (S, 64) FP16
    const torch::Tensor& W2,           // (64, 32) FP16
    const torch::Tensor& gamma,        // (32) FP16
    const torch::Tensor& beta,         // (32) FP16
    torch::Tensor& output,             // (B, 32) FP16
    int batch_size,
    int static_size
);

// Layer normalization
void layer_norm_backward(
    const torch::Tensor& grad_output,  // (B, D) FP16
    const torch::Tensor& input,        // (B, D) FP16
    const torch::Tensor& gamma,        // (D)    FP16
    const torch::Tensor& mean,         // (B)    FP32
    const torch::Tensor& inv_std,      // (B)    FP32
    torch::Tensor& dL_dgamma,          // (D)    FP32
    torch::Tensor& dL_dbeta,           // (D)    FP32
    torch::Tensor& dL_dinput,          // (B, D) FP32
    int batch_size,
    int hidden_size
);

// Interpretability operations
void attention_aggregate(
    const torch::Tensor& attn_weights,   // (B, T, H, T) FP32
    torch::Tensor& temporal_importance,  // (B, T) FP32
    int batch_size,
    int seq_len,
    int num_heads
);

void vsn_aggregate(
    const torch::Tensor& selection_gates, // (B, T, N) FP32
    torch::Tensor& static_importance,     // (B, N) FP32
    const torch::Tensor& V_s,             // (N, N) FP32 (optional; identity if not used)
    int batch_size,
    int seq_len,
    int num_features
);

void static_embedding_importance(
    const torch::Tensor& grads,            // (B, 32) FP32
    const torch::Tensor& static_embeddings,// (B, 32) FP32
    const torch::Tensor& W1,               // (S, 64) FP32
    torch::Tensor& importance,             // (B, S) FP32
    int batch_size,
    int static_size
);

} // namespace tft_cuda