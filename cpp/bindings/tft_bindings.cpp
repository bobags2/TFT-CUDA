"""
PyBind11 bindings for TFT CUDA operations.
Exposes C++/CUDA functions to Python.
"""

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <torch/extension.h>
#include "tft_cuda.h"

namespace py = pybind11;

PYBIND11_MODULE(tft_cuda, m) {
    m.doc() = "TFT CUDA accelerated operations";
    
    // LSTM operations
    m.def("lstm_variable_selection_forward", &tft_cuda::lstm_variable_selection_forward,
          "LSTM with Variable Selection forward pass",
          py::arg("input"), py::arg("W_i"), py::arg("W_h"), py::arg("bias"), py::arg("V_s"),
          py::arg("batch_size"), py::arg("seq_len"), py::arg("hidden_size"));
    
    m.def("lstm_variable_selection_backward", &tft_cuda::lstm_variable_selection_backward,
          "LSTM with Variable Selection backward pass",
          py::arg("grad_output"), py::arg("input"), py::arg("hidden_states"), 
          py::arg("cell_states"), py::arg("selection_gates"), py::arg("W_i"), 
          py::arg("W_h"), py::arg("V_s"), py::arg("batch_size"), py::arg("seq_len"), 
          py::arg("hidden_size"));
    
    // Multi-Head Attention operations
    m.def("multi_head_attention_mp", &tft_cuda::multi_head_attention_forward,
          "Multi-Head Attention forward pass with mixed precision",
          py::arg("Q"), py::arg("K"), py::arg("V"), py::arg("attn_weights"), 
          py::arg("theta"), py::arg("batch_size"), py::arg("seq_len"), 
          py::arg("num_heads"), py::arg("head_dim"));
    
    m.def("mha_backward_mp", &tft_cuda::multi_head_attention_backward,
          "Multi-Head Attention backward pass with mixed precision",
          py::arg("grad_output"), py::arg("Q"), py::arg("K"), py::arg("V"), 
          py::arg("attn_weights"), py::arg("theta"), py::arg("batch_size"), 
          py::arg("seq_len"), py::arg("num_heads"), py::arg("head_dim"));
    
    // Linear layer operations
    m.def("linear_forward_mp", &tft_cuda::linear_forward,
          "Linear layer forward pass with mixed precision",
          py::arg("input"), py::arg("weight"), py::arg("bias"), 
          py::arg("M"), py::arg("K"), py::arg("N"));
    
    m.def("linear_backward_mp", &tft_cuda::linear_backward,
          "Linear layer backward pass",
          py::arg("grad_output"), py::arg("input"), py::arg("weight"),
          py::arg("M"), py::arg("K"), py::arg("N"));
    
    // Quantile head operations
    m.def("quantile_heads", &tft_cuda::quantile_heads_forward,
          "Quantile heads forward pass",
          py::arg("input"), py::arg("weight"), py::arg("bias"), py::arg("quantiles"),
          py::arg("batch_size"), py::arg("seq_len"), py::arg("input_dim"), 
          py::arg("num_quantiles"));
    
    m.def("quantile_loss", &tft_cuda::quantile_loss,
          "Quantile (Pinball) loss computation",
          py::arg("predictions"), py::arg("targets"), py::arg("quantiles"),
          py::arg("batch_size"), py::arg("seq_len"), py::arg("num_quantiles"));
    
    m.def("quantile_heads_backward_mp", &tft_cuda::quantile_heads_backward,
          "Quantile heads backward pass with mixed precision",
          py::arg("grad_output"), py::arg("input"), py::arg("weight"), 
          py::arg("quantiles"), py::arg("predictions"), py::arg("targets"),
          py::arg("batch_size"), py::arg("seq_len"), py::arg("input_dim"), 
          py::arg("num_quantiles"));
    
    // Static encoder operations
    m.def("static_encoder_mp", &tft_cuda::static_encoder_forward,
          "Static covariate encoder with mixed precision",
          py::arg("input"), py::arg("W1"), py::arg("W2"), py::arg("gamma"), py::arg("beta"),
          py::arg("batch_size"), py::arg("static_size"), py::arg("hidden_size"));
    
    // Layer normalization
    m.def("layer_norm_backward_mp", &tft_cuda::layer_norm_backward,
          "Layer normalization backward pass",
          py::arg("grad_output"), py::arg("input"), py::arg("gamma"), 
          py::arg("mean"), py::arg("inv_std"), py::arg("batch_size"), py::arg("hidden_size"));
    
    // Interpretability operations
    m.def("attention_aggregate", &tft_cuda::attention_aggregate,
          "Aggregate attention weights for temporal interpretability",
          py::arg("attn_weights"), py::arg("batch_size"), py::arg("seq_len"), py::arg("num_heads"));
    
    m.def("vsn_aggregate", &tft_cuda::vsn_aggregate,
          "Aggregate Variable Selection Network weights",
          py::arg("selection_weights"), py::arg("batch_size"), py::arg("seq_len"), py::arg("num_features"));
    
    m.def("static_embedding_importance", &tft_cuda::static_embedding_importance,
          "Compute static embedding importance scores",
          py::arg("embeddings"), py::arg("weights"), py::arg("batch_size"), 
          py::arg("hidden_size"), py::arg("output_size"));
}