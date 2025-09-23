/*
Pure PyTorch extension bindings for TFT CUDA operations.
Uses torch::Tensor and pure C++ without PyBind11.
*/

#include <torch/extension.h>
#include "../pytorch_compat.h"  
#include "../tft_cuda.h"

// PyTorch extension module definition
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.doc() = "TFT CUDA accelerated operations - Pure PyTorch Extension";
    
    // LSTM operations
    m.def("lstm_variable_selection_forward", &tft_cuda::lstm_variable_selection_forward,
        "LSTM variable selection forward (raises to trigger PyTorch fallback)");
    m.def("lstm_variable_selection_backward", &tft_cuda::lstm_variable_selection_backward,
        "LSTM variable selection backward (raises to trigger PyTorch fallback)");
    
    // Multi-Head Attention operations
    m.def("multi_head_attention_forward", &tft_cuda::multi_head_attention_forward,
        "Multi-head attention forward (FP16, RoPE)");
    m.def("multi_head_attention_backward", &tft_cuda::multi_head_attention_backward,
        "Multi-head attention backward (FP16 grads->FP32)");
    
    // Linear layer operations
    m.def("linear_forward", &tft_cuda::linear_forward,
        "Linear forward (FP16)");
    m.def("linear_backward", &tft_cuda::linear_backward,
        "Linear backward (FP16 inputs, FP32 grads)");
    
    // Quantile operations
    m.def("quantile_heads_forward", &tft_cuda::quantile_heads_forward,
        "Quantile heads forward (FP32 kernel)");
    m.def("quantile_heads_backward", &tft_cuda::quantile_heads_backward,
        "Quantile heads backward (accumulate FP32 grads)");
    m.def("quantile_loss", &tft_cuda::quantile_loss,
        "Quantile pinball loss (per-quantile)");
    
    // Interpretability operations
    m.def("attention_aggregate", &tft_cuda::attention_aggregate,
        "Aggregate attention weights across heads");
    m.def("vsn_aggregate", &tft_cuda::vsn_aggregate,
        "Aggregate VSN selection gates across time");
    m.def("static_embedding_importance", &tft_cuda::static_embedding_importance,
        "Compute static embedding importance via gradients");
}