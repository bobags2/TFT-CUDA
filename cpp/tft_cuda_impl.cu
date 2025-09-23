#include "tft_cuda.h"
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <stdexcept>
#include <vector>

// Forward declare CUDA kernels (defined in .cu files)
// Forward passes
__global__ void multi_head_attention_mp(const __half* Q, const __half* K, const __half* V,
                                        __half* output, float* attn_weights,
                                        const float theta, const int B, const int T, const int H, const int D);
__global__ void linear_forward_mp(const __half* input, const __half* weight, const __half* bias,
                                  __half* output, const int M, const int K, const int N);
__global__ void static_encoder_forward(const __half* input, const __half* W1, const __half* W2,
                                       const __half* gamma, const __half* beta, __half* output,
                                       const int B, const int S);
__global__ void attention_aggregate(const float* attn_weights, float* temporal_importance,
                                    const int B, const int T, const int H);
__global__ void vsn_aggregate(const float* selection_gates, float* static_importance,
                              const float* V_s, const int B, const int T, const int N);
__global__ void static_embedding_importance(const float* grads, const float* static_embeddings,
                                            float* importance, const float* W1,
                                            const int B, const int S);
__global__ void quantile_heads(const float* input, const float* W, const float* b,
                               float* outputs, const float* quantiles,
                               const int B, const int T, const int D, const int Q);
__global__ void quantile_loss(const float* preds, const float* targets, const float* quantiles,
                              float* losses, const int B, const int T, const int Q);

// Backward passes
__global__ void linear_backward_mp(const __half* dL_doutput, const __half* input,
                                   float* dL_dW, float* dL_db, const int M, const int K, const int N);
__global__ void mha_backward_mp(const __half* Q, const __half* K, const __half* V,
                                const float* attn_weights, const __half* dL_doutput,
                                float* dL_dQ, float* dL_dK, float* dL_dV,
                                const float theta, const int B, const int T, const int H, const int D);
__global__ void layer_norm_backward_mp(const __half* dL_doutput, const __half* input, const __half* gamma,
                                       const float* mean, const float* inv_std,
                                       float* dL_dgamma, float* dL_dbeta, float* dL_dinput,
                                       int B, int D);
__global__ void quantile_heads_backward_mp(const __half* input, const __half* W, const __half* b,
                                           const __half* quantiles, const __half* preds, const __half* targets,
                                           float* dL_dW, float* dL_db, float* dL_dinput,
                                           const int B, const int T, const int D, const int Q);

namespace tft_cuda {

static inline void check_cuda_tensor(const torch::Tensor& t, c10::ScalarType dtype, const char* name) {
    TORCH_CHECK(t.is_cuda(), name, " must be a CUDA tensor");
    TORCH_CHECK(t.is_contiguous(), name, " must be contiguous");
    if (dtype != c10::ScalarType::Undefined) {
        TORCH_CHECK(t.scalar_type() == dtype, name, " has wrong dtype");
    }
}

static inline dim3 ceil_div_dim3(int a, int b) { return dim3((a + b - 1) / b); }

void multi_head_attention_forward(
    const torch::Tensor& Q,
    const torch::Tensor& K,
    const torch::Tensor& V,
    torch::Tensor& output,
    torch::Tensor& attn_weights,
    float theta,
    int B, int T, int H, int D
) {
    c10::cuda::CUDAGuard device_guard(Q.device());
    auto stream = at::cuda::getCurrentCUDAStream();

    check_cuda_tensor(Q, c10::kHalf, "Q");
    check_cuda_tensor(K, c10::kHalf, "K");
    check_cuda_tensor(V, c10::kHalf, "V");
    check_cuda_tensor(output, c10::kHalf, "output");
    check_cuda_tensor(attn_weights, c10::kFloat, "attn_weights");

    TORCH_CHECK(Q.sizes() == std::vector<int64_t>({B, T, H, D}), "Q shape mismatch");
    TORCH_CHECK(K.sizes() == std::vector<int64_t>({B, T, H, D}), "K shape mismatch");
    TORCH_CHECK(V.sizes() == std::vector<int64_t>({B, T, H, D}), "V shape mismatch");
    TORCH_CHECK(output.sizes() == std::vector<int64_t>({B, T, H, D}), "output shape mismatch");
    TORCH_CHECK(attn_weights.sizes() == std::vector<int64_t>({B, T, H, T}), "attn_weights shape mismatch");
    TORCH_CHECK(D <= 1024, "head_dim D must be <= 1024");

    dim3 grid(B, T, H);
    dim3 block(D);
    ::multi_head_attention_mp<<<grid, block, 0, stream>>> (
        reinterpret_cast<const __half*>(Q.data_ptr<at::Half>()),
        reinterpret_cast<const __half*>(K.data_ptr<at::Half>()),
        reinterpret_cast<const __half*>(V.data_ptr<at::Half>()),
        reinterpret_cast<__half*>(output.data_ptr<at::Half>()),
        attn_weights.data_ptr<float>(),
        theta, B, T, H, D
    );
}

void multi_head_attention_backward(
    const torch::Tensor& grad_output,
    const torch::Tensor& attn_weights,
    const torch::Tensor& Q,
    const torch::Tensor& K,
    const torch::Tensor& V,
    torch::Tensor& grad_Q,
    torch::Tensor& grad_K,
    torch::Tensor& grad_V,
    float theta,
    int B, int T, int H, int D
) {
    c10::cuda::CUDAGuard device_guard(Q.device());
    auto stream = at::cuda::getCurrentCUDAStream();

    check_cuda_tensor(grad_output, c10::kHalf, "grad_output");
    check_cuda_tensor(attn_weights, c10::kFloat, "attn_weights");
    check_cuda_tensor(Q, c10::kHalf, "Q");
    check_cuda_tensor(K, c10::kHalf, "K");
    check_cuda_tensor(V, c10::kHalf, "V");
    check_cuda_tensor(grad_Q, c10::kFloat, "grad_Q");
    check_cuda_tensor(grad_K, c10::kFloat, "grad_K");
    check_cuda_tensor(grad_V, c10::kFloat, "grad_V");

    TORCH_CHECK(D <= 1024, "head_dim D must be <= 1024");
    dim3 grid(B, H, T);
    dim3 block(D);
    ::mha_backward_mp<<<grid, block, 0, stream>>>(
        reinterpret_cast<const __half*>(Q.data_ptr<at::Half>()),
        reinterpret_cast<const __half*>(K.data_ptr<at::Half>()),
        reinterpret_cast<const __half*>(V.data_ptr<at::Half>()),
        attn_weights.data_ptr<float>(),
        reinterpret_cast<const __half*>(grad_output.data_ptr<at::Half>()),
        grad_Q.data_ptr<float>(),
        grad_K.data_ptr<float>(),
        grad_V.data_ptr<float>(),
        theta, B, T, H, D
    );
}

void linear_forward(
    const torch::Tensor& input,
    const torch::Tensor& weight,
    const c10::optional<torch::Tensor>& bias,
    torch::Tensor& output,
    int M, int K, int N
) {
    c10::cuda::CUDAGuard device_guard(input.device());
    auto stream = at::cuda::getCurrentCUDAStream();

    check_cuda_tensor(input, c10::kHalf, "input");
    check_cuda_tensor(weight, c10::kHalf, "weight");
    check_cuda_tensor(output, c10::kHalf, "output");

    const __half* bias_ptr = nullptr;
    if (bias.has_value() && bias.value().defined()) {
        check_cuda_tensor(bias.value(), c10::kHalf, "bias");
        bias_ptr = reinterpret_cast<const __half*>(bias.value().data_ptr<at::Half>());
    }

    dim3 grid(M, N);
    dim3 block(std::min(256, K));
    ::linear_forward_mp<<<grid, block, 0, stream>>>(
        reinterpret_cast<const __half*>(input.data_ptr<at::Half>()),
        reinterpret_cast<const __half*>(weight.data_ptr<at::Half>()),
        bias_ptr,
        reinterpret_cast<__half*>(output.data_ptr<at::Half>()),
        M, K, N
    );
}

void linear_backward(
    const torch::Tensor& grad_output,
    const torch::Tensor& input,
    torch::Tensor& dL_dW,
    torch::Tensor& dL_db,
    int M, int K, int N
) {
    c10::cuda::CUDAGuard device_guard(input.device());
    auto stream = at::cuda::getCurrentCUDAStream();

    check_cuda_tensor(grad_output, c10::kHalf, "grad_output");
    check_cuda_tensor(input, c10::kHalf, "input");
    check_cuda_tensor(dL_dW, c10::kFloat, "dL_dW");
    check_cuda_tensor(dL_db, c10::kFloat, "dL_db");

    int blocks = std::max(1, (K * N + 255) / 256);
    ::linear_backward_mp<<<blocks, 256, 0, stream>>>(
        reinterpret_cast<const __half*>(grad_output.data_ptr<at::Half>()),
        reinterpret_cast<const __half*>(input.data_ptr<at::Half>()),
        dL_dW.data_ptr<float>(),
        dL_db.data_ptr<float>(),
        M, K, N
    );
}

void quantile_heads_forward(
    const torch::Tensor& input,
    const torch::Tensor& weight,
    const c10::optional<torch::Tensor>& bias,
    torch::Tensor& output,
    int B, int T, int D, int Q
) {
    c10::cuda::CUDAGuard device_guard(input.device());
    auto stream = at::cuda::getCurrentCUDAStream();

    bool two_dim = (input.dim() == 2);
    if (two_dim) {
        TORCH_CHECK(T == 1, "For 2D input, T must be 1");
    }

    torch::Tensor in32 = input.scalar_type() == c10::kFloat ? input.contiguous() : input.to(c10::kFloat).contiguous();
    torch::Tensor w32 = weight.scalar_type() == c10::kFloat ? weight.contiguous() : weight.to(c10::kFloat).contiguous();
    const float* bptr = nullptr;
    torch::Tensor b32;
    if (bias.has_value() && bias.value().defined()) {
        b32 = bias.value().scalar_type() == c10::kFloat ? bias.value().contiguous() : bias.value().to(c10::kFloat).contiguous();
        bptr = b32.data_ptr<float>();
    }

    bool need_convert_out = output.scalar_type() != c10::kFloat;
    torch::Tensor out32 = need_convert_out ? torch::empty_like(output, output.options().dtype(torch::kFloat))
                                           : output;
    check_cuda_tensor(out32, c10::kFloat, "output32");

    dim3 grid(B, T);
    dim3 block(Q);

    ::quantile_heads<<<grid, block, 0, stream>>>(
        in32.data_ptr<float>(), w32.data_ptr<float>(), bptr,
        out32.data_ptr<float>(), /*quantiles*/ nullptr, B, T, D, Q
    );

    if (need_convert_out) {
        output.copy_(out32.to(output.scalar_type()));
    }
}

void quantile_loss(
    const torch::Tensor& predictions,
    const torch::Tensor& targets,
    const torch::Tensor& quantiles,
    torch::Tensor& losses,
    int B, int T, int Q
) {
    c10::cuda::CUDAGuard guard(predictions.device());
    auto stream = at::cuda::getCurrentCUDAStream();

    torch::Tensor pred32 = predictions.scalar_type() == c10::kFloat ? predictions.contiguous() : predictions.to(c10::kFloat).contiguous();
    torch::Tensor targ32 = targets.scalar_type() == c10::kFloat ? targets.contiguous() : targets.to(c10::kFloat).contiguous();
    torch::Tensor q32 = quantiles.scalar_type() == c10::kFloat ? quantiles.contiguous() : quantiles.to(c10::kFloat).contiguous();
    check_cuda_tensor(losses, c10::kFloat, "losses");

    dim3 grid(B, T);
    dim3 block(Q);
    ::quantile_loss<<<grid, block, 0, stream>>>(
        pred32.data_ptr<float>(), targ32.data_ptr<float>(), q32.data_ptr<float>(),
        losses.data_ptr<float>(), B, T, Q
    );
}

void quantile_heads_backward(
    const torch::Tensor& grad_output,
    const torch::Tensor& input,
    const torch::Tensor& weight,
    torch::Tensor& dL_dinput,
    torch::Tensor& dL_dW,
    torch::Tensor& dL_db,
    int B, int T, int D, int Q
) {
    c10::cuda::CUDAGuard guard(input.device());
    auto stream = at::cuda::getCurrentCUDAStream();

    torch::Tensor go16 = grad_output.scalar_type() == c10::kHalf ? grad_output.contiguous() : grad_output.to(c10::kHalf).contiguous();
    torch::Tensor in16 = input.scalar_type() == c10::kHalf ? input.contiguous() : input.to(c10::kHalf).contiguous();

    int M = B * T;
    torch::Tensor go2d = go16.view({M, Q}).contiguous();
    torch::Tensor in2d = in16.view({M, D}).contiguous();

    check_cuda_tensor(dL_dW, c10::kFloat, "dL_dW");
    check_cuda_tensor(dL_db, c10::kFloat, "dL_db");
    check_cuda_tensor(dL_dinput, c10::kFloat, "dL_dinput");

    int blocks = std::max(1, (D * Q + 255) / 256);
    ::linear_backward_mp<<<blocks, 256, 0, stream>>>(
        reinterpret_cast<const __half*>(go2d.data_ptr<at::Half>()),
        reinterpret_cast<const __half*>(in2d.data_ptr<at::Half>()),
        dL_dW.data_ptr<float>(),
        dL_db.data_ptr<float>(),
        M, D, Q
    );

    torch::Tensor go32 = grad_output.scalar_type() == c10::kFloat ? grad_output.view({M, Q}).contiguous()
                                                                 : grad_output.to(c10::kFloat).view({M, Q}).contiguous();
    torch::Tensor w32t = weight.scalar_type() == c10::kFloat ? weight.t().contiguous()
                                                            : weight.to(c10::kFloat).t().contiguous();
    torch::Tensor dx32 = at::matmul(go32, w32t);
    dL_dinput.view({M, D}).copy_(dx32);
}

void static_encoder_forward(
    const torch::Tensor& input,
    const torch::Tensor& W1,
    const torch::Tensor& W2,
    const torch::Tensor& gamma,
    const torch::Tensor& beta,
    torch::Tensor& output,
    int B, int S
) {
    c10::cuda::CUDAGuard guard(input.device());
    auto stream = at::cuda::getCurrentCUDAStream();

    check_cuda_tensor(input, c10::kHalf, "input");
    check_cuda_tensor(W1, c10::kHalf, "W1");
    check_cuda_tensor(W2, c10::kHalf, "W2");
    check_cuda_tensor(gamma, c10::kHalf, "gamma");
    check_cuda_tensor(beta, c10::kHalf, "beta");
    check_cuda_tensor(output, c10::kHalf, "output");

    size_t smem = (static_cast<size_t>(S) * 64 + 64 * 32) * sizeof(float);
    dim3 grid(B);
    dim3 block(256);
    ::static_encoder_forward<<<grid, block, smem, stream>>>(
        reinterpret_cast<const __half*>(input.data_ptr<at::Half>()),
        reinterpret_cast<const __half*>(W1.data_ptr<at::Half>()),
        reinterpret_cast<const __half*>(W2.data_ptr<at::Half>()),
        reinterpret_cast<const __half*>(gamma.data_ptr<at::Half>()),
        reinterpret_cast<const __half*>(beta.data_ptr<at::Half>()),
        reinterpret_cast<__half*>(output.data_ptr<at::Half>()),
        B, S
    );
}

void layer_norm_backward(
    const torch::Tensor& grad_output,
    const torch::Tensor& input,
    const torch::Tensor& gamma,
    const torch::Tensor& mean,
    const torch::Tensor& inv_std,
    torch::Tensor& dL_dgamma,
    torch::Tensor& dL_dbeta,
    torch::Tensor& dL_dinput,
    int B, int D
) {
    c10::cuda::CUDAGuard guard(input.device());
    auto stream = at::cuda::getCurrentCUDAStream();

    check_cuda_tensor(grad_output, c10::kHalf, "grad_output");
    check_cuda_tensor(input, c10::kHalf, "input");
    check_cuda_tensor(gamma, c10::kHalf, "gamma");
    check_cuda_tensor(mean, c10::kFloat, "mean");
    check_cuda_tensor(inv_std, c10::kFloat, "inv_std");
    check_cuda_tensor(dL_dgamma, c10::kFloat, "dL_dgamma");
    check_cuda_tensor(dL_dbeta, c10::kFloat, "dL_dbeta");
    check_cuda_tensor(dL_dinput, c10::kFloat, "dL_dinput");

    dim3 grid(B);
    dim3 block(D);
    ::layer_norm_backward_mp<<<grid, block, 0, stream>>>(
        reinterpret_cast<const __half*>(grad_output.data_ptr<at::Half>()),
        reinterpret_cast<const __half*>(input.data_ptr<at::Half>()),
        reinterpret_cast<const __half*>(gamma.data_ptr<at::Half>()),
        mean.data_ptr<float>(), inv_std.data_ptr<float>(),
        dL_dgamma.data_ptr<float>(), dL_dbeta.data_ptr<float>(), dL_dinput.data_ptr<float>(),
        B, D
    );
}

void attention_aggregate(
    const torch::Tensor& attn_weights,
    torch::Tensor& temporal_importance,
    int B, int T, int H
) {
    c10::cuda::CUDAGuard guard(attn_weights.device());
    auto stream = at::cuda::getCurrentCUDAStream();
    check_cuda_tensor(attn_weights, c10::kFloat, "attn_weights");
    check_cuda_tensor(temporal_importance, c10::kFloat, "temporal_importance");

    dim3 grid(B);
    dim3 block(T);
    ::attention_aggregate<<<grid, block, 0, stream>>>(
        attn_weights.data_ptr<float>(), temporal_importance.data_ptr<float>(), B, T, H
    );
}

void vsn_aggregate(
    const torch::Tensor& selection_gates,
    torch::Tensor& static_importance,
    const torch::Tensor& V_s,
    int B, int T, int N
) {
    c10::cuda::CUDAGuard guard(selection_gates.device());
    auto stream = at::cuda::getCurrentCUDAStream();
    check_cuda_tensor(selection_gates, c10::kFloat, "selection_gates");
    check_cuda_tensor(static_importance, c10::kFloat, "static_importance");
    check_cuda_tensor(V_s, c10::kFloat, "V_s");

    dim3 grid(B);
    dim3 block(N);
    ::vsn_aggregate<<<grid, block, 0, stream>>>(
        selection_gates.data_ptr<float>(), static_importance.data_ptr<float>(), V_s.data_ptr<float>(), B, T, N
    );
}

void static_embedding_importance(
    const torch::Tensor& grads,
    const torch::Tensor& static_embeddings,
    const torch::Tensor& W1,
    torch::Tensor& importance,
    int B, int S
) {
    c10::cuda::CUDAGuard guard(grads.device());
    auto stream = at::cuda::getCurrentCUDAStream();

    check_cuda_tensor(grads, c10::kFloat, "grads");
    check_cuda_tensor(static_embeddings, c10::kFloat, "static_embeddings");
    check_cuda_tensor(W1, c10::kFloat, "W1");
    check_cuda_tensor(importance, c10::kFloat, "importance");

    dim3 grid(B);
    dim3 block(S);
    ::static_embedding_importance<<<grid, block, 0, stream>>>(
        grads.data_ptr<float>(), static_embeddings.data_ptr<float>(), importance.data_ptr<float>(), W1.data_ptr<float>(), B, S
    );
}

void lstm_variable_selection_forward(
    const torch::Tensor& x,
    const torch::Tensor& h0,
    const torch::Tensor& c0,
    const c10::optional<torch::Tensor>& weight_ih,
    const c10::optional<torch::Tensor>& weight_hh,
    const c10::optional<torch::Tensor>& bias_ih,
    const c10::optional<torch::Tensor>& bias_hh,
    torch::Tensor& output,
    torch::Tensor& hn,
    torch::Tensor& cn,
    int B, int T, int N
) {
    TORCH_CHECK(false, "lstm_variable_selection_forward not yet implemented; using PyTorch fallback");
}

void lstm_variable_selection_backward(
    const torch::Tensor& grad_output,
    const torch::Tensor& grad_hn,
    const torch::Tensor& grad_cn,
    const torch::Tensor& x,
    const torch::Tensor& hidden_states,
    const torch::Tensor& cell_states,
    const torch::Tensor& weight_ih,
    const torch::Tensor& weight_hh,
    torch::Tensor& grad_x,
    torch::Tensor& grad_h0,
    torch::Tensor& grad_c0,
    torch::Tensor& grad_weight_ih,
    torch::Tensor& grad_weight_hh,
    c10::optional<torch::Tensor>& grad_bias_ih,
    c10::optional<torch::Tensor>& grad_bias_hh,
    int B, int T, int N
) {
    TORCH_CHECK(false, "lstm_variable_selection_backward not yet implemented; using PyTorch fallback");
}

} // namespace tft_cuda
