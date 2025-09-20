#include <cuda_fp16.h>
#include <cooperative_groups.h>

__global__ void quantile_heads_backward_mp(
    // Forward inputs (FP16)
    const __half* input,           // (B, T, D)
    const __half* W,               // (D, Q)
    const __half* b,               // (Q)
    const __half* quantiles,       // (Q)
    // Forward outputs (FP16)
    const __half* preds,           // (B, T, Q)
    // Targets (FP16)
    const __half* targets,         // (B, T)
    // Gradients (FP32, accumulated)
    float* dL_dW,                  // (D, Q)
    float* dL_db,                  // (Q)
    float* dL_dinput,              // (B, T, D)
    // Dimensions
    const int B, const int T, const int D, const int Q
) {
    namespace cg = cooperative_groups;
    cg::thread_block block = cg::this_thread_block();
    cg::thread_block_tile<32> warp = cg::tiled_partition<32>(block);

    int b = blockIdx.x;  // Batch
    int t = blockIdx.y;  // Time
    int q = threadIdx.x; // Quantile
    if (b >= B || t >= T || q >= Q) return;

    int pred_idx = b * T * Q + t * Q + q;
    int target_idx = b * T + t;
    int input_base = b * T * D + t * D;

    // Load values (FP16 → FP32)
    float pred = __half2float(preds[pred_idx]);
    float target = __half2float(targets[target_idx]);
    float q_val = __half2float(quantiles[q]);

    // === Pinball Loss Gradient: dL/dpred ===
    float diff = target - pred;
    float dL_dpred = (diff < 0) ? (q_val - 1.0f) : q_val;

    // === dL/db = dL/dpred * dpred/db = dL/dpred * 1 ===
    atomicAdd(&dL_db[q], dL_dpred);

    // === dL/dW = dL/dpred * dpred/dW = dL/dpred * input ===
    for (int d = 0; d < D; d++) {
        float x_d = __half2float(input[input_base + d]);
        atomicAdd(&dL_dW[d * Q + q], dL_dpred * x_d);
    }

    // === dL/dinput = dL/dpred * dpred/dinput = dL/dpred * W ===
    for (int d = 0; d < D; d++) {
        float w_dq = __half2float(W[d * Q + q]);
        float dL_dx = dL_dpred * w_dq;
        atomicAdd(&dL_dinput[input_base + d], dL_dx);
    }
}