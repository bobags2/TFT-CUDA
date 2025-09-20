#include <cuda_fp16.h>
#include <cooperative_groups.h>
#include <cuda_runtime.h>

#define MAX_T 128
#define MAX_D 64
#define WARP_SIZE 32

__global__ void mha_backward_mp(
    // Forward pass (FP16)
    const __half* Q, const __half* K, const __half* V,
    const float* attn_weights,  // [B, T, H, T] - from forward (FP32)
    // Gradients (FP32)
    const __half* dL_doutput,   // [B, T, H, D]
    float* dL_dQ, float* dL_dK, float* dL_dV,
    // RoPE params
    const float theta,
    // Dimensions
    const int B, const int T, const int H, const int D
) {
    namespace cg = cooperative_groups;
    cg::thread_block block = cg::this_thread_block();
    cg::thread_block_tile<32> warp = cg::tiled_partition<32>(block);

    int b = blockIdx.x;  // Batch
    int h = blockIdx.y;  // Head
    int t = blockIdx.z;  // Query time
    int d = threadIdx.x; // Dimension
    if (b >= B || h >= H || t >= T || d >= D) return;

    int base_idx = b * T * H * D + t * H * D + h * D + d;
    float dL_dout = __half2float(dL_doutput[base_idx]);

    // RoPE: Recompute rotated Q/K for this (t,d)
    float inv_freq = __half2float(__float2half(powf(theta, -(float)d / D)));
    float pos_enc = inv_freq * t;
    float cos_p = cosf(pos_enc), sin_p = sinf(pos_enc);

    float q_rot = __half2float(Q[base_idx]) * cos_p +
                  __half2float(Q[base_idx + ((d + D/2) % D)]) * sin_p;
    float k_rot = __half2float(K[base_idx]) * cos_p +
                  __half2float(K[base_idx + ((d + D/2) % D)]) * sin_p;

    // Shared memory for attention weights and gradients
    __shared__ float s_attn[MAX_T];        // [T]
    __shared__ float s_dL_dV[MAX_T];       // [T] - dL/dV for this (h,d)

    // Load attention weights for query t
    if (d == 0) {
        for (int t2 = 0; t2 <= t; t2++) {
            s_attn[t2] = attn_weights[b * T * H * T + t * H * T + h * T + t2];
        }
        for (int t2 = t + 1; t2 < T; t2++) s_attn[t2] = 0.0f; // causal
    }
    block.sync();

    // === dL/dV: dL/dout @ dout/dV = dL/dout * attn_weights ===
    float dL_dV_td = 0.0f;
    for (int t2 = 0; t2 <= t; t2++) {
        float attn_t2 = s_attn[t2];
        int v_idx = b * T * H * D + t2 * H * D + h * D + d;
        dL_dV_td += dL_dout * attn_t2;
    }
    s_dL_dV[t] = dL_dV_td;
    atomicAdd(&dL_dV[base_idx], dL_dV_td);

    // === dL/dA: dL/dout @ dout/dA = dL/dout @ V ===
    float dL_dA_t = 0.0f;
    for (int t2 = 0; t2 <= t; t2++) {
        int v_idx = b * T * H * D + t2 * H * D + h * D + d;
        dL_dA_t += dL_dout * __half2float(V[v_idx]);
    }
    dL_dA_t *= rsqrtf(D);  // from scale

    // === dL/dK, dL/dQ via dA/dK, dA/dQ (softmax derivative) ===
    float dL_dK_td = 0.0f, dL_dQ_td = 0.0f;

    for (int t2 = 0; t2 <= t; t2++) {
        float attn_t2 = s_attn[t2];
        float dA_dK_t2 = attn_t2 * (q_rot - attn_t2 * k_rot);  // dA/dk = A * (q - A * k)
        float dA_dQ_t2 = attn_t2 * (k_rot - attn_t2 * q_rot);  // dA/dq = A * (k - A * q)

        // RoPE-aware: dA/dK affects K[t2], dA/dQ affects Q[t2]
        float dL_dK_val = dL_dA_t * dA_dK_t2;
        float dL_dQ_val = dL_dA_t * dA_dQ_t2;

        // Accumulate dL/dK for K[t2][h][d]
        if (t2 == t) {
            // RoPE: dL/dK affects both K[t][d] and K[t][d+D/2]
            float dK_d = dL_dK_val * cos_p;
            float dK_d2 = dL_dK_val * sin_p;
            atomicAdd(&dL_dK[b * T * H * D + t * H * D + h * D + d], dK_d);
            atomicAdd(&dL_dK[b * T * H * D + t * H * D + h * D + ((d + D/2) % D)], dK_d2);
        }

        // Accumulate dL/dQ for Q[t2][h][d]
        if (t2 == t) {
            float dQ_d = dL_dQ_val * cos_p;
            float dQ_d2 = dL_dQ_val * sin_p;
            atomicAdd(&dL_dQ[b * T * H * D + t * H * D + h * D + d], dQ_d);
            atomicAdd(&dL_dQ[b * T * H * D + t * H * D + h * D + ((d + D/2) % D)], dQ_d2);
        }
    }

    // === dL/dQ for other t2 (cross-time) ===
    for (int t2 = 0; t2 <= t; t2++) {
        if (t2 == t) continue;
        float attn_t2 = s_attn[t2];
        float dA_dQ_t2 = attn_t2 * (k_rot - attn_t2 * q_rot);
        float dL_dQ_val = dL_dA_t * dA_dQ_t2;

        int q2_idx = b * T * H * D + t2 * H * D + h * D + d;
        int q2_idx2 = b * T * H * D + t2 * H * D + h * D + ((d + D/2) % D);

        float dQ_d = dL_dQ_val * cos_p;
        float dQ_d2 = dL_dQ_val * sin_p;
        atomicAdd(&dL_dQ[q2_idx], dQ_d);
        atomicAdd(&dL_dQ[q2_idx2], dQ_d2);
    }

    // === dL/dK for other t2 (cross-time) ===
    for (int t2 = 0; t2 < T; t2++) {
        if (t2 > t) continue; // causal
        float attn_t2 = s_attn[t2];
        float dA_dK_t2 = attn_t2 * (q_rot - attn_t2 * k_rot);
        float dL_dK_val = dL_dA_t * dA_dK_t2;

        int k2_idx = b * T * H * D + t2 * H * D + h * D + d;
        int k2_idx2 = b * T * H * D + t2 * H * D + h * D + ((d + D/2) % D);

        float dK_d = dL_dK_val * cos_p;
        float dK_d2 = dL_dK_val * sin_p;
        atomicAdd(&dL_dK[k2_idx], dK_d);
        atomicAdd(&dL_dK[k2_idx2], dK_d2);
    }

    // === dL/dV for other t2 (already done in s_dL_dV) ===
    for (int t2 = 0; t2 <= t; t2++) {
        if (t2 == t) continue;
        atomicAdd(&dL_dV[b * T * H * D + t2 * H * D + h * D + d], s_dL_dV[t2]);
    }
}