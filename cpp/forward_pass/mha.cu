__global__ void multi_head_attention_mp(
    const __half* Q,          // (B, T, H, D)
    const __half* K,          // (B, T, H, D)
    const __half* V,          // (B, T, H, D)
    __half* output,           // (B, T, H, D)
    float* attn_weights,      // (B, T, H, T) - FP32 for gradients
    const float theta,
    const int B, const int T, const int H, const int D
) {
    namespace cg = cooperative_groups;
    cg::thread_block block = cg::this_thread_block();
    cg::thread_block_tile<32> warp = cg::tiled_partition<32>(block);

    int b = blockIdx.x;  // Batch
    int t = blockIdx.y;  // Time (query)
    int h = blockIdx.z;  // Head
    int d = threadIdx.x; // Dimension
    if (b >= B || t >= T || h >= H || d >= D) return;

    int base_idx = b * T * H * D + t * H * D + h * D + d;
    int attn_base = b * T * H * T + t * H * T + h * T;

    // === RoPE: Apply rotation (FP16 → FP32) ===
    float inv_freq = powf(theta, -(float)d / D);
    float pos_enc = inv_freq * t;
    float cos_p = cosf(pos_enc), sin_p = sinf(pos_enc);

    float q_rot = __half2float(Q[base_idx]) * cos_p +
                  __half2float(Q[base_idx + ((d + D/2) % D)]) * sin_p;
    float k_rot = __half2float(K[base_idx]) * cos_p +
                  __half2float(K[base_idx + ((d + D/2) % D)]) * sin_p;

    // === Shared memory for attention scores (T) ===
    __shared__ float s_scores[MAX_T];
    __shared__ float s_max[MAX_T / WARP_SIZE];  // For warp reduction

    // Compute scores: Q @ K.T (causal)
    float score = q_rot * k_rot * rsqrtf(D);
    s_scores[t] = score;  // Only this thread writes to s_scores[t]
    block.sync();

    // === Warp-level reduction for max (for numerical stability) ===
    float max_val = score;
    for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
        float tmp = __shfl_down_sync(0xffffffff, max_val, offset);
        max_val = fmaxf(max_val, tmp);
    }
    if (d % WARP_SIZE == 0) s_max[d / WARP_SIZE] = max_val;
    block.sync();
    if (d == 0) {
        max_val = s_max[0];
        for (int i = 1; i < (T + WARP_SIZE - 1) / WARP_SIZE; i++)
            max_val = fmaxf(max_val, s_max[i]);
        s_max[0] = max_val;
    }
    block.sync();
    max_val = s_max[0];

    // === Softmax: exp(score - max) / sum ===
    float exp_score = expf(score - max_val);
    s_scores[t] = exp_score;
    block.sync();

    // Warp-level sum
    float sum = exp_score;
    for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
        sum += __shfl_down_sync(0xffffffff, sum, offset);
    }
    if (d % WARP_SIZE == 0) s_max[d / WARP_SIZE] = sum;
    block.sync();
    if (d == 0) {
        sum = s_max[0];
        for (int i = 1; i < (T + WARP_SIZE - 1) / WARP_SIZE; i++)
            sum += s_max[i];
        s_max[0] = sum;
    }
    block.sync();
    sum = s_max[0];

    float attn = exp_score / sum;
    if (d == 0) attn_weights[attn_base + t] = attn;  // Store FP32 for backward

    // === Weighted sum: A @ V ===
    float out_val = 0.0f;
    for (int t2 = 0; t2 <= t; t2++) {  // Causal
        float attn_t2 = (t2 == t) ? attn : 0.0f;  // Only this thread knows its own attn
        // But we need all attn weights — load from shared or recompute
        // Instead: use shared s_scores (only valid for t2 <= t)
        if (t2 < t) {
            // Load from shared (only d=0 thread has it)
            if (d == 0) attn_t2 = s_scores[t2] / sum;
            attn_t2 = __shfl_sync(0xffffffff, attn_t2, d % WARP_SIZE);
        }
        int v_idx = b * T * H * D + t2 * H * D + h * D + d;
        out_val += attn_t2 * __half2float(V[v_idx]);
    }

    output[base_idx] = __float2half(out_val);
}