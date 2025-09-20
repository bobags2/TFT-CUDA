// Aggregate attention weights across heads (for temporal interpretability)
__global__ void attention_aggregate(
    const float* attn_weights,     // (B, T, H, T)
    float* temporal_importance,    // (B, T) - importance of each time step
    const int B, const int T, const int H
) {
    int bid = blockIdx.x;
    int tid = threadIdx.x;
    if (bid >= B || tid >= T) return;

    // Sum attention weights across heads and future time steps
    float sum = 0;
    for (int h = 0; h < H; h++) {
        for (int t = 0; t <= tid; t++) {
            sum += attn_weights[bid * T * H * T + tid * H * T + h * T + t];
        }
    }
    temporal_importance[bid * T + tid] = sum / (H * (tid + 1)); // Normalize
}