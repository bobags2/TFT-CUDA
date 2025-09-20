// Aggregate selection gates across time (for static interpretability)
__global__ void vsn_aggregate(
    const float* selection_gates,  // (B, T, N)
    float* static_importance,      // (B, N)
    const float* V_s,              // (N, N) - variable selection weights
    const int B, const int T, const int N
) {
    int bid = blockIdx.x;
    int nid = threadIdx.x;
    if (bid >= B || nid >= N) return;

    // Sum selection gates across time
    float sum = 0;
    for (int t = 0; t < T; t++) {
        sum += selection_gates[bid * T * N + t * N + nid];
    }
    static_importance[bid * N + nid] = sum / T; // Normalize
}