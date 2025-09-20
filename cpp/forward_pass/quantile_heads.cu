// Quantile heads: (B, T, D) -> (B, T, Q) where Q = num_quantiles (e.g., 3)
__global__ void quantile_heads(
    const float* input,      // (B, T, D)
    const float* W,          // (D, Q) - shared weights
    const float* b,          // (Q) - biases
    float* outputs,          // (B, T, Q) - outputs for all quantiles
    const float* quantiles,  // (Q) - quantile values (e.g., [0.1, 0.5, 0.9])
    const int B, const int T, const int D, const int Q
) {
    int bid = blockIdx.x;
    int tid = blockIdx.y;  // Time step
    int qid = threadIdx.x; // Quantile ID
    if (bid >= B || tid >= T || qid >= Q) return;

    // Compute output for quantile `qid`
    float out = b[qid];
    for (int d = 0; d < D; d++) {
        out += input[bid * T * D + tid * D + d] * W[d * Q + qid];
    }
    outputs[bid * T * Q + tid * Q + qid] = out;
}

// Quantile loss kernel (Pinball Loss)
__global__ void quantile_loss(
    const float* preds,      // (B, T, Q)
    const float* targets,    // (B, T)
    const float* quantiles,  // (Q)
    float* losses,           // (B, T, Q)
    const int B, const int T, const int Q
) {
    int bid = blockIdx.x;
    int tid = blockIdx.y;
    int qid = threadIdx.x;
    if (bid >= B || tid >= T || qid >= Q) return;

    float diff = targets[bid * T + tid] - preds[bid * T * Q + tid * Q + qid];
    float q = quantiles[qid];
    losses[bid * T * Q + tid * Q + qid] = (diff < 0) ? (q - 1) * diff : q * diff;
}