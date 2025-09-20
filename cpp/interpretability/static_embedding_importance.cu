// Compute static feature importance via gradient w.r.t. embeddings
__global__ void static_embedding_importance(
    const float* grads,            // (B, 32) - gradients from static encoder output
    const float* static_embeddings, // (B, 32)
    float* importance,             // (B, S) - importance of original static features
    const float* W1,               // (S, 64) - static encoder weights
    const int B, const int S
) {
    int bid = blockIdx.x;
    int sid = threadIdx.x;
    if (bid >= B || sid >= S) return;

    // Gradient * embedding * weight (chain rule)
    float sum = 0;
    for (int i = 0; i < 32; i++) {
        sum += grads[bid * 32 + i] * static_embeddings[bid * 32 + i] * W1[sid * 64 + i];
    }
    importance[bid * S + sid] = sum;
}