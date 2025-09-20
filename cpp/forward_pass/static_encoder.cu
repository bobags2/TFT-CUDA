__global__ void static_encoder_mp(
    const __half* input,      // (B, S)
    const __half* W1,         // (S, 64)
    const __half* W2,         // (64, 32)
    const __half* gamma,      // (32)
    const __half* beta,       // (32)
    __half* output,           // (B, 32)
    const int B, const int S
) {
    namespace cg = cooperative_groups;
    cg::thread_block block = cg::this_thread_block();
    cg::thread_block_tile<32> warp = cg::tiled_partition<32>(block);

    extern __shared__ float s[];
    float* s_W1 = s;           // (S, 64)
    float* s_W2 = s + S * 64;  // (64, 32)

    // Load weights (FP16 → FP32 for precision)
    for (int i = threadIdx.x; i < S * 64; i += blockDim.x)
        s_W1[i] = __half2float(W1[i]);
    for (int i = threadIdx.x; i < 64 * 32; i += blockDim.x)
        s_W2[i] = __half2float(W2[i]);
    block.sync();

    int bid = blockIdx.x;
    int tid = threadIdx.x;
    if (bid >= B) return;

    // MLP Layer 1: S → 64
    float hidden[64] = {0};
    for (int i = 0; i < S; i++) {
        float x_i = __half2float(input[bid * S + i]);
        for (int j = tid; j < 64; j += blockDim.x)
            hidden[j] += x_i * s_W1[i * 64 + j];
    }
    block.sync();

    // === LayerNorm (64) - WARP-LEVEL REDUCTION ===
    float mean = 0.0f, var = 0.0f;
    for (int i = 0; i < 64; i++) mean += hidden[i];
    mean = warp.reduce(mean, [](float a, float b) { return a + b; }) / 64.0f;

    for (int i = 0; i < 64; i++) var += (hidden[i] - mean) * (hidden[i] - mean);
    var = warp.reduce(var, [](float a, float b) { return a + b; }) / 64.0f;
    float inv_std = rsqrtf(var + 1e-8f);

    for (int i = tid; i < 64; i += blockDim.x)
        hidden[i] = (hidden[i] - mean) * inv_std;

    block.sync();

    // MLP Layer 2: 64 → 32
    float out[32] = {0};
    for (int i = 0; i < 64; i++) {
        float h_i = hidden[i];
        for (int j = tid; j < 32; j += blockDim.x)
            out[j] += h_i * s_W2[i * 32 + j];
    }
    block.sync();

    // === LayerNorm (32) - WARP-LEVEL REDUCTION ===
    mean = 0.0f; var = 0.0f;
    for (int i = 0; i < 32; i++) mean += out[i];
    mean = warp.reduce(mean, [](float a, float b) { return a + b; }) / 32.0f;

    for (int i = 0; i < 32; i++) var += (out[i] - mean) * (out[i] - mean);
    var = warp.reduce(var, [](float a, float b) { return a + b; }) / 32.0f;
    inv_std = rsqrtf(var + 1e-8f);

    for (int i = tid; i < 32; i += blockDim.x) {
        float normed = (out[i] - mean) * inv_std;
        output[bid * 32 + i] = __float2half(
            __half2float(gamma[i]) * normed + __half2float(beta[i])
        );
    }
}