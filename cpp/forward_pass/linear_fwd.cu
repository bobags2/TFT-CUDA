__global__ void linear_forward_mp(
    const __half* input,     // (M, K)
    const __half* weight,    // (K, N)
    const __half* bias,      // (N)
    __half* output,          // (M, N)
    const int M, const int K, const int N
) {
    int m = blockIdx.x;
    int n = blockIdx.y;
    int tid = threadIdx.x;
    if (m >= M || n >= N) return;

    float sum = (bias != nullptr) ? __half2float(bias[n]) : 0.0f;

    // Coalesced load: each thread computes partial dot product
    for (int k = tid; k < K; k += blockDim.x) {
        sum += __half2float(input[m * K + k]) * __half2float(weight[k * N + n]);
    }

    // Warp-level reduction
    for (int offset = 16; offset > 0; offset /= 2)
        sum += __shfl_down_sync(0xffffffff, sum, offset);

    if (tid == 0) {
        output[m * N + n] = __float2half(sum);
    }
}