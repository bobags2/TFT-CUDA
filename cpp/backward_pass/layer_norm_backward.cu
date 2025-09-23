#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cooperative_groups.h>
#include <cmath>

__global__ void layer_norm_backward_mp(
    const __half* dL_doutput,  // (B, D) - gradient from next layer
    const __half* input,       // (B, D) - pre-LN input
    const __half* gamma,       // (D)
    const float* mean,         // (B) - precomputed
    const float* inv_std,      // (B) - precomputed (1/sqrt(var + eps))
    float* dL_dgamma,          // (D) - output
    float* dL_dbeta,           // (D) - output
    float* dL_dinput,          // (B, D) - output
    int B, int D
) {
    namespace cg = cooperative_groups;
    cg::thread_block_tile<32> warp = cg::tiled_partition<32>(cg::this_thread_block());

    int b = blockIdx.x;
    int d = threadIdx.x;
    if (b >= B || d >= D) return;

    float mean_b = mean[b];
    float inv_std_b = inv_std[b];
    float dL_dout = __half2float(dL_doutput[b * D + d]);
    float x = __half2float(input[b * D + d]);
    float normed = (x - mean_b) * inv_std_b;

    // dL/dgamma, dL/dbeta
    atomicAdd(&dL_dgamma[d], dL_dout * normed);
    atomicAdd(&dL_dbeta[d], dL_dout);

    // dL/dinput = dL/dnormed * dnormed/dx + dL/dmean + dL/dvar
    float dL_dnormed = dL_dout * __half2float(gamma[d]);
    float dL_dx = dL_dnormed * inv_std_b;

    // dL/dmean = -dL/dnormed * inv_std - 2 * dL/dvar * mean
    float dL_dmean = -dL_dnormed * inv_std_b;
    float dL_dvar = -dL_dnormed * normed * 0.5f * (inv_std_b * inv_std_b * inv_std_b);

    // Manual warp reduction for dL/dmean and dL/dvar
    for (int offset = 16; offset > 0; offset /= 2) {
        dL_dmean += __shfl_down_sync(0xffffffff, dL_dmean, offset);
        dL_dvar += __shfl_down_sync(0xffffffff, dL_dvar, offset);
    }

    // Final gradient: dL/dx = dL/dnormed * inv_std + dL/dvar * 2 * (x - mean)/D + dL/dmean / D
    float dx = dL_dnormed * inv_std_b;
    dx += dL_dvar * 2 * (x - mean_b) / D;
    dx += dL_dmean / D;

    dL_dinput[b * D + d] = dx;
}