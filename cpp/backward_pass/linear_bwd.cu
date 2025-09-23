#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cublas_v2.h>

__global__ void linear_backward_mp(
    const __half* dL_doutput,  // (M, N)
    const __half* input,       // (M, K)
    float* dL_dW,              // (K, N) - output (FP32)
    float* dL_db,              // (N) - output (FP32)
    const int M, const int K, const int N
) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;

    // === Compute dL/db = sum(dL_doutput, dim=0) ===
    for (int n = threadIdx.x; n < N; n += blockDim.x) {
        float sum = 0.0f;
        for (int m = 0; m < M; m++) {
            sum += __half2float(dL_doutput[m * N + n]);
        }
        dL_db[n] = sum;
    }
    __syncthreads();

    // === Compute dL/dW = input.T @ dL_doutput ===
    // Each thread computes one element of dL_dW
    int total_elements = K * N;
    for (int idx = tid; idx < total_elements; idx += blockDim.x * gridDim.x) {
        int k = idx / N;  // row index
        int n = idx % N;  // column index
        
        float grad_sum = 0.0f;
        for (int m = 0; m < M; m++) {
            grad_sum += __half2float(input[m * K + k]) * __half2float(dL_doutput[m * N + n]);
        }
        dL_dW[k * N + n] = grad_sum;
    }
}