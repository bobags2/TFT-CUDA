#include "cutlass/cutlass.h"
#include "cutlass/gemm/device/gemm.h"

__global__ void linear_backward_mp(
    const __half* dL_doutput,  // (M, N)
    const __half* input,       // (M, K)
    float* dL_dW,              // (N, K) - output (FP32)
    float* dL_db,              // (N) - output (FP32)
    const int M, const int K, const int N
) {
    using namespace cutlass;

    // === dL/db = sum(dL/doutput, dim=0) ===
    for (int n = threadIdx.x; n < N; n += blockDim.x) {
        float sum = 0.0f;
        for (int m = 0; m < M; m++) {
            sum += __half2float(dL_doutput[m * N + n]);
        }
        dL_db[n] = sum;
    }
    __syncthreads();

    // === dL/dW = dL/doutput.T @ input ===
    // We use CUTLASS GEMM: (N, M) @ (M, K) = (N, K)
    using Gemm = gemm::device::Gemm<
        __half, layout::RowMajor,  // dL_doutput.T (N, M)
        __half, layout::RowMajor,  // input (M, K)
        float, layout::RowMajor,   // dL_dW (N, K)
        float,
        arch::OpClassTensorOp,
        arch::Sm80,
        gemm::GemmShape<64, 64, 32>,
        gemm::GemmShape<32, 32, 32>,
        gemm::GemmShape<16, 8, 16>
    >;

    typename Gemm::Arguments args(
        {N, K, M},                    // problem size
        {dL_doutput, M},              // dL_doutput.T (lda = M)
        {input, K},                   // input (ldb = K)
        {dL_dW, K},                   // dL_dW (ldc = K)
        {1.0f, 0.0f},
        1                             // batch count
    );

    Gemm gemm_op;
    cutlass::Status status = gemm_op(args);
    assert(status == cutlass::Status::kSuccess);
}