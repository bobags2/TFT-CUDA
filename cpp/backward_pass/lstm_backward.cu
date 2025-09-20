#include <cuda_fp16.h>
#include <cooperative_groups.h>
#include <cuda_runtime.h>

#define MAX_N 512  // Max features (adjust as needed)
#define TPB 256    // Threads per block (T=10, B=16 → 160 threads)

__global__ void lstm_variable_selection_backward_mp(
    // Forward pass inputs (FP16)
    const __half* x,           // [B, T, N] - input
    const __half* W_i,         // [N, 4*N] - input weights
    const __half* W_h,         // [N, 4*N] - hidden weights
    const __half* V_s,         // [N, N] - variable selection weights
    const __half* b,           // [4*N] - biases

    // Forward pass outputs (FP16, for gradient computation)
    const __half* h_out,       // [B, T, N]
    const __half* c_out,       // [B, T, N]
    const __half* selection_gates, // [B, T, N] - softmax output

    // Gradient from next layer (FP32)
    const float* dL_dh_next,   // [B, T, N] - dL/dh at t+1
    const float* dL_dc_next,   // [B, T, N] - dL/dc at t+1

    // Output gradients (FP32, accumulated)
    float* dL_dW_i,            // [N, 4*N] - dL/dW_i (accumulated)
    float* dL_dW_h,            // [N, 4*N] - dL/dW_h
    float* dL_dV_s,            // [N, N]   - dL/dV_s
    float* dL_db,              // [4*N]    - dL/db
    float* dL_dx,              // [B, T, N] - dL/dx (input gradient)
    float* dL_dh_prev,         // [B, N]   - dL/dh at t=-1 (output)
    float* dL_dc_prev,         // [B, N]   - dL/dc at t=-1

    // Dimensions
    const int B, const int T, const int N
) {
    namespace cg = cooperative_groups;
    cg::thread_block block = cg::this_thread_block();
    cg::thread_block_tile<32> warp = cg::tiled_partition<32>(block);

    int b = blockIdx.x;  // Batch index
    int tid = threadIdx.x;  // Thread ID (0..TPB-1)
    if (b >= B) return;

    // Shared memory for h, c, gates, selection (FP32 for precision)
    __shared__ float s_h[MAX_N], s_c[MAX_N], s_selection[MAX_N];
    __shared__ float s_gates[4 * MAX_N];  // [i, f, c_tilde, o]

    // Local accumulators (FP32)
    float dh[MAX_N] = {0}, dc[MAX_N] = {0};
    float dW_i[MAX_N * 4] = {0}, dW_h[MAX_N * 4] = {0}, dV_s[MAX_N * MAX_N] = {0}, db[4 * MAX_N] = {0};
    float dx_local[MAX_N] = {0};

    // Load initial gradients from next time step
    for (int i = tid; i < N; i += blockDim.x) {
        dh[i] = dL_dh_next[b * T * N + (T-1) * N + i];
        dc[i] = dL_dc_next[b * T * N + (T-1) * N + i];
    }
    block.sync();

    // Backward loop: from T-1 to 0
    for (int t = T - 1; t >= 0; t--) {
        int t_idx = b * T * N + t * N;
        int t_gates_idx = b * T * N + t * N;

        // Load selection gates (FP16 → FP32)
        for (int i = tid; i < N; i += blockDim.x) {
            s_selection[i] = __half2float(selection_gates[t_gates_idx + i]);
            s_h[i] = __half2float(h_out[t_idx + i]);
            s_c[i] = __half2float(c_out[t_idx + i]);
        }
        block.sync();

        // Recompute gates (FP16 → FP32)
        for (int i = tid; i < 4 * N; i += blockDim.x) s_gates[i] = 0.0f;
        for (int j = 0; j < N; j++) {
            float sel_j = s_selection[j];
            for (int i = tid; i < 4 * N; i += blockDim.x) {
                float x_j = __half2float(x[t_idx + j]);
                float h_j = s_h[j];
                s_gates[i] += sel_j * x_j * __half2float(W_i[j * 4 * N + i]);
                s_gates[i] += h_j * __half2float(W_h[j * 4 * N + i]);
                if (i == j || i == j + N || i == j + 2*N || i == j + 3*N) {
                    s_gates[i] += __half2float(b[i]);
                }
            }
        }
        block.sync();

        // Apply activations (FP32)
        for (int i = tid; i < 2 * N; i += blockDim.x) s_gates[i] = 1.0f / (1.0f + expf(-s_gates[i])); // sigmoid
        for (int i = 2 * N; i < 3 * N; i += blockDim.x) s_gates[i] = tanhf(s_gates[i]); // tanh
        for (int i = 3 * N; i < 4 * N; i += blockDim.x) s_gates[i] = 1.0f / (1.0f + expf(-s_gates[i])); // sigmoid
        block.sync();

        // === Backward through LSTM ===
        float dh_prev[MAX_N] = {0}, dc_prev[MAX_N] = {0};
        for (int i = tid; i < N; i += blockDim.x) {
            float i_gate = s_gates[i];
            float f_gate = s_gates[i + N];
            float c_tilde = s_gates[i + 2 * N];
            float o_gate = s_gates[i + 3 * N];
            float c_t = s_c[i];
            float tanh_c_t = tanhf(c_t);

            // dL/do = dL/dh * tanh(c)
            float do_gate = dh[i] * tanh_c_t;
            db[3 * N + i] += do_gate;
            dh[i] -= do_gate * o_gate * (1 - tanh_c_t * tanh_c_t);

            // dL/dc += dL/dh * o * (1 - tanh^2(c))
            dc[i] += dh[i] * o_gate * (1 - tanh_c_t * tanh_c_t);

            // dL/df = dL/dc * c_{t-1} * f'
            float c_prev = (t > 0) ? __half2float(c_out[b * T * N + (t-1) * N + i]) : 0.0f;
            float df_gate = dc[i] * c_prev * f_gate * (1 - f_gate);
            db[N + i] += df_gate;

            // dL/di = dL/dc * c_tilde * i'
            float di_gate = dc[i] * c_tilde * i_gate * (1 - i_gate);
            db[i] += di_gate;

            // dL/dc_tilde = dL/dc * i * tanh'
            float dc_tilde = dc[i] * i_gate * (1 - c_tilde * c_tilde);
            db[2 * N + i] += dc_tilde;

            // dL/dx, dL/dh from gates
            for (int j = 0; j < N; j++) {
                float sel_j = s_selection[j];
                float x_j = __half2float(x[t_idx + j]);
                float h_j = s_h[j];

                // dL/dW_i[j, *] += sel_j * x_j * [di, df, dc_tilde, do]
                atomicAdd(&dW_i[j * 4 + 0], sel_j * x_j * di_gate);
                atomicAdd(&dW_i[j * 4 + 1], sel_j * x_j * df_gate);
                atomicAdd(&dW_i[j * 4 + 2], sel_j * x_j * dc_tilde);
                atomicAdd(&dW_i[j * 4 + 3], sel_j * x_j * do_gate);

                // dL/dW_h[j, *] += h_j * [di, df, dc_tilde, do]
                atomicAdd(&dW_h[j * 4 + 0], h_j * di_gate);
                atomicAdd(&dW_h[j * 4 + 1], h_j * df_gate);
                atomicAdd(&dW_h[j * 4 + 2], h_j * dc_tilde);
                atomicAdd(&dW_h[j * 4 + 3], h_j * do_gate);

                // dL/dx[j] += sel_j * (di * W_i[j,i] + ...)
                float dx_j = sel_j * (di_gate * __half2float(W_i[j * 4 * N + i]) +
                                      df_gate * __half2float(W_i[j * 4 * N + i + N]) +
                                      dc_tilde * __half2float(W_i[j * 4 * N + i + 2*N]) +
                                      do_gate * __half2float(W_i[j * 4 * N + i + 3*N]));
                dx_local[j] += dx_j;

                // dL/dh[j] (from gate gradients)
                dh_prev[j] += di_gate * __half2float(W_h[j * 4 * N + i]) +
                              df_gate * __half2float(W_h[j * 4 * N + i + N]) +
                              dc_tilde * __half2float(W_h[j * 4 * N + i + 2*N]) +
                              do_gate * __half2float(W_h[j * 4 * N + i + 3*N]);
            }

            // dL/dV_s: gradient of selection
            for (int k = 0; k < N; k++) {
                float x_k = __half2float(x[t_idx + k]);
                float dsel = (i == k) ? s_selection[i] * (1 - s_selection[i]) : -s_selection[i] * s_selection[k];
                float dL_dV = 0.0f;
                for (int g = 0; g < 4; g++) {
                    float gate_grad = (g == 0) ? di_gate :
                                      (g == 1) ? df_gate :
                                      (g == 2) ? dc_tilde : do_gate;
                    dL_dV += gate_grad * x_k * __half2float(W_i[k * 4 * N + i + g * N]);
                }
                dV_s[i * N + k] += dL_dV * dsel;
            }

            dc_prev[i] = df_gate * f_gate;
        }
        block.sync();

        // Copy to shared and reduce
        for (int i = tid; i < N; i += blockDim.x) {
            dh[i] = dh_prev[i];
            dc[i] = dc_prev[i];
        }
        block.sync();
    }

    // === Write gradients back (atomic to global) ===
    for (int i = tid; i < N * 4; i += blockDim.x) {
        atomicAdd(&dL_dW_i[i], dW_i[i]);
        atomicAdd(&dL_dW_h[i], dW_h[i]);
        atomicAdd(&dL_db[i], db[i]);
    }
    for (int i = tid; i < N * N; i += blockDim.x) {
        atomicAdd(&dL_dV_s[i], dV_s[i]);
    }
    for (int i = tid; i < N; i += blockDim.x) {
        atomicAdd(&dL_dx[b * T * N + i], dx_local[i]);
        dL_dh_prev[b * N + i] = dh[i];
        dL_dc_prev[b * N + i] = dc[i];
    }
}