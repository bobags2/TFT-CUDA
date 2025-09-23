#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cmath>

#define MAX_FEATURES 512

// LSTM kernel with variable selection (simplified for clarity)
__global__ void lstm_variable_selection(
    const float* x,          // (B, T, N)
    const float* W_i,        // (N, 4*N) - input weights
    const float* W_h,        // (N, 4*N) - hidden weights
    const float* b,          // (4*N) - biases
    const float* V_s,        // (N, N) - variable selection weights
    float* h_out,            // (B, T, N)
    float* c_out,            // (B, T, N)
    float* selection_gates,  // (B, T, N) - interpretability
    const int B, const int T, const int N
) {
    int bid = blockIdx.x;
    int tid = threadIdx.x;
    if (bid >= B || N > MAX_FEATURES) return;

    // Use shared memory for large arrays
    __shared__ float s_h[MAX_FEATURES];
    __shared__ float s_c[MAX_FEATURES];
    __shared__ float s_selection[MAX_FEATURES];
    
    // Initialize hidden/cell states
    if (tid < N) {
        s_h[tid] = 0.0f;
        s_c[tid] = 0.0f;
    }
    __syncthreads();

    for (int t = 0; t < T; t++) {
        // Variable selection: softmax over features
        if (tid < N) {
            float sum = 0;
            for (int j = 0; j < N; j++)
                sum += x[bid * T * N + t * N + j] * V_s[tid * N + j];
            s_selection[tid] = expf(sum);
        }
        __syncthreads();
        
        // Normalize selection gates
        if (tid == 0) {
            float sum_exp = 0;
            for (int i = 0; i < N; i++) sum_exp += s_selection[i];
            for (int i = 0; i < N; i++) s_selection[i] /= sum_exp;
        }
        __syncthreads();
        
        // Store selection gates for interpretability
        if (tid < N) {
            selection_gates[bid * T * N + t * N + tid] = s_selection[tid];
        }

        // LSTM gates computation using shared memory
        __shared__ float s_gates[4 * MAX_FEATURES];
        if (tid < 4 * N) {
            s_gates[tid] = 0.0f;
        }
        __syncthreads();
        
        // Compute gates with proper indexing
        if (tid < N) {
            for (int j = 0; j < 4 * N; j++) {
                atomicAdd(&s_gates[j], s_selection[tid] * x[bid * T * N + t * N + tid] * W_i[tid * 4 * N + j]);
                atomicAdd(&s_gates[j], s_h[tid] * W_h[tid * 4 * N + j]);
            }
        }
        __syncthreads();
        
        // Add biases
        if (tid < 4 * N) {
            s_gates[tid] += b[tid];
        }
        __syncthreads();

        // Activation functions
        if (tid < 2 * N) s_gates[tid] = 1.0f / (1.0f + expf(-s_gates[tid])); // sigmoid for input/forget
        if (tid >= 2 * N && tid < 3 * N) s_gates[tid] = tanhf(s_gates[tid]); // tanh for candidate
        if (tid >= 3 * N && tid < 4 * N) s_gates[tid] = 1.0f / (1.0f + expf(-s_gates[tid])); // sigmoid for output
        __syncthreads();

        // Update cell/hidden states using shared memory
        if (tid < N) {
            float input_gate = s_gates[tid];
            float forget_gate = s_gates[tid + N];
            float candidate_gate = s_gates[tid + 2 * N];
            float output_gate = s_gates[tid + 3 * N];
            
            s_c[tid] = forget_gate * s_c[tid] + input_gate * candidate_gate;
            s_h[tid] = output_gate * tanhf(s_c[tid]);
            
            h_out[bid * T * N + t * N + tid] = s_h[tid];
            c_out[bid * T * N + t * N + tid] = s_c[tid];
        }
        __syncthreads();
    }
}