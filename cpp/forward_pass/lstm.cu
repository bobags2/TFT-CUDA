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
    if (bid >= B) return;

    // Initialize hidden/cell states
    float h[N] = {0}, c[N] = {0};

    for (int t = 0; t < T; t++) {
        // Variable selection: softmax over features
        float selection[N] = {0};
        for (int i = 0; i < N; i++) {
            float sum = 0;
            for (int j = 0; j < N; j++)
                sum += x[bid * T * N + t * N + j] * V_s[i * N + j];
            selection[i] = expf(sum);
        }
        float sum_exp = 0;
        for (int i = 0; i < N; i++) sum_exp += selection[i];
        for (int i = 0; i < N; i++) selection[i] /= sum_exp;
        for (int i = 0; i < N; i++) selection_gates[bid * T * N + t * N + i] = selection[i];

        // LSTM gates (input, forget, candidate, output)
        float gates[4 * N] = {0};
        for (int i = 0; i < N; i++) {
            for (int j = 0; j < 4 * N; j++) {
                gates[j] += selection[i] * x[bid * T * N + t * N + i] * W_i[i * 4 * N + j];
                gates[j] += h[i] * W_h[i * 4 * N + j];
            }
            gates[i] += b[i]; // input gate
            gates[i + N] += b[i + N]; // forget gate
            gates[i + 2 * N] += b[i + 2 * N]; // candidate
            gates[i + 3 * N] += b[i + 3 * N]; // output gate
        }

        // Activation (tanh for candidate, sigmoid for others)
        for (int i = 0; i < 2 * N; i++) gates[i] = 1.0f / (1.0f + expf(-gates[i]));
        for (int i = 2 * N; i < 3 * N; i++) gates[i] = tanhf(gates[i]);
        for (int i = 3 * N; i < 4 * N; i++) gates[i] = 1.0f / (1.0f + expf(-gates[i]));

        // Update cell/hidden states
        for (int i = 0; i < N; i++) {
            c[i] = gates[i + N] * c[i] + gates[i] * gates[i + 2 * N];
            h[i] = gates[i + 3 * N] * tanhf(c[i]);
            h_out[bid * T * N + t * N + i] = h[i];
            c_out[bid * T * N + t * N + i] = c[i];
        }
    }
}