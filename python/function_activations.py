# Forward
mean64 = torch.zeros(B, dtype=torch.float32, device='cuda')
inv_std64 = torch.zeros(B, dtype=torch.float32, device='cuda')
tft_cuda.static_encoder_mp(
    x.half(), W1.half(), W2.half(), gamma.half(), beta.half(),
    out.half(), mean64, inv_std64, B, S
)

# Backward
dL_dinput = torch.zeros(B, 64, dtype=torch.float32, device='cuda')
tft_cuda.layer_norm_backward_mp(
    dL_dhidden.half(), input64.half(), gamma64.half(),
    mean64, inv_std64,
    dL_dgamma64, dL_dbeta64, dL_dinput,
    B, 64
)


#mha 
# Forward
Q_half = Q.float().half().cuda()
attn_weights = torch.zeros(B, T, H, T, dtype=torch.float32, device='cuda')
tft_cuda.multi_head_attention_mp(
    Q_half, K_half, V_half, output_half, attn_weights, 10000.0, B, T, H, D
)

# Backward
dL_dQ = torch.zeros_like(Q, dtype=torch.float32)
tft_cuda.mha_backward_mp(
    Q_half, K_half, V_half, attn_weights, dL_doutput.half(),
    dL_dQ, dL_dK, dL_dV, 10000.0, B, T, H, D
)
dL_dQ = dL_dQ.half()  # Convert back for optimizer


#quantile heads workflow 
# Forward (FP16)
preds = tft_cuda.quantile_heads(input.half(), W.half(), b.half(), ...)
loss = tft_cuda.quantile_loss(preds, targets.half(), quantiles.half(), ...)

# Backward (FP32 gradients)
dL_dW = torch.zeros(D, Q, dtype=torch.float32, device='cuda')
tft_cuda.quantile_heads_backward_mp(
    input.half(), W.half(), b.half(), quantiles.half(),
    preds.half(), targets.half(),
    dL_dW, dL_db, dL_dinput,
    B, T, D, Q
)

# Convert back to FP16 for optimizer
dL_dW = dL_dW.half()


#linear layer workflow
# Forward
output = torch.zeros(M, N, dtype=torch.float16, device='cuda')
tft_cuda.linear_forward_mp(input.half(), weight.half(), bias.half(), output, M, K, N)

# Backward
dL_dW = torch.zeros(N, K, dtype=torch.float32, device='cuda')
dL_db = torch.zeros(N, dtype=torch.float32, device='cuda')
tft_cuda.linear_backward_mp(dL_dout.half(), input.half(), dL_dW, dL_db, M, K, N)

# Convert to FP16 for optimizer
dL_dW = dL_dW.half()
dL_db = dL_db.half()