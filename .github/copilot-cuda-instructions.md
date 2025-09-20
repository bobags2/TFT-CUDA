# CUDA Optimization Instructions

## CUDA Development Focus
You are optimizing CUDA kernels for xLSTM layers in financial time-series processing with emphasis on memory efficiency and numerical precision.

## Key CUDA Patterns
- **Tensor Cores**: Use WMMA API for FP16 mixed precision operations
- **Memory Management**: Implement proper bounds checking and coalesced access
- **Kernel Fusion**: Combine operations to reduce memory bandwidth
- **Shared Memory**: Optimize tile sizes for maximum occupancy

## Performance Priorities
1. **Memory Bandwidth**: Minimize global memory transactions
2. **Occupancy**: Balance shared memory usage with thread blocks
3. **Precision**: Maintain numerical stability with FP16/FP32 mixed precision
4. **Latency**: Optimize for real-time inference in trading systems

## Code Patterns
- Follow existing kernel structure in `cuda/kernels/slstm_kernels.cu`
- Use `__device__` functions for reusable components
- Implement defensive bounds checking with `assert()` in debug builds
- Use `cudaMemcpy` with proper error handling

## Memory Optimization
- Pin memory for host-device transfers
- Use CUDA streams for overlapping computation
- Implement gradient accumulation for large batch sizes
- Consider memory fragmentation in long training runs

## Debugging
- Add comprehensive CUDA error checking
- Use `nvidia-smi` for memory monitoring
- Include timing measurements with CUDA events
- Test with different batch sizes and sequence lengths