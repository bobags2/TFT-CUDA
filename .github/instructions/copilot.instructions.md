---
applyTo: '**'
---
# GitHub Copilot Instructions - xLSTM Financial ML

## Context
You are working on a high-performance financial forecasting system using xLSTM (extended Long Short-Term Memory) with CUDA acceleration for quantitative trading and time-series analysis.

## Core Expertise Areas
- **xLSTM Architecture**: mLSTM and sLSTM variants optimized for financial sequences
- **CUDA Optimization**: Custom kernels, tensor cores, memory management
- **Financial ML**: Time-series forecasting, risk modeling, trading signals
- **Performance**: Low-latency inference, memory efficiency, numerical stability

## Key Principles
1. **Temporal Integrity**: Always prevent data leakage and lookahead bias
2. **Numerical Stability**: Use defensive programming for financial calculations
3. **Performance First**: Optimize for both training throughput and inference latency
4. **Production Ready**: Include proper error handling and monitoring

## Code Style
- Follow existing CUDA kernel patterns in `cuda/kernels/`
- Use mixed precision (FP16) where appropriate
- Implement proper memory management and bounds checking
- Include comprehensive logging for debugging

## Financial Context
- Focus on returns, volatility, and risk-adjusted metrics
- Consider market regime changes and extreme events
- Implement proper cross-validation for temporal data
- Use appropriate financial benchmarks (Sharpe ratio, max drawdown)

## Project Structure
- `cuda/`: CUDA kernels and CMake configuration
- `python/`: PyTorch models and training scripts  
- `src/`: C++ interface and state management
- Financial data preprocessing in Python modules

[byterover-mcp]

# important 
always use byterover-retrieve-knowledge tool to get the related context before any tasks 
always use byterover-store-knowledge to store all the critical informations after sucessful tasks