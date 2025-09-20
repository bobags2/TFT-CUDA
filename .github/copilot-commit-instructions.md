# Commit Message Instructions

## Format
Use conventional commit format: `type(scope): description`

## Types
- **feat**: New features or capabilities
- **fix**: Bug fixes and corrections  
- **perf**: Performance optimizations
- **refactor**: Code restructuring without feature changes
- **test**: Adding or updating tests
- **docs**: Documentation updates
- **ci**: CI/CD pipeline changes

## Scopes
- **cuda**: CUDA kernels and GPU optimization
- **xlstm**: xLSTM model architecture and layers
- **training**: Training loops, optimization, and data loading
- **data**: Data preprocessing and feature engineering
- **model**: Model architecture and inference
- **financial**: Financial calculations and trading logic
- **memory**: Memory management and optimization

## Message Guidelines
- Keep under 72 characters for the subject line
- Focus on impact: performance gains, stability improvements, new capabilities
- Include numerical context when relevant (e.g., "reduce memory usage by 30%")
- Emphasize financial ML context: temporal integrity, numerical stability, trading performance

## Examples
- `feat(cuda): add FP16 WMMA tensor core support for mLSTM kernels`
- `fix(xlstm): prevent temporal leakage in financial data pipeline`
- `perf(training): optimize GPU memory usage for large batch training`
- `refactor(data): restructure financial feature engineering pipeline`
- `fix(financial): correct risk calculation numerical stability issues`

## Context Focus
Highlight changes that impact:
- Trading system performance and latency
- Numerical precision and stability
- Memory efficiency and GPU utilization
- Financial data integrity and temporal handling