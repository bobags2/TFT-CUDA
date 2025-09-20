# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

TFT-CUDA is a high-performance implementation of the Temporal Fusion Transformer (TFT) optimized for financial time-series forecasting with CUDA acceleration. It focuses on multi-asset financial forecasting (ES, VX, ZN futures) with advanced interpretability features.

## Essential Commands

### Build and Setup
- `bash scripts/build.sh` - Full build & install with CUDA support detection
- `make build` - Alternative build command
- `make setup` - Setup development environment with dependencies

### Training and Testing
- `bash scripts/train.sh` - Train the TFT model with automatic data pipeline
- `bash scripts/test.sh` - Run comprehensive unit tests
- `make train` - Alternative training command
- `make test` - Alternative testing command

### Development
- `bash scripts/debug.sh` - Debug common issues with diagnostics
- `make dev` - Quick development cycle (clean, build, test)
- `make clean` - Clean build artifacts
- `make lint` - Run code linting (flake8, black, isort)
- `make format` - Format code with black and isort

### Specific Testing
- `make test-data` - Test data pipeline only
- `make test-model` - Test model creation only
- `make test-cuda` - Test CUDA availability

## Architecture Overview

### Core Components

**Python Modules** (`python/`):
- `data.py` - Financial data processing and feature engineering (30+ indicators)
- `tft_model.py` - Complete TFT implementation with Variable Selection Networks
- `trainer.py` - Training orchestration with early stopping and checkpointing
- `loss.py` - TFT quantile loss for probabilistic forecasting
- `interpretability.py` - Attention visualization and temporal importance analysis

**CUDA Kernels** (`cpp/`):
- `forward_pass/` - CUDA-optimized forward operations (LSTM, MHA, quantile heads)
- `backward_pass/` - Custom backward passes for gradient computation
- `interpretability/` - CUDA kernels for attention aggregation and analysis
- `tft_cuda.h` - Main header with kernel declarations

### Key Architecture Patterns

1. **Dual Implementation Strategy**: Each component has both PyTorch CPU fallback and CUDA-optimized versions
2. **Configuration-Driven**: All model/training parameters centralized in `config/default_config.json`
3. **Financial ML Focus**: Built-in support for microstructure features, regime detection, cross-asset modeling
4. **Production-Ready**: Comprehensive error handling, logging, and memory management

## Data Pipeline

The system expects 10-minute financial data with columns:
```
Date, Time, Open, High, Low, Last, Volume, NumberOfTrades, BidVolume, AskVolume
```

Expected data files in `data/`:
- `*es10m*.csv` - E-mini S&P 500 Futures
- `*vx10m*.csv` - VIX Volatility Index
- `*zn10m*.csv` - 10-Year US Treasury Yield

The pipeline automatically generates 30+ features including:
- Price/volume technical indicators
- Microstructure features (OFI, Kyle's lambda, bid-ask dynamics)
- Cross-asset correlations and regime indicators
- Time-based features for market sessions

## Build System

### Requirements
- Python 3.8+
- PyTorch 1.13+
- CUDA 11.0+ (optional, auto-detected)
- CMake 3.20+ (for CUDA builds)
- pybind11 for Python bindings

### Build Process
1. Auto-detects CUDA availability
2. Installs Python dependencies
3. Builds CUDA kernels with CMake if available
4. Falls back to CPU-only if CUDA unavailable
5. Installs package in development mode

## Model Configuration

Key configuration parameters in `config/default_config.json`:
- `input_size`: Number of input features
- `hidden_size`: Model dimensions (default: 256)
- `num_heads`: Attention heads (default: 8)
- `sequence_length`: Input sequence length (default: 100)
- `quantile_levels`: Prediction quantiles [0.1, 0.5, 0.9]
- `prediction_horizon`: Multi-step predictions [1, 5, 10]

## Development Guidelines

### Code Style
- Follow existing CUDA kernel patterns
- Use mixed precision (FP16) where appropriate
- Implement proper memory management and bounds checking
- Include comprehensive logging and error handling

### Financial ML Principles
- **Temporal Integrity**: Prevent data leakage and lookahead bias
- **Numerical Stability**: Use defensive programming for financial calculations
- **Risk Metrics**: Focus on Sharpe ratio, max drawdown, risk-adjusted returns
- **Market Regimes**: Consider volatility regimes and extreme events

### Testing Strategy
- Unit tests for each Python module
- CUDA kernel tests (if available)
- End-to-end training pipeline tests
- Numerical stability checks (NaN/Inf detection)
- Synthetic data fallbacks for missing real data

## Common Issues

### Build Issues
- Missing CUDA: Install CUDA toolkit or use CPU-only mode
- CMake errors: Ensure CMake 3.20+ and proper C++ compiler
- Import errors: Run `bash scripts/build.sh` to install dependencies

### Memory Issues
- Reduce batch_size in training config
- Use gradient accumulation for large batches
- Monitor GPU memory with debug script

### Data Issues
- Place CSV files in `data/` directory with expected format
- System automatically falls back to synthetic data for testing
- Check column format matches expected schema

## File Paths to Know

- `scripts/` - All automation scripts for build/train/test/debug
- `python/` - Core Python implementation
- `cpp/` - CUDA kernels and C++ extensions
- `config/` - Model and training configurations
- `data/` - Input data and processed features
- `tests/` - Unit tests and integration tests

## Byterover MCP Integration

This project includes Byterover MCP (Model Context Protocol) tools for enhanced development workflows. When working with plans or knowledge management, use the Byterover tools for context retrieval and storage as specified in the existing CLAUDE.md configuration.

## Important Reminders

- Always run `make lint` or equivalent before committing
- Use `bash scripts/debug.sh` to diagnose environment issues
- The system gracefully degrades from CUDA to CPU-only operation
- Synthetic data is automatically generated if real financial data is missing
- Model checkpoints and processed features are cached in `data/`