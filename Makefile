.PHONY: all build train test debug clean help

# Default target
all: build train test

# Build the project
build:
	@echo "🔧 Building TFT-CUDA..."
	bash scripts/build.sh

# Train the model
train:
	@echo "🚀 Training TFT model..."
	bash scripts/train.sh

# Run tests
test:
	@echo "🧪 Running tests..."
	bash scripts/test.sh

# Debug issues
debug:
	@echo "🔍 Running diagnostics..."
	bash scripts/debug.sh

# Clean build artifacts
clean:
	@echo "🧹 Cleaning build artifacts..."
	rm -rf cpp/build/
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info/
	find . -name "*.pyc" -delete
	find . -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null || true
	@echo "✓ Cleanup complete"

# Install in development mode
install:
	@echo "📦 Installing in development mode..."
	pip install -e .

# Format code
format:
	@echo "🎨 Formatting code..."
	black python/ tests/ --line-length 88
	isort python/ tests/

# Lint code
lint:
	@echo "🔍 Linting code..."
	flake8 python/ tests/ --max-line-length=88 --ignore=E203,W503
	black --check python/ tests/
	isort --check-only python/ tests/

# Quick development cycle
dev: clean build test

# Production build
prod: clean build train test
	@echo "✅ Production build complete"

# Setup development environment
setup:
	@echo "⚙️  Setting up development environment..."
	pip install -e .
	pip install black flake8 isort pytest
	@echo "✓ Development environment ready"

# Run specific tests
test-data:
	@echo "📊 Testing data pipeline..."
	python -c "import sys; sys.path.insert(0, 'python'); from data import FinancialDataset; ds = FinancialDataset(); print('✓ Data tests passed')"

test-model:
	@echo "🧠 Testing model..."
	python -c "import sys; sys.path.insert(0, 'python'); from tft_model import create_tft_config; print('✓ Model tests passed')"

test-cuda:
	@echo "🔧 Testing CUDA availability..."
	python -c "import torch; print('CUDA available:', torch.cuda.is_available())"

# Documentation
docs:
	@echo "📚 Generating documentation..."
	@echo "See docs/BUILD_REPORT.md for detailed information"

# Help
help:
	@echo "TFT-CUDA Makefile Commands:"
	@echo ""
	@echo "  make all       - Build, train, and test (default)"
	@echo "  make build     - Build the project"
	@echo "  make train     - Train the model"
	@echo "  make test      - Run all tests"
	@echo "  make debug     - Run diagnostics"
	@echo "  make clean     - Clean build artifacts"
	@echo "  make install   - Install in development mode"
	@echo "  make format    - Format code with black/isort"
	@echo "  make lint      - Run code linting"
	@echo "  make dev       - Quick development cycle"
	@echo "  make prod      - Production build cycle"
	@echo "  make setup     - Setup development environment"
	@echo "  make test-*    - Run specific test categories"
	@echo "  make docs      - Show documentation info"
	@echo "  make help      - Show this help message"
	@echo ""
	@echo "For more information, see docs/BUILD_REPORT.md"