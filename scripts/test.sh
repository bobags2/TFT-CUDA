#!/bin/bash
#
# Test script for TFT-CUDA project
# Runs all tests with clean output

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}🧪 TFT-CUDA Testing Script${NC}"
echo "=========================="

# Set Python path
export PYTHONPATH="${PYTHONPATH}:$(pwd)/python"

# Suppress warnings
export PYTHONWARNINGS="ignore"
export TFT_TESTING="1"

# Check for CUDA tests
echo -e "${BLUE}🔧 Checking CUDA unit tests...${NC}"
if [ -d "tests/cuda" ] && command -v make &> /dev/null; then
    echo "   Running CUDA tests..."
    cd tests/cuda && make test && cd ../..
else
    echo -e "   ${YELLOW}⚠️  CUDA tests not available (no tests/cuda or make command)${NC}"
fi

# Run Python tests with clean runner
echo -e "${BLUE}🐍 Running Python unit tests...${NC}"
echo "   Using clean test runner..."

cd tests
python3 run_tests.py
TEST_RESULT=$?
cd ..

# Report results
echo
if [ $TEST_RESULT -eq 0 ]; then
    echo -e "${GREEN}✅ All tests passed successfully!${NC}"
    exit 0
else
    echo -e "${RED}❌ Some tests failed. Please review the output above.${NC}"
    exit 1
fi
