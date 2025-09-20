#!/usr/bin/env python3
"""
Run tests with suppressed warnings for cleaner output.
"""

import sys
import os
import warnings
import logging

# Suppress all warnings for clean test output
warnings.filterwarnings('ignore')
os.environ['PYTHONWARNINGS'] = 'ignore'
logging.disable(logging.WARNING)

# Set test environment variable
os.environ['TFT_TESTING'] = '1'

# Add python directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'python'))

# Redirect import warnings
import io
old_stdout = sys.stdout
old_stderr = sys.stderr

# Import modules with suppressed output
sys.stdout = io.StringIO()
sys.stderr = io.StringIO()

try:
    import tft_model
    import trainer
    import loss
    import data
except:
    pass
finally:
    sys.stdout = old_stdout
    sys.stderr = old_stderr

# Now run the actual tests
sys.stdout = old_stdout
sys.stderr = old_stderr

# Import test modules
from test_basic import run_all_tests as run_basic_tests
from test_simple import run_tests as run_simple_tests

def main():
    """Main test runner."""
    print("🧪 TFT-CUDA Testing Suite")
    print("=" * 40)
    print()
    
    all_passed = True
    
    # Run basic tests
    print("📋 Running Basic Tests...")
    print("-" * 30)
    if not run_basic_tests():
        all_passed = False
    print()
    
    # Run simple tests
    print("📋 Running Core Tests...")
    print("-" * 30)
    if not run_simple_tests():
        all_passed = False
    print()
    
    # Final summary
    print("=" * 40)
    if all_passed:
        print("✅ All tests completed successfully!")
        sys.exit(0)
    else:
        print("❌ Some tests failed. Review output above.")
        sys.exit(1)

if __name__ == "__main__":
    main()
