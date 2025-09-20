"""
Utility functions and silent imports for the TFT project.
"""

import sys
import warnings
from contextlib import contextmanager
import io

@contextmanager
def suppress_output():
    """Context manager to suppress stdout and stderr."""
    old_stdout = sys.stdout
    old_stderr = sys.stderr
    try:
        sys.stdout = io.StringIO()
        sys.stderr = io.StringIO()
        yield
    finally:
        sys.stdout = old_stdout
        sys.stderr = old_stderr

# Import wandb silently
WANDB_AVAILABLE = False
try:
    with suppress_output():
        import wandb
        WANDB_AVAILABLE = True
except ImportError:
    pass

# Check if we're in test mode
IS_TESTING = any('test' in arg.lower() for arg in sys.argv)

def maybe_import_wandb():
    """Import wandb only if available and not in test mode."""
    if WANDB_AVAILABLE and not IS_TESTING:
        return wandb
    return None

def suppress_wandb_warning():
    """Suppress the Weights & Biases warning if in test mode."""
    if IS_TESTING:
        warnings.filterwarnings('ignore', message='.*Weights & Biases.*')
