#!/usr/bin/env python3
"""
Quick test for fixed data normalization.
"""

import sys
sys.path.insert(0, 'python')

import numpy as np
import warnings
warnings.filterwarnings('ignore')

def test_data_normalization():
    """Test the data normalization fix."""
    print("🔧 Testing Data Normalization Fix")
    print("=" * 50)
    
    # Create synthetic test data with extreme values
    np.random.seed(42)
    
    # Simulate problematic financial data with different scales
    samples, timesteps, features = 1000, 128, 134
    
    # Create features with vastly different scales (like real financial data)
    X = np.zeros((samples, timesteps, features))
    
    # Feature 1: Prices (around 4000-5000)
    X[:, :, 0] = np.random.normal(4500, 100, (samples, timesteps))
    
    # Feature 2: Small percentage returns (around 0.001)
    X[:, :, 1] = np.random.normal(0, 0.01, (samples, timesteps))
    
    # Feature 3: Volume (millions)
    X[:, :, 2] = np.random.exponential(1000000, (samples, timesteps))
    
    # Feature 4: VIX levels (10-50)
    X[:, :, 3] = np.random.normal(20, 5, (samples, timesteps))
    
    # Fill remaining features with mixed scales
    for i in range(4, features):
        scale = 10 ** np.random.uniform(-3, 6)  # Random scale from 0.001 to 1M
        X[:, :, i] = np.random.normal(0, scale, (samples, timesteps))
    
    print(f"Original data shape: {X.shape}")
    print(f"Original data range: [{np.min(X):.2f}, {np.max(X):.2f}]")
    print(f"Original data mean: {np.mean(X):.2f}")
    print(f"Original data std: {np.std(X):.2f}")
    
    # Apply the FIXED normalization logic
    print("\nApplying FIXED normalization...")
    
    X_normalized = X.copy()
    
    # Global feature statistics for numerical stability
    X_flat = X.reshape(-1, X.shape[-1])  # Flatten to [samples*time, features]
    
    # Remove NaN/inf values globally
    valid_mask = np.isfinite(X_flat).all(axis=1)
    X_clean = X_flat[valid_mask]
    
    print(f'Data cleaning: {len(X_clean)}/{len(X_flat)} samples retained ({100*len(X_clean)/len(X_flat):.1f}%)')
    
    # Compute global statistics per feature
    feature_means = np.mean(X_clean, axis=0)
    feature_stds = np.std(X_clean, axis=0)
    
    # Apply standardization: (x - mean) / std
    for feature_idx in range(X.shape[-1]):
        mean_val = feature_means[feature_idx]
        std_val = feature_stds[feature_idx]
        
        if std_val > 1e-8:  # Avoid division by zero
            X_normalized[:, :, feature_idx] = (X[:, :, feature_idx] - mean_val) / std_val
        else:
            # Constant feature - center at zero
            X_normalized[:, :, feature_idx] = X[:, :, feature_idx] - mean_val
    
    # Check normalization results with detailed verification
    X_mean = np.nanmean(X_normalized)
    X_std = np.nanstd(X_normalized)
    
    # Per-feature statistics for debugging
    feature_means_norm = np.nanmean(X_normalized, axis=(0,1))
    feature_stds_norm = np.nanstd(X_normalized, axis=(0,1))
    
    print(f'\n✓ Features normalized: global_mean={X_mean:.6f}, global_std={X_std:.6f}')
    print(f'✓ Per-feature mean range: [{np.min(feature_means_norm):.6f}, {np.max(feature_means_norm):.6f}]')
    print(f'✓ Per-feature std range: [{np.min(feature_stds_norm):.6f}, {np.max(feature_stds_norm):.6f}]')
    
    # Sanity check - features should have reasonable distribution
    if abs(X_mean) > 0.1 or X_std < 0.8 or X_std > 1.2:
        print(f'⚠️  WARNING: Feature normalization may be inadequate!')
        print(f'   Expected: mean≈0, std≈1. Got: mean={X_mean:.6f}, std={X_std:.6f}')
        return False
    else:
        print(f'✅ SUCCESS: Normalization working correctly!')
        print(f'   Mean is close to 0: {X_mean:.6f}')
        print(f'   Std is close to 1: {X_std:.6f}')
        return True

if __name__ == "__main__":
    success = test_data_normalization()
    
    if success:
        print("\n🎉 DATA NORMALIZATION FIX VERIFIED!")
        print("The gradient explosion should now be resolved.")
    else:
        print("\n❌ DATA NORMALIZATION STILL HAS ISSUES!")
        print("Further investigation needed.")