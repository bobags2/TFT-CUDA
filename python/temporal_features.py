"""
Temporal-aware feature engineering to prevent data leakage.
Features are computed incrementally using only past/present data.
"""

import pandas as pd
import numpy as np
from typing import Dict, Tuple, Optional, List
import warnings
warnings.filterwarnings('ignore')


class TemporalFeatureEngine:
    """
    Temporal-aware feature engineering that prevents data leakage.
    Features are computed using only information available up to each time point.
    """
    
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or self._default_config()
        self.feature_stats = {}  # Store rolling statistics
        
    def _default_config(self) -> Dict:
        return {
            'lookback_windows': [5, 10, 20, 50],
            'rolling_stats': ['mean', 'std', 'min', 'max'],
            'price_features': ['return', 'log_return', 'volatility'],
            'volume_features': ['volume_sma', 'volume_std', 'vwap'],
            'microstructure_features': ['ofi', 'kyle_lambda', 'spread']
        }
    
    def temporal_train_val_test_split(self, data: pd.DataFrame, 
                                    train_ratio: float = 0.7, 
                                    val_ratio: float = 0.15) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Temporal split BEFORE feature engineering to prevent leakage.
        
        Args:
            data: Raw time series data with datetime index
            train_ratio: Fraction for training
            val_ratio: Fraction for validation
            
        Returns:
            Tuple of (train_data, val_data, test_data)
        """
        # Ensure data is sorted by time
        data = data.sort_index()
        
        n_samples = len(data)
        train_end = int(n_samples * train_ratio)
        val_end = int(n_samples * (train_ratio + val_ratio))
        
        train_data = data.iloc[:train_end].copy()
        val_data = data.iloc[:val_end].copy()  # Include train for proper feature calculation
        test_data = data.copy()  # Include all for proper feature calculation
        
        print(f"Temporal split - Train: {len(train_data)}, Val period: {train_end}-{val_end}, Test period: {val_end}-{len(data)}")
        
        return train_data, val_data, test_data
    
    def engineer_temporal_features(self, data: pd.DataFrame, 
                                 split_type: str = 'train',
                                 reference_stats: Optional[Dict] = None) -> pd.DataFrame:
        """
        Engineer features using only information available up to each time point.
        
        Args:
            data: Time series data
            split_type: 'train', 'val', or 'test'
            reference_stats: Pre-computed statistics from training set
            
        Returns:
            DataFrame with temporal-aware features
        """
        print(f"   Engineering {split_type} features with temporal awareness...")
        
        # Copy to avoid modifying original
        df = data.copy()
        
        # Basic price features (no leakage)
        df = self._add_price_features(df)
        
        # Rolling features (computed incrementally)
        df = self._add_rolling_features(df, split_type, reference_stats)
        
        # Volume features
        df = self._add_volume_features(df, split_type, reference_stats)
        
        # Technical indicators (computed incrementally)
        df = self._add_technical_indicators(df)
        
        # Cross-asset features (temporal-aware)
        df = self._add_cross_asset_features(df)
        
        # Time-based features (no leakage risk)
        df = self._add_time_features(df)
        
        # Target variables (always computed properly)
        df = self._add_targets(df)
        
        print(f"   ✓ {split_type.capitalize()} features: {df.shape}")
        
        return df
    
    def _add_price_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add basic price features with no leakage risk."""
        price_cols = [col for col in df.columns if col.endswith('_close')]
        
        for col in price_cols:
            asset = col.split('_')[0]
            
            # Returns (safe - only use past prices)
            df[f'{asset}_return_1'] = df[col].pct_change(1)
            df[f'{asset}_return_5'] = df[col].pct_change(5)
            df[f'{asset}_log_return'] = np.log(df[col] / df[col].shift(1))
            
            # Price ranges (safe - only current bar)
            if f'{asset}_high' in df.columns and f'{asset}_low' in df.columns:
                df[f'{asset}_range'] = (df[f'{asset}_high'] - df[f'{asset}_low']) / df[col]
                df[f'{asset}_hl_ratio'] = df[f'{asset}_high'] / df[f'{asset}_low']
        
        return df
    
    def _add_rolling_features(self, df: pd.DataFrame, 
                            split_type: str, 
                            reference_stats: Optional[Dict] = None) -> pd.DataFrame:
        """Add rolling features computed incrementally to prevent leakage."""
        price_cols = [col for col in df.columns if col.endswith('_close')]
        
        for col in price_cols:
            asset = col.split('_')[0]
            
            # Rolling means (expanding for early periods to avoid NaN)
            for window in self.config['lookback_windows']:
                # Use expanding window for first 'window' periods
                rolling_mean = df[col].expanding().mean()
                rolling_mean.iloc[window:] = df[col].rolling(window).mean().iloc[window:]
                df[f'{asset}_sma_{window}'] = rolling_mean
                
                # Rolling volatility
                rolling_std = df[f'{asset}_return_1'].expanding().std()
                rolling_std.iloc[window:] = df[f'{asset}_return_1'].rolling(window).std().iloc[window:]
                df[f'{asset}_vol_{window}'] = rolling_std
                
                # Price relative to moving average
                df[f'{asset}_price_ma_ratio_{window}'] = df[col] / df[f'{asset}_sma_{window}']
        
        return df
    
    def _add_volume_features(self, df: pd.DataFrame, 
                           split_type: str, 
                           reference_stats: Optional[Dict] = None) -> pd.DataFrame:
        """Add volume features with temporal awareness."""
        volume_cols = [col for col in df.columns if col.endswith('_volume')]
        
        for col in volume_cols:
            asset = col.split('_')[0]
            
            # Volume moving averages
            for window in [10, 20]:
                rolling_vol_mean = df[col].expanding().mean()
                rolling_vol_mean.iloc[window:] = df[col].rolling(window).mean().iloc[window:]
                df[f'{asset}_vol_sma_{window}'] = rolling_vol_mean
                
                # Volume ratio
                df[f'{asset}_vol_ratio_{window}'] = df[col] / df[f'{asset}_vol_sma_{window}']
        
        return df
    
    def _add_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add technical indicators computed incrementally."""
        price_cols = [col for col in df.columns if col.endswith('_close')]
        
        for col in price_cols:
            asset = col.split('_')[0]
            
            # RSI (computed incrementally)
            df[f'{asset}_rsi'] = self._compute_rsi(df[col])
            
            # MACD
            ema_12 = df[col].ewm(span=12).mean()
            ema_26 = df[col].ewm(span=26).mean()
            df[f'{asset}_macd'] = ema_12 - ema_26
            df[f'{asset}_macd_signal'] = df[f'{asset}_macd'].ewm(span=9).mean()
            
            # Bollinger Bands
            sma_20 = df[col].rolling(20, min_periods=1).mean()
            std_20 = df[col].rolling(20, min_periods=1).std()
            df[f'{asset}_bb_upper'] = sma_20 + (std_20 * 2)
            df[f'{asset}_bb_lower'] = sma_20 - (std_20 * 2)
            df[f'{asset}_bb_position'] = (df[col] - df[f'{asset}_bb_lower']) / (df[f'{asset}_bb_upper'] - df[f'{asset}_bb_lower'])
        
        return df
    
    def _compute_rsi(self, prices: pd.Series, window: int = 14) -> pd.Series:
        """Compute RSI incrementally to prevent leakage."""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=window, min_periods=1).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window, min_periods=1).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def _add_cross_asset_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add cross-asset features with temporal awareness."""
        # Only compute if we have multiple assets
        price_cols = [col for col in df.columns if col.endswith('_close')]
        
        if len(price_cols) >= 2:
            # Rolling correlations (computed incrementally)
            for i, col1 in enumerate(price_cols):
                for col2 in price_cols[i+1:]:
                    asset1 = col1.split('_')[0]
                    asset2 = col2.split('_')[0]
                    
                    # Use return series for correlation
                    ret1 = df[col1].pct_change()
                    ret2 = df[col2].pct_change()
                    
                    # Rolling correlation
                    corr = ret1.rolling(window=50, min_periods=10).corr(ret2)
                    df[f'{asset1}_{asset2}_corr'] = corr
        
        return df
    
    def _add_time_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add time-based features (no leakage risk)."""
        if not isinstance(df.index, pd.DatetimeIndex):
            return df
        
        df['hour'] = df.index.hour
        df['day_of_week'] = df.index.dayofweek
        df['month'] = df.index.month
        
        # Market session indicators
        df['is_market_open'] = ((df.index.hour >= 9) & (df.index.hour < 16)).astype(int)
        df['is_overnight'] = ((df.index.hour < 9) | (df.index.hour >= 16)).astype(int)
        
        return df
    
    def _add_targets(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add target variables (forward-looking by design)."""
        price_cols = [col for col in df.columns if col.endswith('_close')]
        
        for col in price_cols:
            asset = col.split('_')[0]
            
            # Forward returns (properly computed targets)
            df[f'target_{asset}_return_1'] = df[col].pct_change().shift(-1)
            df[f'target_{asset}_return_5'] = df[col].pct_change(5).shift(-5)
            
            # Forward volatility
            rolling_vol = df[col].pct_change().rolling(5).std()
            df[f'target_{asset}_vol_5'] = rolling_vol.shift(-5)
        
        return df
    
    def process_temporal_pipeline(self, data: pd.DataFrame,
                                train_ratio: float = 0.7,
                                val_ratio: float = 0.15) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Complete temporal-aware processing pipeline.
        
        Args:
            data: Raw time series data
            train_ratio: Training split ratio
            val_ratio: Validation split ratio
            
        Returns:
            Tuple of (train_features, val_features, test_features)
        """
        print("🕒 Starting temporal-aware feature engineering pipeline...")
        
        # 1. CRITICAL: Split BEFORE feature engineering
        train_raw, val_raw, test_raw = self.temporal_train_val_test_split(data, train_ratio, val_ratio)
        
        # 2. Engineer features for each split using only available data
        print("   Engineering training features...")
        train_features = self.engineer_temporal_features(train_raw, split_type='train')
        
        # Compute reference statistics from training set
        train_stats = self._compute_reference_stats(train_features)
        
        print("   Engineering validation features...")  
        val_features = self.engineer_temporal_features(val_raw, split_type='val', reference_stats=train_stats)
        
        print("   Engineering test features...")
        test_features = self.engineer_temporal_features(test_raw, split_type='test', reference_stats=train_stats)
        
        # 3. Extract the actual splits (val and test portions only)
        n_train = len(train_raw)
        n_val_end = len(val_raw)
        
        train_final = train_features.copy()
        val_final = val_features.iloc[n_train:n_val_end].copy()
        test_final = test_features.iloc[n_val_end:].copy()
        
        print(f"✓ Temporal pipeline complete!")
        print(f"  Train: {train_final.shape}")
        print(f"  Val: {val_final.shape}")  
        print(f"  Test: {test_final.shape}")
        
        return train_final, val_final, test_final
    
    def _compute_reference_stats(self, train_data: pd.DataFrame) -> Dict:
        """Compute reference statistics from training data for normalization."""
        stats = {}
        
        # Compute stats for numerical columns
        numeric_cols = train_data.select_dtypes(include=[np.number]).columns
        
        for col in numeric_cols:
            stats[col] = {
                'mean': train_data[col].mean(),
                'std': train_data[col].std(),
                'min': train_data[col].min(),
                'max': train_data[col].max()
            }
        
        return stats


def create_temporal_datasets(data_path: str = "data/") -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Create temporal-aware train/val/test datasets with proper feature engineering.
    
    Returns:
        Tuple of (train_df, val_df, test_df) with engineered features
    """
    from pathlib import Path
    
    # Load raw data
    data_dir = Path(data_path)
    
    # Try to load real data
    try:
        # Look for data files
        es_files = list(data_dir.glob("es10m*.csv"))
        vx_files = list(data_dir.glob("vx10m*.csv"))
        zn_files = list(data_dir.glob("zn10m*.csv"))
        
        if es_files and vx_files and zn_files:
            print("Loading real financial data...")
            # Load and merge real data (implementation would go here)
            # For now, create synthetic data
            raise FileNotFoundError("Real data loading not implemented yet")
        else:
            raise FileNotFoundError("Data files not found")
    
    except (FileNotFoundError, Exception):
        print("Creating synthetic data for demonstration...")
        
        # Create synthetic financial data
        np.random.seed(42)
        dates = pd.date_range('2023-01-01', periods=2000, freq='10min')
        
        # Create realistic financial time series
        data = pd.DataFrame(index=dates)
        
        # ES (S&P 500) - trending with volatility clustering
        es_returns = np.random.randn(len(dates)) * 0.002
        es_returns[500:600] *= 3  # Volatility cluster
        es_returns[1200:1300] *= 2  # Another cluster
        data['es_close'] = 4000 * np.exp(np.cumsum(es_returns))
        data['es_volume'] = np.random.lognormal(8, 0.5, len(dates))
        
        # VX (VIX) - mean-reverting with regime changes
        vx_level = 20
        vx_series = [vx_level]
        for i in range(1, len(dates)):
            shock = np.random.randn() * 0.5
            mean_reversion = -0.1 * (vx_series[-1] - 20)
            vx_series.append(max(5, vx_series[-1] + mean_reversion + shock))
        
        data['vx_close'] = vx_series
        data['vx_volume'] = np.random.lognormal(6, 0.3, len(dates))
        
        # ZN (10-Year Treasury) - slow-moving with regime changes
        zn_returns = np.random.randn(len(dates)) * 0.001
        data['zn_close'] = 110 + np.cumsum(zn_returns)
        data['zn_volume'] = np.random.lognormal(7, 0.4, len(dates))
    
    print(f"Raw data shape: {data.shape}")
    
    # Initialize temporal feature engine
    engine = TemporalFeatureEngine()
    
    # Process with temporal awareness
    train_df, val_df, test_df = engine.process_temporal_pipeline(data)
    
    return train_df, val_df, test_df


if __name__ == "__main__":
    # Example usage
    train_df, val_df, test_df = create_temporal_datasets()
    
    print("\\nFeature engineering summary:")
    print(f"Training features: {train_df.shape}")
    print(f"Validation features: {val_df.shape}")
    print(f"Test features: {test_df.shape}")
    
    # Show feature columns
    feature_cols = [col for col in train_df.columns if not col.startswith('target')]
    target_cols = [col for col in train_df.columns if col.startswith('target')]
    
    print(f"\\nFeature columns ({len(feature_cols)}): {feature_cols[:10]}...")
    print(f"Target columns ({len(target_cols)}): {target_cols}")