"""
Financial data processing and feature engineering for TFT model.
Handles multi-asset 10-minute timeframe data with microstructure features.
"""

import pandas as pd
import numpy as np
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union
import warnings
warnings.filterwarnings('ignore')

try:
    import torch
    import torch.nn.functional as F
    from torch.utils.data import Dataset, DataLoader
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("Warning: PyTorch not available. Some functionality will be limited.")

try:
    from scipy import stats
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    print("Warning: SciPy not available. Some statistical features will be disabled.")


class FinancialDataset:
    """
    Financial dataset loader and feature engineer for multi-asset data.
    Supports ES (S&P 500), VX (VIX), and ZN (10-Year Treasury) data.
    """
    
    def __init__(self, data_dir: str = "data/", config: Optional[Dict] = None):
        self.data_dir = Path(data_dir)
        self.config = config or self._default_config()
        self.raw_data = {}
        self.processed_data = None
        self.feature_columns = []
        self.target_columns = []
        
    def _default_config(self) -> Dict:
        """Default configuration for data processing."""
        return {
            'assets': ['es', 'vx', 'zn'],
            'lookback_window': 100,  # 100 * 10min = 1000 minutes (~16h)
            'prediction_horizon': [1, 5, 10],  # 1, 5, 10 steps ahead
            'quantile_levels': [0.1, 0.5, 0.9],
            'robust_scaling': True,
            'handle_missing': 'forward_fill',
            'min_periods_for_features': 20,
            'rolling_windows': [5, 10, 20, 50],
            'market_hours': {
                'start': '09:30',
                'end': '16:00',
                'timezone': 'US/Eastern'
            }
        }
    
    def load_data(self, file_patterns: Optional[Dict[str, str]] = None) -> Dict[str, pd.DataFrame]:
        """
        Load financial data from CSV files.
        
        Args:
            file_patterns: Dict mapping asset names to file patterns
                          Default: {'es': 'es10m.csv', 'vx': 'vx10m.csv', 'zn': 'zn10m.csv'}
        
        Returns:
            Dict of DataFrames with asset data
        """
        if file_patterns is None:
            file_patterns = {
                'es': 'es10m.csv',
                'vx': 'vx10m.csv', 
                'zn': 'zn10m.csv'
            }
        
        for asset, pattern in file_patterns.items():
            file_path = self.data_dir / pattern
            if file_path.exists():
                print(f"Loading {asset} data from {file_path}")
                df = pd.read_csv(file_path)
                
                # Parse timestamp - combine Date and Time columns
                if 'Date' in df.columns and ' Time' in df.columns:
                    df['timestamp'] = pd.to_datetime(
                        df['Date'] + ' ' + df[' Time'].astype(str),
                        format='%Y%m%d %H:%M:%S'
                    )
                elif 'timestamp' not in df.columns:
                    # Assume first column is timestamp
                    df['timestamp'] = pd.to_datetime(df.iloc[:, 0])
                
                # Standardize column names
                df = self._standardize_columns(df, asset)
                df.set_index('timestamp', inplace=True)
                df.sort_index(inplace=True)
                
                self.raw_data[asset] = df
                print(f"Loaded {len(df)} records for {asset}")
            else:
                print(f"Warning: {file_path} not found. Creating sample data for {asset}")
                self.raw_data[asset] = self._create_sample_data(asset)
        
        return self.raw_data
    
    def _standardize_columns(self, df: pd.DataFrame, asset: str) -> pd.DataFrame:
        """Standardize column names across assets."""
        # Common column mappings
        column_mapping = {
            'Open': f'{asset}_open',
            'High': f'{asset}_high', 
            'Low': f'{asset}_low',
            'Last': f'{asset}_close',
            'Close': f'{asset}_close',
            'Volume': f'{asset}_volume',
            'NumberOfTrades': f'{asset}_num_trades',
            'BidVolume': f'{asset}_bid_volume',
            'AskVolume': f'{asset}_ask_volume'
        }
        
        # Apply mappings
        for old_col, new_col in column_mapping.items():
            if old_col in df.columns:
                df[new_col] = df[old_col]
        
        # Keep timestamp and standardized columns
        keep_columns = ['timestamp'] + [col for col in df.columns if col.startswith(f'{asset}_')]
        return df[keep_columns]
    
    def _create_sample_data(self, asset: str) -> pd.DataFrame:
        """Create sample data for testing when real data is not available."""
        print(f"Creating sample data for {asset}")
        
        # Generate 1000 10-minute bars (about 1 week of data)
        timestamps = pd.date_range(
            start='2024-01-01 09:30:00',
            periods=1000,
            freq='10min'
        )
        
        # Asset-specific price levels
        price_levels = {'es': 4500, 'vx': 15, 'zn': 110}
        base_price = price_levels.get(asset, 100)
        
        # Generate realistic OHLCV data with some autocorrelation
        np.random.seed(42)
        returns = np.random.normal(0, 0.001, len(timestamps))
        returns = np.cumsum(returns)  # Add some trend
        
        prices = base_price * np.exp(returns)
        
        # OHLC with realistic relationships
        close_prices = prices
        open_prices = np.roll(close_prices, 1)
        open_prices[0] = close_prices[0]
        
        # High/Low with realistic spreads
        volatility = np.abs(np.random.normal(0, 0.002, len(timestamps)))
        high_prices = np.maximum(open_prices, close_prices) + volatility * prices
        low_prices = np.minimum(open_prices, close_prices) - volatility * prices
        
        # Volume with some correlation to volatility
        volume = np.random.lognormal(8, 1, len(timestamps)) * (1 + 10 * volatility)
        num_trades = (volume * np.random.uniform(0.001, 0.01, len(timestamps))).astype(int)
        
        # Bid/Ask volumes (microstructure)
        bid_ask_ratio = np.random.beta(2, 2, len(timestamps))  # Centered around 0.5
        bid_volume = volume * bid_ask_ratio
        ask_volume = volume * (1 - bid_ask_ratio)
        
        df = pd.DataFrame({
            'timestamp': timestamps,
            f'{asset}_open': open_prices,
            f'{asset}_high': high_prices,
            f'{asset}_low': low_prices,
            f'{asset}_close': close_prices,
            f'{asset}_volume': volume,
            f'{asset}_num_trades': num_trades,
            f'{asset}_bid_volume': bid_volume,
            f'{asset}_ask_volume': ask_volume
        })
        
        df.set_index('timestamp', inplace=True)
        return df
    
    def merge_datasets(self) -> pd.DataFrame:
        """Merge all asset datasets on timestamp."""
        if not self.raw_data:
            raise ValueError("No data loaded. Call load_data() first.")
        
        # Start with first asset
        assets = list(self.raw_data.keys())
        merged = self.raw_data[assets[0]].copy()
        
        # Inner join with other assets
        for asset in assets[1:]:
            merged = merged.join(self.raw_data[asset], how='inner')
        
        # Forward fill missing values
        if self.config['handle_missing'] == 'forward_fill':
            merged = merged.fillna(method='ffill')
        
        print(f"Merged dataset shape: {merged.shape}")
        print(f"Date range: {merged.index.min()} to {merged.index.max()}")
        
        return merged
    
    def engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Comprehensive feature engineering for financial data.
        Implements 30+ features including microstructure, technical, and cross-asset features.
        """
        print("Starting feature engineering...")
        
        # Make a copy to avoid modifying original
        data = df.copy()
        assets = self.config['assets']
        
        # 1. Core Price & Volume Features
        for asset in assets:
            data = self._add_price_volume_features(data, asset)
        
        # 2. Microstructure Features
        for asset in assets:
            data = self._add_microstructure_features(data, asset)
        
        # 3. Technical Indicators
        data = self._add_technical_indicators(data, 'es')  # Focus on ES for technicals
        
        # 4. Time-based Features
        data = self._add_time_features(data)
        
        # 5. Cross-asset Features
        data = self._add_cross_asset_features(data, assets)
        
        # 6. Lagged Features
        data = self._add_lagged_features(data, assets)
        
        # 7. Target Variables
        data = self._add_target_variables(data, 'es')  # Predict ES returns
        
        # Clean up data
        data = self._clean_features(data)
        
        # Store feature and target columns
        self.feature_columns = [col for col in data.columns if not col.startswith('target_')]
        self.target_columns = [col for col in data.columns if col.startswith('target_')]
        
        print(f"Feature engineering complete. Features: {len(self.feature_columns)}, Targets: {len(self.target_columns)}")
        return data
    
    def _add_price_volume_features(self, data: pd.DataFrame, asset: str) -> pd.DataFrame:
        """Add core price and volume features."""
        # Price features
        o, h, l, c = f'{asset}_open', f'{asset}_high', f'{asset}_low', f'{asset}_close'
        v = f'{asset}_volume'
        
        if c not in data.columns:
            return data
        
        # Returns
        data[f'{asset}_log_return'] = np.log(data[c] / data[c].shift(1))
        data[f'{asset}_abs_return'] = np.abs(data[f'{asset}_log_return'])
        data[f'{asset}_signed_squared_return'] = np.sign(data[f'{asset}_log_return']) * data[f'{asset}_log_return']**2
        
        # Lagged returns
        for lag in [1, 5, 10]:
            data[f'{asset}_log_return_lag_{lag}'] = data[f'{asset}_log_return'].shift(lag)
        
        # Price ranges and ratios
        if all(col in data.columns for col in [o, h, l]):
            data[f'{asset}_price_range'] = (data[h] - data[l]) / data[o]
            data[f'{asset}_body_ratio'] = (data[c] - data[o]) / (data[h] - data[l] + 1e-8)
        
        # Volume features
        if v in data.columns:
            # Volume z-score
            rolling_mean = data[v].rolling(20).mean()
            rolling_std = data[v].rolling(20).std()
            data[f'{asset}_volume_z_score'] = (data[v] - rolling_mean) / (rolling_std + 1e-8)
            
            # Volume MA ratio
            data[f'{asset}_volume_ma_ratio'] = data[v] / (rolling_mean + 1e-8)
        
        return data
    
    def _add_microstructure_features(self, data: pd.DataFrame, asset: str) -> pd.DataFrame:
        """Add microstructure features using bid/ask volume and trade data."""
        bid_vol = f'{asset}_bid_volume'
        ask_vol = f'{asset}_ask_volume'
        num_trades = f'{asset}_num_trades'
        volume = f'{asset}_volume'
        close = f'{asset}_close'
        open_price = f'{asset}_open'
        high = f'{asset}_high'
        low = f'{asset}_low'
        
        if not all(col in data.columns for col in [bid_vol, ask_vol, volume]):
            return data
        
        # Order Flow Imbalance (OFI)
        data[f'{asset}_ofi'] = (data[ask_vol] - data[bid_vol]) / (data[ask_vol] + data[bid_vol] + 1e-8)
        
        # Trade Imbalance (using tick rule approximation)
        if all(col in data.columns for col in [close, open_price, num_trades]):
            # Estimate up/down trades
            up_trades = (data[close] > data[open_price]).astype(float) * data[num_trades]
            down_trades = data[num_trades] - up_trades
            data[f'{asset}_trade_imb'] = (up_trades - down_trades) / (data[num_trades] + 1e-8)
        
        # Kyle's Lambda Proxy
        if close in data.columns:
            abs_return = np.abs(data[f'{asset}_log_return']) if f'{asset}_log_return' in data.columns else 0
            data[f'{asset}_kyle_lambda'] = abs_return / (data[ask_vol] + data[bid_vol] + 1e-8)
        
        # Liquidity measures
        if all(col in data.columns for col in [high, low, open_price]):
            data[f'{asset}_depth_proxy'] = (data[ask_vol] + data[bid_vol]) / (data[high] - data[low] + 1e-8)
            data[f'{asset}_vol_per_bp'] = data[volume] / ((data[high] - data[low]) / data[open_price] * 10000 + 1e-8)
        
        # Bid-Ask Skew
        data[f'{asset}_bid_ask_skew'] = (data[ask_vol] - data[bid_vol]) / data[volume]
        
        return data
    
    def _add_technical_indicators(self, data: pd.DataFrame, asset: str) -> pd.DataFrame:
        """Add technical indicators (focus on ES)."""
        close = f'{asset}_close'
        high = f'{asset}_high'
        low = f'{asset}_low'
        volume = f'{asset}_volume'
        
        if close not in data.columns:
            return data
        
        # RSI
        delta = data[close].diff()
        gain = (delta.where(delta > 0, 0)).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / (loss + 1e-8)
        data[f'{asset}_rsi_14'] = 100 - (100 / (1 + rs))
        
        # MACD
        ema_12 = data[close].ewm(span=12).mean()
        ema_26 = data[close].ewm(span=26).mean()
        data[f'{asset}_macd'] = ema_12 - ema_26
        data[f'{asset}_macd_signal'] = data[f'{asset}_macd'].ewm(span=9).mean()
        data[f'{asset}_macd_hist'] = data[f'{asset}_macd'] - data[f'{asset}_macd_signal']
        
        # Bollinger Bands
        bb_window = 20
        bb_ma = data[close].rolling(bb_window).mean()
        bb_std = data[close].rolling(bb_window).std()
        data[f'{asset}_bollinger_upper'] = bb_ma + 2 * bb_std
        data[f'{asset}_bollinger_lower'] = bb_ma - 2 * bb_std
        data[f'{asset}_bb_width'] = (data[f'{asset}_bollinger_upper'] - data[f'{asset}_bollinger_lower']) / bb_ma
        data[f'{asset}_bb_position'] = (data[close] - data[f'{asset}_bollinger_lower']) / (data[f'{asset}_bollinger_upper'] - data[f'{asset}_bollinger_lower'] + 1e-8)
        
        # ATR (Average True Range)
        if all(col in data.columns for col in [high, low]):
            tr1 = data[high] - data[low]
            tr2 = np.abs(data[high] - data[close].shift(1))
            tr3 = np.abs(data[low] - data[close].shift(1))
            true_range = np.maximum(tr1, np.maximum(tr2, tr3))
            data[f'{asset}_atr_14'] = true_range.rolling(14).mean()
        
        # VWAP
        if volume in data.columns:
            data[f'{asset}_vwap'] = (data[close] * data[volume]).rolling(20).sum() / data[volume].rolling(20).sum()
        
        # Moving averages
        data[f'{asset}_price_ma_10'] = data[close].rolling(10).mean()
        data[f'{asset}_price_ma_50'] = data[close].rolling(50).mean()
        data[f'{asset}_ma_cross'] = (data[f'{asset}_price_ma_10'] > data[f'{asset}_price_ma_50']).astype(float)
        
        return data
    
    def _add_time_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Add time-based features."""
        # Basic time features
        data['hour_of_day'] = data.index.hour
        data['day_of_week'] = data.index.dayofweek
        data['month'] = data.index.month
        
        # Market session features
        market_start = 9.5  # 9:30 AM
        market_end = 16.0   # 4:00 PM
        
        hour_decimal = data.index.hour + data.index.minute / 60.0
        data['is_market_open'] = ((hour_decimal >= market_start) & (hour_decimal <= market_end)).astype(float)
        data['time_since_open'] = np.maximum(0, hour_decimal - market_start)
        data['time_to_close'] = np.maximum(0, market_end - hour_decimal)
        
        # Session indicators
        data['session_pre'] = (hour_decimal < market_start).astype(float)
        data['session_regular'] = ((hour_decimal >= market_start) & (hour_decimal <= market_end)).astype(float)
        data['session_post'] = (hour_decimal > market_end).astype(float)
        
        return data
    
    def _add_cross_asset_features(self, data: pd.DataFrame, assets: List[str]) -> pd.DataFrame:
        """Add cross-asset features for regime detection and correlation."""
        if 'vx' in assets and 'es' in assets:
            # VIX regime (volatility)
            vix_close = 'vx_close'
            if vix_close in data.columns:
                vix_quantiles = data[vix_close].rolling(252).quantile([0.2, 0.8])  # 252 periods for percentiles
                data['vix_regime_low'] = (data[vix_close] <= data[vix_close].rolling(252).quantile(0.2)).astype(float)
                data['vix_regime_high'] = (data[vix_close] >= data[vix_close].rolling(252).quantile(0.8)).astype(float)
                data['vix_regime_medium'] = (1 - data['vix_regime_low'] - data['vix_regime_high'])
                
                # VIX-ES ratio
                es_close = 'es_close'
                if es_close in data.columns:
                    data['vix_es_ratio'] = data[vix_close] / data[es_close]
        
        # Cross-asset correlations
        if len(assets) >= 2:
            for i, asset1 in enumerate(assets):
                for asset2 in assets[i+1:]:
                    ret1 = f'{asset1}_log_return'
                    ret2 = f'{asset2}_log_return'
                    if all(col in data.columns for col in [ret1, ret2]):
                        # Rolling correlation
                        data[f'{asset1}_{asset2}_corr_20'] = data[ret1].rolling(20).corr(data[ret2])
                        
                        # Divergence signals
                        data[f'{asset1}_{asset2}_div'] = data[ret1] - data[ret2]
        
        return data
    
    def _add_lagged_features(self, data: pd.DataFrame, assets: List[str]) -> pd.DataFrame:
        """Add lagged features for temporal dependencies."""
        for asset in assets:
            close_col = f'{asset}_close'
            if close_col in data.columns:
                # Lagged prices
                for lag in range(1, 6):
                    data[f'{asset}_close_lag_{lag}'] = data[close_col].shift(lag)
                
                # Rolling statistics
                for window in [20, 50]:
                    data[f'{asset}_ma_{window}'] = data[close_col].rolling(window).mean()
                    data[f'{asset}_std_{window}'] = data[close_col].rolling(window).std()
                
                # Volume spikes
                vol_col = f'{asset}_volume'
                if vol_col in data.columns:
                    vol_mean = data[vol_col].rolling(20).mean()
                    vol_std = data[vol_col].rolling(20).std()
                    data[f'{asset}_volume_spike'] = (data[vol_col] > vol_mean + 2 * vol_std).astype(float)
        
        return data
    
    def _add_target_variables(self, data: pd.DataFrame, primary_asset: str = 'es') -> pd.DataFrame:
        """Add target variables for prediction."""
        close_col = f'{primary_asset}_close'
        if close_col not in data.columns:
            return data
        
        # Future returns at different horizons
        for horizon in self.config['prediction_horizon']:
            future_close = data[close_col].shift(-horizon)
            data[f'target_return_{horizon}'] = np.log(future_close / data[close_col])
            data[f'target_price_{horizon}'] = future_close
            data[f'target_direction_{horizon}'] = (future_close > data[close_col]).astype(float)
        
        return data
    
    def _clean_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Clean features by handling inf, nan, and outliers."""
        # Replace inf with nan
        data = data.replace([np.inf, -np.inf], np.nan)
        
        # Drop columns with too many missing values
        missing_threshold = 0.5
        missing_ratios = data.isnull().sum() / len(data)
        cols_to_drop = missing_ratios[missing_ratios > missing_threshold].index
        if len(cols_to_drop) > 0:
            print(f"Dropping columns with >{missing_threshold*100}% missing values: {list(cols_to_drop)}")
            data = data.drop(columns=cols_to_drop)
        
        # Forward fill remaining missing values
        data = data.fillna(method='ffill')
        
        # Drop rows with any remaining NaN (typically first few rows)
        data = data.dropna()
        
        return data
    
    def normalize_features(self, data: pd.DataFrame, method: str = 'robust') -> Tuple[pd.DataFrame, Dict]:
        """
        Normalize features using robust scaling or standard scaling.
        
        Args:
            data: DataFrame with features
            method: 'robust' or 'standard'
            
        Returns:
            Tuple of (normalized_data, scaling_params)
        """
        feature_cols = [col for col in data.columns if not col.startswith('target_')]
        scaling_params = {}
        
        normalized_data = data.copy()
        
        for col in feature_cols:
            if method == 'robust':
                median = data[col].median()
                q75 = data[col].quantile(0.75)
                q25 = data[col].quantile(0.25)
                iqr = q75 - q25
                
                if iqr > 1e-8:
                    normalized_data[col] = (data[col] - median) / iqr
                    scaling_params[col] = {'median': median, 'iqr': iqr, 'method': 'robust'}
                else:
                    scaling_params[col] = {'median': median, 'iqr': 1.0, 'method': 'robust'}
                    
            else:  # standard scaling
                mean = data[col].mean()
                std = data[col].std()
                
                if std > 1e-8:
                    normalized_data[col] = (data[col] - mean) / std
                    scaling_params[col] = {'mean': mean, 'std': std, 'method': 'standard'}
                else:
                    scaling_params[col] = {'mean': mean, 'std': 1.0, 'method': 'standard'}
        
        return normalized_data, scaling_params
    
    def create_sequences(self, data: pd.DataFrame, sequence_length: int = 100) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create sequences for time series modeling.
        
        Args:
            data: Processed DataFrame with features and targets
            sequence_length: Length of input sequences
            
        Returns:
            Tuple of (X, y) arrays
        """
        if not TORCH_AVAILABLE:
            print("Warning: PyTorch not available. Returning numpy arrays.")
        
        feature_cols = [col for col in data.columns if not col.startswith('target_')]
        target_cols = [col for col in data.columns if col.startswith('target_')]
        
        X_data = data[feature_cols].values
        y_data = data[target_cols].values if target_cols else None
        
        # Create sequences
        X_sequences = []
        y_sequences = []
        
        for i in range(sequence_length, len(data)):
            X_sequences.append(X_data[i-sequence_length:i])
            if y_data is not None:
                y_sequences.append(y_data[i])
        
        X = np.array(X_sequences, dtype=np.float32)
        y = np.array(y_sequences, dtype=np.float32) if y_sequences else None
        
        print(f"Created {len(X)} sequences of length {sequence_length}")
        print(f"Feature shape: {X.shape}, Target shape: {y.shape if y is not None else None}")
        
        return X, y
    
    def train_val_test_split(self, X: np.ndarray, y: np.ndarray, 
                           train_ratio: float = 0.7, val_ratio: float = 0.15) -> Tuple:
        """
        Time-series aware train/validation/test split.
        
        Args:
            X: Feature sequences
            y: Target sequences  
            train_ratio: Fraction for training
            val_ratio: Fraction for validation
            
        Returns:
            Tuple of (X_train, X_val, X_test, y_train, y_val, y_test)
        """
        n_samples = len(X)
        train_end = int(n_samples * train_ratio)
        val_end = int(n_samples * (train_ratio + val_ratio))
        
        X_train = X[:train_end]
        X_val = X[train_end:val_end]
        X_test = X[val_end:]
        
        y_train = y[:train_end] if y is not None else None
        y_val = y[train_end:val_end] if y is not None else None
        y_test = y[val_end:] if y is not None else None
        
        print(f"Split sizes - Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")
        
        return X_train, X_val, X_test, y_train, y_val, y_test
    
    def save_processed_data(self, data: pd.DataFrame, filepath: str = None):
        """Save processed data and metadata."""
        if filepath is None:
            filepath = self.data_dir / "features.parquet"
        
        # Save main data
        data.to_parquet(filepath)
        print(f"Saved processed data to {filepath}")
        
        # Save feature metadata
        metadata = {
            'feature_columns': self.feature_columns,
            'target_columns': self.target_columns,
            'config': self.config,
            'n_features': len(self.feature_columns),
            'n_targets': len(self.target_columns),
            'data_shape': data.shape,
            'date_range': {
                'start': str(data.index.min()),
                'end': str(data.index.max())
            }
        }
        
        metadata_path = self.data_dir / "feature_metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        print(f"Saved metadata to {metadata_path}")
    
    def process_pipeline(self, file_patterns: Optional[Dict[str, str]] = None) -> pd.DataFrame:
        """
        Complete data processing pipeline.
        
        Args:
            file_patterns: Optional file patterns for loading data
            
        Returns:
            Processed DataFrame ready for modeling
        """
        print("Starting complete data processing pipeline...")
        
        # 1. Load data
        self.load_data(file_patterns)
        
        # 2. Merge datasets
        merged_data = self.merge_datasets()
        
        # 3. Engineer features
        processed_data = self.engineer_features(merged_data)
        
        # 4. Save processed data
        self.save_processed_data(processed_data)
        
        self.processed_data = processed_data
        print(f"Pipeline complete! Final data shape: {processed_data.shape}")
        
        return processed_data


# PyTorch Dataset class for TFT training
if TORCH_AVAILABLE:
    class TFTDataset(Dataset):
        """PyTorch Dataset for TFT model training."""
        
        def __init__(self, X: np.ndarray, y: np.ndarray = None):
            self.X = torch.tensor(X, dtype=torch.float32)
            self.y = torch.tensor(y, dtype=torch.float32) if y is not None else None
        
        def __len__(self):
            return len(self.X)
        
        def __getitem__(self, idx):
            if self.y is not None:
                return self.X[idx], self.y[idx]
            return self.X[idx]


def main():
    """Example usage of the FinancialDataset class."""
    print("Financial Dataset Processing Example")
    print("=" * 50)
    
    # Initialize dataset
    dataset = FinancialDataset(data_dir="data/")
    
    # Process data
    processed_data = dataset.process_pipeline()
    
    # Create sequences
    X, y = dataset.create_sequences(processed_data, sequence_length=100)
    
    # Split data
    X_train, X_val, X_test, y_train, y_val, y_test = dataset.train_val_test_split(X, y)
    
    # Normalize features
    normalized_data, scaling_params = dataset.normalize_features(processed_data)
    
    print("\nProcessing complete!")
    print(f"Features: {len(dataset.feature_columns)}")
    print(f"Targets: {len(dataset.target_columns)}")
    print(f"Training sequences: {len(X_train)}")


if __name__ == "__main__":
    main()