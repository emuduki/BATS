"""
Data preparation pipeline for Phase 3 AI training.
Handles data loading, feature engineering, and train/validation/test splits.
"""

import pandas as pd
import numpy as np
from typing import Dict, Tuple, Optional, List
from sklearn.model_selection import train_test_split
from ai.datasets.horizon_configs import HorizonConfig, HORIZON_CONFIGS
from ai.dataset.feature_engineering import (
    create_all_features, 
    prepare_horizon_dataset,
    get_feature_columns
)
import os
from datetime import datetime, timedelta


class HistoricalDataLoader:
    """Handles loading of historical market data from various sources."""
    
    def __init__(self, data_dir: str = "data/historical"):
        self.data_dir = data_dir
        os.makedirs(data_dir, exist_ok=True)
    
    def load_csv_data(self, filename: str) -> pd.DataFrame:
        """Load data from CSV file."""
        path = os.path.join(self.data_dir, filename)
        if not os.path.exists(path):
            raise FileNotFoundError(f"Data file not found: {path}")
        
        df = pd.read_csv(path)
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
        return df
    
    def generate_synthetic_data(self, n_samples: int = 100000) -> pd.DataFrame:
        """Generate synthetic OHLCV data for testing."""
        dates = pd.date_range(start='2026-01-01', periods=n_samples, freq='1min')
        base_price = 1250.0
        
        # Generate realistic price series with drift
        returns = np.random.randn(n_samples) * 0.0005
        prices = base_price * np.cumprod(1 + returns)
        
        df = pd.DataFrame({
            'timestamp': dates,
            'open': prices * (1 + np.abs(np.random.randn(n_samples) * 0.0001)),
            'high': prices * (1 + np.abs(np.random.randn(n_samples) * 0.00015)),
            'low': prices * (1 - np.abs(np.random.randn(n_samples) * 0.00015)),
            'close': prices,
            'volume': np.random.randint(100000, 1000000, n_samples)
        })
        
        return df


class DataSplitter:
    """Handles time-series aware data splitting."""
    
    @staticmethod
    def chronological_split(
        df: pd.DataFrame, 
        train_ratio: float = 0.7,
        val_ratio: float = 0.15,
        test_ratio: float = 0.15,
        date_column: str = 'timestamp'
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Split data chronologically without shuffling.
        
        Returns:
            train_df, val_df, test_df
        """
        # Ensure chronological order
        df = df.sort_values(by=date_column).reset_index(drop=True)
        
        n = len(df)
        train_end = int(n * train_ratio)
        val_end = int(n * (train_ratio + val_ratio))
        
        train_df = df.iloc[:train_end]
        val_df = df.iloc[train_end:val_end]
        test_df = df.iloc[val_end:]
        
        return train_df, val_df, test_df
    
    @staticmethod
    def walk_forward_split(
        df: pd.DataFrame,
        window_size: int,
        step_size: int,
        date_column: str = 'timestamp'
    ) -> List[Tuple[pd.DataFrame, pd.DataFrame]]:
        """
        Generate walk-forward validation splits.
        
        Returns list of (train_df, test_df) tuples.
        """
        splits = []
        n = len(df)
        start_idx = 0
        
        while start_idx + window_size < n:
            end_idx = start_idx + window_size
            test_end = min(end_idx + step_size, n)
            
            train_df = df.iloc[start_idx:end_idx]
            test_df = df.iloc[end_idx:test_end]
            
            splits.append((train_df, test_df))
            start_idx += step_size
        
        return splits


class DataPreparationPipeline:
    """Complete pipeline for preparing training data."""
    
    def __init__(self, data_loader: Optional[HistoricalDataLoader] = None):
        self.data_loader = data_loader or HistoricalDataLoader()
        self.splitter = DataSplitter()
    
    def prepare_training_data(
        self,
        df: pd.DataFrame,
        horizon_name: str,
        train_ratio: float = 0.7,
        val_ratio: float = 0.15,
        test_ratio: float = 0.15
    ) -> Dict[str, any]:
        """
        Prepare complete dataset for training.
        
        Returns dictionary containing:
        - train_df, val_df, test_df
        - X_train, X_val, X_test
        - y_train, y_val, y_test
        - feature_columns
        - horizon_config
        """
        config = HORIZON_CONFIGS[horizon_name]
        
        # Create features and labels
        df_features, X, y = prepare_horizon_dataset(df, horizon_name)
        
        # Split data chronologically
        train_df, val_df, test_df = self.splitter.chronological_split(
            df_features,
            train_ratio=train_ratio,
            val_ratio=val_ratio,
            test_ratio=test_ratio
        )
        
        # Extract features and labels for each split
        feature_cols = get_feature_columns(train_df)
        
        X_train = train_df[feature_cols].values
        y_train = train_df['label'].values
        
        X_val = val_df[feature_cols].values
        y_val = val_df['label'].values
        
        X_test = test_df[feature_cols].values
        y_test = test_df['label'].values
        
        return {
            'train_df': train_df,
            'val_df': val_df,
            'test_df': test_df,
            'X_train': X_train,
            'y_train': y_train,
            'X_val': X_val,
            'y_val': y_val,
            'X_test': X_test,
            'y_test': y_test,
            'feature_columns': feature_cols,
            'horizon_config': config,
            'feature_count': len(feature_cols)
        }
    
    def prepare_walk_forward_data(
        self,
        df: pd.DataFrame,
        horizon_name: str,
        window_size: int = 1000,
        step_size: int = 200
    ) -> List[Dict[str, any]]:
        """
        Prepare data for walk-forward validation.
        
        Returns list of dictionaries, each containing:
        - X_train, X_val, y_train, y_val
        - train_df, val_df
        - feature_columns
        """
        df_features, X, y = prepare_horizon_dataset(df, horizon_name)
        feature_cols = get_feature_columns(df_features)
        
        splits = self.splitter.walk_forward_split(df_features, window_size, step_size)
        
        prepared_splits = []
        for train_df, val_df in splits:
            X_train = train_df[feature_cols].values
            y_train = train_df['label'].values
            X_val = val_df[feature_cols].values
            y_val = val_df['label'].values
            
            prepared_splits.append({
                'X_train': X_train,
                'y_train': y_train,
                'X_val': X_val,
                'y_val': y_val,
                'train_df': train_df,
                'val_df': val_df,
                'feature_columns': feature_cols
            })
        
        return prepared_splits


# Example usage functions
def load_and_prepare_data(
    data_source: str = "synthetic",
    horizon: str = "60s",
    **kwargs
) -> Dict[str, any]:
    """
    Convenience function to load and prepare data.
    
    data_source: "synthetic", "csv", or path to CSV file
    """
    loader = HistoricalDataLoader()
    
    if data_source == "synthetic":
        df = loader.generate_synthetic_data(**kwargs)
    elif data_source == "csv":
        df = loader.load_csv_data(**kwargs)
    else:
        df = loader.load_csv_data(data_source)
    
    pipeline = DataPreparationPipeline(loader)
    return pipeline.prepare_training_data(df, horizon)