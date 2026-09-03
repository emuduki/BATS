"""
Feature engineering module for Phase 3 AI models.
Creates features and labels from historical market data.
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Optional, Tuple
from ai.datasets.horizon_configs import HORIZON_CONFIGS, HorizonConfig


def calculate_returns(df: pd.DataFrame, periods: List[int]) -> pd.DataFrame:
    """Calculate percentage returns for specified periods."""
    df = df.copy()
    for p in periods:
        df[f'return_{p}'] = df['close'].pct_change(p)
    return df


def calculate_price_position_features(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate price position within candle and body/wick ratios."""
    df = df.copy()
    df['close_position'] = (df['close'] - df['low']) / (df['high'] - df['low'] + 1e-10)
    df['body_size'] = np.abs(df['close'] - df['open']) / (df['high'] - df['low'] + 1e-10)
    df['upper_wick'] = (df['high'] - df[['open', 'close']].max(axis=1)) / (df['high'] - df['low'] + 1e-10)
    df['lower_wick'] = (df[['open', 'close']].min(axis=1) - df['low']) / (df['high'] - df['low'] + 1e-10)
    return df


def calculate_microstructure_features(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate volume and microstructure features."""
    df = df.copy()
    df['volume_10_ma'] = df['volume'].rolling(10).mean()
    df['volume_50_ma'] = df['volume'].rolling(50).mean()
    df['volume_ratio'] = df['volume'] / (df['volume_10_ma'] + 1e-10)
    df['volume_deviation'] = np.log(df['volume'] / (df['volume_50_ma'] + 1e-10) + 1)
    return df


def apply_label_threshold(df: pd.DataFrame, config: HorizonConfig) -> pd.DataFrame:
    """
    Apply meaningful movement threshold to create labels.
    
    Labels:
    - 1 = UP (price increases beyond threshold)
    - 0 = DOWN (price decreases beyond threshold)
    - -1 = NO TRADE (movement not significant enough)
    """
    df = df.copy()
    
    threshold = config.label_threshold_pct
    df['future_close'] = df['close'].shift(-config.horizon_seconds)
    df['future_return'] = (df['future_close'] - df['close']) / df['close']
    
    conditions = [
        df['future_return'] > threshold,
        df['future_return'] < -threshold
    ]
    choices = [1, 0]
    
    df['label'] = np.select(conditions, choices, default=-1)
    df['target_probability'] = np.where(df['label'] == 1, 1.0, 
                                        np.where(df['label'] == 0, 0.0, np.nan))
    
    return df


def filter_valid_labels(df: pd.DataFrame) -> pd.DataFrame:
    """Remove rows where label is -1 (no meaningful movement)."""
    return df[df['label'] != -1].copy()


def create_all_features(df: pd.DataFrame, config: HorizonConfig) -> pd.DataFrame:
    """Create complete feature set for horizon-aware training."""
    # Basic price features
    df = calculate_returns(df, [1, 3, 5, 10, 15, config.horizon_seconds])
    
    # Technical indicators
    from trading.indicators.technical import add_all_indicators
    df = add_all_indicators(df)
    
    # Position features
    df = calculate_price_position_features(df)
    
    # Microstructure
    df = calculate_microstructure_features(df)
    
    # Apply horizon-specific labeling
    df = apply_label_threshold(df, config)
    
    # Filter out non-significant movements
    df = filter_valid_labels(df)
    
    return df.dropna(subset=[
        'return_1', 'return_3', 'return_5', f'return_{config.horizon_seconds}',
        'ema_9', 'ema_21', 'rsi_14', 'macd_line', 'macd_signal', 'macd_hist',
        'close_position', 'body_size', 'volume_ratio', 'label'
    ])


def get_feature_columns(df: pd.DataFrame, include_price_context: bool = False) -> List[str]:
    """Get list of feature columns for model training."""
    exclude = {'timestamp', 'open', 'high', 'low', 'close', 'volume', 
               'label', 'target_probability', 'future_close', 'future_return',
               'symbol', 'timeframe'}
    
    features = [c for c in df.columns if c not in exclude]
    
    if include_price_context:
        price_cols = ['open', 'high', 'low', 'close']
        features = sorted(set(features + price_cols))
    
    return features


def prepare_horizon_dataset(df: pd.DataFrame, horizon_name: str) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """
    Prepare complete dataset for specific horizon.
    
    Returns:
        df_features: DataFrame with all features
        X: Feature matrix (numpy array)
        y: Label vector (numpy array)
    """
    config = HORIZON_CONFIGS[horizon_name]
    
    # Create features and labels
    df_features = create_all_features(df.copy(), config)
    
    # Validate minimum samples
    if len(df_features) < config.min_training_samples:
        raise ValueError(
            f"Insufficient samples for {horizon_name}: "
            f"{len(df_features)} < {config.min_training_samples}"
        )
    
    # Extract features and labels
    feature_cols = get_feature_columns(df_features)
    X = df_features[feature_cols].values
    y = df_features['label'].values
    
    return df_features, X, y