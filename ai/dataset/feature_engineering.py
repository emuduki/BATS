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


def calculate_horizon_shift(df: pd.DataFrame, horizon_seconds: int) -> int:
    """Calculate candle shift steps corresponding to the horizon."""
    if 'timestamp' in df.columns and len(df) > 1:
        try:
            ts = pd.to_datetime(df['timestamp'])
            dt = (ts.diff().dt.total_seconds()).median()
            if dt and dt > 0:
                steps = max(1, int(round(horizon_seconds / dt)))
                return steps
        except Exception:
            pass
    # If 1m data or undefined, 15-60s corresponds to 1 candle ahead
    if horizon_seconds <= 60:
        return 1
    return max(1, horizon_seconds // 60)


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
    shift_steps = calculate_horizon_shift(df, config.horizon_seconds)
    df['future_close'] = df['close'].shift(-shift_steps)
    df['future_return'] = (df['future_close'] - df['close']) / (df['close'] + 1e-10)
    
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


def create_feature_indicators(df: pd.DataFrame, config: Optional[HorizonConfig] = None) -> pd.DataFrame:
    """Calculate indicators and technical features without labels."""
    df = df.copy()
    
    # Basic price features
    df = calculate_returns(df, [1, 3, 5, 10, 15])
    
    # Generate return columns for all standard horizons so any horizon model finds its column
    from ai.datasets.horizon_configs import HORIZON_CONFIGS
    for h_name, h_cfg in HORIZON_CONFIGS.items():
        shift_steps = calculate_horizon_shift(df, h_cfg.horizon_seconds)
        df[f'return_h{h_cfg.horizon_seconds}'] = df['close'].pct_change(shift_steps)
    
    # Technical indicators
    from trading.indicators.technical import add_all_indicators
    df = add_all_indicators(df)
    
    # Position features
    df = calculate_price_position_features(df)
    
    # Microstructure
    df = calculate_microstructure_features(df)
    
    return df


def create_all_features(df: pd.DataFrame, config: HorizonConfig) -> pd.DataFrame:
    """Create complete feature set with labels for horizon-aware training."""
    df = create_feature_indicators(df, config)
    
    # Apply horizon-specific labeling
    df = apply_label_threshold(df, config)
    
    # Filter out non-significant movements
    df = filter_valid_labels(df)
    
    core_cols = [
        'return_1', 'return_3', 'return_5',
        'ema_9', 'ema_21', 'rsi_14', 'macd_line', 'macd_signal', 'macd_hist',
        'close_position', 'body_size', 'volume_ratio', 'label'
    ]
    avail_cols = [c for c in core_cols if c in df.columns]
    df = df.dropna(subset=avail_cols)
    return df


def get_feature_columns(df: pd.DataFrame, include_price_context: bool = False) -> List[str]:
    """Get list of feature columns for model training in deterministic order."""
    exclude = {'timestamp', 'open', 'high', 'low', 'close', 'volume', 
               'label', 'target_probability', 'future_close', 'future_return',
               'symbol', 'timeframe'}
    
    features = sorted([c for c in df.columns if c not in exclude])
    
    if include_price_context:
        price_cols = ['open', 'high', 'low', 'close']
        features = sorted(set(features + price_cols))
    
    return features


def create_inference_features(df: pd.DataFrame, config: Optional[HorizonConfig] = None, feature_columns: Optional[List[str]] = None) -> Tuple[np.ndarray, List[str]]:
    """
    Prepare feature vector for real-time inference (no future labels required).
    
    Returns:
        X: 2D numpy array shaped (1, n_features) for the latest candle.
        feature_cols: list of column names.
    """
    df_features = create_feature_indicators(df, config)
    
    if feature_columns:
        cols = list(feature_columns)
        for c in cols:
            if c not in df_features.columns:
                df_features[c] = 0.0
    else:
        cols = get_feature_columns(df_features)
    
    # Forward fill then backward fill any indicator warm-up NaNs
    df_features[cols] = df_features[cols].ffill().bfill().fillna(0.0)
    
    latest_row = df_features[cols].iloc[-1:].values.astype(np.float32)
    return latest_row, cols


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