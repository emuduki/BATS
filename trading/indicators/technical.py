import numpy as np
import pandas as pd
from typing import Dict, Tuple, List


def calculate_ema(series: pd.Series, period: int) -> pd.Series:
    """Calculates Exponential Moving Average (EMA)."""
    return series.ewm(span=period, adjust=False).mean()


def calculate_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    """Calculates Relative Strength Index (RSI)."""
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()

    # Fallback to exponential moving average smoothing if rolling has NaNs
    if gain.isna().all():
        gain = delta.clip(lower=0).ewm(alpha=1/period, adjust=False).mean()
        loss = (-delta.clip(upper=0)).ewm(alpha=1/period, adjust=False).mean()

    rs = gain / loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    return rsi.fillna(50.0)


def calculate_macd(
    series: pd.Series,
    fast_period: int = 12,
    slow_period: int = 26,
    signal_period: int = 9
) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """
    Calculates Moving Average Convergence Divergence (MACD).
    Returns (macd_line, signal_line, histogram).
    """
    fast_ema = calculate_ema(series, fast_period)
    slow_ema = calculate_ema(series, slow_period)
    macd_line = fast_ema - slow_ema
    signal_line = calculate_ema(macd_line, signal_period)
    histogram = macd_line - signal_line
    return macd_line, signal_line, histogram


def calculate_support_resistance(
    high_series: pd.Series,
    low_series: pd.Series,
    window: int = 20
) -> Tuple[pd.Series, pd.Series]:
    """
    Calculates dynamic Support (local min) and Resistance (local max) levels over a rolling window.
    """
    resistance = high_series.rolling(window=window).max()
    support = low_series.rolling(window=window).min()
    return support.bfill(), resistance.bfill()


def add_all_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Enriches an OHLC DataFrame with standard technical indicators:
    - ema_9, ema_21
    - rsi_14
    - macd_line, macd_signal, macd_hist
    - support, resistance
    """
    df = df.copy()

    # Ensure required columns exist
    for col in ["open", "high", "low", "close"]:
        if col not in df.columns:
            raise ValueError(f"DataFrame missing required column: {col}")

    # EMAs
    df["ema_9"] = calculate_ema(df["close"], 9)
    df["ema_21"] = calculate_ema(df["close"], 21)

    # RSI
    df["rsi_14"] = calculate_rsi(df["close"], 14)

    # MACD
    macd, signal, hist = calculate_macd(df["close"], 12, 26, 9)
    df["macd_line"] = macd
    df["macd_signal"] = signal
    df["macd_hist"] = hist

    # Support & Resistance
    support, resistance = calculate_support_resistance(df["high"], df["low"], 20)
    df["support"] = support
    df["resistance"] = resistance

    return df
