import pandas as pd
from trading.strategies.base import BaseStrategy, StrategySignal, SignalDirection
from trading.indicators.technical import calculate_macd


class MACDStrategy(BaseStrategy):
    """
    Strategy 3: MACD Momentum Strategy.
    Detects momentum shifts using MACD line, signal line, and histogram.
    """

    def __init__(self, fast_period: int = 12, slow_period: int = 26, signal_period: int = 9):
        super().__init__(name="MACD")
        self.fast_period = fast_period
        self.slow_period = slow_period
        self.signal_period = signal_period

    def evaluate(self, df: pd.DataFrame, index: int = -1) -> StrategySignal:
        if len(df) < self.slow_period + self.signal_period:
            return StrategySignal(
                direction=SignalDirection.NEUTRAL,
                confidence=0.0,
                strategy_name=self.name,
                reason="Insufficient data for MACD calculation"
            )

        if "macd_line" not in df.columns or "macd_signal" not in df.columns:
            df = df.copy()
            macd, signal, hist = calculate_macd(df["close"], self.fast_period, self.slow_period, self.signal_period)
            df["macd_line"] = macd
            df["macd_signal"] = signal
            df["macd_hist"] = hist

        idx = index if index >= 0 else len(df) + index
        prev_idx = idx - 1

        curr_macd = df["macd_line"].iloc[idx]
        curr_sig = df["macd_signal"].iloc[idx]
        curr_hist = df["macd_hist"].iloc[idx]

        prev_macd = df["macd_line"].iloc[prev_idx]
        prev_sig = df["macd_signal"].iloc[prev_idx]

        # Bullish MACD Crossover
        if prev_macd <= prev_sig and curr_macd > curr_sig:
            return StrategySignal(
                direction=SignalDirection.UP,
                confidence=0.75,
                strategy_name=self.name,
                reason=f"MACD line ({curr_macd:.4f}) crossed above Signal ({curr_sig:.4f})",
                metadata={"macd": curr_macd, "signal": curr_sig, "hist": curr_hist}
            )

        # Bearish MACD Crossover
        elif prev_macd >= prev_sig and curr_macd < curr_sig:
            return StrategySignal(
                direction=SignalDirection.DOWN,
                confidence=0.75,
                strategy_name=self.name,
                reason=f"MACD line ({curr_macd:.4f}) crossed below Signal ({curr_sig:.4f})",
                metadata={"macd": curr_macd, "signal": curr_sig, "hist": curr_hist}
            )

        # Momentum Continuation
        elif curr_hist > 0:
            return StrategySignal(
                direction=SignalDirection.UP,
                confidence=0.55,
                strategy_name=self.name,
                reason=f"MACD Histogram is positive (+{curr_hist:.4f})",
                metadata={"macd": curr_macd, "signal": curr_sig, "hist": curr_hist}
            )

        elif curr_hist < 0:
            return StrategySignal(
                direction=SignalDirection.DOWN,
                confidence=0.55,
                strategy_name=self.name,
                reason=f"MACD Histogram is negative ({curr_hist:.4f})",
                metadata={"macd": curr_macd, "signal": curr_sig, "hist": curr_hist}
            )

        return StrategySignal(
            direction=SignalDirection.NEUTRAL,
            confidence=0.0,
            strategy_name=self.name,
            reason="MACD histogram is zero"
        )
