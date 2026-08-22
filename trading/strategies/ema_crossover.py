import pandas as pd
from trading.strategies.base import BaseStrategy, StrategySignal, SignalDirection
from trading.indicators.technical import calculate_ema


class EMACrossoverStrategy(BaseStrategy):
    """
    Strategy 1: EMA Crossover.
    Fast EMA: 9, Slow EMA: 21
    - EMA 9 crosses above EMA 21 -> Potential UP signal
    - EMA 9 crosses below EMA 21 -> Potential DOWN signal
    """

    def __init__(self, fast_period: int = 9, slow_period: int = 21):
        super().__init__(name="EMA_Crossover")
        self.fast_period = fast_period
        self.slow_period = slow_period

    def evaluate(self, df: pd.DataFrame, index: int = -1) -> StrategySignal:
        if len(df) < self.slow_period + 2:
            return StrategySignal(
                direction=SignalDirection.NEUTRAL,
                confidence=0.0,
                strategy_name=self.name,
                reason="Insufficient historical data for EMA calculation"
            )

        # Compute EMAs if missing
        if "ema_9" not in df.columns:
            df = df.copy()
            df["ema_9"] = calculate_ema(df["close"], self.fast_period)
            df["ema_21"] = calculate_ema(df["close"], self.slow_period)

        idx = index if index >= 0 else len(df) + index
        prev_idx = idx - 1

        curr_fast = df["ema_9"].iloc[idx]
        curr_slow = df["ema_21"].iloc[idx]
        prev_fast = df["ema_9"].iloc[prev_idx]
        prev_slow = df["ema_21"].iloc[prev_idx]

        # Bullish Crossover
        if prev_fast <= prev_slow and curr_fast > curr_slow:
            diff_ratio = min(1.0, abs(curr_fast - curr_slow) / curr_slow * 1000)
            confidence = round(min(0.95, 0.70 + (diff_ratio * 0.1)), 2)
            return StrategySignal(
                direction=SignalDirection.UP,
                confidence=confidence,
                strategy_name=self.name,
                reason=f"EMA 9 ({curr_fast:.2f}) crossed above EMA 21 ({curr_slow:.2f})",
                metadata={"ema_9": curr_fast, "ema_21": curr_slow}
            )

        # Bearish Crossover
        elif prev_fast >= prev_slow and curr_fast < curr_slow:
            diff_ratio = min(1.0, abs(curr_fast - curr_slow) / curr_slow * 1000)
            confidence = round(min(0.95, 0.70 + (diff_ratio * 0.1)), 2)
            return StrategySignal(
                direction=SignalDirection.DOWN,
                confidence=confidence,
                strategy_name=self.name,
                reason=f"EMA 9 ({curr_fast:.2f}) crossed below EMA 21 ({curr_slow:.2f})",
                metadata={"ema_9": curr_fast, "ema_21": curr_slow}
            )

        # Continuation trend
        elif curr_fast > curr_slow:
            return StrategySignal(
                direction=SignalDirection.UP,
                confidence=0.60,
                strategy_name=self.name,
                reason="EMA 9 is above EMA 21 (Bullish trend)",
                metadata={"ema_9": curr_fast, "ema_21": curr_slow}
            )

        elif curr_fast < curr_slow:
            return StrategySignal(
                direction=SignalDirection.DOWN,
                confidence=0.60,
                strategy_name=self.name,
                reason="EMA 9 is below EMA 21 (Bearish trend)",
                metadata={"ema_9": curr_fast, "ema_21": curr_slow}
            )

        return StrategySignal(
            direction=SignalDirection.NEUTRAL,
            confidence=0.0,
            strategy_name=self.name,
            reason="EMA lines are equal"
        )
