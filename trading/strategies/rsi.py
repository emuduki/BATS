import pandas as pd
from trading.strategies.base import BaseStrategy, StrategySignal, SignalDirection
from trading.indicators.technical import calculate_rsi


class RSIStrategy(BaseStrategy):
    """
    Strategy 2: RSI Strategy.
    - RSI < 30 -> Oversold -> Potential UP signal
    - RSI > 70 -> Overbought -> Potential DOWN signal
    """

    def __init__(self, period: int = 14, oversold: float = 30.0, overbought: float = 70.0):
        super().__init__(name="RSI")
        self.period = period
        self.oversold = oversold
        self.overbought = overbought

    def evaluate(self, df: pd.DataFrame, index: int = -1) -> StrategySignal:
        if len(df) < self.period + 1:
            return StrategySignal(
                direction=SignalDirection.NEUTRAL,
                confidence=0.0,
                strategy_name=self.name,
                reason="Insufficient data for RSI calculation"
            )

        if "rsi_14" not in df.columns:
            df = df.copy()
            df["rsi_14"] = calculate_rsi(df["close"], self.period)

        idx = index if index >= 0 else len(df) + index
        rsi_val = df["rsi_14"].iloc[idx]

        if rsi_val < self.oversold:
            # More extreme oversold = higher confidence
            confidence = round(min(0.95, 0.70 + ((self.oversold - rsi_val) / 30.0) * 0.25), 2)
            return StrategySignal(
                direction=SignalDirection.UP,
                confidence=confidence,
                strategy_name=self.name,
                reason=f"RSI ({rsi_val:.1f}) is oversold (< {self.oversold})",
                metadata={"rsi_14": rsi_val}
            )

        elif rsi_val > self.overbought:
            # More extreme overbought = higher confidence
            confidence = round(min(0.95, 0.70 + ((rsi_val - self.overbought) / 30.0) * 0.25), 2)
            return StrategySignal(
                direction=SignalDirection.DOWN,
                confidence=confidence,
                strategy_name=self.name,
                reason=f"RSI ({rsi_val:.1f}) is overbought (> {self.overbought})",
                metadata={"rsi_14": rsi_val}
            )

        return StrategySignal(
            direction=SignalDirection.NEUTRAL,
            confidence=0.0,
            strategy_name=self.name,
            reason=f"RSI ({rsi_val:.1f}) in neutral zone (30 - 70)",
            metadata={"rsi_14": rsi_val}
        )
