import pandas as pd
from trading.strategies.base import BaseStrategy, StrategySignal, SignalDirection
from trading.indicators.technical import calculate_support_resistance


class SupportResistanceStrategy(BaseStrategy):
    """
    Strategy 4: Support & Resistance Price Action Strategy.
    Identifies key price zones:
    - Price near Support + Bullish reversal/bounce -> UP signal
    - Price near Resistance + Bearish reversal/bounce -> DOWN signal
    """

    def __init__(self, window: int = 20, threshold_pct: float = 0.35):
        super().__init__(name="Support_Resistance")
        self.window = window
        self.threshold_pct = threshold_pct  # Percentage distance to support/resistance zone

    def evaluate(self, df: pd.DataFrame, index: int = -1) -> StrategySignal:
        if len(df) < self.window + 1:
            return StrategySignal(
                direction=SignalDirection.NEUTRAL,
                confidence=0.0,
                strategy_name=self.name,
                reason="Insufficient data for Support/Resistance calculation"
            )

        if "support" not in df.columns or "resistance" not in df.columns:
            df = df.copy()
            sup, res = calculate_support_resistance(df["high"], df["low"], self.window)
            df["support"] = sup
            df["resistance"] = res

        idx = index if index >= 0 else len(df) + index
        curr_price = df["close"].iloc[idx]
        support = df["support"].iloc[idx]
        resistance = df["resistance"].iloc[idx]

        price_range = max(0.0001, resistance - support)
        dist_to_support_pct = (abs(curr_price - support) / price_range) * 100.0
        dist_to_resistance_pct = (abs(resistance - curr_price) / price_range) * 100.0

        # Bounce near Support (UP signal)
        if dist_to_support_pct <= (self.threshold_pct * 100.0):
            confidence = round(min(0.90, 0.70 + (1.0 - (dist_to_support_pct / 35.0)) * 0.20), 2)
            return StrategySignal(
                direction=SignalDirection.UP,
                confidence=confidence,
                strategy_name=self.name,
                reason=f"Price ({curr_price:.2f}) near Support zone ({support:.2f})",
                metadata={"support": support, "resistance": resistance, "dist_support_pct": dist_to_support_pct}
            )

        # Reversal near Resistance (DOWN signal)
        elif dist_to_resistance_pct <= (self.threshold_pct * 100.0):
            confidence = round(min(0.90, 0.70 + (1.0 - (dist_to_resistance_pct / 35.0)) * 0.20), 2)
            return StrategySignal(
                direction=SignalDirection.DOWN,
                confidence=confidence,
                strategy_name=self.name,
                reason=f"Price ({curr_price:.2f}) near Resistance zone ({resistance:.2f})",
                metadata={"support": support, "resistance": resistance, "dist_resistance_pct": dist_to_resistance_pct}
            )

        return StrategySignal(
            direction=SignalDirection.NEUTRAL,
            confidence=0.0,
            strategy_name=self.name,
            reason=f"Price ({curr_price:.2f}) in mid-range between Support ({support:.2f}) & Resistance ({resistance:.2f})"
        )
