from typing import List, Dict, Any
import pandas as pd

from trading.strategies.base import BaseStrategy, StrategySignal, SignalDirection
from trading.strategies.ema_crossover import EMACrossoverStrategy
from trading.strategies.rsi import RSIStrategy
from trading.strategies.macd import MACDStrategy
from trading.strategies.support_resistance import SupportResistanceStrategy
from trading.indicators.technical import add_all_indicators


class CombinedConsensusEngine(BaseStrategy):
    """
    Strategy 5: Multi-Strategy Consensus Engine.
    Aggregates signals from EMA Crossover, RSI, MACD, and Support/Resistance strategies,
    weighing consensus alignment to output final Signal direction and Confidence Score.
    """

    def __init__(self):
        super().__init__(name="Combined_Consensus")
        self.strategies: List[BaseStrategy] = [
            EMACrossoverStrategy(fast_period=9, slow_period=21),
            RSIStrategy(period=14, oversold=30.0, overbought=70.0),
            MACDStrategy(fast_period=12, slow_period=26, signal_period=9),
            SupportResistanceStrategy(window=20, threshold_pct=0.35)
        ]
        # Weights assigned to each strategy component
        self.weights = {
            "EMA_Crossover": 0.30,
            "RSI": 0.25,
            "MACD": 0.25,
            "Support_Resistance": 0.20
        }

    def evaluate(self, df: pd.DataFrame, index: int = -1) -> StrategySignal:
        if len(df) < 30:
            return StrategySignal(
                direction=SignalDirection.NEUTRAL,
                confidence=0.0,
                strategy_name=self.name,
                reason="Insufficient historical candles for multi-strategy consensus evaluation"
            )

        # Enrich DataFrame with indicators if not present
        if "ema_9" not in df.columns:
            df = add_all_indicators(df)

        up_score = 0.0
        down_score = 0.0
        total_weight = sum(self.weights.values())

        breakdown: Dict[str, str] = {}
        signals: List[StrategySignal] = []

        for strat in self.strategies:
            sig = strat.evaluate(df, index=index)
            signals.append(sig)
            w = self.weights.get(strat.name, 0.25)

            if sig.direction == SignalDirection.UP:
                up_score += w * sig.confidence
                breakdown[strat.name] = "UP"
            elif sig.direction == SignalDirection.DOWN:
                down_score += w * sig.confidence
                breakdown[strat.name] = "DOWN"
            else:
                breakdown[strat.name] = "NEUTRAL"

        normalized_up = up_score / total_weight
        normalized_down = down_score / total_weight

        # Consensus Thresholds
        if normalized_up > normalized_down and normalized_up >= 0.40:
            final_direction = SignalDirection.UP
            raw_confidence = min(0.98, normalized_up * 1.15)
            confidence_pct = round(raw_confidence, 2)
            reason = f"Bullish Consensus: {breakdown}"
        elif normalized_down > normalized_up and normalized_down >= 0.40:
            final_direction = SignalDirection.DOWN
            raw_confidence = min(0.98, normalized_down * 1.15)
            confidence_pct = round(raw_confidence, 2)
            reason = f"Bearish Consensus: {breakdown}"
        else:
            final_direction = SignalDirection.NEUTRAL
            confidence_pct = 0.0
            reason = f"Neutral / Conflict Consensus: {breakdown}"

        return StrategySignal(
            direction=final_direction,
            confidence=confidence_pct,
            strategy_name=self.name,
            reason=reason,
            metadata={
                "breakdown": breakdown,
                "up_score": round(normalized_up, 2),
                "down_score": round(normalized_down, 2)
            }
        )


# Global combined strategy instance
combined_strategy_engine = CombinedConsensusEngine()
