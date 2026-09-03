"""Duration Agent for Phase 4 - Multi-Agent Decision System.
Decides appropriate trade duration based on market regime and strategy performance.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Any
import pandas as pd
from datetime import datetime

from trading.agents.regime.regime_agent import RegimeSignal, MarketRegime


@dataclass
class DurationRecommendation:
    """Duration recommendation output."""
    duration_seconds: int
    duration_label: str
    reason: str
    confidence: float
    metadata: Dict[str, Any] = None


class DurationAgent:
    """
    Duration Selection Agent
    
    This agent decides which supported contract duration is most appropriate.
    
    Decision factors:
    - Market volatility (high vol -> short duration, low vol -> long duration)
    - Trend strength (strong trend -> longer duration, weak trend -> shorter)
    - Historical strategy performance for each duration
    - Time of day effects
    """
    
    # Supported durations (must match contract availability)
    SUPPORTED_DURATIONS = [
        {"seconds": 30, "label": "30s"},
        {"seconds": 60, "label": "60s"},
        {"seconds": 120, "label": "120s"},
        {"seconds": 300, "label": "5m"},
    ]
    
    def __init__(self):
        # Historical performance lookup (would come from backtest results)
        self.historical_performance: Dict[str, Dict[int, float]] = {}
        self.default_durations = self.SUPPORTED_DURATIONS.copy()
    
    def recommend(self, regime_signal: RegimeSignal, strategy_name: Optional[str] = None) -> DurationRecommendation:
        """
        Recommend trade duration based on market conditions.
        """
        regime = regime_signal.regime
        trend_strength = regime_signal.trend_strength
        volatility_pct = regime_signal.volatility_score
        trend_direction = regime_signal.trend_direction
        
        # Rule-based duration selection
        duration_choice = self._rule_based_selection(regime, trend_strength, volatility_pct)
        
        # Adjust based on historical performance if available
        if strategy_name and strategy_name in self.historical_performance:
            duration_choice = self._adjust_for_performance(
                duration_choice, strategy_name, regime
            )
        
        # Calculate confidence
        confidence = self._calculate_confidence(regime, volatility_pct, trend_strength)
        
        duration_label = next(
            (d["label"] for d in self.SUPPORTED_DURATIONS if d["seconds"] == duration_choice),
            f"{duration_choice}s"
        )
        
        reason = self._generate_reason(regime, volatility_pct, trend_strength, duration_choice)
        
        return DurationRecommendation(
            duration_seconds=duration_choice,
            duration_label=duration_label,
            reason=reason,
            confidence=round(confidence, 4),
            metadata={
                "regime": regime.value,
                "trend_strength": trend_strength,
                "volatility": volatility_pct,
                "trend_direction": trend_direction,
            }
        )
    
    def _rule_based_selection(self, regime: MarketRegime, trend_strength: float, volatility_pct: float) -> int:
        """
        Select optimal duration using rule-based system.
        """
        # High volatility -> shorter durations due to noise
        if volatility_pct >= 0.70:
            if trend_strength >= 0.6:
                return 60  # Strong trend with high vol -> medium-term
            else:
                return 30  # Weak trend, high vol -> quick exits
        
        # Low volatility -> longer durations
        if volatility_pct <= 0.30:
            if trend_strength >= 0.5:
                return 300  # Strong trend, low vol -> ride the wave
            else:
                return 120  # Weak trend, low vol -> medium patience
        
        # Medium volatility -> standard durations
        if trend_strength >= 0.6:
            return 120  # Strong trend -> longer duration
        elif trend_strength >= 0.4:
            return 60   # Moderate trend -> standard
        else:
            return 30   # Weak/no trend -> quick exits
    
    def _adjust_for_performance(self, duration: int, strategy: str, regime: MarketRegime) -> int:
        """
        Adjust duration based on historical strategy performance.
        """
        # This would look up actual historical results
        # For now, return unchanged
        return duration
    
    def _calculate_confidence(self, regime: MarketRegime, volatility_pct: float, trend_strength: float) -> float:
        """
        Calculate confidence in duration recommendation.
        """
        # Higher confidence in clear market conditions
        vol_conf = 1.0 - abs(volatility_pct - 0.5) * 2  # Peak confidence at 50th percentile
        trend_conf = 1.0 if trend_strength > 0.3 else trend_strength / 0.3
        
        return min(0.95, vol_conf * 0.4 + trend_conf * 0.6)
    
    def _generate_reason(self, regime: MarketRegime, vol_pct: float, trend_strength: float, duration_sec: int) -> str:
        """Generate human-readable explanation."""
        vol_adj = "High" if vol_pct >= 0.7 else "Low" if vol_pct <= 0.3 else "Medium"
        trend_adj = ""
        if trend_strength >= 0.6:
            trend_adj = "Strong"
        elif trend_strength >= 0.4:
            trend_adj = "Moderate"
        else:
            trend_adj = "Weak"
            
        return f"{vol_adj} volatility ({vol_pct:.1%}) with {trend_adj.lower()} trend ({trend_strength:.1%}) → {duration_sec}s expiry optimal for this regime."


# Global agent instance
duration_agent = DurationAgent()