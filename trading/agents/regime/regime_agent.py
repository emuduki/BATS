"""Market Regime Agent for Phase 4 - Multi-Agent Decision System.
Determines market regime: trending, ranging, high/low volatility.
"""

from dataclasses import dataclass
from enum import Enum
from typing import Dict, Any
import pandas as pd
import numpy as np


class MarketRegime(Enum):
    TRENDING_UP = "TRENDING_UP"
    TRENDING_DOWN = "TRENDING_DOWN"
    RANGING = "RANGING"
    HIGH_VOLATILITY = "HIGH_VOLATILITY"
    LOW_VOLATILITY = "LOW_VOLATILITY"
    NEUTRAL = "NEUTRAL"


@dataclass
class RegimeSignal:
    """Regime detection output."""
    regime: MarketRegime
    trend_direction: str  # "UP", "DOWN", "NEUTRAL"
    trend_strength: float  # 0.0 to 1.0
    volatility_level: str  # "HIGH", "MEDIUM", "LOW"
    volatility_score: float  # 0.0 to 1.0
    confidence: float  # 0.0 to 1.0
    metadata: Dict[str, Any] = None


class RegimeAgent:
    """
    Market Regime Detection Agent
    
    Determines:
    - Trending / Ranging
    - High / Low Volatility
    - Trend direction and strength
    
    This is critical because strategies perform differently in different regimes.
    """
    
    def __init__(self):
        # Volatility thresholds (percentile-based)
        self.high_vol_threshold = 0.70
        self.low_vol_threshold = 0.30
        
        # Trend strength thresholds
        self.strong_trend_threshold = 0.60
        self.weak_trend_threshold = 0.30
    
    def analyze(self, df: pd.DataFrame) -> RegimeSignal:
        """Analyze market regime."""
        if len(df) < 30:
            return RegimeSignal(
                regime=MarketRegime.NEUTRAL,
                trend_direction="NEUTRAL",
                trend_strength=0.0,
                volatility_level="MEDIUM",
                volatility_score=0.5,
                confidence=0.0,
                metadata={"reason": "Insufficient data"}
            )
        
        # Calculate metrics
        returns = df['close'].pct_change().dropna()
        volatility = returns.rolling(20).std().iloc[-1]
        
        # Trend detection
        trend_strength, trend_direction = self._detect_trend(df)
        
        # Volatility analysis
        vol_percentile = self._estimate_volatility_percentile(volatility, df)
        
        # Determine regime
        regime = self._determine_regime(trend_strength, trend_direction, vol_percentile)
        
        # Confidence calculation
        confidence = self._calculate_confidence(trend_strength, vol_percentile)
        
        return RegimeSignal(
            regime=regime,
            trend_direction=trend_direction,
            trend_strength=round(trend_strength, 4),
            volatility_level=self._volatility_level(vol_percentile),
            volatility_score=round(vol_percentile, 4),
            confidence=round(confidence, 4),
            metadata={
                "volatility": round(volatility, 6),
                "trend_threshold": self.strong_trend_threshold,
                "vol_percentile": round(vol_percentile, 4)
            }
        )
    
    def _detect_trend(self, df: pd.DataFrame) -> tuple:
        """Detect trend direction and strength."""
        try:
            # Use EMA alignment for trend detection
            ema_9 = df['close'].ewm(span=9, adjust=False).mean()
            ema_21 = df['close'].ewm(span=21, adjust=False).mean()
            ema_50 = df['close'].ewm(span=50, adjust=False).mean()
            
            curr_9 = ema_9.iloc[-1]
            curr_21 = ema_21.iloc[-1]
            curr_50 = ema_50.iloc[-1]
            
            # Trend direction
            if curr_9 > curr_21 > curr_50:
                direction = "UP"
            elif curr_9 < curr_21 < curr_50:
                direction = "DOWN"
            else:
                direction = "NEUTRAL"
            
            # Trend strength based on EMA separation
            if direction == "UP":
                strength = min(1.0, (curr_9 - curr_50) / abs(curr_50) * 100)
            elif direction == "DOWN":
                strength = min(1.0, (curr_50 - curr_9) / abs(curr_50) * 100)
            else:
                strength = 0.0
            
            return strength, direction
            
        except Exception:
            return 0.0, "NEUTRAL"
    
    def _estimate_volatility_percentile(self, current_vol: float, df: pd.DataFrame) -> float:
        """Estimate volatility percentile relative to historical."""
        try:
            returns = df['close'].pct_change().dropna()
            historical_vol = returns.rolling(20).std().dropna()
            
            if len(historical_vol) < 20:
                return 0.5
            
            # Calculate percentile rank
            vol_series = pd.Series([current_vol] + historical_vol.tolist())
            percentile = (vol_series.rank().iloc[0] - 1) / (len(vol_series) - 1)
            
            return min(1.0, max(0.0, percentile))
            
        except Exception:
            return 0.5
    
    def _volatility_level(self, percentile: float) -> str:
        """Convert percentile to volatility level."""
        if percentile >= self.high_vol_threshold:
            return "HIGH"
        elif percentile <= self.low_vol_threshold:
            return "LOW"
        else:
            return "MEDIUM"
    
    def _determine_regime(self, trend_strength: float, trend_direction: str, vol_percentile: float) -> MarketRegime:
        """Determine primary market regime."""
        # High volatility overrides trend
        if vol_percentile >= self.high_vol_threshold:
            if trend_strength >= self.strong_trend_threshold:
                if trend_direction == "UP":
                    return MarketRegime.TRENDING_UP
                else:
                    return MarketRegime.TRENDING_DOWN
            return MarketRegime.HIGH_VOLATILITY
        
        # Low volatility
        if vol_percentile <= self.low_vol_threshold:
            if trend_strength >= self.weak_trend_threshold:
                if trend_direction == "UP":
                    return MarketRegime.TRENDING_UP
                else:
                    return MarketRegime.TRENDING_DOWN
            return MarketRegime.LOW_VOLATILITY
        
        # Medium volatility with trend
        if trend_strength >= self.strong_trend_threshold:
            if trend_direction == "UP":
                return MarketRegime.TRENDING_UP
            else:
                return MarketRegime.TRENDING_DOWN
        
        # Ranging market
        return MarketRegime.RANGING
    
    def _calculate_confidence(self, trend_strength: float, vol_percentile: float) -> float:
        """Calculate confidence in regime detection."""
        # Higher confidence when trend is strong and volatility is clear
        trend_conf = min(1.0, trend_strength / self.strong_trend_threshold)
        vol_conf = min(1.0, max(vol_percentile, 1 - vol_percentile) / 0.5)
        
        return round((trend_conf * 0.6 + vol_conf * 0.4), 4)


# Global agent instance
regime_agent = RegimeAgent()