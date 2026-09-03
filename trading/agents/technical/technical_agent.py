"""Technical Agent for Phase 4 - Multi-Agent Decision System.
Analyzes technical indicators to produce trading direction and confidence.
"""

from dataclasses import dataclass
from enum import Enum
from typing import List, Optional, Dict, Any
import pandas as pd
import numpy as np
from datetime import datetime


class Direction(Enum):
    UP = "UP"
    DOWN = "DOWN"
    NEUTRAL = "NEUTRAL"


@dataclass
class AgentSignal:
    """Standardized signal output from agents."""
    direction: Direction
    confidence: float  # 0.0 to 1.0
    reason: str = ""
    metadata: Dict[str, Any] = None


class TechnicalAgent:
    """
    Technical Analysis Agent
    
    Analyzes:
    - RSI
    - MACD
    - EMA
    - Support/Resistance
    - Candlestick patterns
    
    Output:
    - Direction: UP / DOWN / NEUTRAL
    - Confidence: 0.0 to 1.0
    """
    
    def __init__(self):
        # Strategy weights (can be tuned)
        self.strategy_weights = {
            "EMA_Crossover": 0.30,
            "RSI": 0.25,
            "MACD": 0.25,
            "Support_Resistance": 0.20
        }
    
    def analyze(self, df: pd.DataFrame) -> AgentSignal:
        """Analyze technical indicators and produce signal."""
        if len(df) < 30:
            return AgentSignal(
                direction=Direction.NEUTRAL,
                confidence=0.0,
                reason="Insufficient data for technical analysis"
            )
        
        # Calculate indicators
        df = self._calculate_indicators(df)
        
        # Run individual strategies
        signals = {
            "EMA_Crossover": self._ema_strategy(df),
            "RSI": self._rsi_strategy(df),
            "MACD": self._macd_strategy(df),
            "Support_Resistance": self._sr_strategy(df),
        }
        
        # Aggregate weighted signals
        up_score = sum(
            signals[k].confidence * self.strategy_weights[k]
            for k in signals
            if signals[k].direction == Direction.UP
        )
        down_score = sum(
            signals[k].confidence * self.strategy_weights[k]
            for k in signals
            if signals[k].direction == Direction.DOWN
        )
        
        total_weight = sum(self.strategy_weights.values())
        up_score /= total_weight
        down_score /= total_weight
        
        # Determine final direction
        if up_score > down_score and up_score >= 0.40:
            confidence = min(0.95, up_score * 1.15)
            return AgentSignal(
                direction=Direction.UP,
                confidence=round(confidence, 4),
                reason=f"Technical indicators bullish (score: {up_score:.2f})",
                metadata={k: v.__dict__ for k, v in signals.items()}
            )
        elif down_score > up_score and down_score >= 0.40:
            confidence = min(0.95, down_score * 1.15)
            return AgentSignal(
                direction=Direction.DOWN,
                confidence=round(confidence, 4),
                reason=f"Technical indicators bearish (score: {down_score:.2f})",
                metadata={k: v.__dict__ for k, v in signals.items()}
            )
        else:
            return AgentSignal(
                direction=Direction.NEUTRAL,
                confidence=0.0,
                reason=f"Conflicting signals or insufficient consensus",
                metadata={k: v.__dict__ for k, v in signals.items()}
            )
    
    def _calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate all technical indicators."""
        from trading.indicators.technical import (
            calculate_ema, calculate_rsi, calculate_macd,
            calculate_support_resistance
        )
        
        df = df.copy()
        df['ema_9'] = calculate_ema(df['close'], 9)
        df['ema_21'] = calculate_ema(df['close'], 21)
        df['ema_50'] = calculate_ema(df['close'], 50)
        df['rsi_14'] = calculate_rsi(df['close'], 14)
        df['macd_line'], df['macd_signal'], df['macd_hist'] = calculate_macd(df['close'])
        df['support'], df['resistance'] = calculate_support_resistance(df['high'], df['low'], 20)
        return df
    
    def _ema_strategy(self, df: pd.DataFrame) -> AgentSignal:
        """EMA crossover strategy."""
        idx = -1
        prev = idx - 1
        
        try:
            curr_fast = df["ema_9"].iloc[idx]
            curr_slow = df["ema_21"].iloc[idx]
            prev_fast = df["ema_9"].iloc[prev]
            prev_slow = df["ema_21"].iloc[prev]
            
            if prev_fast <= prev_slow and curr_fast > curr_slow:
                diff = abs(curr_fast - curr_slow) / curr_slow * 1000
                conf = round(min(0.95, 0.70 + diff * 0.1), 2)
                return AgentSignal(
                    direction=Direction.UP,
                    confidence=conf,
                    reason=f"EMA9 ({curr_fast:.2f}) crossed above EMA21 ({curr_slow:.2f})"
                )
            elif prev_fast >= prev_slow and curr_fast < curr_slow:
                diff = abs(curr_fast - curr_slow) / curr_slow * 1000
                conf = round(min(0.95, 0.70 + diff * 0.1), 2)
                return AgentSignal(
                    direction=Direction.DOWN,
                    confidence=conf,
                    reason=f"EMA9 ({curr_fast:.2f}) crossed below EMA21 ({curr_slow:.2f})"
                )
            elif curr_fast > curr_slow:
                return AgentSignal(
                    direction=Direction.UP,
                    confidence=0.60,
                    reason="EMA9 above EMA21 (bullish trend)"
                )
            elif curr_fast < curr_slow:
                return AgentSignal(
                    direction=Direction.DOWN,
                    confidence=0.60,
                    reason="EMA9 below EMA21 (bearish trend)"
                )
        except:
            pass
        
        return AgentSignal(
            direction=Direction.NEUTRAL,
            confidence=0.0,
            reason="No clear EMA signal"
        )
    
    def _rsi_strategy(self, df: pd.DataFrame) -> AgentSignal:
        """RSI oscillator strategy."""
        try:
            rsi_val = df["rsi_14"].iloc[-1]
            
            if rsi_val < 30:
                conf = round(min(0.95, 0.70 + ((30 - rsi_val) / 30.0) * 0.25), 2)
                return AgentSignal(
                    direction=Direction.UP,
                    confidence=conf,
                    reason=f"RSI ({rsi_val:.1f}) oversold (<30)"
                )
            elif rsi_val > 70:
                conf = round(min(0.95, 0.70 + ((rsi_val - 70) / 30.0) * 0.25), 2)
                return AgentSignal(
                    direction=Direction.DOWN,
                    confidence=conf,
                    reason=f"RSI ({rsi_val:.1f}) overbought (>70)"
                )
        except:
            pass
        
        return AgentSignal(
            direction=Direction.NEUTRAL,
            confidence=0.0,
            reason=f"RSI ({df['rsi_14'].iloc[-1]:.1f}) in neutral zone"
        )
    
    def _macd_strategy(self, df: pd.DataFrame) -> AgentSignal:
        """MACD momentum strategy."""
        try:
            idx = -1
            prev = idx - 1
            
            curr_macd = df["macd_line"].iloc[idx]
            curr_sig = df["macd_signal"].iloc[idx]
            curr_hist = df["macd_hist"].iloc[idx]
            
            prev_macd = df["macd_line"].iloc[prev]
            prev_sig = df["macd_signal"].iloc[prev]
            
            if prev_macd <= prev_sig and curr_macd > curr_sig:
                return AgentSignal(
                    direction=Direction.UP,
                    confidence=0.75,
                    reason=f"MACD ({curr_macd:.4f}) crossed above signal ({curr_sig:.4f})"
                )
            elif prev_macd >= prev_sig and curr_macd < curr_sig:
                return AgentSignal(
                    direction=Direction.DOWN,
                    confidence=0.75,
                    reason=f"MACD ({curr_macd:.4f}) crossed below signal ({curr_sig:.4f})"
                )
            elif curr_hist > 0:
                return AgentSignal(
                    direction=Direction.UP,
                    confidence=0.55,
                    reason=f"MACD histogram positive ({curr_hist:.4f})"
                )
            elif curr_hist < 0:
                return AgentSignal(
                    direction=Direction.DOWN,
                    confidence=0.55,
                    reason=f"MACD histogram negative ({curr_hist:.4f})"
                )
        except:
            pass
        
        return AgentSignal(
            direction=Direction.NEUTRAL,
            confidence=0.0,
            reason="No clear MACD signal"
        )
    
    def _sr_strategy(self, df: pd.DataFrame) -> AgentSignal:
        """Support/Resistance strategy."""
        try:
            idx = -1
            curr_price = df["close"].iloc[idx]
            support = df["support"].iloc[idx]
            resistance = df["resistance"].iloc[idx]
            
            price_range = max(0.0001, resistance - support)
            dist_to_support = abs(curr_price - support) / price_range
            dist_to_resistance = abs(resistance - curr_price) / price_range
            
            if dist_to_support <= 0.35:
                confidence = round(min(0.90, 0.70 + (1.0 - dist_to_support / 35.0) * 0.20), 2)
                return AgentSignal(
                    direction=Direction.UP,
                    confidence=confidence,
                    reason=f"Price ({curr_price:.2f}) near support ({support:.2f})"
                )
            elif dist_to_resistance <= 0.35:
                confidence = round(min(0.90, 0.70 + (1.0 - dist_to_resistance / 35.0) * 0.20), 2)
                return AgentSignal(
                    direction=Direction.DOWN,
                    confidence=confidence,
                    reason=f"Price ({curr_price:.2f}) near resistance ({resistance:.2f})"
                )
        except:
            pass
        
        return AgentSignal(
            direction=Direction.NEUTRAL,
            confidence=0.0,
            reason=f"Price in mid-range between support/resistance"
        )


# Global agent instance
technical_agent = TechnicalAgent()