"""Decision Agent for Phase 4 - Multi-Agent Decision System.
Combines signals from all agents to make final trading decisions.
"""

from dataclasses import dataclass
from enum import Enum
from typing import Dict, Any, List, Optional
import numpy as np
from datetime import datetime

from trading.agents.technical.technical_agent import AgentSignal as TechnicalSignal
from trading.agents.regime.regime_agent import RegimeSignal
from trading.agents.ai.ai_prediction_agent import AIPrediction
from trading.agents.duration.duration_agent import DurationRecommendation
from trading.agents.technical.technical_agent import Direction


class Decision(Enum):
    TRADE_UP = "TRADE_UP"
    TRADE_DOWN = "TRADE_DOWN"
    NO_TRADE = "NO_TRADE"


@dataclass
class FinalDecision:
    """Final trading decision from multi-agent system."""
    action: Decision
    direction: Optional[Direction]
    confidence: float
    duration_seconds: int
    duration_label: str
    reason: str
    agent_contributions: Dict[str, Any]
    timestamp: datetime = None


class DecisionAgent:
    """
    Decision Agent (Final Arbiter)
    
    Receives:
    - Technical Agent → UP/DOWN/NORTUAL with confidence
    - Regime Agent → Market regime information
    - AI Agent → UP: 84% / DOWN: 16% probabilities
    - Duration Agent → Recommended duration
    
    Decision Logic:
    - Requires consensus among agents for a trade signal
    - Conflicting signals result in NO TRADE
    - Regime information filters signals (e.g., avoid strong trends in ranging markets)
    - Duration recommendation overrides conflicting duration preferences
    
    Output:
    - TRADE: UP / DOWN with confidence and duration
    - OR: NO TRADE with reason
    """
    
    def __init__(self):
        # Agent weights in final decision (can be tuned)
        self.agent_weights = {
            "technical": 0.25,
            "regime": 0.20,
            "ai": 0.35,
            "duration": 0.20  # Duration doesn't affect direction but affects trade size/confidence
        }
        
        # Thresholds for decision making
        self.min_confidence_threshold = 0.55
        self.consensus_threshold = 0.55  # Minimum weighted agreement needed
        self.regime_filter_strength = 0.3  # How much regime can override other signals
    
    def decide(
        self,
        technical_signal: TechnicalSignal,
        regime_signal: RegimeSignal,
        ai_signal: AIPrediction,
        duration_recommendation: DurationRecommendation
    ) -> FinalDecision:
        """Make final trading decision based on all agent inputs."""
        
        # Analyze each agent's contribution
        tech_contrib = self._analyze_technical_signal(technical_signal)
        regime_contrib = self._analyze_regime_signal(regime_signal)
        ai_contrib = self._analyze_ai_signal(ai_signal)
        duration_contrib = self._analyze_duration_signal(duration_recommendation)
        
        # Check for conflicting signals that would result in NO TRADE
        no_trade_reason = self._check_conflicts(
            tech_contrib, regime_contrib, ai_contrib, duration_contrib
        )
        
        if no_trade_reason:
            return FinalDecision(
                action=Decision.NO_TRADE,
                direction=None,
                confidence=0.0,
                duration_seconds=duration_recommendation.duration_seconds,
                duration_label=duration_recommendation.duration_label,
                reason=no_trade_reason,
                agent_contributions={
                    "technical": tech_contrib,
                    "regime": regime_contrib,
                    "ai": ai_contrib,
                    "duration": duration_contrib
                },
                timestamp=datetime.now()
            )
        
        # Calculate weighted direction score
        up_score = 0.0
        down_score = 0.0
        
        # Technical contribution
        if tech_contrib["direction"] == Direction.UP:
            up_score += tech_contrib["weighted_strength"]
        elif tech_contrib["direction"] == Direction.DOWN:
            down_score += tech_contrib["weighted_strength"]
        
        # AI contribution
        if ai_contrib["direction"] == Direction.UP:
            up_score += ai_contrib["weighted_strength"]
        elif ai_contrib["direction"] == Direction.DOWN:
            down_score += ai_contrib["weighted_strength"]
        
        # Regime contribution (can boost or suppress signals)
        regime_multiplier = regime_contrib["signal_boost"]
        up_score *= regime_multiplier
        down_score *= regime_multiplier
        
        # Normalize scores by total directional weights so consensus is on [0.0, 1.0] scale
        total_dir_weight = self.agent_weights["technical"] + self.agent_weights["ai"]
        norm_up_score = (up_score / total_dir_weight) if total_dir_weight > 0 else up_score
        norm_down_score = (down_score / total_dir_weight) if total_dir_weight > 0 else down_score

        # Determine final direction
        if norm_up_score > norm_down_score and norm_up_score >= self.min_confidence_threshold:
            final_direction = Direction.UP
            confidence = min(0.95, norm_up_score)
            action = Decision.TRADE_UP
            reason = f"Consensus bullish: Tech({tech_contrib['strength']:.2f}) + AI({ai_contrib['strength']:.2f}) × Regime({regime_multiplier:.2f})"
        elif norm_down_score > norm_up_score and norm_down_score >= self.min_confidence_threshold:
            final_direction = Direction.DOWN
            confidence = min(0.95, norm_down_score)
            action = Decision.TRADE_DOWN
            reason = f"Consensus bearish: Tech({tech_contrib['strength']:.2f}) + AI({ai_contrib['strength']:.2f}) × Regime({regime_multiplier:.2f})"
        else:
            return FinalDecision(
                action=Decision.NO_TRADE,
                direction=None,
                confidence=max(norm_up_score, norm_down_score),
                duration_seconds=duration_recommendation.duration_seconds,
                duration_label=duration_recommendation.duration_label,
                reason=f"Insufficient consensus (UP: {norm_up_score:.2f}, DOWN: {norm_down_score:.2f})",
                agent_contributions={
                    "technical": tech_contrib,
                    "regime": regime_contrib,
                    "ai": ai_contrib,
                    "duration": duration_contrib
                },
                timestamp=datetime.now()
            )
        
        return FinalDecision(
            action=action,
            direction=final_direction,
            confidence=round(confidence, 4),
            duration_seconds=duration_recommendation.duration_seconds,
            duration_label=duration_recommendation.duration_label,
            reason=reason,
            agent_contributions={
                "technical": tech_contrib,
                "regime": regime_contrib,
                "ai": ai_contrib,
                "duration": duration_contrib
            },
            timestamp=datetime.now()
        )
    
    def _analyze_technical_signal(self, signal: TechnicalSignal) -> Dict[str, Any]:
        """Analyze technical agent signal."""
        strength = signal.confidence
        direction = signal.direction
        
        # Weight by agent importance
        weighted_strength = strength * self.agent_weights["technical"]
        
        return {
            "direction": direction,
            "strength": strength,
            "weighted_strength": weighted_strength,
            "confidence": signal.confidence,
            "reason": signal.reason
        }
    
    def _analyze_regime_signal(self, signal: RegimeSignal) -> Dict[str, Any]:
        """Analyze regime agent signal - determines if regime supports trading."""
        # Determine if regime is favorable for the detected trend
        regime = signal.regime
        trend_strength = signal.trend_strength
        
        # Regime boost/suppression factors
        boost_factor = 1.0  # Neutral
        
        if regime in [signal.regime.TRENDING_UP, signal.regime.TRENDING_DOWN]:
            # Trending markets boost trend-following signals
            boost_factor = 1.0 + (trend_strength * 0.3)  # Up to 30% boost
        elif regime == signal.regime.RANGING:
            # Ranging markets suppress strong directional bets
            boost_factor = 0.7  # 30% suppression
        elif regime == signal.regime.HIGH_VOLATILITY:
            # High volatility - be cautious
            boost_factor = 0.8  # 20% suppression
        elif regime == signal.regime.LOW_VOLATILITY:
            # Low volatility - can be good for breakouts
            boost_factor = 1.1  # 10% boost
        
        # Determine if regime aligns with detected trend
        regime_aligned = True
        if signal.trend_direction == "UP" and regime == signal.regime.TRENDING_DOWN:
            regime_aligned = False
        elif signal.trend_direction == "DOWN" and regime == signal.regime.TRENDING_UP:
            regime_aligned = False
        
        if not regime_aligned:
            boost_factor *= 0.5  # Strong penalty for regime/trend mismatch
        
        return {
            "regime": signal.regime.value,
            "trend_strength": signal.trend_strength,
            "volatility": signal.volatility_level,
            "signal_boost": boost_factor,
            "regime_aligned": regime_aligned,
            "confidence": signal.confidence
        }
    
    def _analyze_ai_signal(self, signal: AIPrediction) -> Dict[str, Any]:
        """Analyze AI prediction agent signal."""
        strength = signal.confidence
        direction = signal.direction
        
        # Weight by agent importance
        weighted_strength = strength * self.agent_weights["ai"]
        
        return {
            "direction": direction,
            "strength": strength,
            "weighted_strength": weighted_strength,
            "probabilities": signal.probabilities,
            "model_type": signal.model_type,
            "confidence": signal.confidence
        }
    
    def _analyze_duration_signal(self, signal: DurationRecommendation) -> Dict[str, Any]:
        """Analyze duration agent signal - doesn't affect direction but affects trade viability."""
        return {
            "duration_seconds": signal.duration_seconds,
            "duration_label": signal.duration_label,
            "reason": signal.reason,
            "confidence": signal.confidence,
            "weight": self.agent_weights["duration"]
        }
    
    def _check_conflicts(
        self,
        tech: Dict[str, Any],
        regime: Dict[str, Any],
        ai: Dict[str, Any],
        duration: Dict[str, Any]
    ) -> Optional[str]:
        """Check for conflicting signals that would result in NO TRADE."""
        
        # Check technical vs AI conflict
        tech_dir = tech["direction"]
        ai_dir = ai["direction"]
        
        if tech_dir != Direction.NEUTRAL and ai_dir != Direction.NEUTRAL:
            if tech_dir != ai_dir:
                # Check if the conflict is significant enough to warrant NO TRADE
                tech_conf = tech["confidence"]
                ai_conf = ai["confidence"]
                
                # If both are confident (>0.6) but disagree, no trade
                if tech_conf > 0.6 and ai_conf > 0.6:
                    return f"Technical ({tech_dir.value} {tech_conf:.0%}) conflicts with AI ({ai_dir.value} {ai_conf:.0%})"
        
        # Check regime alignment
        if not regime["regime_aligned"] and regime["signal_boost"] < 0.7:
            return f"Regime ({regime['regime']}) misaligned with detected trend"
        
        # Check if confidence is too low overall
        max_confidence = max(tech["confidence"], ai["confidence"])
        if max_confidence < self.min_confidence_threshold:
            return f"Insufficient confidence (max: {max_confidence:.0%})"
        
        return None


# Global agent instance
decision_agent = DecisionAgent()