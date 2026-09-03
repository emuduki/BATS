"""Multi-Agent Decision System Orchestrator for Phase 4.
Coordinates all agents and produces unified trading decisions.
"""

from dataclasses import dataclass
from typing import Dict, Any, Optional
import pandas as pd
from datetime import datetime

from trading.agents.technical.technical_agent import TechnicalAgent, AgentSignal, Direction
from trading.agents.regime.regime_agent import RegimeAgent, RegimeSignal
from trading.agents.ai.ai_prediction_agent import AIPredictionAgent, AIPrediction
from trading.agents.duration.duration_agent import DurationAgent, DurationRecommendation
from trading.agents.decision.decision_agent import DecisionAgent, FinalDecision


@dataclass
class MultiAgentSignal:
    """Complete multi-agent signal output."""
    technical: AgentSignal
    regime: RegimeSignal
    ai: AIPrediction
    duration: DurationRecommendation
    decision: FinalDecision
    timestamp: datetime


class MultiAgentSystem:
    """
    Multi-Agent Decision System Orchestrator
    
    Coordinates all 5 agents:
    1. Technical Agent - RSI, MACD, EMA, Support/Resistance
    2. Regime Agent - Trending, Ranging, High/Low Volatility
    3. AI Prediction Agent - ML model predictions
    4. Duration Agent - Optimal contract duration
    5. Decision Agent - Final arbiter
    
    Output:
    ┌───────────────────────┐
    │ TECHNICAL AGENT       │
    │ UP — 72%              │
    └───────────┬───────────┘
                │
    ┌───────────▼───────────┐
    │ REGIME AGENT          │
    │ TRENDING              │
    └───────────┬───────────┘
                │
    ┌───────────▼───────────┐
    │ AI AGENT              │
    │ UP — 84%              │
    └───────────┬───────────┘
                │
                ▼
           DECISION
                │
           TRADE UP
           60 SECONDS
    """
    
    def __init__(self, ai_model_path: Optional[str] = None):
        self.technical_agent = TechnicalAgent()
        self.regime_agent = RegimeAgent()
        self.ai_agent = AIPredictionAgent(model_path=ai_model_path)
        self.duration_agent = DurationAgent()
        self.decision_agent = DecisionAgent()
        
        # Track performance for adaptive weighting
        self.decision_history: list = []
    
    def analyze(self, df: pd.DataFrame) -> MultiAgentSignal:
        """Run full multi-agent analysis on market data."""
        
        # Agent 1: Technical Analysis
        technical_signal = self.technical_agent.analyze(df)
        
        # Agent 2: Market Regime Detection
        regime_signal = self.regime_agent.analyze(df)
        
        # Agent 3: AI Prediction
        ai_signal = self.ai_agent.predict(df)
        
        # Agent 4: Duration Recommendation
        duration_recommendation = self.duration_agent.recommend(regime_signal)
        
        # Agent 5: Final Decision
        final_decision = self.decision_agent.decide(
            technical_signal=technical_signal,
            regime_signal=regime_signal,
            ai_signal=ai_signal,
            duration_recommendation=duration_recommendation
        )
        
        # Create complete signal
        multi_signal = MultiAgentSignal(
            technical=technical_signal,
            regime=regime_signal,
            ai=ai_signal,
            duration=duration_recommendation,
            decision=final_decision,
            timestamp=datetime.now()
        )
        
        # Store for performance tracking
        self.decision_history.append(multi_signal)
        
        return multi_signal
    
    def get_decision_summary(self, signal: MultiAgentSignal) -> str:
        """Format decision output for display."""
        decision = signal.decision
        
        lines = [
            "┌─────────────────────────────────────────┐",
            "│       MULTI-AGENT DECISION SYSTEM       │",
            "└─────────────────────────────────────────┘",
            "",
            f"┌───────────────────────┐",
            f"│ TECHNICAL AGENT       │",
            f"│ {signal.technical.direction.value:>4} — {signal.technical.confidence:.0%}              │",
            f"└───────────┬───────────┘",
            f"            │",
            f"┌───────────▼───────────┐",
            f"│ REGIME AGENT          │",
            f"│ {signal.regime.regime.value:<15} │",
            f"└───────────┬───────────┘",
            f"            │",
            f"┌───────────▼───────────┐",
            f"│ AI AGENT              │",
            f"│ UP  — {signal.ai.probabilities['UP']:.0%}              │",
            f"│ DOWN — {signal.ai.probabilities['DOWN']:.0%}              │",
            f"└───────────┬───────────┘",
            f"            │",
            f"            ▼",
            f"       DECISION",
        ]
        
        if decision.action == "TRADE_UP":
            lines.extend([
                f"       TRADE UP",
                f"       {signal.duration.duration_label}"
            ])
        elif decision.action == "TRADE_DOWN":
            lines.extend([
                f"       TRADE DOWN",
                f"       {signal.duration.duration_label}"
            ])
        else:
            lines.extend([
                f"       NO TRADE",
                f"       Reason: {decision.reason}"
            ])
        
        return "\n".join(lines)
    
    def get_agent_details(self, signal: MultiAgentSignal) -> Dict[str, Any]:
        """Get detailed breakdown of all agent signals."""
        return {
            "technical": {
                "direction": signal.technical.direction.value,
                "confidence": signal.technical.confidence,
                "reason": signal.technical.reason,
                "metadata": signal.technical.metadata
            },
            "regime": {
                "regime": signal.regime.regime.value,
                "trend_direction": signal.regime.trend_direction,
                "trend_strength": signal.regime.trend_strength,
                "volatility": signal.regime.volatility_level,
                "confidence": signal.regime.confidence
            },
            "ai": {
                "direction": signal.ai.direction.value,
                "confidence": signal.ai.confidence,
                "probabilities": signal.ai.probabilities,
                "model_type": signal.ai.model_type
            },
            "duration": {
                "duration_seconds": signal.duration.duration_seconds,
                "duration_label": signal.duration.duration_label,
                "reason": signal.duration.reason,
                "confidence": signal.duration.confidence
            },
            "decision": {
                "action": signal.decision.action.value,
                "direction": signal.decision.direction.value if signal.decision.direction else None,
                "confidence": signal.decision.confidence,
                "duration_seconds": signal.decision.duration_seconds,
                "duration_label": signal.decision.duration_label,
                "reason": signal.decision.reason
            },
            "timestamp": signal.timestamp.isoformat()
        }


# Global system instance
multi_agent_system = MultiAgentSystem()