from dataclasses import dataclass
from enum import Enum
from typing import Optional, Dict, Any
import pandas as pd


class SignalDirection(str, Enum):
    UP = "UP"
    DOWN = "DOWN"
    NEUTRAL = "NEUTRAL"


@dataclass
class StrategySignal:
    direction: SignalDirection
    confidence: float  # 0.0 to 1.0 (or 0% to 100%)
    strategy_name: str
    reason: str
    metadata: Optional[Dict[str, Any]] = None


class BaseStrategy:
    """Abstract Base Class for all BATS Trading Strategies."""

    def __init__(self, name: str):
        self.name = name

    def evaluate(self, df: pd.DataFrame, index: int = -1) -> StrategySignal:
        """
        Evaluates current DataFrame at specified index and returns a StrategySignal.
        Must be implemented by subclasses.
        """
        raise NotImplementedError("Subclasses must implement evaluate()")
