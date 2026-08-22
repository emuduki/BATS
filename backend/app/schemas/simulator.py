from datetime import datetime
from enum import Enum
from typing import Optional, List
from pydantic import BaseModel, Field


class TradeDirection(str, Enum):
    UP = "UP"
    DOWN = "DOWN"


class TradeOutcome(str, Enum):
    WIN = "WIN"
    LOSS = "LOSS"
    TIE = "TIE"


class SimulatorTradeRequest(BaseModel):
    symbol: str = Field(default="R_100", example="R_100")
    direction: TradeDirection = Field(..., example=TradeDirection.UP)
    duration: int = Field(default=60, gt=0, description="Duration in seconds", example=60)
    stake: float = Field(default=5.0, gt=0, description="Stake amount in USD", example=5.0)
    payout_rate: float = Field(default=0.85, ge=0.5, le=1.0, description="Payout rate (e.g. 0.85 = 85%)", example=0.85)


class SimulatorTradeResponse(BaseModel):
    id: str
    symbol: str
    direction: TradeDirection
    stake: float
    duration: int
    entry_price: float
    settlement_price: float
    payout: float
    result: TradeOutcome
    entry_time: datetime
    settlement_time: datetime
    balance_after: float


class SimulatorStatsResponse(BaseModel):
    initial_balance: float = Field(default=100.0)
    current_balance: float
    total_trades: int
    wins: int
    losses: int
    win_rate: float = Field(..., description="Win rate percentage (0.0 to 100.0)")
    total_profit_loss: float
    max_drawdown: float = Field(..., description="Maximum drawdown percentage")
    consecutive_wins: int
    consecutive_losses: int
    max_consecutive_losses: int


class DemoSimulationRun(BaseModel):
    status: str = Field(default="COMPLETED")
    signal: str = Field(..., example="UP")
    symbol: str = Field(..., example="R_100")
    duration: str = Field(..., example="60 seconds")
    entry: float = Field(..., example=1250.50)
    settlement: float = Field(..., example=1251.20)
    result: str = Field(..., example="WIN")
    stake: float = Field(..., example=5.0)
    payout: float = Field(..., example=4.25)
    balance: float = Field(..., example=104.25)
    stats: SimulatorStatsResponse
