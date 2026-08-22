from typing import Dict, Any, List, Optional
from pydantic import BaseModel, Field


class BacktestRequest(BaseModel):
    strategy_name: str = Field(default="combined", example="combined")  # ema, rsi, macd, support_resistance, combined
    symbol: str = Field(default="R_100", example="R_100")
    num_candles: int = Field(default=1000, ge=100, le=10000, example=1000)
    duration_seconds: int = Field(default=60, example=60)
    stake: float = Field(default=10.00, gt=0, example=10.00)
    payout_rate: float = Field(default=0.85, ge=0.5, le=1.0, example=0.85)
    min_confidence: float = Field(default=0.50, ge=0.0, le=1.0, example=0.50)


class BacktestResponse(BaseModel):
    strategy: str
    total_trades: int
    wins: int
    losses: int
    ties: int
    win_rate: float
    break_even_win_rate: float
    expected_value_per_trade: float
    total_pnl: float
    initial_balance: float
    final_balance: float
    max_drawdown: float
    max_losing_streak: int
    duration_seconds: int


class DeliverableReportResponse(BaseModel):
    report_text: str = Field(..., description="Formatted text deliverable report")
    parsed_stats: Dict[str, Any]
