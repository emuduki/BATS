from datetime import datetime
from enum import Enum
from typing import Any, Dict, Optional
from pydantic import BaseModel, Field


class SignalDirection(str, Enum):
    UP = "UP"
    DOWN = "DOWN"


class SignalDurationUnit(str, Enum):
    SECONDS = "seconds"
    MINUTES = "minutes"
    HOURS = "hours"
    TICKS = "ticks"


class TradingSignalCreate(BaseModel):
    symbol: str = Field(..., description="Market asset symbol", example="R_100")
    direction: SignalDirection = Field(..., description="Trade direction (UP/DOWN)", example=SignalDirection.UP)
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence level between 0 and 1", example=0.87)
    duration: int = Field(..., gt=0, description="Trade duration integer", example=60)
    duration_unit: SignalDurationUnit = Field(
        default=SignalDurationUnit.SECONDS,
        description="Duration unit (seconds, minutes, hours, ticks)",
        example=SignalDurationUnit.SECONDS
    )
    strategy: str = Field(..., description="Strategy or agent producing the signal", example="ema_rsi")
    timestamp: datetime = Field(
        default_factory=datetime.utcnow,
        description="UTC timestamp of signal generation"
    )
    metadata: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Additional technical indicator values or metadata"
    )


class TradingSignalResponse(TradingSignalCreate):
    id: str = Field(..., description="Unique signal identifier")
    status: str = Field(default="active", description="Signal lifecycle status (active, executed, expired, cancelled)")

    class Config:
        from_attributes = True
