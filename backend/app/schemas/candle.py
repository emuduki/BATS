from datetime import datetime
from pydantic import BaseModel, Field


class CandleBase(BaseModel):
    symbol: str = Field(..., example="R_100")
    timeframe: str = Field(default="1m", example="1m")
    timestamp: datetime
    open: float = Field(..., example=1250.20)
    high: float = Field(..., example=1251.50)
    low: float = Field(..., example=1249.80)
    close: float = Field(..., example=1251.10)
    volume: float = Field(default=0.0, example=100.0)


class CandleCreate(CandleBase):
    pass


class CandleResponse(CandleBase):
    id: str

    class Config:
        from_attributes = True
