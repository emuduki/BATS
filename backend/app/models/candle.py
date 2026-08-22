import uuid
from datetime import datetime
from sqlalchemy import Column, String, Float, DateTime, Index
from app.database.session import Base


class CandleModel(Base):
    __tablename__ = "market_candles"

    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    symbol = Column(String(50), nullable=False, index=True)
    timeframe = Column(String(10), nullable=False, default="1m", index=True)  # 1m, 5m, 15m, 1h
    timestamp = Column(DateTime, nullable=False, index=True)
    open = Column(Float, nullable=False)
    high = Column(Float, nullable=False)
    low = Column(Float, nullable=False)
    close = Column(Float, nullable=False)
    volume = Column(Float, default=0.0)

    __table_args__ = (
        Index("idx_symbol_timeframe_timestamp", "symbol", "timeframe", "timestamp"),
    )

    def to_dict(self):
        return {
            "id": self.id,
            "symbol": self.symbol,
            "timeframe": self.timeframe,
            "timestamp": self.timestamp.isoformat() if self.timestamp else None,
            "open": self.open,
            "high": self.high,
            "low": self.low,
            "close": self.close,
            "volume": self.volume,
        }
