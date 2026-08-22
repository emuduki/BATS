import uuid
from datetime import datetime
from sqlalchemy import Column, String, Float, Integer, DateTime, JSON
from app.database.session import Base


class SignalModel(Base):
    __tablename__ = "trading_signals"

    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    symbol = Column(String(50), nullable=False, index=True)
    direction = Column(String(10), nullable=False)
    confidence = Column(Float, nullable=False)
    duration = Column(Integer, nullable=False)
    duration_unit = Column(String(20), nullable=False, default="seconds")
    strategy = Column(String(100), nullable=False, index=True)
    timestamp = Column(DateTime, default=datetime.utcnow, index=True)
    status = Column(String(20), default="active", index=True)
    signal_metadata = Column(JSON, nullable=True)

    def to_dict(self):
        return {
            "id": self.id,
            "symbol": self.symbol,
            "direction": self.direction,
            "confidence": self.confidence,
            "duration": self.duration,
            "duration_unit": self.duration_unit,
            "strategy": self.strategy,
            "timestamp": self.timestamp.isoformat() if self.timestamp else None,
            "status": self.status,
            "metadata": self.signal_metadata,
        }
