import uuid
from datetime import datetime
from sqlalchemy import Column, String, Float, DateTime, JSON, ForeignKey
from app.database.session import Base


class TradeModel(Base):
    __tablename__ = "trades"

    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    signal_id = Column(String(36), ForeignKey("trading_signals.id"), nullable=True)
    symbol = Column(String(50), nullable=False, index=True)
    direction = Column(String(10), nullable=False)
    stake = Column(Float, nullable=False)
    payout = Column(Float, nullable=True)
    entry_price = Column(Float, nullable=True)
    exit_price = Column(Float, nullable=True)
    status = Column(String(20), default="open", index=True)  # open, won, lost, cancelled
    entry_time = Column(DateTime, default=datetime.utcnow)
    exit_time = Column(DateTime, nullable=True)
    broker = Column(String(50), default="deriv")
    trade_metadata = Column(JSON, nullable=True)
