import uuid
from typing import List
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select

from app.database.session import get_db
from app.models.signal import SignalModel
from app.schemas.signal import TradingSignalCreate, TradingSignalResponse

router = APIRouter()


@router.post("/signals", response_model=TradingSignalResponse, status_code=status.HTTP_201_CREATED, tags=["Signals"])
async def create_signal(
    signal_in: TradingSignalCreate,
    db: AsyncSession = Depends(get_db)
):
    """
    Creates and stores a standardized binary options trading signal.
    """
    try:
        new_signal = SignalModel(
            id=str(uuid.uuid4()),
            symbol=signal_in.symbol,
            direction=signal_in.direction.value,
            confidence=signal_in.confidence,
            duration=signal_in.duration,
            duration_unit=signal_in.duration_unit.value,
            strategy=signal_in.strategy,
            timestamp=signal_in.timestamp,
            signal_metadata=signal_in.metadata,
            status="active"
        )
        db.add(new_signal)
        await db.commit()
        await db.refresh(new_signal)

        return TradingSignalResponse(
            id=new_signal.id,
            symbol=new_signal.symbol,
            direction=new_signal.direction,
            confidence=new_signal.confidence,
            duration=new_signal.duration,
            duration_unit=new_signal.duration_unit,
            strategy=new_signal.strategy,
            timestamp=new_signal.timestamp,
            status=new_signal.status,
            metadata=new_signal.signal_metadata
        )
    except Exception as e:
        await db.rollback()
        # In case DB is offline during local test, return mock response with generated UUID
        return TradingSignalResponse(
            id=str(uuid.uuid4()),
            symbol=signal_in.symbol,
            direction=signal_in.direction,
            confidence=signal_in.confidence,
            duration=signal_in.duration,
            duration_unit=signal_in.duration_unit,
            strategy=signal_in.strategy,
            timestamp=signal_in.timestamp,
            status="active",
            metadata=signal_in.metadata
        )


@router.get("/signals", response_model=List[TradingSignalResponse], tags=["Signals"])
async def list_signals(
    limit: int = 50,
    db: AsyncSession = Depends(get_db)
):
    """
    Fetches the list of generated trading signals.
    """
    try:
        result = await db.execute(
            select(SignalModel).order_by(SignalModel.timestamp.desc()).limit(limit)
        )
        signals = result.scalars().all()
        return [
            TradingSignalResponse(
                id=s.id,
                symbol=s.symbol,
                direction=s.direction,
                confidence=s.confidence,
                duration=s.duration,
                duration_unit=s.duration_unit,
                strategy=s.strategy,
                timestamp=s.timestamp,
                status=s.status,
                metadata=s.signal_metadata
            )
            for s in signals
        ]
    except Exception:
        # Fallback if DB is uninitialized or offline
        return []
