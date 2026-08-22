from typing import List, Optional
from fastapi import APIRouter, Depends, Query
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select

from app.database.session import get_db
from app.models.candle import CandleModel
from app.schemas.candle import CandleResponse
from trading.data.collector import tick_collector
from trading.data.candle_builder import candle_builder

router = APIRouter()


@router.get("/market/ticks", tags=["Market Data"])
async def get_latest_ticks():
    """
    Returns latest price ticks for active market symbols.
    """
    return {
        symbol: {
            "price": tick_collector.get_latest_price(symbol),
            "symbol": symbol
        }
        for symbol in tick_collector.symbols
    }


@router.get("/market/candles", response_model=List[CandleResponse], tags=["Market Data"])
async def get_candles(
    symbol: str = Query(default="R_100", description="Asset symbol"),
    timeframe: str = Query(default="1m", description="Timeframe interval (1m, 5m)"),
    limit: int = Query(default=50, ge=1, le=500),
    db: AsyncSession = Depends(get_db)
):
    """
    Returns historical OHLC candlestick data for the specified symbol and timeframe.
    """
    # 1. Check in-memory/candle builder stream
    candles_in_mem = candle_builder.get_candles(symbol, limit=limit)
    if candles_in_mem:
        return [
            CandleResponse(
                id=c.get("id", f"mem_{idx}"),
                symbol=c["symbol"],
                timeframe=c.get("timeframe", timeframe),
                timestamp=c["timestamp"],
                open=c["open"],
                high=c["high"],
                low=c["low"],
                close=c["close"],
                volume=c.get("volume", 0.0)
            )
            for idx, c in enumerate(candles_in_mem)
        ]

    # 2. Database query fallback
    try:
        query = select(CandleModel).where(
            CandleModel.symbol == symbol,
            CandleModel.timeframe == timeframe
        ).order_by(CandleModel.timestamp.desc()).limit(limit)

        result = await db.execute(query)
        db_candles = result.scalars().all()
        return [
            CandleResponse(
                id=c.id,
                symbol=c.symbol,
                timeframe=c.timeframe,
                timestamp=c.timestamp,
                open=c.open,
                high=c.high,
                low=c.low,
                close=c.close,
                volume=c.volume
            )
            for c in reversed(db_candles)
        ]
    except Exception:
        return []
