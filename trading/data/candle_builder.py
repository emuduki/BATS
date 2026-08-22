import time
from datetime import datetime, timezone
from typing import Dict, List, Optional, Callable


class CandleBuilder:
    """
    Converts incoming real-time ticks into OHLC (Open, High, Low, Close) Candlestick data.
    Supports intervals like 1m (60s), 5m (300s).
    """

    def __init__(self, timeframe: str = "1m", timeframe_seconds: int = 60):
        self.timeframe = timeframe
        self.timeframe_seconds = timeframe_seconds
        self.current_candles: Dict[str, dict] = {}
        self.completed_candles: Dict[str, List[dict]] = {}
        self.listeners: List[Callable[[dict], None]] = []

    def subscribe(self, callback: Callable[[dict], None]):
        """Subscribe callback to receive completed OHLC candles."""
        self.listeners.append(callback)

    def process_tick(self, tick: dict) -> Optional[dict]:
        """
        Ingests a tick object and updates the current forming candle.
        Returns a completed candle dict if a timeframe boundary is crossed.
        """
        symbol = tick["symbol"]
        price = tick["price"]
        timestamp_epoch = tick.get("time_epoch", time.time())

        # Determine candle start timestamp boundary
        candle_start_epoch = int(timestamp_epoch // self.timeframe_seconds) * self.timeframe_seconds
        candle_start_dt = datetime.fromtimestamp(candle_start_epoch, tz=timezone.utc)

        if symbol not in self.current_candles:
            self.current_candles[symbol] = {
                "symbol": symbol,
                "timeframe": self.timeframe,
                "timestamp": candle_start_dt.isoformat(),
                "start_epoch": candle_start_epoch,
                "open": price,
                "high": price,
                "low": price,
                "close": price,
                "volume": 1.0,
                "is_complete": False
            }
            return None

        curr = self.current_candles[symbol]

        # Check if tick belongs to a new candle period
        if candle_start_epoch > curr["start_epoch"]:
            # Mark previous candle as completed
            completed_candle = curr.copy()
            completed_candle["is_complete"] = True

            if symbol not in self.completed_candles:
                self.completed_candles[symbol] = []
            self.completed_candles[symbol].append(completed_candle)

            # Notify subscribers
            for cb in self.listeners:
                try:
                    cb(completed_candle)
                except Exception:
                    pass

            # Start new candle
            self.current_candles[symbol] = {
                "symbol": symbol,
                "timeframe": self.timeframe,
                "timestamp": candle_start_dt.isoformat(),
                "start_epoch": candle_start_epoch,
                "open": price,
                "high": price,
                "low": price,
                "close": price,
                "volume": 1.0,
                "is_complete": False
            }
            return completed_candle

        else:
            # Update current candle OHLC
            curr["high"] = max(curr["high"], price)
            curr["low"] = min(curr["low"], price)
            curr["close"] = price
            curr["volume"] += 1.0
            return None

    def get_candles(self, symbol: str, limit: int = 100) -> List[dict]:
        """Returns completed candles for a given symbol."""
        completed = self.completed_candles.get(symbol, [])
        current = self.current_candles.get(symbol)
        res = list(completed)
        if current:
            res.append(current)
        return res[-limit:]


# Global candle builder instance for 1m timeframe
candle_builder = CandleBuilder(timeframe="1m", timeframe_seconds=60)
