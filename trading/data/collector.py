import asyncio
import json
import random
import time
from datetime import datetime, timezone
from typing import Dict, List, Optional, Callable


class TickCollector:
    """
    Market Tick Data Collector.
    Continuously streams or simulates price ticks for binary options assets (e.g., R_100, EUR/USD).
    """

    def __init__(self, symbols: Optional[List[str]] = None):
        self.symbols = symbols or ["R_100", "EUR/USD"]
        self.prices: Dict[str, float] = {
            "R_100": 1250.50,
            "EUR/USD": 1.08420
        }
        self.volatilities: Dict[str, float] = {
            "R_100": 0.45,
            "EUR/USD": 0.00015
        }
        self.is_running = False
        self.listeners: List[Callable[[dict], None]] = []
        self._history: Dict[str, List[dict]] = {s: [] for s in self.symbols}

    def subscribe(self, callback: Callable[[dict], None]):
        """Subscribe listener callback to live tick updates."""
        self.listeners.append(callback)

    def get_latest_price(self, symbol: str) -> float:
        """Returns the latest price tick for a symbol."""
        return self.prices.get(symbol, 1250.0)

    def generate_next_tick(self, symbol: str) -> dict:
        """Generates the next price tick for a given symbol."""
        current_price = self.prices[symbol]
        vol = self.volatilities[symbol]

        # Geometric Brownian Motion step
        change = random.gauss(0, vol)
        new_price = round(max(0.0001, current_price + change), 5 if "EUR" in symbol else 2)

        self.prices[symbol] = new_price
        tick = {
            "symbol": symbol,
            "price": new_price,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "time_epoch": time.time()
        }

        # Store in rolling buffer
        self._history[symbol].append(tick)
        if len(self._history[symbol]) > 1000:
            self._history[symbol].pop(0)

        # Notify subscribers
        for callback in self.listeners:
            try:
                callback(tick)
            except Exception:
                pass

        return tick

    async def start_streaming(self, interval_seconds: float = 1.0):
        """Asynchronously streams price ticks at regular intervals."""
        self.is_running = True
        while self.is_running:
            for symbol in self.symbols:
                self.generate_next_tick(symbol)
            await asyncio.sleep(interval_seconds)

    def stop_streaming(self):
        self.is_running = False


# Global singleton instance
tick_collector = TickCollector()
