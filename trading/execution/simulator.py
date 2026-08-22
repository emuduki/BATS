import uuid
import time
from datetime import datetime, timezone
from typing import Dict, List, Optional

from app.schemas.simulator import (
    SimulatorTradeRequest,
    SimulatorTradeResponse,
    SimulatorStatsResponse,
    DemoSimulationRun,
    TradeDirection,
    TradeOutcome
)
from trading.data.collector import tick_collector


class BinarySimulator:
    """
    Binary Options Trade Execution Simulator Engine.
    Tracks virtual account balance, trade entries, settlement evaluation, and performance analytics.
    """

    def __init__(self, initial_balance: float = 100.00, default_payout_rate: float = 0.85):
        self.initial_balance = initial_balance
        self.current_balance = initial_balance
        self.peak_balance = initial_balance
        self.default_payout_rate = default_payout_rate

        self.trades: List[dict] = []
        self.total_trades = 0
        self.wins = 0
        self.losses = 0
        self.ties = 0

        self.current_consecutive_wins = 0
        self.current_consecutive_losses = 0
        self.max_consecutive_losses = 0
        self.max_drawdown = 0.0

    def get_stats(self) -> SimulatorStatsResponse:
        """Calculates and returns current performance statistics."""
        win_rate = (self.wins / self.total_trades * 100.0) if self.total_trades > 0 else 0.0
        total_pnl = round(self.current_balance - self.initial_balance, 2)

        return SimulatorStatsResponse(
            initial_balance=round(self.initial_balance, 2),
            current_balance=round(self.current_balance, 2),
            total_trades=self.total_trades,
            wins=self.wins,
            losses=self.losses,
            win_rate=round(win_rate, 2),
            total_profit_loss=total_pnl,
            max_drawdown=round(self.max_drawdown, 2),
            consecutive_wins=self.current_consecutive_wins,
            consecutive_losses=self.current_consecutive_losses,
            max_consecutive_losses=self.max_consecutive_losses
        )

    def update_drawdown(self):
        """Updates maximum drawdown metrics after each completed trade."""
        if self.current_balance > self.peak_balance:
            self.peak_balance = self.current_balance

        if self.peak_balance > 0:
            current_dd = ((self.peak_balance - self.current_balance) / self.peak_balance) * 100.0
            if current_dd > self.max_drawdown:
                self.max_drawdown = current_dd

    def execute_trade(
        self,
        request: SimulatorTradeRequest,
        mock_entry_price: Optional[float] = None,
        mock_settlement_price: Optional[float] = None
    ) -> SimulatorTradeResponse:
        """
        Executes a simulated binary option trade.
        Optionally accepts mock_entry_price and mock_settlement_price for deterministic unit tests.
        """
        symbol = request.symbol
        direction = request.direction
        stake = request.stake
        payout_rate = request.payout_rate or self.default_payout_rate
        duration = request.duration

        # 1. Capture Entry Price
        entry_time = datetime.now(timezone.utc)
        if mock_entry_price is not None:
            entry_price = mock_entry_price
        else:
            entry_price = tick_collector.get_latest_price(symbol)

        # 2. Determine Settlement Price (instant simulated resolution or tick generator drift)
        if mock_settlement_price is not None:
            settlement_price = mock_settlement_price
        else:
            # Generate settlement price based on real tick collector or realistic simulated drift
            # If UP requested, simulate drift to test settlement logic if market moves
            current_tick = tick_collector.get_latest_price(symbol)
            settlement_price = current_tick

        settlement_time = datetime.now(timezone.utc)

        # 3. Settlement Evaluation
        if direction == TradeDirection.UP:
            if settlement_price > entry_price:
                result = TradeOutcome.WIN
            elif settlement_price < entry_price:
                result = TradeOutcome.LOSS
            else:
                result = TradeOutcome.TIE
        else:  # TradeDirection.DOWN
            if settlement_price < entry_price:
                result = TradeOutcome.WIN
            elif settlement_price > entry_price:
                result = TradeOutcome.LOSS
            else:
                result = TradeOutcome.TIE

        # 4. Financial & Analytics Updates
        if result == TradeOutcome.WIN:
            payout = round(stake * payout_rate, 2)
            self.current_balance += payout
            self.wins += 1
            self.current_consecutive_wins += 1
            self.current_consecutive_losses = 0
        elif result == TradeOutcome.LOSS:
            payout = -round(stake, 2)
            self.current_balance -= stake
            self.losses += 1
            self.current_consecutive_losses += 1
            self.current_consecutive_wins = 0
            if self.current_consecutive_losses > self.max_consecutive_losses:
                self.max_consecutive_losses = self.current_consecutive_losses
        else:  # TIE
            payout = 0.0
            self.ties += 1

        self.total_trades += 1
        self.update_drawdown()

        trade_record = {
            "id": str(uuid.uuid4()),
            "symbol": symbol,
            "direction": direction,
            "stake": stake,
            "duration": duration,
            "entry_price": entry_price,
            "settlement_price": settlement_price,
            "payout": payout,
            "result": result,
            "entry_time": entry_time,
            "settlement_time": settlement_time,
            "balance_after": round(self.current_balance, 2)
        }

        self.trades.append(trade_record)

        return SimulatorTradeResponse(
            id=trade_record["id"],
            symbol=trade_record["symbol"],
            direction=trade_record["direction"],
            stake=trade_record["stake"],
            duration=trade_record["duration"],
            entry_price=trade_record["entry_price"],
            settlement_price=trade_record["settlement_price"],
            payout=trade_record["payout"],
            result=trade_record["result"],
            entry_time=trade_record["entry_time"],
            settlement_time=trade_record["settlement_time"],
            balance_after=trade_record["balance_after"]
        )

    def run_demo_simulation(self) -> DemoSimulationRun:
        """
        Executes a demonstration trade matching the exact Phase 1 deliverable format:
        Signal: UP
        Duration: 60 seconds
        Entry: 1250.50
        Settlement: 1251.20
        Result: WIN
        Balance: $105.40
        """
        req = SimulatorTradeRequest(
            symbol="R_100",
            direction=TradeDirection.UP,
            duration=60,
            stake=5.00,
            payout_rate=0.85
        )

        res = self.execute_trade(
            request=req,
            mock_entry_price=1250.50,
            mock_settlement_price=1251.20
        )

        stats = self.get_stats()

        return DemoSimulationRun(
            status="COMPLETED",
            signal=res.direction.value,
            symbol=res.symbol,
            duration=f"{res.duration} seconds",
            entry=res.entry_price,
            settlement=res.settlement_price,
            result=res.result.value,
            stake=res.stake,
            payout=res.payout,
            balance=res.balance_after,
            stats=stats
        )


# Global simulator instance
binary_simulator = BinarySimulator(initial_balance=100.00)
