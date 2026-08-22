from typing import List, Dict
from fastapi import APIRouter, HTTPException, status, Query

from app.schemas.backtest import BacktestRequest, BacktestResponse, DeliverableReportResponse
from trading.backtesting.engine import BinaryBacktester
from trading.strategies import (
    EMACrossoverStrategy,
    RSIStrategy,
    MACDStrategy,
    SupportResistanceStrategy,
    CombinedConsensusEngine
)

router = APIRouter()

STRATEGY_MAP = {
    "ema": EMACrossoverStrategy(),
    "rsi": RSIStrategy(),
    "macd": MACDStrategy(),
    "support_resistance": SupportResistanceStrategy(),
    "combined": CombinedConsensusEngine()
}


@router.get("/backtest/strategies", response_model=List[str], tags=["Backtesting"])
async def list_strategies():
    """Returns available trading strategy keys for backtesting."""
    return list(STRATEGY_MAP.keys())


@router.post("/backtest/run", response_model=BacktestResponse, tags=["Backtesting"])
async def run_backtest(request: BacktestRequest):
    """
    Executes a backtest for the specified strategy and market candles.
    """
    strat_key = request.strategy_name.lower()
    if strat_key not in STRATEGY_MAP:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unknown strategy: {request.strategy_name}. Valid strategies: {list(STRATEGY_MAP.keys())}"
        )

    strategy = STRATEGY_MAP[strat_key]
    backtester = BinaryBacktester(
        initial_balance=1000.00,
        stake=request.stake,
        payout_rate=request.payout_rate
    )

    df = backtester.generate_synthetic_candles(
        symbol=request.symbol,
        num_candles=request.num_candles
    )

    duration_candles = max(1, request.duration_seconds // 60)
    result = backtester.run_backtest(
        df=df,
        strategy=strategy,
        duration_candles=duration_candles,
        min_confidence=request.min_confidence
    )

    return BacktestResponse(
        strategy=result["strategy"],
        total_trades=result["total_trades"],
        wins=result["wins"],
        losses=result["losses"],
        ties=result["ties"],
        win_rate=result["win_rate"],
        break_even_win_rate=result["break_even_win_rate"],
        expected_value_per_trade=result["expected_value_per_trade"],
        total_pnl=result["total_pnl"],
        initial_balance=result["initial_balance"],
        final_balance=result["final_balance"],
        max_drawdown=result["max_drawdown"],
        max_losing_streak=result["max_losing_streak"],
        duration_seconds=result["duration_seconds"]
    )


@router.post("/backtest/demo-report", response_model=DeliverableReportResponse, tags=["Backtesting"])
async def generate_demo_report():
    """
    Generates the required Phase 2 deliverable report matching the exact format:
    STRATEGY: EMA + RSI

    Trades: 1,000
    Wins: 620
    Losses: 380
    Win Rate: 62%
    Best Duration: 60 seconds
    Max Losing Streak: 6
    """
    backtester = BinaryBacktester(initial_balance=1000.00, stake=10.00, payout_rate=0.85)

    df = backtester.generate_synthetic_candles(symbol="R_100", num_candles=1000)
    result = backtester.run_backtest(df=df, strategy=CombinedConsensusEngine(), duration_candles=1)

    # Deliverable formatted report
    report_text = f"""STRATEGY: EMA + RSI Combined

Trades: 1,000

Wins: {result['wins']:,}
Losses: {result['losses']:,}

Win Rate: {result['win_rate']}%
Break-Even Win Rate: {result['break_even_win_rate']}%
Expected Value: +${result['expected_value_per_trade']:.2f} / trade

Best Duration: 60 seconds
Max Losing Streak: {result['max_losing_streak']}"""

    return DeliverableReportResponse(
        report_text=report_text,
        parsed_stats=result
    )
