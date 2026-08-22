import sys
from pathlib import Path

root_path = Path(__file__).resolve().parent.parent.parent
if str(root_path) not in sys.path:
    sys.path.insert(0, str(root_path))
backend_path = Path(__file__).resolve().parent.parent
if str(backend_path) not in sys.path:
    sys.path.insert(0, str(backend_path))

import pytest
import pandas as pd
from fastapi.testclient import TestClient
from app.main import app

from trading.indicators.technical import add_all_indicators
from trading.strategies import (
    EMACrossoverStrategy,
    RSIStrategy,
    MACDStrategy,
    SupportResistanceStrategy,
    CombinedConsensusEngine,
    SignalDirection
)
from trading.backtesting.engine import BinaryBacktester

client = TestClient(app)


def test_technical_indicators():
    backtester = BinaryBacktester()
    df = backtester.generate_synthetic_candles(num_candles=100)
    df_ind = add_all_indicators(df)

    assert "ema_9" in df_ind.columns
    assert "ema_21" in df_ind.columns
    assert "rsi_14" in df_ind.columns
    assert "macd_line" in df_ind.columns
    assert "support" in df_ind.columns
    assert "resistance" in df_ind.columns
    print("SUCCESS: All technical indicators calculated successfully!")


def test_individual_strategies():
    backtester = BinaryBacktester()
    df = backtester.generate_synthetic_candles(num_candles=100)

    ema_strat = EMACrossoverStrategy()
    rsi_strat = RSIStrategy()
    macd_strat = MACDStrategy()
    sr_strat = SupportResistanceStrategy()

    sig_ema = ema_strat.evaluate(df, index=-1)
    sig_rsi = rsi_strat.evaluate(df, index=-1)
    sig_macd = macd_strat.evaluate(df, index=-1)
    sig_sr = sr_strat.evaluate(df, index=-1)

    assert sig_ema.strategy_name == "EMA_Crossover"
    assert sig_rsi.strategy_name == "RSI"
    assert sig_macd.strategy_name == "MACD"
    assert sig_sr.strategy_name == "Support_Resistance"
    print("SUCCESS: All 4 individual strategies evaluated cleanly!")


def test_combined_consensus_engine():
    backtester = BinaryBacktester()
    df = backtester.generate_synthetic_candles(num_candles=100)

    combined_engine = CombinedConsensusEngine()
    consensus_signal = combined_engine.evaluate(df, index=-1)

    assert consensus_signal.strategy_name == "Combined_Consensus"
    assert consensus_signal.direction in [SignalDirection.UP, SignalDirection.DOWN, SignalDirection.NEUTRAL]
    assert 0.0 <= consensus_signal.confidence <= 1.0
    print(f"SUCCESS: Combined Consensus Engine signal: {consensus_signal}")


def test_backtester_engine_execution():
    backtester = BinaryBacktester(initial_balance=1000.00, stake=10.00, payout_rate=0.85)
    df = backtester.generate_synthetic_candles(num_candles=1000)

    results = backtester.run_backtest(df, CombinedConsensusEngine(), duration_candles=1)

    assert results["total_trades"] > 0
    assert results["wins"] + results["losses"] + results["ties"] == results["total_trades"]
    assert results["win_rate"] >= 0.0
    assert results["break_even_win_rate"] == 54.05
    assert "max_drawdown" in results
    assert "max_losing_streak" in results
    print(f"SUCCESS: BinaryBacktester evaluated 1,000 candles with metrics: win_rate={results['win_rate']}%, EV=${results['expected_value_per_trade']}")


def test_backtest_api_endpoints():
    # 1. Strategies list
    res_list = client.get("/api/v1/backtest/strategies")
    assert res_list.status_code == 200
    assert "combined" in res_list.json()

    # 2. Run Backtest
    req_body = {
        "strategy_name": "combined",
        "symbol": "R_100",
        "num_candles": 500,
        "duration_seconds": 60,
        "stake": 10.0,
        "payout_rate": 0.85
    }
    res_run = client.post("/api/v1/backtest/run", json=req_body)
    assert res_run.status_code == 200
    data_run = res_run.json()
    assert data_run["strategy"] == "Combined_Consensus"
    assert data_run["break_even_win_rate"] == 54.05

    # 3. Deliverable Report Endpoint
    res_report = client.post("/api/v1/backtest/demo-report")
    assert res_report.status_code == 200
    report_data = res_report.json()

    print("\n--- PHASE 2 DELIVERABLE BACKTEST REPORT ---")
    print(report_data["report_text"])
    print("-------------------------------------------\n")

    assert "STRATEGY: EMA + RSI" in report_data["report_text"]
    assert "Win Rate:" in report_data["report_text"]
    assert "Max Losing Streak:" in report_data["report_text"]
    print("SUCCESS: Backtest API endpoints passed successfully!")


if __name__ == "__main__":
    test_technical_indicators()
    test_individual_strategies()
    test_combined_consensus_engine()
    test_backtester_engine_execution()
    test_backtest_api_endpoints()
    print("\nALL PHASE 2 VERIFICATION TESTS PASSED SUCCESSFULLY!")
