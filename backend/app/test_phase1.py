import sys
import time
from pathlib import Path

root_path = Path(__file__).resolve().parent.parent.parent
if str(root_path) not in sys.path:
    sys.path.insert(0, str(root_path))
backend_path = Path(__file__).resolve().parent.parent
if str(backend_path) not in sys.path:
    sys.path.insert(0, str(backend_path))

from datetime import datetime, timezone
from fastapi.testclient import TestClient
from app.main import app

from trading.data.collector import TickCollector
from trading.data.candle_builder import CandleBuilder
from trading.execution.simulator import BinarySimulator
from app.schemas.simulator import SimulatorTradeRequest, TradeDirection, TradeOutcome

client = TestClient(app)


def test_tick_collector():
    collector = TickCollector(symbols=["R_100"])
    tick = collector.generate_next_tick("R_100")
    assert tick["symbol"] == "R_100"
    assert "price" in tick
    assert tick["price"] > 0
    print("SUCCESS: TickCollector generated valid tick:", tick)


def test_candle_builder():
    builder = CandleBuilder(timeframe="1m", timeframe_seconds=60)

    # Feed ticks in same timeframe
    tick1 = {"symbol": "R_100", "price": 1250.00, "time_epoch": 1700000001}
    tick2 = {"symbol": "R_100", "price": 1255.00, "time_epoch": 1700000010}
    tick3 = {"symbol": "R_100", "price": 1245.00, "time_epoch": 1700000020}
    tick4 = {"symbol": "R_100", "price": 1252.00, "time_epoch": 1700000030}

    builder.process_tick(tick1)
    builder.process_tick(tick2)
    builder.process_tick(tick3)
    builder.process_tick(tick4)

    curr = builder.current_candles["R_100"]
    assert curr["open"] == 1250.00
    assert curr["high"] == 1255.00
    assert curr["low"] == 1245.00
    assert curr["close"] == 1252.00

    # Cross timeframe boundary (tick in next minute)
    tick5 = {"symbol": "R_100", "price": 1254.00, "time_epoch": 1700000065}
    completed = builder.process_tick(tick5)
    assert completed is not None
    assert completed["close"] == 1252.00
    print("SUCCESS: CandleBuilder aggregated OHLC candle correctly:", completed)


def test_binary_simulator_win_and_loss():
    sim = BinarySimulator(initial_balance=100.00, default_payout_rate=0.85)

    # 1. Test WIN Trade: UP, Entry 1250.50, Settlement 1251.20
    req1 = SimulatorTradeRequest(symbol="R_100", direction=TradeDirection.UP, duration=60, stake=5.00)
    res1 = sim.execute_trade(req1, mock_entry_price=1250.50, mock_settlement_price=1251.20)
    assert res1.result == TradeOutcome.WIN
    assert res1.payout == 4.25
    assert res1.balance_after == 104.25

    # 2. Test LOSS Trade: DOWN, Entry 1250.50, Settlement 1251.20
    req2 = SimulatorTradeRequest(symbol="R_100", direction=TradeDirection.DOWN, duration=60, stake=5.00)
    res2 = sim.execute_trade(req2, mock_entry_price=1250.50, mock_settlement_price=1251.20)
    assert res2.result == TradeOutcome.LOSS
    assert res2.payout == -5.00
    assert res2.balance_after == 99.25

    stats = sim.get_stats()
    assert stats.total_trades == 2
    assert stats.wins == 1
    assert stats.losses == 1
    assert stats.win_rate == 50.0
    print("SUCCESS: BinarySimulator evaluated WIN/LOSS and updated metrics correctly:", stats.model_dump())


def test_market_api_endpoints():
    res_ticks = client.get("/api/v1/market/ticks")
    assert res_ticks.status_code == 200
    assert "R_100" in res_ticks.json()

    res_candles = client.get("/api/v1/market/candles?symbol=R_100&timeframe=1m")
    assert res_candles.status_code == 200
    print("SUCCESS: Market API endpoints responded cleanly!")


def test_simulator_api_endpoints():
    # Test POST /api/v1/simulator/run-demo
    res_demo = client.post("/api/v1/simulator/run-demo")
    assert res_demo.status_code == 200
    data = res_demo.json()

    print("\n--- DEMO SIMULATION DELIVERABLE OUTPUT ---")
    print(f"Status:     {data['status']}")
    print(f"Signal:     {data['signal']}")
    print(f"Duration:   {data['duration']}")
    print(f"Entry:      {data['entry']}")
    print(f"Settlement: {data['settlement']}")
    print(f"Result:     {data['result']}")
    print(f"Balance:    ${data['balance']}")
    print("------------------------------------------\n")

    assert data["signal"] == "UP"
    assert data["entry"] == 1250.50
    assert data["settlement"] == 1251.20
    assert data["result"] == "WIN"

    # Test GET /api/v1/simulator/stats
    res_stats = client.get("/api/v1/simulator/stats")
    assert res_stats.status_code == 200
    print("SUCCESS: Simulator API endpoints passed successfully!")


if __name__ == "__main__":
    test_tick_collector()
    test_candle_builder()
    test_binary_simulator_win_and_loss()
    test_market_api_endpoints()
    test_simulator_api_endpoints()
    print("\nALL PHASE 1 VERIFICATION TESTS PASSED SUCCESSFULLY!")
