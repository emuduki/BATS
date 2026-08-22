import pytest
import asyncio
from datetime import datetime
from fastapi.testclient import TestClient
from app.main import app
from app.schemas.signal import TradingSignalCreate, SignalDirection, SignalDurationUnit

client = TestClient(app)


def test_health_endpoint():
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert "status" in data
    assert "database" in data
    assert "redis" in data
    assert data["status"] == "running"


def test_trading_signal_schema():
    signal_input = TradingSignalCreate(
        symbol="R_100",
        direction=SignalDirection.UP,
        confidence=0.87,
        duration=60,
        duration_unit=SignalDurationUnit.SECONDS,
        strategy="ema_rsi",
        timestamp=datetime.utcnow()
    )
    assert signal_input.symbol == "R_100"
    assert signal_input.direction == "UP"
    assert signal_input.confidence == 0.87
    assert signal_input.duration == 60
    assert signal_input.strategy == "ema_rsi"


def test_create_signal_endpoint():
    payload = {
        "symbol": "R_100",
        "direction": "UP",
        "confidence": 0.87,
        "duration": 60,
        "duration_unit": "seconds",
        "strategy": "ema_rsi"
    }
    response = client.post("/api/v1/signals", json=payload)
    assert response.status_code in [200, 201]
    res_json = response.json()
    assert res_json["symbol"] == "R_100"
    assert res_json["direction"] == "UP"


if __name__ == "__main__":
    test_health_endpoint()
    test_trading_signal_schema()
    test_create_signal_endpoint()
    print("ALL PHASE 0 VERIFICATION TESTS PASSED SUCCESSFULLY IN BACKEND!")
