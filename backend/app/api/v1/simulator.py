from fastapi import APIRouter, status
from app.schemas.simulator import (
    SimulatorTradeRequest,
    SimulatorTradeResponse,
    SimulatorStatsResponse,
    DemoSimulationRun
)
from trading.execution.simulator import binary_simulator

router = APIRouter()


@router.post("/simulator/trade", response_model=SimulatorTradeResponse, status_code=status.HTTP_201_CREATED, tags=["Simulator"])
async def execute_simulated_trade(request: SimulatorTradeRequest):
    """
    Executes a simulated binary options trade and calculates financial outcome & metrics.
    """
    return binary_simulator.execute_trade(request)


@router.get("/simulator/stats", response_model=SimulatorStatsResponse, tags=["Simulator"])
async def get_simulator_stats():
    """
    Fetches real-time binary options simulator performance metrics
    (Win Rate, Total Trades, Wins, Losses, Max Drawdown, Consecutive Losses).
    """
    return binary_simulator.get_stats()


@router.post("/simulator/run-demo", response_model=DemoSimulationRun, tags=["Simulator"])
async def run_demo_simulation():
    """
    Executes a demonstration trade matching the exact Phase 1 deliverable format:
    Signal: UP | Duration: 60s | Entry: 1250.50 | Settlement: 1251.20 | Result: WIN | Balance: $104.25
    """
    return binary_simulator.run_demo_simulation()
