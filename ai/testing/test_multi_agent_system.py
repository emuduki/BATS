"""Test multi-agent system with sample market data."""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Import agents
from trading.agents.multi_agent_system import multi_agent_system


def create_test_data(n_bars: int = 200, trend: str = "UP", volatility: str = "MEDIUM") -> pd.DataFrame:
    """Create synthetic OHLCV data for testing."""
    
    base_price = 100.0
    
    # Trend component
    if trend == "UP":
        trend_component = np.linspace(0, 5, n_bars)
    elif trend == "DOWN":
        trend_component = np.linspace(0, -5, n_bars)
    else:
        trend_component = np.zeros(n_bars)
    
    # Volatility
    if volatility == "HIGH":
        vol = 0.03
    elif volatility == "LOW":
        vol = 0.005
    else:
        vol = 0.015
    
    # Generate prices
    noise = np.random.normal(0, vol, n_bars)
    prices = base_price + trend_component + np.cumsum(noise)
    
    # Generate OHLCV
    dates = pd.date_range(start=datetime.now() - timedelta(hours=n_bars), periods=n_bars, freq='5min')
    
    df = pd.DataFrame({
        'timestamp': dates,
        'open': prices,
        'high': prices + np.abs(np.random.normal(0, vol/2, n_bars)),
        'low': prices - np.abs(np.random.normal(0, vol/2, n_bars)),
        'close': prices,
        'volume': np.random.uniform(100, 1000, n_bars)
    })
    
    return df


def test_agents():
    """Test all agents individually."""
    
    # Create test data
    print("\n=== Creating Test Data ===")
    df = create_test_data(n_bars=200, trend="UP", volatility="MEDIUM")
    print(f"Created {len(df)} bars of synthetic data")
    print(f"Price range: {df['close'].min():.2f} - {df['close'].max():.2f}")
    print()
    
    # Run multi-agent analysis
    print("=== Running Multi-Agent Analysis ===")
    signal = multi_agent_system.analyze(df)
    
    # Display decision summary
    print("\n" + multi_agent_system.get_decision_summary(signal))
    
    # Display agent details
    print("\n=== Agent Details ===")
    details = multi_agent_system.get_agent_details(signal)
    for agent_name, agent_data in details.items():
        if agent_name != "timestamp":
            print(f"\n{agent_name.upper()}:")
            for key, value in agent_data.items():
                if isinstance(value, dict):
                    print(f"  {key}:")
                    for k, v in value.items():
                        print(f"    {k}: {v}")
                else:
                    print(f"  {key}: {value}")
    
    return signal


def test_no_trade_scenario():
    """Test scenario where conflicting signals result in NO TRADE."""
    
    print("\n\n=== Testing NO TRADE Scenario ===")
    print("Creating data with conflicting signals (uptrend but oversold RSI)...")
    
    # Create data with high volatility and unclear trend
    df = create_test_data(n_bars=200, trend="NEUTRAL", volatility="HIGH")
    print(f"Created {len(df)} bars of high volatility, ranging data")
    
    # Run multi-agent analysis
    signal = multi_agent_system.analyze(df)
    
    # Display decision summary
    print("\n" + multi_agent_system.get_decision_summary(signal))
    
    return signal


if __name__ == "__main__":
    print("=" * 60)
    print("PHASE 4 — MULTI-AGENT DECISION SYSTEM TEST")
    print("=" * 60)
    
    # Test 1: Standard market scenario
    signal1 = test_agents()
    
    # Test 2: NO TRADE scenario
    signal2 = test_no_trade_scenario()
    
    print("\n" + "=" * 60)
    print("TESTS COMPLETED")
    print("=" * 60)
