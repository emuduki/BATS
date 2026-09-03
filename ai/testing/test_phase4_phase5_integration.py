"""Integration test for Phase 4 + Phase 5: Multi-Agent System with Risk Management."""

import sys
import os
import asyncio
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from trading.agents.multi_agent_system import multi_agent_system
from trading.execution.execution_engine import BinaryTradingEngine, RiskManager


def create_test_data(n_bars: int = 200, trend: str = "UP", volatility: str = "MEDIUM") -> pd.DataFrame:
    """Create synthetic OHLCV data for testing."""
    base_price = 100.0
    
    if trend == "UP":
        trend_component = np.linspace(0, 5, n_bars)
    elif trend == "DOWN":
        trend_component = np.linspace(0, -5, n_bars)
    else:
        trend_component = np.zeros(n_bars)
    
    if volatility == "HIGH":
        vol = 0.03
    elif volatility == "LOW":
        vol = 0.005
    else:
        vol = 0.015
    
    noise = np.random.normal(0, vol, n_bars)
    prices = base_price + trend_component + np.cumsum(noise)
    
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


def test_phase4_multi_agent():
    """Test Phase 4: Multi-Agent Decision System."""
    print("=" * 60)
    print("PHASE 4 — MULTI-AGENT DECISION SYSTEM TEST")
    print("=" * 60)
    
    df = create_test_data(n_bars=200, trend="UP", volatility="MEDIUM")
    print(f"Created {len(df)} bars of UP-trending data")
    
    signal = multi_agent_system.analyze(df)
    
    print("\n" + multi_agent_system.get_decision_summary(signal))
    
    return signal


def test_phase5_risk_manager():
    """Test Phase 5: Risk Management System."""
    print("\n" + "=" * 60)
    print("PHASE 5 — RISK MANAGEMENT SYSTEM TEST")
    print("=" * 60)
    
    # Create risk manager with test parameters
    rm = RiskManager(
        balance=100.0,
        max_risk_pct=1.0,          # 1% max stake
        max_drawdown_pct=10.0,
        daily_loss_limit=10.0,     # $10 daily loss limit
        max_open_trades=3,
        confidence_threshold=0.75, # 75% min confidence
        cooldown_seconds=60
    )
    
    print(f"Initial Balance: ${rm.balance:.2f}")
    print(f"Max Stake: ${rm.balance * (rm.max_risk_pct/100):.2f} ({rm.max_risk_pct}%)")
    print(f"Daily Loss Limit: ${rm.daily_loss_limit:.2f}")
    print(f"Confidence Threshold: {rm.confidence_threshold:.0%}")
    print(f"Max Open Trades: {rm.max_open_trades}")
    print(f"Cooldown: {rm.cooldown_seconds}s")
    
    # Test 1: Valid trade (high confidence)
    print("\n--- Test 1: Valid Trade (80% confidence, $2 stake) ---")
    result = rm.validate_trade(
        signal_direction="UP",
        signal_confidence=0.80,
        entry_price=100.0,
        recommended_stake=2.0,
        current_balance=rm.balance
    )
    print(f"Result: {'✅ APPROVED' if result['is_valid'] else '❌ REJECTED'}")
    print(f"Reason: {result['reason']}")
    
    # Test 2: Low confidence rejection
    print("\n--- Test 2: Low Confidence Rejection (60% confidence) ---")
    result = rm.validate_trade(
        signal_direction="UP",
        signal_confidence=0.60,
        entry_price=100.0,
        recommended_stake=2.0,
        current_balance=rm.balance
    )
    print(f"Result: {'✅ APPROVED' if result['is_valid'] else '❌ REJECTED'}")
    print(f"Reason: {result['reason']}")
    
    # Test 3: Stake too high
    print("\n--- Test 3: Stake Too High ($2 stake > $1 max) ---")
    result = rm.validate_trade(
        signal_direction="UP",
        signal_confidence=0.85,
        entry_price=100.0,
        recommended_stake=2.0,
        current_balance=rm.balance
    )
    print(f"Result: {'✅ APPROVED' if result['is_valid'] else '❌ REJECTED'}")
    print(f"Reason: {result['reason']}")
    
    # Test 4: Simulate consecutive losses triggering pause
    print("\n--- Test 4: Simulate 3 Consecutive Losses ---")
    for i in range(3):
        rm.update_after_trade("LOSS", -1.0)
        print(f"  Loss {i+1}: Consecutive={rm.consecutive_losses}, Paused={rm.is_paused}")
    
    # Test 5: After 3 losses, new trade should be rejected
    print("\n--- Test 5: Trade After 3 Losses (Should Reject) ---")
    result = rm.validate_trade(
        signal_direction="UP",
        signal_confidence=0.85,
        entry_price=100.0,
        recommended_stake=0.5,
        current_balance=rm.balance
    )
    print(f"Result: {'✅ APPROVED' if result['is_valid'] else '❌ REJECTED'}")
    print(f"Reason: {result['reason']}")
    
    # Test 6: Reset and test daily loss limit
    print("\n--- Test 6: Daily Loss Limit ($10 limit) ---")
    rm2 = RiskManager(balance=100.0, daily_loss_limit=10.0)
    for i in range(4):
        rm2.update_after_trade("LOSS", -3.0)
        print(f"  Loss {i+1}: Daily loss=${rm2.daily_loss_amount:.2f}, Paused={rm2.is_paused}")
    
    result = rm2.validate_trade("UP", 0.85, 100.0, 0.5, rm2.balance)
    print(f"Result: {'✅ APPROVED' if result['is_valid'] else '❌ REJECTED'}")
    print(f"Reason: {result['reason']}")
    
    print("\n✅ Phase 5 tests completed")


def test_full_integration():
    """Test full Phase 4 + Phase 5 integration."""
    print("\n" + "=" * 60)
    print("FULL INTEGRATION TEST: Phase 4 + Phase 5")
    print("=" * 60)
    
    # Create trading engine with Phase 5 risk parameters
    engine = BinaryTradingEngine(
        initial_balance=100.0,
        max_risk_pct=1.0,
        daily_loss_limit=10.0,
        max_open_trades=3,
        confidence_threshold=0.75,
        cooldown_seconds=60
    )
    
    # Create test market data
    df = create_test_data(n_bars=200, trend="UP", volatility="MEDIUM")
    current_price = df['close'].iloc[-1]
    
    print(f"Market: UP trending, Price: ${current_price:.2f}")
    print(f"Balance: ${engine.balance:.2f}")
    
    # Run multi-agent analysis
    print("\n--- Running Multi-Agent Analysis ---")
    decision = engine.process_multi_agent_signal(df, current_price)
    
    print(f"Decision Action: {decision['decision_action']}")
    print(f"Direction: {decision.get('direction', 'N/A')}")
    print(f"Confidence: {decision.get('confidence', 0):.0%}")
    print(f"Duration: {decision.get('duration_label', 'N/A')}")
    print(f"Trade Status: {decision['trade_status']}")
    
    if 'reject_reason' in decision:
        print(f"Reject Reason: {decision['reject_reason']}")
    
    if decision.get('trade_status') == 'PENDING_EXECUTION':
        print("\n--- Executing Trade ---")
        trade_result = asyncio.run(engine.execute_decision_trade(decision, current_price))
        print(f"Result: {trade_result['result']}")
        print(f"Stake: ${trade_result['stake_amount']:.2f}")
        print(f"Payout: ${trade_result['payout_amount']:.2f}")
        print(f"New Balance: ${trade_result['balance_after']:.2f}")
    
    # Check risk status
    print("\n--- Risk Status ---")
    risk_status = engine.risk_manager.get_risk_status()
    print(f"Balance: ${risk_status['balance']:.2f}")
    print(f"Daily P&L: ${risk_status['daily_pnl']:.2f}")
    print(f"Consecutive Losses: {risk_status['consecutive_losses']}")
    print(f"Open Trades: {risk_status['open_trades']}")
    print(f"Paused: {risk_status['is_paused']}")
    
    print("\n✅ Full integration test completed")


if __name__ == "__main__":
    # Test Phase 4
    test_phase4_multi_agent()
    
    # Test Phase 5
    test_phase5_risk_manager()
    
    # Test Full Integration
    test_full_integration()
    
    print("\n" + "=" * 60)
    print("ALL TESTS COMPLETED SUCCESSFULLY")
    print("=" * 60)