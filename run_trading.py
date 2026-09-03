# run_trading.py
from trading.broker import DerivBroker
from trading.execution.execution_engine import BinaryTradingEngine

# Replace with your actual token from Deriv.com
broker = DerivBroker(
    api_token="YOUR_COPIED_TOKEN_HERE",  # Get from https://app.deriv.com/
    demo_mode=True  # Keep True while testing (set False for real money)
)

engine = BinaryTradingEngine(
    broker=broker, 
    symbol="R_100",
    initial_balance=100.0,
    max_risk_pct=1.0,
    daily_loss_limit=10.0,
    max_open_trades=3,
    confidence_threshold=0.75,
    cooldown_seconds=60
)

# Example: Run one analysis cycle (you'd typically run this in a loop)
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Generate sample data for testing
def generate_sample_data():
    base_price = 100.0
    n_bars = 200
    trend = np.linspace(0, 2, n_bars)  # Slight upward trend
    noise = np.random.normal(0, 0.015, n_bars)
    prices = base_price + trend + np.cumsum(noise)
    
    dates = pd.date_range(start=datetime.now() - timedelta(hours=n_bars), periods=n_bars, freq='5min')
    
    df = pd.DataFrame({
        'timestamp': dates,
        'open': prices,
        'high': prices + np.abs(np.random.normal(0, 0.005, n_bars)),
        'low': prices - np.abs(np.random.normal(0, 0.005, n_bars)),
        'close': prices + np.random.normal(0, 0.01, n_bars),
        'volume': np.random.uniform(100, 1000, n_bars)
    })
    return df

# Run analysis
df = generate_sample_data()
current_price = df['close'].iloc[-1]

# Get multi-agent decision
from trading.agents.multi_agent_system import multi_agent_system
decision = multi_agent_system.analyze(df)

print("=== BATS Trading System ===")
print(f"Symbol: R_100")
print(f"Current Price: ${current_price:.2f}")
print(f"Decision: {decision.decision.action.value}")
print(f"Direction: {decision.decision.direction.value if decision.decision.direction else 'N/A'}")
print(f"Confidence: {decision.decision.confidence:.0%}")
print(f"duration_label: {decision.duration.duration_label}")

# Check risk approval
risk_result = engine.risk_manager.validate_trade(
    signal_direction=decision.decision.direction.value if decision.decision.direction else "UP",
    signal_confidence=decision.decision.confidence,
    entry_price=current_price,
    recommended_stake=1.0,
    current_balance=engine.balance
)

print(f"Risk Check: {'APPROVED' if risk_result['is_valid'] else 'REJECTED'}")
print(f"Reason: {risk_result['reason']}")

if risk_result['is_valid'] and decision.decision.action.value != "NO_TRADE":
    print("Trade would be executed!")
else:
    print("Trade blocked by risk management or decision system")