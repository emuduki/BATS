"""
Trading loop for Phase 7 - runs in background process.
"""

import asyncio
import time
import random
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Optional

from trading.execution.execution_engine import BinaryTradingEngine
from trading.broker import DemoBroker, DemoBrokerConfig
from trading.state import TradingState


def generate_live_data(n_bars: int = 200, base_price: float = 100.0) -> pd.DataFrame:
    """Generate realistic-looking OHLCV data."""
    dates = pd.date_range(start=datetime.now() - timedelta(hours=n_bars), periods=n_bars, freq='5min')
    
    # Random walk with trend
    returns = np.random.normal(0.0001, 0.002, n_bars)
    prices = base_price * np.exp(np.cumsum(returns))
    
    # Generate OHLC from close prices
    df = pd.DataFrame({
        'timestamp': dates,
        'open': prices * (1 + np.random.uniform(-0.001, 0.001, n_bars)),
        'high': prices * (1 + np.abs(np.random.uniform(0, 0.002, n_bars))),
        'low': prices * (1 - np.abs(np.random.uniform(0, 0.002, n_bars))),
        'close': prices,
        'volume': np.random.uniform(100, 1000, n_bars)
    })
    
    return df


async def run_trading_cycle(
    engine: BinaryTradingEngine,
    state: TradingState,
    symbol: str = "R_100",
    interval_seconds: int = 30
) -> dict:
    """Run a single trading cycle: analyze -> validate -> execute."""
    try:
        # Generate/load market data
        df = generate_live_data(n_bars=200)
        current_price = df['close'].iloc[-1]
        
        # Update state with current risk status
        risk_status = engine.risk_manager.get_risk_status()
        state.update_risk_status({
            "is_paused": risk_status.get("is_paused", False),
            "pause_reason": risk_status.get("pause_reason", ""),
            "daily_loss_amount": risk_status.get("daily_loss_amount", 0.0),
            "consecutive_losses": risk_status.get("consecutive_losses", 0),
            "open_trades": risk_status.get("open_trades", 0),
            "can_trade": risk_status.get("can_trade_now", True)
        })
        
        # Check if trading is allowed
        can_trade, reason = engine.risk_manager.can_trade_now()
        if not can_trade:
            return {
                "action": "SKIPPED",
                "reason": reason,
                "balance": engine.balance
            }
        
        # Run multi-agent analysis
        decision = engine.process_multi_agent_signal(df, current_price)
        
        # Update state with signal
        state.update_signal({
            "timestamp": datetime.now().isoformat(),
            "symbol": symbol,
            "direction": decision.get("direction"),
            "confidence": decision.get("confidence"),
            "duration": decision.get("duration_seconds"),
            "action": decision.get("decision_action"),
            "reason": decision.get("reason", "")
        })
        
        # Execute if pending
        if decision.get("trade_status") == "PENDING_EXECUTION":
            trade_result = await engine.execute_decision_trade(decision, current_price)
            
            # Update state with trade result
            state.update_trade({
                "timestamp": datetime.now().isoformat(),
                "symbol": symbol,
                "direction": trade_result.get("direction"),
                "stake": trade_result.get("stake_amount"),
                "result": trade_result.get("result"),
                "pnl": trade_result.get("payout_amount", 0),
                "balance_after": trade_result.get("balance_after"),
                "contract_id": trade_result.get("contract_id")
            })
            
            state.update_balance(engine.balance)
            
            return {
                "action": "EXECUTED",
                "result": trade_result.get("result"),
                "pnl": trade_result.get("payout_amount", 0),
                "balance": engine.balance
            }
        
        return {
            "action": decision.get("decision_action"),
            "reason": decision.get("reject_reason", decision.get("reason", "")),
            "balance": engine.balance
        }
        
    except Exception as e:
        return {
            "action": "ERROR",
            "error": str(e),
            "balance": engine.balance
        }


def trading_loop_background(
    symbol: str = "R_100",
    interval_seconds: int = 30,
    max_trades_per_session: int = 20
) -> None:
    """
    Main trading loop - runs in background process.
    
    This is the entry point for the background process.
    """
    print(f"[TRADING LOOP] Starting with symbol={symbol}, interval={interval_seconds}s")
    
    # Initialize broker and engine
    broker = DemoBroker(DemoBrokerConfig(
        initial_balance=100.0,
        win_probability=0.55,
        payout_rate=0.85
    ))
    
    engine = BinaryTradingEngine(
        initial_balance=100.0,
        max_risk_pct=1.0,
        daily_loss_limit=10.0,
        max_open_trades=3,
        confidence_threshold=0.55,
        cooldown_seconds=30,
        broker=broker,
        symbol=symbol
    )
    
    state = TradingState()
    
    # Reset state for new session
    state.save_state({
        "balance": 100.0,
        "daily_pnl": 0.0,
        "win_rate": 0.0,
        "active_trades": 0,
        "total_trades": 0,
        "winning_trades": 0,
        "last_signal": None,
        "last_trade": None,
        "risk_status": {
            "is_paused": False,
            "pause_reason": "",
            "daily_loss_amount": 0.0,
            "consecutive_losses": 0,
            "open_trades": 0,
            "can_trade": True
        },
        "session_started": datetime.now().isoformat(),
        "updated_at": datetime.now().isoformat()
    })
    
    trade_count = 0
    
    # Main loop
    while trade_count < max_trades_per_session:
        try:
            # Run async trading cycle
            result = asyncio.run(run_trading_cycle(engine, state, symbol, interval_seconds))
            
            print(f"[TRADING LOOP] Cycle: {result}")
            
            if result.get("action") == "EXECUTED":
                trade_count += 1
            
            # Wait for next cycle
            time.sleep(interval_seconds)
            
        except KeyboardInterrupt:
            print("[TRADING LOOP] Interrupted by user")
            break
        except Exception as e:
            print(f"[TRADING LOOP] Error: {e}")
            time.sleep(5)  # Wait before retrying
    
    print(f"[TRADING LOOP] Session ended. Total trades: {trade_count}")
    
    # Save final state
    state.save_state({
        "session_ended": datetime.now().isoformat(),
        "total_trades": trade_count,
        "final_balance": engine.balance
    })


if __name__ == "__main__":
    # For testing
    trading_loop_background(symbol="R_100", interval_seconds=5, max_trades_per_session=5)
