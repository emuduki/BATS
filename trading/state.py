"""
Trading state management for Phase 7 Dashboard & Monitoring.
Handles persistence of trading state to JSON file.
"""

import json
import os
from datetime import datetime
from typing import Dict, Any, Optional


class TradingState:
    """Manages trading state persistence."""
    
    def __init__(self, filepath: str = "trading_state.json"):
        self.filepath = filepath
        self.state: Dict[str, Any] = self._load_state()
    
    def _load_state(self) -> Dict[str, Any]:
        """Load state from file, or return default if file doesn't exist."""
        default_state = {
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
            "updated_at": datetime.now().isoformat()
        }
        
        if os.path.exists(self.filepath):
            try:
                with open(self.filepath, 'r') as f:
                    state = json.load(f)
                # Ensure all keys are present
                for key, value in default_state.items():
                    if key not in state:
                        state[key] = value
                return state
            except Exception:
                return default_state
        return default_state
    
    def save_state(self, state: Optional[Dict[str, Any]] = None) -> None:
        """Save state to file."""
        if state is not None:
            self.state = state
        self.state["updated_at"] = datetime.now().isoformat()
        with open(self.filepath, 'w') as f:
            json.dump(self.state, f, indent=2)
    
    def get_state(self) -> Dict[str, Any]:
        """Get current state."""
        return self.state.copy()
    
    def update_balance(self, balance: float) -> None:
        """Update balance and recalculate daily P&L."""
        self.state["balance"] = balance
        # Daily P&L is balance - starting balance (we don't store starting balance, so approximate)
        # In a real system, we'd store starting balance of the day
        self.state["daily_pnl"] = balance - 100.0  # Assuming starting balance of 100
        self.save_state()
    
    def update_trade(self, trade_result: Dict[str, Any]) -> None:
        """Update trade statistics."""
        self.state["total_trades"] += 1
        if trade_result.get("result") == "WIN":
            self.state["winning_trades"] += 1
        self.state["win_rate"] = (
            self.state["winning_trades"] / self.state["total_trades"]
            if self.state["total_trades"] > 0 else 0.0
        )
        self.state["last_trade"] = trade_result
        self.save_state()
    
    def update_signal(self, signal: Dict[str, Any]) -> None:
        """Update last signal."""
        self.state["last_signal"] = signal
        self.save_state()
    
    def update_risk_status(self, risk_status: Dict[str, Any]) -> None:
        """Update risk status."""
        self.state["risk_status"] = risk_status
        self.save_state()
    
    def increment_active_trades(self) -> None:
        """Increment active trades count."""
        self.state["active_trades"] += 1
        self.save_state()
    
    def decrement_active_trades(self) -> None:
        """Decrement active trades count."""
        self.state["active_trades"] = max(0, self.state["active_trades"] - 1)
        self.save_state()


# Global state instance
trading_state = TradingState()