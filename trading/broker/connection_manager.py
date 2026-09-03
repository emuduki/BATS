"""Broker Connection Manager - Phase 6.3, 6.6, 6.9.
Manages connection, authentication, reconnection, and account synchronization.
"""

import asyncio
from typing import Dict, Any, Optional, Callable
from datetime import datetime, timedelta
from enum import Enum
import logging

from trading.broker.base_broker import (
    BaseBroker, ConnectionState, AccountInfo, 
    ConnectionException, AuthenticationException
)


logger = logging.getLogger("BATS.ConnectionManager")


class BrokerState(Enum):
    """High-level broker states for UI display."""
    UNKNOWN = "UNKNOWN"
    DISCONNECTED = "DISCONNECTED"
    CONNECTING = "CONNECTING"
    AUTHENTICATING = "AUTHENTICATING"
    VERIFYING_ACCOUNT = "VERIFYING_ACCOUNT"
    DISCOVERING_CAPABILITIES = "DISCOVERING_CAPABILITIES"
    CONNECTED = "CONNECTED"
    READY = "READY"
    PAUSED = "PAUSED"
    ERROR = "ERROR"


class ConnectionManager:
    """
    Manages broker connection lifecycle with:
    - Auto-reconnection
    - Account synchronization
    - State management
    - Health monitoring
    """
    
    def __init__(
        self,
        broker: BaseBroker,
        reconnect_interval: float = 5.0,
        max_reconnect_attempts: int = 10
    ):
        self.broker = broker
        self.reconnect_interval = reconnect_interval
        self.max_reconnect_attempts = max_reconnect_attempts
        
        self._state = BrokerState.UNKNOWN
        self._last_connected: Optional[datetime] = None
        self._reconnect_attempts = 0
        self._connection_task: Optional[asyncio.Task] = None
        self._health_check_task: Optional[asyncio.Task] = None
        
    @property
    def state(self) -> BrokerState:
        """Current broker state."""
        return self._state
    
    @property
    def is_trading_ready(self) -> bool:
        """Check if trading is enabled."""
        return (
            self._state == BrokerState.READY and
            self.broker.is_ready and
            self.broker.enforce_demo_mode_check()
        )
    
    async def connect_and_authenticate(
        self,
        credentials: Dict[str, Any]
    ) -> bool:
        """
        Full connection and authentication sequence.
        
        Returns:
            True if fully connected and authenticated
        """
        self._set_state(BrokerState.CONNECTING)
        
        try:
            # Step 1: Connect
            logger.info("Connecting to broker...")
            if not await self.broker.connect():
                raise ConnectionException("Connection failed")
            
            self._set_state(BrokerState.AUTHENTICATING)
            
            # Step 2: Authenticate
            logger.info("Authenticating...")
            if not await self.broker.authenticate(credentials):
                raise AuthenticationException("Authentication failed")
            
            # Step 3: Verify account
            self._set_state(BrokerState.VERIFYING_ACCOUNT)
            logger.info("Verifying account...")
            account_info = await self.broker.verify_account()
            
            # Step 4: Discover capabilities
            self._set_state(BrokerState.DISCOVERING_CAPABILITIES)
            logger.info("Discovering capabilities...")
            capabilities = await self.broker.discover_capabilities()
            
            self._set_state(BrokerState.READY)
            self._last_connected = datetime.now()
            self._reconnect_attempts = 0
            
            logger.info("Broker connection READY")
            return True
            
        except Exception as e:
            logger.error(f"Connection failed: {e}")
            self._transition_to_error(str(e))
            return False
    
    async def disconnect(self) -> bool:
        """Gracefully disconnect from broker."""
        if self._connection_task:
            self._connection_task.cancel()
            self._connection_task = None
        
        if self._health_check_task:
            self._health_check_task.cancel()
            self._health_check_task = None
        
        await self.broker.disconnect()
        self._set_state(BrokerState.DISCONNECTED)
        return True
    
    async def start_auto_reconnect(self, credentials: Dict[str, Any]) -> None:
        """Start background task for auto-reconnection."""
        if self._connection_task:
            return  # Already running
        
        self._connection_task = asyncio.create_task(
            self._reconnect_loop(credentials)
        )
    
    async def _reconnect_loop(self, credentials: Dict[str, Any]) -> None:
        """Background reconnection loop."""
        while True:
            if not self.broker.is_connected:
                logger.warning(f"Connection lost, reconnect attempt {self._reconnect_attempts + 1}")
                
                try:
                    success = await self.connect_and_authenticate(credentials)
                    if success:
                        logger.info("Reconnected successfully")
                    else:
                        raise Exception("Reconnect failed")
                        
                except Exception as e:
                    self._reconnect_attempts += 1
                    
                    if self._reconnect_attempts >= self.max_reconnect_attempts:
                        logger.error(f"Max reconnect attempts ({self.max_reconnect_attempts}) reached")
                        break
                    
                    logger.warning(f"Reconnect failed: {e}. Retrying in {self.reconnect_interval}s")
                    await asyncio.sleep(self.reconnect_interval)
            
            await asyncio.sleep(self.reconnect_interval)
    
    def pause_trading(self, reason: str = ""):
        """Pause trading operations."""
        self._set_state(BrokerState.PAUSED)
        logger.warning(f"Trading paused: {reason}")
    
    def resume_trading(self):
        """Resume trading operations after pause."""
        if self._state == BrokerState.PAUSED:
            self._set_state(BrokerState.READY)
            logger.info("Trading resumed")
    
    def force_error_state(self, error_message: str):
        """Force error state (used when broker fails unexpectedly)."""
        self._transition_to_error(error_message)
    
    def _set_state(self, state: BrokerState):
        """Update broker state."""
        old_state = self._state
        self._state = state
        logger.debug(f"Broker state: {old_state.value} -> {state.value}")
    
    def _transition_to_error(self, error_message: str):
        """Transition to error state."""
        self._state = BrokerState.ERROR
        self._last_error = error_message
        logger.error(f"Broker in ERROR state: {error_message}")
    
    async def health_check(self) -> Dict[str, Any]:
        """
        Get comprehensive health status.
        
        Returns:
            Dict with connection status, latency, account info, etc.
        """
        try:
            health = await self.broker.health_check()
            health["broker_state"] = self._state.value
            health["is_trading_enabled"] = self.is_trading_ready
            health["account_mode"] = (
                self.broker.account_info.account_mode.value 
                if self.broker.account_info else "unknown"
            )
            health["balance"] = self.broker.account_info.balance if self.broker.account_info else 0
            health["last_connected"] = (
                self._last_connected.isoformat() if self._last_connected else None
            )
            return health
        except Exception as e:
            return {
                "status": "error",
                "error": str(e),
                "broker_state": self._state.value,
                "is_trading_enabled": False
            }
    
    def get_connection_log(self, hours: int = 24) -> list:
        """Get connection events for the last N hours."""
        events = []
        
        if self._last_connected:
            cutoff = datetime.now() - timedelta(hours=hours)
            if self._last_connected >= cutoff:
                events.append({
                    "time": self._last_connected.isoformat(),
                    "event": "connected",
                    "state": self._state.value
                })
        
        if self._state == BrokerState.ERROR:
            events.append({
                "time": datetime.now().isoformat(),
                "event": "error",
                "state": self._state.value,
                "message": getattr(self, '_last_error', None)
            })
        
        return events
    
    # === Convenience Methods ===
    
    async def get_available_symbols(self) -> list:
        """Get list of available symbols."""
        if not self.broker.capabilities:
            return []
        return list(self.broker.capabilities.symbols.keys())
    
    def is_symbol_available(self, symbol: str) -> bool:
        """Check if symbol is available for trading."""
        return self.broker.validate_symbol(symbol)
    
    def can_execute_trade(
        self,
        symbol: str,
        direction: str,
        stake: float,
        duration: int,
        duration_unit: str = "s"
    ) -> Dict[str, bool]:
        """
        Pre-validate trade parameters before sending to broker.
        
        Returns:
            Dict with validation results for each check
        """
        results = {
            "symbol_valid": False,
            "direction_valid": False,
            "stake_valid": False,
            "duration_valid": False,
            "demo_mode_ok": False,
            "overall_ok": False
        }
        
        if not self.is_trading_ready:
            return results
        
        results["symbol_valid"] = self.broker.validate_symbol(symbol)
        results["direction_valid"] = self.broker.validate_direction(direction)
        results["stake_valid"] = self.broker.validate_stake(symbol, stake)
        results["duration_valid"] = self.broker.validate_duration(symbol, duration, duration_unit)
        results["demo_mode_ok"] = self.broker.enforce_demo_mode_check()
        
        results["overall_ok"] = all(results.values())
        
        return results


# Singleton instance helper
_connection_manager: Optional[ConnectionManager] = None


def get_connection_manager(broker: BaseBroker = None) -> ConnectionManager:
    """Get or create the singleton connection manager."""
    global _connection_manager
    if _connection_manager is None and broker is not None:
        _connection_manager = ConnectionManager(broker)
    return _connection_manager


def reset_connection_manager():
    """Reset the singleton (for testing)."""
    global _connection_manager
    _connection_manager = None