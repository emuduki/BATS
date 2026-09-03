"""Base Broker Interface - Phase 6.1.
Abstract interface that all broker implementations must follow.
Separates BATS from any specific broker API.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional
from datetime import datetime
from enum import Enum
import uuid

from trading.broker.binary_broker import (
    ContractType, OrderStatus, TradeResult,
    MarketData, AccountBalance, ContractSpec, Proposal, Contract, BinaryBroker
)


class ConnectionState(Enum):
    """Broker connection states."""
    DISCONNECTED = "disconnected"
    CONNECTING = "connecting"
    CONNECTED = "connected"
    AUTHENTICATING = "authenticating"
    AUTHENTICATED = "authenticated"
    READY = "ready"
    ERROR = "error"


class AccountMode(Enum):
    """Account type (demo vs real)."""
    DEMO = "demo"
    REAL = "real"


@dataclass
class AccountInfo:
    """Verified account information."""
    account_id: str
    account_mode: AccountMode
    currency: str
    balance: float
    equity: float
    trading_enabled: bool
    permissions: List[str]
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class SymbolCapability:
    """Broker-supported symbol and its capabilities."""
    symbol: str
    display_name: str
    contract_types: List[str]
    available_durations: Dict[str, List[int]]  # unit -> durations
    min_stake: float
    max_stake: float
    payout_percentage: float
    trading_available: bool


@dataclass
class BrokerCapabilities:
    """Discovered broker capabilities."""
    symbols: Dict[str, SymbolCapability] = field(default_factory=dict)
    supported_durations: List[str] = field(default_factory=list)  # ["s", "m", "t"]
    supported_contract_types: List[str] = field(default_factory=list)
    timestamp: datetime = field(default_factory=datetime.now)


class BaseBroker(ABC):
    """
    Abstract Broker Interface.
    
    All broker implementations must implement this interface.
    BATS communicates only through this abstraction.
    
    Architecture:
    
    ML / Strategy Models
            ↓
    Multi-Agent Decision
            ↓
    Risk Manager
            ↓
    Execution Engine
            ↓
    Broker Adapter (this layer)
            ↓
    Broker API / WebSocket
    """
    
    def __init__(self, enforce_demo_mode: bool = True):
        """
        Args:
            enforce_demo_mode: If True, rejects all live trading attempts
        """
        self.enforce_demo_mode = enforce_demo_mode
        self._connection_state = ConnectionState.DISCONNECTED
        self._account_info: Optional[AccountInfo] = None
        self._capabilities: Optional[BrokerCapabilities] = None
        self._last_error: Optional[str] = None
        
    @property
    def connection_state(self) -> ConnectionState:
        return self._connection_state
    
    @property
    def account_info(self) -> Optional[AccountInfo]:
        return self._account_info
    
    @property
    def capabilities(self) -> Optional[BrokerCapabilities]:
        return self._capabilities
    
    @property
    def is_ready(self) -> bool:
        return (
            self._connection_state == ConnectionState.READY and
            self._account_info is not None and
            self._capabilities is not None
        )
    
    # === Connection Management ===
    
    @abstractmethod
    async def connect(self) -> bool:
        """Establish connection to broker. Returns True on success."""
        pass
    
    @abstractmethod
    async def disconnect(self) -> bool:
        """Close connection to broker."""
        pass
    
    @abstractmethod
    async def reconnect(self, max_attempts: int = 3) -> bool:
        """Attempt to reconnect with retries."""
        pass
    
    # === Authentication ===
    
    @abstractmethod
    async def authenticate(self, credentials: Dict[str, str]) -> bool:
        """
        Authenticate with broker using credentials.
        
        Args:
            credentials: Dict with required auth info (e.g., {"api_token": "..."})
            
        Returns:
            True if authenticated successfully
        """
        pass
    
    @abstractmethod
    async def verify_account(self) -> AccountInfo:
        """
        Verify account and return account information.
        Enforces demo mode if configured.
        """
        pass
    
    # === Capability Discovery ===
    
    @abstractmethod
    async def discover_capabilities(self) -> BrokerCapabilities:
        """
        Query broker for available symbols, contracts, durations.
        Should be called after authentication.
        """
        pass
    
    @abstractmethod
    async def get_symbol_info(self, symbol: str) -> SymbolCapability:
        """Get capabilities for a specific symbol."""
        pass
    
    # === Order Proposal ===
    
    @abstractmethod
    async def request_proposal(
        self,
        symbol: str,
        direction: str,
        stake: float,
        duration: int,
        duration_unit: str = "s"
    ) -> Dict[str, Any]:
        """
        Request a contract proposal from broker.
        
        Returns broker-generated proposal with terms and pricing.
        Does NOT execute the trade.
        """
        pass
    
    # === Trade Execution ===
    
    @abstractmethod
    async def execute_trade(self, proposal: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute a trade using a previously obtained proposal.
        
        Args:
            proposal: Proposal dict from request_proposal()
            
        Returns:
            Execution result with contract_id
        """
        pass
    
    # === Contract Monitoring ===
    
    @abstractmethod
    async def get_contract_status(self, contract_id: str) -> Dict[str, Any]:
        """Get current status of a contract."""
        pass
    
    @abstractmethod
    async def subscribe_contract(
        self, 
        contract_id: str, 
        callback=None
    ) -> None:
        """Subscribe to contract updates (if broker supports it)."""
        pass
    
    @abstractmethod
    async def unsubscribe_contract(self, contract_id: str) -> None:
        """Unsubscribe from contract updates."""
        pass
    
    # === Utility ===
    
    @abstractmethod
    async def get_balance(self) -> float:
        """Get current account balance."""
        pass
    
    @abstractmethod
    async def health_check(self) -> Dict[str, Any]:
        """
        Check broker connection health.
        
        Returns:
            Dict with connection_state, latency, etc.
        """
        pass
    
    # === Validation Helpers ===
    
    def validate_symbol(self, symbol: str) -> bool:
        """Check if symbol is supported."""
        if self._capabilities and symbol in self._capabilities.symbols:
            return self._capabilities.symbols[symbol].trading_available
        return False
    
    def validate_direction(self, direction: str) -> bool:
        """Check if direction is valid."""
        return direction.upper() in ["UP", "DOWN", "CALL", "PUT"]
    
    def validate_duration(self, symbol: str, duration: int, unit: str = "s") -> bool:
        """Check if duration is valid for symbol."""
        if not self._capabilities or symbol not in self._capabilities.symbols:
            return False
        cap = self._capabilities.symbols[symbol]
        if unit in cap.available_durations:
            return duration in cap.available_durations[unit]
        return False
    
    def validate_stake(self, symbol: str, stake: float) -> bool:
        """Check if stake is within limits for symbol."""
        if not self._capabilities or symbol not in self._capabilities.symbols:
            return False
        cap = self._capabilities.symbols[symbol]
        return cap.min_stake <= stake <= cap.max_stake
    
    def enforce_demo_mode_check(self) -> bool:
        """Check if live trading should be rejected."""
        if not self.enforce_demo_mode:
            return True
        if self._account_info and self._account_info.account_mode != AccountMode.DEMO:
            self._last_error = "Live trading blocked: Demo mode enforced"
            return False
        return True


class BrokerException(Exception):
    """Base exception for broker errors."""
    
    def __init__(self, message: str, code: str = None, details: Dict = None):
        self.message = message
        self.code = code
        self.details = details or {}
        super().__init__(message)


class ConnectionException(BrokerException):
    """Connection-related errors."""
    pass


class AuthenticationException(BrokerException):
    """Authentication errors."""
    pass


class ProposalException(BrokerException):
    """Proposal creation errors."""
    pass


class ExecutionException(BrokerException):
    """Trade execution errors."""
    pass


class MonitoringException(BrokerException):
    """Contract monitoring errors."""
    pass
