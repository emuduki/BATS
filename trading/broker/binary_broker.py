"""Abstract Binary Broker Interface - Phase 6 Broker Integration.
Defines the contract that all broker implementations must follow.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, Any, List, Optional
from datetime import datetime
from enum import Enum


class ContractType(Enum):
    """Binary option contract types."""
    RISE_FALL = "RISE_FALL"
    HIGHER_LOWER = "HIGHER_LOWER"
    TOUCH_NOTOUCH = "TOUCH_NOTOUCH"
    IN_OUT = "IN_OUT"


class OrderStatus(Enum):
    """Order execution status."""
    PENDING = "PENDING"
    PROPOSED = "PROPOSED"
    PURCHASED = "PURCHASED"
    CONFIRMED = "CONFIRMED"
    REJECTED = "REJECTED"
    EXPIRED = "EXPIRED"
    CANCELLED = "CANCELLED"


class TradeResult(Enum):
    """Trade settlement result."""
    WIN = "WIN"
    LOSS = "LOSS"
    REFUNDED = "REFUNDED"
    PENDING = "PENDING"


@dataclass
class MarketData:
    """Market data snapshot."""
    symbol: str
    bid: float
    ask: float
    spot: float
    timestamp: datetime
    source: str


@dataclass
class AccountBalance:
    """Account balance information."""
    balance: float
    currency: str
    equity: float
    available: float
    timestamp: datetime


@dataclass
class ContractSpec:
    """Available contract specification."""
    contract_type: ContractType
    symbol: str
    direction: str  # UP/DOWN, etc.
    min_stake: float
    max_stake: float
    min_duration: int  # seconds
    max_duration: int  # seconds
    available_durations: List[int]
    payout: float  # e.g., 0.85 for 85%


@dataclass
class Proposal:
    """Contract proposal from broker."""
    proposal_id: str
    contract_spec: ContractSpec
    stake: float
    duration: int
    direction: str
    payout: float
    spot_price: float
    timestamp: datetime
    raw_response: Dict[str, Any]


@dataclass
class Contract:
    """Purchased contract."""
    contract_id: str
    proposal_id: str
    buy_price: float
    payout: float
    stake: float
    symbol: str
    direction: str
    duration: int
    entry_spot: float
    expiry_time: datetime
    status: OrderStatus
    result: Optional[TradeResult] = None
    settlement_price: Optional[float] = None
    profit: Optional[float] = None
    raw_response: Dict[str, Any] = None


@dataclass
class BrokerError:
    """Broker error details."""
    code: str
    message: str
    details: Dict[str, Any] = None


class BinaryBroker(ABC):
    """
    Abstract Binary Broker Interface.
    
    All broker implementations (Deriv, Demo, etc.) must implement these methods.
    The rest of BATS communicates only through this interface.
    """
    
    def __init__(self, enforce_demo_mode: bool = True):
        self.enforce_demo_mode = enforce_demo_mode
        self._connected = False
        self._authenticated = False
    
    @abstractmethod
    async def connect(self) -> bool:
        """Establish connection to broker. Returns True on success."""
        pass
    
    @abstractmethod
    async def disconnect(self) -> bool:
        """Close connection to broker."""
        pass
    
    @abstractmethod
    async def is_connected(self) -> bool:
        """Check if connected to broker."""
        pass
    
    @abstractmethod
    async def get_balance(self) -> AccountBalance:
        """Get current account balance."""
        pass
    
    @abstractmethod
    async def get_market_data(self, symbol: str) -> MarketData:
        """Get current market data for symbol."""
        pass
    
    @abstractmethod
    async def get_available_contracts(self, symbol: str) -> List[ContractSpec]:
        """Get available contract types and parameters for symbol."""
        pass
    
    @abstractmethod
    async def create_proposal(
        self,
        symbol: str,
        direction: str,
        stake: float,
        duration: int,
        contract_type: ContractType = ContractType.RISE_FALL
    ) -> Proposal:
        """
        Request a contract proposal from broker.
        
        Args:
            symbol: Trading symbol (e.g., "R_100")
            direction: "UP" or "DOWN"
            stake: Stake amount in USD
            duration: Duration in seconds
            contract_type: Type of binary contract
            
        Returns:
            Proposal with broker-calculated payout and terms
        """
        pass
    
    @abstractmethod
    async def buy_contract(self, proposal_id: str) -> Contract:
        """
        Purchase a contract using a proposal ID.
        
        Args:
            proposal_id: ID from create_proposal
            
        Returns:
            Contract with contract_id and purchase details
        """
        pass
    
    @abstractmethod
    async def monitor_contract(self, contract_id: str) -> Contract:
        """
        Monitor contract until expiration.
        
        Args:
            contract_id: ID from buy_contract
            
        Returns:
            Updated Contract with result (WIN/LOSS)
        """
        pass
    
    @abstractmethod
    async def get_contract_status(self, contract_id: str) -> Contract:
        """Get current status of a contract."""
        pass
    
    @abstractmethod
    async def cancel_contract(self, contract_id: str) -> bool:
        """Attempt to cancel a pending contract (if supported)."""
        pass


class BrokerException(Exception):
    """Base exception for broker errors."""
    
    def __init__(self, message: str, code: str = None, details: Dict = None):
        self.message = message
        self.code = code
        self.details = details or {}
        super().__init__(message)


class ConnectionError(BrokerException):
    """Connection-related errors."""
    pass


class AuthenticationError(BrokerException):
    """Authentication/authorization errors."""
    pass


class ProposalError(BrokerException):
    """Proposal creation errors."""
    pass


class PurchaseError(BrokerException):
    """Contract purchase errors."""
    pass


class MonitoringError(BrokerException):
    """Contract monitoring errors."""
    pass