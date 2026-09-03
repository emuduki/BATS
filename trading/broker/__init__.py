"""Broker package for Phase 6 integration.
Provides abstract interface and implementations for binary options brokers.
"""

# Base interface and core models
from trading.broker.binary_broker import (
    BinaryBroker,
    ContractType,
    OrderStatus,
    TradeResult,
    MarketData,
    AccountBalance,
    ContractSpec,
    Proposal,
    Contract,
    BrokerError,
    BrokerException,
    ConnectionError,
    AuthenticationError,
    ProposalError,
    PurchaseError,
    MonitoringError
)

from trading.broker.base_broker import (
    BaseBroker,
    ConnectionState,
    AccountMode,
    AccountInfo,
    SymbolCapability,
    BrokerCapabilities,
    ConnectionException,
    AuthenticationException,
    ProposalException,
    ExecutionException,
    MonitoringException
)

# Connection management
from trading.broker.connection_manager import (
    ConnectionManager,
    BrokerState,
    get_connection_manager,
    reset_connection_manager
)

# Broker implementations
from trading.broker.demo_broker import DemoBroker, DemoBrokerConfig
from trading.broker.deriv_broker import DerivBroker

__all__ = [
    # Base interface
    "BaseBroker",
    "ConnectionState",
    "AccountMode",
    "AccountInfo",
    "SymbolCapability",
    "BrokerCapabilities",
    
    # Exceptions
    "BrokerException",
    "ConnectionException",
    "AuthenticationException",
    "ProposalException",
    "ExecutionException",
    "MonitoringException",
    "ConnectionError",
    "AuthenticationError",
    "ProposalError",
    "PurchaseError",
    
    # Core abstractions and data models
    "BinaryBroker",
    "ContractType",
    "OrderStatus",
    "TradeResult",
    "MarketData",
    "AccountBalance",
    "ContractSpec",
    "Proposal",
    "Contract",
    "BrokerError",
    
    # Connection management
    "ConnectionManager",
    "BrokerState",
    "get_connection_manager",
    "reset_connection_manager",
    
    # Implementations
    "DemoBroker",
    "DemoBrokerConfig",
    "DerivBroker"
]