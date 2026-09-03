"""Demo Broker Implementation - Phase 6.
Simulates broker API for safe testing without real money.
Uses configurable parameters to control simulation behavior.
"""

import asyncio
from dataclasses import dataclass
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
import random
import uuid

from trading.broker.base_broker import (
    AccountInfo, SymbolCapability, BrokerCapabilities, ConnectionState, AccountMode
)
from trading.broker.binary_broker import (
    BinaryBroker, ContractType, OrderStatus, TradeResult,
    MarketData, AccountBalance, ContractSpec, Proposal, Contract,
    BrokerException, ConnectionError, AuthenticationError,
    ProposalError, PurchaseError, MonitoringError
)

ConnectionException = ConnectionError
AuthenticationException = AuthenticationError
ProposalException = ProposalError
ExecutionException = MonitoringError


@dataclass
class DemoBrokerConfig:
    """Configuration for DemoBroker simulation."""
    initial_balance: float = 100.0
    currency: str = "USD"
    volatility: float = 0.02  # Price movement volatility
    win_probability: float = 0.55  # Simulated win rate (can be biased by direction)
    payout_rate: float = 0.85  # Standard binary options payout
    connection_delay: float = 0.1  # Simulated network latency
    proposal_validity: float = 1.0  # Probability proposal succeeds
    min_stake: float = 0.1
    max_stake: float = 100.0
    cooldowns: bool = True  # Simulate cooldown after trade
    crash_mode: bool = False  # Force losses for testing risk manager


class DemoBroker(BinaryBroker):
    """
    Demo Broker for safe testing of BATS.
    
    Simulates broker behavior including:
    - Market data fluctuations
    - Random win/loss outcomes
    - Balance tracking
    - Contract lifecycle
    - Error scenarios (when configured)
    """
    
    def __init__(self, config: Optional[DemoBrokerConfig] = None, enforce_demo_mode: bool = True):
        super().__init__(enforce_demo_mode=enforce_demo_mode)
        self.config = config or DemoBrokerConfig()
        self._connected = False
        self._balance = self.config.initial_balance
        self._contracts: Dict[str, Contract] = {}
        self._proposals: Dict[str, Proposal] = {}
        self._symbols = ["R_100", "R_500", "DEMO_SYMBOL"]
        self._prices = {"R_100": 100.0, "R_500": 500.0, "DEMO_SYMBOL": 100.0}
        self._last_trade_time: Optional[datetime] = None
    
    async def connect(self) -> bool:
        await asyncio.sleep(self.config.connection_delay)
        self._connected = True
        return True
    
    async def disconnect(self) -> bool:
        self._connected = False
        return True
    
    async def is_connected(self) -> bool:
        return self._connected
    
    async def get_balance(self) -> AccountBalance:
        if not self._connected:
            raise ConnectionError("Not connected to broker")
        
        await asyncio.sleep(self.config.connection_delay)
        return AccountBalance(
            balance=self._balance,
            currency=self.config.currency,
            equity=self._balance,
            available=self._balance,
            timestamp=datetime.now()
        )
    
    async def get_market_data(self, symbol: str) -> MarketData:
        if not self._connected:
            raise ConnectionError("Not connected to broker")
        
        if symbol not in self._symbols:
            raise BrokerException(f"Unknown symbol: {symbol}")
        
        await asyncio.sleep(self.config.connection_delay)
        
        # Simulate price movement
        price = self._prices[symbol]
        if not self.config.crash_mode:
            volatility = self.config.volatility * (1 if symbol == "R_100" else 0.5)
            price_change = random.gauss(0, volatility)
            self._prices[symbol] = max(0.01, price * (1 + price_change))
        
        return MarketData(
            symbol=symbol,
            bid=round(self._prices[symbol] * (1 - 0.0001), 4),
            ask=round(self._prices[symbol] * (1 + 0.0001), 4),
            spot=round(self._prices[symbol], 4),
            timestamp=datetime.now(),
            source="DemoBroker"
        )
    
    async def get_available_contracts(self, symbol: str) -> List[ContractSpec]:
        if not self._connected:
            raise ConnectionError("Not connected to broker")
        
        if symbol not in self._symbols:
            raise BrokerException(f"Unknown symbol: {symbol}")
        
        await asyncio.sleep(self.config.connection_delay * 0.5)
        
        return [
            ContractSpec(
                contract_type=ContractType.RISE_FALL,
                symbol=symbol,
                direction="UP",
                min_stake=self.config.min_stake,
                max_stake=self.config.max_stake,
                min_duration=10,
                max_duration=300,
                available_durations=[15, 30, 60, 120, 180, 300],
                payout=self.config.payout_rate
            ),
            ContractSpec(
                contract_type=ContractType.RISE_FALL,
                symbol=symbol,
                direction="DOWN",
                min_stake=self.config.min_stake,
                max_stake=self.config.max_stake,
                min_duration=10,
                max_duration=300,
                available_durations=[15, 30, 60, 120, 180, 300],
                payout=self.config.payout_rate
            )
        ]
    
    async def create_proposal(
        self,
        symbol: str,
        direction: str,
        stake: float,
        duration: int,
        contract_type: ContractType = ContractType.RISE_FALL
    ) -> Proposal:
        if not self._connected:
            raise ConnectionError("Not connected to broker")
        
        if symbol not in self._symbols:
            raise ProposalError(f"Unknown symbol: {symbol}")
        
        if stake < self.config.min_stake or stake > self.config.max_stake:
            raise ProposalError(f"Stake {stake} outside allowed range")
        
        if duration < 10 or duration > 300:
            raise ProposalError(f"Duration {duration}s outside allowed range")
        
        if direction not in ["UP", "DOWN"]:
            raise ProposalError(f"Invalid direction: {direction}")
        
        if random.random() > self.config.proposal_validity:
            raise ProposalError("Proposal generation failed")
        
        await asyncio.sleep(self.config.connection_delay)
        
        # Get current market data
        market_data = await self.get_market_data(symbol)
        
        proposal_id = f"PROP_{uuid.uuid4().hex[:12]}"
        
        proposal = Proposal(
            proposal_id=proposal_id,
            contract_spec=ContractSpec(
                contract_type=contract_type,
                symbol=symbol,
                direction=direction,
                min_stake=self.config.min_stake,
                max_stake=self.config.max_stake,
                min_duration=10,
                max_duration=300,
                available_durations=[15, 30, 60, 120, 180, 300],
                payout=self.config.payout_rate
            ),
            stake=stake,
            duration=duration,
            direction=direction,
            payout=self.config.payout_rate,
            spot_price=market_data.spot,
            timestamp=datetime.now(),
            raw_response={
                "proposal_id": proposal_id,
                "symbol": symbol,
                "direction": direction,
                "stake": stake,
                "duration": duration,
                "payout": self.config.payout_rate,
                "spot": market_data.spot
            }
        )
        self._proposals[proposal_id] = proposal
        return proposal
    
    async def buy_contract(self, proposal_id: str) -> Contract:
        if not self._connected:
            raise ConnectionError("Not connected to broker")
        
        if proposal_id not in self._proposals:
            # Fetch from internal state if we stored it
            if hasattr(self, '_pending_proposals'):
                if proposal_id not in self._pending_proposals:
                    raise PurchaseError(f"Unknown proposal: {proposal_id}")
                proposal = self._pending_proposals[proposal_id]
            else:
                raise PurchaseError(f"Unknown proposal: {proposal_id}")
        else:
            proposal = self._proposals[proposal_id]
        
        await asyncio.sleep(self.config.connection_delay)
        
        # Check balance
        if self._balance < proposal.stake:
            raise PurchaseError("Insufficient balance")
        
        # Check cooldown
        if (self.config.cooldowns and self._last_trade_time and 
            (datetime.now() - self._last_trade_time).total_seconds() < 60):
            raise PurchaseError("Cooldown period active")
        
        contract_id = f"CONTRACT_{uuid.uuid4().hex[:12]}"
        expiry_time = datetime.now() + timedelta(seconds=proposal.duration)
        
        # Deduct stake from balance
        self._balance -= proposal.stake
        
        contract = Contract(
            contract_id=contract_id,
            proposal_id=proposal_id,
            buy_price=proposal.stake,
            payout=proposal.payout,
            stake=proposal.stake,
            symbol=proposal.contract_spec.symbol if hasattr(proposal, 'contract_spec') else getattr(proposal, 'symbol', 'R_100'),
            direction=proposal.direction,
            duration=proposal.duration,
            entry_spot=proposal.spot_price,
            expiry_time=expiry_time,
            status=OrderStatus.PURCHASED,
            raw_response={
                "contract_id": contract_id,
                "proposal_id": proposal_id,
                "expiry": expiry_time.isoformat()
            }
        )
        
        self._contracts[contract_id] = contract
        if not hasattr(self, '_pending_proposals'):
            self._pending_proposals = {}
        self._pending_proposals[proposal_id] = proposal
        
        contract.status = OrderStatus.PURCHASED
        return contract
    
    async def monitor_contract(self, contract_id: str) -> Contract:
        if contract_id not in self._contracts:
            raise MonitoringError(f"Unknown contract: {contract_id}")
        
        contract = self._contracts[contract_id]
        
        if contract.result is not None:
            return contract
        
        await asyncio.sleep(self.config.connection_delay)
        
        # Check if expired
        if datetime.now() >= contract.expiry_time:
            # Simulate result
            contract.status = OrderStatus.CONFIRMED
            
            if self.config.crash_mode:
                # Force loss for testing
                contract.result = TradeResult.LOSS
            else:
                # Simulate outcome based on direction and volatility
                entry_price = contract.entry_spot
                
                # Simulate price movement during contract duration
                volatility_factor = contract.duration / 3600  # Normalize duration
                if contract.direction == "UP":
                    price_movement = random.gauss(volatility_factor * 0.5, self.config.volatility)
                    final_price = entry_price * (1 + price_movement)
                    contract.result = TradeResult.WIN if final_price > entry_price else TradeResult.LOSS
                else:
                    price_movement = random.gauss(volatility_factor * 0.5, self.config.volatility)
                    final_price = entry_price * (1 - price_movement)
                    contract.result = TradeResult.WIN if final_price < entry_price else TradeResult.LOSS
            
            # Calculate profit
            if contract.result == TradeResult.WIN:
                profit = contract.stake * contract.payout
                self._balance += profit
                contract.profit = profit
            else:
                contract.profit = -contract.stake
            
            contract.settlement_price = contract.entry_spot  # Simplified
            
            self._last_trade_time = datetime.now()
        
        return contract
    
    async def get_contract_status(self, contract_id: str) -> Contract:
        if contract_id not in self._contracts:
            raise MonitoringError(f"Unknown contract: {contract_id}")
        return self._contracts[contract_id]
    
    async def cancel_contract(self, contract_id: str) -> bool:
        if contract_id not in self._contracts:
            return False
        
        contract = self._contracts[contract_id]
        if contract.status == OrderStatus.PROCESSING:
            # Full refund if still processing
            self._balance += contract.stake
            contract.status = OrderStatus.CANCELLED
            return True
        return False
    
    # === BaseBroker Interface Methods ===
    
    async def reconnect(self, max_attempts: int = 3) -> bool:
        """Reconnect with retries."""
        for attempt in range(max_attempts):
            if await self.connect():
                return True
            await asyncio.sleep(1.0)
        return False
    
    async def authenticate(self, credentials: Dict[str, str]) -> bool:
        """Demo broker always authenticates successfully."""
        return True
    
    async def verify_account(self) -> AccountInfo:
        """Verify account - always returns demo account."""
        return AccountInfo(
            account_id="DEMO_ACCOUNT_001",
            account_mode=AccountMode.DEMO,
            currency=self.config.currency,
            balance=self._balance,
            equity=self._balance,
            trading_enabled=True,
            permissions=["trade", "read"]
        )
    
    async def discover_capabilities(self) -> BrokerCapabilities:
        """Discover broker capabilities."""
        symbols = {}
        for sym in self._symbols:
            symbols[sym] = SymbolCapability(
                symbol=sym,
                display_name=sym,
                contract_types=["CALL", "PUT"],
                available_durations={"s": [15, 30, 60, 120, 180, 300]},
                min_stake=self.config.min_stake,
                max_stake=self.config.max_stake,
                payout_percentage=self.config.payout_rate * 100,
                trading_available=True
            )
        
        return BrokerCapabilities(
            symbols=symbols,
            supported_durations=["s"],
            supported_contract_types=["CALL", "PUT"]
        )
    
    async def get_symbol_info(self, symbol: str) -> SymbolCapability:
        """Get symbol capabilities."""
        if symbol not in self._symbols:
            raise Exception(f"Unknown symbol: {symbol}")
        
        return SymbolCapability(
            symbol=symbol,
            display_name=symbol,
            contract_types=["CALL", "PUT"],
            available_durations={"s": [15, 30, 60, 120, 180, 300]},
            min_stake=self.config.min_stake,
            max_stake=self.config.max_stake,
            payout_percentage=self.config.payout_rate * 100,
            trading_available=True
        )
    
    async def request_proposal(
        self,
        symbol: str,
        direction: str,
        stake: float,
        duration: int,
        duration_unit: str = "s"
    ) -> Dict[str, Any]:
        """Request a contract proposal - returns proposal dict."""
        if not self._connected:
            raise ConnectionException("Not connected")
        
        # Validate
        if not self.validate_symbol(symbol):
            raise ProposalException(f"Symbol not supported: {symbol}")
        if not self.validate_direction(direction):
            raise ProposalException(f"Invalid direction: {direction}")
        if not self.validate_stake(symbol, stake):
            raise ProposalException(f"Invalid stake: {stake}")
        if not self.validate_duration(symbol, duration, duration_unit):
            raise ProposalException(f"Invalid duration: {duration} {duration_unit}")
        
        # Get market data
        market = await self.get_market_data(symbol)
        
        proposal_id = f"PROP_{uuid.uuid4().hex[:12]}"
        
        return {
            "proposal_id": proposal_id,
            "symbol": symbol,
            "direction": direction,
            "stake": stake,
            "duration": duration,
            "duration_unit": duration_unit,
            "payout": self.config.payout_rate,
            "spot_price": market.spot,
            "timestamp": datetime.now().isoformat(),
            "contract_type": "CALL" if direction == "UP" else "PUT"
        }
    
    async def execute_trade(self, proposal: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a trade using a proposal."""
        if not self._connected:
            raise ConnectionException("Not connected")
        
        # Check demo mode enforcement
        if not self.enforce_demo_mode_check():
            raise ExecutionException("Live trading blocked: Demo mode enforced")
        
        # Check balance
        stake = proposal.get("stake", 0)
        if self._balance < stake:
            raise ExecutionException("Insufficient balance")
        
        # Check cooldown
        if (self.config.cooldowns and self._last_trade_time and 
            (datetime.now() - self._last_trade_time).total_seconds() < 60):
            raise ExecutionException("Cooldown period active")
        
        contract_id = f"CONTRACT_{uuid.uuid4().hex[:12]}"
        duration = proposal.get("duration", 60)
        expiry_time = datetime.now() + timedelta(seconds=duration)
        
        # Deduct stake
        self._balance -= stake
        
        contract = Contract(
            contract_id=contract_id,
            proposal_id=proposal.get("proposal_id", ""),
            buy_price=stake,
            payout=proposal.get("payout", self.config.payout_rate),
            stake=stake,
            symbol=proposal.get("symbol", ""),
            direction=proposal.get("direction", "UP"),
            duration=duration,
            entry_spot=proposal.get("spot_price", 0),
            expiry_time=expiry_time,
            status=OrderStatus.PROCESSING,
            raw_response={
                "contract_id": contract_id,
                "proposal": proposal
            }
        )
        
        self._contracts[contract_id] = contract
        contract.status = OrderStatus.PURCHASED
        
        return {
            "contract_id": contract_id,
            "proposal_id": proposal.get("proposal_id", ""),
            "buy_price": stake,
            "payout": proposal.get("payout", self.config.payout_rate),
            "expiry_time": expiry_time.isoformat(),
            "status": "purchased"
        }
    
    async def subscribe_contract(self, contract_id: str, callback=None) -> None:
        """Subscribe to contract updates (demo - no-op)."""
        pass
    
    async def unsubscribe_contract(self, contract_id: str) -> None:
        """Unsubscribe from contract updates (demo - no-op)."""
        pass
    
    async def health_check(self) -> Dict[str, Any]:
        """Check connection health."""
        return {
            "connected": self._connected,
            "balance": self._balance,
            "symbols": self._symbols,
            "last_trade_time": self._last_trade_time.isoformat() if self._last_trade_time else None
        }
    
    async def get_balance(self) -> float:
        """Get current balance as float (BaseBroker interface)."""
        if not self._connected:
            raise ConnectionException("Not connected")
        await asyncio.sleep(self.config.connection_delay)
        return self._balance
    
    def get_internal_state(self) -> Dict[str, Any]:
        """Get internal state for testing/debugging."""
        return {
            "balance": self._balance,
            "connected": self._connected,
            "prices": dict(self._prices),
            "contracts": {k: v.__dict__ for k, v in self._contracts.items()},
            "last_trade_time": self._last_trade_time.isoformat() if self._last_trade_time else None
        }


# Convenience function for quick demo
async def demo_trading():
    """Run a quick demo of the broker API."""
    broker = DemoBroker(DemoBrokerConfig(initial_balance=100.0))
    
    await broker.connect()
    
    # Get balance
    balance = await broker.get_balance()
    print(f"Balance: ${balance.balance:.2f}")
    
    # Get market data
    market = await broker.get_market_data("R_100")
    print(f"R_100 Price: {market.spot:.4f}")
    
    # Create proposal
    proposal = await broker.create_proposal(
        symbol="R_100",
        direction="UP",
        stake=1.0,
        duration=60
    )
    print(f"Proposal: {proposal.proposal_id}, Payout: {proposal.payout:.0%}")
    
    # Buy contract
    contract = await broker.buy_contract(proposal.proposal_id)
    print(f"Contract: {contract.contract_id}, Expires: {contract.expiry_time}")
    
    # Monitor (wait for expiry in real scenario)
    await asyncio.sleep(0.1)  # Simulate passage of time
    
    result = await broker.monitor_contract(contract.contract_id)
    print(f"Result: {result.result.value}, Profit: ${result.profit or 0:.2f}")
    
    new_balance = await broker.get_balance()
    print(f"New Balance: ${new_balance.balance:.2f}")
    
    await broker.disconnect()


if __name__ == "__main__":
    import asyncio
    asyncio.run(demo_trading())