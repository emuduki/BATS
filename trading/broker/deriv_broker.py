"""Deriv.com Broker Implementation - Phase 6.
Connects BATS to the actual Deriv API for live trading.
Uses WebSocket for real-time communication.

Note: This requires proper API credentials and uses Deriv's binary.com API.
For demonstration only - actual production use requires proper error handling.
"""

import asyncio
import json
import uuid
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta

from trading.broker.base_broker import (
    AccountInfo, SymbolCapability, BrokerCapabilities, ConnectionState, AccountMode
)
from trading.broker.binary_broker import (
    BinaryBroker, ContractType, OrderStatus, TradeResult,
    MarketData, AccountBalance, ContractSpec, Proposal, Contract,
    BrokerException, ConnectionError, AuthenticationError,
    ProposalError, PurchaseError, MonitoringError
)


class DerivBroker(BinaryBroker):
    """
    Deriv.com Broker Integration.
    
    Connects to Deriv's API for real binary options trading.
    Implements the BinaryBroker interface for seamless integration.
    
    Uses Deriv's API format:
    - authorize: Authentication
    - api_price: Get current price
    - proposal: Request contract proposal
    - buy: Purchase contract
    - proposal_open_contract: Monitor contract status
    - balance: Get account balance
    - ping/pong: Keep connection alive
    """
    
    # Deriv API endpoints
    DERIV_API_URL = "wss://ws.binaryws.com/websocket"
    DERIV_API_APP_ID = "10049"  # Demo app ID - replace with your own
    
    def __init__(
        self,
        api_token: str = None,
        app_id: str = None,
        use_testnet: bool = True,
        demo_mode: bool = True,
        max_retries: int = 3,
        retry_delay: float = 1.0
    ):
        """
        Initialize Deriv Broker.
        
        Args:
            api_token: Deriv API token (from https://developers.deriv.com/)
            app_id: Application ID for API access
            use_testnet: Use Deriv test environment
            demo_mode: Use demo account (no real money)
            max_retries: Maximum connection retries
            retry_delay: Delay between retries in seconds
        """
        super().__init__(enforce_demo_mode=demo_mode)
        self.api_token = api_token or ""
        self.app_id = app_id or self.DERIV_API_APP_ID
        self.use_testnet = use_testnet
        self.demo_mode = demo_mode
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        
        self._connected = False
        self._authenticated = False
        self._ws = None
        self._reconnect_attempts = 0
        self._message_handler = {}
        self._req_id_counter = 0
        
        # Use testnet URL if requested
        if self.use_testnet:
            self.DERIV_API_URL = "wss://ws.binaryws.com/websocket"
    
    async def connect(self) -> bool:
        """Establish WebSocket connection to Deriv API."""
        import websockets
        
        for attempt in range(self.max_retries):
            try:
                url = f"{self.DERIV_API_URL}?app_id={self.app_id}"
                self._ws = await websockets.connect(url, ping_interval=30, ping_timeout=10)
                self._connected = True
                self._reconnect_attempts = 0
                
                # Authenticate if token provided
                if self.api_token:
                    await self._authenticate()
                
                # Start heartbeat
                asyncio.create_task(self._heartbeat())
                
                return True
                
            except Exception as e:
                self._reconnect_attempts += 1
                if attempt < self.max_retries - 1:
                    await asyncio.sleep(self.retry_delay * (attempt + 1))
                else:
                    raise ConnectionError(f"Failed to connect after {self.max_retries} attempts: {str(e)}")
        
        return False
    
    async def disconnect(self) -> bool:
        """Close WebSocket connection."""
        if self._ws:
            await self._ws.close()
        self._connected = False
        self._authenticated = False
        return True
    
    async def is_connected(self) -> bool:
        """Check if connected and authenticated."""
        if not self._ws or self._ws.closed:
            self._connected = False
            return False
        return self._connected and self._authenticated
    
    async def _authenticate(self) -> bool:
        """Authenticate with API token."""
        response = await self._send_request({
            "authorize": self.api_token
        })
        
        if "error" in response:
            raise AuthenticationError(response["error"]["message"])
        
        if response.get("auth_status") == "OK":
            self._authenticated = True
            return True
        
        raise AuthenticationError(f"Authentication failed: {response.get('error', 'Unknown error')}")
    
    async def _heartbeat(self):
        """Send periodic ping to keep connection alive."""
        while self._connected:
            try:
                await self._send_request({"ping": 1})
                await asyncio.sleep(25)  # Ping every 25 seconds
            except Exception:
                self._connected = False
                break
    
    async def _send_request(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Send request to Deriv API and get response."""
        if not self._ws:
            raise ConnectionError("WebSocket connection not established")
        
        self._req_id_counter += 1
        request["req_id"] = self._req_id_counter
        
        await self._ws.send(json.dumps(request))
        response = await self._ws.recv()
        return json.loads(response)
    
    async def get_balance(self) -> AccountBalance:
        """Get current account balance from Deriv."""
        response = await self._send_request({
            "balance": 1,
            "subscribe": 1
        })
        
        if "error" in response:
            raise BrokerException(f"Balance request failed: {response['error']['message']}")
        
        balance_info = response.get("balance", {})
        
        return AccountBalance(
            balance=balance_info.get("balance", 0.0),
            currency=balance_info.get("currency", "USD"),
            equity=balance_info.get("equity", 0.0),
            available=balance_info.get("balance", 0.0),  # No margin in binary options
            timestamp=datetime.now()
        )
    
    async def get_market_data(self, symbol: str) -> MarketData:
        """Get current market data from Deriv."""
        response = await self._send_request({
            "api_price": symbol,
            "granularity": 60  # 1-minute granularity
        })
        
        if "error" in response:
            raise BrokerException(f"Market data request failed: {response['error']['message']}")
        
        prices = response.get("prices", [])
        if not prices:
            raise BrokerException(f"No price data for symbol: {symbol}")
        
        latest = prices[-1]
        return MarketData(
            symbol=symbol,
            bid=latest.get("bid", 0.0),
            ask=latest.get("ask", 0.0),
            spot=(latest.get("bid", 0) + latest.get("ask", 0)) / 2,
            timestamp=datetime.now(),
            source="Deriv"
        )
    
    async def get_available_contracts(self, symbol: str) -> List[ContractSpec]:
        """Get available contracts from Deriv."""
        response = await self._send_request({
            "contracts_for": symbol,
            "contract_type": "CALL"  # Also get PUT
        })
        
        if "error" in response:
            raise BrokerException(f"Contracts request failed: {response['error']['message']}")
        
        contract_for = response.get("contracts_for", {})
        specs = contract_for.get("contracts", [])
        
        result = []
        for spec in specs:
            if spec.get("contract_type") in ["CALL", "PUT"]:
                result.append(ContractSpec(
                    contract_type=ContractType.RISE_FALL,
                    symbol=symbol,
                    direction="UP" if spec.get("contract_type") == "CALL" else "DOWN",
                    min_stake=spec.get("min_stake", 0.1),
                    max_stake=spec.get("max_stake", 100.0),
                    min_duration=spec.get("min_duration", 10),
                    max_duration=spec.get("max_duration", 300),
                    available_durations=spec.get("available_durations", [15, 30, 60, 120]),
                    payout=spec.get("payout", 0.85)
                ))
        
        return result
    
    async def create_proposal(
        self,
        symbol: str,
        direction: str,
        stake: float,
        duration: int,
        contract_type: ContractType = ContractType.RISE_FALL
    ) -> Proposal:
        """Create a contract proposal via Deriv API."""
        call_put = "CALL" if direction == "UP" else "PUT"
        
        request = {
            "proposal": 1,
            "amount": stake,
            "asset": symbol,
            "duration": duration,
            "duration_unit": "s",
            "contract_type": call_put,
            "currency": "USD",
            "basis": "stake"
        }
        
        response = await self._send_request(request)
        
        if "error" in response:
            raise ProposalError(response["error"]["message"])
        
        proposal_data = response.get("proposal", {})
        
        if not proposal_data.get("id"):
            raise ProposalError("No proposal ID returned")
        
        return Proposal(
            proposal_id=proposal_data["id"],
            contract_spec=ContractSpec(
                contract_type=contract_type,
                symbol=symbol,
                direction=direction,
                min_stake=0.1,
                max_stake=100.0,
                min_duration=10,
                max_duration=300,
                available_durations=[15, 30, 60, 120, 180, 300],
                payout=proposal_data.get("payout", 0.85)
            ),
            stake=stake,
            duration=duration,
            direction=direction,
            payout=proposal_data.get("payout", 0.85),
            spot_price=proposal_data.get("spot", 0.0),
            timestamp=datetime.now(),
            raw_response=response
        )
    
    async def buy_contract(self, proposal_id: str) -> Contract:
        """Buy contract using proposal ID."""
        response = await self._send_request({
            "buy": proposal_id
        })
        
        if "error" in response:
            raise PurchaseError(response["error"]["message"])
        
        buy_data = response.get("buy", {})
        contract_id = buy_data.get("contract_id") or buy_data.get("id")
        
        if not contract_id:
            raise PurchaseError("No contract ID returned")
        
        return Contract(
            contract_id=contract_id,
            proposal_id=proposal_id,
            buy_price=buy_data.get("buy_price", 0),
            payout=buy_data.get("payout", 0.85),
            stake=buy_data.get("buy_price", 0),
            symbol=buy_data.get("symbol", ""),
            direction="UP" if buy_data.get("contract_type") == "CALL" else "DOWN",
            duration=buy_data.get("duration", 0),
            entry_spot=buy_data.get("spot", 0.0),
            expiry_time=datetime.now() + timedelta(seconds=buy_data.get("duration", 0)),
            status=OrderStatus.PURCHASED,
            raw_response=response
        )
    
    async def monitor_contract(self, contract_id: str) -> Contract:
        """Monitor contract status via Deriv API."""
        response = await self._send_request({
            "proposal_open_contract": 1,
            "contract_id": contract_id
        })
        
        if "error" in response:
            raise MonitoringError(response["error"]["message"])
        
        contract_data = response.get("proposal_open_contract", {})
        
        contract = Contract(
            contract_id=contract_id,
            proposal_id=contract_data.get("proposal_id", ""),
            buy_price=contract_data.get("buy_price", 0),
            payout=contract_data.get("payout", 0.85),
            stake=contract_data.get("buy_price", 0),
            symbol=contract_data.get("symbol", ""),
            direction="UP" if contract_data.get("contract_type") == "CALL" else "DOWN",
            duration=contract_data.get("duration", 0),
            entry_spot=contract_data.get("entry_spot", 0.0),
            expiry_time=datetime.now(),
            status=OrderStatus.CONFIRMED,
            settlement_price=contract_data.get("exit_spot"),
            profit=contract_data.get("profit", 0.0),
            raw_response=response
        )
        
        # Determine result
        if contract_data.get("status") == "won":
            contract.result = TradeResult.WIN
        elif contract_data.get("status") == "lost":
            contract.result = TradeResult.LOSS
        elif contract_data.get("status") == "refunded":
            contract.result = TradeResult.REFUNDED
        else:
            contract.result = TradeResult.PENDING
        
        return contract
    
    async def get_contract_status(self, contract_id: str) -> Contract:
        """Get current contract status."""
        return await self.monitor_contract(contract_id)
    
    async def cancel_contract(self, contract_id: str) -> bool:
        """Attempt to cancel a contract (Deriv may not support this)."""
        # Deriv generally does not allow cancellation of purchased contracts
        # This is implemented for interface compliance
        try:
            response = await self._send_request({
                "cancel": contract_id
            })
            return "error" not in response
        except Exception:
            return False

    # === BaseBroker Interface Methods ===
    
    async def reconnect(self, max_attempts: int = 3) -> bool:
        """Attempt to reconnect with retries."""
        for attempt in range(max_attempts):
            if await self.connect():
                return True
            await asyncio.sleep(self.retry_delay)
        return False
    
    async def authenticate(self, credentials: Dict[str, str]) -> bool:
        """Authenticate with Deriv API."""
        if not credentials.get("api_token"):
            raise AuthenticationException("API token required")
        
        self.api_token = credentials["api_token"]
        return await self._authenticate()
    
    async def verify_account(self) -> AccountInfo:
        """Verify account and return account information."""
        response = await self._send_request({
            "balance": 1
        })
        
        if "error" in response:
            raise Exception(f"Account verification failed: {response['error']['message']}")
        
        balance_info = response.get("balance", {})
        is_demo = balance_info.get("is_virtual", 1) == 1
        
        return AccountInfo(
            account_id=balance_info.get("loginid", ""),
            account_mode=AccountMode.DEMO if is_demo else AccountMode.REAL,
            currency=balance_info.get("currency", "USD"),
            balance=float(balance_info.get("balance", 0.0)),
            equity=float(balance_info.get("equity", 0.0)),
            trading_enabled=True,  # Deriv doesn't provide specific trading enabled flag
            permissions=["trade", "read"] if is_demo else ["trade", "read", "withdraw"]
        )
    
    async def discover_capabilities(self) -> BrokerCapabilities:
        """Discover broker capabilities from Deriv API."""
        # For simplicity, we'll return static capabilities for common symbols
        # In production, you might want to query active_symbols endpoint
        symbols = {
            "R_100": SymbolCapability(
                symbol="R_100",
                display_name="Volatility 100 Index",
                contract_types=["CALL", "PUT"],
                available_durations={"s": [15, 30, 60, 120, 180, 300]},
                min_stake=0.35,
                max_stake=20000.0,
                payout_percentage=95.0,  # Variable, using typical value
                trading_available=True
            ),
            "R_50": SymbolCapability(
                symbol="R_50",
                display_name="Volatility 50 Index",
                contract_types=["CALL", "PUT"],
                available_durations={"s": [15, 30, 60, 120, 180, 300]},
                min_stake=0.35,
                max_stake=20000.0,
                payout_percentage=90.0,
                trading_available=True
            ),
            "frxEURUSD": SymbolCapability(
                symbol="frxEURUSD",
                display_name="EUR/USD",
                contract_types=["CALL", "PUT"],
                available_durations={"s": [60, 120, 300]},
                min_stake=0.35,
                max_stake=20000.0,
                payout_percentage=85.0,
                trading_available=True
            )
        }
        
        return BrokerCapabilities(
            symbols=symbols,
            supported_durations=["s"],
            supported_contract_types=["CALL", "PUT"]
        )
    
    async def get_symbol_info(self, symbol: str) -> SymbolCapability:
        """Get symbol capabilities."""
        capabilities = await self.discover_capabilities()
        if symbol not in capabilities.symbols:
            raise Exception(f"Symbol not supported: {symbol}")
        return capabilities.symbols[symbol]
    
    async def request_proposal(
        self,
        symbol: str,
        direction: str,
        stake: float,
        duration: int,
        duration_unit: str = "s"
    ) -> Dict[str, Any]:
        """Request a contract proposal from Deriv."""
        if not self._connected or not self._authenticated:
            raise ConnectionException("Not connected or not authenticated")
        
        call_put = "CALL" if direction == "UP" else "PUT"
        
        request = {
            "proposal": 1,
            "amount": stake,
            "asset": symbol,
            "duration": duration,
            "duration_unit": duration_unit,
            "contract_type": call_put,
            "currency": "USD"
        }
        
        response = await self._send_request(request)
        
        if "error" in response:
            raise ProposalException(response["error"]["message"])
        
        proposal_data = response.get("proposal", {})
        
        if not proposal_data.get("id"):
            raise ProposalException("No proposal ID returned")
        
        return {
            "proposal_id": proposal_data["id"],
            "symbol": symbol,
            "direction": direction,
            "stake": stake,
            "duration": duration,
            "duration_unit": duration_unit,
            "payout": proposal_data.get("payout", 0.0),
            "spot_price": proposal_data.get("spot", 0.0),
            "timestamp": datetime.now().isoformat(),
            "contract_type": call_put,
            "raw_response": response
        }
    
    async def execute_trade(self, proposal: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a trade using a proposal from Deriv."""
        if not self._connected or not self._authenticated:
            raise ConnectionException("Not connected or not authenticated")
        
        # Check demo mode enforcement
        if not self.enforce_demo_mode_check():
            raise ExecutionException("Live trading blocked: Demo mode enforced")
        
        # Get the proposal ID from the proposal dict
        proposal_id = proposal.get("proposal_id")
        if not proposal_id:
            raise ExecutionException("Proposal ID missing")
        
        # Buy the contract
        response = await self._send_request({
            "buy": proposal_id
        })
        
        if "error" in response:
            raise ExecutionException(response["error"]["message"])
        
        buy_data = response.get("buy", {})
        contract_id = buy_data.get("contract_id") or buy_data.get("id")
        
        if not contract_id:
            raise ExecutionException("No contract ID returned")
        
        return {
            "contract_id": contract_id,
            "proposal_id": proposal_id,
            "buy_price": buy_data.get("buy_price", 0),
            "payout": buy_data.get("payout", 0.0),
            "expiry_time": (datetime.now() + timedelta(seconds=buy_data.get("duration", 0))).isoformat(),
            "status": "purchased",
            "raw_response": response
        }
    
    async def subscribe_contract(self, contract_id: str, callback=None) -> None:
        """Subscribe to contract updates - not implemented for simplicity."""
        # In a full implementation, you would maintain subscriptions
        pass
    
    async def unsubscribe_contract(self, contract_id: str) -> None:
        """Unsubscribe from contract updates - not implemented for simplicity."""
        pass
    
    async def health_check(self) -> Dict[str, Any]:
        """Check connection health."""
        try:
            latency = None
            start = datetime.now()
            await self._send_request({"ping": 1})
            latency = (datetime.now() - start).total_seconds() * 1000
            
            return {
                "connected": self._connected,
                "authenticated": self._authenticated,
                "latency_ms": round(latency, 2) if latency else None,
                "symbols_available": 3  # Approximate
            }
        except Exception as e:
            return {
                "connected": False,
                "authenticated": False,
                "error": str(e)
            }
    
    async def get_balance(self) -> float:
        """Get current balance as float."""
        if not self._connected or not self._authenticated:
            raise ConnectionException("Not connected or not authenticated")
        
        response = await self._send_request({
            "balance": 1
        })
        
        if "error" in response:
            raise Exception(f"Balance request failed: {response['error']['message']}")
        
        balance_info = response.get("balance", {})
        return float(balance_info.get("balance", 0.0))