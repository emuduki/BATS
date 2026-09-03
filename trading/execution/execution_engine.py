"""
Binary options trading execution module.
Connects AI predictions to broker execution and trade monitoring.
Integrates Phase 4 Multi-Agent Decision System.
"""

import time
import asyncio
from typing import Dict, Any, Optional, List, Tuple
from datetime import datetime, timedelta
import pandas as pd

from trading.agents.multi_agent_system import MultiAgentSignal
from trading.broker import BinaryBroker, DemoBroker
from trading.broker.binary_broker import Proposal, Contract, TradeResult


class BinaryTradingEngine:
    """
    Executes binary options trades based on AI signal predictions.
    Integrates signal validation, risk management, and trade execution.
    Now uses the Phase 4 Multi-Agent Decision System for trade decisions.
    """
    
    def __init__(self, 
                 initial_balance: float = 100.0,
                 max_risk_pct: float = 1.0,
                 max_drawdown_pct: float = 10.0,
                 daily_loss_limit: float = 10.0,
                 max_open_trades: int = 3,
                 confidence_threshold: float = 0.55,
                 cooldown_seconds: int = 60,
                 broker: Optional[BinaryBroker] = None,
                 symbol: str = "R_100"):
        self.balance = initial_balance
        self.max_risk_pct = max_risk_pct
        self.max_drawdown_pct = max_drawdown_pct
        self.daily_loss_limit = daily_loss_limit
        self.max_open_trades = max_open_trades
        self.confidence_threshold = confidence_threshold
        self.cooldown_seconds = cooldown_seconds
        self.symbol = symbol
        
        self.daily_trades_completed = 0
        self.daily_loss = 0.0
        self.daily_wins = 0
        self.current_drawdown = 0.0
        self.equity_curve: List[float] = [initial_balance]
        
        self.risk_manager = RiskManager(
            balance=initial_balance,
            max_risk_pct=max_risk_pct,
            max_drawdown_pct=max_drawdown_pct,
            daily_loss_limit=daily_loss_limit,
            max_open_trades=max_open_trades,
            confidence_threshold=confidence_threshold,
            cooldown_seconds=cooldown_seconds
        )
        
        from trading.agents.multi_agent_system import multi_agent_system
        self.multi_agent_system = multi_agent_system
        
        self.broker = broker if broker is not None else DemoBroker()
        self._broker_connected = False
        self._open_contracts: Dict[str, Contract] = {}
    
    def process_multi_agent_signal(
        self,
        df: pd.DataFrame,
        current_price: float,
        investment_amount: Optional[float] = None
    ) -> Dict[str, Any]:
        signal: MultiAgentSignal = self.multi_agent_system.analyze(df)
        
        decision = signal.decision
        direction = signal.decision.direction
        confidence = signal.decision.confidence
        duration_seconds = signal.decision.duration_seconds
        duration_label = signal.decision.duration_label
        reason = signal.decision.reason
        
        response: Dict[str, Any] = {
            'timestamp': datetime.now().isoformat(),
            'multi_agent_analysis': True,
            'decision_action': decision.action.value,
            'direction': direction.value if direction else None,
            'confidence': confidence,
            'duration_seconds': duration_seconds,
            'duration_label': duration_label,
            'reason': reason,
            'agent_contributions': {
                'technical': {
                    'direction': signal.technical.direction.value,
                    'confidence': signal.technical.confidence,
                },
                'regime': {
                    'regime': signal.regime.regime.value,
                    'trend_direction': signal.regime.trend_direction,
                    'volatility': signal.regime.volatility_level,
                },
                'ai': {
                    'direction': signal.ai.direction.value,
                    'confidence': signal.ai.confidence,
                    'probabilities': signal.ai.probabilities,
                },
                'duration': {
                    'duration_label': signal.duration.duration_label,
                    'reason': signal.duration.reason,
                    'confidence': signal.duration.confidence,
                }
            }
        }
        
        response['broker'] = {
            'type': self.broker.__class__.__name__,
            'symbol': self.symbol,
            'connected': self._broker_connected
        }
        
        if decision.action.name == "NO_TRADE":
            response['trade_status'] = 'NO_TRADE'
            response['reject_reason'] = reason
            response['trade_executed'] = False
            self.equity_curve.append(self.balance)
            return response
        
        direction_str = direction.value if direction else 'UP'
        max_risk_amount = self.balance * (self.max_risk_pct / 100)
        stake_amount = investment_amount if investment_amount is not None else min(confidence * 10, max_risk_amount)
        if stake_amount <= 0:
            stake_amount = max_risk_amount
        
        risk_result = self.risk_manager.validate_trade(
            signal_direction=direction_str,
            signal_confidence=confidence,
            entry_price=current_price,
            recommended_stake=stake_amount,
            current_balance=self.balance,
            current_time=datetime.now()
        )
        
        if not risk_result['is_valid']:
            response['trade_status'] = 'REJECTED'
            response['reject_reason'] = risk_result['reason']
            response['trade_executed'] = False
            response['risk_details'] = risk_result
            self.equity_curve.append(self.balance)
            return response
        
        if investment_amount is not None:
            stake_amount = investment_amount
        else:
            max_risk_amount = self.balance * (self.max_risk_pct / 100)
            stake_amount = min(confidence * 10, max_risk_amount)
        
        response['trade_status'] = 'PENDING_EXECUTION'
        response['stake_amount'] = float(stake_amount)
        response['trade_executed'] = False
        
        self.equity_curve.append(self.balance)
        return response
    
    async def execute_decision_trade(
        self,
        decision: Dict[str, Any],
        current_price: float
    ) -> Dict[str, Any]:
        if decision.get('trade_status') != 'PENDING_EXECUTION':
            return {
                'status': 'ERROR',
                'reason': 'Decision is not in PENDING_EXECUTION state',
                'timestamp': datetime.now().isoformat()
            }
        
        direction = decision.get('direction')
        duration_seconds = decision.get('duration_seconds')
        stake_amount = decision.get('stake_amount', 0.0)
        confidence = decision.get('confidence', 0.0)
        
        # Risk validation
        risk_result = self.risk_manager.can_trade_now()
        if not risk_result[0]:
            self._last_trade_time = datetime.now()
            self.risk_manager.update_after_trade('LOSS', -stake_amount, {'direction': direction})
            return {
                'status': 'REJECTED',
                'reason': f'Risk check failed: {risk_result[1]}',
                'timestamp': datetime.now().isoformat()
            }
        
        # Connect to broker
        if not self._broker_connected:
            try:
                await self.broker.connect()
                self._broker_connected = True
            except Exception as e:
                return {
                    'status': 'EXECUTION_ERROR',
                    'reason': f'Broker connection failed: {str(e)}',
                    'timestamp': datetime.now().isoformat()
                }
        
        # Get market data
        try:
            market_data = await self.broker.get_market_data(self.symbol)
        except Exception as e:
            return {
                'status': 'EXECUTION_ERROR',
                'reason': f'Market data unavailable: {str(e)}',
                'timestamp': datetime.now().isoformat()
            }
        
        # Create proposal
        try:
            proposal = await self.broker.create_proposal(
                symbol=self.symbol,
                direction=direction,
                stake=stake_amount,
                duration=duration_seconds
            )
        except Exception as e:
            return {
                'status': 'EXECUTION_ERROR',
                'reason': f'Proposal creation failed: {str(e)}',
                'timestamp': datetime.now().isoformat()
            }
        
        # Buy contract
        try:
            contract = await self.broker.buy_contract(proposal.proposal_id)
        except Exception as e:
            return {
                'status': 'EXECUTION_ERROR',
                'reason': f'Contract purchase failed: {str(e)}',
                'timestamp': datetime.now().isoformat()
            }
        
        # Track open contract
        self._open_contracts[contract.contract_id] = contract
        self.risk_manager.open_trade()
        
        # Monitor until expiry
        try:
            contract_result = await asyncio.wait_for(
                self.broker.monitor_contract(contract.contract_id),
                timeout=duration_seconds + 5
            )
            
            if contract.contract_id in self._open_contracts:
                del self._open_contracts[contract.contract_id]
        except asyncio.TimeoutError:
            contract_result = contract
        
        # Calculate PnL
        pnl = contract_result.profit if contract_result.profit else (
            stake_amount * 0.85 if contract_result.result == TradeResult.WIN else -stake_amount
        )
        
        result_name = contract_result.result.value if contract_result.result else "PENDING"
        
        # Update balance
        self.balance += pnl
        
        # Update risk manager
        self.risk_manager.update_after_trade(
            'WIN' if contract_result.result == TradeResult.WIN else 'LOSS',
            pnl,
            {
                'direction': direction,
                'stake_amount': stake_amount,
                'contract_id': contract.contract_id
            }
        )
        
        self.equity_curve.append(self.balance)
        
        return {
            'status': 'EXECUTED',
            'result': result_name,
            'stake_amount': float(stake_amount),
            'payout_amount': float(pnl),
            'direction': direction,
            'duration_seconds': duration_seconds,
            'entry_price': current_price,
            'exit_price': contract_result.settlement_price,
            'contract_id': contract.contract_id,
            'proposal_id': contract.proposal_id,
            'balance_after': self.balance,
            'equity_curve_index': len(self.equity_curve) - 1,
            'timestamp': datetime.now().isoformat(),
            'broker_status': contract_result.status.value,
            'broker_result': contract_result.result.value if contract_result.result else None,
            'broker_profit': float(contract_result.profit or 0)
        }


class RiskManager:
    """Phase 5: Advanced Risk Management System."""
    
    def __init__(self, 
                 balance: float, 
                 max_risk_pct: float = 1.0,
                 max_drawdown_pct: float = 10.0,
                 daily_loss_limit: float = 10.0,
                 max_open_trades: int = 3,
                 confidence_threshold: float = 0.55,
                 cooldown_seconds: int = 60):
        self.initial_balance = balance
        self.balance = balance
        self.max_risk_pct = max_risk_pct
        self.max_drawdown_pct = max_drawdown_pct
        self.daily_loss_limit = daily_loss_limit
        self.max_open_trades = max_open_trades
        self.confidence_threshold = confidence_threshold
        self.cooldown_seconds = cooldown_seconds
        
        self.current_drawdown = 0.0
        self.consecutive_losses = 0
        self.consecutive_wins = 0
        self.daily_loss_limit_breached = False
        self.daily_win_count = 0
        self.daily_loss_count = 0
        self.daily_loss_amount = 0.0
        self.daily_win_amount = 0.0
        self.max_drawdown = 0.0
        self.open_trades_count = 0
        self.last_trade_time = None
        self.is_paused = False
        self.pause_reason = ""
        self.equity_curve: List[float] = [balance]
        self.daily_trades: List[Dict] = []
    
    def validate_trade(self,
                      signal_direction: str,
                      signal_confidence: float,
                      entry_price: float,
                      recommended_stake: float,
                      current_balance: float,
                      current_time: Optional[datetime] = None) -> Dict[str, Any]:
        if current_time is None:
            current_time = datetime.now()
        
        checks = {
            'confidence_check': False,
            'stake_check': False,
            'daily_loss_check': False,
            'consecutive_loss_check': False,
            'max_trades_check': False,
            'cooldown_check': False,
            'drawdown_check': False
        }
        rejection_reasons = []
        
        if self.is_paused:
            return {
                'is_valid': False,
                'reason': f'SYSTEM_PAUSED: {self.pause_reason}',
                'risk_recommendation': 0.0,
                'checks': checks,
                'rejection_reasons': [self.pause_reason],
                'timestamp': current_time.isoformat()
            }
        
        if signal_confidence < self.confidence_threshold:
            rejection_reasons.append(f"Confidence {signal_confidence:.0%} < threshold {self.confidence_threshold:.0%}")
        else:
            checks['confidence_check'] = True
        
        max_stake_amount = current_balance * (self.max_risk_pct / 100)
        if recommended_stake > max_stake_amount:
            rejection_reasons.append(f"Stake ${recommended_stake:.2f} > max ${max_stake_amount:.2f}")
        elif recommended_stake <= 0:
            rejection_reasons.append(f"Invalid stake: ${recommended_stake:.2f}")
        else:
            checks['stake_check'] = True
        
        if self.daily_loss_amount >= self.daily_loss_limit:
            rejection_reasons.append(f"Daily loss limit: ${self.daily_loss_amount:.2f} >= ${self.daily_loss_limit:.2f}")
            self.daily_loss_limit_breached = True
            self.is_paused = True
            self.pause_reason = "Daily loss limit reached"
        else:
            checks['daily_loss_check'] = True
        
        if self.consecutive_losses >= 3:
            rejection_reasons.append(f"Consecutive losses: {self.consecutive_losses} >= 3")
            self.is_paused = True
            self.pause_reason = "Maximum consecutive losses reached"
        else:
            checks['consecutive_loss_check'] = True
        
        if self.open_trades_count >= self.max_open_trades:
            rejection_reasons.append(f"Open trades: {self.open_trades_count} >= {self.max_open_trades}")
        else:
            checks['max_trades_check'] = True
        
        if self.last_trade_time is not None:
            elapsed = (current_time - self.last_trade_time).total_seconds()
            if elapsed < self.cooldown_seconds:
                rejection_reasons.append(f"In cooldown: {int(self.cooldown_seconds - elapsed)}s remaining")
            else:
                checks['cooldown_check'] = True
        else:
            checks['cooldown_check'] = True
        
        if self.current_drawdown >= self.max_drawdown_pct:
            rejection_reasons.append(f"Drawdown: {self.current_drawdown:.1f}% >= {self.max_drawdown_pct:.1f}%")
            self.is_paused = True
            self.pause_reason = "Maximum drawdown exceeded"
        else:
            checks['drawdown_check'] = True
        
        is_valid = all(checks.values()) and not rejection_reasons
        final_stake = min(recommended_stake, max_stake_amount) if is_valid else 0.0
        
        return {
            'is_valid': is_valid,
            'reason': 'ALL_CHECKS_PASSED' if is_valid else '; '.join(rejection_reasons),
            'risk_recommendation': final_stake,
            'message': 'Trade validated' if is_valid else 'Trade rejected',
            'checks': checks,
            'rejection_reasons': rejection_reasons,
            'max_allowed_stake': max_stake_amount,
            'timestamp': current_time.isoformat()
        }
    
    def update_after_trade(self, outcome: str, pnl: float, trade_info: Optional[Dict] = None):
        current_time = datetime.now()
        
        if outcome == 'WIN':
            self.daily_win_count += 1
            self.daily_win_amount += pnl
            self.consecutive_wins += 1
            self.consecutive_losses = 0
        else:
            self.daily_loss_count += 1
            self.daily_loss_amount += abs(pnl)
            self.consecutive_losses += 1
            self.consecutive_wins = 0
        
        self.balance += pnl
        self.last_trade_time = current_time
        self.equity_curve.append(self.balance)
        
        if self.open_trades_count > 0:
            self.open_trades_count -= 1
        
        peak = max(self.equity_curve)
        self.current_drawdown = ((peak - self.balance) / peak) * 100
        self.max_drawdown = max(self.max_drawdown, self.current_drawdown)
        
        self.daily_trades.append({
            'timestamp': current_time.isoformat(),
            'outcome': outcome,
            'pnl': pnl,
            'balance_after': self.balance,
            **(trade_info or {})
        })
    
    def open_trade(self):
        if self.open_trades_count < self.max_open_trades:
            self.open_trades_count += 1
            return True
        return False
    
    def force_pause(self, reason: str):
        self.is_paused = True
        self.pause_reason = reason
    
    def resume_trading(self):
        self.is_paused = False
        self.pause_reason = ""
    
    def reset_daily_stats(self):
        self.daily_win_count = 0
        self.daily_loss_count = 0
        self.daily_loss_amount = 0.0
        self.daily_win_amount = 0.0
        self.daily_trades = []
        self.daily_loss_limit_breached = False
        if self.is_paused and "daily loss" in self.pause_reason.lower():
            self.is_paused = False
            self.pause_reason = ""
    
    def get_risk_status(self) -> Dict[str, Any]:
        cooldown_remaining = 0
        if self.last_trade_time:
            elapsed = (datetime.now() - self.last_trade_time).total_seconds()
            cooldown_remaining = max(0, self.cooldown_seconds - elapsed)
        
        return {
            'balance': self.balance,
            'daily_pnl': self.daily_win_amount - self.daily_loss_amount,
            'daily_loss_amount': self.daily_loss_amount,
            'daily_loss_limit': self.daily_loss_limit,
            'daily_loss_remaining': max(0, self.daily_loss_limit - self.daily_loss_amount),
            'consecutive_losses': self.consecutive_losses,
            'consecutive_wins': self.consecutive_wins,
            'open_trades': self.open_trades_count,
            'max_open_trades': self.max_open_trades,
            'is_paused': self.is_paused,
            'pause_reason': self.pause_reason,
            'current_drawdown': self.current_drawdown,
            'max_drawdown_pct': self.max_drawdown_pct,
            'seconds_to_cooldown': int(cooldown_remaining),
            'confidence_threshold': self.confidence_threshold,
            'max_risk_pct': self.max_risk_pct,
            'trades_today': len(self.daily_trades),
            'win_rate_today': (self.daily_win_count / max(1, self.daily_win_count + self.daily_loss_count)) * 100
        }
    
    def can_trade_now(self) -> tuple:
        if self.is_paused:
            return False, self.pause_reason
        if self.daily_loss_amount >= self.daily_loss_limit:
            return False, "Daily loss limit reached"
        if self.consecutive_losses >= 3:
            return False, "Maximum consecutive losses"
        if self.current_drawdown >= self.max_drawdown_pct:
            return False, "Maximum drawdown exceeded"
        if self.last_trade_time:
            elapsed = (datetime.now() - self.last_trade_time).total_seconds()
            if elapsed < self.cooldown_seconds:
                return False, f"In cooldown ({int(self.cooldown_seconds - elapsed)}s)"
        if self.open_trades_count >= self.max_open_trades:
            return False, f"Max open trades ({self.open_trades_count}/{self.max_open_trades})"
        return True, "Trading allowed"
