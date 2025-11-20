#!/usr/bin/env python3
"""
Live Trading Engine 2.0 - The Trifecta Integration

Integrates Strategy Framework, Risk Management, and Execution Engine
into a unified production-ready live trading system.

Features:
- Real-time market data processing
- Enterprise risk management with Kelly Criterion and Circuit Breaker
- Smart order routing with slippage protection
- Performance tracking and analytics
"""

import asyncio
import logging
import time
from typing import Optional, Dict, Any, List
from datetime import datetime

from ..execution.router import SmartRouter
from ..risk.core import RiskManager
from ..strategies.base_strategy import BaseStrategy, Signal
from ..utils.validators import validate_market_data, ValidationError
from ..monitoring.metrics_collector import MetricsCollector


logger = logging.getLogger(__name__)


class LiveTradingEngineV2:
    """
    Live Trading Engine 2.0 - Enterprise Integration

    Orchestrates the complete trading flow:
    1. Market Data → Strategy → Signal
    2. Signal → Risk Manager → Position Size
    3. Position Size → Smart Router → Execution
    4. Execution → Performance Tracking → Risk Updates
    """

    def __init__(self, exchange_client, strategy: BaseStrategy, config: Dict[str, Any]):
        """
        Initialize the live trading engine.

        Args:
            exchange_client: Exchange API client (ccxt or similar)
            strategy: Trading strategy instance
            config: Engine configuration
        """
        self.logger = logging.getLogger("LiveEngineV2")
        self.strategy = strategy
        self.exchange = exchange_client

        # INTEGRATION 1: Enterprise Risk Manager
        risk_config = config.get('risk_config', {})
        # Ensure all required config keys are present
        risk_config.setdefault('max_risk_per_trade', 0.02)
        risk_config.setdefault('kelly_mode', 'half')
        risk_config.setdefault('daily_loss_limit', 0.05)
        risk_config.setdefault('risk_free_rate', 0.0)
        risk_config.setdefault('max_position_pct', 0.10)
        risk_config.setdefault('max_portfolio_pct', 0.50)

        self.risk_manager = RiskManager(
            capital=config.get('initial_capital', 10000.0),
            config=risk_config
        )

        # INTEGRATION 2: Smart Execution Router
        router_config = {
            "max_slippage": config.get('max_slippage', 0.01),
            "max_market_impact": 0.01,
            "iceberg_threshold": 1000,
            "max_chunk_size": 500,
            "liquidity_check": True,
            "impact_analysis": True,
            "twap_enabled": True,
            "execution_timeout": 30
        }
        self.router = SmartRouter(exchange_client, config=router_config)

        # Engine state
        self.is_running = False
        self.current_positions = {}  # symbol -> position_info
        self.trade_history = []

        # Performance tracking
        self.total_trades = 0
        self.winning_trades = 0
        self.total_pnl = 0.0

        # Configuration
        self.config = config
        self.update_interval = config.get('update_interval', 60)  # seconds

        # CRITICAL FIX: Add concurrency protection
        self._state_lock = asyncio.Lock()
        self._event_queue = asyncio.Queue()

        # PERFORMANCE MONITORING: Initialize metrics collector
        self.metrics = MetricsCollector()
        self.metrics.initialize(config.get('initial_capital', 10000.0))

        # NEW: Performance Monitoring
        self.metrics = MetricsCollector()
        self.metrics.initialize(config.get('initial_capital', 10000.0))

        # INTEGRATION 3: Performance Monitoring (The Heartbeat)
        self.metrics = MetricsCollector()
        self.metrics.initialize(config.get('initial_capital', 10000.0))

        self.logger.info("LiveTradingEngine V2 initialized with Trifecta Integration, Critical Fixes, and Performance Monitoring")

    async def on_market_update(self, market_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        The Heartbeat of the System - Process market update and execute trades.

        Flow:
        1. Generate Signal from Strategy
        2. Validate Signal
        3. Risk Check (Kelly + Circuit Breaker)
        4. Execute via Smart Router
        5. Update Performance Metrics

        Args:
            market_data: Current market data (OHLCV + metadata)

        Returns:
            Execution result or None if no action taken
        """
        # PERFORMANCE MONITORING: Start latency measurement
        start_time = time.perf_counter()

        # CRITICAL FIX: Validate input first
        try:
            validated_data = validate_market_data(market_data)
        except ValidationError as e:
            self.logger.warning(f"Invalid market data: {e}")
            return None

        symbol = validated_data.symbol

        # CRITICAL FIX: Use lock to protect shared state
        async with self._state_lock:
            try:
                # STEP 1: Generate Signal from Strategy
                signal = self.strategy.generate_signal(validated_data.dict())

                if not signal:
                    return None  # No action needed

                # Validate signal
                if not self.strategy.validate_signal(signal):
                    self.logger.warning(f"Invalid signal for {symbol}")
                    return None

                self.logger.info(f"Signal generated: {signal.side} {symbol} @ {signal.price} (strength: {signal.strength:.2f})")

                # STEP 2: Risk Check - Get Position Size
                # Use strategy's historical performance or default
                win_rate = self._get_strategy_win_rate()
                rr_ratio = self._get_strategy_rr_ratio()

                target_size = self.risk_manager.get_target_size(
                    win_rate=win_rate,
                    reward_risk_ratio=rr_ratio,
                    current_exposure=0.0  # TODO: Calculate actual exposure
                )

                if target_size <= 0:
                    self.logger.warning(f"Risk Manager rejected trade for {symbol}: Circuit breaker active or Kelly = 0")
                    return {'status': 'REJECTED', 'reason': 'risk_manager_rejection'}

                # Convert size from $ amount to quantity
                quantity = target_size / signal.price
                self.logger.info(f"Risk Manager approved: ${target_size:.2f} (~{quantity:.4f} {symbol})")

                # STEP 3: Execute via Smart Router
                self.logger.info(f"Executing {signal.side} {quantity:.4f} {symbol} via SmartRouter")

                result = await self.router.execute_order(
                    symbol=symbol,
                    side=signal.side,
                    amount=quantity,  # SmartRouter uses 'amount' not 'quantity'
                    order_type='market'  # Can be enhanced with limit orders
                )

                # STEP 4: Post-Trade Processing
                # Convert ExecutionResult to dict for consistency
                result_dict = result.to_dict() if hasattr(result, 'to_dict') else result

                if result_dict.get('status') == 'SUCCESS':
                    await self._on_trade_executed(signal, result_dict, target_size)

                return result_dict

            except Exception as e:
                self.logger.error(f"Error in market update processing: {e}", exc_info=True)
                return {'status': 'ERROR', 'error': str(e)}

        # PERFORMANCE MONITORING: Record latency
        end_time = time.perf_counter()
        latency_ms = (end_time - start_time) * 1000
        self.metrics.record_latency(latency_ms)

    async def _on_trade_executed(self, signal: Signal, execution_result: Dict[str, Any], position_size: float):
        """
        Handle post-trade updates and tracking.

        Args:
            signal: Original signal that triggered the trade
            execution_result: Result from Smart Router
            position_size: Position size in dollars
        """
        symbol = signal.symbol

        # Update position tracking
        if signal.side == 'buy':
            self.current_positions[symbol] = {
                'side': 'LONG',
                'quantity': execution_result.get('filled_quantity', 0),
                'entry_price': execution_result.get('avg_fill_price', signal.price),
                'timestamp': datetime.now(),
                'signal_metadata': signal.metadata
            }
        elif signal.side == 'sell' and symbol in self.current_positions:
            # Close position
            position = self.current_positions.pop(symbol)
            pnl = self._calculate_pnl(position, execution_result)

            # Update risk manager with trade result
            self.risk_manager.record_trade(pnl)

            # Update performance stats
            self.total_trades += 1
            if pnl > 0:
                self.winning_trades += 1
            self.total_pnl += pnl

            # PERFORMANCE MONITORING: Record trade in metrics collector
            self.metrics.record_trade(pnl, symbol)

            # Notify strategy
            self.strategy.on_order_filled({
                'symbol': symbol,
                'side': signal.side,
                'quantity': execution_result.get('filled_quantity', 0),
                'price': execution_result.get('avg_fill_price', signal.price),
                'pnl': pnl
            })

            self.logger.info(f"Position closed: {symbol}, PnL: ${pnl:.2f}, Total PnL: ${self.total_pnl:.2f}")

        # Record trade history
        self.trade_history.append({
            'timestamp': datetime.now(),
            'symbol': symbol,
            'signal': signal.to_dict(),
            'execution': execution_result,
            'position_size': position_size
        })

    def _calculate_pnl(self, position: Dict[str, Any], exit_result: Dict[str, Any]) -> float:
        """Calculate realized PnL for a closed position."""
        entry_price = position['entry_price']
        exit_price = exit_result.get('avg_fill_price', 0)
        quantity = position['quantity']

        if position['side'] == 'LONG':
            pnl = (exit_price - entry_price) * quantity
        else:  # SHORT
            pnl = (entry_price - exit_price) * quantity

        return pnl

    def _get_strategy_win_rate(self) -> float:
        """Get strategy's historical win rate or use default."""
        if self.total_trades > 0:
            return self.winning_trades / self.total_trades
        return 0.5  # Default 50% win rate

    def _get_strategy_rr_ratio(self) -> float:
        """Get strategy's risk/reward ratio or use default."""
        # TODO: Calculate from actual trade history
        return 2.0  # Default 2:1 reward:risk ratio

    async def start(self):
        """Start the live trading engine."""
        self.is_running = True
        self.logger.info("Live Trading Engine V2 started")

    async def stop(self):
        """Stop the live trading engine."""
        self.is_running = False
        self.logger.info("Live Trading Engine V2 stopped")

        # Close any open positions (optional, can be configurable)
        if self.current_positions:
            self.logger.warning(f"Stopping with {len(self.current_positions)} open positions")

    def get_status(self) -> Dict[str, Any]:
        """Get comprehensive engine status."""
        return {
            'is_running': self.is_running,
            'strategy': self.strategy.name,
            'total_trades': self.total_trades,
            'winning_trades': self.winning_trades,
            'win_rate': self._get_strategy_win_rate(),
            'total_pnl': self.total_pnl,
            'current_positions': len(self.current_positions),
            'circuit_breaker_active': self.risk_manager.circuit_breaker.is_active,
            'portfolio_value': self.risk_manager.current_capital,
            'risk_metrics': {
                'current_daily_drawdown': self.risk_manager.circuit_breaker.current_daily_drawdown,
                'current_weekly_drawdown': self.risk_manager.circuit_breaker.current_weekly_drawdown,
                'daily_limit': self.risk_manager.circuit_breaker.daily_limit
            }
        }

    def get_trade_history(self, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """Get trade history."""
        if limit:
            return self.trade_history[-limit:]
        return self.trade_history
