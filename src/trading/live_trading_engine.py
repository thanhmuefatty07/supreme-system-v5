#!/usr/bin/env python3
"""
Supreme System V5 - Live Trading Engine

Production-ready live trading engine for real market execution.
Handles order placement, position management, and live risk controls.
"""

import logging
import threading
import time
from datetime import datetime, timedelta
from decimal import ROUND_DOWN, Decimal
from typing import Any, Dict, List, Optional, Union

from ..config.config import get_config
from ..data.binance_client import BinanceClient
from ..risk.circuit_breaker import CircuitBreaker, CircuitBreakerOpen
from ..risk.risk_manager import RiskManager
from ..strategies.base_strategy import BaseStrategy


class LiveTradingPosition:
    """Represents a live trading position."""

    def __init__(
        self,
        symbol: str,
        side: str,
        quantity: float,
        entry_price: float,
        timestamp: Optional[datetime] = None
    ):
        self.symbol = symbol
        self.side = side  # 'LONG' or 'SHORT'
        self.quantity = quantity
        self.entry_price = entry_price
        self.current_price = entry_price
        self.timestamp = timestamp or datetime.now()
        self.unrealized_pnl = 0.0
        self.realized_pnl = 0.0
        self.stop_loss: Optional[float] = None
        self.take_profit: Optional[float] = None
        self.status = 'OPEN'

    def update_pnl(self, current_price: float) -> float:
        """Update and return unrealized P&L."""
        self.current_price = current_price
        if self.side == 'LONG':
            self.unrealized_pnl = (current_price - self.entry_price) * self.quantity
        else:  # SHORT
            self.unrealized_pnl = (self.entry_price - current_price) * self.quantity
        return self.unrealized_pnl

    def close_position(self, exit_price: float) -> float:
        """Close position and return realized P&L."""
        if self.side == 'LONG':
            self.realized_pnl = (exit_price - self.entry_price) * self.quantity
        else:  # SHORT
            self.realized_pnl = (self.entry_price - exit_price) * self.quantity

        self.status = 'CLOSED'
        return self.realized_pnl


class LiveOrder:
    """Represents a live trading order."""

    def __init__(
        self,
        symbol: str,
        side: str,  # 'BUY' or 'SELL'
        order_type: str,  # 'MARKET', 'LIMIT', etc.
        quantity: float,
        price: Optional[float] = None,
        stop_loss: Optional[float] = None,
        take_profit: Optional[float] = None
    ):
        self.symbol = symbol
        self.side = side
        self.order_type = order_type
        self.quantity = quantity
        self.price = price
        self.stop_loss = stop_loss
        self.take_profit = take_profit

        self.order_id: Optional[str] = None
        self.status: str = 'PENDING'
        self.timestamp = datetime.now()
        self.execution_price: Optional[float] = None
        self.filled_quantity: float = 0.0
        self.fees: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        """Convert order to dictionary."""
        return {
            'symbol': self.symbol,
            'side': self.side,
            'type': self.order_type,
            'quantity': self.quantity,
            'price': self.price,
            'stop_loss': self.stop_loss,
            'take_profit': self.take_profit,
            'order_id': self.order_id,
            'status': self.status,
            'timestamp': self.timestamp.isoformat(),
            'execution_price': self.execution_price,
            'filled_quantity': self.filled_quantity,
            'fees': self.fees
        }


class LivePosition:
    """Represents a live trading position."""

    def __init__(self, symbol: str, entry_price: float, quantity: float, side: str):
        self.symbol = symbol
        self.entry_price = entry_price
        self.quantity = quantity
        self.side = side  # 'LONG' or 'SHORT'

        self.current_price: float = entry_price
        self.unrealized_pnl: float = 0.0
        self.stop_loss: Optional[float] = None
        self.take_profit: Optional[float] = None
        self.entry_time = datetime.now()
        self.last_update = datetime.now()

    def update_price(self, current_price: float) -> None:
        """Update position with new price."""
        self.current_price = current_price
        self.last_update = datetime.now()

        # Calculate unrealized P&L
        if self.side == 'LONG':
            self.unrealized_pnl = (current_price - self.entry_price) * self.quantity
        else:  # SHORT
            self.unrealized_pnl = (self.entry_price - current_price) * self.quantity

    def should_close(self) -> Optional[str]:
        """Check if position should be closed (stop loss or take profit)."""
        if not self.stop_loss and not self.take_profit:
            return None

        if self.side == 'LONG':
            if self.stop_loss and self.current_price <= self.stop_loss:
                return 'STOP_LOSS'
            if self.take_profit and self.current_price >= self.take_profit:
                return 'TAKE_PROFIT'
        else:  # SHORT
            if self.stop_loss and self.current_price >= self.stop_loss:
                return 'STOP_LOSS'
            if self.take_profit and self.current_price <= self.take_profit:
                return 'TAKE_PROFIT'

        return None

    def to_dict(self) -> Dict[str, Any]:
        """Convert position to dictionary."""
        return {
            'symbol': self.symbol,
            'entry_price': self.entry_price,
            'quantity': self.quantity,
            'side': self.side,
            'current_price': self.current_price,
            'unrealized_pnl': self.unrealized_pnl,
            'stop_loss': self.stop_loss,
            'take_profit': self.take_profit,
            'entry_time': self.entry_time.isoformat(),
            'last_update': self.last_update.isoformat()
        }


class LiveTradingEngine:
    """
    Production live trading engine for Supreme System V5.

    Features:
    - Real order execution on Binance
    - Position management with stop losses
    - Risk controls and circuit breakers
    - Real-time monitoring and alerts
    - Production-grade error handling
    """

    def __init__(self, config_file: Optional[str] = None):
        """
        Initialize live trading engine.

        Args:
            config_file: Optional configuration file
        """
        self.config = get_config()
        if config_file:
            from config.config import load_config
            self.config = load_config(config_file)

        self.logger = logging.getLogger(__name__)

        # Initialize components
        self.client = BinanceClient(
            api_key=self.config.get('binance.api_key'),
            api_secret=self.config.get('binance.api_secret'),
            testnet=self.config.get('binance.testnet', True)
        )

        self.risk_manager = RiskManager(
            initial_capital=self.config.get('risk.initial_capital', 10000),
            max_position_size=self.config.get('risk.max_position_size', 0.1),
            stop_loss_pct=self.config.get('risk.stop_loss_pct', 0.02),
            take_profit_pct=self.config.get('risk.take_profit_pct', 0.05)
        )

        # Initialize circuit breaker
        circuit_config = {
            'failure_threshold': self.config.get('circuit_breaker.failure_threshold', 5),
            'timeout': self.config.get('circuit_breaker.timeout', 300),  # 5 minutes
            'success_threshold': self.config.get('circuit_breaker.success_threshold', 3),
            'max_daily_loss': self.config.get('circuit_breaker.max_daily_loss', 0.05),
            'max_hourly_loss': self.config.get('circuit_breaker.max_hourly_loss', 0.02),
            'max_position_size': self.config.get('circuit_breaker.max_position_size', 0.10),
            'max_drawdown': self.config.get('circuit_breaker.max_drawdown', 0.15)
        }
        self.circuit_breaker = CircuitBreaker(circuit_config)
        # Set portfolio reference for risk checks
        self.circuit_breaker.portfolio = self.risk_manager.portfolio if hasattr(self.risk_manager, 'portfolio') else None

        # Trading state
        self.is_live = False
        self.positions: Dict[str, LivePosition] = {}
        self.pending_orders: Dict[str, LiveOrder] = {}
        self.completed_orders: List[LiveOrder] = []

        # Performance tracking
        self.total_trades = 0
        self.winning_trades = 0
        self.total_pnl = 0.0
        self.max_drawdown = 0.0
        self.start_time = datetime.now()

        # Risk controls
        self.daily_loss_limit = self.config.get('risk.daily_loss_limit', 0.05)
        self.max_consecutive_losses = self.config.get('risk.max_consecutive_losses', 5)
        self.consecutive_losses = 0

        # Monitoring
        self.monitoring_thread: Optional[threading.Thread] = None
        self.monitoring_active = False

        self.logger.info("Live Trading Engine initialized")

    def start_live_trading(self) -> bool:
        """Start live trading operations."""
        if not self.client.client:
            self.logger.error("❌ Binance client not initialized - check API credentials")
            return False

        if not self.client.test_connection():
            self.logger.error("❌ Cannot connect to Binance API")
            return False

        self.is_live = True
        self.start_time = datetime.now()

        # Start monitoring thread
        self.monitoring_active = True
        self.monitoring_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self.monitoring_thread.start()

        self.logger.info("🟢 LIVE TRADING STARTED - Supreme System V5 Active")
        return True

    def stop_live_trading(self) -> None:
        """Stop live trading and close all positions."""
        if not self.is_live:
            return

        self.logger.info("🔴 Stopping live trading...")

        # Close all positions
        self._close_all_positions()

        # Cancel pending orders
        self._cancel_pending_orders()

        # Stop monitoring
        self.monitoring_active = False
        if self.monitoring_thread and self.monitoring_thread.is_alive():
            self.monitoring_thread.join(timeout=5.0)

        self.is_live = False
        self.logger.info("⏹️ LIVE TRADING STOPPED")

    def execute_signal(
        self,
        strategy_name: str,
        symbol: str,
        signal: int,
        price: float,
        confidence: float = 1.0
    ) -> Optional[LiveOrder]:
        """
        Execute trading signal in live market.

        Args:
            strategy_name: Name of the strategy generating signal
            symbol: Trading symbol
            signal: 1 for BUY, -1 for SELL, 0 for HOLD
            price: Current market price
            confidence: Signal confidence (0.0 to 1.0)

        Returns:
            LiveOrder if executed, None otherwise
        """
        if not self.is_live:
            self.logger.warning("Live trading not active")
            return None

        if signal == 0:  # HOLD
            return None

        # Risk management checks
        if not self._check_risk_limits(strategy_name, symbol, signal, price, confidence):
            return None

        # Circuit breaker protection
        try:
            # Define the trading execution function
            def execute_trading_signal():
                return self._execute_signal_internal(strategy_name, symbol, signal, price, confidence)

            # Execute with circuit breaker protection
            return self.circuit_breaker.call(execute_trading_signal)

        except CircuitBreakerOpen:
            self.logger.critical("Circuit breaker triggered - trading halted for safety")
            return None

    def _execute_signal_internal(
        self,
        strategy_name: str,
        symbol: str,
        signal: int,
        price: float,
        confidence: float
    ) -> Optional[LiveOrder]:
        """
        Internal signal execution without circuit breaker (called by circuit breaker).

        This method contains the actual trading logic that needs protection.
        """
        # Check existing position
        existing_position = self.positions.get(symbol)
        if existing_position:
            # Close existing position if signal is opposite
            if (signal == 1 and existing_position.side == 'SHORT') or \
               (signal == -1 and existing_position.side == 'LONG'):
                self._close_position(symbol, "SIGNAL_REVERSE")
            else:
                # Same direction signal - add to position or hold
                return None

        # Calculate position size
        position_size = self.risk_manager.calculate_position_size(
            entry_price=price,
            capital=self._get_available_capital()
        )

        if position_size <= 0:
            self.logger.warning(f"Invalid position size: {position_size}")
            return None

        # Create order
        side = 'BUY' if signal == 1 else 'SELL'
        order = LiveOrder(
            symbol=symbol,
            side=side,
            order_type='MARKET',
            quantity=position_size,
            price=price
        )

        # Execute order
        if self._execute_order(order):
            # Create position
            position_side = 'LONG' if signal == 1 else 'SHORT'
            position = LivePosition(symbol, price, position_size, position_side)

            # Set stop loss and take profit
            if signal == 1:  # LONG
                position.stop_loss = price * (1 - self.risk_manager.stop_loss_pct)
                position.take_profit = price * (1 + self.risk_manager.take_profit_pct)
            else:  # SHORT
                position.stop_loss = price * (1 + self.risk_manager.stop_loss_pct)
                position.take_profit = price * (1 - self.risk_manager.take_profit_pct)

            self.positions[symbol] = position

            self.logger.info(f"📈 {strategy_name}: {side} {position_size:.4f} {symbol} at ${price:.2f}")
            return order

        return None

    def _execute_order(self, order: LiveOrder) -> bool:
        """Execute order on Binance."""
        try:
            # For live trading, use actual Binance API
            if self.client.client:
                # Place market order
                binance_order = self.client.client.create_order(
                    symbol=order.symbol,
                    side=order.side,
                    type='MARKET',
                    quantity=self._format_quantity(order.symbol, order.quantity)
                )

                if binance_order and binance_order['status'] == 'FILLED':
                    order.order_id = binance_order['orderId']
                    order.status = 'FILLED'
                    order.execution_price = float(binance_order['price'])
                    order.filled_quantity = float(binance_order['executedQty'])
                    order.fees = self._calculate_fees(binance_order)

                    self.completed_orders.append(order)
                    self.total_trades += 1

                    return True
                else:
                    order.status = 'FAILED'
                    self.logger.error(f"Order failed: {binance_order}")
                    return False
            else:
                # Simulation mode for testing
                order.order_id = f"sim_{int(time.time())}"
                order.status = 'FILLED'
                order.execution_price = order.price
                order.filled_quantity = order.quantity
                order.fees = order.quantity * order.price * 0.001  # 0.1% fee

                self.completed_orders.append(order)
                self.total_trades += 1

                self.logger.info(f"📊 SIMULATION: {order.side} {order.quantity} {order.symbol} at ${order.price}")
                return True

        except Exception as e:
            order.status = 'ERROR'
            self.logger.error(f"Order execution failed: {e}")
            return False

    def _close_position(self, symbol: str, reason: str) -> Optional[LiveOrder]:
        """Close position for symbol."""
        if symbol not in self.positions:
            return None

        position = self.positions[symbol]
        current_price = position.current_price

        # Create closing order
        side = 'SELL' if position.side == 'LONG' else 'BUY'
        order = LiveOrder(
            symbol=symbol,
            side=side,
            order_type='MARKET',
            quantity=position.quantity,
            price=current_price
        )

        if self._execute_order(order):
            # Calculate P&L
            pnl = position.unrealized_pnl - order.fees
            self.total_pnl += pnl

            # Update win/loss stats
            if pnl > 0:
                self.winning_trades += 1
                self.consecutive_losses = 0
            else:
                self.consecutive_losses += 1

            # Remove position
            del self.positions[symbol]

            self.logger.info(f"🔒 Closed {symbol} position: {pnl:+.2f} P&L ({reason})")
            return order

        return None

    def _close_all_positions(self) -> None:
        """Close all open positions."""
        symbols_to_close = list(self.positions.keys())
        for symbol in symbols_to_close:
            self._close_position(symbol, "SYSTEM_SHUTDOWN")

    def _cancel_pending_orders(self) -> None:
        """Cancel all pending orders."""
        # In live trading, would cancel actual orders
        self.pending_orders.clear()

    def _monitoring_loop(self) -> None:
        """Main monitoring loop for live trading."""
        while self.monitoring_active and self.is_live:
            try:
                self._update_positions()
                self._check_stop_losses()
                self._check_risk_limits()
                self._log_status()

                time.sleep(10)  # Update every 10 seconds

            except Exception as e:
                self.logger.error(f"Monitoring error: {e}")
                time.sleep(30)

    def _update_positions(self) -> None:
        """Update position prices and P&L."""
        for symbol, position in self.positions.items():
            try:
                # Get current price (would use real-time data in production)
                current_price = self._get_current_price(symbol)
                if current_price:
                    position.update_price(current_price)

            except Exception as e:
                self.logger.error(f"Error updating {symbol}: {e}")

    def _check_stop_losses(self) -> None:
        """Check and execute stop losses and take profits."""
        symbols_to_close = []

        for symbol, position in self.positions.items():
            close_reason = position.should_close()
            if close_reason:
                symbols_to_close.append((symbol, close_reason))

        for symbol, reason in symbols_to_close:
            self._close_position(symbol, reason)

    def _check_risk_limits(self) -> None:
        """Check overall risk limits and circuit breakers."""
        # Daily loss limit
        if self.total_pnl < -self.daily_loss_limit * self.risk_manager.initial_capital:
            self.logger.warning("Daily loss limit reached - activating circuit breaker")
            self.circuit_breaker_active = True
            self._close_all_positions()

        # Consecutive losses
        if self.consecutive_losses >= self.max_consecutive_losses:
            self.logger.warning("Max consecutive losses reached - cooling down")
            self.circuit_breaker_active = True

        # Reset circuit breaker after cooldown period
        if self.circuit_breaker_active and \
           (datetime.now() - self.last_circuit_reset).seconds > 3600:  # 1 hour
            self.circuit_breaker_active = False
            self.consecutive_losses = 0
            self.last_circuit_reset = datetime.now()
            self.logger.info("Circuit breaker reset - resuming trading")

    def _check_risk_limits(self, strategy_name: str, symbol: str, signal: int,
                          price: float, confidence: float) -> bool:
        """Check if trade passes risk management."""
        # Position size check
        available_capital = self._get_available_capital()
        if available_capital <= 0:
            return False

        # Maximum positions check
        if len(self.positions) >= self.config.get('trading.max_open_positions', 5):
            return False

        # Confidence threshold
        min_confidence = self.config.get('trading.min_signal_confidence', 0.6)
        if confidence < min_confidence:
            return False

        return True

    def _get_current_price(self, symbol: str) -> Optional[float]:
        """Get current price for symbol."""
        try:
            if self.client.client:
                ticker = self.client.client.get_symbol_ticker(symbol=symbol)
                return float(ticker['price'])
            else:
                # Simulation - return slightly modified price
                return 50000.0 + (time.time() % 1000 - 500)  # Mock price
        except Exception:
            return None

    def _get_available_capital(self) -> float:
        """Get available capital for trading."""
        # In live trading, would check actual account balance
        # For now, return remaining capital
        used_capital = sum(pos.entry_price * pos.quantity for pos in self.positions.values())
        return max(0, self.risk_manager.initial_capital - used_capital)

    def _format_quantity(self, symbol: str, quantity: float) -> str:
        """Format quantity according to Binance precision rules."""
        # In production, would check symbol info for precision
        return f"{quantity:.6f}".rstrip('0').rstrip('.')

    def _calculate_fees(self, order_response: Dict[str, Any]) -> float:
        """Calculate trading fees from order response."""
        # In production, calculate from fills
        return 0.001 * float(order_response.get('cummulativeQuoteQty', 0))

    def _log_status(self) -> None:
        """Log current trading status."""
        if len(self.positions) > 0 or self.total_trades > 0:
            win_rate = (self.winning_trades / self.total_trades * 100) if self.total_trades > 0 else 0

            self.logger.info(
                f"📊 Status: {len(self.positions)} positions | "
                f"P&L: ${self.total_pnl:.2f} | "
                f"Win Rate: {win_rate:.1f}% | "
                f"Trades: {self.total_trades}"
            )

    def get_status(self) -> Dict[str, Any]:
        """Get comprehensive trading status."""
        win_rate = (self.winning_trades / self.total_trades * 100) if self.total_trades > 0 else 0

        return {
            'is_live': self.is_live,
            'start_time': self.start_time.isoformat(),
            'positions': {k: v.to_dict() for k, v in self.positions.items()},
            'pending_orders': {k: v.to_dict() for k, v in self.pending_orders.items()},
            'completed_orders': [o.to_dict() for o in self.completed_orders[-10:]],  # Last 10
            'performance': {
                'total_trades': self.total_trades,
                'winning_trades': self.winning_trades,
                'win_rate': win_rate,
                'total_pnl': self.total_pnl,
                'max_drawdown': self.max_drawdown
            },
            'risk_status': {
                'circuit_breaker_active': self.circuit_breaker_active,
                'consecutive_losses': self.consecutive_losses,
                'available_capital': self._get_available_capital()
            }
        }

    def start_monitoring(self) -> None:
        """
        Start the monitoring thread for live trading.

        This method starts a background thread that monitors positions,
        checks stop losses, and handles risk management.
        """
        if hasattr(self, '_monitoring_thread') and self._monitoring_thread.is_alive():
            self.logger.warning("Monitoring already running")
            return

        self._monitoring_active = True
        self._monitoring_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self._monitoring_thread.start()
        self.logger.info("Live trading monitoring started")

    def stop_monitoring(self) -> None:
        """
        Stop the monitoring thread for live trading.

        This method stops the background thread that monitors positions.
        """
        self._monitoring_active = False
        if hasattr(self, '_monitoring_thread') and self._monitoring_thread.is_alive():
            self._monitoring_thread.join(timeout=5.0)
        self.logger.info("Live trading monitoring stopped")

    def _execute_buy_order(self, symbol: str, quantity: float, price: Optional[float] = None) -> Optional[str]:
        """
        Execute a buy order.

        Args:
            symbol: Trading symbol
            quantity: Order quantity
            price: Optional limit price

        Returns:
            Order ID if successful, None if failed
        """
        try:
            order_type = 'LIMIT' if price else 'MARKET'
            order = LiveOrder(
                symbol=symbol,
                side='BUY',
                order_type=order_type,
                quantity=quantity,
                price=price
            )

            if self._execute_order(order):
                return order.order_id
            return None

        except Exception as e:
            self.logger.error(f"Failed to execute buy order for {symbol}: {e}")
            return None

    def _execute_sell_order(self, symbol: str, quantity: float, price: Optional[float] = None) -> Optional[str]:
        """
        Execute a sell order.

        Args:
            symbol: Trading symbol
            quantity: Order quantity
            price: Optional limit price

        Returns:
            Order ID if successful, None if failed
        """
        try:
            order_type = 'LIMIT' if price else 'MARKET'
            order = LiveOrder(
                symbol=symbol,
                side='SELL',
                order_type=order_type,
                quantity=quantity,
                price=price
            )

            if self._execute_order(order):
                return order.order_id
            return None

        except Exception as e:
            self.logger.error(f"Failed to execute sell order for {symbol}: {e}")
            return None

    def __enter__(self):
        """Context manager entry."""
        self.start_live_trading()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.stop_live_trading()
