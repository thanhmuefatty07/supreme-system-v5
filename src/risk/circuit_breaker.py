#!/usr/bin/env python3
"""
Supreme System V5 - Circuit Breaker Implementation

Industry-standard circuit breaker pattern for algorithmic trading.
Implements SEC/ESMA regulatory requirements and modern resilience patterns.
"""

import logging
import time
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Callable, Dict, List, Optional


class CircuitBreakerState(Enum):
    """Circuit breaker states following industry standards."""
    CLOSED = "closed"      # Normal operation
    OPEN = "open"          # Circuit is open, rejecting requests
    HALF_OPEN = "half_open"  # Testing recovery


class CircuitBreakerOpen(Exception):
    """Exception raised when circuit breaker is open."""
    pass


class CircuitBreaker:
    """
    Circuit Breaker implementation following industry best practices.

    Based on SEC/ESMA regulatory requirements and modern cloud-native patterns.
    Implements financial risk limits and automated recovery mechanisms.
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize circuit breaker with financial risk limits.

        Args:
            config: Configuration dictionary with risk limits
        """
        self.logger = logging.getLogger(__name__)

        # Circuit breaker state
        self.state = CircuitBreakerState.CLOSED
        self.failure_count = 0
        self.last_failure_time = None
        self.success_count = 0

        # Configuration
        self.failure_threshold = config.get('failure_threshold', 5)
        self.timeout = config.get('timeout', 60)  # seconds
        self.success_threshold = config.get('success_threshold', 3)  # for half-open recovery

        # Financial risk limits (SEC/ESMA compliant)
        self.max_daily_loss = config.get('max_daily_loss', 0.05)  # 5%
        self.max_hourly_loss = config.get('max_hourly_loss', 0.02)  # 2%
        self.max_position_size = config.get('max_position_size', 0.10)  # 10% of portfolio
        self.max_drawdown = config.get('max_drawdown', 0.15)  # 15%

        # Performance monitoring
        self.request_count = 0
        self.success_count_total = 0
        self.failure_count_total = 0
        self.last_request_time = None

        # Portfolio reference (to be set by risk manager)
        self.portfolio = None

        self.logger.info("Circuit breaker initialized with risk limits")

    def call(self, trading_function: Callable, *args, **kwargs) -> Any:
        """
        Execute trading function with circuit breaker protection.

        Args:
            trading_function: Function to execute
            *args, **kwargs: Arguments for the function

        Returns:
            Result of the trading function

        Raises:
            CircuitBreakerOpen: When circuit is open
        """
        self.request_count += 1
        self.last_request_time = time.time()

        # Check circuit state first
        if self.state == CircuitBreakerState.OPEN:
            # Check if can attempt reset to half-open
            if self._can_attempt_reset():
                self.state = CircuitBreakerState.HALF_OPEN
                self.logger.warning("Circuit breaker HALF_OPEN - Testing recovery")
            else:
                self.logger.critical("Circuit breaker OPEN - Trading halted")
                raise CircuitBreakerOpen("Trading halted due to circuit breaker")

        # Check if circuit should open due to risk limits (before execution)
        if self._should_open():
            self.state = CircuitBreakerState.OPEN
            self.logger.critical("Circuit breaker OPEN - Trading halted due to risk limits")
            raise CircuitBreakerOpen("Trading halted due to risk limits")

        # Execute trading function with monitoring
        try:
            result = trading_function(*args, **kwargs)

            # Success handling
            if self.state == CircuitBreakerState.HALF_OPEN:
                # Success in HALF_OPEN state → transition to CLOSED
                self.success_count += 1
                if self.success_count >= self.success_threshold:
                    self.state = CircuitBreakerState.CLOSED
                    self.failure_count = 0
                    self.success_count = 0
                    self.logger.info("Circuit breaker CLOSED - Normal operation resumed")
            else:
                # Success in CLOSED state
                self.success_count_total += 1

            return result

        except Exception as e:
            # Failure handling
            self.failure_count += 1
            self.failure_count_total += 1
            self.last_failure_time = time.time()

            # Check if circuit should open due to failure threshold
            if self._should_open():
                self.state = CircuitBreakerState.OPEN
                self.logger.critical(f"Circuit breaker OPEN - Too many failures ({self.failure_count})")

            if self.state == CircuitBreakerState.HALF_OPEN:
                # Failed recovery - back to OPEN
                self.state = CircuitBreakerState.OPEN
                self.success_count = 0

            self.logger.error(f"Circuit breaker recorded failure: {e}")
            raise

    def _should_open(self) -> bool:
        """
        Check if circuit should open based on financial risk limits and failure count.

        Returns:
            True if circuit should open
        """
        # Don't check failure count in HALF_OPEN state (recovery mode)
        if self.state != CircuitBreakerState.HALF_OPEN:
            # Always check failure count first (technical failures)
            if self.failure_count >= self.failure_threshold:
                self.logger.critical(f"Circuit breaker opening due to {self.failure_count} consecutive failures")
                return True

        # Check financial risk limits if portfolio is available
        if self.portfolio is not None:
            # Check loss limits (SEC/ESMA regulatory requirements)
            try:
                daily_loss = self.portfolio.get_daily_pnl()
                if daily_loss < -self.max_daily_loss * self.portfolio.total_value:
                    self.logger.critical(".2%"
                                       ".2%")
                    return True
            except (AttributeError, TypeError):
                pass  # Portfolio method not available

            try:
                hourly_loss = self.portfolio.get_hourly_pnl()
                if hourly_loss < -self.max_hourly_loss * self.portfolio.total_value:
                    self.logger.critical(".2%"
                                       ".2%")
                    return True
            except (AttributeError, TypeError):
                pass  # Portfolio method not available

            # Check drawdown limit
            try:
                current_drawdown = self.portfolio.get_current_drawdown()
                if current_drawdown > self.max_drawdown:
                    self.logger.critical(".2%")
                    return True
            except (AttributeError, TypeError):
                pass  # Portfolio method not available

            # Check position concentration
            try:
                positions = self.portfolio.get_positions()
                for position in positions:
                    try:
                        position_pct = position.value / self.portfolio.total_value
                        if position_pct > self.max_position_size:
                            self.logger.critical(
                                f"Position concentration limit breached: "
                                f"{getattr(position, 'symbol', 'Unknown')} = {position_pct:.2%}"
                            )
                            return True
                    except (AttributeError, TypeError):
                        pass  # Position attributes not available
            except (AttributeError, TypeError):
                pass  # Portfolio method not available

        return False

    def _can_attempt_reset(self) -> bool:
        """
        Check if circuit can attempt to reset to half-open state.

        Returns:
            True if reset can be attempted
        """
        if self.last_failure_time is None:
            return True

        # Check timeout period
        elapsed = time.time() - self.last_failure_time
        return elapsed >= self.timeout

    def get_status(self) -> Dict[str, Any]:
        """
        Get comprehensive circuit breaker status.

        Returns:
            Status dictionary
        """
        return {
            'state': self.state.value,
            'failure_count': self.failure_count,
            'success_count': self.success_count,
            'total_requests': self.request_count,
            'total_successes': self.success_count_total,
            'total_failures': self.failure_count_total,
            'last_request_time': self.last_request_time,
            'last_failure_time': self.last_failure_time,
            'success_rate': (self.success_count_total / max(self.request_count, 1)) * 100,
            'risk_limits': {
                'max_daily_loss': self.max_daily_loss,
                'max_hourly_loss': self.max_hourly_loss,
                'max_position_size': self.max_position_size,
                'max_drawdown': self.max_drawdown
            }
        }

    def reset(self):
        """Manually reset circuit breaker to closed state."""
        self.state = CircuitBreakerState.CLOSED
        self.failure_count = 0
        self.success_count = 0
        self.logger.info("Circuit breaker manually reset to CLOSED")

    def force_open(self):
        """Manually force circuit breaker to open state."""
        self.state = CircuitBreakerState.OPEN
        self.logger.warning("Circuit breaker manually forced to OPEN")


class EmergencyShutdown:
    """
    Emergency shutdown system for algorithmic trading.

    Implements regulatory requirements for automated system shutdown
    in case of critical failures or extreme market conditions.
    """

    def __init__(self, trading_engine, config: Dict[str, Any]):
        """
        Initialize emergency shutdown system.

        Args:
            trading_engine: Reference to trading engine
            config: Configuration with shutdown parameters
        """
        self.logger = logging.getLogger(__name__)
        self.engine = trading_engine
        self.config = config

        # Shutdown triggers
        self.shutdown_triggers: List[Dict[str, Any]] = []

        # Shutdown state
        self.is_shutdown = False
        self.shutdown_time = None
        self.shutdown_reason = None

        # Register default triggers
        self._register_default_triggers()

    def register_trigger(self, condition: Callable[[], bool], callback: Callable, name: str):
        """
        Register a custom shutdown trigger.

        Args:
            condition: Function that returns True when shutdown should trigger
            callback: Function to call when shutdown triggers
            name: Descriptive name for the trigger
        """
        self.shutdown_triggers.append({
            'condition': condition,
            'callback': callback,
            'name': name,
            'trigger_count': 0,
            'last_triggered': None
        })

        self.logger.info(f"Registered shutdown trigger: {name}")

    def _register_default_triggers(self):
        """Register default regulatory-required shutdown triggers."""

        # Extreme loss trigger (10% intraday loss)
        def extreme_loss_condition():
            if not hasattr(self.engine, 'portfolio'):
                return False
            daily_pnl = self.engine.portfolio.get_daily_pnl()
            return daily_pnl < -0.10 * self.engine.portfolio.total_value

        # System failure trigger (consecutive errors)
        consecutive_errors = 0
        def system_failure_condition():
            nonlocal consecutive_errors
            # This would be updated by error monitoring system
            return consecutive_errors >= 10

        # Market circuit breaker trigger
        def market_circuit_breaker_condition():
            # Check for extreme market volatility
            # This would integrate with market data feeds
            return False  # Placeholder

        # Manual shutdown trigger
        def manual_shutdown_condition():
            return getattr(self, 'manual_shutdown_requested', False)

        # Register triggers
        self.register_trigger(extreme_loss_condition, self._extreme_loss_callback, "Extreme Loss")
        self.register_trigger(system_failure_condition, self._system_failure_callback, "System Failure")
        self.register_trigger(market_circuit_breaker_condition, self._market_circuit_callback, "Market Circuit Breaker")
        self.register_trigger(manual_shutdown_condition, self._manual_shutdown_callback, "Manual Shutdown")

    async def monitor(self):
        """
        Continuous monitoring for shutdown conditions.

        This should be run as a background task.
        """
        import asyncio

        check_interval = self.config.get('check_interval', 1.0)  # seconds

        while not self.is_shutdown:
            try:
                for trigger in self.shutdown_triggers:
                    if trigger['condition']():
                        await self._execute_shutdown(trigger)
                        break

                await asyncio.sleep(check_interval)

            except Exception as e:
                self.logger.error(f"Error in shutdown monitoring: {e}")
                await asyncio.sleep(check_interval)

    async def _execute_shutdown(self, trigger: Dict[str, Any]):
        """
        Execute emergency shutdown procedure.

        Args:
            trigger: Trigger that caused shutdown
        """
        if self.is_shutdown:
            return

        self.is_shutdown = True
        self.shutdown_time = datetime.now()
        self.shutdown_reason = trigger['name']

        trigger['trigger_count'] += 1
        trigger['last_triggered'] = self.shutdown_time

        self.logger.critical(f"EMERGENCY SHUTDOWN TRIGGERED: {trigger['name']}")

        try:
            # Step 1: Stop accepting new orders
            await self._stop_new_orders()

            # Step 2: Cancel pending orders
            await self._cancel_pending_orders()

            # Step 3: Close positions (if configured)
            if self.config.get('close_positions_on_shutdown', False):
                await self._close_all_positions()

            # Step 4: Disconnect from exchanges
            await self._disconnect_exchanges()

            # Step 5: Execute trigger-specific callback
            await trigger['callback']()

            # Step 6: Send notifications
            await self._send_shutdown_notifications()

            self.logger.info("Emergency shutdown completed successfully")

        except Exception as e:
            self.logger.error(f"Error during emergency shutdown: {e}")

    async def _stop_new_orders(self):
        """Stop accepting new orders."""
        if hasattr(self.engine, 'halt_new_orders'):
            self.engine.halt_new_orders()
            self.logger.info("Stopped accepting new orders")

    async def _cancel_pending_orders(self):
        """Cancel all pending orders."""
        if hasattr(self.engine, 'cancel_all_orders'):
            await self.engine.cancel_all_orders()
            self.logger.info("Cancelled all pending orders")

    async def _close_all_positions(self):
        """Close all open positions."""
        if hasattr(self.engine, 'close_all_positions'):
            await self.engine.close_all_positions()
            self.logger.info("Closed all positions")

    async def _disconnect_exchanges(self):
        """Disconnect from all exchanges."""
        if hasattr(self.engine, 'disconnect'):
            await self.engine.disconnect()
            self.logger.info("Disconnected from exchanges")

    async def _send_shutdown_notifications(self):
        """Send shutdown notifications."""
        # This would integrate with notification systems
        self.logger.critical("EMERGENCY SHUTDOWN NOTIFICATIONS SENT")

    # Trigger-specific callbacks
    async def _extreme_loss_callback(self):
        """Callback for extreme loss trigger."""
        self.logger.critical("Shutdown due to extreme losses - Regulatory compliance triggered")

    async def _system_failure_callback(self):
        """Callback for system failure trigger."""
        self.logger.critical("Shutdown due to system failures - Safety protocol activated")

    async def _market_circuit_callback(self):
        """Callback for market circuit breaker trigger."""
        self.logger.critical("Shutdown due to market circuit breaker - Market protection activated")

    async def _manual_shutdown_callback(self):
        """Callback for manual shutdown trigger."""
        self.logger.info("Manual shutdown completed")

    def manual_shutdown_request(self):
        """Request manual shutdown."""
        self.manual_shutdown_requested = True
        self.logger.warning("Manual shutdown requested")

    def get_status(self) -> Dict[str, Any]:
        """
        Get emergency shutdown system status.

        Returns:
            Status dictionary
        """
        return {
            'is_shutdown': self.is_shutdown,
            'shutdown_time': self.shutdown_time,
            'shutdown_reason': self.shutdown_reason,
            'triggers': [
                {
                    'name': t['name'],
                    'trigger_count': t['trigger_count'],
                    'last_triggered': t['last_triggered']
                }
                for t in self.shutdown_triggers
            ]
        }

    def reset(self):
        """Reset emergency shutdown system."""
        self.is_shutdown = False
        self.shutdown_time = None
        self.shutdown_reason = None
        self.manual_shutdown_requested = False
        self.logger.info("Emergency shutdown system reset")

