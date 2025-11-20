#!/usr/bin/env python3
"""
Risk Limits and Circuit Breakers - Enterprise Risk Controls

Implements circuit breaker patterns, daily loss limits, and position size controls
to prevent catastrophic losses and ensure responsible trading.
"""

import logging
from datetime import datetime, date
from typing import Optional, Dict, Any

logger = logging.getLogger(__name__)


class CircuitBreaker:
    """
    Circuit Breaker implementation for risk management.

    Automatically stops trading when loss limits are exceeded to prevent
    further damage during adverse market conditions.
    """

    def __init__(self, daily_limit_pct: float = 0.05, weekly_limit_pct: Optional[float] = None):
        """
        Initialize circuit breaker.

        Args:
            daily_limit_pct: Maximum daily drawdown before triggering (default 5%)
            weekly_limit_pct: Maximum weekly drawdown (optional)
        """
        self.daily_limit = abs(daily_limit_pct)  # Ensure positive
        self.weekly_limit = abs(weekly_limit_pct) if weekly_limit_pct else None

        # State tracking
        self.current_daily_drawdown = 0.0
        self.current_weekly_drawdown = 0.0
        self.is_active = False
        self.trigger_reason = None

        # Date tracking for resets
        self.last_reset_date = date.today()
        self.last_reset_week = self._get_week_number()

        # History for analysis
        self.daily_history = []
        self.trigger_history = []

    def _get_week_number(self) -> int:
        """Get current week number for weekly resets."""
        return datetime.now().isocalendar()[1]

    def _check_daily_reset(self) -> bool:
        """Check if daily reset is needed."""
        today = date.today()
        if today != self.last_reset_date:
            # Reset daily drawdown
            if self.current_daily_drawdown > 0:
                self.daily_history.append({
                    'date': self.last_reset_date.isoformat(),
                    'drawdown': self.current_daily_drawdown,
                    'triggered': self.is_active
                })

            self.current_daily_drawdown = 0.0
            self.last_reset_date = today

            # Reset circuit breaker if it was daily-triggered
            if self.trigger_reason == 'daily_limit':
                self.is_active = False
                self.trigger_reason = None
                logger.info("Circuit breaker reset for new trading day")

            return True
        return False

    def _check_weekly_reset(self) -> bool:
        """Check if weekly reset is needed."""
        current_week = self._get_week_number()
        if current_week != self.last_reset_week:
            # Reset weekly drawdown
            self.current_weekly_drawdown = 0.0
            self.last_reset_week = current_week

            # Reset circuit breaker if it was weekly-triggered
            if self.trigger_reason == 'weekly_limit':
                self.is_active = False
                self.trigger_reason = None
                logger.info("Circuit breaker reset for new trading week")

            return True
        return False

    def update(self, pnl_pct: float) -> bool:
        """
        Update drawdown status based on trade PnL percentage.

        Args:
            pnl_pct: Profit/Loss as percentage of capital

        Returns:
            False if breaker triggered (STOP TRADING), True otherwise
        """
        # Check for resets first
        self._check_daily_reset()
        self._check_weekly_reset()

        # Skip updates if already active
        if self.is_active:
            return False

        # Only count losses (negative PnL)
        if pnl_pct < 0:
            loss_pct = abs(pnl_pct)

            # Update daily drawdown
            self.current_daily_drawdown += loss_pct

            # Update weekly drawdown
            self.current_weekly_drawdown += loss_pct

            # Check daily limit
            if self.current_daily_drawdown >= self.daily_limit:
                self.is_active = True
                self.trigger_reason = 'daily_limit'
                self.trigger_history.append({
                    'timestamp': datetime.now().isoformat(),
                    'reason': 'daily_limit',
                    'drawdown': self.current_daily_drawdown,
                    'limit': self.daily_limit
                })
                logger.warning(".2f")
                return False

            # Check weekly limit
            if self.weekly_limit and self.current_weekly_drawdown >= self.weekly_limit:
                self.is_active = True
                self.trigger_reason = 'weekly_limit'
                self.trigger_history.append({
                    'timestamp': datetime.now().isoformat(),
                    'reason': 'weekly_limit',
                    'drawdown': self.current_weekly_drawdown,
                    'limit': self.weekly_limit
                })
                logger.warning(".2f")
                return False

        return True

    def reset(self):
        """Manually reset the circuit breaker."""
        self.current_daily_drawdown = 0.0
        self.current_weekly_drawdown = 0.0
        self.is_active = False
        self.trigger_reason = None
        logger.info("Circuit breaker manually reset")

    def get_status(self) -> Dict[str, Any]:
        """Get current circuit breaker status."""
        return {
            'is_active': self.is_active,
            'trigger_reason': self.trigger_reason,
            'daily_drawdown': self.current_daily_drawdown,
            'daily_limit': self.daily_limit,
            'weekly_drawdown': self.current_weekly_drawdown,
            'weekly_limit': self.weekly_limit,
            'last_reset_date': self.last_reset_date.isoformat(),
            'trigger_count': len(self.trigger_history)
        }


class PositionSizeLimiter:
    """
    Limits position sizes based on account risk and market conditions.
    """

    def __init__(self, max_position_pct: float = 0.1, max_portfolio_pct: float = 0.5):
        """
        Initialize position size limiter.

        Args:
            max_position_pct: Maximum size for any single position (as % of capital)
            max_portfolio_pct: Maximum total exposure across all positions
        """
        self.max_position_pct = max_position_pct
        self.max_portfolio_pct = max_portfolio_pct

    def validate_position_size(self, position_size: float, capital: float,
                             current_exposure: float = 0.0) -> tuple[bool, str]:
        """
        Validate if a position size is acceptable.

        Args:
            position_size: Requested position size in dollars
            capital: Current account capital
            current_exposure: Current total exposure in dollars

        Returns:
            (is_valid, reason) tuple
        """
        if position_size <= 0:
            return False, "Position size must be positive"

        if capital <= 0:
            return False, "Invalid capital amount"

        position_pct = position_size / capital

        if position_pct > self.max_position_pct:
            return False, f"Position size {position_pct:.2%} exceeds maximum {self.max_position_pct:.2%} limit"

        total_exposure = current_exposure + position_size
        total_pct = total_exposure / capital

        if total_pct > self.max_portfolio_pct:
            return False, f"Total exposure {total_pct:.2%} exceeds portfolio limit {self.max_portfolio_pct:.2%}"

        return True, "Position size approved"
