#!/usr/bin/env python3
"""
Core Risk Manager - Enterprise Risk Management Orchestrator

Integrates all risk components: calculations, limits, position sizing,
and portfolio risk monitoring for comprehensive risk control.
"""

import logging
from typing import Dict, Any, Optional, List
from datetime import datetime

from .calculations import (
    calculate_kelly_criterion,
    calculate_position_size,
    calculate_var_historical,
    calculate_sharpe_ratio,
    KellyInput
)
from .limits import CircuitBreaker, PositionSizeLimiter

logger = logging.getLogger(__name__)


class RiskManager:
    """
    Enterprise Risk Manager - Comprehensive risk control system.

    Features:
    - Kelly Criterion position sizing
    - Circuit breaker protection
    - Portfolio risk monitoring
    - Performance analytics
    - Configurable risk parameters
    """

    def __init__(self, capital: float = 10000.0, config: Optional[Dict[str, Any]] = None):
        """
        Initialize Risk Manager.

        Args:
            capital: Starting account capital
            config: Risk configuration dictionary
        """
        self.initial_capital = float(capital)
        self.current_capital = float(capital)

        # Default configuration - optimized for safety
        self.config = config or {
            "max_risk_per_trade": 0.02,      # 2% max risk per trade
            "kelly_mode": "half",            # Use half Kelly for safety
            "daily_loss_limit": 0.05,        # 5% daily loss limit
            "weekly_loss_limit": 0.15,       # 15% weekly loss limit
            "max_position_pct": 0.10,        # 10% max per position
            "max_portfolio_pct": 0.50,       # 50% max total exposure
            "risk_free_rate": 0.02,          # 2% risk-free rate for Sharpe
            "var_confidence": 0.95           # 95% VaR confidence
        }

        # Initialize components
        self.circuit_breaker = CircuitBreaker(
            daily_limit_pct=self.config["daily_loss_limit"],
            weekly_limit_pct=self.config.get("weekly_loss_limit")
        )

        self.position_limiter = PositionSizeLimiter(
            max_position_pct=self.config["max_position_pct"],
            max_portfolio_pct=self.config["max_portfolio_pct"]
        )

        # State tracking
        self.trades = []
        self.current_exposure = 0.0
        self.peak_capital = self.current_capital

        # Performance metrics
        self.total_trades = 0
        self.winning_trades = 0
        self.total_pnl = 0.0

        logger.info(f"Risk Manager initialized with ${self.initial_capital:.2f} capital")

    def get_target_size(self, win_rate: float, reward_risk_ratio: float,
                       current_exposure: float = 0.0) -> float:
        """
        Calculate optimal position size based on Kelly Criterion and risk limits.

        Args:
            win_rate: Historical win rate (0.0 to 1.0)
            reward_risk_ratio: Average reward to risk ratio
            current_exposure: Current portfolio exposure in dollars

        Returns:
            Target position size in dollars (0.0 if trading blocked)
        """
        # 1. Check Circuit Breaker
        if self.circuit_breaker.is_active:
            logger.warning("Circuit breaker active - no new positions allowed")
            return 0.0

        # 2. Validate inputs
        if not (0.0 <= win_rate <= 1.0) or reward_risk_ratio <= 0:
            logger.warning(f"Invalid inputs: win_rate={win_rate}, rr_ratio={reward_risk_ratio}")
            return 0.0

        # 3. Calculate Kelly Size
        kelly_fraction = calculate_kelly_criterion(KellyInput(win_rate=win_rate, reward_risk_ratio=reward_risk_ratio))

        if kelly_fraction <= 0:
            logger.info("Kelly fraction <= 0 - skipping trade")
            return 0.0

        # 4. Apply Position Sizing Rules
        position_size = calculate_position_size(
            self.current_capital,
            kelly_fraction,
            self.config["max_risk_per_trade"],
            self.config["kelly_mode"]
        )

        # 5. Validate with Position Limiter
        is_valid, reason = self.position_limiter.validate_position_size(
            position_size, self.current_capital, current_exposure
        )

        if not is_valid:
            logger.warning(f"Position size rejected: {reason}")
            return 0.0

        logger.info(".2f")
        return position_size

    def record_trade(self, pnl: float, symbol: str = "UNKNOWN",
                    timestamp: Optional[datetime] = None):
        """
        Record a completed trade and update risk state.

        Args:
            pnl: Profit/Loss in dollars
            symbol: Trading symbol
            timestamp: Trade timestamp (defaults to now)
        """
        if timestamp is None:
            timestamp = datetime.now()

        # Calculate percentage change
        pnl_pct = pnl / self.current_capital if self.current_capital > 0 else 0

        # Update capital
        self.current_capital += pnl
        self.total_pnl += pnl

        # Update peak capital for drawdown calculations
        self.peak_capital = max(self.peak_capital, self.current_capital)

        # Record trade
        trade_record = {
            'timestamp': timestamp,
            'symbol': symbol,
            'pnl': pnl,
            'pnl_pct': pnl_pct,
            'capital_before': self.current_capital - pnl,
            'capital_after': self.current_capital
        }
        self.trades.append(trade_record)

        # Update trade statistics
        self.total_trades += 1
        if pnl > 0:
            self.winning_trades += 1

        # Update circuit breaker
        self.circuit_breaker.update(pnl_pct)

        # Log trade result
        win_rate = self.winning_trades / self.total_trades if self.total_trades > 0 else 0
        logger.info(
            f"Trade recorded: {symbol} | PnL: ${pnl:.2f} ({pnl_pct:.2%}) | "
            f"Capital: ${self.current_capital:.2f} | Win Rate: {win_rate:.1%}"
        )

    def get_risk_metrics(self) -> Dict[str, Any]:
        """
        Calculate comprehensive risk metrics.

        Returns:
            Dictionary with risk and performance metrics
        """
        if not self.trades:
            return {
                'total_trades': 0,
                'win_rate': 0.0,
                'total_pnl': 0.0,
                'current_capital': self.current_capital,
                'peak_capital': self.peak_capital,
                'current_drawdown': 0.0,
                'sharpe_ratio': 0.0,
                'var_95': 0.0
            }

        # Calculate returns for each trade
        returns = [trade['pnl_pct'] for trade in self.trades]

        # Calculate metrics
        win_rate = self.winning_trades / self.total_trades
        current_drawdown = (self.peak_capital - self.current_capital) / self.peak_capital
        sharpe_ratio = calculate_sharpe_ratio(returns, self.config["risk_free_rate"])
        var_95 = calculate_var_historical(returns, self.config["var_confidence"])

        return {
            'total_trades': self.total_trades,
            'win_rate': win_rate,
            'total_pnl': self.total_pnl,
            'current_capital': self.current_capital,
            'peak_capital': self.peak_capital,
            'current_drawdown': current_drawdown,
            'sharpe_ratio': sharpe_ratio,
            'var_95': var_95,
            'circuit_breaker_status': self.circuit_breaker.get_status()
        }

    def reset(self):
        """Reset risk manager to initial state."""
        self.current_capital = self.initial_capital
        self.peak_capital = self.initial_capital
        self.trades = []
        self.current_exposure = 0.0
        self.total_trades = 0
        self.winning_trades = 0
        self.total_pnl = 0.0
        self.circuit_breaker.reset()
        logger.info("Risk Manager reset to initial state")

    def is_trading_allowed(self) -> bool:
        """
        Check if trading is currently allowed.

        Returns:
            True if trading is permitted, False if blocked
        """
        return not self.circuit_breaker.is_active
