# python/supreme_system_v5/risk.py
"""
Risk Manager - Enterprise-grade risk management system
ULTRA SFL implementation with circuit breakers and adaptive limits
"""

import time
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

from loguru import logger
from prometheus_client import Counter, Gauge

# Metrics
RISK_CHECKS = Counter("risk_checks_total", "Total risk evaluations", ["result"])
RISK_VIOLATIONS = Counter(
    "risk_violations_total", "Risk violations by type", ["violation_type"]
)
CURRENT_DRAWDOWN = Gauge("current_drawdown_percent", "Current portfolio drawdown")
DAILY_PNL = Gauge("daily_pnl_usd", "Daily P&L in USD")
ACTIVE_POSITIONS = Gauge("active_positions_count", "Number of active positions")


class RiskViolation(Enum):
    """Types of risk violations"""

    MAX_DRAWDOWN = "max_drawdown"
    DAILY_LOSS_LIMIT = "daily_loss_limit"
    POSITION_SIZE = "position_size"
    LEVERAGE_LIMIT = "leverage_limit"
    MAX_POSITIONS = "max_positions"
    CIRCUIT_BREAKER = "circuit_breaker"
    CONCENTRATION = "concentration"


@dataclass
class RiskLimits:
    """Risk management limits and thresholds"""

    # Portfolio-level limits
    max_drawdown_percent: float = 12.0  # 12% max drawdown
    max_daily_loss_usd: float = 100.0  # $100 max daily loss
    max_daily_loss_percent: float = 5.0  # 5% max daily loss percentage

    # Position-level limits
    max_position_size_usd: float = 1000.0  # $1000 max position size
    max_position_size_percent: float = 2.0  # 2% of portfolio max position
    max_leverage: float = 2.0  # 2x max leverage

    # Portfolio composition
    max_active_positions: int = 5  # Max 5 active positions
    max_single_symbol_concentration: float = 25.0  # 25% max per symbol

    # Circuit breaker settings
    circuit_breaker_threshold: int = 3  # Violations before circuit breaker
    circuit_breaker_timeout_minutes: int = 30  # 30 min circuit breaker
    cool_off_period_minutes: int = 15  # 15 min cool-off between trades

    # Adaptive risk settings
    enable_adaptive_limits: bool = True
    volatility_adjustment: bool = True


@dataclass
class PortfolioState:
    """Current portfolio state for risk calculations"""

    total_value: float = 10000.0  # Starting portfolio value
    cash_balance: float = 10000.0  # Available cash
    peak_value: float = 10000.0  # Highest portfolio value (for drawdown calc)

    # Daily tracking (resets at midnight UTC)
    daily_start_value: float = 10000.0
    daily_pnl: float = 0.0
    daily_trade_count: int = 0

    # Position tracking
    positions: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    position_values: Dict[str, float] = field(default_factory=dict)

    # Risk state
    circuit_breaker_active: bool = False
    circuit_breaker_until: float = 0.0
    last_trade_time: float = 0.0
    violation_count: int = 0

    # Performance tracking
    total_trades: int = 0
    winning_trades: int = 0
    total_pnl: float = 0.0


@dataclass
class RiskAssessment:
    """Risk assessment result"""

    approved: bool
    violations: List[RiskViolation] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    adjusted_position_size: Optional[float] = None
    reasoning: str = ""

    def add_violation(self, violation: RiskViolation, message: str):
        """Add a risk violation"""
        self.violations.append(violation)
        self.approved = False
        self.warnings.append(f"{violation.value}: {message}")
        RISK_VIOLATIONS.labels(violation_type=violation.value).inc()


class RiskManager:
    """
    Enterprise-grade risk management system
    ULTRA SFL implementation with circuit breakers and adaptive limits
    """

    def __init__(
        self,
        limits: Optional[RiskLimits] = None,
        portfolio_state: Optional[PortfolioState] = None,
    ):
        self.limits = limits or RiskLimits()
        self.portfolio = portfolio_state or PortfolioState()

        # Daily reset tracking
        self.last_daily_reset = self._get_day_start()

        logger.info("🛡️ Risk Manager initialized with enterprise-grade controls")
        logger.info(f"📊 Max drawdown: {self.limits.max_drawdown_percent}%")
        logger.info(f"📊 Max daily loss: ${self.limits.max_daily_loss_usd}")
        logger.info(f"📊 Max position size: ${self.limits.max_position_size_usd}")

    def _get_day_start(self) -> float:
        """Get start of current UTC day"""
        now = datetime.utcnow()
        day_start = now.replace(hour=0, minute=0, second=0, microsecond=0)
        return day_start.timestamp()

    def _check_daily_reset(self):
        """Reset daily counters if new day"""
        current_day_start = self._get_day_start()
        if current_day_start > self.last_daily_reset:
            # Reset daily counters
            self.portfolio.daily_start_value = self.portfolio.total_value
            self.portfolio.daily_pnl = 0.0
            self.portfolio.daily_trade_count = 0
            self.last_daily_reset = current_day_start

            logger.info("🌅 Daily risk counters reset")

    def _calculate_drawdown(self) -> float:
        """Calculate current portfolio drawdown percentage"""
        if self.portfolio.peak_value <= 0:
            return 0.0

        current_value = self.portfolio.total_value
        drawdown = (
            (self.portfolio.peak_value - current_value)
            / self.portfolio.peak_value
            * 100
        )
        return drawdown

    def _calculate_daily_loss(self) -> float:
        """Calculate daily loss amount"""
        daily_loss = self.portfolio.daily_start_value - self.portfolio.total_value
        return daily_loss

    def _calculate_daily_loss_percent(self) -> float:
        """Calculate daily loss percentage"""
        if self.portfolio.daily_start_value <= 0:
            return 0.0

        loss_percent = (
            (self.portfolio.daily_start_value - self.portfolio.total_value)
            / self.portfolio.daily_start_value
            * 100
        )
        return loss_percent

    def _check_portfolio_limits(self, trade_value: float) -> RiskAssessment:
        """Check portfolio-level risk limits"""
        assessment = RiskAssessment(approved=True)

        # Update daily counters
        self._check_daily_reset()

        # Check circuit breaker
        if self.portfolio.circuit_breaker_active:
            if time.time() < self.portfolio.circuit_breaker_until:
                assessment.add_violation(
                    RiskViolation.CIRCUIT_BREAKER,
                    f"Circuit breaker active until {datetime.fromtimestamp(self.portfolio.circuit_breaker_until)}",
                )
                return assessment
            else:
                # Reset circuit breaker
                self.portfolio.circuit_breaker_active = False
                self.portfolio.violation_count = 0
                logger.info("🔄 Circuit breaker reset")

        # Check cool-off period
        if self.limits.cool_off_period_minutes > 0:
            cool_off_until = self.portfolio.last_trade_time + (
                self.limits.cool_off_period_minutes * 60
            )
            if time.time() < cool_off_until:
                remaining_minutes = (cool_off_until - time.time()) / 60
                assessment.warnings.append(
                    f"Cool-off period active: {remaining_minutes:.1f} minutes remaining"
                )
                # Not a violation, just a warning

        # Check drawdown limit
        current_drawdown = self._calculate_drawdown()
        CURRENT_DRAWDOWN.set(current_drawdown)

        if current_drawdown >= self.limits.max_drawdown_percent:
            assessment.add_violation(RiskViolation.MAX_DRAWDOWN, ".2f")

        # Check daily loss limits
        daily_loss = self._calculate_daily_loss()
        daily_loss_percent = self._calculate_daily_loss_percent()
        DAILY_PNL.set(-daily_loss)  # Negative for losses

        if daily_loss >= self.limits.max_daily_loss_usd:
            assessment.add_violation(RiskViolation.DAILY_LOSS_LIMIT, ".2f")

        if daily_loss_percent >= self.limits.max_daily_loss_percent:
            assessment.add_violation(RiskViolation.DAILY_LOSS_LIMIT, ".2f")

        return assessment

    def _check_position_limits(
        self, symbol: str, position_value: float, leverage: float = 1.0
    ) -> RiskAssessment:
        """Check position-level risk limits"""
        assessment = RiskAssessment(approved=True)

        # Check position size limits
        if position_value > self.limits.max_position_size_usd:
            assessment.add_violation(RiskViolation.POSITION_SIZE, ".2f")
            # Suggest adjusted size
            assessment.adjusted_position_size = self.limits.max_position_size_usd

        # Check position size as percentage of portfolio
        portfolio_percent = (position_value / self.portfolio.total_value) * 100
        if portfolio_percent > self.limits.max_position_size_percent:
            assessment.add_violation(RiskViolation.POSITION_SIZE, ".2f")
            # Suggest adjusted size
            max_portfolio_size = self.portfolio.total_value * (
                self.limits.max_position_size_percent / 100
            )
            if (
                assessment.adjusted_position_size is None
                or max_portfolio_size < assessment.adjusted_position_size
            ):
                assessment.adjusted_position_size = max_portfolio_size

        # Check leverage limit
        if leverage > self.limits.max_leverage:
            assessment.add_violation(RiskViolation.LEVERAGE_LIMIT, ".1f")

        # Check maximum active positions
        if len(self.portfolio.positions) >= self.limits.max_active_positions:
            assessment.add_violation(
                RiskViolation.MAX_POSITIONS,
                f"Already have {len(self.portfolio.positions)} active positions, max allowed: {self.limits.max_active_positions}",
            )

        # Check concentration limit
        current_symbol_value = self.portfolio.position_values.get(symbol, 0)
        total_symbol_value = current_symbol_value + position_value
        symbol_concentration = (total_symbol_value / self.portfolio.total_value) * 100

        if symbol_concentration > self.limits.max_single_symbol_concentration:
            assessment.add_violation(RiskViolation.CONCENTRATION, ".2f")

        return assessment

    def evaluate_trade(
        self, symbol: str, position_value: float, leverage: float = 1.0
    ) -> RiskAssessment:
        """
        Comprehensive risk assessment for a trade
        Returns RiskAssessment with approval status and any violations
        """
        RISK_CHECKS.labels(result="total").inc()

        # Combine portfolio and position assessments
        portfolio_assessment = self._check_portfolio_limits(position_value)
        position_assessment = self._check_position_limits(
            symbol, position_value, leverage
        )

        # Merge assessments
        final_assessment = RiskAssessment(
            approved=portfolio_assessment.approved and position_assessment.approved,
            violations=portfolio_assessment.violations + position_assessment.violations,
            warnings=portfolio_assessment.warnings + position_assessment.warnings,
        )

        # Use the most restrictive position size adjustment
        if (
            portfolio_assessment.adjusted_position_size is not None
            or position_assessment.adjusted_position_size is not None
        ):
            final_assessment.adjusted_position_size = min(
                portfolio_assessment.adjusted_position_size or position_value,
                position_assessment.adjusted_position_size or position_value,
            )

        # Build reasoning
        if final_assessment.approved:
            final_assessment.reasoning = "✅ Trade approved - all risk checks passed"
            RISK_CHECKS.labels(result="approved").inc()
        else:
            final_assessment.reasoning = (
                f"❌ Trade rejected - {len(final_assessment.violations)} risk violations"
            )
            RISK_CHECKS.labels(result="rejected").inc()

            # Check if we should trigger circuit breaker
            if len(final_assessment.violations) > 0:
                self.portfolio.violation_count += 1

                if (
                    self.portfolio.violation_count
                    >= self.limits.circuit_breaker_threshold
                ):
                    self.portfolio.circuit_breaker_active = True
                    self.portfolio.circuit_breaker_until = time.time() + (
                        self.limits.circuit_breaker_timeout_minutes * 60
                    )

                    logger.error(
                        f"🚨 Circuit breaker activated! {self.portfolio.violation_count} violations in sequence"
                    )
                    logger.error(
                        f"⏰ Circuit breaker active until {datetime.fromtimestamp(self.portfolio.circuit_breaker_until)}"
                    )

        # Log assessment
        logger.info(f"🔍 Risk assessment for {symbol}: {final_assessment.reasoning}")

        if final_assessment.warnings:
            for warning in final_assessment.warnings:
                logger.warning(f"⚠️ {warning}")

        return final_assessment

    def record_trade(
        self, symbol: str, side: str, quantity: float, price: float, pnl: float = 0.0
    ):
        """
        Record a completed trade for risk tracking
        """
        # Update portfolio value
        trade_value = quantity * price
        if side.upper() == "BUY":
            self.portfolio.cash_balance -= trade_value
            self.portfolio.positions[symbol] = {
                "side": side,
                "quantity": quantity,
                "entry_price": price,
                "timestamp": time.time(),
            }
            self.portfolio.position_values[symbol] = trade_value
        elif side.upper() == "SELL":
            self.portfolio.cash_balance += trade_value
            if symbol in self.portfolio.positions:
                del self.portfolio.positions[symbol]
                del self.portfolio.position_values[symbol]

        # Update portfolio value and peak
        self.portfolio.total_value = self.portfolio.cash_balance + sum(
            self.portfolio.position_values.values()
        )
        self.portfolio.peak_value = max(
            self.portfolio.peak_value, self.portfolio.total_value
        )

        # Update daily tracking
        self.portfolio.daily_pnl += pnl
        self.portfolio.daily_trade_count += 1
        self.portfolio.last_trade_time = time.time()

        # Update performance tracking
        self.portfolio.total_trades += 1
        self.portfolio.total_pnl += pnl
        if pnl > 0:
            self.portfolio.winning_trades += 1

        # Update metrics
        ACTIVE_POSITIONS.set(len(self.portfolio.positions))
        CURRENT_DRAWDOWN.set(self._calculate_drawdown())
        DAILY_PNL.set(self.portfolio.daily_pnl)

        logger.info(f"💰 Trade recorded: {symbol} {side} {quantity} @ ${price:.2f}")
        logger.info(
            f"📊 Portfolio value: ${self.portfolio.total_value:.2f} (PnL: ${self.portfolio.total_pnl:.2f})"
        )

    def get_risk_status(self) -> Dict[str, Any]:
        """
        Get comprehensive risk status for monitoring
        """
        self._check_daily_reset()

        return {
            "portfolio_value": self.portfolio.total_value,
            "cash_balance": self.portfolio.cash_balance,
            "peak_value": self.portfolio.peak_value,
            "current_drawdown_percent": self._calculate_drawdown(),
            "daily_pnl": self.portfolio.daily_pnl,
            "daily_loss_percent": self._calculate_daily_loss_percent(),
            "daily_trade_count": self.portfolio.daily_trade_count,
            "active_positions": len(self.portfolio.positions),
            "circuit_breaker_active": self.portfolio.circuit_breaker_active,
            "violation_count": self.portfolio.violation_count,
            "total_trades": self.portfolio.total_trades,
            "win_rate": self.portfolio.winning_trades
            / max(1, self.portfolio.total_trades),
            "total_pnl": self.portfolio.total_pnl,
            "limits": {
                "max_drawdown_percent": self.limits.max_drawdown_percent,
                "max_daily_loss_usd": self.limits.max_daily_loss_usd,
                "max_daily_loss_percent": self.limits.max_daily_loss_percent,
                "max_position_size_usd": self.limits.max_position_size_usd,
                "max_position_size_percent": self.limits.max_position_size_percent,
                "max_leverage": self.limits.max_leverage,
                "max_active_positions": self.limits.max_active_positions,
                "max_single_symbol_concentration": self.limits.max_single_symbol_concentration,
            },
        }

    def reset_portfolio(self, new_value: float = 10000.0):
        """
        Reset portfolio to new value (for testing/backtesting)
        """
        self.portfolio = PortfolioState(
            total_value=new_value,
            cash_balance=new_value,
            peak_value=new_value,
            daily_start_value=new_value,
        )
        self.last_daily_reset = self._get_day_start()

        logger.info(f"🔄 Portfolio reset to ${new_value:,.2f}")

    def get_adaptive_limits(self) -> RiskLimits:
        """
        Get adaptive risk limits based on current portfolio performance
        """
        if not self.limits.enable_adaptive_limits:
            return self.limits

        # Base limits
        adaptive_limits = RiskLimits()
        adaptive_limits.__dict__.update(self.limits.__dict__)

        # Adjust based on recent performance
        win_rate = self.portfolio.winning_trades / max(1, self.portfolio.total_trades)
        current_drawdown = self._calculate_drawdown()

        # Reduce limits if drawdown is high
        if current_drawdown > 5.0:  # Over 5% drawdown
            drawdown_factor = max(
                0.5, 1.0 - (current_drawdown / self.limits.max_drawdown_percent)
            )
            adaptive_limits.max_position_size_percent *= drawdown_factor
            adaptive_limits.max_daily_loss_percent *= drawdown_factor
            logger.info(
                f"🔴 High drawdown detected: {current_drawdown:.2f}%, reducing limits by {drawdown_factor:.2f}x"
            )
        # Increase limits if win rate is high and drawdown is low
        elif (
            win_rate > 0.6
            and current_drawdown < 2.0
            and self.portfolio.total_trades > 10
        ):
            confidence_factor = min(1.2, 1.0 + (win_rate - 0.5))
            adaptive_limits.max_position_size_percent *= confidence_factor
            logger.info(
                f"🟢 Strong performance detected: {win_rate:.1%} win rate, increasing limits by {confidence_factor:.2f}x"
            )
        return adaptive_limits
