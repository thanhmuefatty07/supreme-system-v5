from dataclasses import dataclass
from typing import List, Tuple



@dataclass(slots=True)
class RiskLimitsConfig:
    """Configuration for risk limits."""

    daily_loss_limit_pct: float = 0.05  # 5% daily drawdown
    max_daily_loss_amount: float = 100.0 # $100 absolute loss
    max_consecutive_losses: int = 3      # Stop after 3 losses in a row


@dataclass(slots=True)
class PositionLimitsConfig:
    """Configuration for position size limits."""

    max_position_pct: float = 0.1   # 10% max per position
    max_portfolio_pct: float = 0.5  # 50% max total exposure


class CircuitBreaker:
    """
    Protects the account from extreme losses and psychological spirals.

    optimized with __slots__ for low memory footprint.
    """

    __slots__ = [
        'config',
        'current_daily_drawdown',
        'current_daily_loss',
        'consecutive_losses',
        'is_active',
        'trigger_reason',
        'daily_limit'  # Legacy compatibility
    ]

    def __init__(self, config: RiskLimitsConfig = None, daily_limit_pct: float = 0.05, weekly_limit_pct: float = None):
        """
        Initialize CircuitBreaker.

        Args:
            config: RiskLimitsConfig object (preferred)
            daily_limit_pct: Daily loss limit percentage (legacy support)
            weekly_limit_pct: Weekly loss limit percentage (legacy support, not used in new implementation)
        """
        if config is None:
            config = RiskLimitsConfig(
                daily_loss_limit_pct=daily_limit_pct,
                max_daily_loss_amount=1000.0,  # Default
                max_consecutive_losses=3
            )
        self.config = config
        # Legacy compatibility
        self.daily_limit = daily_limit_pct
        self.reset()

    def reset(self):
        """Resets daily counters (call this at start of trading day)."""
        self.current_daily_drawdown = 0.0
        self.current_daily_loss = 0.0
        self.consecutive_losses = 0
        self.is_active = False
        self.trigger_reason = ""

    def update(self, pnl_pct: float) -> bool:
        """
        Updates risk state based on trade result (legacy interface).
        Returns: True if trading is ALLOWED, False if BLOCKED.

        Args:
            pnl_pct: Profit/loss as percentage (e.g., -0.05 for 5% loss)
        """
        # If already tripped, stay tripped
        if self.is_active:
            return False

        # 1. Update Loss Counters
        if pnl_pct < 0:
            self.current_daily_loss += abs(pnl_pct)
            self.consecutive_losses += 1
        else:
            # Reset consecutive losses on a win (optional, or keep counting bad streak)
            # Here we reset to represent "psychological reset"
            self.consecutive_losses = 0

        # Update Drawdown
        if pnl_pct < 0:
            self.current_daily_drawdown += abs(pnl_pct)

        # 2. Check Triggers

        # A. Percentage Drawdown
        if self.current_daily_drawdown >= self.config.daily_loss_limit_pct:
            self.is_active = True
            self.trigger_reason = "daily_limit"
            return False

        # B. Consecutive Losses (Psychological Breaker)
        if self.consecutive_losses >= self.config.max_consecutive_losses:
            self.is_active = True
            self.trigger_reason = f"Consecutive Losses ({self.consecutive_losses}) Limit Hit"
            return False

        return True

    def can_trade(self) -> bool:
        return not self.is_active



class PositionSizeLimiter:
    """
    Limits position sizes to prevent excessive exposure.
    """

    __slots__ = ['max_position_pct', 'max_portfolio_pct']

    def __init__(self, max_position_pct: float = None, max_portfolio_pct: float = None, config: PositionLimitsConfig = None):
        """
        Initialize PositionSizeLimiter.

        Args:
            max_position_pct: Max percentage of capital per position (0.0-1.0) - legacy
            max_portfolio_pct: Max percentage of total portfolio exposure (0.0-1.0) - legacy
            config: PositionLimitsConfig object (preferred)
        """
        if config is None:
            # Legacy support
            config = PositionLimitsConfig(
                max_position_pct=max_position_pct or 0.1,
                max_portfolio_pct=max_portfolio_pct or 0.5
            )
        self.max_position_pct = config.max_position_pct
        self.max_portfolio_pct = config.max_portfolio_pct

    def validate_position_size(self, position_size: float, current_capital: float, current_exposure: float) -> Tuple[bool, str]:
        """
        Validates if a position size is acceptable.

        Args:
            position_size: Size of the position to validate
            current_capital: Current account capital
            current_exposure: Current total exposure

        Returns:
            Tuple of (is_valid: bool, reason: str)
        """
        if position_size <= 0:
            return False, "Position size must be positive"

        if current_capital <= 0:
            return False, "Invalid capital amount"

        # Check position size vs capital
        position_pct = position_size / current_capital
        if position_pct > self.max_position_pct:
            return False, f"Position size {position_pct:.2%} exceeds max {self.max_position_pct:.2%}"

        # Check total exposure vs capital
        total_exposure = current_exposure + position_size
        exposure_pct = total_exposure / current_capital
        if exposure_pct > self.max_portfolio_pct:
            return False, f"portfolio exposure {exposure_pct:.2%} exceeds max {self.max_portfolio_pct:.2%}"

        return True, "Position size approved"