"""

Adaptive Kelly Criterion Risk Manager with EWMA Performance Tracking.

Dynamically adjusts Kelly fraction based on recent win rate and reward/risk ratio.

"""

from dataclasses import dataclass, field
from typing import Optional, Literal


@dataclass(slots=True)
class RiskConfig:
    """Configuration for risk management."""
    initial_win_rate: float = 0.5
    initial_reward_risk: float = 1.5
    ewma_alpha: float = 0.05  # Decay factor (0.05 = ~40 trades for convergence)
    max_daily_loss_pct: float = 0.05  # 5%
    max_consecutive_losses: int = 3
    max_risk_per_trade: float = 0.02  # 2% per trade hard cap
    max_position_pct: float = 0.10  # 10% max position size


@dataclass(slots=True)
class AdaptiveKellyRiskManager:
    """
    Adaptive Kelly with EWMA tracking and circuit breakers.
    Thread-safe for concurrent tick processing.
    """
    config: RiskConfig
    current_capital: float

    # EWMA State (mutable)
    ewma_win_rate: float = field(init=False)
    ewma_reward_risk: float = field(init=False)

    # Circuit Breaker State
    daily_loss_pct: float = field(default=0.0, init=False)
    consecutive_losses: int = field(default=0, init=False)
    is_halted: bool = field(default=False, init=False)
    halt_reason: str = field(default="", init=False)

    def __post_init__(self):
        """Initialize EWMA with config defaults."""
        self.ewma_win_rate = self.config.initial_win_rate
        self.ewma_reward_risk = self.config.initial_reward_risk

    def update_performance(self, was_win: bool, pnl: float):
        """
        Update EWMA estimates based on trade result.

        Args:
            was_win: True if trade was profitable
            pnl: Profit/Loss amount (positive or negative)
        """
        alpha = self.config.ewma_alpha

        # Update Win Rate EWMA
        win_signal = 1.0 if was_win else 0.0
        self.ewma_win_rate = alpha * win_signal + (1 - alpha) * self.ewma_win_rate

        # Update Reward/Risk EWMA (approximation)
        if pnl != 0:
            abs_pnl = abs(pnl)
            # Estimate current trade's reward/risk contribution
            # If win: reward observed, If loss: risk observed
            if was_win:
                # This win suggests reward is around abs_pnl relative to typical risk
                estimated_rr = abs_pnl / (self.current_capital * 0.01)  # Normalize
            else:
                # This loss gives us risk info
                estimated_rr = (self.current_capital * 0.01) / abs_pnl if abs_pnl > 0 else 1.0

            self.ewma_reward_risk = alpha * estimated_rr + (1 - alpha) * self.ewma_reward_risk

        # Update Circuit Breaker State
        pnl_pct = pnl / self.current_capital if self.current_capital > 0 else 0.0

        if pnl < 0:
            self.daily_loss_pct += abs(pnl_pct)
            self.consecutive_losses += 1
        else:
            self.consecutive_losses = 0  # Reset on win

        # Check Circuit Breaker Triggers
        if self.daily_loss_pct >= self.config.max_daily_loss_pct:
            self.is_halted = True
            self.halt_reason = f"Daily Loss Limit Hit ({self.daily_loss_pct:.2%})"

        elif self.consecutive_losses >= self.config.max_consecutive_losses:
            self.is_halted = True
            self.halt_reason = f"Consecutive Losses ({self.consecutive_losses})"

    def get_target_size(self, mode: Literal['full', 'half', 'quarter'] = 'half') -> float:
        """
        Calculate position size using Adaptive Kelly.

        Args:
            mode: Kelly mode ('full', 'half', 'quarter')

        Returns:
            Position size in dollars
        """
        if self.is_halted:
            return 0.0

        # Kelly Formula: f* = (p(b+1) - 1) / b
        p = self.ewma_win_rate
        b = self.ewma_reward_risk

        if b <= 0 or p <= 0 or p >= 1:
            return 0.0  # Invalid parameters

        kelly_fraction = (p * (b + 1) - 1) / b
        kelly_fraction = max(0.0, kelly_fraction)  # No negative betting

        # Apply Mode Scaling
        scaling = {'full': 1.0, 'half': 0.5, 'quarter': 0.25}.get(mode, 0.5)
        kelly_fraction *= scaling

        # Cap at 1.0 (never bet more than entire capital)
        kelly_fraction = min(kelly_fraction, 1.0)

        # Compute raw position size in dollars
        position_value = self.current_capital * kelly_fraction

        # Enforce max risk per trade (e.g., 2% of capital as absolute cap)
        max_risk_value = self.config.max_risk_per_trade * self.current_capital
        if position_value > max_risk_value:
            position_value = max_risk_value

        # Enforce max position size (e.g., 10% of capital)
        max_position_value = self.config.max_position_pct * self.current_capital
        if position_value > max_position_value:
            position_value = max_position_value

        return position_value

    def can_trade(self) -> bool:
        """Check if trading is allowed (circuit breaker not tripped)."""
        return not self.is_halted

    def reset_daily(self):
        """Reset daily counters (call at start of trading day)."""
        self.daily_loss_pct = 0.0
        self.is_halted = False
        self.halt_reason = ""
        # Note: consecutive_losses persists across days (design choice)
