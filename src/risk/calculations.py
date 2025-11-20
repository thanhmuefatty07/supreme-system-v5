#!/usr/bin/env python3
"""
Risk Management Calculations - Pure Functions for Risk Math

Enterprise-grade risk calculations with Kelly Criterion, position sizing,
and statistical risk metrics.
"""

import math
from typing import Optional


def calculate_kelly_criterion(win_rate: float, reward_risk_ratio: float) -> float:
    """
    Calculate Kelly fraction: f = (p(b+1) - 1) / b

    Where:
    - p = win_rate (probability of winning)
    - b = reward_risk_ratio (reward/risk ratio)

    Args:
        win_rate: Probability of winning (0.0 to 1.0)
        reward_risk_ratio: Reward to risk ratio (must be > 0)

    Returns:
        Kelly fraction (0.0 to 1.0+), or 0.0 if invalid inputs
    """
    if reward_risk_ratio <= 0 or win_rate < 0 or win_rate > 1:
        return 0.0

    # Kelly formula: f = (p(b+1) - 1) / b
    kelly_fraction = (win_rate * (reward_risk_ratio + 1) - 1) / reward_risk_ratio

    # Kelly can be negative (avoid these trades) or very large
    return max(0.0, kelly_fraction)


def apply_position_sizing(capital: float, kelly_fraction: float, max_risk_pct: float, mode: str = 'half') -> float:
    """
    Apply sizing rules with Kelly fraction and hard caps.

    Args:
        capital: Current account capital
        kelly_fraction: Raw Kelly fraction from calculate_kelly_criterion
        max_risk_pct: Maximum risk per trade as percentage of capital
        mode: Sizing mode - 'full', 'half', 'quarter', 'conservative'

    Returns:
        Position size in dollars (not percentage)
    """
    if capital <= 0 or kelly_fraction < 0:
        return 0.0

    # Apply mode multiplier
    multiplier = {
        'full': 1.0,
        'half': 0.5,
        'quarter': 0.25,
        'conservative': 0.1
    }.get(mode, 0.5)  # Default to half Kelly

    adjusted_kelly = kelly_fraction * multiplier

    # Calculate risk amount
    risk_amount = capital * adjusted_kelly

    # Apply hard cap
    max_allowed_risk = capital * max_risk_pct

    return min(risk_amount, max_allowed_risk)


def calculate_var_historical(returns: list, confidence_level: float = 0.95) -> float:
    """
    Calculate Value at Risk using historical simulation.

    Args:
        returns: List of historical returns (as decimals)
        confidence_level: Confidence level (default 95%)

    Returns:
        VaR as positive percentage (e.g., 0.05 for 5% VaR)
    """
    if not returns or len(returns) < 2:
        return 0.0

    # Sort returns in ascending order (worst to best)
    sorted_returns = sorted(returns)

    # Find the return at the confidence level
    index = int((1 - confidence_level) * len(sorted_returns))

    # VaR is the negative of the worst case (since returns are negative for losses)
    var = -sorted_returns[index]

    return max(0.0, var)


def calculate_sharpe_ratio(returns: list, risk_free_rate: float = 0.02) -> float:
    """
    Calculate Sharpe ratio: (mean return - risk_free) / std_dev

    Args:
        returns: List of returns
        risk_free_rate: Risk-free rate (annualized)

    Returns:
        Sharpe ratio, or 0.0 if calculation impossible
    """
    if not returns or len(returns) < 2:
        return 0.0

    # Convert to numpy for calculations (import here to avoid dependency issues)
    try:
        import numpy as np

        returns_array = np.array(returns)
        mean_return = np.mean(returns_array)
        std_dev = np.std(returns_array, ddof=1)  # Sample standard deviation

        if std_dev == 0:
            return 0.0

        sharpe = (mean_return - risk_free_rate) / std_dev
        return sharpe

    except ImportError:
        # Fallback without numpy
        mean_return = sum(returns) / len(returns)
        variance = sum((r - mean_return) ** 2 for r in returns) / (len(returns) - 1)
        std_dev = math.sqrt(variance)

        if std_dev == 0:
            return 0.0

        return (mean_return - risk_free_rate) / std_dev
