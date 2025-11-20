from dataclasses import dataclass
from typing import Optional, List



@dataclass(slots=True, frozen=True)
class KellyInput:
    """
    Immutable input for Kelly calculation.
    Using slots for memory efficiency.
    """

    win_rate: float
    reward_risk_ratio: float



def calculate_kelly_criterion(inp: KellyInput) -> float:
    """
    Calculates the Kelly fraction (f*) for position sizing.
    Formula: f* = (p(b + 1) - 1) / b

    Where:
        p = probability of winning (win_rate)
        b = odds received on the wager (reward_risk_ratio)

    Returns:
        float: A value between 0.0 and 1.0 (clamped).
               Returns 0.0 for invalid inputs or negative expectation.
    """

    # 1. Input Validation (Fail-safe)
    if inp.reward_risk_ratio <= 0:
        return 0.0
    if not (0.0 <= inp.win_rate <= 1.0):
        return 0.0

    # 2. Calculation
    # f = (p * (b + 1) - 1) / b
    numerator = inp.win_rate * (inp.reward_risk_ratio + 1.0) - 1.0
    f = numerator / inp.reward_risk_ratio

    # 3. Safety Clamp (Conservative)
    # Never recommend leveraging > 1.0x base capital based purely on Kelly here
    # Negative Kelly means "Don't trade" -> 0.0
    return min(max(0.0, f), 1.0)



def calculate_position_size(capital: float,
                            kelly_fraction: float,
                            max_risk_per_trade: float,
                            mode: str = 'half') -> float:
    """
    Applies Kelly fraction with safety modes to determine position size.

    Args:
        capital: Total available capital
        kelly_fraction: Raw Kelly output (0.0 to 1.0)
        max_risk_per_trade: Hard cap on risk per trade (e.g., 0.02 for 2%)
        mode: 'full', 'half', 'quarter'
    """

    if capital <= 0 or kelly_fraction <= 0:
        return 0.0

    # 1. Apply Mode Multiplier
    multiplier = 1.0
    if mode == 'half':
        multiplier = 0.5
    elif mode == 'quarter':
        multiplier = 0.25

    adjusted_fraction = kelly_fraction * multiplier

    # 2. Calculate Raw Size
    raw_size = capital * adjusted_fraction

    # 3. Apply Hard Cap (Risk Management Override)
    max_allowed_size = capital * max_risk_per_trade

    # Return the smaller of the two (Safety First)
    return min(raw_size, max_allowed_size)



def calculate_var_historical(returns: List[float], confidence: float = 0.95) -> float:
    """
    Calculate Value at Risk (VaR) using historical simulation.

    Args:
        returns: List of historical returns (as decimals, e.g., 0.01 for 1%)
        confidence: Confidence level (0.95 for 95% VaR)

    Returns:
        float: VaR value (positive number, e.g., 0.05 means 5% loss at 95% confidence)
    """
    if not returns:
        return 0.0

    # Sort returns in ascending order (worst to best)
    sorted_returns = sorted(returns)

    # Find the index for the confidence level
    # For 95% confidence, we want the 5th percentile (worst 5% of returns)
    index = int((1 - confidence) * len(sorted_returns))

    # Ensure index is within bounds
    index = max(0, min(index, len(sorted_returns) - 1))

    # VaR is the absolute value of the worst return at confidence level
    return abs(sorted_returns[index])



def calculate_sharpe_ratio(returns: List[float], risk_free_rate: float = 0.0) -> float:
    """
    Calculate Sharpe Ratio.

    Sharpe Ratio = (Expected Return - Risk-Free Rate) / Volatility

    Args:
        returns: List of historical returns (as decimals)
        risk_free_rate: Risk-free rate (as decimal, e.g., 0.02 for 2%)

    Returns:
        float: Sharpe ratio (higher is better)
    """
    if not returns or len(returns) < 2:
        return 0.0

    # Calculate average return
    avg_return = sum(returns) / len(returns)

    # Calculate volatility (standard deviation)
    if len(returns) > 1:
        variance = sum((r - avg_return) ** 2 for r in returns) / (len(returns) - 1)
        volatility = variance ** 0.5
    else:
        volatility = 0.0

    # Avoid division by zero
    if volatility == 0:
        return 0.0  # No risk = no Sharpe ratio (conservative approach)

    # Calculate Sharpe ratio
    return (avg_return - risk_free_rate) / volatility