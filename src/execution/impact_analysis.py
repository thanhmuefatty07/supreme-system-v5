#!/usr/bin/env python3
"""
Execution Impact Analysis - Market Microstructure Analysis

Enterprise-grade analysis of market impact, slippage calculation,
and liquidity assessment for optimal trade execution.
"""

import logging
from typing import List, Dict, Any, Optional, Tuple

logger = logging.getLogger(__name__)


def calculate_slippage(order_size: float, order_book_depth: List[Dict[str, float]],
                      initial_price: Optional[float] = None) -> float:
    """
    Calculate expected slippage for an order based on order book depth.

    Args:
        order_size: Size of the order to execute
        order_book_depth: List of dicts with 'price' and 'amount' keys
        initial_price: Reference price (defaults to first level)

    Returns:
        Slippage as percentage (positive for adverse impact)
    """
    if not order_book_depth or order_size <= 0:
        return 0.0

    if initial_price is None:
        initial_price = order_book_depth[0]['price']

    if initial_price <= 0:
        return 0.0

    remaining = order_size
    total_cost = 0.0
    total_volume = 0.0

    # Simulate market order execution through order book
    for level in order_book_depth:
        if remaining <= 0:
            break

        price = level['price']
        available = level['amount']

        take = min(remaining, available)
        total_cost += take * price
        total_volume += take
        remaining -= take

    if total_volume == 0:
        return 0.0

    avg_execution_price = total_cost / total_volume
    slippage_pct = abs(avg_execution_price - initial_price) / initial_price

    logger.debug(".2f")
    return slippage_pct


def is_liquidity_sufficient(order_size: float, order_book_depth: List[Dict[str, float]],
                           max_slippage: float, initial_price: Optional[float] = None) -> Tuple[bool, str]:
    """
    Assess if market has sufficient liquidity for the order within slippage tolerance.

    Args:
        order_size: Size of the order to execute
        order_book_depth: Order book depth data
        max_slippage: Maximum acceptable slippage percentage
        initial_price: Reference price

    Returns:
        (is_sufficient, reason) tuple
    """
    if not order_book_depth:
        return False, "No order book data available"

    if order_size <= 0:
        return True, "Order size must be positive"

    # Calculate expected slippage first
    slippage = calculate_slippage(order_size, order_book_depth, initial_price)

    # Check if order book has enough total liquidity
    total_liquidity = sum(level['amount'] for level in order_book_depth)
    if total_liquidity < order_size:
        return False, f"Order book has only {total_liquidity:.2f} liquidity, need {order_size:.2f}"

    if slippage > max_slippage:
        return False, f"Slippage {slippage:.2f} exceeds maximum {max_slippage:.2f} limit"

    return True, "Liquidity sufficient"


def estimate_market_impact(order_size: float, avg_daily_volume: float,
                          volatility: float = 0.02) -> float:
    """
    Estimate market impact using simplified square-root model.

    Args:
        order_size: Size of the order
        avg_daily_volume: Average daily volume for the asset
        volatility: Asset volatility (default 2%)

    Returns:
        Estimated market impact as percentage
    """
    if avg_daily_volume <= 0:
        return float('inf')  # Infinite impact if no volume

    participation_rate = order_size / avg_daily_volume

    # Square-root impact model (simplified)
    impact = volatility * (participation_rate ** 0.5)

    return min(impact, 1.0)  # Cap at 100% impact


def get_optimal_execution_time(order_size: float, avg_daily_volume: float,
                              target_participation: float = 0.1) -> int:
    """
    Calculate optimal execution time to minimize market impact.

    Args:
        order_size: Total order size
        avg_daily_volume: Average daily volume
        target_participation: Target percentage of daily volume per minute

    Returns:
        Estimated execution time in minutes
    """
    if avg_daily_volume <= 0:
        return 1  # Default to 1 minute

    max_minute_volume = avg_daily_volume * target_participation / (24 * 60)
    minutes_needed = max(1, int(order_size / max_minute_volume))

    return min(minutes_needed, 24 * 60)  # Cap at 1 day
