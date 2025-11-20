#!/usr/bin/env python3
"""
Algorithmic Order Execution - Smart Order Splitting

Enterprise-grade order splitting algorithms including Iceberg, TWAP, and VWAP
for minimizing market impact and maximizing execution quality.
"""

import math
import random
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)


def split_iceberg_order(total_size: float, num_chunks: int = 0,
                        max_chunk_size: float = 0, add_noise: bool = True) -> List[float]:
    """
    Split a large order into smaller chunks using Iceberg algorithm.

    Args:
        total_size: Total order size to execute
        num_chunks: Number of chunks (calculated from max_chunk_size if 0)
        max_chunk_size: Maximum size per chunk
        add_noise: Add random noise to chunk sizes for camouflage

    Returns:
        List of chunk sizes
    """
    if total_size <= 0:
        return []

    # Determine number of chunks
    if max_chunk_size > 0:
        num_chunks = max(1, math.ceil(total_size / max_chunk_size))
    elif num_chunks <= 0:
        # Default: split into 3-5 chunks
        num_chunks = min(5, max(1, int(total_size / 10)))  # Assuming 10 is reasonable chunk size

    if num_chunks == 1:
        return [total_size]

    # Base chunk size
    base_chunk = total_size / num_chunks

    chunks = []
    remaining = total_size

    for i in range(num_chunks - 1):
        if add_noise:
            # Add ±20% random noise to avoid detection
            noise_factor = 0.8 + (random.random() * 0.4)  # 0.8 to 1.2
            chunk_size = base_chunk * noise_factor
        else:
            chunk_size = base_chunk

        # Ensure we don't exceed remaining
        chunk_size = min(chunk_size, remaining)
        chunks.append(chunk_size)
        remaining -= chunk_size

    # Last chunk gets remaining amount
    if remaining > 0:
        chunks.append(remaining)

    return chunks


def generate_twap_schedule(total_size: float, duration_minutes: int,
                          interval_minutes: int = 5) -> List[Dict[str, Any]]:
    """
    Generate TWAP (Time Weighted Average Price) execution schedule.

    Args:
        total_size: Total order size
        duration_minutes: Total execution duration
        interval_minutes: Time interval between orders

    Returns:
        List of execution slots with timing and size
    """
    if total_size <= 0 or duration_minutes <= 0:
        return []

    num_intervals = max(1, duration_minutes // interval_minutes)
    chunk_size = total_size / num_intervals

    schedule = []
    current_time = datetime.now()

    for i in range(num_intervals):
        schedule.append({
            'sequence': i + 1,
            'scheduled_time': current_time + timedelta(minutes=i * interval_minutes),
            'size': chunk_size,
            'cumulative_size': chunk_size * (i + 1)
        })

    return schedule


def generate_vwap_schedule(total_size: float, volume_profile: List[float],
                          time_intervals: int = 60) -> List[Dict[str, Any]]:
    """
    Generate VWAP (Volume Weighted Average Price) execution schedule.

    Args:
        total_size: Total order size
        volume_profile: List of volume weights for each time interval
        time_intervals: Number of time intervals

    Returns:
        List of execution slots weighted by volume profile
    """
    if not volume_profile or total_size <= 0:
        return []

    # Normalize volume profile
    total_volume = sum(volume_profile)
    if total_volume == 0:
        return []

    weights = [v / total_volume for v in volume_profile]

    schedule = []
    current_time = datetime.now()
    interval_duration = 1  # minute

    for i, weight in enumerate(weights):
        size = total_size * weight
        schedule.append({
            'sequence': i + 1,
            'scheduled_time': current_time + timedelta(minutes=i * interval_duration),
            'size': size,
            'weight': weight,
            'cumulative_size': sum(s['size'] for s in schedule)
        })

    return schedule


def optimize_chunk_timing(chunks: List[float], market_hours: Dict[str, Any],
                         avoid_high_volatility: bool = True) -> List[Dict[str, Any]]:
    """
    Optimize timing of order chunks based on market conditions.

    Args:
        chunks: List of chunk sizes
        market_hours: Market hours information
        avoid_high_volatility: Whether to avoid high volatility periods

    Returns:
        List of timed execution slots
    """
    # Simple implementation - spread evenly across market hours
    # In production, this would consider volatility, news events, etc.

    schedule = []
    current_time = datetime.now()

    for i, chunk_size in enumerate(chunks):
        schedule.append({
            'sequence': i + 1,
            'scheduled_time': current_time + timedelta(minutes=i * 5),  # 5-min intervals
            'size': chunk_size,
            'market_condition': 'normal'  # Could be 'high_vol', 'news', etc.
        })

    return schedule


def calculate_execution_quality(executed_orders: List[Dict[str, Any]],
                              benchmark_price: float) -> Dict[str, Any]:
    """
    Calculate execution quality metrics.

    Args:
        executed_orders: List of executed order details
        benchmark_price: Benchmark price for comparison

    Returns:
        Dictionary with execution quality metrics
    """
    if not executed_orders:
        return {'quality_score': 0.0, 'metrics': {}}

    total_volume = sum(order.get('size', 0) for order in executed_orders)
    total_cost = sum(order.get('price', 0) * order.get('size', 0) for order in executed_orders)

    if total_volume == 0:
        return {'quality_score': 0.0, 'metrics': {}}

    avg_execution_price = total_cost / total_volume

    # Price improvement (positive = better than benchmark)
    price_improvement = (benchmark_price - avg_execution_price) / benchmark_price

    # Slippage analysis
    slippage = abs(avg_execution_price - benchmark_price) / benchmark_price

    # Market impact estimate
    market_impact = slippage * 0.5  # Simplified

    # Overall quality score (0-100, higher is better)
    quality_score = max(0, min(100, 50 + (price_improvement * 1000)))

    metrics = {
        'avg_execution_price': avg_execution_price,
        'benchmark_price': benchmark_price,
        'price_improvement': price_improvement,
        'slippage': slippage,
        'market_impact': market_impact,
        'total_volume': total_volume,
        'total_orders': len(executed_orders)
    }

    return {
        'quality_score': quality_score,
        'metrics': metrics
    }
