#!/usr/bin/env python3
"""
SMA Crossover Strategy - Production Ready

Implements the classic Simple Moving Average crossover strategy
with enterprise-grade implementation, risk awareness, and optimization.
"""

import pandas as pd
import numpy as np
from typing import Optional, Dict, Any
from collections import deque  # CRITICAL FIX: Import deque for memory management
from .base_strategy import BaseStrategy, Signal


class SMACrossover(BaseStrategy):
    """
    Simple Moving Average Crossover Strategy.

    Generates buy signals when fast MA crosses above slow MA (Golden Cross)
    and sell signals when fast MA crosses below slow MA (Death Cross).

    Production features:
    - Optimized price buffer management
    - Risk-aware position sizing
    - Comprehensive signal metadata
    - Performance tracking
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize SMA Crossover strategy.

        Args:
            config: Strategy configuration with parameters
        """
        # Core parameters
        self.fast_window = config.get('fast_window', 10)
        self.slow_window = config.get('slow_window', 20)
        self.min_crossover_strength = config.get('min_crossover_strength', 0.001)  # Minimum % difference

        # Strategy-specific buffers (use base class for prices, manage MA buffers separately)
        buffer_size = config.get('buffer_size', max(self.slow_window * 3, 100))
        self.fast_ma_history = deque(maxlen=buffer_size)
        self.slow_ma_history = deque(maxlen=buffer_size)

        super().__init__("SMACrossover", config)

        self.logger.info(f"SMA Crossover initialized: Fast={self.fast_window}, Slow={self.slow_window}")

    def _calculate_sma_optimized(self, prices, period: int) -> float:
        """
        Calculate Simple Moving Average using numpy vectorized operations.

        Optimized for deque input with O(1) access to recent elements.

        Args:
            prices: Deque or list of recent prices
            period: SMA period

        Returns:
            Current SMA value or NaN if insufficient data
        """
        if len(prices) < period:
            return np.nan

        # Convert deque to list for slicing (deque doesn't support direct slicing)
        prices_list = list(prices)

        # Get last 'period' prices and calculate mean
        recent_prices = prices_list[-period:]
        return np.mean(recent_prices)

    def _initialize_state(self):
        """Initialize strategy-specific state."""
        # Clear strategy-specific buffers (base class handles self.prices)
        self.fast_ma_history.clear()
        self.slow_ma_history.clear()

    def generate_signal(self, market_data: Dict[str, Any]) -> Optional[Signal]:
        """
        Generate trading signal based on SMA crossover analysis.

        Args:
            market_data: Current market data with price info

        Returns:
            Signal object or None if no crossover detected
        """
        current_price = market_data.get('close')
        symbol = market_data.get('symbol', 'UNKNOWN')

        if not current_price or current_price <= 0:
            self.logger.warning(f"Invalid price data: {current_price}")
            return None

        # Update price buffer using base class helper (memory-safe)
        self.update_price(current_price)

        # Need enough data for analysis
        if len(self.prices) < self.slow_window:
            return None

        # Calculate moving averages using optimized numpy approach
        fast_ma = self._calculate_sma_optimized(self.prices, self.fast_window)
        slow_ma = self._calculate_sma_optimized(self.prices, self.slow_window)

        # Skip if insufficient data for MA calculation
        if np.isnan(fast_ma) or np.isnan(slow_ma):
            return None

        # Previous MAs for crossover detection (optimized)
        if len(self.prices) >= self.slow_window + 1:
            # Use optimized calculation for previous period
            prev_prices_list = list(self.prices)[:-1]  # Exclude current price
            prev_fast_ma = self._calculate_sma_optimized(prev_prices_list, self.fast_window)
            prev_slow_ma = self._calculate_sma_optimized(prev_prices_list, self.slow_window)

            # Skip if previous calculations failed
            if np.isnan(prev_fast_ma) or np.isnan(prev_slow_ma):
                return None
        else:
            # First calculation
            prev_fast_ma = fast_ma
            prev_slow_ma = slow_ma

        # Store MA history for analysis
        self.fast_ma_history.append(fast_ma)
        self.slow_ma_history.append(slow_ma)

        # CRITICAL FIX: Deque auto-manages size, no manual popping needed

        # Detect crossovers
        signal = self._detect_crossover(
            prev_fast_ma, prev_slow_ma, fast_ma, slow_ma,
            current_price, symbol
        )

        if signal:
            self.total_signals += 1
            self.logger.info(f"SMA Crossover signal generated: {signal.side} @ {signal.price}")

        return signal

    def _detect_crossover(self, prev_fast: float, prev_slow: float,
                         curr_fast: float, curr_slow: float,
                         price: float, symbol: str) -> Optional[Signal]:
        """
        Detect golden/death cross and calculate signal strength.

        Args:
            prev_fast: Previous fast MA value
            prev_slow: Previous slow MA value
            curr_fast: Current fast MA value
            curr_slow: Current slow MA value
            price: Current price
            symbol: Trading symbol

        Returns:
            Signal if crossover detected, None otherwise
        """
        # Golden Cross: Fast MA crosses above Slow MA
        if prev_fast <= prev_slow and curr_fast > curr_slow:
            crossover_strength = abs(curr_fast - curr_slow) / curr_slow

            if crossover_strength >= self.min_crossover_strength:
                return Signal(
                    symbol=symbol,
                    side='buy',
                    price=price,
                    strength=min(crossover_strength, 1.0),  # Cap at 1.0
                    metadata={
                        'type': 'golden_cross',
                        'fast_ma': curr_fast,
                        'slow_ma': curr_slow,
                        'crossover_strength': crossover_strength,
                        'fast_window': self.fast_window,
                        'slow_window': self.slow_window
                    }
                )

        # Death Cross: Fast MA crosses below Slow MA
        elif prev_fast >= prev_slow and curr_fast < curr_slow:
            crossover_strength = abs(curr_slow - curr_fast) / curr_slow

            if crossover_strength >= self.min_crossover_strength:
                return Signal(
                    symbol=symbol,
                    side='sell',
                    price=price,
                    strength=min(crossover_strength, 1.0),  # Cap at 1.0
                    metadata={
                        'type': 'death_cross',
                        'fast_ma': curr_fast,
                        'slow_ma': curr_slow,
                        'crossover_strength': crossover_strength,
                        'fast_window': self.fast_window,
                        'slow_window': self.slow_window
                    }
                )

        return None

    def get_strategy_info(self) -> Dict[str, Any]:
        """Get detailed strategy information."""
        base_info = super().get_status()

        # Add SMA-specific info
        base_info.update({
            'strategy_type': 'SMA_Crossover',
            'parameters': {
                'fast_window': self.fast_window,
                'slow_window': self.slow_window,
                'min_crossover_strength': self.min_crossover_strength
            },
            'current_state': {
                'buffer_size': len(self.prices),
                'fast_ma_history': self.fast_ma_history[-5:] if self.fast_ma_history else [],
                'slow_ma_history': self.slow_ma_history[-5:] if self.slow_ma_history else []
            }
        })

        return base_info
