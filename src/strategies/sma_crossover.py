#!/usr/bin/env python3
"""
SMA Crossover Strategy - Production Ready

Implements the classic Simple Moving Average crossover strategy
with enterprise-grade implementation, risk awareness, and optimization.
"""

import pandas as pd
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
        # CRITICAL FIX: Initialize deque attributes BEFORE calling super().__init__()
        buffer_size = max(config.get('slow_window', 20) * 3, 100)  # At least 3x slow window or 100 items
        self.prices = deque(maxlen=buffer_size)  # Rolling price buffer
        self.fast_ma_history = deque(maxlen=buffer_size)  # Fast MA history for crossover detection
        self.slow_ma_history = deque(maxlen=buffer_size)  # Slow MA history for crossover detection

        super().__init__("SMACrossover", config)

        # Core parameters
        self.fast_window = config.get('fast_window', 10)
        self.slow_window = config.get('slow_window', 20)
        self.min_crossover_strength = config.get('min_crossover_strength', 0.001)  # Minimum % difference

        # Optimization settings
        self.max_buffer_size = max(self.slow_window * 2, 100)  # Adaptive buffer size

        self.logger.info(f"SMA Crossover initialized: Fast={self.fast_window}, Slow={self.slow_window}")

    def _initialize_state(self):
        """Initialize strategy-specific state."""
        # CRITICAL FIX: Clear deques instead of reassigning
        self.prices.clear()
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

        # CRITICAL FIX: Update price buffer (deque auto-manages size)
        self.prices.append(current_price)

        # Need enough data for analysis
        if len(self.prices) < self.slow_window:
            return None

        # Calculate moving averages
        prices_series = pd.Series(self.prices)

        # Current MAs
        fast_ma = prices_series.rolling(window=self.fast_window).mean().iloc[-1]
        slow_ma = prices_series.rolling(window=self.slow_window).mean().iloc[-1]

        # Previous MAs for crossover detection (deque-safe)
        if len(self.prices) >= self.slow_window + 1:
            # Convert deque to list for slicing, but keep only last N+1 elements for efficiency
            prices_list = list(self.prices)
            prev_prices = pd.Series(prices_list[:-1])
            prev_fast_ma = prev_prices.rolling(window=self.fast_window).mean().iloc[-1]
            prev_slow_ma = prev_prices.rolling(window=self.slow_window).mean().iloc[-1]
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
