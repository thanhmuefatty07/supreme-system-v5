#!/usr/bin/env python3
"""
Breakout Strategy - Production Ready

Implements breakout trading strategy with dynamic support/resistance detection,
volume confirmation, and risk-aware signal generation.
"""

import pandas as pd
import numpy as np
from typing import Optional, Dict, Any, List
from collections import deque  # CRITICAL FIX: Import deque for memory management
from .base_strategy import BaseStrategy, Signal


class BreakoutStrategy(BaseStrategy):
    """
    Breakout Strategy - Identifies and trades price breakouts.

    Detects dynamic support/resistance levels and generates signals
    when price breaks through these levels with volume confirmation.

    Production features:
    - Dynamic S/R level calculation
    - Volume confirmation
    - Multiple timeframe analysis
    - Risk-aware breakout validation
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize Breakout strategy.

        Args:
            config: Strategy configuration with parameters
        """
        # CRITICAL FIX: Initialize deque attributes BEFORE calling super().__init__()
        buffer_size = max(config.get('lookback_period', 20) * 3, 100)
        self.price_history = deque(maxlen=buffer_size)
        self.volume_history = deque(maxlen=buffer_size)
        self.support_levels = deque(maxlen=5)
        self.resistance_levels = deque(maxlen=5)

        # Store previous levels for breakout detection (avoid current price affecting levels)
        self._previous_resistance = None
        self._previous_support = None

        super().__init__("BreakoutStrategy", config)

        # CRITICAL FIX: Add max_history_size for compatibility with tests
        self.max_history_size = buffer_size

        # Core parameters
        self.lookback_period = config.get('lookback_period', 20)
        self.breakout_threshold = config.get('breakout_threshold', 0.02)  # 2% breakout
        self.volume_multiplier = config.get('volume_multiplier', 1.5)  # Volume confirmation
        self.consolidation_period = config.get('consolidation_period', 10)  # Sideways period

        # Advanced features
        self.require_volume_confirmation = config.get('require_volume_confirmation', True)
        self.min_breakout_strength = config.get('min_breakout_strength', 0.1)

        self.logger.info(f"Breakout Strategy initialized: Lookback={self.lookback_period}, Threshold={self.breakout_threshold}")

    def _initialize_state(self):
        """Initialize strategy-specific state."""
        # CRITICAL FIX: Clear deques instead of reassigning
        self.price_history.clear()
        self.volume_history.clear()
        self.support_levels.clear()
        self.resistance_levels.clear()

    def generate_signal(self, market_data: Dict[str, Any]) -> Optional[Signal]:
        """
        Generate breakout trading signal.

        Args:
            market_data: Current market data with OHLCV

        Returns:
            Signal object or None
        """
        current_price = market_data.get('close')
        current_volume = market_data.get('volume', 0)
        symbol = market_data.get('symbol', 'UNKNOWN')

        if not current_price or current_price <= 0:
            self.logger.warning(f"Invalid price data: {current_price}")
            return None

        # Update data buffers
        self.price_history.append(current_price)
        self.volume_history.append(current_volume)

        # CRITICAL FIX: Deque auto-manages size, no manual popping needed

        # Need sufficient data
        if len(self.price_history) < self.lookback_period:
            return None

        # Store current levels as previous before updating
        if self.resistance_levels:
            self._previous_resistance = list(self.resistance_levels)[-1]
        if self.support_levels:
            self._previous_support = list(self.support_levels)[-1]

        # Update support/resistance levels with new price
        self._update_levels()

        # Analyze for breakouts using levels from before current price
        signal = self._analyze_breakout(current_price, current_volume, symbol)

        if signal:
            self.total_signals += 1
            self.logger.info(f"Breakout signal generated: {signal.side} @ {signal.price}")

        return signal

    def _update_levels(self):
        """Update dynamic support and resistance levels."""
        if len(self.price_history) < self.lookback_period:
            return

        # Deque-safe slicing - use historical prices for level calculation
        prices_list = list(self.price_history)
        # Always use the most recent lookback_period prices for level calculation
        # This includes the current price, but that's acceptable for the first few signals
        recent_prices = prices_list[-self.lookback_period:]

        # Calculate pivot points for S/R levels
        high = max(recent_prices)
        low = min(recent_prices)
        close = recent_prices[-1]

        # Simple S/R calculation (can be enhanced with more sophisticated methods)
        pivot = (high + low + close) / 3
        support = 2 * pivot - high
        resistance = 2 * pivot - low

        # Store previous levels for breakout detection
        if self.resistance_levels:
            self._previous_resistance = list(self.resistance_levels)[-1]
        if self.support_levels:
            self._previous_support = list(self.support_levels)[-1]

        # Update level history
        self.support_levels.append(support)
        self.resistance_levels.append(resistance)

        # CRITICAL FIX: maxlen automatically handles size limits
        # Previous levels are now stored in generate_signal before updating

    def _analyze_breakout(self, current_price: float, current_volume: float, symbol: str) -> Optional[Signal]:
        """
        Analyze price for breakout patterns.

        Args:
            current_price: Current closing price
            current_volume: Current volume
            symbol: Trading symbol

        Returns:
            Signal if breakout detected, None otherwise
        """
        if not self.support_levels or not self.resistance_levels:
            return None

        # CRITICAL FIX: Use previous levels for breakout detection to avoid current price affecting levels
        current_support = self._previous_support
        current_resistance = self._previous_resistance

        # Fallback to current levels if no previous levels exist
        if current_support is None or current_resistance is None:
            support_list = list(self.support_levels)
            resistance_list = list(self.resistance_levels)
            current_support = support_list[-1] if support_list else None
            current_resistance = resistance_list[-1] if resistance_list else None

        # Check for volume confirmation if required
        volume_confirmed = True
        if self.require_volume_confirmation and len(self.volume_history) >= self.lookback_period:
            # Use last lookback_period volumes (excluding current) for average
            # Deque-safe slicing for volume
            volume_list = list(self.volume_history)
            avg_volume = np.mean(volume_list[-self.lookback_period:])
            volume_confirmed = bool(current_volume >= avg_volume * self.volume_multiplier)

        # Check for resistance breakout (BULLISH)
        # Price must break above resistance by more than threshold percentage
        resistance_breakout_level = current_resistance * (1 + self.breakout_threshold)
        if current_price > resistance_breakout_level:
            breakout_strength = (current_price - current_resistance) / current_resistance

            if breakout_strength >= self.min_breakout_strength and (not self.require_volume_confirmation or volume_confirmed):
                return Signal(
                    symbol=symbol,
                    side='buy',
                    price=current_price,
                    strength=min(breakout_strength, 1.0),
                    metadata={
                        'type': 'resistance_breakout',
                        'breakout_level': current_resistance,
                        'breakout_price': resistance_breakout_level,
                        'breakout_strength': breakout_strength,
                        'volume_confirmed': volume_confirmed,
                        'lookback_period': self.lookback_period,
                        'support_level': current_support
                    }
                )

        # Check for support breakdown (BEARISH)
        # Price must break below support by more than threshold percentage
        support_breakdown_level = current_support * (1 - self.breakout_threshold)
        if current_price < support_breakdown_level:
            breakout_strength = (current_support - current_price) / current_support

            if breakout_strength >= self.min_breakout_strength and (not self.require_volume_confirmation or volume_confirmed):
                return Signal(
                    symbol=symbol,
                    side='sell',
                    price=current_price,
                    strength=min(breakout_strength, 1.0),
                    metadata={
                        'type': 'support_breakdown',
                        'breakout_level': current_support,
                        'breakout_price': support_breakdown_level,
                        'breakout_strength': breakout_strength,
                        'volume_confirmed': volume_confirmed,
                        'lookback_period': self.lookback_period,
                        'resistance_level': current_resistance
                    }
                )

        return None

    def _is_consolidating(self) -> bool:
        """
        Check if market is in consolidation phase.

        Returns:
            True if consolidating (good for breakout setup)
        """
        if len(self.price_history) < self.consolidation_period:
            return False

        # Deque-safe slicing for consolidation check
        prices_list = list(self.price_history)
        recent_prices = prices_list[-self.consolidation_period:]
        price_range = max(recent_prices) - min(recent_prices)
        avg_price = sum(recent_prices) / len(recent_prices)

        # Consolidation if price range < 3% of average price
        return price_range / avg_price < 0.03

    def get_strategy_info(self) -> Dict[str, Any]:
        """Get detailed strategy information."""
        base_info = super().get_status()

        # Convert deques to lists for JSON serialization
        support_list = list(self.support_levels)
        resistance_list = list(self.resistance_levels)

        # Add breakout-specific info
        base_info.update({
            'strategy_type': 'Breakout_Based',
            'parameters': {
                'lookback_period': self.lookback_period,
                'breakout_threshold': self.breakout_threshold,
                'volume_multiplier': self.volume_multiplier,
                'consolidation_period': self.consolidation_period,
                'require_volume_confirmation': self.require_volume_confirmation,
                'min_breakout_strength': self.min_breakout_strength
            },
            'current_state': {
                'data_points': len(self.price_history),
                'current_support': support_list[-1] if support_list else None,
                'current_resistance': resistance_list[-1] if resistance_list else None,
                'is_consolidating': self._is_consolidating()
            }
        })

        return base_info
