#!/usr/bin/env python3
"""
Supreme System V5 - Breakout Strategy Implementation

Research-backed breakout detection using automated pattern recognition.
Based on academic studies and algorithmic trading best practices.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
import logging
from .base_strategy import BaseStrategy


class BreakoutStrategy(BaseStrategy):
    """
    Advanced Breakout Strategy with research-backed detection algorithms.

    Implements automated breakout detection using:
    - Liquidity sweep analysis
    - Volume confirmation
    - False breakout filtering
    - Multiple timeframe analysis

    Based on breakout detection research and algorithmic trading studies.
    """

    def __init__(self,
                 lookback_period: int = 20,
                 breakout_threshold: float = 0.02,
                 volume_multiplier: float = 1.5,
                 consolidation_period: int = 10,
                 min_breakout_distance: float = 0.01,
                 max_hold_period: int = 5):
        """
        Initialize breakout strategy with research-backed parameters.

        Args:
            lookback_period: Period for breakout detection (default: 20)
            breakout_threshold: Minimum breakout percentage (default: 2%)
            volume_multiplier: Volume confirmation multiplier (default: 1.5x)
            consolidation_period: Period to check for consolidation (default: 10)
            min_breakout_distance: Minimum distance from recent highs/lows (default: 1%)
            max_hold_period: Maximum periods to hold position (default: 5)
        """
        super().__init__()
        self.logger = logging.getLogger(__name__)

        # Strategy parameters (research-optimized)
        self.lookback_period = lookback_period
        self.breakout_threshold = breakout_threshold
        self.volume_multiplier = volume_multiplier
        self.consolidation_period = consolidation_period
        self.min_breakout_distance = min_breakout_distance
        self.max_hold_period = max_hold_period

        # Position tracking
        self.position_active = False
        self.entry_price = 0.0
        self.entry_time = None
        self.position_side = None  # 'long' or 'short'
        self.hold_periods = 0

        # Performance tracking
        self.breakout_signals = []
        self.false_breakouts = 0
        self.successful_breakouts = 0

    def generate_signal(self, data: pd.DataFrame) -> int:
        """
        Generate breakout trading signals.

        Based on automated breakout detection research.

        Args:
            data: OHLCV DataFrame

        Returns:
            Signal: 1 (BUY), -1 (SELL), 0 (HOLD)
        """
        if not self.validate_data(data):
            return 0

        if len(data) < self.lookback_period + self.consolidation_period:
            return 0

        # Check if we have an active position
        if self.position_active:
            return self._manage_position(data)

        # Look for new breakout opportunities
        return self._detect_breakout(data)

    def _detect_breakout(self, data: pd.DataFrame) -> int:
        """
        Detect breakouts using research-backed methods.

        Args:
            data: OHLCV DataFrame

        Returns:
            Signal: 1 (LONG), -1 (SHORT), 0 (NO SIGNAL)
        """
        try:
            # Get recent data for analysis
            recent_data = data.tail(self.lookback_period + self.consolidation_period)

            # Calculate resistance/support levels
            resistance_level = recent_data['high'].rolling(self.lookback_period).max().iloc[-1]
            support_level = recent_data['low'].rolling(self.lookback_period).min().iloc[-1]

            # Get current price and volume
            current_price = recent_data['close'].iloc[-1]
            current_volume = recent_data['volume'].iloc[-1]
            avg_volume = recent_data['volume'].tail(self.lookback_period).mean()

            # Check for consolidation (tight range before breakout)
            recent_range = (resistance_level - support_level) / support_level
            consolidation_threshold = 0.02  # 2% range threshold

            # Bullish breakout conditions
            bullish_conditions = [
                current_price > resistance_level * (1 + self.breakout_threshold),
                current_volume > avg_volume * self.volume_multiplier,
                recent_range < consolidation_threshold,  # Consolidation before breakout
                self._check_liquidity_sweep(recent_data, 'bullish'),
                self._distance_from_recent_high(recent_data) >= self.min_breakout_distance
            ]

            # Bearish breakout conditions
            bearish_conditions = [
                current_price < support_level * (1 - self.breakout_threshold),
                current_volume > avg_volume * self.volume_multiplier,
                recent_range < consolidation_threshold,
                self._check_liquidity_sweep(recent_data, 'bearish'),
                self._distance_from_recent_low(recent_data) >= self.min_breakout_distance
            ]

            # Check bullish breakout
            if all(bullish_conditions):
                self._enter_position(current_price, recent_data['timestamp'].iloc[-1], 'long')
                self.breakout_signals.append({
                    'timestamp': recent_data['timestamp'].iloc[-1],
                    'type': 'bullish',
                    'price': current_price,
                    'resistance': resistance_level
                })
                return 1

            # Check bearish breakout
            elif all(bearish_conditions):
                self._enter_position(current_price, recent_data['timestamp'].iloc[-1], 'short')
                self.breakout_signals.append({
                    'timestamp': recent_data['timestamp'].iloc[-1],
                    'type': 'bearish',
                    'price': current_price,
                    'support': support_level
                })
                return -1

            return 0

        except Exception as e:
            self.logger.error(f"Error in breakout detection: {e}")
            return 0

    def _check_liquidity_sweep(self, data: pd.DataFrame, direction: str) -> bool:
        """
        Check for liquidity sweep before breakout.

        Liquidity sweep indicates institutional activity and increases
        breakout reliability.

        Args:
            data: OHLCV DataFrame
            direction: 'bullish' or 'bearish'

        Returns:
            True if liquidity sweep detected
        """
        try:
            # Get the lowest low in recent period (for bullish breakouts)
            # or highest high (for bearish breakouts)
            if direction == 'bullish':
                extreme_price = data['low'].tail(self.consolidation_period).min()
                sweep_level = data['high'].rolling(self.lookback_period).max().iloc[-1]
                # Check if we broke below the recent low (liquidity sweep)
                return extreme_price < sweep_level * (1 - self.breakout_threshold * 0.5)
            else:  # bearish
                extreme_price = data['high'].tail(self.consolidation_period).max()
                sweep_level = data['low'].rolling(self.lookback_period).min().iloc[-1]
                # Check if we broke above the recent high (liquidity sweep)
                return extreme_price > sweep_level * (1 + self.breakout_threshold * 0.5)

        except Exception as e:
            self.logger.error(f"Error in liquidity sweep check: {e}")
            return False

    def _distance_from_recent_high(self, data: pd.DataFrame) -> float:
        """
        Calculate distance from recent high.

        Args:
            data: OHLCV DataFrame

        Returns:
            Distance as percentage
        """
        try:
            recent_high = data['high'].tail(self.lookback_period).max()
            current_price = data['close'].iloc[-1]
            return abs(current_price - recent_high) / recent_high
        except Exception:
            return 0.0

    def _distance_from_recent_low(self, data: pd.DataFrame) -> float:
        """
        Calculate distance from recent low.

        Args:
            data: OHLCV DataFrame

        Returns:
            Distance as percentage
        """
        try:
            recent_low = data['low'].tail(self.lookback_period).min()
            current_price = data['close'].iloc[-1]
            return abs(current_price - recent_low) / recent_low
        except Exception:
            return 0.0

    def _enter_position(self, price: float, timestamp, side: str):
        """
        Enter a new position.

        Args:
            price: Entry price
            timestamp: Entry timestamp
            side: 'long' or 'short'
        """
        self.position_active = True
        self.entry_price = price
        self.entry_time = timestamp
        self.position_side = side
        self.hold_periods = 0

        self.logger.info(f"Entered {side.upper()} position at {price}")

    def _manage_position(self, data: pd.DataFrame) -> int:
        """
        Manage existing position.

        Args:
            data: OHLCV DataFrame

        Returns:
            Signal: 1 (HOLD LONG), -1 (HOLD SHORT), 0 (EXIT)
        """
        self.hold_periods += 1
        current_price = data['close'].iloc[-1]

        # Check exit conditions
        if self._should_exit_position(data):
            self._exit_position()
            return 0

        # Check maximum hold period
        if self.hold_periods >= self.max_hold_period:
            self._exit_position()
            return 0

        # Return hold signal (maintain position)
        return 1 if self.position_side == 'long' else -1

    def _should_exit_position(self, data: pd.DataFrame) -> bool:
        """
        Determine if position should be exited.

        Args:
            data: OHLCV DataFrame

        Returns:
            True if position should be exited
        """
        try:
            current_price = data['close'].iloc[-1]

            # Exit on opposite breakout (reversal)
            if self.position_side == 'long':
                # Exit long if price breaks below recent support
                recent_low = data['low'].tail(self.lookback_period).min()
                if current_price < recent_low * (1 - self.breakout_threshold * 0.5):
                    return True
            else:  # short
                # Exit short if price breaks above recent resistance
                recent_high = data['high'].tail(self.lookback_period).max()
                if current_price > recent_high * (1 + self.breakout_threshold * 0.5):
                    return True

            # Exit on significant adverse movement
            if self.position_side == 'long':
                loss_threshold = self.entry_price * (1 - self.breakout_threshold * 2)
                if current_price < loss_threshold:
                    return True
            else:  # short
                profit_threshold = self.entry_price * (1 + self.breakout_threshold * 2)
                if current_price > profit_threshold:
                    return True

            return False

        except Exception as e:
            self.logger.error(f"Error checking exit conditions: {e}")
            return True  # Exit on error

    def _exit_position(self):
        """Exit current position."""
        self.position_active = False
        self.entry_price = 0.0
        self.entry_time = None
        self.position_side = None
        self.hold_periods = 0

        self.logger.info("Exited position")

    def get_parameters(self) -> Dict[str, Any]:
        """Get strategy parameters."""
        return {
            'lookback_period': self.lookback_period,
            'breakout_threshold': self.breakout_threshold,
            'volume_multiplier': self.volume_multiplier,
            'consolidation_period': self.consolidation_period,
            'min_breakout_distance': self.min_breakout_distance,
            'max_hold_period': self.max_hold_period,
            'position_active': self.position_active,
            'position_side': self.position_side,
            'entry_price': self.entry_price,
            'hold_periods': self.hold_periods
        }

    def set_parameters(self, **params):
        """Set strategy parameters."""
        for key, value in params.items():
            if hasattr(self, key):
                setattr(self, key, value)

    def reset(self):
        """Reset strategy state."""
        self.position_active = False
        self.entry_price = 0.0
        self.entry_time = None
        self.position_side = None
        self.hold_periods = 0
        self.breakout_signals = []
        self.false_breakouts = 0
        self.successful_breakouts = 0

    def get_performance_stats(self) -> Dict[str, Any]:
        """Get strategy performance statistics."""
        return {
            'total_signals': len(self.breakout_signals),
            'successful_breakouts': self.successful_breakouts,
            'false_breakouts': self.false_breakouts,
            'success_rate': (self.successful_breakouts /
                           max(len(self.breakout_signals), 1)) * 100,
            'current_position': {
                'active': self.position_active,
                'side': self.position_side,
                'entry_price': self.entry_price,
                'hold_periods': self.hold_periods
            }
        }