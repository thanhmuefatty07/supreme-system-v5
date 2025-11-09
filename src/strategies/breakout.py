#!/usr/bin/env python3
"""
Supreme System V5 - Breakout Strategy

Real implementation of breakout trading strategy.
Trades based on price breaking through support/resistance levels.
"""

import pandas as pd
import numpy as np
from typing import Optional, List

from .base_strategy import BaseStrategy


class BreakoutStrategy(BaseStrategy):
    """
    Breakout Strategy.

    Identifies key support/resistance levels and trades breakouts
    through these levels with volume confirmation.
    """

    def __init__(
        self,
        lookback_period: int = 20,
        breakout_threshold: float = 0.02,  # 2% breakout threshold
        consolidation_period: int = 10,
        volume_multiplier: float = 1.5,    # Volume must be 1.5x average
        pullback_tolerance: float = 0.005, # 0.5% pullback tolerance
        name: str = "Breakout"
    ):
        """
        Initialize the breakout strategy.

        Args:
            lookback_period: Period to identify support/resistance levels
            breakout_threshold: Minimum breakout percentage
            consolidation_period: Period to check for consolidation
            volume_multiplier: Required volume multiplier for breakout
            pullback_tolerance: Maximum allowed pullback after breakout
            name: Strategy name
        """
        super().__init__(name)

        self.lookback_period = lookback_period
        self.breakout_threshold = breakout_threshold
        self.consolidation_period = consolidation_period
        self.volume_multiplier = volume_multiplier
        self.pullback_tolerance = pullback_tolerance

        # Set parameters for tracking
        self.set_parameters(
            lookback_period=lookback_period,
            breakout_threshold=breakout_threshold,
            consolidation_period=consolidation_period,
            volume_multiplier=volume_multiplier,
            pullback_tolerance=pullback_tolerance
        )

    def generate_signal(self, data: pd.DataFrame) -> int:
        """
        Generate trading signal based on breakout logic.

        Args:
            data: DataFrame with OHLCV data

        Returns:
            1 for buy (breakout up), -1 for sell (breakout down), 0 for hold
        """
        if not self.validate_data(data):
            return 0

        # Need enough data for analysis
        if len(data) < self.lookback_period + self.consolidation_period:
            return 0

        current_price = data['close'].iloc[-1]

        # Check for bullish breakout
        if self._is_bullish_breakout(data, current_price):
            self.logger.debug(".2f")
            return 1

        # Check for bearish breakout
        if self._is_bearish_breakout(data, current_price):
            self.logger.debug(".2f")
            return -1

        return 0

    def _is_bullish_breakout(self, data: pd.DataFrame, current_price: float) -> bool:
        """Check for bullish breakout conditions."""
        try:
            # Identify resistance level (recent high)
            recent_highs = data['high'].tail(self.lookback_period)
            resistance_level = recent_highs.max()

            # Check if current price broke through resistance
            breakout_pct = (current_price - resistance_level) / resistance_level

            if breakout_pct < self.breakout_threshold:
                return False

            # Check for consolidation before breakout
            if not self._is_consolidated(data, self.consolidation_period):
                return False

            # Check volume confirmation
            if not self._has_volume_confirmation(data):
                return False

            # Check for false breakout (price must stay above resistance)
            recent_low = data['low'].tail(5).min()
            if recent_low < resistance_level * (1 - self.pullback_tolerance):
                return False  # False breakout - price pulled back too much

            return True

        except Exception as e:
            self.logger.error(f"Error checking bullish breakout: {e}")
            return False

    def _is_bearish_breakout(self, data: pd.DataFrame, current_price: float) -> bool:
        """Check for bearish breakout conditions."""
        try:
            # Identify support level (recent low)
            recent_lows = data['low'].tail(self.lookback_period)
            support_level = recent_lows.min()

            # Check if current price broke through support
            breakout_pct = (support_level - current_price) / support_level

            if breakout_pct < self.breakout_threshold:
                return False

            # Check for consolidation before breakout
            if not self._is_consolidated(data, self.consolidation_period):
                return False

            # Check volume confirmation
            if not self._has_volume_confirmation(data):
                return False

            # Check for false breakout (price must stay below support)
            recent_high = data['high'].tail(5).max()
            if recent_high > support_level * (1 + self.pullback_tolerance):
                return False  # False breakout - price rallied back too much

            return True

        except Exception as e:
            self.logger.error(f"Error checking bearish breakout: {e}")
            return False

    def _is_consolidated(self, data: pd.DataFrame, period: int) -> bool:
        """Check if price has been consolidating (low volatility)."""
        try:
            recent_data = data.tail(period)

            # Calculate price range (high - low) / average price
            price_range = (recent_data['high'].max() - recent_data['low'].min())
            avg_price = recent_data['close'].mean()

            if avg_price == 0:
                return False

            volatility = price_range / avg_price

            # Consider consolidated if volatility is below threshold
            # Consolidation threshold: price movement less than 3% of average price
            consolidation_threshold = 0.03

            return volatility < consolidation_threshold

        except Exception as e:
            self.logger.error(f"Error checking consolidation: {e}")
            return False

    def _has_volume_confirmation(self, data: pd.DataFrame) -> bool:
        """Check for volume confirmation on breakout."""
        try:
            # Compare recent volume to historical average
            recent_volume = data['volume'].tail(5).mean()
            historical_volume = data['volume'].tail(20).mean()

            if historical_volume == 0:
                return True  # No volume data, assume confirmed

            volume_ratio = recent_volume / historical_volume

            return volume_ratio >= self.volume_multiplier

        except Exception as e:
            self.logger.error(f"Error checking volume confirmation: {e}")
            return False

    def identify_support_resistance(self, data: pd.DataFrame) -> dict:
        """
        Identify key support and resistance levels.

        Args:
            data: OHLCV data

        Returns:
            Dictionary with support and resistance levels
        """
        try:
            lookback = min(self.lookback_period, len(data))

            recent_data = data.tail(lookback)

            # Simple approach: recent highs/lows
            support_level = recent_data['low'].min()
            resistance_level = recent_data['high'].max()

            # Calculate pivot points
            high = recent_data['high'].max()
            low = recent_data['low'].min()
            close = recent_data['close'].iloc[-1]

            pivot = (high + low + close) / 3
            r1 = (2 * pivot) - low
            s1 = (2 * pivot) - high

            return {
                'support': support_level,
                'resistance': resistance_level,
                'pivot': pivot,
                'r1': r1,  # First resistance
                's1': s1   # First support
            }

        except Exception as e:
            self.logger.error(f"Error identifying support/resistance: {e}")
            return {}

    def calculate_breakout_probability(self, data: pd.DataFrame) -> dict:
        """
        Calculate breakout probability based on current conditions.

        Args:
            data: Price data

        Returns:
            Dictionary with bullish/bearish breakout probabilities
        """
        try:
            levels = self.identify_support_resistance(data)

            if not levels:
                return {'bullish': 0.0, 'bearish': 0.0}

            current_price = data['close'].iloc[-1]
            support = levels['support']
            resistance = levels['resistance']

            # Distance to levels
            dist_to_support = abs(current_price - support) / support
            dist_to_resistance = abs(current_price - resistance) / resistance

            # Consolidation check
            is_consolidated = self._is_consolidated(data, self.consolidation_period)

            # Volume check
            has_volume = self._has_volume_confirmation(data)

            # Calculate probabilities
            base_probability = 0.3  # Base probability

            if is_consolidated:
                base_probability += 0.2

            if has_volume:
                base_probability += 0.2

            # Distance factor
            if dist_to_resistance < 0.01:  # Within 1% of resistance
                bullish_prob = base_probability + 0.3
            elif dist_to_support < 0.01:  # Within 1% of support
                bearish_prob = base_probability + 0.3
            else:
                bullish_prob = base_probability
                bearish_prob = base_probability

            return {
                'bullish': min(bullish_prob, 1.0),
                'bearish': min(bearish_prob, 1.0)
            }

        except Exception as e:
            self.logger.error(f"Error calculating breakout probability: {e}")
            return {'bullish': 0.0, 'bearish': 0.0}

    def add_breakout_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Add breakout-related indicators to data.

        Args:
            data: Input DataFrame

        Returns:
            DataFrame with breakout indicators added
        """
        df = data.copy()

        try:
            # Rolling highs and lows
            df['rolling_high'] = df['high'].rolling(window=self.lookback_period).max()
            df['rolling_low'] = df['low'].rolling(window=self.lookback_period).min()

            # Consolidation indicator (price range / average price)
            df['price_range'] = (df['high'] - df['low']) / df['close']
            df['consolidation'] = df['price_range'].rolling(window=self.consolidation_period).mean()

            # Volume ratio
            df['volume_ratio'] = df['volume'] / df['volume'].rolling(window=20).mean()

            # Distance to support/resistance
            levels = self.identify_support_resistance(df)
            if levels:
                df['dist_to_support'] = abs(df['close'] - levels['support']) / levels['support']
                df['dist_to_resistance'] = abs(df['close'] - levels['resistance']) / levels['resistance']

        except Exception as e:
            self.logger.error(f"Error adding breakout indicators: {e}")

        return df

    def get_breakout_signal_strength(self, data: pd.DataFrame) -> float:
        """
        Get overall breakout signal strength (-1 to 1).

        Args:
            data: Price data

        Returns:
            Signal strength: positive for bullish, negative for bearish
        """
        try:
            probabilities = self.calculate_breakout_probability(data)

            # Weight the probabilities
            strength = probabilities['bullish'] - probabilities['bearish']

            return strength

        except Exception as e:
            self.logger.error(f"Error calculating signal strength: {e}")
            return 0.0
