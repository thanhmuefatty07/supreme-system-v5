#!/usr/bin/env python3
"""
Supreme System V5 - Momentum Strategy

Real implementation of momentum-based trading strategy.
Trades based on the principle that trending assets continue to trend.
"""

import pandas as pd
import numpy as np
from typing import Optional

from .base_strategy import BaseStrategy


class MomentumStrategy(BaseStrategy):
    """
    Momentum Strategy.

    Identifies strong trending assets and trades in the direction of momentum.
    Uses various momentum indicators like MACD, ROC, and trend strength.
    """

    def __init__(
        self,
        short_period: int = 12,
        long_period: int = 26,
        signal_period: int = 9,
        roc_period: int = 10,
        trend_threshold: float = 0.02,  # 2% trend strength threshold
        volume_confirmation: bool = True,
        name: str = "Momentum"
    ):
        """
        Initialize the momentum strategy.

        Args:
            short_period: Short EMA period for MACD
            long_period: Long EMA period for MACD
            signal_period: Signal line period for MACD
            roc_period: Period for Rate of Change calculation
            trend_threshold: Minimum trend strength for signals
            volume_confirmation: Whether to require volume confirmation
            name: Strategy name
        """
        super().__init__(name)

        self.short_period = short_period
        self.long_period = long_period
        self.signal_period = signal_period
        self.roc_period = roc_period
        self.trend_threshold = trend_threshold
        self.volume_confirmation = volume_confirmation

        # Set parameters for tracking
        self.set_parameters(
            short_period=short_period,
            long_period=long_period,
            signal_period=signal_period,
            roc_period=roc_period,
            trend_threshold=trend_threshold,
            volume_confirmation=volume_confirmation
        )

    def generate_signal(self, data: pd.DataFrame) -> int:
        """
        Generate trading signal based on momentum indicators.

        Args:
            data: DataFrame with OHLCV data

        Returns:
            1 for buy (bullish momentum), -1 for sell (bearish momentum), 0 for hold
        """
        if not self.validate_data(data):
            return 0

        # Need enough data for calculations
        min_periods = max(self.long_period + self.signal_period, self.roc_period)
        if len(data) < min_periods:
            return 0

        # Calculate momentum signals
        macd_signal = self._calculate_macd_signal(data)
        roc_signal = self._calculate_roc_signal(data)
        trend_signal = self._calculate_trend_signal(data)

        # Volume confirmation (if enabled)
        volume_signal = 1
        if self.volume_confirmation:
            volume_signal = self._calculate_volume_confirmation(data)

        # Combine signals with voting system
        signals = [macd_signal, roc_signal, trend_signal]
        if self.volume_confirmation:
            signals.append(volume_signal)

        # Majority vote (at least 60% agreement)
        bullish_signals = sum(1 for s in signals if s == 1)
        bearish_signals = sum(1 for s in signals if s == -1)

        total_signals = len(signals)
        bullish_ratio = bullish_signals / total_signals
        bearish_ratio = bearish_signals / total_signals

        if bullish_ratio >= 0.6:
            return 1
        elif bearish_ratio >= 0.6:
            return -1
        else:
            return 0

    def _calculate_macd_signal(self, data: pd.DataFrame) -> int:
        """Calculate MACD-based momentum signal."""
        try:
            prices = data['close']

            # Calculate EMAs
            short_ema = prices.ewm(span=self.short_period, adjust=False).mean()
            long_ema = prices.ewm(span=self.long_period, adjust=False).mean()

            # Calculate MACD line
            macd_line = short_ema - long_ema

            # Calculate signal line
            signal_line = macd_line.ewm(span=self.signal_period, adjust=False).mean()

            # Calculate histogram
            histogram = macd_line - signal_line

            # Get latest values
            latest_macd = macd_line.iloc[-1]
            latest_signal = signal_line.iloc[-1]
            latest_histogram = histogram.iloc[-1]

            # Previous values for crossover detection
            prev_macd = macd_line.iloc[-2] if len(macd_line) > 1 else latest_macd
            prev_signal = signal_line.iloc[-2] if len(signal_line) > 1 else latest_signal

            # MACD crossover signals
            if prev_macd <= prev_signal and latest_macd > latest_signal:
                # Bullish crossover
                self.logger.debug(".4f")
                return 1
            elif prev_macd >= prev_signal and latest_macd < latest_signal:
                # Bearish crossover
                self.logger.debug(".4f")
                return -1
            else:
                # Check histogram momentum
                if latest_histogram > 0 and abs(latest_histogram) > abs(histogram.iloc[-2] if len(histogram) > 1 else 0):
                    return 1  # Increasing bullish momentum
                elif latest_histogram < 0 and abs(latest_histogram) > abs(histogram.iloc[-2] if len(histogram) > 1 else 0):
                    return -1  # Increasing bearish momentum
                else:
                    return 0

        except Exception as e:
            self.logger.error(f"Error calculating MACD signal: {e}")
            return 0

    def _calculate_roc_signal(self, data: pd.DataFrame) -> int:
        """Calculate Rate of Change (ROC) based momentum signal."""
        try:
            prices = data['close']

            # Calculate ROC: (current_price - price_n_periods_ago) / price_n_periods_ago
            roc = ((prices - prices.shift(self.roc_period)) / prices.shift(self.roc_period)) * 100

            latest_roc = roc.iloc[-1]

            # ROC threshold signals
            if latest_roc > self.trend_threshold * 100:  # Strong upward momentum
                self.logger.debug(".2f")
                return 1
            elif latest_roc < -self.trend_threshold * 100:  # Strong downward momentum
                self.logger.debug(".2f")
                return -1
            else:
                return 0

        except Exception as e:
            self.logger.error(f"Error calculating ROC signal: {e}")
            return 0

    def _calculate_trend_signal(self, data: pd.DataFrame) -> int:
        """Calculate trend strength signal."""
        try:
            prices = data['close']
            lookback = min(20, len(prices) - 1)  # Use up to 20 periods

            # Calculate trend slope using linear regression
            x = np.arange(lookback)
            y = prices.tail(lookback).values

            # Simple trend calculation (slope of price over lookback period)
            slope = np.polyfit(x, y, 1)[0]

            # Calculate trend strength as percentage change
            start_price = prices.iloc[-lookback]
            end_price = prices.iloc[-1]
            trend_pct = (end_price - start_price) / start_price

            # Signal based on trend strength
            if abs(trend_pct) >= self.trend_threshold:
                if trend_pct > 0:
                    self.logger.debug(".2f")
                    return 1
                else:
                    self.logger.debug(".2f")
                    return -1
            else:
                return 0

        except Exception as e:
            self.logger.error(f"Error calculating trend signal: {e}")
            return 0

    def _calculate_volume_confirmation(self, data: pd.DataFrame) -> int:
        """Calculate volume confirmation signal."""
        try:
            volume = data['volume']
            prices = data['close']

            # Calculate volume trend (recent vs historical average)
            recent_volume = volume.tail(5).mean()
            historical_volume = volume.tail(20).mean()

            # Calculate price trend
            recent_prices = prices.tail(5)
            price_trend = (recent_prices.iloc[-1] - recent_prices.iloc[0]) / recent_prices.iloc[0]

            # Volume confirmation logic
            volume_ratio = recent_volume / historical_volume if historical_volume > 0 else 1

            if price_trend > 0 and volume_ratio > 1.2:  # Price up + volume increasing
                return 1
            elif price_trend < 0 and volume_ratio > 1.2:  # Price down + volume increasing
                return -1
            else:
                return 0  # No clear volume confirmation

        except Exception as e:
            self.logger.error(f"Error calculating volume confirmation: {e}")
            return 0

    def calculate_momentum_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate all momentum indicators for analysis.

        Args:
            data: Input DataFrame with OHLCV data

        Returns:
            DataFrame with momentum indicators added
        """
        df = data.copy()

        try:
            prices = df['close']

            # MACD calculation
            short_ema = prices.ewm(span=self.short_period, adjust=False).mean()
            long_ema = prices.ewm(span=self.long_period, adjust=False).mean()
            df['macd_line'] = short_ema - long_ema
            df['macd_signal'] = df['macd_line'].ewm(span=self.signal_period, adjust=False).mean()
            df['macd_histogram'] = df['macd_line'] - df['macd_signal']

            # ROC calculation
            df['roc'] = ((prices - prices.shift(self.roc_period)) / prices.shift(self.roc_period)) * 100

            # Trend strength (simple slope)
            lookback = 10
            df['trend_strength'] = prices.rolling(window=lookback).apply(
                lambda x: np.polyfit(range(len(x)), x, 1)[0] if len(x) > 1 else 0,
                raw=True
            )

            # Volume momentum (if volume available)
            if 'volume' in df.columns:
                df['volume_momentum'] = df['volume'] / df['volume'].rolling(window=10).mean()

        except Exception as e:
            self.logger.error(f"Error calculating momentum indicators: {e}")

        return df

    def get_momentum_score(self, data: pd.DataFrame) -> float:
        """
        Calculate overall momentum score (-1 to 1).

        Args:
            data: Price data

        Returns:
            Momentum score: positive for bullish, negative for bearish
        """
        try:
            signals = [
                self._calculate_macd_signal(data),
                self._calculate_roc_signal(data),
                self._calculate_trend_signal(data)
            ]

            if self.volume_confirmation:
                signals.append(self._calculate_volume_confirmation(data))

            # Calculate weighted score
            bullish_count = sum(1 for s in signals if s == 1)
            bearish_count = sum(1 for s in signals if s == -1)

            total_signals = len(signals)
            score = (bullish_count - bearish_count) / total_signals

            return score

        except Exception as e:
            self.logger.error(f"Error calculating momentum score: {e}")
            return 0.0
