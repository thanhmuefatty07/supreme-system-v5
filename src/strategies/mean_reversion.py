#!/usr/bin/env python3
"""
Supreme System V5 - Mean Reversion Strategy

Real implementation of mean reversion trading strategy.
Trades based on the assumption that prices tend to revert to their mean.
"""

from typing import Optional

import numpy as np
import pandas as pd

from .base_strategy import BaseStrategy


class MeanReversionStrategy(BaseStrategy):
    """
    Mean Reversion Strategy.

    Identifies overbought/oversold conditions using Bollinger Bands or RSI,
    and trades on the expectation of price reversion to the mean.
    """

    def __init__(
        self,
        lookback_period: int = 20,
        entry_threshold: float = 2.0,  # Standard deviations for Bollinger Bands
        exit_threshold: float = 0.5,   # Exit when price reverts halfway back
        rsi_period: int = 14,
        rsi_overbought: float = 70.0,
        rsi_oversold: float = 30.0,
        use_rsi: bool = False,
        name: str = "MeanReversion"
    ):
        """
        Initialize the mean reversion strategy.

        Args:
            lookback_period: Period for calculating moving average and std dev
            entry_threshold: Standard deviation threshold for entry signals
            exit_threshold: Threshold for exit signals (reversion distance)
            rsi_period: Period for RSI calculation
            rsi_overbought: RSI level considered overbought
            rsi_oversold: RSI level considered oversold
            use_rsi: Whether to use RSI in addition to Bollinger Bands
            name: Strategy name
        """
        super().__init__(name)

        self.lookback_period = lookback_period
        self.entry_threshold = entry_threshold
        self.exit_threshold = exit_threshold
        self.rsi_period = rsi_period
        self.rsi_overbought = rsi_overbought
        self.rsi_oversold = rsi_oversold
        self.use_rsi = use_rsi

        # Set parameters for tracking
        self.set_parameters(
            lookback_period=lookback_period,
            entry_threshold=entry_threshold,
            exit_threshold=exit_threshold,
            rsi_period=rsi_period,
            rsi_overbought=rsi_overbought,
            rsi_oversold=rsi_oversold,
            use_rsi=use_rsi
        )

    def generate_signal(self, data: pd.DataFrame) -> int:
        """
        Generate trading signal based on mean reversion logic.

        Args:
            data: DataFrame with OHLCV data

        Returns:
            1 for buy (oversold), -1 for sell (overbought), 0 for hold
        """
        if not self.validate_data(data):
            return 0

        # Need enough data for calculations
        min_periods = max(self.lookback_period, self.rsi_period if self.use_rsi else 0)
        if len(data) < min_periods:
            return 0

        current_price = data['close'].iloc[-1]

        # Calculate Bollinger Bands
        bb_signal = self._calculate_bollinger_signal(data, current_price)
        rsi_signal = 0

        if self.use_rsi:
            rsi_signal = self._calculate_rsi_signal(data)

        # Combine signals (both must agree if using RSI)
        if self.use_rsi and rsi_signal != 0:
            # RSI and Bollinger Bands must agree
            if bb_signal == rsi_signal:
                signal = bb_signal
            else:
                signal = 0  # No signal if indicators disagree
        else:
            # Use only Bollinger Bands
            signal = bb_signal

        return signal

    def _calculate_bollinger_signal(self, data: pd.DataFrame, current_price: float) -> int:
        """Calculate signal based on Bollinger Bands."""
        try:
            # Calculate rolling mean and standard deviation
            prices = data['close']
            rolling_mean = prices.rolling(window=self.lookback_period).mean()
            rolling_std = prices.rolling(window=self.lookback_period).std()

            # Calculate Bollinger Bands
            upper_band = rolling_mean + (rolling_std * self.entry_threshold)
            lower_band = rolling_mean - (rolling_std * self.entry_threshold)

            # Get latest values
            latest_mean = rolling_mean.iloc[-1]
            latest_upper = upper_band.iloc[-1]
            latest_lower = lower_band.iloc[-1]

            # Calculate z-score (how many standard deviations from mean)
            if rolling_std.iloc[-1] > 0:
                z_score = (current_price - latest_mean) / rolling_std.iloc[-1]
            else:
                z_score = 0

            # Generate signal based on distance from mean
            if current_price <= latest_lower:
                # Price touched or broke lower band - potential buy signal
                self.logger.debug(f"Bollinger Buy Signal: price {current_price:.2f} <= lower {latest_lower:.2f}")
                return 1
            elif current_price >= latest_upper:
                # Price touched or broke upper band - potential sell signal
                self.logger.debug(f"Bollinger Sell Signal: price {current_price:.2f} >= upper {latest_upper:.2f}")
                return -1
            else:
                # Price within bands - check for partial reversion
                return 0

        except Exception as e:
            self.logger.error(f"Error calculating Bollinger signal: {e}")
            return 0

    def _calculate_rsi_signal(self, data: pd.DataFrame) -> int:
        """Calculate signal based on RSI."""
        try:
            prices = data['close']

            # Calculate RSI
            rsi = self._calculate_rsi(prices, self.rsi_period)

            if rsi is None or np.isnan(rsi):
                return 0

            latest_rsi = rsi.iloc[-1]

            # Generate RSI-based signal
            if latest_rsi <= self.rsi_oversold:
                # Oversold condition
                self.logger.debug(".2f")
                return 1
            elif latest_rsi >= self.rsi_overbought:
                # Overbought condition
                self.logger.debug(".2f")
                return -1
            else:
                return 0

        except Exception as e:
            self.logger.error(f"Error calculating RSI signal: {e}")
            return 0

    def _calculate_rsi(self, prices: pd.Series, period: int) -> Optional[pd.Series]:
        """Calculate RSI (Relative Strength Index)."""
        try:
            # Calculate price changes
        delta = prices.diff()

            # Separate gains and losses
            gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()

            # Calculate RS (Relative Strength)
        rs = gain / loss

            # Calculate RSI
        rsi = 100 - (100 / (1 + rs))

        return rsi
    
        except Exception as e:
            self.logger.error(f"Error calculating RSI: {e}")
            return None

    def calculate_bollinger_bands(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate Bollinger Bands for analysis.

        Args:
            data: Input DataFrame with 'close' column

        Returns:
            DataFrame with Bollinger Bands added
        """
        df = data.copy()

        try:
            prices = df['close']
            rolling_mean = prices.rolling(window=self.lookback_period).mean()
            rolling_std = prices.rolling(window=self.lookback_period).std()

            df['bb_middle'] = rolling_mean
            df['bb_upper'] = rolling_mean + (rolling_std * self.entry_threshold)
            df['bb_lower'] = rolling_mean - (rolling_std * self.entry_threshold)

            # Calculate %B (position within bands)
            df['bb_percent_b'] = (prices - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])

            # Calculate bandwidth (volatility indicator)
            df['bb_bandwidth'] = (df['bb_upper'] - df['bb_lower']) / rolling_mean

        except Exception as e:
            self.logger.error(f"Error calculating Bollinger Bands: {e}")

        return df

    def get_reversion_probability(self, data: pd.DataFrame, current_price: float) -> float:
        """
        Calculate probability of mean reversion based on historical data.

        Args:
            data: Historical price data
            current_price: Current price to analyze

        Returns:
            Probability of reversion (0.0 to 1.0)
        """
        try:
            prices = data['close']

            # Calculate z-score of current price
            mean_price = prices.tail(self.lookback_period).mean()
            std_price = prices.tail(self.lookback_period).std()

            if std_price > 0:
                z_score = abs(current_price - mean_price) / std_price
        else:
                return 0.5  # Neutral if no volatility

            # Calculate reversion probability based on historical data
            # Simple approach: probability based on how extreme the z-score is
            if z_score >= self.entry_threshold:
                # Very extreme - high reversion probability
                return 0.8
            elif z_score >= 1.5:
                return 0.6
            elif z_score >= 1.0:
                return 0.4
        else:
                # Not extreme enough for strong reversion signal
                return 0.2

        except Exception as e:
            self.logger.error(f"Error calculating reversion probability: {e}")
            return 0.5
