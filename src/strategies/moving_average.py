#!/usr/bin/env python3
"""
Supreme System V5 - Moving Average Crossover Strategy

Real implementation of a moving average crossover trading strategy.
"""

from typing import Optional

import numpy as np
import pandas as pd

from .base_strategy import BaseStrategy


class MovingAverageStrategy(BaseStrategy):
    """
    Moving Average Crossover Strategy.

    Generates buy signals when short MA crosses above long MA,
    and sell signals when short MA crosses below long MA.
    """

    def __init__(
        self,
        short_window: int = 10,
        long_window: int = 30,
        name: str = "MovingAverage"
    ):
        """
        Initialize the moving average strategy.

        Args:
            short_window: Period for short moving average
            long_window: Period for long moving average
            name: Strategy name
        """
        super().__init__(name)
        self.short_window = short_window
        self.long_window = long_window

        # Set parameters for tracking
        self.set_parameters(
            short_window=short_window,
            long_window=long_window
        )

    def generate_signal(self, data: pd.DataFrame) -> int:
        """
        Generate trading signal based on moving average crossover.

        Args:
            data: DataFrame with OHLCV data

        Returns:
            1 for buy, -1 for sell, 0 for hold
        """
        if not self.validate_data(data):
            return 0

        # Need enough data for the longer MA
        if len(data) < self.long_window:
            return 0

        # Calculate moving averages
        data = data.copy()
        data['short_ma'] = data['close'].rolling(window=self.short_window).mean()
        data['long_ma'] = data['close'].rolling(window=self.long_window).mean()

        # Get the last two complete data points
        recent_data = data.dropna().tail(2)

        if len(recent_data) < 2:
            return 0

        # Check for crossover
        prev_short = recent_data['short_ma'].iloc[-2]
        prev_long = recent_data['long_ma'].iloc[-2]
        curr_short = recent_data['short_ma'].iloc[-1]
        curr_long = recent_data['long_ma'].iloc[-1]

        # Buy signal: short MA crosses above long MA
        if prev_short <= prev_long and curr_short > curr_long:
            self.logger.debug(f"📈 BUY signal: Short MA ({curr_short:.2f}) crossed above Long MA ({curr_long:.2f})")
            return 1

        # Sell signal: short MA crosses below long MA
        elif prev_short >= prev_long and curr_short < curr_long:
            self.logger.debug(f"📉 SELL signal: Short MA ({curr_short:.2f}) crossed below Long MA ({curr_long:.2f})")
            return -1

        # Hold
        return 0

    def calculate_moving_averages(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate moving averages for analysis.

        Args:
            data: Input DataFrame with 'close' column

        Returns:
            DataFrame with added MA columns
        """
        df = data.copy()
        df['short_ma'] = df['close'].rolling(window=self.short_window).mean()
        df['long_ma'] = df['close'].rolling(window=self.long_window).mean()
        return df
