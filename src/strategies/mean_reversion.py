#!/usr/bin/env python3
"""
Supreme System V5 - Mean Reversion Strategy

Enterprise-grade mean reversion trading strategy with Bollinger Bands.
Trades based on the assumption that prices tend to revert to their mean.
"""

from typing import Optional, Dict, Any
from collections import deque

import numpy as np
import pandas as pd

from .base_strategy import BaseStrategy, Signal


class MeanReversionStrategy(BaseStrategy):
    """
    Mean Reversion Strategy - Enterprise Grade

    Identifies overbought/oversold conditions using Bollinger Bands and RSI,
    and trades on the expectation of price reversion to the mean.

    Production features:
    - Bollinger Bands for statistical mean reversion
    - RSI confirmation for enhanced signals
    - Memory-safe buffer management
    - Risk-aware signal generation
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the mean reversion strategy with enterprise config.

        Args:
            config: Strategy configuration with parameters
        """
        # CRITICAL FIX: Initialize deque attributes BEFORE calling super().__init__()
        buffer_size = max(config.get('lookback_period', 20) * 3, 100)
        self.price_history = deque(maxlen=buffer_size)
        self.bollinger_history = deque(maxlen=buffer_size)
        self.rsi_history = deque(maxlen=buffer_size) if config.get('use_rsi', False) else None

        super().__init__("MeanReversionStrategy", config)

        # Core parameters
        self.lookback_period = config.get('lookback_period', 20)
        self.entry_threshold = config.get('entry_threshold', 2.0)  # Standard deviations
        self.exit_threshold = config.get('exit_threshold', 0.5)   # Exit threshold
        self.rsi_period = config.get('rsi_period', 14)
        self.rsi_overbought = config.get('rsi_overbought', 70.0)
        self.rsi_oversold = config.get('rsi_oversold', 30.0)
        self.use_rsi = config.get('use_rsi', False)
        self.min_signal_strength = config.get('min_signal_strength', 0.1)

        self.logger.info(f"Mean Reversion Strategy initialized: Lookback={self.lookback_period}, Entry={self.entry_threshold}σ, RSI={self.use_rsi}")

    def _initialize_state(self):
        """Initialize strategy-specific state."""
        # CRITICAL FIX: Clear deques instead of reassigning
        self.price_history.clear()
        self.bollinger_history.clear()
        if self.rsi_history:
            self.rsi_history.clear()

    def generate_signal(self, market_data: Dict[str, Any]) -> Optional[Signal]:
        """
        Generate trading signal based on mean reversion logic.

        Args:
            market_data: Current market data with OHLCV

        Returns:
            Signal object or None if no action needed
        """
        current_price = market_data.get('close')
        symbol = market_data.get('symbol', 'UNKNOWN')

        if not current_price or current_price <= 0:
            self.logger.warning(f"Invalid price data: {current_price}")
            return None

        # Update price buffer using base class helper (memory-safe)
        self.update_price(current_price)

        # Need enough data for calculations
        min_periods = max(self.lookback_period, self.rsi_period if self.use_rsi else 0)
        if len(self.prices) < min_periods:
            return None

        # Calculate signals
        bb_signal = self._calculate_bollinger_signal(current_price)
        rsi_signal = 0

        if self.use_rsi:
            rsi_signal = self._calculate_rsi_signal()

        # Combine signals (both must agree if using RSI)
        final_signal = 0
        if self.use_rsi and rsi_signal != 0:
            # RSI and Bollinger Bands must agree for confirmation
            if bb_signal == rsi_signal:
                final_signal = bb_signal
        else:
            # Use only Bollinger Bands
            final_signal = bb_signal

        if final_signal == 0:
            return None

        # Generate Signal object
        signal = self._create_signal(final_signal, current_price, symbol)
        if signal:
            self.total_signals += 1
            self.logger.info(f"Mean Reversion signal generated: {signal.side} @ {signal.price} (strength: {signal.strength:.2f})")

        return signal

    def _calculate_bollinger_signal(self, current_price: float) -> int:
        """Calculate signal based on Bollinger Bands using price buffer."""
        try:
            if len(self.prices) < self.lookback_period:
                return 0

            # Deque-safe: Convert to list for pandas operations
            prices_list = list(self.prices)
            prices = pd.Series(prices_list)

            # Calculate rolling mean and standard deviation
            rolling_mean = prices.rolling(window=self.lookback_period).mean()
            rolling_std = prices.rolling(window=self.lookback_period).std()

            # Calculate Bollinger Bands
            upper_band = rolling_mean + (rolling_std * self.entry_threshold)
            lower_band = rolling_mean - (rolling_std * self.entry_threshold)

            # Get latest values
            latest_mean = rolling_mean.iloc[-1]
            latest_upper = upper_band.iloc[-1]
            latest_lower = lower_band.iloc[-1]

            # Store bands for analysis
            self.bollinger_history.append({
                'mean': latest_mean,
                'upper': latest_upper,
                'lower': latest_lower,
                'price': current_price
            })

            # Generate signal based on Bollinger Band position
            if current_price <= latest_lower:
                # Price touched or broke lower band - oversold signal (buy)
                deviation = abs(current_price - latest_mean) / latest_mean
                if deviation >= self.min_signal_strength:
                    self.logger.debug(".2f")
                    return 1
            elif current_price >= latest_upper:
                # Price touched or broke upper band - overbought signal (sell)
                deviation = abs(current_price - latest_mean) / latest_mean
                if deviation >= self.min_signal_strength:
                    self.logger.debug(".2f")
                    return -1

            return 0

        except Exception as e:
            self.logger.error(f"Error calculating Bollinger signal: {e}")
            return 0

    def _calculate_rsi_signal(self) -> int:
        """Calculate signal based on RSI using price buffer."""
        try:
            if not self.rsi_history or len(self.prices) < self.rsi_period + 1:
                return 0

            # Calculate RSI using price buffer
            rsi_value = self._calculate_rsi()
            if rsi_value is None or np.isnan(rsi_value):
                return 0

            # Store RSI value
            self.rsi_history.append(rsi_value)

            # Generate RSI-based signal
            if rsi_value <= self.rsi_oversold:
                # Oversold condition - buy signal
                rsi_deviation = (self.rsi_oversold - rsi_value) / self.rsi_oversold
                if rsi_deviation >= self.min_signal_strength:
                    self.logger.debug(".2f")
                    return 1
            elif rsi_value >= self.rsi_overbought:
                # Overbought condition - sell signal
                rsi_deviation = (rsi_value - self.rsi_overbought) / (100 - self.rsi_overbought)
                if rsi_deviation >= self.min_signal_strength:
                    self.logger.debug(".2f")
                    return -1

            return 0

        except Exception as e:
            self.logger.error(f"Error calculating RSI signal: {e}")
            return 0

    def _calculate_rsi(self) -> Optional[float]:
        """Calculate RSI using price buffer."""
        try:
            if len(self.prices) < self.rsi_period + 1:
                return None

            # Deque-safe: Convert to list for calculations
            prices_list = list(self.prices)
            prices = pd.Series(prices_list)

            # Calculate price changes
            delta = prices.diff()

            # Separate gains and losses
            gain = (delta.where(delta > 0, 0)).rolling(window=self.rsi_period).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=self.rsi_period).mean()

            # Get latest values
            latest_gain = gain.iloc[-1]
            latest_loss = loss.iloc[-1]

            if latest_loss == 0:
                return 100.0  # No losses = extremely strong momentum

            # Calculate RS and RSI
            rs = latest_gain / latest_loss
            rsi = 100 - (100 / (1 + rs))

            return rsi

        except Exception as e:
            self.logger.error(f"Error calculating RSI: {e}")
            return None

    def _create_signal(self, signal_type: int, price: float, symbol: str) -> Optional[Signal]:
        """Create Signal object based on signal type."""
        try:
            if signal_type == 0:
                return None

            side = 'buy' if signal_type == 1 else 'sell'
            signal_strength = self._calculate_signal_strength(signal_type, price)

            if signal_strength < self.min_signal_strength:
                return None

            # Get Bollinger Band info for metadata
            bb_info = {}
            if self.bollinger_history:
                latest_bb = self.bollinger_history[-1]
                bb_info = {
                    'bb_mean': latest_bb['mean'],
                    'bb_upper': latest_bb['upper'],
                    'bb_lower': latest_bb['lower'],
                    'deviation_from_mean': abs(price - latest_bb['mean']) / latest_bb['mean']
                }

            metadata = {
                'type': 'mean_reversion_bollinger' if not self.use_rsi else 'mean_reversion_combined',
                'lookback_period': self.lookback_period,
                'entry_threshold': self.entry_threshold,
                'signal_strength': signal_strength,
                **bb_info
            }

            if self.use_rsi and self.rsi_history:
                metadata['rsi_value'] = self.rsi_history[-1]
                metadata['rsi_period'] = self.rsi_period

            return Signal(
                symbol=symbol,
                side=side,
                price=price,
                strength=min(signal_strength, 1.0),
                metadata=metadata
            )

        except Exception as e:
            self.logger.error(f"Error creating signal: {e}")
            return None

    def _calculate_signal_strength(self, signal_type: int, price: float) -> float:
        """Calculate signal strength based on deviation from mean."""
        try:
            if not self.bollinger_history:
                return 0.0

            latest_bb = self.bollinger_history[-1]
            mean_price = latest_bb['mean']

            # Calculate deviation from mean (normalized)
            deviation = abs(price - mean_price) / mean_price

            # Scale to 0-1 range (higher deviation = stronger signal)
            strength = min(deviation * 2.0, 1.0)  # Cap at 1.0

            return strength

        except Exception as e:
            self.logger.error(f"Error calculating signal strength: {e}")
            return 0.0

    def get_strategy_info(self) -> Dict[str, Any]:
        """Get detailed strategy information."""
        base_info = super().get_status()

        # Add mean reversion-specific info
        base_info.update({
            'strategy_type': 'Mean_Reversion',
            'parameters': {
                'lookback_period': self.lookback_period,
                'entry_threshold': self.entry_threshold,
                'exit_threshold': self.exit_threshold,
                'rsi_period': self.rsi_period,
                'rsi_overbought': self.rsi_overbought,
                'rsi_oversold': self.rsi_oversold,
                'use_rsi': self.use_rsi,
                'min_signal_strength': self.min_signal_strength
            },
            'current_state': {
                'data_points': len(self.prices),
                'bollinger_signals': len(self.bollinger_history),
                'rsi_signals': len(self.rsi_history) if self.rsi_history else 0,
                'latest_price': list(self.prices)[-1] if self.prices else None,
                'latest_bb': self.bollinger_history[-1] if self.bollinger_history else None,
                'latest_rsi': self.rsi_history[-1] if self.rsi_history else None
            }
        })

        return base_info
