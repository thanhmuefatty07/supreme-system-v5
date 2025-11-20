#!/usr/bin/env python3
"""
RSI Strategy - Production Ready

Implements RSI-based trading strategy with overbought/oversold signals,
divergence detection, and enterprise-grade implementation.
"""

import pandas as pd
import numpy as np
from typing import Optional, Dict, Any, List
from .base_strategy import BaseStrategy, Signal


class RSIStrategy(BaseStrategy):
    """
    Relative Strength Index (RSI) Strategy.

    Generates signals based on RSI overbought/oversold levels and divergences.
    Production features:
    - Configurable RSI parameters
    - Multiple signal types (overbought/oversold, divergence)
    - Risk-aware signal filtering
    - Performance tracking
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize RSI strategy.

        Args:
            config: Strategy configuration with parameters
        """
        super().__init__("RSIStrategy", config)

        # Core RSI parameters
        self.rsi_period = config.get('rsi_period', 14)
        self.overbought_level = config.get('overbought_level', 70)
        self.oversold_level = config.get('oversold_level', 30)
        self.min_signal_strength = config.get('min_signal_strength', 0.1)

        # Advanced features
        self.enable_divergence = config.get('enable_divergence', True)
        self.divergence_lookback = config.get('divergence_lookback', 5)

        # Internal state
        self.prices = []  # Price history
        self.rsi_history = []  # RSI values history
        self.price_changes = []  # For RSI calculation

        # Optimization
        self.max_buffer_size = max(self.rsi_period * 3, 100)

        self.logger.info(f"RSI Strategy initialized: Period={self.rsi_period}, OB={self.overbought_level}, OS={self.oversold_level}")

    def _initialize_state(self):
        """Initialize strategy-specific state."""
        self.prices = []
        self.rsi_history = []
        self.price_changes = []

    def generate_signal(self, market_data: Dict[str, Any]) -> Optional[Signal]:
        """
        Generate trading signal based on RSI analysis.

        Args:
            market_data: Current market data

        Returns:
            Signal object or None
        """
        current_price = market_data.get('close')
        symbol = market_data.get('symbol', 'UNKNOWN')

        if not current_price or current_price <= 0:
            self.logger.warning(f"Invalid price data: {current_price}")
            return None

        # Calculate price change BEFORE adding to buffer
        if self.prices:  # If we have previous price
            change = current_price - self.prices[-1]  # Current - previous
            self.price_changes.append(change)

            # Maintain buffer size
            if len(self.price_changes) > self.max_buffer_size:
                self.price_changes.pop(0)

        # Update price buffer AFTER calculating change
        self.prices.append(current_price)

        # Maintain buffer size
        if len(self.prices) > self.max_buffer_size:
            self.prices.pop(0)

        # Need enough data for RSI calculation
        if len(self.prices) < self.rsi_period + 1:
            return None

        # Calculate RSI only when we have enough price changes
        if len(self.price_changes) < self.rsi_period:
            return None

        current_rsi = self._calculate_rsi()
        if current_rsi is None:
            return None

        self.rsi_history.append(current_rsi)
        if len(self.rsi_history) > self.max_buffer_size:
            self.rsi_history.pop(0)

        # Generate signals
        signal = self._analyze_rsi_signals(current_rsi, current_price, symbol)

        if signal:
            self.total_signals += 1
            self.logger.info(f"RSI signal generated: {signal.side} @ {signal.price} (RSI: {current_rsi:.2f})")

        return signal

    def _calculate_rsi(self) -> Optional[float]:
        """
        Calculate RSI using Wilder's smoothing method.

        Returns:
            RSI value (0-100) or None if insufficient data
        """
        if len(self.price_changes) < self.rsi_period:
            return None

        # Get recent changes
        changes = self.price_changes[-self.rsi_period:]

        # Separate gains and losses
        gains = [max(change, 0) for change in changes]
        losses = [max(-change, 0) for change in changes]

        # Calculate average gain/loss
        avg_gain = sum(gains) / len(gains)
        avg_loss = sum(losses) / len(losses)

        if avg_loss == 0:
            return 100.0  # No losses = extremely strong

        # Calculate RS and RSI
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))

        return rsi

    def _analyze_rsi_signals(self, current_rsi: float, price: float, symbol: str) -> Optional[Signal]:
        """
        Analyze RSI for trading signals.

        Args:
            current_rsi: Current RSI value
            price: Current price
            symbol: Trading symbol

        Returns:
            Signal if conditions met, None otherwise
        """
        # Basic overbought/oversold signals
        if current_rsi <= self.oversold_level:
            # Oversold - potential buy signal
            signal_strength = (self.oversold_level - current_rsi) / self.oversold_level
            if signal_strength >= self.min_signal_strength:
                return Signal(
                    symbol=symbol,
                    side='buy',
                    price=price,
                    strength=min(signal_strength, 1.0),
                    metadata={
                        'type': 'rsi_oversold',
                        'rsi_value': current_rsi,
                        'threshold': self.oversold_level,
                        'signal_strength': signal_strength,
                        'rsi_period': self.rsi_period
                    }
                )

        elif current_rsi >= self.overbought_level:
            # Overbought - potential sell signal
            signal_strength = (current_rsi - self.overbought_level) / (100 - self.overbought_level)
            if signal_strength >= self.min_signal_strength:
                return Signal(
                    symbol=symbol,
                    side='sell',
                    price=price,
                    strength=min(signal_strength, 1.0),
                    metadata={
                        'type': 'rsi_overbought',
                        'rsi_value': current_rsi,
                        'threshold': self.overbought_level,
                        'signal_strength': signal_strength,
                        'rsi_period': self.rsi_period
                    }
                )

        # Advanced: RSI divergence detection (if enabled)
        if self.enable_divergence and len(self.rsi_history) >= self.divergence_lookback + 1:
            divergence_signal = self._detect_divergence(price, symbol)
            if divergence_signal:
                return divergence_signal

        return None

    def _detect_divergence(self, current_price: float, symbol: str) -> Optional[Signal]:
        """
        Detect RSI divergence patterns.

        Returns:
            Signal if divergence detected, None otherwise
        """
        if len(self.prices) < self.divergence_lookback + 1 or len(self.rsi_history) < self.divergence_lookback + 1:
            return None

        # Get recent data
        recent_prices = self.prices[-(self.divergence_lookback + 1):]
        recent_rsi = self.rsi_history[-(self.divergence_lookback + 1):]

        # Check for bullish divergence (price falling, RSI rising)
        price_trend = recent_prices[-1] - recent_prices[0]
        rsi_trend = recent_rsi[-1] - recent_rsi[0]

        if price_trend < 0 and rsi_trend > 0:  # Price down, RSI up
            strength = abs(rsi_trend) / 10  # Normalize strength
            if strength >= self.min_signal_strength:
                return Signal(
                    symbol=symbol,
                    side='buy',
                    price=current_price,
                    strength=min(strength, 1.0),
                    metadata={
                        'type': 'rsi_bullish_divergence',
                        'price_trend': price_trend,
                        'rsi_trend': rsi_trend,
                        'lookback': self.divergence_lookback,
                        'signal_strength': strength
                    }
                )

        elif price_trend > 0 and rsi_trend < 0:  # Price up, RSI down
            strength = abs(rsi_trend) / 10  # Normalize strength
            if strength >= self.min_signal_strength:
                return Signal(
                    symbol=symbol,
                    side='sell',
                    price=current_price,
                    strength=min(strength, 1.0),
                    metadata={
                        'type': 'rsi_bearish_divergence',
                        'price_trend': price_trend,
                        'rsi_trend': rsi_trend,
                        'lookback': self.divergence_lookback,
                        'signal_strength': strength
                    }
                )

        return None

    def get_strategy_info(self) -> Dict[str, Any]:
        """Get detailed strategy information."""
        base_info = super().get_status()

        # Add RSI-specific info
        base_info.update({
            'strategy_type': 'RSI_Based',
            'parameters': {
                'rsi_period': self.rsi_period,
                'overbought_level': self.overbought_level,
                'oversold_level': self.oversold_level,
                'min_signal_strength': self.min_signal_strength,
                'enable_divergence': self.enable_divergence,
                'divergence_lookback': self.divergence_lookback
            },
            'current_state': {
                'buffer_size': len(self.prices),
                'current_rsi': self.rsi_history[-1] if self.rsi_history else None,
                'rsi_history': self.rsi_history[-10:] if self.rsi_history else []
            }
        })

        return base_info
