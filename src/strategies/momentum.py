#!/usr/bin/env python3
"""
Supreme System V5 - Momentum Strategy

Enterprise-grade momentum-based trading strategy.
Trades based on the principle that trending assets continue to trend.
"""

from typing import Any, Dict, Optional
from collections import deque

import numpy as np
import pandas as pd

from .base_strategy import BaseStrategy, Signal


class MomentumStrategy(BaseStrategy):
    """
    Momentum Strategy - Enterprise Grade

    Identifies strong trending assets and trades in the direction of momentum.
    Uses MACD, ROC, and trend strength indicators with memory-safe buffers.

    Production features:
    - MACD (Moving Average Convergence Divergence)
    - Rate of Change (ROC) momentum
    - Trend strength filtering
    - Memory-safe buffer management
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the momentum strategy with enterprise config.

        Args:
            config: Strategy configuration with parameters
        """
        # CRITICAL FIX: Initialize deque attributes BEFORE calling super().__init__()
        buffer_size = max(config.get('long_period', 26) * 3, 100)
        self.price_history = deque(maxlen=buffer_size)
        self.macd_history = deque(maxlen=buffer_size)
        self.roc_history = deque(maxlen=buffer_size)
        self.signal_history = deque(maxlen=buffer_size)

        super().__init__("MomentumStrategy", config)

        # Core MACD parameters
        self.short_period = config.get('short_period', 12)
        self.long_period = config.get('long_period', 26)
        self.signal_period = config.get('signal_period', 9)
        self.roc_period = config.get('roc_period', 10)

        # Momentum filters
        self.trend_threshold = config.get('trend_threshold', 0.02)  # 2% trend strength
        self.volume_confirmation = config.get('volume_confirmation', True)
        self.min_signal_strength = config.get('min_signal_strength', 0.1)

        self.logger.info(f"Momentum Strategy initialized: MACD({self.short_period},{self.long_period},{self.signal_period}), ROC({self.roc_period}), Trend Threshold={self.trend_threshold}")

    def _initialize_state(self):
        """Initialize strategy-specific state."""
        # CRITICAL FIX: Clear deques instead of reassigning
        self.price_history.clear()
        self.macd_history.clear()
        self.roc_history.clear()
        self.signal_history.clear()

    def _calculate_momentum_indicators(self) -> Dict[str, float]:
        """Calculate momentum indicators using price buffer."""
        try:
            # Deque-safe: Convert to list for pandas operations
            prices_list = list(self.prices)
            prices = pd.Series(prices_list)

            # Calculate MACD
            short_ema = prices.ewm(span=self.short_period).mean()
            long_ema = prices.ewm(span=self.long_period).mean()
            macd = short_ema - long_ema
            signal_line = macd.ewm(span=self.signal_period).mean()
            macd_histogram = macd - signal_line

            # Calculate ROC (Rate of Change)
            roc = ((prices - prices.shift(self.roc_period)) / prices.shift(self.roc_period)) * 100

            # Get latest values
            latest_macd = macd.iloc[-1]
            latest_signal = signal_line.iloc[-1]
            latest_histogram = macd_histogram.iloc[-1]
            latest_roc = roc.iloc[-1] if not np.isnan(roc.iloc[-1]) else 0

            # Store in history
            self.macd_history.append({
                'macd': latest_macd,
                'signal': latest_signal,
                'histogram': latest_histogram
            })
            self.roc_history.append(latest_roc)

            return {
                'macd': latest_macd,
                'signal': latest_signal,
                'histogram': latest_histogram,
                'roc': latest_roc
            }

        except Exception as e:
            self.logger.error(f"Error calculating momentum indicators: {e}")
            return {'macd': 0, 'signal': 0, 'histogram': 0, 'roc': 0}

    def _evaluate_momentum_signals(self, indicators: Dict[str, float], current_volume: float) -> int:
        """Evaluate momentum signals and return signal type."""
        try:
            macd = indicators['macd']
            signal_line = indicators['signal']
            histogram = indicators['histogram']
            roc = indicators['roc']

            # MACD signals
            macd_bullish = macd > signal_line and histogram > 0
            macd_bearish = macd < signal_line and histogram < 0

            # ROC signals (momentum strength)
            roc_bullish = roc > self.trend_threshold
            roc_bearish = roc < -self.trend_threshold

            # Combine signals
            bullish_signals = sum([macd_bullish, roc_bullish])
            bearish_signals = sum([macd_bearish, roc_bearish])

            # Volume confirmation (if enabled)
            if self.volume_confirmation and len(self.prices) > 10:
                # Check if volume is above average
                volumes_list = []  # In real implementation, we'd need volume history
                # For now, assume volume confirmation passes
                volume_confirmed = True
            else:
                volume_confirmed = True

            # Decision logic
            if bullish_signals >= 1 and volume_confirmed:
                return 1  # Buy signal
            elif bearish_signals >= 1 and volume_confirmed:
                return -1  # Sell signal
            else:
                return 0  # No signal

        except Exception as e:
            self.logger.error(f"Error evaluating momentum signals: {e}")
            return 0

    def _create_momentum_signal(self, signal_type: int, price: float, symbol: str, indicators: Dict[str, float]) -> Optional[Signal]:
        """Create Signal object for momentum strategy."""
        try:
            if signal_type == 0:
                return None

            side = 'buy' if signal_type == 1 else 'sell'
            signal_strength = self._calculate_momentum_strength(indicators)

            if signal_strength < self.min_signal_strength:
                return None

            metadata = {
                'type': 'momentum_macd',
                'macd': indicators['macd'],
                'signal_line': indicators['signal'],
                'histogram': indicators['histogram'],
                'roc': indicators['roc'],
                'trend_threshold': self.trend_threshold,
                'signal_strength': signal_strength,
                'macd_periods': f"{self.short_period}/{self.long_period}/{self.signal_period}",
                'roc_period': self.roc_period
            }

            return Signal(
                symbol=symbol,
                side=side,
                price=price,
                strength=min(signal_strength, 1.0),
                metadata=metadata
            )

        except Exception as e:
            self.logger.error(f"Error creating momentum signal: {e}")
            return None

    def _calculate_momentum_strength(self, indicators: Dict[str, float]) -> float:
        """Calculate signal strength based on momentum indicators."""
        try:
            macd_strength = abs(indicators['histogram']) / abs(indicators['macd']) if indicators['macd'] != 0 else 0
            roc_strength = abs(indicators['roc']) / 10.0  # Normalize ROC percentage

            # Average of MACD and ROC strength
            strength = (macd_strength + roc_strength) / 2.0

            # Scale to 0-1 range
            return min(strength, 1.0)

        except Exception as e:
            self.logger.error(f"Error calculating momentum strength: {e}")
            return 0.0

        self.short_period = short_period
        self.long_period = long_period
        self.signal_period = signal_period
        self.roc_period = roc_period
        self.trend_threshold = trend_threshold
        self.volume_confirmation = volume_confirmation

        # Performance optimization cache
        self._indicators_cache = {}
        self._last_data_hash = None

        # Use hardware-optimized calculations when available
        from ..utils.vectorized_ops import HardwareDetector
        self.hardware_detector = HardwareDetector()

        # Set parameters for tracking
        self.set_parameters(
            short_period=short_period,
            long_period=long_period,
            signal_period=signal_period,
            roc_period=roc_period,
            trend_threshold=trend_threshold,
            volume_confirmation=volume_confirmation
        )

    def generate_signal(self, market_data: Dict[str, Any]) -> Optional[Signal]:
        """
        Generate trading signal based on momentum indicators.

        Args:
            market_data: Current market data with OHLCV

        Returns:
            Signal object or None if no action needed
        """
        current_price = market_data.get('close')
        current_volume = market_data.get('volume', 0)
        symbol = market_data.get('symbol', 'UNKNOWN')

        if not current_price or current_price <= 0:
            self.logger.warning(f"Invalid price data: {current_price}")
            return None

        # Update price buffer using base class helper (memory-safe)
        self.update_price(current_price)

        # Need enough data for calculations
        min_periods = max(self.long_period + self.signal_period, self.roc_period)
        if len(self.prices) < min_periods:
            return None

        # Calculate momentum indicators
        indicators = self._calculate_momentum_indicators()

        # Generate signal based on momentum strength
        signal_type = self._evaluate_momentum_signals(indicators, current_volume)
        if signal_type == 0:
            return None

        # Create Signal object
        signal = self._create_momentum_signal(signal_type, current_price, symbol, indicators)
        if signal:
            self.total_signals += 1
            self.logger.info(f"Momentum signal generated: {signal.side} @ {signal.price} (strength: {signal.strength:.2f})")

        return signal

    def get_strategy_info(self) -> Dict[str, Any]:
        """Get detailed strategy information."""
        base_info = super().get_status()

        # Add momentum-specific info
        base_info.update({
            'strategy_type': 'Momentum_Based',
            'parameters': {
                'short_period': self.short_period,
                'long_period': self.long_period,
                'signal_period': self.signal_period,
                'roc_period': self.roc_period,
                'trend_threshold': self.trend_threshold,
                'volume_confirmation': self.volume_confirmation,
                'min_signal_strength': self.min_signal_strength
            },
            'current_state': {
                'data_points': len(self.prices),
                'macd_signals': len(self.macd_history),
                'roc_signals': len(self.roc_history),
                'signal_history': len(self.signal_history),
                'latest_price': list(self.prices)[-1] if self.prices else None,
                'latest_macd': self.macd_history[-1] if self.macd_history else None,
                'latest_roc': self.roc_history[-1] if self.roc_history else None
            }
        })

        return base_info
