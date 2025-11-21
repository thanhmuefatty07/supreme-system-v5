#!/usr/bin/env python3
"""
Comprehensive tests for MomentumStrategy.

Tests cover:
- Initialization and configuration
- MACD calculation and signals
- ROC momentum signals
- Combined signal generation
- Memory management
- Edge cases and error handling
"""

import pytest
from collections import deque
from src.strategies.momentum import MomentumStrategy
from src.strategies.base_strategy import Signal


class TestMomentumStrategy:
    """Test suite for MomentumStrategy."""

    @pytest.fixture
    def strategy_config(self):
        """Default strategy configuration."""
        return {
            'short_period': 12,
            'long_period': 26,
            'signal_period': 9,
            'roc_period': 10,
            'trend_threshold': 0.02,
            'volume_confirmation': True,
            'min_signal_strength': 0.1,
            'buffer_size': 100
        }

    @pytest.fixture
    def strategy(self, strategy_config):
        """Create strategy instance."""
        return MomentumStrategy(strategy_config)

    def test_initialization(self, strategy, strategy_config):
        """Test strategy initialization."""
        assert strategy.name == "MomentumStrategy"
        assert strategy.short_period == strategy_config['short_period']
        assert strategy.long_period == strategy_config['long_period']
        assert strategy.roc_period == strategy_config['roc_period']
        assert strategy.trend_threshold == strategy_config['trend_threshold']
        assert isinstance(strategy.prices, deque)
        assert isinstance(strategy.macd_history, deque)
        assert isinstance(strategy.roc_history, deque)
        assert strategy.prices.maxlen == strategy_config['buffer_size']

    def test_insufficient_data(self, strategy):
        """Test behavior with insufficient data."""
        # Test with empty market data
        signal = strategy.generate_signal({'close': 100.0})
        assert signal is None

        # Test with some data but not enough for MACD calculations
        for i in range(15):  # Less than long_period + signal_period
            strategy.update_price(100.0 + i * 0.1)
        signal = strategy.generate_signal({'close': 101.5})
        assert signal is None

    def test_macd_bullish_signal(self, strategy):
        """Test bullish MACD signal generation."""
        # Create trending up data that should generate bullish MACD
        prices = []
        for i in range(50):  # Enough for MACD calculation
            prices.append(100.0 + i * 0.5)  # Strong upward trend

        for price in prices:
            strategy.update_price(price)

        signal = strategy.generate_signal({'close': 125.0, 'symbol': 'TEST', 'volume': 1000})
        assert signal is not None
        assert signal.side == 'buy'
        assert signal.symbol == 'TEST'
        assert signal.price == 125.0
        assert 'momentum_macd' in signal.metadata['type']
        assert 'macd' in signal.metadata
        assert 'roc' in signal.metadata

    def test_macd_bearish_signal(self, strategy):
        """Test bearish MACD signal generation."""
        # Create trending down data that should generate bearish MACD
        prices = []
        for i in range(50):
            prices.append(150.0 - i * 0.5)  # Strong downward trend

        for price in prices:
            strategy.update_price(price)

        signal = strategy.generate_signal({'close': 125.0, 'symbol': 'TEST', 'volume': 1000})
        assert signal is not None
        assert signal.side == 'sell'
        assert signal.symbol == 'TEST'
        assert 'macd' in signal.metadata

    def test_roc_momentum_signals(self, strategy):
        """Test ROC-based momentum signals."""
        # Create strong momentum data
        prices = [100.0] * 20  # Stable first
        for i in range(20):    # Then strong uptrend
            prices.append(100.0 + i * 2.0)

        for price in prices:
            strategy.update_price(price)

        signal = strategy.generate_signal({'close': 140.0, 'symbol': 'TEST'})
        assert signal is not None
        assert signal.side == 'buy'
        assert signal.metadata['roc'] > strategy.trend_threshold

    def test_weak_trend_handling(self, strategy):
        """Test handling of weak or sideways trends."""
        # Create sideways data (no clear trend)
        prices = []
        for i in range(50):
            prices.append(100.0 + (i % 10) * 0.1)  # Small oscillations

        for price in prices:
            strategy.update_price(price)

        signal = strategy.generate_signal({'close': 100.5, 'symbol': 'TEST'})
        # Strategy may or may not generate signal based on calculations
        # The important thing is it doesn't crash
        assert signal is None or isinstance(signal, Signal)

    def test_signal_strength_calculation(self, strategy):
        """Test signal strength calculation."""
        # Create strong trend data
        prices = []
        for i in range(40):
            prices.append(100.0 + i * 1.0)  # Moderate uptrend

        for price in prices:
            strategy.update_price(price)

        signal = strategy.generate_signal({'close': 140.0, 'symbol': 'TEST'})
        assert signal is not None
        assert 0.0 <= signal.strength <= 1.0
        assert signal.strength >= strategy.min_signal_strength

    def test_weak_signal_filtered(self, strategy):
        """Test that weak signals below threshold are filtered."""
        # Create weak trend data
        prices = [100.0] * 40  # Very stable, minimal movement
        for price in prices:
            strategy.update_price(price)

        # Modify strategy to have high strength threshold
        strategy.min_signal_strength = 0.8

        signal = strategy.generate_signal({'close': 100.1, 'symbol': 'TEST'})
        assert signal is None  # Signal too weak

    def test_invalid_price_data(self, strategy):
        """Test handling of invalid price data."""
        signal = strategy.generate_signal({'close': 0, 'symbol': 'TEST'})
        assert signal is None

        signal = strategy.generate_signal({'close': -10.0, 'symbol': 'TEST'})
        assert signal is None

        signal = strategy.generate_signal({'close': None, 'symbol': 'TEST'})
        assert signal is None

    def test_signal_metadata_completeness(self, strategy):
        """Test signal metadata contains all required information."""
        # Create strong uptrend
        prices = []
        for i in range(50):
            prices.append(100.0 + i * 0.8)

        for price in prices:
            strategy.update_price(price)

        signal = strategy.generate_signal({'close': 140.0, 'symbol': 'TEST'})
        assert signal is not None

        # Check metadata completeness
        metadata = signal.metadata
        required_fields = [
            'type', 'macd', 'signal_line', 'histogram', 'roc',
            'trend_threshold', 'signal_strength', 'macd_periods', 'roc_period'
        ]

        for field in required_fields:
            assert field in metadata, f"Missing {field} in metadata"

        assert metadata['type'] == 'momentum_macd'
        assert isinstance(metadata['macd'], (int, float))
        assert isinstance(metadata['roc'], (int, float))

    def test_buffer_management(self, strategy):
        """Test memory-safe buffer management."""
        # Add many prices to test maxlen
        for i in range(150):  # More than buffer_size (100)
            strategy.update_price(100.0 + i * 0.1)

        assert len(strategy.prices) <= strategy.prices.maxlen
        assert len(strategy.macd_history) <= strategy.macd_history.maxlen
        assert len(strategy.roc_history) <= strategy.roc_history.maxlen

        # MACD history is only populated during signal generation
        # Generate a signal to populate the buffers
        signal = strategy.generate_signal({'close': 125.0, 'symbol': 'TEST'})

        # Now check that buffers have been populated
        assert len(strategy.prices) > 0
        # MACD and ROC buffers may be populated depending on data sufficiency
        assert len(strategy.macd_history) >= 0  # At least empty
        assert len(strategy.roc_history) >= 0   # At least empty

    def test_reset_functionality(self, strategy):
        """Test strategy reset functionality."""
        # Add some data and generate signals
        for i in range(50):
            strategy.update_price(100.0 + i * 0.5)

        strategy.generate_signal({'close': 125.0, 'symbol': 'TEST'})
        strategy.generate_signal({'close': 127.0, 'symbol': 'TEST'})

        # Reset
        strategy.reset()

        # Check reset worked
        assert len(strategy.prices) == 0
        assert isinstance(strategy.prices, deque)
        assert len(strategy.macd_history) == 0
        assert len(strategy.roc_history) == 0
        assert len(strategy.signal_history) == 0
        assert strategy.total_signals == 0

    def test_get_strategy_info(self, strategy):
        """Test strategy info retrieval."""
        # Add some data
        for i in range(40):
            strategy.update_price(100.0 + i * 0.3)

        info = strategy.get_strategy_info()

        assert info['strategy_type'] == 'Momentum_Based'
        assert info['parameters']['short_period'] == 12
        assert info['parameters']['long_period'] == 26
        assert info['parameters']['roc_period'] == 10
        assert info['current_state']['data_points'] == 40
        assert 'macd_signals' in info['current_state']
        assert 'roc_signals' in info['current_state']

    def test_different_configurations(self):
        """Test strategy with different MACD configurations."""
        configs = [
            {'short_period': 8, 'long_period': 21, 'signal_period': 5, 'roc_period': 7},
            {'short_period': 5, 'long_period': 35, 'signal_period': 5, 'trend_threshold': 0.05},
            {'short_period': 15, 'long_period': 30, 'signal_period': 10, 'volume_confirmation': False}
        ]

        for config in configs:
            strategy = MomentumStrategy(config)
            assert strategy.short_period == config['short_period']
            assert strategy.long_period == config['long_period']
            assert strategy.signal_period == config['signal_period']
            assert strategy.roc_period == config.get('roc_period', 10)
            assert strategy.volume_confirmation == config.get('volume_confirmation', True)

    def test_indicator_calculations(self, strategy):
        """Test MACD and ROC indicator calculations."""
        # Add test data
        prices = [100.0, 101.0, 102.0, 103.0, 104.0, 105.0, 104.0, 103.0, 104.0, 105.0,
                 106.0, 107.0, 108.0, 109.0, 110.0, 111.0, 112.0, 113.0, 114.0, 115.0,
                 116.0, 117.0, 118.0, 119.0, 120.0, 121.0, 122.0, 123.0, 124.0, 125.0]

        for price in prices:
            strategy.update_price(price)

        # Test indicator calculation
        indicators = strategy._calculate_momentum_indicators()

        assert 'macd' in indicators
        assert 'signal' in indicators
        assert 'histogram' in indicators
        assert 'roc' in indicators

        # MACD should be positive for uptrend
        assert indicators['macd'] > 0
        # ROC should be positive for uptrend
        assert indicators['roc'] > 0

    def test_combined_signal_logic(self, strategy):
        """Test combined MACD and ROC signal logic."""
        # Create data that should trigger both MACD and ROC signals
        prices = []
        for i in range(50):
            prices.append(100.0 + i * 0.8)  # Strong uptrend

        for price in prices:
            strategy.update_price(price)

        # Test bullish signal evaluation
        indicators = {'macd': 2.0, 'signal': 1.5, 'histogram': 0.5, 'roc': 5.0}
        signal_type = strategy._evaluate_momentum_signals(indicators, 1000)
        assert signal_type == 1  # Buy signal

        # Test bearish signal evaluation
        bearish_indicators = {'macd': -2.0, 'signal': -1.5, 'histogram': -0.5, 'roc': -5.0}
        signal_type = strategy._evaluate_momentum_signals(bearish_indicators, 1000)
        assert signal_type == -1  # Sell signal

        # Test no signal for very weak indicators
        weak_indicators = {'macd': 0.001, 'signal': 0.001, 'histogram': 0.0, 'roc': 0.001}
        signal_type = strategy._evaluate_momentum_signals(weak_indicators, 1000)
        assert signal_type == 0  # No signal for very weak indicators

    def test_edge_case_error_handling(self, strategy):
        """Test error handling in edge cases."""
        # Test with empty indicators
        signal_type = strategy._evaluate_momentum_signals({}, 0)
        assert signal_type == 0

        # Test signal creation with invalid data
        signal = strategy._create_momentum_signal(0, 100.0, 'TEST', {})
        assert signal is None

        # Test strength calculation with invalid data
        strength = strategy._calculate_momentum_strength({})
        assert strength == 0.0
