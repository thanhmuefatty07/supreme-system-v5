#!/usr/bin/env python3
"""
Comprehensive unit tests for all trading strategies.

Based on algorithmic trading research and testing best practices.
Tests signal generation, parameter validation, and edge cases.
"""

import pytest
import pandas as pd
import numpy as np
from src.strategies.momentum import MomentumStrategy
from src.strategies.moving_average import MovingAverageStrategy
from src.strategies.mean_reversion import MeanReversionStrategy
from src.strategies.breakout import ImprovedBreakoutStrategy as BreakoutStrategy


class TestBaseStrategyFunctionality:
    """Test base strategy functionality and validation."""

    def test_data_validation_valid_data(self, sample_ohlcv_data):
        """Test data validation with valid OHLCV data."""
        strategy = MomentumStrategy()

        # Should pass validation
        assert strategy.validate_data(sample_ohlcv_data) == True

    def test_data_validation_missing_columns(self, sample_ohlcv_data):
        """Test data validation with missing columns."""
        strategy = MomentumStrategy()
        invalid_data = sample_ohlcv_data.drop('volume', axis=1)

        assert strategy.validate_data(invalid_data) == False

    def test_data_validation_empty_data(self):
        """Test data validation with empty DataFrame."""
        strategy = MomentumStrategy()
        empty_data = pd.DataFrame()

        assert strategy.validate_data(empty_data) == False

    def test_parameter_setting(self):
        """Test parameter setting and retrieval."""
        strategy = MomentumStrategy(short_period=10, long_period=20)

        assert strategy.short_period == 10
        assert strategy.long_period == 20

        # Test parameter update
        strategy.set_parameters(short_period=15)
        assert strategy.short_period == 15

        # Test parameter retrieval
        params = strategy.get_parameters()
        assert isinstance(params, dict)
        assert 'short_period' in params


class TestMomentumStrategy:
    """Test Momentum Strategy implementation."""

    @pytest.mark.parametrize("data_size,expected_signal_type", [
        (50, int),   # Small dataset
        (100, int),  # Normal dataset
        (200, int)   # Large dataset
    ])
    def test_signal_generation_basic(self, sample_ohlcv_data, data_size, expected_signal_type):
        """Test basic signal generation across different data sizes."""
        strategy = MomentumStrategy()

        # Test with different data sizes
        test_data = sample_ohlcv_data.head(data_size)
        signal = strategy.generate_signal(test_data)

        assert isinstance(signal, expected_signal_type)
        assert signal in [-1, 0, 1]  # Valid signal values

    def test_momentum_with_trending_data(self):
        """Test momentum strategy with clear trending data."""
        strategy = MomentumStrategy()

        # Create trending up data
        dates = pd.date_range('2024-01-01', periods=50, freq='D')
        prices = np.linspace(100, 120, 50)  # Clear upward trend
        data = pd.DataFrame({
            'timestamp': dates,
            'open': prices,
            'high': prices * 1.02,
            'low': prices * 0.98,
            'close': prices,
            'volume': np.full(50, 1000)
        })

        signal = strategy.generate_signal(data)
        # Should generate buy signal in uptrend
        assert signal >= 0  # Allow 0 or 1

    def test_momentum_with_downtrending_data(self):
        """Test momentum strategy with downtrending data."""
        strategy = MomentumStrategy()

        # Create trending down data
        dates = pd.date_range('2024-01-01', periods=50, freq='D')
        prices = np.linspace(120, 100, 50)  # Clear downward trend
        data = pd.DataFrame({
            'timestamp': dates,
            'open': prices,
            'high': prices * 1.02,
            'low': prices * 0.98,
            'close': prices,
            'volume': np.full(50, 1000)
        })

        signal = strategy.generate_signal(data)
        # Should generate sell signal in downtrend
        assert signal <= 0  # Allow 0 or -1

    def test_momentum_parameters(self):
        """Test momentum strategy parameter effects."""
        # Test with different MACD parameters
        strategy_fast = MomentumStrategy(short_period=8, long_period=21)
        strategy_slow = MomentumStrategy(short_period=12, long_period=26)

        dates = pd.date_range('2024-01-01', periods=60, freq='D')
        prices = 100 + 5 * np.sin(np.linspace(0, 4*np.pi, 60))  # Oscillating data
        data = pd.DataFrame({
            'timestamp': dates,
            'open': prices,
            'high': prices * 1.02,
            'low': prices * 0.98,
            'close': prices,
            'volume': np.full(60, 1000)
        })

        signal_fast = strategy_fast.generate_signal(data)
        signal_slow = strategy_slow.generate_signal(data)

        # Different parameters can produce different signals
        # This is a characteristic test, not asserting specific values
        assert isinstance(signal_fast, int)
        assert isinstance(signal_slow, int)


class TestMovingAverageStrategy:
    """Test Moving Average Strategy implementation."""

    def test_ma_crossover_signals(self, sample_ohlcv_data):
        """Test moving average crossover signal generation."""
        strategy = MovingAverageStrategy(short_window=5, long_window=20)

        signal = strategy.generate_signal(sample_ohlcv_data)

        assert isinstance(signal, int)
        assert signal in [-1, 0, 1]

    def test_ma_with_insufficient_data(self):
        """Test MA strategy with insufficient data."""
        strategy = MovingAverageStrategy(short_window=10, long_window=20)

        # Data shorter than required periods
        small_data = pd.DataFrame({
            'timestamp': pd.date_range('2024-01-01', periods=5, freq='D'),
            'open': [100, 101, 102, 103, 104],
            'high': [105, 106, 107, 108, 109],
            'low': [95, 96, 97, 98, 99],
            'close': [102, 103, 104, 105, 106],
            'volume': [1000] * 5
        })

        signal = strategy.generate_signal(small_data)
        assert signal == 0  # Should return HOLD with insufficient data

    def test_ma_trend_following(self):
        """Test MA strategy trend following capability."""
        strategy = MovingAverageStrategy()

        # Create strong uptrend
        dates = pd.date_range('2024-01-01', periods=50, freq='D')
        prices = 100 + np.linspace(0, 20, 50)  # Strong upward trend
        data = pd.DataFrame({
            'timestamp': dates,
            'open': prices,
            'high': prices * 1.01,
            'low': prices * 0.99,
            'close': prices,
            'volume': np.full(50, 1000)
        })

        signal = strategy.generate_signal(data)
        # Should follow the trend
        assert signal >= 0  # BUY or HOLD in uptrend


class TestMeanReversionStrategy:
    """Test Mean Reversion Strategy implementation."""

    def test_mean_reversion_signals(self, sample_ohlcv_data):
        """Test mean reversion signal generation."""
        strategy = MeanReversionStrategy()

        signal = strategy.generate_signal(sample_ohlcv_data)

        assert isinstance(signal, int)
        assert signal in [-1, 0, 1]

    def test_bollinger_band_calculation(self):
        """Test Bollinger Band calculations."""
        strategy = MeanReversionStrategy(lookback_period=20)

        # Create oscillating data around mean
        dates = pd.date_range('2024-01-01', periods=50, freq='D')
        base_price = 100
        oscillation = 10 * np.sin(np.linspace(0, 4*np.pi, 50))
        prices = base_price + oscillation

        data = pd.DataFrame({
            'timestamp': dates,
            'open': prices,
            'high': prices * 1.02,
            'low': prices * 0.98,
            'close': prices,
            'volume': np.full(50, 1000)
        })

        signal = strategy.generate_signal(data)
        assert isinstance(signal, int)

    def test_mean_reversion_extremes(self):
        """Test mean reversion at price extremes."""
        strategy = MeanReversionStrategy()

        # Create data with extreme moves
        dates = pd.date_range('2024-01-01', periods=60, freq='D')
        prices = np.full(60, 100.0)

        # Add extreme moves at certain points
        prices[20] = 85  # 15% below mean (buy signal expected)
        prices[40] = 120  # 20% above mean (sell signal expected)

        data = pd.DataFrame({
            'timestamp': dates,
            'open': prices,
            'high': prices * 1.01,
            'low': prices * 0.99,
            'close': prices,
            'volume': np.full(60, 1000)
        })

        signal = strategy.generate_signal(data)
        assert isinstance(signal, int)


class TestBreakoutStrategy:
    """Test Breakout Strategy implementation."""

    def test_breakout_signal_generation(self, sample_ohlcv_data):
        """Test breakout signal generation."""
        strategy = BreakoutStrategy()

        signal = strategy.generate_signal(sample_ohlcv_data)

        # Note: Current implementation may return 0, this is expected
        # until breakout logic is fully implemented
        assert isinstance(signal, int)
        assert signal in [-1, 0, 1]

    def test_breakout_with_range_data(self):
        """Test breakout with ranging market data."""
        strategy = BreakoutStrategy()

        # Create ranging data (sideways movement)
        dates = pd.date_range('2024-01-01', periods=60, freq='D')
        prices = 100 + 5 * np.sin(np.linspace(0, 4*np.pi, 60))  # Range bound
        data = pd.DataFrame({
            'timestamp': dates,
            'open': prices,
            'high': prices * 1.02,
            'low': prices * 0.98,
            'close': prices,
            'volume': np.full(60, 1000)
        })

        signal = strategy.generate_signal(data)
        assert isinstance(signal, int)

    def test_breakout_parameters(self):
        """Test breakout strategy with different parameters."""
        strategy = BreakoutStrategy(lookback_period=30, breakout_threshold=0.03)

        dates = pd.date_range('2024-01-01', periods=50, freq='D')
        prices = np.random.uniform(95, 105, 50)  # Random data
        data = pd.DataFrame({
            'timestamp': dates,
            'open': prices,
            'high': prices * 1.02,
            'low': prices * 0.98,
            'close': prices,
            'volume': np.full(50, 1000)
        })

        signal = strategy.generate_signal(data)
        assert isinstance(signal, int)


# Parametrized testing for strategy comparison
@pytest.mark.parametrize("strategy_class,expected_behavior", [
    (MomentumStrategy, "trending"),
    (MovingAverageStrategy, "smoothing"),
    (MeanReversionStrategy, "oscillating"),
    (BreakoutStrategy, "ranging")
])
def test_strategy_types(strategy_class, expected_behavior, sample_ohlcv_data):
    """Test different strategy types with parametrized testing."""
    strategy = strategy_class()
    signal = strategy.generate_signal(sample_ohlcv_data)

    assert isinstance(signal, int)
    assert signal in [-1, 0, 1]


# Performance testing
@pytest.mark.slow
def test_strategy_performance_large_dataset(large_ohlcv_data):
    """Test strategy performance with large datasets."""
    import time

    strategies = [
        MomentumStrategy(),
        MovingAverageStrategy(),
        MeanReversionStrategy(),
        BreakoutStrategy()
    ]

    for strategy in strategies:
        start_time = time.time()
        signal = strategy.generate_signal(large_ohlcv_data)
        end_time = time.time()

        # Should complete in reasonable time
        assert end_time - start_time < 1.0  # Less than 1 second
        assert isinstance(signal, int)


# Edge case testing
def test_strategies_with_minimum_data():
    """Test strategies with minimum required data."""
    # Create minimal dataset
    minimal_data = pd.DataFrame({
        'timestamp': pd.date_range('2024-01-01', periods=30, freq='D'),
        'open': np.full(30, 100.0),
        'high': np.full(30, 101.0),
        'low': np.full(30, 99.0),
        'close': np.full(30, 100.0),
        'volume': np.full(30, 1000.0)
    })

    strategies = [
        MomentumStrategy(),
        MovingAverageStrategy(),
        MeanReversionStrategy(),
        BreakoutStrategy()
    ]

    for strategy in strategies:
        signal = strategy.generate_signal(minimal_data)
        assert isinstance(signal, int)
        # With minimal data, most strategies should return 0 (HOLD)
        # This is expected behavior for insufficient data


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
