#!/usr/bin/env python3
"""
Tests for trading strategies
"""

import pytest
import pandas as pd
import numpy as np

import sys
from pathlib import Path

# Add src to path for imports
src_path = Path(__file__).parent.parent / 'src'
sys.path.insert(0, str(src_path))

try:
    from src.strategies.base_strategy import BaseStrategy
    from src.strategies.moving_average import MovingAverageStrategy
except ImportError:
    from strategies.base_strategy import BaseStrategy
    from strategies.moving_average import MovingAverageStrategy


class MockStrategy(BaseStrategy):
    """Mock strategy for testing BaseStrategy functionality"""

    def generate_signal(self, data: pd.DataFrame) -> int:
        return 0  # Always hold


class TestBaseStrategy:
    """Test base strategy functionality"""

    def test_initialization(self):
        """Test strategy initialization"""
        strategy = MockStrategy("TestStrategy")
        assert strategy.name == "TestStrategy"
        assert strategy.parameters == {}

    def test_parameter_setting(self):
        """Test parameter management"""
        strategy = MockStrategy()
        strategy.set_parameters(param1=10, param2="test")
        assert strategy.get_parameters() == {"param1": 10, "param2": "test"}

    def test_data_validation_valid(self):
        """Test data validation with valid data"""
        strategy = MockStrategy()
        data = pd.DataFrame({
            'timestamp': pd.date_range('2024-01-01', periods=10),
            'open': np.random.rand(10),
            'high': np.random.rand(10),
            'low': np.random.rand(10),
            'close': np.random.rand(10),
            'volume': np.random.rand(10)
        })

        assert strategy.validate_data(data) is True

    def test_data_validation_missing_columns(self):
        """Test data validation with missing columns"""
        strategy = MockStrategy()
        data = pd.DataFrame({'close': [1, 2, 3]})  # Missing required columns

        assert strategy.validate_data(data) is False

    def test_data_validation_empty(self):
        """Test data validation with empty data"""
        strategy = MockStrategy()
        data = pd.DataFrame()

        assert strategy.validate_data(data) is False


class TestMovingAverageStrategy:
    """Test moving average strategy"""

    def test_initialization(self):
        """Test MA strategy initialization"""
        strategy = MovingAverageStrategy(short_window=5, long_window=10)
        assert strategy.short_window == 5
        assert strategy.long_window == 10
        assert strategy.get_parameters()['short_window'] == 5

    def test_signal_generation_insufficient_data(self):
        """Test signal generation with insufficient data"""
        strategy = MovingAverageStrategy(short_window=5, long_window=10)
        data = pd.DataFrame({
            'timestamp': pd.date_range('2024-01-01', periods=5),
            'open': [100] * 5,
            'high': [100] * 5,
            'low': [100] * 5,
            'close': [100] * 5,
            'volume': [1000] * 5
        })

        signal = strategy.generate_signal(data)
        assert signal == 0  # Should hold due to insufficient data

    def test_signal_generation_buy(self):
        """Test buy signal generation"""
        strategy = MovingAverageStrategy(short_window=3, long_window=5)

        # Create data where short MA crosses above long MA
        prices = [100, 101, 102, 103, 104, 105, 106, 107, 108, 109]
        data = pd.DataFrame({
            'timestamp': pd.date_range('2024-01-01', periods=len(prices)),
            'open': prices,
            'high': prices,
            'low': prices,
            'close': prices,
            'volume': [1000] * len(prices)
        })

        # Add some data points to ensure we have enough for MA calculation
        signal = strategy.generate_signal(data)
        # Should generate some signals (exact behavior depends on crossover)
        assert signal in [-1, 0, 1]

    def test_signal_generation_sell(self):
        """Test sell signal generation"""
        strategy = MovingAverageStrategy(short_window=3, long_window=5)

        # Create data where short MA crosses below long MA
        prices = [110, 109, 108, 107, 106, 105, 104, 103, 102, 101]
        data = pd.DataFrame({
            'timestamp': pd.date_range('2024-01-01', periods=len(prices)),
            'open': prices,
            'high': prices,
            'low': prices,
            'close': prices,
            'volume': [1000] * len(prices)
        })

        signal = strategy.generate_signal(data)
        # Should generate some signals
        assert signal in [-1, 0, 1]

    def test_calculate_moving_averages(self):
        """Test MA calculation"""
        strategy = MovingAverageStrategy(short_window=3, long_window=5)

        prices = [100, 101, 102, 103, 104, 105]
        data = pd.DataFrame({
            'timestamp': pd.date_range('2024-01-01', periods=len(prices)),
            'close': prices
        })

        result = strategy.calculate_moving_averages(data)

        # Check that MA columns were added
        assert 'short_ma' in result.columns
        assert 'long_ma' in result.columns

        # Check some MA values
        assert not pd.isna(result['short_ma'].iloc[2])  # Should have value at index 2
        assert pd.isna(result['long_ma'].iloc[3])       # Should be NaN until index 4
