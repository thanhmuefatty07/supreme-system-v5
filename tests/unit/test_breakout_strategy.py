#!/usr/bin/env python3
"""
Test Suite for Breakout Strategy

Comprehensive tests for breakout trading strategy including
level detection, breakout signals, volume confirmation, and edge cases.
"""

import pytest
import numpy as np
from collections import deque
from unittest.mock import MagicMock
from src.strategies.breakout_strategy import BreakoutStrategy


class TestBreakoutStrategy:
    """Test suite for Breakout strategy."""

    @pytest.fixture
    def strategy_config(self):
        """Standard strategy configuration."""
        return {
            'lookback_period': 20,
            'breakout_threshold': 0.02,
            'volume_multiplier': 1.5,
            'consolidation_period': 10,
            'require_volume_confirmation': True,
            'min_breakout_strength': 0.1,
            'initial_capital': 10000.0,
            'max_position_size': 0.1,
            'max_daily_loss': 0.05
        }

    @pytest.fixture
    def strategy(self, strategy_config):
        """Create strategy instance."""
        return BreakoutStrategy(strategy_config)

    @pytest.fixture
    def strategy_no_volume(self, strategy_config):
        """Create strategy instance without volume confirmation."""
        config = strategy_config.copy()
        config['require_volume_confirmation'] = False
        return BreakoutStrategy(config)

    def test_initialization(self, strategy, strategy_config):
        """Test strategy initialization."""
        assert strategy.name == "BreakoutStrategy"
        assert strategy.lookback_period == 20
        assert strategy.breakout_threshold == 0.02
        assert strategy.volume_multiplier == 1.5
        assert strategy.require_volume_confirmation is True
        # CRITICAL FIX: Verify deque initialization (prevents memory leaks)
        assert len(strategy.price_history) == 0
        assert isinstance(strategy.price_history, deque)
        assert strategy.price_history.maxlen == 100  # Buffer size limit
        assert len(strategy.volume_history) == 0
        assert isinstance(strategy.volume_history, deque)
        assert strategy.volume_history.maxlen == 100

    def test_insufficient_data(self, strategy):
        """Test behavior with insufficient data."""
        # Need at least lookback_period data points
        for i in range(19):
            signal = strategy.generate_signal({
                'symbol': 'BTC/USDT',
                'close': 100.0,
                'volume': 1000.0
            })
            assert signal is None

    def test_resistance_breakout_buy_signal(self, strategy_no_volume):
        """Test resistance breakout generates buy signal."""
        strategy = strategy_no_volume

        # Create simple scenario: establish resistance at 105, then breakout
        # Use fixed high prices to create clear resistance level
        high_prices = [103, 104, 105, 106, 105, 104, 105, 106, 105, 104,
                      105, 106, 105, 104, 105, 106, 105, 104, 105, 106]

        # Feed data to establish levels
        for price in high_prices:
            strategy.generate_signal({
                'symbol': 'BTC/USDT',
                'close': price,
                'volume': 1000
            })

        # Now test breakout - price significantly above recent highs
        breakout_price = 120.0  # Much higher above 106 resistance to meet min_breakout_strength

        signal = strategy.generate_signal({
            'symbol': 'BTC/USDT',
            'close': breakout_price,
            'volume': 1000
        })

        # Should generate breakout signal since volume confirmation is disabled
        assert signal is not None
        assert signal.side == 'buy'
        assert signal.price == breakout_price
        assert signal.metadata['type'] == 'resistance_breakout'
        assert signal.metadata['volume_confirmed'] is True  # Considered confirmed when disabled

    def test_support_breakdown_sell_signal(self, strategy):
        """Test support breakdown generates sell signal."""
        # Create data with clear support level, then breakdown
        base_price = 100.0
        base_volume = 1000.0

        # First create history to establish support (need >= lookback_period = 20)
        for i in range(20):
            price = base_price - (i % 5)  # Oscillate between 96-100
            strategy.generate_signal({
                'symbol': 'BTC/USDT',
                'close': price,
                'volume': base_volume
            })

        # Now breakdown below support with high volume
        breakdown_price = 85.0  # Sufficient breakdown strength below support
        high_volume = base_volume * 2

        signal = strategy.generate_signal({
            'symbol': 'BTC/USDT',
            'close': breakdown_price,
            'volume': high_volume
        })

        assert signal is not None
        assert signal.side == 'sell'
        assert signal.price == breakdown_price
        assert signal.metadata['type'] == 'support_breakdown'
        assert signal.metadata['volume_confirmed'] is True

    def test_volume_confirmation_required(self, strategy_config):
        """Test that volume confirmation is required."""
        config = strategy_config.copy()
        config['require_volume_confirmation'] = True

        strategy = BreakoutStrategy(config)

        # Setup breakout scenario but with low volume
        base_price = 100.0
        base_volume = 1000.0

        # Create history
        for i in range(19):
            price = base_price + (i % 5)
            strategy.generate_signal({
                'symbol': 'BTC/USDT',
                'close': price,
                'volume': base_volume
            })

        # Attempt breakout with insufficient volume
        breakout_price = 106.0
        low_volume = base_volume * 0.5  # Below threshold

        signal = strategy.generate_signal({
            'symbol': 'BTC/USDT',
            'close': breakout_price,
            'volume': low_volume
        })

        # Should not generate signal due to low volume
        assert signal is None

    def test_volume_confirmation_disabled(self, strategy_config):
        """Test strategy without volume confirmation."""
        config = strategy_config.copy()
        config['require_volume_confirmation'] = False

        strategy = BreakoutStrategy(config)

        # Setup breakout scenario
        base_price = 100.0
        base_volume = 1000.0

        # Create history (need >= lookback_period = 20)
        for i in range(20):
            price = base_price + (i % 5)
            strategy.generate_signal({
                'symbol': 'BTC/USDT',
                'close': price,
                'volume': base_volume
            })

        # Breakout with any volume (since confirmation disabled)
        breakout_price = 116.0  # Need sufficient strength above resistance
        any_volume = 1.0

        signal = strategy.generate_signal({
            'symbol': 'BTC/USDT',
            'close': breakout_price,
            'volume': any_volume
        })

        assert signal is not None
        assert signal.side == 'buy'
        assert signal.metadata['volume_confirmed'] is True  # Considered confirmed when disabled

    def test_weak_breakout_filtered(self, strategy_config):
        """Test that weak breakouts are filtered."""
        config = strategy_config.copy()
        config['min_breakout_strength'] = 0.5  # High threshold

        strategy = BreakoutStrategy(config)

        # Setup scenario
        base_price = 100.0
        base_volume = 1000.0

        # Create history
        for i in range(19):
            price = base_price + (i % 5)
            strategy.generate_signal({
                'symbol': 'BTC/USDT',
                'close': price,
                'volume': base_volume * 3  # High volume
            })

        # Very weak breakout
        weak_breakout_price = 102.0  # Only 2% above resistance
        high_volume = base_volume * 3

        signal = strategy.generate_signal({
            'symbol': 'BTC/USDT',
            'close': weak_breakout_price,
            'volume': high_volume
        })

        # Should be filtered due to weak strength
        assert signal is None

    def test_invalid_data_handling(self, strategy):
        """Test handling of invalid market data."""
        # Zero price
        signal = strategy.generate_signal({
            'symbol': 'BTC/USDT',
            'close': 0,
            'volume': 1000.0
        })
        assert signal is None

        # Negative price
        signal = strategy.generate_signal({
            'symbol': 'BTC/USDT',
            'close': -100.0,
            'volume': 1000.0
        })
        assert signal is None

        # Missing price
        signal = strategy.generate_signal({
            'symbol': 'BTC/USDT',
            'volume': 1000.0
        })
        assert signal is None

    def test_level_calculation(self, strategy):
        """Test support/resistance level calculation."""
        # Add price data
        prices = [100, 101, 102, 103, 104, 103, 102, 101, 100, 101,
                 102, 103, 104, 103, 102, 101, 100, 101, 102, 103]

        for price in prices:
            strategy.generate_signal({
                'symbol': 'BTC/USDT',
                'close': price,
                'volume': 1000.0
            })

        # Check that levels are calculated
        assert len(strategy.support_levels) > 0
        assert len(strategy.resistance_levels) > 0

        # Basic sanity checks
        support = strategy.support_levels[-1]
        resistance = strategy.resistance_levels[-1]
        assert support < resistance

    def test_consolidation_detection(self, strategy):
        """Test market consolidation detection."""
        # Create consolidating (sideways) market
        consolidation_prices = [100, 100.5, 99.8, 100.2, 99.9, 100.1, 99.7, 100.3, 99.8, 100.0]

        for price in consolidation_prices:
            strategy.generate_signal({
                'symbol': 'BTC/USDT',
                'close': price,
                'volume': 1000.0
            })

        # Should detect consolidation
        assert strategy._is_consolidating() is True

    def test_non_consolidation_detection(self, strategy):
        """Test trending market (non-consolidation)."""
        # Create trending market
        trending_prices = [100, 105, 110, 115, 120, 125, 130, 135, 140, 145]

        for price in trending_prices:
            strategy.generate_signal({
                'symbol': 'BTC/USDT',
                'close': price,
                'volume': 1000.0
            })

        # Should not be consolidating
        assert strategy._is_consolidating() is False

    def test_signal_validation(self, strategy):
        """Test signal validation functionality."""
        from src.strategies.base_strategy import Signal

        # Valid signal
        valid_signal = Signal('BTC/USDT', 'buy', 100.0, 0.8)
        assert strategy.validate_signal(valid_signal) is True

        # Invalid side
        invalid_signal = Signal('BTC/USDT', 'hold', 100.0, 0.8)
        assert strategy.validate_signal(invalid_signal) is False

    def test_buffer_management(self, strategy):
        """Test data buffer size management."""
        # Add many data points
        for i in range(150):
            strategy.generate_signal({
                'symbol': 'BTC/USDT',
                'close': 100.0 + i,
                'volume': 1000.0
            })

        # Buffers should be trimmed
        assert len(strategy.price_history) <= strategy.max_history_size
        assert len(strategy.volume_history) <= strategy.max_history_size

    def test_get_strategy_info(self, strategy):
        """Test strategy info retrieval."""
        info = strategy.get_strategy_info()

        assert info['name'] == 'BreakoutStrategy'
        assert info['version'] == '2.0'
        assert 'parameters' in info
        assert 'current_state' in info
        assert info['parameters']['lookback_period'] == 20
        assert info['parameters']['breakout_threshold'] == 0.02

    @pytest.mark.parametrize("lookback,threshold", [
        (10, 0.01),
        (30, 0.05),
        (20, 0.02),  # Standard values
    ])
    def test_different_parameters(self, strategy_config, lookback, threshold):
        """Test strategy with different parameters."""
        config = strategy_config.copy()
        config['lookback_period'] = lookback
        config['breakout_threshold'] = threshold

        strategy = BreakoutStrategy(config)
        assert strategy.lookback_period == lookback
        assert strategy.breakout_threshold == threshold

    def test_reset_functionality(self, strategy):
        """Test strategy reset."""
        # Add some data
        for i in range(25):
            strategy.generate_signal({
                'symbol': 'BTC/USDT',
                'close': 100.0,
                'volume': 1000.0
            })

        strategy.total_signals = 3

        strategy.reset()

        assert strategy.total_signals == 0
        # CRITICAL FIX: After reset, deques should be cleared but remain deques
        assert len(strategy.price_history) == 0
        assert isinstance(strategy.price_history, deque)
        assert len(strategy.volume_history) == 0
        assert isinstance(strategy.volume_history, deque)
        assert len(strategy.support_levels) == 0
        assert isinstance(strategy.support_levels, deque)
        assert len(strategy.resistance_levels) == 0
        assert isinstance(strategy.resistance_levels, deque)
