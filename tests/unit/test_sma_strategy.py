#!/usr/bin/env python3
"""
Test Suite for SMA Crossover Strategy

Comprehensive tests covering all aspects of the SMA crossover strategy
including signal generation, validation, edge cases, and performance.
"""

import pytest
import pandas as pd
from collections import deque  # CRITICAL FIX: Import deque for test assertions
from unittest.mock import MagicMock
from src.strategies.sma_crossover import SMACrossover


class TestSMACrossover:
    """Test suite for SMA Crossover strategy."""

    @pytest.fixture
    def strategy_config(self):
        """Standard strategy configuration."""
        return {
            'fast_window': 2,
            'slow_window': 5,
            'min_crossover_strength': 0.001,
            'initial_capital': 10000.0,
            'max_position_size': 0.1,
            'max_daily_loss': 0.05
        }

    @pytest.fixture
    def strategy(self, strategy_config):
        """Create strategy instance."""
        return SMACrossover(strategy_config)

    def test_initialization(self, strategy, strategy_config):
        """Test strategy initialization."""
        assert strategy.name == "SMACrossover"
        assert strategy.fast_window == 2
        assert strategy.slow_window == 5
        assert strategy.min_crossover_strength == 0.001
        assert strategy.portfolio_value == 10000.0
        # CRITICAL FIX: Check deque initialization (prevents memory leaks)
        assert len(strategy.prices) == 0
        assert strategy.prices.maxlen == 100  # Buffer size limit
        assert len(strategy.fast_ma_history) == 0
        assert strategy.fast_ma_history.maxlen == 100
        assert len(strategy.slow_ma_history) == 0
        assert strategy.slow_ma_history.maxlen == 100

    def test_not_enough_data(self, strategy):
        """Test behavior when insufficient data for analysis."""
        # Single price point
        signal = strategy.generate_signal({'symbol': 'BTC/USDT', 'close': 100.0})
        assert signal is None

        # Few price points
        for i in range(4):
            signal = strategy.generate_signal({'symbol': 'BTC/USDT', 'close': 100.0 + i})
            assert signal is None

    def test_golden_cross_buy_signal(self, strategy):
        """Test golden cross generates buy signal."""
        # Setup: Slow MA flat, Fast MA rising
        # Prices: [10, 10, 10, 10, 20]
        # Fast MA (2): [10, 10, 10, 15] -> [10, 10, 15, 15] wait, let me calculate properly
        # Actually: prices = [10, 10, 10, 10, 10, 20]
        # Fast MA: 10, 10, 10, 10, 15 (last two: 10+20/2=15)
        # Slow MA: when we have 6 prices, slow MA is average of last 5

        prices = [10, 10, 10, 10, 10, 20]
        signal = None

        for price in prices:
            signal = strategy.generate_signal({'symbol': 'BTC/USDT', 'close': price})

        assert signal is not None
        assert signal.side == 'buy'
        assert signal.price == 20.0
        assert signal.symbol == 'BTC/USDT'
        assert signal.metadata['type'] == 'golden_cross'
        assert 'fast_ma' in signal.metadata
        assert 'slow_ma' in signal.metadata
        assert signal.metadata['crossover_strength'] > 0

    def test_death_cross_sell_signal(self, strategy):
        """Test death cross generates sell signal."""
        # Setup: Fast MA above Slow MA, then Fast MA drops below
        prices = [20, 20, 20, 20, 20, 10]  # Fast MA drops from 20 to 15, Slow MA stays ~18

        signal = None
        for price in prices:
            signal = strategy.generate_signal({'symbol': 'BTC/USDT', 'close': price})

        assert signal is not None
        assert signal.side == 'sell'
        assert signal.price == 10.0
        assert signal.metadata['type'] == 'death_cross'

    def test_no_crossover_when_trending(self, strategy):
        """Test no signal when both MAs trending in same direction."""
        # Both MAs rising together (no crossover)
        prices = [10, 11, 12, 13, 14, 15, 16, 17]

        signal = None
        for price in prices:
            signal = strategy.generate_signal({'symbol': 'BTC/USDT', 'close': price})

        # Should be no crossover signal
        assert signal is None

    def test_weak_crossover_filtered(self, strategy_config):
        """Test that very weak crossovers are filtered out."""
        # Create strategy with high minimum crossover strength
        config = strategy_config.copy()
        config['min_crossover_strength'] = 0.1  # 10% minimum

        strategy = SMACrossover(config)

        # Very small crossover
        prices = [10, 10, 10, 10, 10, 10.01]  # Tiny change

        signal = None
        for price in prices:
            signal = strategy.generate_signal({'symbol': 'BTC/USDT', 'close': price})

        assert signal is None  # Should be filtered due to weak strength

    def test_invalid_price_data(self, strategy):
        """Test handling of invalid price data."""
        # Zero price
        signal = strategy.generate_signal({'symbol': 'BTC/USDT', 'close': 0})
        assert signal is None

        # Negative price
        signal = strategy.generate_signal({'symbol': 'BTC/USDT', 'close': -100})
        assert signal is None

        # Missing price
        signal = strategy.generate_signal({'symbol': 'BTC/USDT'})
        assert signal is None

    def test_signal_validation(self, strategy):
        """Test signal validation functionality."""
        from src.strategies.base_strategy import Signal

        # Valid signal
        valid_signal = Signal('BTC/USDT', 'buy', 100.0, 0.8)
        assert strategy.validate_signal(valid_signal) is True

        # Invalid side
        invalid_signal = Signal('BTC/USDT', 'hold', 100.0, 0.8)
        assert strategy.validate_signal(invalid_signal) is False

        # Invalid price
        invalid_signal = Signal('BTC/USDT', 'buy', -100.0, 0.8)
        assert strategy.validate_signal(invalid_signal) is False

        # Strength clamping
        signal_high = Signal('BTC/USDT', 'buy', 100.0, 1.5)  # > 1.0
        strategy.validate_signal(signal_high)
        assert signal_high.strength <= 1.0

    def test_portfolio_state_updates(self, strategy):
        """Test portfolio state update functionality."""
        portfolio_info = {
            'total_value': 12000.0,
            'current_position': {'symbol': 'BTC/USDT', 'quantity': 0.5}
        }

        strategy.update_portfolio_state(portfolio_info)

        assert strategy.portfolio_value == 12000.0
        assert strategy.current_position == portfolio_info['current_position']

    def test_order_filled_callback(self, strategy):
        """Test order filled callback updates state."""
        initial_pnl = strategy.total_pnl
        initial_executed = strategy.executed_signals

        trade_info = {
            'symbol': 'BTC/USDT',
            'side': 'buy',
            'quantity': 0.1,
            'price': 50000.0,
            'pnl': 100.0
        }

        strategy.on_order_filled(trade_info)

        assert strategy.executed_signals == initial_executed + 1
        assert strategy.total_pnl == initial_pnl + 100.0
        assert strategy.portfolio_value == 10100.0  # Initial 10000 + 100
        assert strategy.current_position is not None
        assert strategy.current_position['symbol'] == 'BTC/USDT'

    def test_reset_functionality(self, strategy):
        """Test strategy reset clears all state."""
        # Add some data and signals
        for i in range(10):
            strategy.generate_signal({'symbol': 'BTC/USDT', 'close': 100.0 + i})

        strategy.total_signals = 5
        strategy.total_pnl = 1000.0

        # Reset
        strategy.reset()

        assert strategy.total_signals == 0
        assert strategy.total_pnl == 0.0
        # CRITICAL FIX: After reset, deques should be cleared but remain deques
        assert len(strategy.prices) == 0
        assert isinstance(strategy.prices, deque)
        assert len(strategy.fast_ma_history) == 0
        assert isinstance(strategy.fast_ma_history, deque)
        assert len(strategy.slow_ma_history) == 0
        assert isinstance(strategy.slow_ma_history, deque)

    def test_get_strategy_info(self, strategy):
        """Test strategy info retrieval."""
        info = strategy.get_strategy_info()

        assert info['name'] == 'SMACrossover'
        assert info['version'] == '2.0'
        assert 'parameters' in info
        assert 'current_state' in info
        assert info['parameters']['fast_window'] == 2
        assert info['parameters']['slow_window'] == 5

    def test_buffer_management(self, strategy):
        """Test price buffer size management."""
        # Add many prices to test buffer trimming
        for i in range(150):  # More than max_buffer_size
            strategy.generate_signal({'symbol': 'BTC/USDT', 'close': 100.0 + i})

        # CRITICAL FIX: Deque auto-manages size up to maxlen
        assert len(strategy.prices) <= 100  # Buffer size limit
        assert len(strategy.fast_ma_history) <= 100  # MA history maxlen
        assert len(strategy.slow_ma_history) <= 100  # MA history maxlen

    @pytest.mark.parametrize("fast_window,slow_window", [
        (5, 10),
        (10, 20),
        (20, 50),
    ])
    def test_different_window_sizes(self, strategy_config, fast_window, slow_window):
        """Test strategy with different MA window combinations."""
        config = strategy_config.copy()
        config['fast_window'] = fast_window
        config['slow_window'] = slow_window

        strategy = SMACrossover(config)
        assert strategy.fast_window == fast_window
        assert strategy.slow_window == slow_window

        # Test basic functionality still works
        signal = strategy.generate_signal({'symbol': 'BTC/USDT', 'close': 100.0})
        assert signal is None  # Not enough data yet
