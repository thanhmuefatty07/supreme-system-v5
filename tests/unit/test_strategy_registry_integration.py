#!/usr/bin/env python3
"""
Integration Test for Strategy Registry

Tests the integration of all strategies with the registry system.
"""

import pytest
from src.strategies import registry, SMACrossover, RSIStrategy, BreakoutStrategy


class TestStrategyRegistryIntegration:
    """Test strategy registry integration."""

    def test_registry_initialization(self):
        """Test registry has built-in strategies registered."""
        strategies = registry.list_strategies()

        # Should have our 3 strategies
        assert len(strategies) >= 3

        # Check strategy names in the list of dicts
        strategy_names = [s['name'] for s in strategies]
        assert "sma_crossover" in strategy_names
        assert "rsi_strategy" in strategy_names
        assert "breakout_strategy" in strategy_names

    def test_sma_crossover_creation(self):
        """Test creating SMA crossover strategy from registry."""
        strategy = registry.create_strategy("sma_crossover")

        assert strategy is not None
        assert isinstance(strategy, SMACrossover)
        assert strategy.name == "SMACrossover"
        assert strategy.fast_window == 10
        assert strategy.slow_window == 20

    def test_rsi_strategy_creation(self):
        """Test creating RSI strategy from registry."""
        strategy = registry.create_strategy("rsi_strategy")

        assert strategy is not None
        assert isinstance(strategy, RSIStrategy)
        assert strategy.name == "RSIStrategy"
        assert strategy.rsi_period == 14
        assert strategy.overbought_level == 70
        assert strategy.oversold_level == 30

    def test_breakout_strategy_creation(self):
        """Test creating breakout strategy from registry."""
        strategy = registry.create_strategy("breakout_strategy")

        assert strategy is not None
        assert isinstance(strategy, BreakoutStrategy)
        assert strategy.name == "BreakoutStrategy"
        assert strategy.lookback_period == 20
        assert strategy.breakout_threshold == 0.02

    def test_strategy_creation_with_custom_params(self):
        """Test creating strategy with custom parameters."""
        custom_config = {
            'fast_window': 5,
            'slow_window': 15,
            'initial_capital': 50000.0,
            'max_position_size': 0.05
        }

        strategy = registry.create_strategy("sma_crossover", **custom_config)

        assert strategy is not None
        assert strategy.fast_window == 5
        assert strategy.slow_window == 15
        assert strategy.portfolio_value == 50000.0

    def test_sma_strategy_functionality(self):
        """Test SMA strategy works through registry."""
        strategy = registry.create_strategy("sma_crossover")

        # Test basic functionality
        signal = strategy.generate_signal({'symbol': 'BTC/USDT', 'close': 100.0})
        assert signal is None  # Not enough data

        # Add more data
        for i in range(25):
            signal = strategy.generate_signal({'symbol': 'BTC/USDT', 'close': 100.0 + i})

        # Should have processed data
        assert len(strategy.prices) > 0

    def test_rsi_strategy_functionality(self):
        """Test RSI strategy works through registry."""
        strategy = registry.create_strategy("rsi_strategy")

        # Test basic functionality
        signal = strategy.generate_signal({'symbol': 'BTC/USDT', 'close': 100.0})
        assert signal is None  # Not enough data

        # Add more data
        for i in range(20):
            signal = strategy.generate_signal({'symbol': 'BTC/USDT', 'close': 100.0})

        # Should have processed data
        assert len(strategy.prices) > 0
        assert len(strategy.rsi_history) > 0

    def test_strategy_metadata(self):
        """Test strategy metadata is properly stored."""
        sma_info = registry.get_strategy_info("sma_crossover")

        assert sma_info is not None
        assert sma_info['name'] == 'sma_crossover'
        assert 'metadata' in sma_info
        assert 'parameters' in sma_info
        assert sma_info['metadata']['type'] == 'trend_following'
        assert sma_info['metadata']['indicators'] == ['SMA']

        rsi_info = registry.get_strategy_info("rsi_strategy")
        assert rsi_info['metadata']['type'] == 'oscillator'
        assert rsi_info['metadata']['indicators'] == ['RSI']

        breakout_info = registry.get_strategy_info("breakout_strategy")
        assert breakout_info['metadata']['type'] == 'breakout'
        assert breakout_info['metadata']['indicators'] == ['support_resistance', 'volume']

    def test_invalid_strategy_creation(self):
        """Test creating non-existent strategy."""
        strategy = registry.create_strategy("non_existent_strategy")
        assert strategy is None

    def test_registry_list_strategies(self):
        """Test listing all registered strategies."""
        strategies = registry.list_strategies()

        assert isinstance(strategies, list)
        assert len(strategies) >= 3

        # Check strategy names are included
        strategy_names = [s['name'] for s in strategies]
        assert "sma_crossover" in strategy_names
        assert "rsi_strategy" in strategy_names
        assert "breakout_strategy" in strategy_names

    def test_strategy_status_reporting(self):
        """Test strategy status reporting."""
        strategy = registry.create_strategy("sma_crossover")

        status = strategy.get_status()

        assert status['name'] == 'SMACrossover'
        assert status['version'] == '2.0'
        assert 'total_signals' in status
        assert 'portfolio_value' in status
        assert status['portfolio_value'] == 10000.0
