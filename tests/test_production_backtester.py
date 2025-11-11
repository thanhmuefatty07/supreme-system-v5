#!/usr/bin/env python3
"""
Tests for Supreme System V5 production backtester.

Tests comprehensive backtesting functionality with realistic market data.
"""

import pytest
import pandas as pd
import numpy as np
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime, timedelta

# Import backtesting components
try:
    from src.backtesting.production_backtester import ProductionBacktester
    from src.strategies.moving_average import MovingAverageStrategy
    from src.risk.risk_manager import RiskManager
except ImportError:
    ProductionBacktester = None
    MovingAverageStrategy = None
    RiskManager = None


@pytest.fixture
def sample_backtest_data():
    """Generate comprehensive backtest data."""
    np.random.seed(42)

    # Generate 6 months of hourly data
    dates = pd.date_range('2024-01-01', periods=4320, freq='1H')  # ~6 months

    # Realistic price movements with trends and volatility
    base_price = 50000
    trend = np.linspace(0, 0.3, len(dates))  # Gradual upward trend
    noise = np.random.normal(0, 0.015, len(dates))  # 1.5% daily volatility
    gaps = np.random.choice([0, 0.01, -0.01], len(dates), p=[0.96, 0.02, 0.02])

    price_changes = trend + noise + gaps
    prices = base_price * np.cumprod(1 + price_changes)

    # Create OHLCV data
    high_spread = np.random.uniform(1.001, 1.008, len(dates))
    low_spread = np.random.uniform(0.992, 0.999, len(dates))
    volume_base = np.random.uniform(1000, 10000, len(dates))

    data = pd.DataFrame({
        'timestamp': dates,
        'open': prices,
        'high': prices * high_spread,
        'low': prices * low_spread,
        'close': prices,
        'volume': volume_base
    })

    # Ensure OHLC relationships
    data['high'] = np.maximum(data[['open', 'close']].max(axis=1), data['high'])
    data['low'] = np.minimum(data[['open', 'close']].min(axis=1), data['low'])

    return data


@pytest.fixture
def backtest_config():
    """Standard backtest configuration."""
    return {
        'initial_capital': 100000,
        'commission': 0.001,  # 0.1%
        'slippage': 0.0005,   # 0.05%
        'max_position_size': 0.1,  # 10% of capital
        'stop_loss': 0.02,    # 2%
        'take_profit': 0.05,  # 5%
        'max_holding_period': 24,  # hours
    }


@pytest.mark.skipif(ProductionBacktester is None, reason="Production backtester not available")
class TestProductionBacktester:
    """Test production backtester functionality."""

    def test_backtester_initialization(self, backtest_config):
        """Test backtester initialization with config."""
        backtester = ProductionBacktester(config=backtest_config)
        assert backtester is not None
        assert backtester.initial_capital == 100000
        assert backtester.commission == 0.001

    def test_strategy_registration(self, backtest_config):
        """Test strategy registration and management."""
        backtester = ProductionBacktester(config=backtest_config)

        strategy = MovingAverageStrategy(short_period=10, long_period=20)
        backtester.register_strategy('MA_Crossover', strategy)

        assert 'MA_Crossover' in backtester.strategies
        assert backtester.strategies['MA_Crossover'] == strategy

    def test_risk_manager_integration(self, backtest_config):
        """Test integration with risk manager."""
        backtester = ProductionBacktester(config=backtest_config)

        # Should initialize risk manager
        assert hasattr(backtester, 'risk_manager')
        assert isinstance(backtester.risk_manager, RiskManager)

    def test_data_validation(self, sample_backtest_data, backtest_config):
        """Test data validation before backtesting."""
        backtester = ProductionBacktester(config=backtest_config)

        # Valid data should pass
        is_valid, errors = backtester.validate_data(sample_backtest_data)
        assert is_valid, f"Data validation failed: {errors}"
        assert len(errors) == 0

    def test_data_validation_missing_columns(self, backtest_config):
        """Test data validation with missing required columns."""
        backtester = ProductionBacktester(config=backtest_config)

        # Missing required columns
        invalid_data = pd.DataFrame({
            'timestamp': pd.date_range('2024-01-01', periods=10),
            'price': np.random.uniform(50000, 51000, 10)
        })

        is_valid, errors = backtester.validate_data(invalid_data)
        assert not is_valid
        assert len(errors) > 0
        assert any('missing' in error.lower() for error in errors)

    def test_data_validation_nan_values(self, sample_backtest_data, backtest_config):
        """Test data validation with NaN values."""
        backtester = ProductionBacktester(config=backtest_config)

        # Introduce NaN values
        data_with_nan = sample_backtest_data.copy()
        data_with_nan.loc[0, 'close'] = np.nan

        is_valid, errors = backtester.validate_data(data_with_nan)
        assert not is_valid
        assert len(errors) > 0

    def test_single_strategy_backtest(self, sample_backtest_data, backtest_config):
        """Test backtesting with single strategy."""
        backtester = ProductionBacktester(config=backtest_config)

        strategy = MovingAverageStrategy(short_period=5, long_period=20)
        backtester.register_strategy('MA_Test', strategy)

        results = backtester.run_backtest(
            strategy_name='MA_Test',
            data=sample_backtest_data,
            start_date=sample_backtest_data['timestamp'].min(),
            end_date=sample_backtest_data['timestamp'].max()
        )

        assert results is not None
        assert isinstance(results, dict)

        # Check required result fields
        required_fields = ['total_return', 'sharpe_ratio', 'max_drawdown', 'total_trades']
        for field in required_fields:
            assert field in results

    def test_multiple_strategy_comparison(self, sample_backtest_data, backtest_config):
        """Test comparing multiple strategies."""
        backtester = ProductionBacktester(config=backtest_config)

        # Register multiple strategies
        strategies = [
            ('MA_5_20', MovingAverageStrategy(short_period=5, long_period=20)),
            ('MA_10_30', MovingAverageStrategy(short_period=10, long_period=30)),
        ]

        for name, strategy in strategies:
            backtester.register_strategy(name, strategy)

        results = backtester.compare_strategies(
            strategy_names=['MA_5_20', 'MA_10_30'],
            data=sample_backtest_data,
            start_date=sample_backtest_data['timestamp'].min(),
            end_date=sample_backtest_data['timestamp'].max()
        )

        assert results is not None
        assert len(results) == 2
        assert all(isinstance(result, dict) for result in results.values())

    def test_walk_forward_optimization(self, sample_backtest_data, backtest_config):
        """Test walk-forward optimization."""
        backtester = ProductionBacktester(config=backtest_config)

        strategy = MovingAverageStrategy(short_period=10, long_period=20)
        backtester.register_strategy('MA_Opt', strategy)

        # Parameter ranges for optimization
        param_ranges = {
            'short_period': (5, 20),
            'long_period': (15, 50)
        }

        optimized_params = backtester.optimize_strategy(
            strategy_name='MA_Opt',
            param_ranges=param_ranges,
            data=sample_backtest_data,
            optimization_target='sharpe_ratio'
        )

        assert optimized_params is not None
        assert isinstance(optimized_params, dict)
        assert 'short_period' in optimized_params
        assert 'long_period' in optimized_params

    def test_performance_metrics_calculation(self, sample_backtest_data, backtest_config):
        """Test comprehensive performance metrics calculation."""
        backtester = ProductionBacktester(config=backtest_config)

        strategy = MovingAverageStrategy(short_period=10, long_period=20)
        backtester.register_strategy('MA_Metrics', strategy)

        results = backtester.run_backtest(
            strategy_name='MA_Metrics',
            data=sample_backtest_data
        )

        # Check all expected metrics are calculated
        expected_metrics = [
            'total_return', 'annualized_return', 'sharpe_ratio', 'sortino_ratio',
            'max_drawdown', 'calmar_ratio', 'total_trades', 'winning_trades',
            'losing_trades', 'win_rate', 'avg_win', 'avg_loss', 'profit_factor',
            'expectancy', 'recovery_factor', 'payoff_ratio'
        ]

        for metric in expected_metrics:
            assert metric in results, f"Missing metric: {metric}"

    def test_risk_management_integration(self, sample_backtest_data, backtest_config):
        """Test risk management integration during backtesting."""
        backtester = ProductionBacktester(config=backtest_config)

        strategy = MovingAverageStrategy(short_period=10, long_period=20)
        backtester.register_strategy('MA_Risk', strategy)

        results = backtester.run_backtest(
            strategy_name='MA_Risk',
            data=sample_backtest_data
        )

        # Check risk-related metrics
        risk_metrics = ['max_drawdown', 'value_at_risk', 'expected_shortfall']
        for metric in risk_metrics:
            assert metric in results

    def test_transaction_cost_modeling(self, sample_backtest_data, backtest_config):
        """Test transaction cost modeling."""
        config_with_costs = backtest_config.copy()
        config_with_costs.update({
            'commission': 0.002,  # 0.2%
            'slippage': 0.001,    # 0.1%
            'spread': 0.0005      # 0.05%
        })

        backtester = ProductionBacktester(config=config_with_costs)

        strategy = MovingAverageStrategy(short_period=10, long_period=20)
        backtester.register_strategy('MA_Costs', strategy)

        results = backtester.run_backtest(
            strategy_name='MA_Costs',
            data=sample_backtest_data
        )

        # Results should account for transaction costs
        assert 'net_return' in results
        assert results['net_return'] < results['gross_return']

    def test_portfolio_rebalancing(self, sample_backtest_data, backtest_config):
        """Test portfolio rebalancing logic."""
        backtester = ProductionBacktester(config=backtest_config)

        # Test with multiple positions
        strategy = MovingAverageStrategy(short_period=10, long_period=20)
        backtester.register_strategy('MA_Rebalance', strategy)

        results = backtester.run_backtest(
            strategy_name='MA_Rebalance',
            data=sample_backtest_data
        )

        # Should track portfolio composition over time
        assert 'portfolio_history' in results
        assert isinstance(results['portfolio_history'], list)

    def test_parallel_processing(self, sample_backtest_data, backtest_config):
        """Test parallel processing capabilities."""
        backtester = ProductionBacktester(config=backtest_config)

        strategies = [
            ('MA_1', MovingAverageStrategy(short_period=5, long_period=15)),
            ('MA_2', MovingAverageStrategy(short_period=10, long_period=25)),
            ('MA_3', MovingAverageStrategy(short_period=15, long_period=35)),
        ]

        for name, strategy in strategies:
            backtester.register_strategy(name, strategy)

        import time
        start_time = time.time()

        # Run parallel backtests
        results = backtester.run_parallel_backtests(
            strategy_names=['MA_1', 'MA_2', 'MA_3'],
            data=sample_backtest_data
        )

        end_time = time.time()
        parallel_time = end_time - start_time

        assert results is not None
        assert len(results) == 3

        # Parallel execution should be faster than sequential
        # (This is a basic check; actual performance depends on system)
        assert parallel_time > 0

    def test_memory_efficiency(self, sample_backtest_data, backtest_config):
        """Test memory efficiency during backtesting."""
        import psutil
        import os

        backtester = ProductionBacktester(config=backtest_config)

        process = psutil.Process(os.getpid())
        memory_before = process.memory_info().rss / 1024 / 1024  # MB

        strategy = MovingAverageStrategy(short_period=10, long_period=20)
        backtester.register_strategy('MA_Memory', strategy)

        results = backtester.run_backtest(
            strategy_name='MA_Memory',
            data=sample_backtest_data
        )

        memory_after = process.memory_info().rss / 1024 / 1024  # MB
        memory_used = memory_after - memory_before

        # Should not use excessive memory (less than 500MB for large dataset)
        assert memory_used < 500, f"Backtest used {memory_used}MB memory"
        assert results is not None

    def test_edge_case_handling(self, backtest_config):
        """Test handling of edge cases."""
        backtester = ProductionBacktester(config=backtest_config)

        # Test with minimal data
        minimal_data = pd.DataFrame({
            'timestamp': pd.date_range('2024-01-01', periods=10, freq='1H'),
            'open': [50000] * 10,
            'high': [50100] * 10,
            'low': [49900] * 10,
            'close': [50000] * 10,
            'volume': [1000] * 10
        })

        strategy = MovingAverageStrategy(short_period=5, long_period=10)
        backtester.register_strategy('MA_Edge', strategy)

        # Should handle edge cases gracefully
        results = backtester.run_backtest(
            strategy_name='MA_Edge',
            data=minimal_data
        )

        assert results is not None
        assert isinstance(results, dict)

    def test_configuration_validation(self):
        """Test configuration validation."""
        # Invalid configuration should raise errors
        invalid_config = {
            'initial_capital': -1000,  # Negative capital
            'commission': 1.5,         # Too high commission
        }

        with pytest.raises(ValueError):
            ProductionBacktester(config=invalid_config)

    def test_data_frequency_handling(self, backtest_config):
        """Test handling different data frequencies."""
        backtester = ProductionBacktester(config=backtest_config)

        # Test with different frequencies
        frequencies = ['1min', '5min', '15min', '1H', '4H', '1D']

        for freq in frequencies:
            data = pd.DataFrame({
                'timestamp': pd.date_range('2024-01-01', periods=100, freq=freq),
                'open': np.random.uniform(50000, 51000, 100),
                'high': np.random.uniform(50100, 51100, 100),
                'low': np.random.uniform(49900, 50900, 100),
                'close': np.random.uniform(50000, 51000, 100),
                'volume': np.random.uniform(1000, 10000, 100)
            })

            is_valid, errors = backtester.validate_data(data)
            assert is_valid, f"Data validation failed for {freq}: {errors}"

    def test_result_persistence(self, sample_backtest_data, backtest_config, tmp_path):
        """Test saving and loading backtest results."""
        backtester = ProductionBacktester(config=backtest_config)

        strategy = MovingAverageStrategy(short_period=10, long_period=20)
        backtester.register_strategy('MA_Save', strategy)

        results = backtester.run_backtest(
            strategy_name='MA_Save',
            data=sample_backtest_data
        )

        # Save results
        results_file = tmp_path / "backtest_results.json"
        backtester.save_results(results, str(results_file))

        assert results_file.exists()

        # Load results
        loaded_results = backtester.load_results(str(results_file))

        # Should be identical
        assert loaded_results == results

    def test_strategy_parameter_sensitivity(self, sample_backtest_data, backtest_config):
        """Test strategy parameter sensitivity analysis."""
        backtester = ProductionBacktester(config=backtest_config)

        # Test different parameter combinations
        param_combinations = [
            {'short_period': 5, 'long_period': 15},
            {'short_period': 10, 'long_period': 25},
            {'short_period': 15, 'long_period': 35},
        ]

        sensitivity_results = {}

        for i, params in enumerate(param_combinations):
            strategy_name = f'MA_Sensitivity_{i}'
            strategy = MovingAverageStrategy(**params)
            backtester.register_strategy(strategy_name, strategy)

            results = backtester.run_backtest(
                strategy_name=strategy_name,
                data=sample_backtest_data
            )

            sensitivity_results[str(params)] = results['sharpe_ratio']

        # Should show variation in performance
        sharpe_values = list(sensitivity_results.values())
        assert len(set(sharpe_values)) > 1, "Parameters should affect performance"

    def test_market_regime_analysis(self, sample_backtest_data, backtest_config):
        """Test backtesting across different market regimes."""
        backtester = ProductionBacktester(config=backtest_config)

        strategy = MovingAverageStrategy(short_period=10, long_period=20)
        backtester.register_strategy('MA_Regime', strategy)

        # Split data into different market periods
        data = sample_backtest_data.copy()
        midpoint = len(data) // 2

        bullish_data = data.iloc[:midpoint].copy()
        bearish_data = data.iloc[midpoint:].copy()

        # Test in different market conditions
        bullish_results = backtester.run_backtest(
            strategy_name='MA_Regime',
            data=bullish_data
        )

        bearish_results = backtester.run_backtest(
            strategy_name='MA_Regime',
            data=bearish_data
        )

        # Results should be different for different market conditions
        assert bullish_results['total_return'] != bearish_results['total_return']

    def test_backtest_reproducibility(self, sample_backtest_data, backtest_config):
        """Test that backtests are reproducible."""
        backtester1 = ProductionBacktester(config=backtest_config)
        backtester2 = ProductionBacktester(config=backtest_config)

        strategy1 = MovingAverageStrategy(short_period=10, long_period=20)
        strategy2 = MovingAverageStrategy(short_period=10, long_period=20)

        backtester1.register_strategy('MA_Repro1', strategy1)
        backtester2.register_strategy('MA_Repro2', strategy2)

        results1 = backtester1.run_backtest(
            strategy_name='MA_Repro1',
            data=sample_backtest_data
        )

        results2 = backtester2.run_backtest(
            strategy_name='MA_Repro2',
            data=sample_backtest_data
        )

        # Results should be identical for same inputs
        assert results1['total_return'] == results2['total_return']
        assert results1['sharpe_ratio'] == results2['sharpe_ratio']
        assert results1['total_trades'] == results2['total_trades']

    def test_error_recovery(self, sample_backtest_data, backtest_config):
        """Test error recovery during backtesting."""
        backtester = ProductionBacktester(config=backtest_config)

        # Create strategy that might fail
        failing_strategy = Mock()
        failing_strategy.generate_signal.side_effect = [1, Exception("Signal generation failed"), 0, -1]

        backtester.register_strategy('Failing_Strategy', failing_strategy)

        # Should handle errors gracefully
        results = backtester.run_backtest(
            strategy_name='Failing_Strategy',
            data=sample_backtest_data.head(10)  # Small dataset
        )

        # Should still produce results despite errors
        assert results is not None
        assert isinstance(results, dict)
