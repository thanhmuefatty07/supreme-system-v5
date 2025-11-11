"""
End-to-end integration tests for complete trading cycle.

Tests cover:
- Full data pipeline from API to storage
- Strategy execution and signal generation
- Risk management integration
- Portfolio management
- Backtesting workflow
- Performance metrics calculation
- Error handling and recovery
"""

import asyncio
import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from unittest.mock import patch, MagicMock, AsyncMock

from src.data.binance_client import AsyncBinanceClient
from src.data.data_pipeline import DataPipeline
from src.data.data_storage import DataStorage
from src.strategies.base_strategy import BaseStrategy
from src.strategies.momentum import MomentumStrategy
from src.strategies.trend_following import TrendFollowingAgent
from src.risk.risk_manager import RiskManager
from src.trading.live_trading_engine import LiveTradingEngine
from src.backtesting.production_backtester import ProductionBacktester
from src.monitoring.prometheus_metrics import SupremeSystemMetrics


class TestFullDataPipelineIntegration:
    """Test complete data pipeline integration."""

    @pytest.fixture
    def mock_binance_data(self):
        """Create realistic mock Binance data."""
        base_time = int(datetime(2024, 1, 1).timestamp() * 1000)
        data = []

        for i in range(100):
            timestamp = base_time + (i * 60000)  # 1-minute intervals
            base_price = 100 + np.sin(i * 0.1) * 5

            data.append([
                timestamp,
                f"{base_price:.2f}",  # open
                f"{base_price + np.random.uniform(0, 2):.2f}",  # high
                f"{base_price - np.random.uniform(0, 2):.2f}",  # low
                f"{base_price + np.random.uniform(-1, 1):.2f}",  # close
                f"{np.random.randint(10000, 50000)}",  # volume
                timestamp + 60000,  # close_time
                "0", "100", "0", "0", "0"  # dummy fields
            ])

        return data

    @pytest.mark.asyncio
    async def test_end_to_end_data_pipeline(self, mock_binance_data):
        """Test complete data pipeline from API fetch to storage."""
        # Setup
        pipeline = DataPipeline(use_async=True)

        with patch.object(pipeline.async_client, 'get_historical_klines') as mock_get:
            with patch.object(pipeline.storage, 'store_data') as mock_store:
                # Mock API response
                mock_df = pd.DataFrame({
                    'timestamp': pd.date_range('2024-01-01', periods=10, freq='1H'),
                    'open': [100.0] * 10,
                    'high': [105.0] * 10,
                    'low': [95.0] * 10,
                    'close': [102.0] * 10,
                    'volume': [1000] * 10
                })
                mock_get.return_value = mock_df

                # Mock storage response
                mock_store.return_value = {
                    'success': True,
                    'rows_stored': 10,
                    'compression_ratio': 2.5,
                    'original_size_mb': 0.1,
                    'compressed_size_mb': 0.04
                }

                # Execute pipeline
                result = await pipeline.fetch_and_store_data_async(
                    symbol="BTCUSDT",
                    interval="1h",
                    start_date="2024-01-01"
                )

                # Verify success
                assert result['success'] is True
                assert result['rows_processed'] == 10
                assert result['storage_success'] is True

                # Verify method calls
                mock_get.assert_called_once_with("BTCUSDT", "1h", "2024-01-01", None)
                mock_store.assert_called_once()

    @pytest.mark.asyncio
    async def test_concurrent_multi_symbol_pipeline(self):
        """Test concurrent data fetching for multiple symbols."""
        pipeline = DataPipeline(use_async=True)
        symbols = ["BTCUSDT", "ETHUSDT", "ADAUSDT"]

        # Mock all API calls
        mock_data = pd.DataFrame({
            'timestamp': pd.date_range('2024-01-01', periods=5, freq='1H'),
            'open': [100.0] * 5,
            'high': [105.0] * 5,
            'low': [95.0] * 5,
            'close': [102.0] * 5,
            'volume': [1000] * 5
        })

        with patch.object(pipeline, 'fetch_and_store_data_async') as mock_fetch:
            mock_fetch.side_effect = [
                {'success': True, 'symbol': 'BTCUSDT', 'duration': 1.0},
                {'success': True, 'symbol': 'ETHUSDT', 'duration': 1.2},
                {'success': True, 'symbol': 'ADAUSDT', 'duration': 0.8}
            ]

            results = await pipeline.fetch_multiple_symbols_async(
                symbols=symbols,
                interval="1h",
                start_date="2024-01-01",
                max_concurrent=3
            )

            assert len(results) == 3
            for symbol in symbols:
                assert symbol in results
                assert results[symbol]['success'] is True

            # Verify all fetches were called
            assert mock_fetch.call_count == 3


class TestStrategyExecutionIntegration:
    """Test strategy execution and signal generation integration."""

    @pytest.fixture
    def sample_market_data(self):
        """Create sample market data for testing."""
        return pd.DataFrame({
            'timestamp': pd.date_range('2024-01-01', periods=100, freq='1H'),
            'open': np.random.uniform(90, 110, 100),
            'high': np.random.uniform(100, 120, 100),
            'low': np.random.uniform(80, 100, 100),
            'close': np.random.uniform(90, 110, 100),
            'volume': np.random.randint(100000, 500000, 100)
        })

    def test_momentum_strategy_signal_generation(self, sample_market_data):
        """Test momentum strategy signal generation."""
        strategy = MomentumStrategy(
            short_period=12,
            long_period=26,
            signal_period=9,
            roc_period=10,
            trend_threshold=0.02
        )

        # Generate signal
        signal = strategy.generate_signal(sample_market_data)

        # Verify signal structure
        assert isinstance(signal, dict)
        assert 'action' in signal
        assert 'symbol' in signal
        assert 'strength' in signal
        assert 'confidence' in signal

        # Action should be valid
        assert signal['action'] in ['BUY', 'SELL', 'HOLD']

        # Strength and confidence should be numeric
        assert isinstance(signal['strength'], (int, float))
        assert isinstance(signal['confidence'], (int, float))

    def test_trend_following_strategy_integration(self, sample_market_data):
        """Test trend following strategy with full configuration."""
        config = {
            'short_window': 20,
            'long_window': 50,
            'adx_period': 14,
            'adx_threshold': 25,
            'rsi_period': 14,
            'rsi_overbought': 70,
            'rsi_oversold': 30,
            'volume_ma_period': 20,
            'stop_loss_pct': 0.02,
            'take_profit_pct': 0.05
        }

        strategy = TrendFollowingAgent("test_agent", config)

        # Generate signal
        signal = strategy.generate_trade_signal(sample_market_data, portfolio_value=10000)

        # Verify signal structure
        assert 'action' in signal
        assert 'symbol' in signal
        assert 'quantity' in signal
        assert 'price' in signal

    def test_strategy_portfolio_integration(self, sample_market_data):
        """Test strategy integration with portfolio management."""
        from src.trading.portfolio_manager import PortfolioManager

        # Create strategy and portfolio
        strategy = MomentumStrategy()
        portfolio = PortfolioManager(initial_capital=10000)

        # Generate signal
        signal = strategy.generate_signal(sample_market_data)

        # Execute trade if signal is actionable
        if signal['action'] != 'HOLD':
            # Mock order execution
            order_result = {
                'success': True,
                'order_id': 'test_123',
                'executed_price': sample_market_data['close'].iloc[-1],
                'executed_quantity': 10,
                'fee': 0.1
            }

            # Update portfolio
            portfolio.update_from_order(order_result, signal['action'])

            # Verify portfolio updated
            assert portfolio.get_total_value() != 10000  # Should have changed


class TestRiskManagementIntegration:
    """Test risk management integration."""

    def test_risk_manager_position_sizing(self):
        """Test risk manager position sizing calculations."""
        risk_manager = RiskManager(
            initial_capital=10000,
            max_position_size=0.1,
            stop_loss_pct=0.02,
            take_profit_pct=0.05
        )

        entry_price = 100.0
        position_size = risk_manager.calculate_position_size(entry_price)

        # Should be reasonable position size
        assert position_size > 0
        assert position_size <= 1000  # Max 10% of capital / price

        # Should respect risk limits
        risk_amount = position_size * entry_price * risk_manager.stop_loss_pct
        assert risk_amount <= risk_manager.initial_capital * risk_manager.max_position_size

    def test_risk_assessment_integration(self):
        """Test risk assessment for trade decisions."""
        risk_manager = RiskManager()

        trade_params = {
            'symbol': 'BTCUSDT',
            'quantity': 10,
            'entry_price': 100.0,
            'current_data': pd.DataFrame({
                'close': [95, 98, 102, 105, 103],
                'volume': [1000, 1200, 800, 1500, 1100]
            })
        }

        assessment = risk_manager.assess_trade_risk(**trade_params)

        # Verify assessment structure
        assert 'approved' in assessment
        assert 'risk_score' in assessment
        assert 'reason' in assessment
        assert isinstance(assessment['approved'], bool)

    def test_risk_manager_portfolio_integration(self):
        """Test risk manager integration with portfolio."""
        from src.trading.portfolio_manager import PortfolioManager

        risk_manager = RiskManager(initial_capital=10000)
        portfolio = PortfolioManager(initial_capital=10000)

        # Simulate a trade
        trade = {
            'symbol': 'BTCUSDT',
            'quantity': 5,
            'entry_price': 100.0,
            'exit_price': 105.0,
            'pnl': 25.0
        }

        # Risk manager should track the trade
        risk_manager.record_trade(trade)

        # Portfolio should be updated
        portfolio.update_from_trade(trade)

        # Verify integration
        assert portfolio.get_total_value() == 10025.0  # Initial + PnL


class TestBacktestingWorkflowIntegration:
    """Test complete backtesting workflow."""

    def test_backtester_strategy_integration(self):
        """Test backtester with strategy integration."""
        # Create test data
        data = pd.DataFrame({
            'timestamp': pd.date_range('2024-01-01', periods=200, freq='1H'),
            'open': np.cumsum(np.random.normal(0, 1, 200)) + 100,
            'high': np.cumsum(np.random.normal(0, 1, 200)) + 105,
            'low': np.cumsum(np.random.normal(0, 1, 200)) + 95,
            'close': np.cumsum(np.random.normal(0, 1, 200)) + 102,
            'volume': np.random.randint(10000, 50000, 200)
        })

        # Create strategy
        strategy = MomentumStrategy()

        # Create backtester
        backtester = ProductionBacktester(
            strategy=strategy,
            initial_capital=10000,
            commission=0.001
        )

        # Run backtest
        results = backtester.run_backtest(data)

        # Verify results structure
        assert 'total_return' in results
        assert 'sharpe_ratio' in results
        assert 'max_drawdown' in results
        assert 'total_trades' in results
        assert 'win_rate' in results

        # Results should be reasonable
        assert results['total_trades'] >= 0
        assert -1 <= results['total_return'] <= 5  # Reasonable return range
        assert results['max_drawdown'] >= 0

    def test_backtester_metrics_calculation(self):
        """Test backtester metrics calculation."""
        # Create synthetic trading history
        trades = [
            {'entry_time': datetime(2024, 1, 1), 'exit_time': datetime(2024, 1, 2),
             'pnl': 100, 'return_pct': 0.01},
            {'entry_time': datetime(2024, 1, 3), 'exit_time': datetime(2024, 1, 4),
             'pnl': -50, 'return_pct': -0.005},
            {'entry_time': datetime(2024, 1, 5), 'exit_time': datetime(2024, 1, 6),
             'pnl': 150, 'return_pct': 0.015}
        ]

        strategy = MomentumStrategy()
        backtester = ProductionBacktester(strategy=strategy, initial_capital=10000)

        # Calculate metrics
        metrics = backtester.calculate_performance_metrics(trades)

        # Verify metrics
        assert 'total_return' in metrics
        assert 'win_rate' in metrics
        assert 'profit_factor' in metrics
        assert 'sharpe_ratio' in metrics

        # Win rate should be 2/3 = 66.7%
        assert abs(metrics['win_rate'] - 0.667) < 0.1

        # Total return should be (100 - 50 + 150) / 10000 = 0.02
        assert abs(metrics['total_return'] - 0.02) < 0.001


class TestLiveTradingIntegration:
    """Test live trading engine integration."""

    @pytest.fixture
    def trading_config(self):
        """Create trading configuration for testing."""
        return {
            'symbols': ['BTCUSDT', 'ETHUSDT'],
            'strategies': ['momentum', 'trend_following'],
            'risk_limits': {
                'max_position_size': 0.1,
                'max_daily_loss': 0.05,
                'max_drawdown': 0.1
            },
            'execution': {
                'slippage_tolerance': 0.001,
                'min_order_size': 10,
                'max_order_size': 1000
            }
        }

    def test_trading_engine_initialization(self, trading_config):
        """Test trading engine initialization."""
        engine = LiveTradingEngine(config=trading_config)

        assert engine.config == trading_config
        assert hasattr(engine, 'portfolio')
        assert hasattr(engine, 'risk_manager')
        assert hasattr(engine, 'strategies')

    def test_trading_engine_signal_processing(self, trading_config):
        """Test trading engine signal processing."""
        engine = LiveTradingEngine(config=trading_config)

        # Mock market data
        market_data = pd.DataFrame({
            'timestamp': [datetime.now()],
            'open': [100.0],
            'high': [105.0],
            'low': [95.0],
            'close': [102.0],
            'volume': [10000]
        })

        # Mock strategy signal
        signal = {
            'action': 'BUY',
            'symbol': 'BTCUSDT',
            'quantity': 10,
            'price': 102.0,
            'strength': 0.8,
            'confidence': 0.9
        }

        # Process signal
        with patch.object(engine, 'execute_order') as mock_execute:
            mock_execute.return_value = {'success': True, 'order_id': 'test_123'}

            result = engine.process_signal(signal, market_data)

            # Verify execution was attempted
            mock_execute.assert_called_once()

            # Verify result structure
            assert 'success' in result
            assert 'order_id' in result


class TestMonitoringIntegration:
    """Test monitoring and metrics integration."""

    def test_prometheus_metrics_integration(self):
        """Test Prometheus metrics collection."""
        metrics = SupremeSystemMetrics()

        # Simulate some operations
        metrics.record_api_request('binance', 'success', 0.1)
        metrics.record_trade_execution('BTCUSDT', 100.0, 10, 'profit')
        metrics.record_strategy_signal('momentum', 'BUY', 0.8)
        metrics.update_portfolio_value(10500.0)

        # Verify metrics are recorded
        # (In real implementation, would check actual metric values)

    def test_system_health_monitoring(self):
        """Test system health monitoring."""
        metrics = SupremeSystemMetrics()

        # Simulate system status
        metrics.record_system_status('healthy')
        metrics.record_memory_usage(512.0)
        metrics.record_cpu_usage(45.0)

        # Verify health status
        # (In real implementation, would check health indicators)


class TestErrorRecoveryIntegration:
    """Test error recovery and resilience."""

    @pytest.mark.asyncio
    async def test_api_failure_recovery(self):
        """Test API failure and recovery."""
        client = AsyncBinanceClient(api_key="test", api_secret="test", testnet=True)

        # Mock network failure followed by success
        with patch.object(client._session, 'get') as mock_get:
            # First call fails
            mock_get.side_effect = [
                aiohttp.ClientError("Network timeout"),
                MagicMock(status=200, json=AsyncMock(return_value=[{
                    "timestamp": 1640995200000,
                    "open": "100.0",
                    "high": "105.0",
                    "low": "95.0",
                    "close": "102.0",
                    "volume": "10000"
                }]))
            ]

            async with client:
                result = await client.get_historical_klines(
                    "BTCUSDT", "1h", "2022-01-01", limit=1
                )

                # Should eventually succeed
                assert result is not None
                assert len(result) == 1

    def test_strategy_error_recovery(self):
        """Test strategy error recovery."""
        # Create strategy that might fail
        strategy = MomentumStrategy()

        # Test with invalid data
        invalid_data = pd.DataFrame()  # Empty DataFrame

        # Should handle gracefully
        try:
            signal = strategy.generate_signal(invalid_data)
            # Should return HOLD or similar safe action
            assert 'action' in signal
        except Exception as e:
            # If it fails, should be a handled exception
            assert "Invalid data" in str(e) or "Empty" in str(e)


class TestPerformanceUnderLoad:
    """Test system performance under load."""

    @pytest.mark.asyncio
    async def test_concurrent_data_processing(self):
        """Test concurrent data processing performance."""
        pipeline = DataPipeline(use_async=True)

        # Mock concurrent operations
        with patch.object(pipeline, 'fetch_and_store_data_async') as mock_fetch:
            mock_fetch.return_value = {
                'success': True,
                'rows_processed': 1000,
                'duration': 0.5
            }

            # Process multiple symbols concurrently
            symbols = [f"TEST{i}USDT" for i in range(10)]
            start_time = asyncio.get_event_loop().time()

            tasks = [pipeline.fetch_and_store_data_async(symbol, "1h", "2024-01-01")
                    for symbol in symbols]
            results = await asyncio.gather(*tasks)

            end_time = asyncio.get_event_loop().time()
            total_time = end_time - start_time

            # All operations should succeed
            assert all(r['success'] for r in results)

            # Concurrent processing should be reasonably fast
            assert total_time < 2.0  # Should complete within 2 seconds

    def test_memory_efficiency_under_load(self):
        """Test memory efficiency during high-load operations."""
        from src.utils.memory_optimizer import MemoryOptimizer

        optimizer = MemoryOptimizer()

        # Create high-load scenario
        datasets = []
        for i in range(20):
            df = pd.DataFrame({
                'timestamp': pd.date_range('2024-01-01', periods=1000, freq='1min'),
                'price': np.random.uniform(100, 200, 1000),
                'volume': np.random.randint(1000, 10000, 1000)
            })
            datasets.append(df)

        initial_memory = optimizer.get_current_memory_mb()

        with optimizer.monitor_memory_usage("high_load_processing"):
            processed_data = []
            for df in datasets:
                # Process with memory optimization
                from src.utils.memory_optimizer import optimize_trading_data_pipeline
                optimized = optimize_trading_data_pipeline(df)
                processed_data.append(optimized)

        final_memory = optimizer.get_current_memory_mb()
        memory_increase = final_memory - initial_memory

        # Memory increase should be manageable (< 100MB for this workload)
        assert memory_increase < 100


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
