#!/usr/bin/env python3
"""
Integration tests for LiveDataManager with LiveTradingEngineV2.

Tests cover:
- End-to-end data flow from WebSocket to trading decisions
- Callback integration between components
- Error handling in integrated system
- Performance and concurrency
"""

import pytest
import asyncio
import json
from unittest.mock import AsyncMock, MagicMock, patch
from src.trading.live_trading_engine_v2 import LiveTradingEngineV2
from src.strategies.mean_reversion import MeanReversionStrategy
from src.data.live_data_manager import LiveDataManager


class TestLiveDataIntegration:
    """Integration tests for live data and trading engine."""

    @pytest.fixture
    def mock_exchange(self):
        """Create mock exchange client."""
        exchange = AsyncMock()
        exchange.fetch_order_book = AsyncMock(return_value={
            'asks': [[100.0, 10.0]],
            'bids': [[99.0, 10.0]]
        })
        exchange.create_order = AsyncMock(return_value={'id': 'test_order_123'})
        return exchange

    @pytest.fixture
    def strategy_config(self):
        """Strategy configuration for testing."""
        return {
            'lookback_period': 5,  # Small for testing
            'entry_threshold': 1.0,
            'use_rsi': False,
            'min_signal_strength': 0.1,
            'buffer_size': 10
        }

    @pytest.fixture
    def engine_config(self):
        """Engine configuration for testing."""
        return {
            'initial_capital': 10000.0,
            'symbols': ['BTCUSDT'],
            'data_interval': '1m',
            'risk_config': {
                'max_risk_per_trade': 0.02,
                'kelly_mode': 'half',
                'daily_loss_limit': 0.05,
                'max_position_pct': 0.1,
                'max_portfolio_pct': 0.5
            },
            'data_config': {
                'validate_data': True,
                'buffer_size': 10,
                'reconnect_delay': 0.1,
                'timeout': 1.0
            },
            'trade_log_path': 'test_integration_trades.jsonl'
        }

    @pytest.fixture
    def strategy(self, strategy_config):
        """Create strategy instance."""
        return MeanReversionStrategy(strategy_config)

    @pytest.fixture
    def engine(self, mock_exchange, strategy, engine_config):
        """Create LiveTradingEngineV2 instance."""
        return LiveTradingEngineV2(mock_exchange, strategy, engine_config)

    def test_engine_initialization_with_data_manager(self, engine):
        """Test that engine initializes with LiveDataManager."""
        assert hasattr(engine, 'data_manager')
        assert isinstance(engine.data_manager, LiveDataManager)
        assert len(engine.data_manager.stream_configs) == 1
        assert engine.data_manager.stream_configs[0].symbol == 'BTCUSDT'
        assert engine.data_manager.stream_configs[0].interval == '1m'

    def test_data_callback_registration(self, engine):
        """Test that data callbacks are properly registered."""
        assert len(engine.data_manager.data_callbacks) == 1

        # The callback should be the engine's market data handler
        callback = engine.data_manager.data_callbacks[0]
        assert callable(callback)

    @pytest.mark.asyncio
    async def test_market_data_flow(self, engine):
        """Test complete market data flow from WebSocket to trading decision."""
        # Sample market data
        market_data = {
            "symbol": "BTCUSDT",
            "timestamp": int(asyncio.get_event_loop().time() * 1000),
            "open": 50000.0,
            "high": 50100.0,
            "low": 49900.0,
            "close": 50050.0,
            "volume": 10.5
        }

        # Process market data (this should not crash)
        try:
            result = await engine.on_market_update(market_data)
            # Result can be None (no signal) or a dict (trade executed)
            assert result is None or isinstance(result, dict)
        except Exception as e:
            # If it fails due to insufficient data, that's expected
            assert "insufficient" in str(e).lower() or "enough" in str(e).lower()

    def test_data_stream_configuration(self, engine, engine_config):
        """Test that data streams are configured correctly."""
        streams = engine.data_manager.stream_configs

        assert len(streams) == len(engine_config['symbols'])
        for stream in streams:
            assert stream.symbol in engine_config['symbols']
            assert stream.interval == engine_config['data_interval']
            assert stream.stream_type == 'kline'

    def test_get_data_status_integration(self, engine):
        """Test integrated data status reporting."""
        status = engine.get_data_status()

        # Should include data manager status
        assert 'data_connected' in status
        assert 'buffered_data_count' in status
        assert 'messages_received' in status
        assert 'reconnect_count' in status

        # Should include uptime if connected
        assert 'uptime_seconds' in status

    @pytest.mark.asyncio
    async def test_shutdown_integration(self, engine):
        """Test integrated shutdown process."""
        # Should be able to shutdown without errors
        await engine._shutdown()
        assert engine.is_running is False

    @pytest.mark.asyncio
    async def test_multiple_data_points(self, engine):
        """Test processing multiple data points in sequence."""
        # Create a series of market data points
        base_price = 50000.0
        data_points = []

        for i in range(10):
            price = base_price + (i - 5) * 100  # Prices around base
            data_points.append({
                "symbol": "BTCUSDT",
                "timestamp": int(asyncio.get_event_loop().time() * 1000) + i * 1000,
                "open": price - 50,
                "high": price + 50,
                "low": price - 50,
                "close": price,
                "volume": 10.0 + i
            })

        # Process each data point
        for data in data_points:
            try:
                result = await engine.on_market_update(data)
                # Should not crash, even if no signals generated
                assert result is None or isinstance(result, dict)
            except Exception as e:
                # Expected for initial data points with insufficient history
                assert "enough" in str(e).lower() or "insufficient" in str(e).lower()

    def test_buffered_data_access(self, engine):
        """Test accessing buffered data through the engine."""
        # Add some test data to the data manager buffer
        test_data = [
            {"symbol": "BTCUSDT", "close": 50000, "timestamp": 1000},
            {"symbol": "BTCUSDT", "close": 50100, "timestamp": 2000},
        ]
        engine.data_manager.data_buffer = test_data

        # Access through data manager
        buffered = engine.data_manager.get_buffered_data("BTCUSDT")
        assert len(buffered) == 2
        assert all(d["symbol"] == "BTCUSDT" for d in buffered)

    @pytest.mark.asyncio
    async def test_error_handling_integration(self, engine):
        """Test error handling in integrated system."""
        # Test with invalid data
        invalid_data = {
            "symbol": "BTCUSDT",
            "timestamp": "invalid_timestamp",  # Should cause validation error
            "close": 50000
        }

        # Should handle error gracefully
        result = await engine.on_market_update(invalid_data)
        assert result is None  # Should return None on validation error

    def test_configuration_persistence(self, engine, engine_config):
        """Test that configuration is properly applied."""
        # Check that data manager has correct config
        assert engine.data_manager.config["validate_data"] == engine_config["data_config"]["validate_data"]
        assert engine.data_manager.config["buffer_size"] == engine_config["data_config"]["buffer_size"]

        # Check that streams are configured
        assert len(engine.data_manager.stream_configs) > 0

    @pytest.mark.asyncio
    async def test_connection_status_integration(self, engine):
        """Test connection status integration."""
        # Initially should not be connected
        status = engine.get_data_status()
        assert status['data_connected'] is False
        assert status['reconnect_count'] == 0

        # Simulate some connection activity
        engine.data_manager.connection_status.messages_received = 5
        engine.data_manager.connection_status.errors_count = 1

        status = engine.get_data_status()
        assert status['messages_received'] == 5
        assert status['data_errors'] == 1

    def test_symbol_configuration(self):
        """Test engine with different symbol configurations."""
        configs = [
            {'symbols': ['BTCUSDT']},
            {'symbols': ['BTCUSDT', 'ETHUSDT']},
            {'symbols': ['BTCUSDT', 'ETHUSDT', 'ADAUSDT'], 'data_interval': '5m'}
        ]

        for config in configs:
            full_config = {
                'initial_capital': 10000.0,
                'symbols': config['symbols'],
                'data_interval': config.get('data_interval', '1m'),
                'risk_config': {'max_risk_per_trade': 0.02},
                'data_config': {'validate_data': True}
            }

            mock_exchange = AsyncMock()
            strategy = MeanReversionStrategy({'lookback_period': 5})
            engine = LiveTradingEngineV2(mock_exchange, strategy, full_config)

            assert len(engine.data_manager.stream_configs) == len(config['symbols'])
            for stream in engine.data_manager.stream_configs:
                assert stream.symbol in config['symbols']
                assert stream.interval == config.get('data_interval', '1m')

    @pytest.mark.asyncio
    async def test_data_callback_execution(self, engine):
        """Test that data callbacks are executed properly."""
        callback_executed = False
        received_data = None

        # Add a test callback
        def test_callback(data):
            nonlocal callback_executed, received_data
            callback_executed = True
            received_data = data

        engine.data_manager.add_data_callback(test_callback)

        # Simulate receiving data
        test_data = {
            "symbol": "BTCUSDT",
            "timestamp": int(asyncio.get_event_loop().time() * 1000),
            "close": 50000.0,
            "open": 49900.0,
            "high": 50100.0,
            "low": 49800.0,
            "volume": 10.5
        }

        # Manually trigger callback (simulating WebSocket message processing)
        await engine.data_manager.data_callbacks[1](test_data)  # Index 1 because engine adds its own callback at 0

        assert callback_executed
        assert received_data == test_data
