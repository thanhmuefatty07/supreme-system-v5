#!/usr/bin/env python3
"""
Comprehensive tests for LiveDataManager.

Tests cover:
- WebSocket connection management
- Auto-reconnect functionality
- Data stream configuration
- Message parsing and validation
- Error handling and recovery
- Callback system
"""

import pytest
import asyncio
import json
from unittest.mock import AsyncMock, MagicMock, patch
from collections import deque
from src.data.live_data_manager import LiveDataManager, StreamConfig, ConnectionStatus


class TestLiveDataManager:
    """Test suite for LiveDataManager."""

    @pytest.fixture
    def data_manager(self):
        """Create LiveDataManager instance."""
        config = {
            "reconnect_delay": 0.1,  # Fast for testing
            "max_reconnect_delay": 1.0,
            "max_reconnect_attempts": 3,
            "ping_interval": 1.0,
            "timeout": 1.0,
            "validate_data": True,
            "buffer_size": 10
        }
        return LiveDataManager(config)

    def test_initialization(self, data_manager):
        """Test LiveDataManager initialization."""
        assert data_manager.websocket is None
        assert len(data_manager.stream_configs) == 0
        assert len(data_manager.data_callbacks) == 0
        assert isinstance(data_manager.data_buffer, list)
        assert data_manager.running is False
        assert data_manager.config["reconnect_delay"] == 0.1

    def test_add_stream(self, data_manager):
        """Test adding data streams."""
        # Add a stream
        data_manager.add_stream("BTCUSDT", "1m", "kline")

        assert len(data_manager.stream_configs) == 1
        config = data_manager.stream_configs[0]
        assert config.symbol == "BTCUSDT"
        assert config.interval == "1m"
        assert config.stream_type == "kline"

    def test_add_multiple_streams(self, data_manager):
        """Test adding multiple data streams."""
        data_manager.add_stream("BTCUSDT", "1m")
        data_manager.add_stream("ETHUSDT", "5m")

        assert len(data_manager.stream_configs) == 2

    def test_build_websocket_url_single_stream(self, data_manager):
        """Test building WebSocket URL for single stream."""
        data_manager.add_stream("BTCUSDT", "1m", "kline")
        url = data_manager._build_websocket_url()
        expected = "wss://stream.binance.com:9443/ws/btcusdt@kline_1m"
        assert url == expected

    def test_build_websocket_url_multiple_streams(self, data_manager):
        """Test building WebSocket URL for multiple streams."""
        data_manager.add_stream("BTCUSDT", "1m", "kline")
        data_manager.add_stream("ETHUSDT", "1m", "kline")
        url = data_manager._build_websocket_url()
        expected = "wss://stream.binance.com:9443/ws/btcusdt@kline_1m/ethusdt@kline_1m"
        assert url == expected

    def test_build_websocket_url_different_types(self, data_manager):
        """Test building WebSocket URL for different stream types."""
        data_manager.add_stream("BTCUSDT", stream_type="ticker")
        data_manager.add_stream("ETHUSDT", stream_type="trade")
        url = data_manager._build_websocket_url()
        expected = "wss://stream.binance.com:9443/ws/btcusdt@ticker/ethusdt@trade"
        assert url == expected

    def test_build_websocket_url_no_streams(self, data_manager):
        """Test error when building URL with no streams."""
        with pytest.raises(ValueError, match="No streams configured"):
            data_manager._build_websocket_url()

    @pytest.mark.asyncio
    async def test_connection_failure(self, data_manager):
        """Test handling of connection failures."""
        # Mock websockets.connect to raise an exception
        with patch('websockets.connect', side_effect=Exception("Connection failed")):
            result = await data_manager.connect()
            assert result is False
            assert data_manager.connection_status.errors_count == 1

    def test_add_data_callback(self, data_manager):
        """Test adding data callbacks."""
        callback = lambda x: None
        data_manager.add_data_callback(callback)
        assert len(data_manager.data_callbacks) == 1
        assert data_manager.data_callbacks[0] == callback

    def test_parse_kline_message(self, data_manager):
        """Test parsing Binance kline WebSocket messages."""
        # Sample Binance kline message
        message = {
            "stream": "btcusdt@kline_1m",
            "data": {
                "e": "kline",
                "E": 1690000000000,
                "s": "BTCUSDT",
                "k": {
                    "t": 1690000000000,  # Kline start time
                    "T": 1690000059999,  # Kline close time
                    "s": "BTCUSDT",      # Symbol
                    "i": "1m",           # Interval
                    "f": 100,            # First trade ID
                    "L": 150,            # Last trade ID
                    "o": "50000.00",     # Open price
                    "c": "50100.00",     # Close price
                    "h": "50200.00",     # High price
                    "l": "49900.00",     # Low price
                    "v": "10.5",         # Base asset volume
                    "n": 50,             # Number of trades
                    "x": True,           # Is kline closed?
                    "q": "525000.00",    # Quote asset volume
                    "V": "5.5",          # Taker buy base asset volume
                    "Q": "275000.00",    # Taker buy quote asset volume
                    "B": "0"             # Ignore
                }
            }
        }

        result = data_manager._parse_kline_message(message["data"])

        assert result is not None
        assert result["symbol"] == "BTCUSDT"
        assert result["timestamp"] == 1690000000000
        assert result["open"] == 50000.00
        assert result["high"] == 50200.00
        assert result["low"] == 49900.00
        assert result["close"] == 50100.00
        assert result["volume"] == 10.5
        assert result["is_closed"] is True

    def test_parse_kline_message_invalid(self, data_manager):
        """Test parsing invalid kline messages."""
        # Message without kline data
        result = data_manager._parse_kline_message({})
        assert result is None

        # Message with invalid data
        invalid_message = {"k": {}}
        result = data_manager._parse_kline_message(invalid_message)
        assert result is None

    @pytest.mark.asyncio
    async def test_process_message(self, data_manager):
        """Test processing WebSocket messages."""
        # Mock callback
        callback_called = False
        callback_data = None

        def mock_callback(data):
            nonlocal callback_called, callback_data
            callback_called = True
            callback_data = data

        data_manager.add_data_callback(mock_callback)

        # Sample message
        message = json.dumps({
            "stream": "btcusdt@kline_1m",
            "data": {
                "e": "kline",
                "s": "BTCUSDT",
                "k": {
                    "t": 1690000000000,
                    "o": "50000.00",
                    "c": "50100.00",
                    "h": "50200.00",
                    "l": "49900.00",
                    "v": "10.5",
                    "x": True
                }
            }
        })

        await data_manager._process_message(message)

        assert callback_called
        assert callback_data is not None
        assert callback_data["symbol"] == "BTCUSDT"
        assert callback_data["close"] == 50100.00

    def test_buffer_management(self, data_manager):
        """Test data buffer management."""
        # Add data to buffer
        for i in range(15):  # More than buffer_size (10)
            data_manager.data_buffer.append({"test": i})

        # Buffer should be truncated to max size
        assert len(data_manager.data_buffer) == 10
        # Should keep most recent data
        assert data_manager.data_buffer[-1]["test"] == 14

    def test_get_latest_data(self, data_manager):
        """Test retrieving latest data for a symbol."""
        # Add test data
        data_manager.data_buffer = [
            {"symbol": "ETHUSDT", "close": 3000},
            {"symbol": "BTCUSDT", "close": 50000},
            {"symbol": "ETHUSDT", "close": 3100},
        ]

        # Get latest BTC data
        btc_data = data_manager.get_latest_data("BTCUSDT")
        assert btc_data["close"] == 50000

        # Get latest ETH data
        eth_data = data_manager.get_latest_data("ETHUSDT")
        assert eth_data["close"] == 3100

        # Non-existent symbol
        none_data = data_manager.get_latest_data("NONEXISTENT")
        assert none_data is None

    def test_get_buffered_data(self, data_manager):
        """Test retrieving buffered data."""
        # Add test data
        data_manager.data_buffer = [
            {"symbol": "BTCUSDT", "close": 50000},
            {"symbol": "ETHUSDT", "close": 3000},
            {"symbol": "BTCUSDT", "close": 50100},
        ]

        # Get all data
        all_data = data_manager.get_buffered_data()
        assert len(all_data) == 3

        # Get BTC data only
        btc_data = data_manager.get_buffered_data("BTCUSDT")
        assert len(btc_data) == 2
        assert all(d["symbol"] == "BTCUSDT" for d in btc_data)

        # Get limited data
        limited_data = data_manager.get_buffered_data(limit=2)
        assert len(limited_data) == 2

    def test_connection_status(self, data_manager):
        """Test connection status tracking."""
        status = data_manager.get_connection_status()

        assert isinstance(status, ConnectionStatus)
        assert status.connected is False
        assert status.reconnect_count == 0
        assert status.messages_received == 0
        assert status.errors_count == 0

    def test_get_data_status(self, data_manager):
        """Test data status retrieval."""
        # Add some test data
        data_manager.data_buffer = [{"test": "data"}]

        status = data_manager.get_data_status()

        assert "data_connected" in status
        assert "buffered_data_count" in status
        assert status["buffered_data_count"] == 1

    @pytest.mark.asyncio
    async def test_disconnect(self, data_manager):
        """Test disconnection."""
        # Mock websocket
        data_manager.websocket = AsyncMock()

        await data_manager.disconnect()

        assert data_manager.websocket is None
        assert data_manager.connection_status.connected is False

    def test_error_handling_in_parsing(self, data_manager):
        """Test error handling in message parsing."""
        # Invalid JSON should not crash
        result = asyncio.run(data_manager._process_message("invalid json"))
        # Should not raise exception

        # Invalid message structure
        result = asyncio.run(data_manager._process_message(json.dumps({"invalid": "message"})))
        # Should not raise exception
