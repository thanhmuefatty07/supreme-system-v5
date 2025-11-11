#!/usr/bin/env python3
"""
Tests for Binance API client
"""

import pytest
import pandas as pd
from unittest.mock import Mock, patch

import sys
from pathlib import Path

# Add src to path for imports
src_path = Path(__file__).parent.parent / 'src'
sys.path.insert(0, str(src_path))

try:
    from data.binance_client import BinanceClient
except ImportError:
    from data.binance_client import BinanceClient


class TestBinanceClient:
    """Test Binance API client functionality"""

    def test_initialization_without_credentials(self):
        """Test client initialization without API credentials"""
        client = BinanceClient()
        assert client.client is None
        assert client.testnet is True

    def test_initialization_with_credentials(self):
        """Test client initialization with API credentials"""
        client = BinanceClient(
            api_key="test_key",
            api_secret="test_secret",
            testnet=False
        )
        assert client.api_key == "test_key"
        assert client.api_secret == "test_secret"
        assert client.testnet is False

    @patch('data.binance_client.Client')
    def test_test_connection_success(self, mock_client_class):
        """Test successful connection test"""
        # Setup mock
        mock_client = Mock()
        mock_client.ping.return_value = {}
        mock_client_class.return_value = mock_client

        # Test
        client = BinanceClient("key", "secret")
        result = client.test_connection()

        assert result is True
        mock_client.ping.assert_called_once()

    @patch('data.binance_client.Client')
    def test_test_connection_failure(self, mock_client_class):
        """Test connection failure"""
        # Setup mock to raise exception
        mock_client_class.side_effect = Exception("Connection failed")

        # Test
        client = BinanceClient("key", "secret")
        result = client.test_connection()

        assert result is False

    @patch('data.binance_client.Client')
    def test_get_historical_klines_success(self, mock_client_class):
        """Test successful historical data download"""
        # Setup mock data
        mock_klines = [
            [1640995200000, "100.0", "105.0", "95.0", "102.0", "1000.0",
             1640998800000, "100000.0", 100, "500.0", "50000.0", "0"]
        ]

        mock_client = Mock()
        mock_client.get_historical_klines.return_value = mock_klines
        mock_client_class.return_value = mock_client

        # Test
        client = BinanceClient("key", "secret")
        result = client.get_historical_klines(
            symbol="ETHUSDT",
            interval="1h",
            start_date="2024-01-01",
            end_date="2024-01-02"
        )

        assert result is not None
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 1
        assert 'timestamp' in result.columns
        assert 'open' in result.columns
        assert 'close' in result.columns
        assert result['close'].iloc[0] == 102.0

    @patch('data.binance_client.Client')
    def test_get_historical_klines_empty_data(self, mock_client_class):
        """Test handling of empty data response"""
        mock_client = Mock()
        mock_client.get_historical_klines.return_value = []
        mock_client_class.return_value = mock_client

        client = BinanceClient("key", "secret")
        result = client.get_historical_klines("ETHUSDT", "1h", "2024-01-01")

        assert result is not None
        assert len(result) == 0

    @patch('data.binance_client.Client')
    def test_get_historical_klines_api_error(self, mock_client_class):
        """Test API error handling"""
        mock_client = Mock()
        mock_client.get_historical_klines.side_effect = Exception("API Error")
        mock_client_class.return_value = mock_client

        client = BinanceClient("key", "secret")
        result = client.get_historical_klines("ETHUSDT", "1h", "2024-01-01")

        assert result is None

    def test_rate_limiting(self):
        """Test rate limiting functionality"""
        client = BinanceClient()

        # First call
        client._rate_limit_wait()
        first_time = client.last_request_time

        # Second call (should wait)
        import time
        time.sleep(0.01)  # Small delay
        client._rate_limit_wait()
        second_time = client.last_request_time

        # Should have waited at least rate_limit_delay
        assert second_time >= first_time + client.rate_limit_delay
