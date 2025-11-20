"""
Comprehensive unit tests for async operations.

Tests cover:
- Async Binance client functionality
- Concurrent data fetching
- Rate limiting and retry logic
- Error handling in async context
- Performance benchmarking
- Connection pooling
"""

import asyncio
import pytest
import aiohttp
import pandas as pd
import numpy as np
from unittest.mock import patch, MagicMock, AsyncMock
from aioresponses import aioresponses

from src.data.binance_client import AsyncBinanceClient, ValidationResult
from src.data.data_pipeline import DataPipeline


class TestAsyncBinanceClient:
    """Test AsyncBinanceClient functionality."""

    @pytest.fixture
    def client(self):
        """Create test client instance."""
        return AsyncBinanceClient(
            api_key="test_key",
            api_secret="test_secret",
            testnet=True,
            max_concurrent_requests=5
        )

    @pytest.mark.asyncio
    async def test_initialization(self, client):
        """Test client initialization."""
        assert client.api_key == "test_key"
        assert client.api_secret == "test_secret"
        assert client.testnet is True
        assert client.max_concurrent_requests == 5

    @pytest.mark.asyncio
    async def test_context_manager(self, client):
        """Test async context manager."""
        async with client:
            assert client._session is not None

        # Session should be closed
        assert client._session is None

    @pytest.mark.asyncio
    async def test_test_connection_success(self, client):
        """Test successful connection test."""
        with aioresponses() as m:
            m.get(f"{client.base_url}/api/v3/ping", payload={})

            async with client:
                result = await client.test_connection()
                assert result is True
                assert client.request_count == 1

    @pytest.mark.asyncio
    async def test_test_connection_failure(self, client):
        """Test connection failure."""
        with aioresponses() as m:
            m.get(f"{client.base_url}/api/v3/ping", status=500)

            async with client:
                result = await client.test_connection()
                assert result is False
                assert client.error_count == 1

    @pytest.mark.asyncio
    async def test_get_historical_klines_success(self, client):
        """Test successful klines data fetching."""
        # Mock response data
        mock_klines = [
            [1640995200000, "100.0", "105.0", "95.0", "102.0", "10000.0",
             1640998800000, "0", "100", "0", "0", "0"]
        ]

        with aioresponses() as m:
            m.get(f"{client.base_url}/api/v3/klines", payload=mock_klines)

            async with client:
                result = await client.get_historical_klines(
                    symbol="BTCUSDT",
                    interval="1h",
                    start_date="2022-01-01",
                    limit=1
                )

                assert isinstance(result, pd.DataFrame)
                assert len(result) == 1
                assert list(result.columns) == ['timestamp', 'open', 'high', 'low', 'close', 'volume']
                assert client.request_count == 1

    @pytest.mark.asyncio
    async def test_get_historical_klines_validation_failure(self, client):
        """Test klines fetching with validation failure."""
        # Test with invalid symbol
        async with client:
            result = await client.get_historical_klines(
                symbol="",  # Invalid symbol
                interval="1h",
                start_date="2022-01-01"
            )

            assert result is None
            assert client.error_count == 0  # Validation happens before request

    @pytest.mark.asyncio
    async def test_get_historical_klines_rate_limit_retry(self, client):
        """Test rate limit handling and retry."""
        mock_klines = [
            [1640995200000, "100.0", "105.0", "95.0", "102.0", "10000.0",
             1640998800000, "0", "100", "0", "0", "0"]
        ]

        with aioresponses() as m:
            # First request gets rate limited
            m.get(f"{client.base_url}/api/v3/klines", status=429,
                  headers={'Retry-After': '1'})
            # Second request succeeds
            m.get(f"{client.base_url}/api/v3/klines", payload=mock_klines)

            async with client:
                result = await client.get_historical_klines(
                    symbol="BTCUSDT",
                    interval="1h",
                    start_date="2022-01-01",
                    limit=1
                )

                assert isinstance(result, pd.DataFrame)
                assert len(result) == 1
                # Should have made 2 requests (original + retry)
                assert client.request_count == 2

    @pytest.mark.asyncio
    async def test_get_multiple_symbols_concurrent(self, client):
        """Test concurrent fetching of multiple symbols."""
        symbols = ["BTCUSDT", "ETHUSDT"]
        mock_klines = [
            [1640995200000, "100.0", "105.0", "95.0", "102.0", "10000.0",
             1640998800000, "0", "100", "0", "0", "0"]
        ]

        with aioresponses() as m:
            # Mock both symbol requests
            for symbol in symbols:
                m.get(f"{client.base_url}/api/v3/klines",
                     payload=mock_klines)

            async with client:
                results = await client.get_multiple_symbols_data(
                    symbols=symbols,
                    interval="1h",
                    start_date="2022-01-01",
                    limit=1
                )

                assert len(results) == 2
                for symbol in symbols:
                    assert symbol in results
                    assert isinstance(results[symbol], pd.DataFrame)
                    assert len(results[symbol]) == 1

    @pytest.mark.asyncio
    async def test_get_symbol_info(self, client):
        """Test symbol info retrieval."""
        mock_exchange_info = {
            "symbols": [
                {
                    "symbol": "BTCUSDT",
                    "status": "TRADING",
                    "baseAsset": "BTC",
                    "quoteAsset": "USDT"
                }
            ]
        }

        with aioresponses() as m:
            m.get(f"{client.base_url}/api/v3/exchangeInfo", payload=mock_exchange_info)

            async with client:
                result = await client.get_symbol_info("BTCUSDT")

                assert result is not None
                assert result["symbol"] == "BTCUSDT"
                assert result["status"] == "TRADING"

    @pytest.mark.asyncio
    async def test_get_server_time(self, client):
        """Test server time retrieval."""
        mock_time = {"serverTime": 1640995200000}

        with aioresponses() as m:
            m.get(f"{client.base_url}/api/v3/time", payload=mock_time)

            async with client:
                result = await client.get_server_time()

                assert result == 1640995200000

    @pytest.mark.asyncio
    async def test_request_signing(self, client):
        """Test HMAC-SHA256 request signing."""
        # Test the signing method directly
        query_string = "symbol=BTCUSDT&timestamp=1640995200000"
        signature = client._create_signature(query_string)

        assert isinstance(signature, str)
        assert len(signature) == 64  # SHA256 produces 64 hex chars
        assert signature.isalnum()  # Should be hexadecimal

    def test_validation_methods(self, client):
        """Test input validation methods."""
        # Test query params validation
        valid_params = {
            'symbol': 'BTCUSDT',
            'interval': '1h',
            'start_date': '2022-01-01',
            'end_date': '2022-01-02',
            'limit': 100
        }

        result = client._validate_query_params(valid_params)
        assert result.valid is True

        # Test invalid params
        invalid_params = {
            'symbol': '',  # Invalid symbol
            'interval': '1h',
            'start_date': '2022-01-01',
            'limit': 100
        }

        result = client._validate_query_params(invalid_params)
        assert result.valid is False


class TestAsyncDataPipeline:
    """Test async data pipeline functionality."""

    @pytest.fixture
    def pipeline(self):
        """Create test pipeline instance."""
        return DataPipeline(use_async=True)

    @pytest.mark.asyncio
    async def test_async_pipeline_initialization(self, pipeline):
        """Test async pipeline initialization."""
        assert pipeline.use_async is True
        assert pipeline.async_client is not None
        assert pipeline.client is None

    @pytest.mark.asyncio
    async def test_fetch_and_store_data_async(self, pipeline):
        """Test async data fetching and storage."""
        # Mock the async client
        with patch.object(pipeline.async_client, 'get_historical_klines') as mock_get:
            # Create mock data
            mock_data = pd.DataFrame({
                'timestamp': pd.date_range('2024-01-01', periods=10, freq='1H'),
                'open': [100.0] * 10,
                'high': [105.0] * 10,
                'low': [95.0] * 10,
                'close': [102.0] * 10,
                'volume': [1000] * 10
            })
            mock_get.return_value = mock_data

            # Mock storage
            with patch.object(pipeline.storage, 'store_data') as mock_store:
                mock_store.return_value = {
                    'success': True,
                    'rows_stored': 10,
                    'compression_ratio': 2.5
                }

                result = await pipeline.fetch_and_store_data_async(
                    symbol="TESTUSDT",
                    interval="1h",
                    start_date="2024-01-01"
                )

                assert result['success'] is True
                assert result['rows_processed'] == 10
                assert result['storage_success'] is True
                mock_get.assert_called_once()
                mock_store.assert_called_once()

    @pytest.mark.asyncio
    async def test_fetch_multiple_symbols_async(self, pipeline):
        """Test concurrent multi-symbol fetching."""
        symbols = ["BTCUSDT", "ETHUSDT"]

        # Mock async operations
        with patch.object(pipeline, 'fetch_and_store_data_async') as mock_fetch:
            mock_fetch.side_effect = [
                {'success': True, 'symbol': 'BTCUSDT', 'duration': 1.0},
                {'success': True, 'symbol': 'ETHUSDT', 'duration': 1.5}
            ]

            results = await pipeline.fetch_multiple_symbols_async(
                symbols=symbols,
                interval="1h",
                start_date="2024-01-01",
                max_concurrent=2
            )

            assert len(results) == 2
            assert 'BTCUSDT' in results
            assert 'ETHUSDT' in results
            assert results['BTCUSDT']['success'] is True
            assert results['ETHUSDT']['success'] is True

    @pytest.mark.asyncio
    async def test_async_pipeline_error_handling(self, pipeline):
        """Test error handling in async pipeline."""
        with patch.object(pipeline.async_client, 'get_historical_klines') as mock_get:
            mock_get.side_effect = aiohttp.ClientError("Network error")

            result = await pipeline.fetch_and_store_data_async(
                symbol="TESTUSDT",
                interval="1h",
                start_date="2024-01-01"
            )

            assert result['success'] is False
            assert 'errors' in result
            assert len(result['errors']) > 0


class TestRateLimiting:
    """Test rate limiting functionality."""

    @pytest.fixture
    def client(self):
        """Create client with fast rate limiting for testing."""
        return AsyncBinanceClient(
            api_key="test_key",
            api_secret="test_secret",
            testnet=True,
            rate_limit_delay=0.01  # Very fast for testing
        )

    @pytest.mark.asyncio
    async def test_rate_limiting(self, client):
        """Test rate limiting between requests."""
        import time

        with aioresponses() as m:
            # Mock ping endpoint
            m.get(f"{client.base_url}/api/v3/ping", payload={})

            async with client:
                start_time = time.time()

                # Make multiple requests
                for i in range(3):
                    await client.test_connection()

                end_time = time.time()

                # Should have some delay between requests
                total_time = end_time - start_time
                min_expected_time = 3 * client.rate_limit_delay  # At least 3 delays

                assert total_time >= min_expected_time * 0.8  # Allow some tolerance
                assert client.request_count == 3

    @pytest.mark.asyncio
    async def test_concurrent_rate_limiting(self, client):
        """Test rate limiting with concurrent requests."""
        with aioresponses() as m:
            m.get(f"{client.base_url}/api/v3/ping", payload={})

            async with client:
                # Create concurrent requests
                tasks = []
                for i in range(5):
                    tasks.append(client.test_connection())

                # Execute concurrently
                results = await asyncio.gather(*tasks, return_exceptions=True)

                # All should succeed
                successful_results = [r for r in results if not isinstance(r, Exception)]
                assert len(successful_results) == 5

                # Total requests should equal concurrent requests
                assert client.request_count == 5


class TestAsyncErrorHandling:
    """Test error handling in async operations."""

    @pytest.fixture
    def client(self):
        """Create test client."""
        return AsyncBinanceClient(api_key="test", api_secret="test", testnet=True)

    @pytest.mark.asyncio
    async def test_timeout_handling(self, client):
        """Test timeout error handling."""
        with aioresponses() as m:
            # Simulate timeout
            m.get(f"{client.base_url}/api/v3/ping", exception=asyncio.TimeoutError())

            async with client:
                result = await client.test_connection()

                assert result is False
                assert client.error_count > 0

    @pytest.mark.asyncio
    async def test_network_error_handling(self, client):
        """Test network error handling."""
        with aioresponses() as m:
            m.get(f"{client.base_url}/api/v3/ping", exception=aiohttp.ClientError("Network failed"))

            async with client:
                result = await client.test_connection()

                assert result is False
                assert client.error_count > 0

    @pytest.mark.asyncio
    async def test_retry_logic(self, client):
        """Test retry logic with failures."""
        with aioresponses() as m:
            # First two attempts fail, third succeeds
            m.get(f"{client.base_url}/api/v3/ping", status=500)
            m.get(f"{client.base_url}/api/v3/ping", status=500)
            m.get(f"{client.base_url}/api/v3/ping", payload={})

            async with client:
                result = await client.test_connection()

                assert result is True
                # Should have made 3 requests (2 failures + 1 success)
                assert client.request_count == 3
                assert client.error_count == 2


class TestPerformanceBenchmarking:
    """Test async operation performance."""

    @pytest.fixture
    def client(self):
        """Create client for performance testing."""
        return AsyncBinanceClient(
            api_key="test",
            api_secret="test",
            testnet=True,
            rate_limit_delay=0.001  # Minimal delay for testing
        )

    @pytest.mark.asyncio
    async def test_concurrent_vs_sequential_performance(self, client):
        """Compare concurrent vs sequential performance."""
        import time

        symbols = ["BTCUSDT", "ETHUSDT", "ADAUSDT", "DOTUSDT"]
        mock_klines = [
            [1640995200000, "100.0", "105.0", "95.0", "102.0", "10000.0",
             1640998800000, "0", "100", "0", "0", "0"]
        ]

        with aioresponses() as m:
            # Mock all symbol requests
            for symbol in symbols:
                m.get(f"{client.base_url}/api/v3/klines", payload=mock_klines)

            async with client:
                # Test sequential fetching
                sequential_start = time.time()
                sequential_results = {}
                for symbol in symbols:
                    result = await client.get_historical_klines(
                        symbol, "1h", "2022-01-01", limit=1
                    )
                    sequential_results[symbol] = result
                sequential_time = time.time() - sequential_start

                # Reset client state
                client.request_count = 0

                # Test concurrent fetching
                concurrent_start = time.time()
                concurrent_results = await client.get_multiple_symbols_data(
                    symbols, "1h", "2022-01-01", limit=1
                )
                concurrent_time = time.time() - concurrent_start

                # Concurrent should be faster (allowing for some overhead)
                assert concurrent_time <= sequential_time * 1.5
                assert len(concurrent_results) == len(symbols)
                assert len(sequential_results) == len(symbols)


class TestAsyncIntegration:
    """Test integration of async components."""

    @pytest.mark.asyncio
    async def test_full_async_workflow(self):
        """Test complete async workflow."""
        # Create pipeline with async client
        pipeline = DataPipeline(use_async=True)

        # Mock all async operations
        with patch.object(pipeline.async_client, 'get_historical_klines') as mock_get:
            with patch.object(pipeline.storage, 'store_data') as mock_store:
                # Setup mocks
                mock_data = pd.DataFrame({
                    'timestamp': pd.date_range('2024-01-01', periods=100, freq='1H'),
                    'open': np.random.uniform(100, 110, 100),
                    'high': np.random.uniform(105, 115, 100),
                    'low': np.random.uniform(95, 105, 100),
                    'close': np.random.uniform(100, 110, 100),
                    'volume': np.random.randint(1000, 10000, 100)
                })
                mock_get.return_value = mock_data
                mock_store.return_value = {
                    'success': True,
                    'rows_stored': 100,
                    'compression_ratio': 3.2
                }

                # Execute workflow
                result = await pipeline.fetch_and_store_data_async(
                    symbol="BTCUSDT",
                    interval="1h",
                    start_date="2024-01-01"
                )

                # Verify success
                assert result['success'] is True
                assert result['rows_processed'] == 100
                assert result['storage_success'] is True

                # Verify method calls
                mock_get.assert_called_once()
                mock_store.assert_called_once()


class TestAsyncValidation:
    """Test validation in async context."""

    def test_pydantic_validation_integration(self):
        """Test Pydantic validation integration."""
        from src.data.data_validator import OHLCVDataPoint, ValidationResult

        # Test valid data
        valid_data = {
            'timestamp': pd.Timestamp('2024-01-01'),
            'open': 100.0,
            'high': 105.0,
            'low': 95.0,
            'close': 102.0,
            'volume': 1000.0
        }

        try:
            point = OHLCVDataPoint(**valid_data)
            assert point.close == 102.0
        except Exception as e:
            pytest.fail(f"Valid data should not raise exception: {e}")

        # Test invalid data
        invalid_data = {
            'timestamp': pd.Timestamp('2024-01-01'),
            'open': -100.0,  # Negative price should fail
            'high': 105.0,
            'low': 95.0,
            'close': 102.0,
            'volume': 1000.0
        }

        with pytest.raises(Exception):  # Should raise validation error
            OHLCVDataPoint(**invalid_data)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
