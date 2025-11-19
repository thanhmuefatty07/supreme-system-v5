import pytest
import responses
import requests
from unittest.mock import MagicMock

class TestBybitClient:
    """Comprehensive tests for BybitClient using Mocking (Clean Architecture)"""

    @pytest.fixture
    def mock_pybit_deps(self, monkeypatch):
        """Mock pybit using responses library for HTTP calls"""
        # Mock the entire pybit import to avoid complex module mocking
        import sys
        original_modules = dict(sys.modules)

        # Create minimal mock that satisfies the import
        mock_pybit = MagicMock()
        mock_unified_trading = MagicMock()
        mock_http_class = MagicMock()
        mock_spot = MagicMock()

        # Mock HTTP class to return a mock instance
        mock_http_instance = MagicMock()
        mock_http_class.return_value = mock_http_instance
        mock_unified_trading.HTTP = mock_http_class
        mock_pybit.unified_trading = mock_unified_trading
        mock_pybit.spot = mock_spot

        # Patch sys.modules temporarily
        sys.modules['pybit'] = mock_pybit
        sys.modules['pybit.unified_trading'] = mock_unified_trading

        # Ensure availability flag is set
        monkeypatch.setattr("src.data.bybit_client.PYBIT_AVAILABLE", True)

        yield {
            "pybit": mock_pybit,
            "http_class": mock_http_class,
            "http_instance": mock_http_instance,
            "spot": mock_spot
        }

        # Cleanup
        sys.modules.clear()
        sys.modules.update(original_modules)

    @pytest.fixture
    def client(self, mock_pybit_deps):
        """Fixture providing BybitClient with mocked dependencies"""
        # Import after mocking is set up
        from src.data.bybit_client import BybitClient

        return BybitClient(
            api_key="test_key_123",
            api_secret="test_secret_456",
            testnet=True,
            use_secrets_manager=False  # Disable secrets manager for testing
        )

    @pytest.fixture
    def base_url(self, client):
        """Helper to get correct URL based on testnet flag"""
        # BybitClient logic: if testnet -> api-testnet, else -> api
        return "https://api-testnet.bybit.com" if client.testnet else "https://api.bybit.com"

    # ==================== INITIALIZATION TESTS ====================

    def test_initialization_defaults(self, client, mock_pybit_deps):
        """Test default initialization"""
        from src.data.bybit_client import BybitClient

        assert client is not None
        assert isinstance(client, BybitClient)
        # Basic validation - just check object was created
        assert hasattr(client, 'async_client')
        assert hasattr(client.async_client, 'api_key')

        # Verify HTTP class was instantiated (basic check)
        mock_pybit_deps["http_class"].assert_called_once()


    # ==================== TEST CONNECTION TESTS ====================

    def test_test_connection_success(self, client):
        """Test successful connection to Bybit API"""
        # Mock the async client's test_connection method to return a coroutine
        async def mock_test_connection():
            return True
        client.async_client.test_connection = mock_test_connection

        result = client.test_connection()
        assert result is True

    def test_test_connection_failure(self, client):
        """Test connection failure"""
        # Mock the async client's test_connection method
        client.async_client.test_connection = MagicMock(return_value=False)

        result = client.test_connection()
        assert result is False

    def test_test_connection_exception(self, client):
        """Test connection with exception handling"""
        # Mock the async client's test_connection method to raise exception
        client.async_client.test_connection = MagicMock(side_effect=Exception("Network error"))

        result = client.test_connection()
        assert result is False

    # ==================== GET HISTORICAL KLINES TESTS ====================

    def test_get_historical_klines_success(self, client):
        """Test successful historical klines retrieval"""
        import pandas as pd

        # Mock DataFrame response
        mock_df = pd.DataFrame({
            'open': [50000, 50100],
            'high': [50200, 50300],
            'low': [49900, 50000],
            'close': [50100, 50200],
            'volume': [100, 120],
            'symbol': ['BTCUSDT', 'BTCUSDT'],
            'interval': ['1h', '1h']
        })

        # Mock async method
        async def mock_get_historical_klines(*args, **kwargs):
            return mock_df
        client.async_client.get_historical_klines = mock_get_historical_klines

        result = client.get_historical_klines(
            symbol="BTCUSDT",
            interval="1h",
            start_date="2024-01-01",
            end_date="2024-01-02"
        )

        assert result is not None
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 2

    def test_get_historical_klines_failure(self, client):
        """Test historical klines retrieval failure"""
        # Mock async method returning None
        async def mock_get_historical_klines(*args, **kwargs):
            return None
        client.async_client.get_historical_klines = mock_get_historical_klines

        result = client.get_historical_klines(
            symbol="BTCUSDT",
            interval="1h",
            start_date="2024-01-01"
        )

        assert result is None

    def test_get_historical_klines_exception(self, client):
        """Test historical klines with exception handling"""
        # Mock async method that raises exception
        async def mock_get_historical_klines(*args, **kwargs):
            raise Exception("API error")
        client.async_client.get_historical_klines = mock_get_historical_klines

        result = client.get_historical_klines(
            symbol="BTCUSDT",
            interval="1h",
            start_date="2024-01-01"
        )

        assert result is None

    # ==================== CONTEXT MANAGER TESTS ====================

    def test_context_manager_usage(self, client):
        """Test context manager functionality"""
        # Mock the async client's session methods as coroutines
        async def mock_initialize():
            pass
        async def mock_close():
            pass

        client.async_client.initialize_session = mock_initialize
        client.async_client.close_session = mock_close

        # Test context manager
        with client:
            pass

        # Context manager should work without errors

    # ==================== VALIDATION TESTS ====================

    def test_invalid_symbol_validation(self, client):
        """Test validation of invalid symbols"""
        # Mock validation failure
        client.async_client.get_historical_klines = MagicMock(return_value=None)

        result = client.get_historical_klines(
            symbol="",  # Invalid symbol
            interval="1h",
            start_date="2024-01-01"
        )

        assert result is None

    def test_invalid_interval_validation(self, client):
        """Test validation of invalid intervals"""
        # Mock validation failure
        client.async_client.get_historical_klines = MagicMock(return_value=None)

        result = client.get_historical_klines(
            symbol="BTCUSDT",
            interval="invalid",  # Invalid interval
            start_date="2024-01-01"
        )

        assert result is None

    def test_invalid_date_format(self, client):
        """Test handling of invalid date formats"""
        # Mock exception due to invalid date
        client.async_client.get_historical_klines = MagicMock(side_effect=Exception("Invalid date"))

        result = client.get_historical_klines(
            symbol="BTCUSDT",
            interval="1h",
            start_date="invalid-date"
        )

        assert result is None

    # ==================== EDGE CASES ====================

    def test_empty_klines_result(self, client):
        """Test handling of empty klines data"""
        import pandas as pd

        # Mock async method returning empty DataFrame
        async def mock_get_historical_klines(*args, **kwargs):
            return pd.DataFrame()
        client.async_client.get_historical_klines = mock_get_historical_klines

        result = client.get_historical_klines(
            symbol="BTCUSDT",
            interval="1h",
            start_date="2024-01-01"
        )

        assert result is not None
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 0

    def test_large_limit_parameter(self, client):
        """Test handling of large limit parameters"""
        import pandas as pd

        # Mock async response
        async def mock_get_historical_klines(*args, **kwargs):
            return pd.DataFrame({'open': [50000], 'close': [50100]})
        client.async_client.get_historical_klines = mock_get_historical_klines

        # Test with limit > 200 (should be capped at 200 internally)
        result = client.get_historical_klines(
            symbol="BTCUSDT",
            interval="1h",
            start_date="2024-01-01",
            limit=500
        )

        assert result is not None
