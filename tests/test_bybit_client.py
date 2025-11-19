import pytest
import responses
import requests
import sys
from unittest.mock import MagicMock, patch

# --- MOCKING PYBIT MODULE (CRITICAL STEP) ---

# Mock pybit and its submodules before import BybitClient
mock_pybit = MagicMock()
mock_unified_trading = MagicMock()
mock_http = MagicMock()
mock_spot = MagicMock()

# Set up the mock structure
mock_pybit.HTTP = mock_http
mock_unified_trading.HTTP = mock_http
mock_pybit.spot = mock_spot

# Mock the modules in sys.modules
sys.modules["pybit"] = mock_pybit
sys.modules["pybit.unified_trading"] = mock_unified_trading

# Mock the PYBIT_AVAILABLE flag
mock_pybit_available = patch('src.data.bybit_client.PYBIT_AVAILABLE', True)
mock_pybit_available.start()

# --------------------------------------------

# Bây giờ mới import client
from src.data.bybit_client import BybitClient

class TestBybitClient:
    """Comprehensive tests for BybitClient using Mocking (Clean Architecture)"""

    @pytest.fixture
    def client(self):
        """Fixture providing BybitClient with mocked dependencies"""
        # Re-enforce mock for each test
        sys.modules["pybit"] = mock_pybit

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

    def test_initialization_defaults(self, client):
        """Test default initialization"""
        assert client is not None
        assert client.async_client.api_key == "test_key_123"
        assert client.async_client.api_secret == "test_secret_456"
        assert client.async_client.testnet is True
        # Verify pybit.HTTP was called (internal logic verification)
        mock_pybit.HTTP.assert_called()


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
