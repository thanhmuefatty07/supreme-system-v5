import pytest
import responses
from unittest.mock import MagicMock
from src.data.bybit_client import BybitClient

class TestBybitClient:
    """Comprehensive tests for BybitClient using Clean Mocking"""

    @pytest.fixture
    def client(self, monkeypatch):
        """Fixture for BybitClient with mocked dependencies"""
        # Mock HTTP class inside pybit module
        mock_http = MagicMock()
        monkeypatch.setattr("src.data.bybit_client.pybit.HTTP", mock_http)

        return BybitClient(
            api_key="test_key_123",
            api_secret="test_secret_456",
            testnet=True
        )

    def test_initialization(self, client):
        assert client is not None
        assert client.api_key == "test_key_123"
        assert client.testnet is True
