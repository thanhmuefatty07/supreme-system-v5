"""
Test configuration and common fixtures for Supreme System V5 tests.
"""
import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import sys
import os

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))


@pytest.fixture(scope="session")
def test_config():
    """Test configuration fixture"""
    return {
        'initial_capital': 100000,
        'symbols': ['AAPL', 'MSFT', 'GOOGL'],
        'risk_per_trade': 0.02
    }


@pytest.fixture(scope="function")
def sample_ohlcv_data():
    """Sample OHLCV data for testing"""
    dates = pd.date_range(end=datetime.now(), periods=100, freq='D')
    return pd.DataFrame({
        'timestamp': dates,
        'open': np.random.uniform(90, 110, 100),
        'high': np.random.uniform(100, 120, 100),
        'low': np.random.uniform(80, 100, 100),
        'close': np.random.uniform(90, 110, 100),
        'volume': np.random.randint(1000000, 5000000, 100)
    })


@pytest.fixture(scope="function")
def sample_market_data():
    """Sample market data with multiple symbols"""
    symbols = ['AAPL', 'MSFT', 'GOOGL']
    dates = pd.date_range(end=datetime.now(), periods=50, freq='D')

    data = {}
    for symbol in symbols:
        data[symbol] = pd.DataFrame({
            'timestamp': dates,
            'open': np.random.uniform(90, 110, 50),
            'high': np.random.uniform(100, 120, 50),
            'low': np.random.uniform(80, 100, 50),
            'close': np.random.uniform(90, 110, 50),
            'volume': np.random.randint(1000000, 5000000, 50)
        })

    return data


@pytest.fixture(scope="function")
def mock_binance_client():
    """Mock Binance client for testing"""
    class MockBinanceClient:
        def __init__(self, api_key=None, api_secret=None):
            self.api_key = api_key
            self.api_secret = api_secret

        def get_historical_klines(self, symbol, interval, start_str, end_str=None):
            """Mock historical data"""
            dates = pd.date_range(start=start_str, periods=100, freq='1h')
            return [
                [
                    int(d.timestamp() * 1000),  # Open time
                    '100.0',  # Open
                    '105.0',  # High
                    '95.0',   # Low
                    '102.0',  # Close
                    '1000000', # Volume
                    int((d + pd.Timedelta(hours=1)).timestamp() * 1000),  # Close time
                    '0',      # Quote asset volume
                    '1000',   # Number of trades
                    '0',      # Taker buy base asset volume
                    '0',      # Taker buy quote asset volume
                    '0'       # Unused field
                ]
                for d in dates
            ]

    return MockBinanceClient()


@pytest.fixture(scope="function")
def temp_data_dir(tmp_path):
    """Temporary directory for data storage tests"""
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    return data_dir