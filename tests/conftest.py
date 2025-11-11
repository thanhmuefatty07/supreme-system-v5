#!/usr/bin/env python3
"""
Pytest configuration and shared fixtures for Supreme System V5 testing.

Based on pytest best practices and industry standards [pytest documentation].
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))


@pytest.fixture(scope="session")
def sample_ohlcv_data():
    """
    Generate realistic OHLCV test data.

    Based on financial data generation best practices.
    """
    np.random.seed(42)  # For reproducible tests

    # Generate 100 periods of realistic price data
    n_periods = 100
    base_price = 100.0

    # Simulate realistic price movements with trends and volatility
    trend = np.linspace(0, 0.1, n_periods)  # Slight upward trend
    noise = np.random.normal(0, 0.02, n_periods)  # 2% daily volatility
    gaps = np.random.choice([0, 0.02, -0.02], n_periods, p=[0.94, 0.03, 0.03])

    # Generate price series
    price_changes = trend + noise + gaps
    prices = base_price * np.cumprod(1 + price_changes)

    # Ensure positive prices
    prices = np.maximum(prices, 1.0)

    # Create OHLCV data with realistic spreads
    data = pd.DataFrame({
        'timestamp': pd.date_range('2024-01-01', periods=n_periods, freq='1H'),
        'open': prices,
        'high': prices * np.random.uniform(1.001, 1.008, n_periods),
        'low': prices * np.random.uniform(0.992, 0.999, n_periods),
        'close': prices,
        'volume': np.random.uniform(1000, 10000, n_periods)
    })

    # Ensure OHLC relationships are correct
    data['high'] = np.maximum(data[['open', 'close']].max(axis=1), data['high'])
    data['low'] = np.minimum(data[['open', 'close']].min(axis=1), data['low'])

    return data


@pytest.fixture(scope="session")
def large_ohlcv_data():
    """
    Generate larger dataset for performance testing.
    """
    np.random.seed(123)
    n_periods = 1000

    # Generate more volatile data for testing
    prices = 100 * np.cumprod(1 + np.random.normal(0.001, 0.03, n_periods))

    data = pd.DataFrame({
        'timestamp': pd.date_range('2024-01-01', periods=n_periods, freq='1min'),
        'open': prices,
        'high': prices * np.random.uniform(1.0005, 1.005, n_periods),
        'low': prices * np.random.uniform(0.995, 0.9995, n_periods),
        'close': prices,
        'volume': np.random.uniform(100, 1000, n_periods)
    })

    return data


@pytest.fixture
def mock_binance_client(mocker):
    """
    Mock Binance client for testing without API calls.
    """
    mock_client = mocker.MagicMock()
    mock_client.get_historical_klines.return_value = [
        [1640995200000, '100.0', '105.0', '95.0', '102.0', '1000.0'],
        [1641081600000, '102.0', '108.0', '98.0', '105.0', '1200.0']
    ]
    mock_client.test_connection.return_value = True
    return mock_client


@pytest.fixture
def mock_websocket_client(mocker):
    """
    Mock WebSocket client for testing.
    """
    mock_ws = mocker.MagicMock()
    mock_ws.is_connected = True
    mock_ws.active_streams = {'btcusdt@ticker', 'ethusdt@trade'}
    return mock_ws


@pytest.fixture
def sample_portfolio_state():
    """
    Sample portfolio state for testing risk management.
    """
    return {
        'cash': 9500.0,
        'positions': {
            'ETHUSDT': {
                'quantity': 10.0,
                'entry_price': 100.0,
                'current_price': 105.0
            },
            'BTCUSDT': {
                'quantity': 2.0,
                'entry_price': 200.0,
                'current_price': 210.0
            }
        },
        'total_value': 10000.0
    }


@pytest.fixture
def mock_risk_manager(mocker, sample_portfolio_state):
    """
    Mock risk manager for testing.
    """
    mock_rm = mocker.MagicMock()
    mock_rm.assess_trade_risk.return_value = {
        'approved': True,
        'risk_score': 0.3,
        'recommended_size': 5.0,
        'warnings': [],
        'reasons': ['Trade approved']
    }
    mock_rm.calculate_position_size.return_value = 5.0
    return mock_rm


@pytest.fixture(autouse=True)
def reset_random_seed():
    """
    Reset random seed before each test for reproducibility.
    """
    np.random.seed(42)


# Pytest configuration
def pytest_configure(config):
    """Configure pytest with custom markers."""
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line(
        "markers", "integration: marks tests as integration tests"
    )
    config.addinivalue_line(
        "markers", "e2e: marks tests as end-to-end tests"
    )


# Custom pytest hooks for performance monitoring
@pytest.fixture(autouse=True)
def performance_monitor(request):
    """Monitor test performance."""
    import time
    start_time = time.time()

    def fin():
        duration = time.time() - start_time
        if duration > 1.0:  # Log slow tests
            print(f"\nSLOW TEST: {request.node.name} took {duration:.2f}s")

    request.addfinalizer(fin)

