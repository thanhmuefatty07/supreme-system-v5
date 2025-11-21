#!/usr/bin/env python3
"""
Simple tests for Live Trading Engine V2 - Router Integration
"""

import pytest
import asyncio
from unittest.mock import MagicMock, AsyncMock, patch
from datetime import datetime

from src.trading.live_trading_engine_v2 import LiveTradingEngineV2
from src.strategies.sma_crossover import SMACrossover


class TestLiveTradingEngineV2:
    """Basic tests for LiveTradingEngineV2 with SmartRouter integration."""

    @pytest.fixture
    def engine(self):
        """Create LiveTradingEngineV2 instance with mocked dependencies."""
        # Mock exchange client
        mock_exchange = MagicMock()
        mock_exchange.fetch_order_book = AsyncMock(return_value={
            'asks': [[100.0, 10.0]],
            'bids': [[99.0, 10.0]]
        })
        mock_exchange.create_order = AsyncMock(return_value={'id': 'test_order_123'})

        # Mock strategy that returns a buy signal
        mock_strategy = MagicMock()
        mock_strategy.generate_signal = MagicMock(return_value=None)
        mock_strategy.name = "MockStrategy"

        # Create engine with test config
        config = {
            'initial_capital': 10000.0,
            'trade_log_path': 'test_trades_v2.jsonl',
            'update_interval': 60,
            'max_slippage': 0.01,
            'risk_config': {
                'max_risk_per_trade': 0.02,
                'kelly_mode': 'half',
                'daily_loss_limit': 0.05,
                'max_position_pct': 0.1,
                'max_portfolio_pct': 0.5
            }
        }

        engine = LiveTradingEngineV2(mock_exchange, mock_strategy, config)
        yield engine
        # Cleanup - could remove log file if needed

    @pytest.mark.asyncio
    async def test_engine_initialization(self, engine):
        """Test that engine initializes correctly with router."""
        assert engine.exchange is not None
        assert engine.strategy is not None
        assert hasattr(engine, 'router')
        assert engine.router is not None
        assert engine.is_running is False

    @pytest.mark.asyncio
    async def test_market_update_processing(self, engine):
        """Test market data processing through the engine."""
        # Mock strategy to return a buy signal
        signal_mock = MagicMock()
        signal_mock.symbol = 'BTC/USDT'
        signal_mock.side = 'buy'
        signal_mock.price = 100.0
        signal_mock.strength = 0.8

        engine.strategy.generate_signal = MagicMock(return_value=signal_mock)

        # Process market data
        market_data = {
            'symbol': 'BTC/USDT',
            'close': 100.0,
            'timestamp': int(datetime.now().timestamp() * 1000)  # Unix timestamp in milliseconds
        }

        result = await engine.on_market_update(market_data)

        # Verify signal was generated and processed
        engine.strategy.generate_signal.assert_called_once()
        assert 'status' in result

    @pytest.mark.asyncio
    async def test_no_signal_scenario(self, engine):
        """Test processing when no signal is generated."""
        # Strategy returns None (no signal)
        engine.strategy.generate_signal = MagicMock(return_value=None)

        market_data = {
            'symbol': 'ETH/USDT',
            'close': 2000.0,
            'timestamp': int(datetime.now().timestamp() * 1000)  # Unix timestamp in milliseconds
        }

        result = await engine.on_market_update(market_data)

        # Should return None when no signal is generated
        assert result is None

    def test_engine_configuration(self, engine):
        """Test that configuration is properly applied."""
        assert engine.config['initial_capital'] == 10000.0
        assert engine.config['trade_log_path'] == 'test_trades_v2.jsonl'
        assert hasattr(engine, 'risk_manager')
        assert hasattr(engine, 'router')
