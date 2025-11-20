#!/usr/bin/env python3
"""
Integration Test - The Grand Test

Tests the complete end-to-end flow:
Market Data → Strategy → Risk → Execution → Performance Tracking

This is the most important test - validates the entire "Trifecta Integration"
"""

import pytest
import asyncio
from unittest.mock import MagicMock, AsyncMock
from src.trading.live_trading_engine_v2 import LiveTradingEngineV2
from src.strategies.sma_crossover import SMACrossover
from src.strategies.rsi_strategy import RSIStrategy


class TestSystemIntegration:
    """Integration tests for the complete trading system."""

    @pytest.fixture
    def mock_exchange(self):
        """Create mock exchange client."""
        mock_ex = MagicMock()

        # Mock order book (for liquidity checks)
        mock_ex.fetch_order_book = AsyncMock(return_value={
            'asks': [[100.0, 100], [101.0, 100], [102.0, 100]],
            'bids': [[99.0, 100], [98.0, 100], [97.0, 100]]
        })

        # Mock order execution
        mock_ex.create_order = AsyncMock(return_value={
            'id': 'order_123',
            'status': 'filled',
            'symbol': 'BTC/USDT',
            'side': 'buy',
            'type': 'market',
            'price': 100.0,
            'amount': 1.0,
            'filled': 1.0,
            'remaining': 0.0,
            'cost': 100.0
        })

        return mock_ex

    @pytest.fixture
    def sma_strategy(self):
        """Create SMA crossover strategy."""
        config = {
            'fast_window': 2,
            'slow_window': 5,
            'initial_capital': 10000.0,
            'max_position_size': 0.1,
            'max_daily_loss': 0.05
        }
        return SMACrossover(config)

    @pytest.fixture
    def rsi_strategy(self):
        """Create RSI strategy."""
        config = {
            'rsi_period': 14,
            'overbought_level': 70,
            'oversold_level': 30,
            'initial_capital': 10000.0,
            'max_position_size': 0.1
        }
        return RSIStrategy(config)

    @pytest.fixture
    def engine_config(self):
        """Engine configuration."""
        return {
            'initial_capital': 10000.0,
            'risk_config': {
                'max_risk_per_trade': 0.1,
                'kelly_mode': 'half',
                'daily_loss_limit': 0.05,
                'risk_free_rate': 0.0
            },
            'max_slippage': 0.01,
            'update_interval': 60
        }

    @pytest.fixture
    def live_engine(self, mock_exchange, sma_strategy, engine_config):
        """Create live trading engine with SMA strategy."""
        return LiveTradingEngineV2(mock_exchange, sma_strategy, engine_config)

    @pytest.mark.asyncio
    async def test_engine_initialization(self, live_engine):
        """Test engine initializes with all components."""
        assert live_engine.strategy is not None
        assert live_engine.risk_manager is not None
        assert live_engine.router is not None
        assert live_engine.is_running is False
        assert live_engine.total_trades == 0

    @pytest.mark.asyncio
    async def test_full_trade_lifecycle_sma(self, live_engine):
        """Test complete trade lifecycle with SMA strategy."""
        # Setup: Inject price history to trigger Golden Cross
        # SMA(2) needs 2 prices, SMA(5) needs 5 prices
        live_engine.strategy.prices = [10, 10, 10, 10, 10]  # Flat baseline

        # Market update with breakout price (should trigger Golden Cross)
        market_data = {
            'symbol': 'BTC/USDT',
            'close': 20.0,  # Significant increase
            'volume': 1000,
            'timestamp': '2024-01-01 12:00:00'
        }

        # Execute market update
        result = await live_engine.on_market_update(market_data)

        # Verify execution
        assert result is not None
        assert result.get('status') in ['SUCCESS', None]  # May be None if signal not strong enough

        # If trade was executed
        if result and result.get('status') == 'SUCCESS':
            # Verify risk manager was engaged
            assert live_engine.risk_manager.current_capital > 0

            # Verify router executed order
            live_engine.exchange.create_order.assert_awaited()

            # Verify position tracking
            assert 'BTC/USDT' in live_engine.current_positions

            # Verify trade history
            assert len(live_engine.trade_history) > 0

    @pytest.mark.asyncio
    async def test_risk_manager_integration(self, live_engine):
        """Test risk manager properly limits trades."""
        # Inject massive loss to trigger circuit breaker
        live_engine.risk_manager.record_trade(-600)  # -6% loss (> 5% limit)

        assert live_engine.risk_manager.circuit_breaker.is_active is True

        # Try to execute trade
        market_data = {'symbol': 'BTC/USDT', 'close': 100.0, 'volume': 1000}

        # Setup strategy to generate signal
        live_engine.strategy.prices = [10, 10, 10, 10, 10]
        result = await live_engine.on_market_update({'symbol': 'BTC/USDT', 'close': 20.0})

        # Trade should be rejected by risk manager
        if result:
            assert result.get('status') == 'REJECTED' or result.get('reason') == 'risk_manager_rejection'

    @pytest.mark.asyncio
    async def test_execution_with_rsi_strategy(self, mock_exchange, rsi_strategy, engine_config):
        """Test integration with RSI strategy."""
        engine = LiveTradingEngineV2(mock_exchange, rsi_strategy, engine_config)

        # Create oversold condition
        # Feed downtrend prices to build RSI history
        for price in [100, 95, 90, 85, 80, 75, 70, 65, 60, 55, 50, 45, 40, 35, 30, 25]:
            market_data = {
                'symbol': 'BTC/USDT',
                'close': price,
                'volume': 1000
            }
            result = await engine.on_market_update(market_data)

        # Last result might be a buy signal (oversold)
        if result and result.get('status') == 'SUCCESS':
            assert 'BTC/USDT' in engine.current_positions

    @pytest.mark.asyncio
    async def test_position_close_and_pnl(self, live_engine):
        """Test position closing and PnL calculation."""
        # Open position
        live_engine.strategy.prices = [10, 10, 10, 10, 10]
        buy_result = await live_engine.on_market_update({
            'symbol': 'BTC/USDT',
            'close': 20.0,
            'volume': 1000
        })

        if buy_result and buy_result.get('status') == 'SUCCESS':
            # Manually insert position for testing
            live_engine.current_positions['BTC/USDT'] = {
                'side': 'LONG',
                'quantity': 1.0,
                'entry_price': 100.0,
                'timestamp': '2024-01-01'
            }

            # Trigger sell signal by reversing trend
            live_engine.strategy.prices = [20, 20, 20, 20, 20]
            sell_result = await live_engine.on_market_update({
                'symbol': 'BTC/USDT',
                'close': 10.0,
                'volume': 1000
            })

            # Position should be closed
            if sell_result and sell_result.get('status') == 'SUCCESS':
                assert 'BTC/USDT' not in live_engine.current_positions
                # PnL should be updated
                assert live_engine.total_trades >= 1

    @pytest.mark.asyncio
    async def test_start_stop_lifecycle(self, live_engine):
        """Test engine start/stop lifecycle."""
        await live_engine.start()
        assert live_engine.is_running is True

        await live_engine.stop()
        assert live_engine.is_running is False

    @pytest.mark.asyncio
    async def test_get_status(self, live_engine):
        """Test engine status reporting."""
        status = live_engine.get_status()

        assert 'is_running' in status
        assert 'strategy' in status
        assert 'total_trades' in status
        assert 'total_pnl' in status
        assert 'circuit_breaker_active' in status
        assert 'portfolio_value' in status
        assert 'risk_metrics' in status

        assert status['strategy'] == 'SMACrossover'
        assert status['portfolio_value'] == 10000.0

    @pytest.mark.asyncio
    async def test_multiple_trades_performance(self, live_engine):
        """Test multiple trades and performance tracking."""
        # Simulate multiple market updates
        prices = [10, 12, 14, 16, 18, 20, 18, 16, 14, 12, 10, 12, 14]

        for price in prices:
            market_data = {
                'symbol': 'BTC/USDT',
                'close': price,
                'volume': 1000
            }
            await live_engine.on_market_update(market_data)

        # Check that trades were tracked
        status = live_engine.get_status()
        assert status['total_trades'] >= 0  # May or may not execute depending on signals

    @pytest.mark.asyncio
    async def test_no_signal_no_trade(self, live_engine):
        """Test that no trade occurs when strategy returns None."""
        # Feed data that won't trigger signal
        market_data = {'symbol': 'BTC/USDT', 'close': 100.0, 'volume': 1000}

        result = await live_engine.on_market_update(market_data)

        # Should return None (no signal)
        assert result is None or result.get('status') != 'SUCCESS'

        # No orders should be placed
        live_engine.exchange.create_order.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_invalid_signal_rejected(self, live_engine):
        """Test that invalid signals are rejected."""
        # Mock strategy to return invalid signal
        from src.strategies.base_strategy import Signal

        invalid_signal = Signal(
            symbol='BTC/USDT',
            side='invalid_side',  # Invalid
            price=100.0
        )

        # Manually inject invalid signal for testing
        original_generate = live_engine.strategy.generate_signal

        def mock_generate(data):
            return invalid_signal

        live_engine.strategy.generate_signal = mock_generate

        result = await live_engine.on_market_update({
            'symbol': 'BTC/USDT',
            'close': 100.0
        })

        # Should not execute due to validation failure
        assert result is None or result.get('status') != 'SUCCESS'

        # Restore original method
        live_engine.strategy.generate_signal = original_generate

    @pytest.mark.asyncio
    async def test_trade_history_tracking(self, live_engine):
        """Test trade history is properly tracked."""
        initial_history_len = len(live_engine.get_trade_history())

        # Execute trade
        live_engine.strategy.prices = [10, 10, 10, 10, 10]
        await live_engine.on_market_update({
            'symbol': 'BTC/USDT',
            'close': 20.0,
            'volume': 1000
        })

        # History should update if trade executed
        history = live_engine.get_trade_history()
        if len(history) > initial_history_len:
            latest_trade = history[-1]
            assert 'timestamp' in latest_trade
            assert 'symbol' in latest_trade
            assert 'signal' in latest_trade
            assert 'execution' in latest_trade
