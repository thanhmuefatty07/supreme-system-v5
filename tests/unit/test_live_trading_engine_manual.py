#!/usr/bin/env python3
"""
Comprehensive Manual Tests for Live Trading Engine

These tests focus on thorough coverage of the LiveTradingEngine functionality.
Goal: Achieve 60%+ coverage for the trading engine module.
"""

import pytest
import asyncio
import threading
from datetime import datetime, timedelta
from decimal import Decimal
from unittest.mock import Mock, patch, MagicMock, AsyncMock
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

try:
    from trading.live_trading_engine import (
        LiveTradingEngine,
        LiveTradingPosition,
        OrderExecutionError,
        InsufficientFundsError
    )
    from risk.risk_manager import RiskManager
    from strategies.base_strategy import BaseStrategy
    from data.binance_client import BinanceClient
except ImportError:
    # Skip tests if imports fail
    pytest.skip("Required modules not available", allow_module_level=True)


class TestLiveTradingPosition:
    """Comprehensive tests for LiveTradingPosition class."""

    def test_position_initialization_long(self):
        """Test LONG position initialization."""
        pos = LiveTradingPosition(
            symbol="BTCUSDT",
            side="LONG",
            quantity=1.0,
            entry_price=50000.0
        )

        assert pos.symbol == "BTCUSDT"
        assert pos.side == "LONG"
        assert pos.quantity == 1.0
        assert pos.entry_price == 50000.0
        assert pos.current_price == 50000.0
        assert pos.unrealized_pnl == 0.0
        assert pos.realized_pnl == 0.0
        assert pos.status == 'OPEN'
        assert pos.stop_loss is None
        assert pos.take_profit is None

    def test_position_initialization_short(self):
        """Test SHORT position initialization."""
        pos = LiveTradingPosition(
            symbol="ETHUSDT",
            side="SHORT",
            quantity=10.0,
            entry_price=3000.0
        )

        assert pos.symbol == "ETHUSDT"
        assert pos.side == "SHORT"
        assert pos.quantity == 10.0
        assert pos.entry_price == 3000.0

    def test_update_pnl_long_profitable(self):
        """Test P&L update for profitable LONG position."""
        pos = LiveTradingPosition("BTCUSDT", "LONG", 1.0, 50000.0)

        # Price goes up
        pnl = pos.update_pnl(55000.0)

        assert pnl == 5000.0  # (55000 - 50000) * 1.0
        assert pos.unrealized_pnl == 5000.0
        assert pos.current_price == 55000.0

    def test_update_pnl_long_loss(self):
        """Test P&L update for losing LONG position."""
        pos = LiveTradingPosition("BTCUSDT", "LONG", 2.0, 50000.0)

        # Price goes down
        pnl = pos.update_pnl(45000.0)

        assert pnl == -10000.0  # (45000 - 50000) * 2.0
        assert pos.unrealized_pnl == -10000.0

    def test_update_pnl_short_profitable(self):
        """Test P&L update for profitable SHORT position."""
        pos = LiveTradingPosition("BTCUSDT", "SHORT", 1.0, 50000.0)

        # Price goes down (profitable for short)
        pnl = pos.update_pnl(45000.0)

        assert pnl == 5000.0  # (50000 - 45000) * 1.0
        assert pos.unrealized_pnl == 5000.0

    def test_update_pnl_short_loss(self):
        """Test P&L update for losing SHORT position."""
        pos = LiveTradingPosition("BTCUSDT", "SHORT", 1.0, 50000.0)

        # Price goes up (loss for short)
        pnl = pos.update_pnl(55000.0)

        assert pnl == -5000.0  # (50000 - 55000) * 1.0
        assert pos.unrealized_pnl == -5000.0

    def test_set_stop_loss(self):
        """Test setting stop loss."""
        pos = LiveTradingPosition("BTCUSDT", "LONG", 1.0, 50000.0)

        pos.stop_loss = 48000.0
        assert pos.stop_loss == 48000.0

    def test_set_take_profit(self):
        """Test setting take profit."""
        pos = LiveTradingPosition("BTCUSDT", "LONG", 1.0, 50000.0)

        pos.take_profit = 55000.0
        assert pos.take_profit == 55000.0

    def test_position_status_change(self):
        """Test position status changes."""
        pos = LiveTradingPosition("BTCUSDT", "LONG", 1.0, 50000.0)

        assert pos.status == 'OPEN'

        pos.status = 'CLOSED'
        assert pos.status == 'CLOSED'

    def test_realized_pnl_update(self):
        """Test realized P&L tracking."""
        pos = LiveTradingPosition("BTCUSDT", "LONG", 1.0, 50000.0)

        pos.realized_pnl = 1000.0
        assert pos.realized_pnl == 1000.0

        # Update again (should accumulate)
        pos.realized_pnl += 500.0
        assert pos.realized_pnl == 1500.0

    def test_position_with_custom_timestamp(self):
        """Test position with custom timestamp."""
        custom_time = datetime(2024, 1, 1, 12, 0, 0)
        pos = LiveTradingPosition(
            "BTCUSDT", "LONG", 1.0, 50000.0,
            timestamp=custom_time
        )

        assert pos.timestamp == custom_time


class TestLiveTradingEngine:
    """Comprehensive tests for LiveTradingEngine class."""

    @pytest.fixture
    def mock_binance_client(self):
        """Mock Binance client for testing."""
        client = Mock(spec=BinanceClient)
        client.get_account_balance = AsyncMock(return_value=10000.0)
        client.get_symbol_price = AsyncMock(return_value=50000.0)
        client.place_market_order = AsyncMock(return_value={
            'order_id': '12345',
            'status': 'FILLED',
            'executed_qty': '1.00000000',
            'price': '50000.00'
        })
        return client

    @pytest.fixture
    def mock_risk_manager(self):
        """Mock risk manager for testing."""
        rm = Mock(spec=RiskManager)
        rm.assess_trade_risk = Mock(return_value={'approved': True, 'max_quantity': 2.0})
        rm.update_portfolio_risk = Mock()
        return rm

    @pytest.fixture
    def mock_strategy(self):
        """Mock trading strategy for testing."""
        strategy = Mock(spec=BaseStrategy)
        strategy.generate_signals = Mock(return_value=[])
        strategy.name = "TestStrategy"
        return strategy

    @pytest.fixture
    def engine(self, mock_binance_client, mock_risk_manager, mock_strategy):
        """Create LiveTradingEngine instance with mocks."""
        with patch('trading.live_trading_engine.get_config') as mock_config:
            mock_config.return_value = {
                'trading': {
                    'max_position_size': 2.0,
                    'risk_per_trade': 0.02,
                    'max_open_positions': 5
                },
                'risk': {
                    'max_drawdown': 0.1,
                    'daily_loss_limit': 1000.0
                }
            }

            engine = LiveTradingEngine(
                binance_client=mock_binance_client,
                risk_manager=mock_risk_manager,
                strategies=[mock_strategy]
            )
            return engine

    def test_engine_initialization(self, engine, mock_binance_client, mock_risk_manager):
        """Test engine initialization."""
        assert engine.binance_client == mock_binance_client
        assert engine.risk_manager == mock_risk_manager
        assert len(engine.strategies) == 1
        assert len(engine.positions) == 0
        assert engine.is_running == False

    def test_get_account_balance(self, engine, mock_binance_client):
        """Test getting account balance."""
        balance = asyncio.run(engine.get_account_balance())
        assert balance == 10000.0
        mock_binance_client.get_account_balance.assert_called_once()

    def test_get_current_price(self, engine, mock_binance_client):
        """Test getting current price."""
        price = asyncio.run(engine.get_current_price("BTCUSDT"))
        assert price == 50000.0
        mock_binance_client.get_symbol_price.assert_called_once_with("BTCUSDT")

    def test_calculate_position_size_valid(self, engine):
        """Test position size calculation with valid parameters."""
        account_balance = 10000.0
        entry_price = 50000.0
        risk_per_trade = 0.02  # 2%
        stop_loss_pct = 0.02   # 2%

        position_size = engine.calculate_position_size(
            account_balance, entry_price, risk_per_trade, stop_loss_pct
        )

        # Expected: (10000 * 0.02) / (50000 * 0.02) = 200 / 1000 = 0.2
        expected_size = 0.2
        assert abs(position_size - expected_size) < 0.001

    def test_calculate_position_size_edge_cases(self, engine):
        """Test position size calculation edge cases."""
        # Zero balance
        size = engine.calculate_position_size(0, 50000, 0.02, 0.02)
        assert size == 0

        # Zero price
        size = engine.calculate_position_size(10000, 0, 0.02, 0.02)
        assert size == 0

        # Zero stop loss
        size = engine.calculate_position_size(10000, 50000, 0.02, 0)
        assert size == 0

    def test_validate_order_parameters_valid(self, engine):
        """Test order parameter validation with valid inputs."""
        is_valid, error = engine.validate_order_parameters(
            symbol="BTCUSDT",
            side="BUY",
            quantity=1.0,
            price=50000.0
        )

        assert is_valid == True
        assert error is None

    def test_validate_order_parameters_invalid_symbol(self, engine):
        """Test order validation with invalid symbol."""
        is_valid, error = engine.validate_order_parameters(
            symbol="", side="BUY", quantity=1.0, price=50000.0
        )

        assert is_valid == False
        assert "symbol" in error.lower()

    def test_validate_order_parameters_invalid_side(self, engine):
        """Test order validation with invalid side."""
        is_valid, error = engine.validate_order_parameters(
            symbol="BTCUSDT", side="INVALID", quantity=1.0, price=50000.0
        )

        assert is_valid == False
        assert "side" in error.lower()

    def test_validate_order_parameters_invalid_quantity(self, engine):
        """Test order validation with invalid quantity."""
        is_valid, error = engine.validate_order_parameters(
            symbol="BTCUSDT", side="BUY", quantity=0, price=50000.0
        )

        assert is_valid == False
        assert "quantity" in error.lower()

        is_valid, error = engine.validate_order_parameters(
            symbol="BTCUSDT", side="BUY", quantity=-1.0, price=50000.0
        )

        assert is_valid == False
        assert "quantity" in error.lower()

    def test_validate_order_parameters_invalid_price(self, engine):
        """Test order validation with invalid price."""
        is_valid, error = engine.validate_order_parameters(
            symbol="BTCUSDT", side="BUY", quantity=1.0, price=0
        )

        assert is_valid == False
        assert "price" in error.lower()

    @patch('asyncio.sleep', new_callable=AsyncMock)
    def test_execute_market_order_success(self, mock_sleep, engine, mock_binance_client, mock_risk_manager):
        """Test successful market order execution."""
        # Execute order
        result = asyncio.run(engine.execute_market_order(
            symbol="BTCUSDT",
            side="BUY",
            quantity=1.0
        ))

        # Verify result
        assert result['order_id'] == '12345'
        assert result['status'] == 'FILLED'

        # Verify calls
        mock_binance_client.place_market_order.assert_called_once_with(
            symbol="BTCUSDT",
            side="BUY",
            quantity=1.0
        )
        mock_risk_manager.update_portfolio_risk.assert_called_once()

    @patch('asyncio.sleep', new_callable=AsyncMock)
    def test_execute_market_order_failure(self, mock_sleep, engine, mock_binance_client):
        """Test market order execution failure."""
        # Mock order failure
        mock_binance_client.place_market_order.side_effect = Exception("API Error")

        # Execute order (should raise)
        with pytest.raises(OrderExecutionError):
            asyncio.run(engine.execute_market_order(
                symbol="BTCUSDT",
                side="BUY",
                quantity=1.0
            ))

    def test_get_open_positions(self, engine):
        """Test getting open positions."""
        # Initially empty
        positions = engine.get_open_positions()
        assert len(positions) == 0

        # Add a position manually (for testing)
        pos = LiveTradingPosition("BTCUSDT", "LONG", 1.0, 50000.0)
        engine.positions["BTCUSDT"] = pos

        positions = engine.get_open_positions()
        assert len(positions) == 1
        assert positions[0].symbol == "BTCUSDT"

    def test_get_position_by_symbol(self, engine):
        """Test getting position by symbol."""
        # Position doesn't exist
        pos = engine.get_position_by_symbol("BTCUSDT")
        assert pos is None

        # Add position
        test_pos = LiveTradingPosition("BTCUSDT", "LONG", 1.0, 50000.0)
        engine.positions["BTCUSDT"] = test_pos

        pos = engine.get_position_by_symbol("BTCUSDT")
        assert pos is not None
        assert pos.symbol == "BTCUSDT"

    def test_calculate_portfolio_value(self, engine, mock_binance_client):
        """Test portfolio value calculation."""
        # Mock current prices
        mock_binance_client.get_symbol_price.side_effect = lambda symbol: {
            "BTCUSDT": 55000.0,
            "ETHUSDT": 3500.0
        }.get(symbol, 50000.0)

        # Add positions
        btc_pos = LiveTradingPosition("BTCUSDT", "LONG", 1.0, 50000.0)
        eth_pos = LiveTradingPosition("ETHUSDT", "SHORT", 10.0, 3000.0)

        engine.positions["BTCUSDT"] = btc_pos
        engine.positions["ETHUSDT"] = eth_pos

        # Calculate portfolio value
        btc_pos.update_pnl(55000.0)  # +5000
        eth_pos.update_pnl(3500.0)   # -5000 (short loss)

        portfolio_value = engine.calculate_portfolio_value()

        # Should be 5000 - 5000 = 0 (positions cancel out)
        assert portfolio_value == 0.0

    def test_close_position_full(self, engine, mock_binance_client):
        """Test closing position completely."""
        # Add position
        pos = LiveTradingPosition("BTCUSDT", "LONG", 1.0, 50000.0)
        pos.update_pnl(55000.0)  # Unrealized P&L: +5000
        engine.positions["BTCUSDT"] = pos

        # Mock closing order
        mock_binance_client.place_market_order.return_value = {
            'order_id': 'close_123',
            'status': 'FILLED',
            'executed_qty': '1.00000000',
            'price': '55000.00'
        }

        # Close position
        result = asyncio.run(engine.close_position("BTCUSDT"))

        # Verify position is closed
        assert pos.status == 'CLOSED'
        assert pos.realized_pnl == 5000.0

        # Verify order was placed
        mock_binance_client.place_market_order.assert_called_once_with(
            symbol="BTCUSDT",
            side="SELL",  # Close LONG with SELL
            quantity=1.0
        )

    def test_close_position_partial(self, engine, mock_binance_client):
        """Test closing position partially."""
        # Add position with larger size
        pos = LiveTradingPosition("BTCUSDT", "LONG", 5.0, 50000.0)
        engine.positions["BTCUSDT"] = pos

        # Mock partial close
        mock_binance_client.place_market_order.return_value = {
            'order_id': 'partial_close_123',
            'status': 'FILLED',
            'executed_qty': '2.00000000',
            'price': '55000.00'
        }

        # Close partial position
        result = asyncio.run(engine.close_position("BTCUSDT", quantity=2.0))

        # Verify position size reduced
        assert pos.quantity == 3.0  # 5.0 - 2.0
        assert pos.status == 'OPEN'  # Still open

    def test_update_positions_with_price_data(self, engine):
        """Test updating positions with new price data."""
        # Add positions
        btc_pos = LiveTradingPosition("BTCUSDT", "LONG", 1.0, 50000.0)
        eth_pos = LiveTradingPosition("ETHUSDT", "SHORT", 10.0, 3000.0)

        engine.positions["BTCUSDT"] = btc_pos
        engine.positions["ETHUSDT"] = eth_pos

        # Price update data
        price_data = {
            "BTCUSDT": 55000.0,  # BTC up
            "ETHUSDT": 3200.0    # ETH down (good for short)
        }

        # Update positions
        engine.update_positions(price_data)

        # Check P&L updates
        assert btc_pos.current_price == 55000.0
        assert btc_pos.unrealized_pnl == 5000.0  # (55000 - 50000) * 1.0

        assert eth_pos.current_price == 3200.0
        assert eth_pos.unrealized_pnl == -2000.0  # (3000 - 3200) * 10.0

    def test_check_stop_loss_take_profit(self, engine):
        """Test stop loss and take profit checking."""
        # Create position with SL and TP
        pos = LiveTradingPosition("BTCUSDT", "LONG", 1.0, 50000.0)
        pos.stop_loss = 48000.0   # 4% stop loss
        pos.take_profit = 55000.0  # 10% take profit

        engine.positions["BTCUSDT"] = pos

        # Test stop loss trigger
        should_close, reason = engine.check_stop_loss_take_profit(pos, 47000.0)
        assert should_close == True
        assert "stop loss" in reason.lower()

        # Reset position
        pos.stop_loss = 48000.0

        # Test take profit trigger
        should_close, reason = engine.check_stop_loss_take_profit(pos, 56000.0)
        assert should_close == True
        assert "take profit" in reason.lower()

        # Test no trigger
        should_close, reason = engine.check_stop_loss_take_profit(pos, 52000.0)
        assert should_close == False

    def test_get_portfolio_summary(self, engine):
        """Test portfolio summary generation."""
        # Add test positions
        btc_pos = LiveTradingPosition("BTCUSDT", "LONG", 1.0, 50000.0)
        eth_pos = LiveTradingPosition("ETHUSDT", "SHORT", 10.0, 3000.0)

        btc_pos.update_pnl(55000.0)  # +5000
        eth_pos.update_pnl(3200.0)   # -2000

        engine.positions["BTCUSDT"] = btc_pos
        engine.positions["ETHUSDT"] = eth_pos

        summary = engine.get_portfolio_summary()

        assert summary['total_positions'] == 2
        assert summary['open_positions'] == 2
        assert summary['total_unrealized_pnl'] == 3000.0  # 5000 - 2000
        assert summary['total_realized_pnl'] == 0.0

    def test_get_trading_statistics(self, engine):
        """Test trading statistics generation."""
        # Add some mock trading history
        engine.trade_history = [
            {'symbol': 'BTCUSDT', 'pnl': 1000.0, 'timestamp': datetime.now()},
            {'symbol': 'ETHUSDT', 'pnl': -500.0, 'timestamp': datetime.now()},
            {'symbol': 'BTCUSDT', 'pnl': 1500.0, 'timestamp': datetime.now()},
        ]

        stats = engine.get_trading_statistics()

        assert stats['total_trades'] == 3
        assert stats['winning_trades'] == 2
        assert stats['losing_trades'] == 1
        assert stats['win_rate'] == pytest.approx(66.67, abs=0.01)
        assert stats['total_pnl'] == 2000.0
        assert stats['average_win'] == 1250.0
        assert stats['average_loss'] == -500.0

    def test_engine_shutdown(self, engine):
        """Test engine shutdown process."""
        # Set engine as running
        engine.is_running = True

        # Shutdown
        engine.shutdown()

        # Verify shutdown state
        assert engine.is_running == False

    def test_max_open_positions_limit(self, engine):
        """Test maximum open positions limit."""
        # Add maximum allowed positions
        for i in range(5):  # max_open_positions = 5
            pos = LiveTradingPosition(f"TEST{i}USDT", "LONG", 1.0, 50000.0)
            engine.positions[f"TEST{i}USDT"] = pos

        # Try to add one more
        can_open = engine.can_open_position("EXTRAUSDT")
        assert can_open == False

    def test_risk_limits_enforcement(self, engine, mock_risk_manager):
        """Test that risk limits are enforced."""
        # Mock risk manager rejection
        mock_risk_manager.assess_trade_risk.return_value = {
            'approved': False,
            'reason': 'Risk limit exceeded'
        }

        # Try to open position
        with pytest.raises(Exception):  # Should raise due to risk rejection
            asyncio.run(engine.execute_market_order(
                symbol="BTCUSDT",
                side="BUY",
                quantity=1.0
            ))

    def test_error_handling_network_failure(self, engine, mock_binance_client):
        """Test error handling for network failures."""
        mock_binance_client.get_account_balance.side_effect = Exception("Network timeout")

        with pytest.raises(Exception):
            asyncio.run(engine.get_account_balance())

    def test_concurrent_order_execution(self, engine, mock_binance_client):
        """Test concurrent order execution handling."""
        # This would require more complex mocking for concurrent scenarios
        # For now, just verify the engine can handle basic concurrent calls

        async def execute_order():
            return await engine.execute_market_order("BTCUSDT", "BUY", 1.0)

        # Execute multiple orders concurrently
        results = asyncio.run(asyncio.gather(*[execute_order() for _ in range(3)]))

        assert len(results) == 3
        assert all(r['status'] == 'FILLED' for r in results)
