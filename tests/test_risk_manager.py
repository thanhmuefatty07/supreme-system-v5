#!/usr/bin/env python3
"""
Tests for risk management system
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# Add src to path for imports
src_path = Path(__file__).parent.parent / 'src'
sys.path.insert(0, str(src_path))

try:
    from src.risk.risk_manager import RiskManager
    from src.strategies.moving_average import MovingAverageStrategy
except ImportError:
    from risk.risk_manager import RiskManager
    from strategies.moving_average import MovingAverageStrategy


class TestRiskManager:
    """Test risk management functionality"""

    def test_initialization(self):
        """Test risk manager initialization"""
        rm = RiskManager(
            initial_capital=10000,
            max_position_size=0.1,
            stop_loss_pct=0.02,
            take_profit_pct=0.05
        )

        assert rm.initial_capital == 10000
        assert rm.current_capital == 10000
        assert rm.max_position_size == 0.1
        assert rm.stop_loss_pct == 0.02
        assert rm.take_profit_pct == 0.05
        assert rm.positions == []
        assert rm.trades == []

    def test_calculate_position_size(self):
        """Test position size calculation"""
        rm = RiskManager(initial_capital=10000, max_position_size=0.1)

        # 1% risk of 10000 = 100 risk amount (default risk_pct=0.01)
        # Position size = 100 / 100 = 1
        size = rm.calculate_position_size(100.0)
        assert size == 1.0

    def test_calculate_position_size_insufficient_capital(self):
        """Test position sizing with insufficient capital"""
        rm = RiskManager(initial_capital=500, max_position_size=0.1)

        # Risk amount = 500 * 0.01 = 5 (default risk_pct=0.01)
        # Position size = 5 / 100 = 0.05
        size = rm.calculate_position_size(100.0)
        assert size == 0.05  # Risk-based sizing

    def test_stop_loss_check_long_position(self):
        """Test stop loss for long position"""
        rm = RiskManager(stop_loss_pct=0.02)  # 2% stop loss

        # 2% loss: 100 -> 98
        assert rm.check_stop_loss(100.0, 98.0, True) is True

        # 1% loss: 100 -> 99 (below threshold)
        assert rm.check_stop_loss(100.0, 99.0, True) is False

        # No loss
        assert rm.check_stop_loss(100.0, 100.0, True) is False

    def test_stop_loss_check_short_position(self):
        """Test stop loss for short position"""
        rm = RiskManager(stop_loss_pct=0.02)  # 2% stop loss

        # 2% adverse move: short at 100, price goes to 102
        assert rm.check_stop_loss(100.0, 102.0, False) is True

        # 1% adverse move: short at 100, price goes to 101
        assert rm.check_stop_loss(100.0, 101.0, False) is False

    def test_take_profit_check_long_position(self):
        """Test take profit for long position"""
        rm = RiskManager(take_profit_pct=0.05)  # 5% take profit

        # 5% profit: 100 -> 105
        assert rm.check_take_profit(100.0, 105.0, True) is True

        # 3% profit: 100 -> 103 (below threshold)
        assert rm.check_take_profit(100.0, 103.0, True) is False

    def test_take_profit_check_short_position(self):
        """Test take profit for short position"""
        rm = RiskManager(take_profit_pct=0.05)  # 5% take profit

        # 5% profit: short at 100, price goes to 95
        assert rm.check_take_profit(100.0, 95.0, False) is True

        # 3% profit: short at 100, price goes to 97
        assert rm.check_take_profit(100.0, 97.0, False) is False

    def test_enter_position(self):
        """Test position entry"""
        rm = RiskManager(initial_capital=10000, max_position_size=0.1)

        # Enter long position
        rm._enter_position(1, 100.0, pd.Timestamp('2024-01-01'))

        assert len(rm.positions) == 1
        position = rm.positions[0]
        assert position['entry_price'] == 100.0
        assert position['size'] == 1.0  # 1% risk of capital / price
        assert position['is_long'] is True
        assert rm.current_capital < 10000  # Capital reduced by position cost + fees

    def test_exit_position_long_profit(self):
        """Test profitable position exit"""
        rm = RiskManager(initial_capital=10000)

        # Enter position
        rm._enter_position(1, 100.0, pd.Timestamp('2024-01-01'))
        initial_capital = rm.current_capital

        # Exit at profit (simulate take profit condition)
        position = rm.positions[0].copy()  # Copy before removal
        rm._exit_position(position, 110.0, pd.Timestamp('2024-01-02'), 'take_profit')
        rm.positions.pop(0)  # Manually remove since _exit_position doesn't do this

        assert len(rm.positions) == 0
        assert len(rm.trades) == 1
        assert rm.current_capital > initial_capital  # Should have profit

        trade = rm.trades[0]
        assert trade['pnl'] > 0  # Profitable trade
        assert trade['exit_reason'] == 'take_profit'

    def test_backtest_simple_scenario(self):
        """Test simple backtest scenario"""
        # Create simple uptrend data
        dates = pd.date_range('2024-01-01', periods=50, freq='D')
        prices = np.linspace(100, 120, 50)  # Steady uptrend
        data = pd.DataFrame({
            'timestamp': dates,
            'open': prices,
            'high': prices * 1.01,
            'low': prices * 0.99,
            'close': prices,
            'volume': np.random.uniform(1000, 10000, 50)
        })

        # Create strategy and risk manager
        strategy = MovingAverageStrategy(short_window=5, long_window=20)
        rm = RiskManager(initial_capital=10000)

        # Run backtest
        results = rm.run_backtest(data, strategy)

        # Verify results structure
        assert 'initial_capital' in results
        assert 'final_capital' in results
        assert 'total_return' in results
        assert 'total_trades' in results
        assert 'sharpe_ratio' in results
        assert 'max_drawdown' in results
        assert 'trades' in results

        # Basic sanity checks
        assert results['initial_capital'] == 10000
        assert results['final_capital'] >= 0  # Should not go negative
        assert isinstance(results['total_trades'], int)
        assert isinstance(results['trades'], list)

    def test_performance_metrics_calculation(self):
        """Test performance metrics calculation"""
        rm = RiskManager(initial_capital=10000)

        # Mock some trades
        rm.trades = [
            {'pnl': 100, 'total_fees': 10},
            {'pnl': -50, 'total_fees': 8},
            {'pnl': 200, 'total_fees': 15}
        ]

        # Mock capital history (steady growth)
        capital_history = [10000, 10100, 10050, 10250]

        # Create mock data
        data = pd.DataFrame({
            'timestamp': pd.date_range('2024-01-01', periods=10),
            'close': [100] * 10
        })

        results = rm._calculate_performance_metrics(capital_history, data)

        assert results['total_trades'] == 3
        assert results['winning_trades'] == 2
        assert results['losing_trades'] == 1
        assert results['win_rate'] == 2/3
        assert results['total_return'] == 0.025  # 10250/10000 - 1
        assert 'sharpe_ratio' in results
        assert 'max_drawdown' in results
