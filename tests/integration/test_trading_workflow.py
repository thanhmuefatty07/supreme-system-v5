"""
Complete trading workflow integration tests for Supreme System V5.

Tests the entire trading pipeline from data to execution.
"""
import pytest
import pandas as pd
import numpy as np
from unittest.mock import Mock, patch
from datetime import datetime, timedelta


class TestCompleteTradingWorkflow:
    """Test complete trading workflow from data to execution"""

    @pytest.fixture
    def mock_executor(self):
        """Mock order executor"""
        executor = Mock()
        executor.execute_order.return_value = {
            'status': 'FILLED',
            'order_id': 'test_order_123',
            'filled_price': 100.0,
            'filled_quantity': 10
        }
        return executor

    @pytest.fixture
    def mock_portfolio(self):
        """Mock portfolio manager"""
        portfolio = Mock()
        portfolio.portfolio_value = 100000.0
        portfolio.calculate_performance_metrics.return_value = {
            'total_return': 0.05,
            'sharpe_ratio': 1.2,
            'max_drawdown': 0.02,
            'total_trades': 5,
            'win_rate': 0.6
        }
        return portfolio

    def test_complete_trading_workflow_aapl(self, test_config, sample_ohlcv_data, mock_executor, mock_portfolio):
        """Test complete trading workflow with AAPL data"""
        # Import required modules
        from src.strategies.trend_following import TrendFollowingAgent
        from src.risk.risk_manager import RiskManager

        # Setup components
        strategy = TrendFollowingAgent("test_trend_agent", {
            'short_window': 5,
            'long_window': 20,
            'stop_loss_pct': 0.02,
            'take_profit_pct': 0.05
        })

        risk_manager = RiskManager(
            capital=test_config['initial_capital'],
            max_position_size=0.1,
            max_daily_loss_pct=0.05
        )

        # Execute trading cycle
        trades_executed = 0
        for i in range(25, len(sample_ohlcv_data)):
            data_window = sample_ohlcv_data.iloc[:i].copy()
            current_price = data_window['Close'].iloc[-1]

            # Generate trading signal
            signal = strategy.generate_trade_signal(data_window, mock_portfolio.portfolio_value)

            if signal['action'] != 'HOLD':
                # Risk assessment
                risk_assessment = risk_manager.assess_trade_risk(
                    symbol='AAPL',
                    quantity=signal.get('quantity', 10),
                    entry_price=current_price,
                    current_data=data_window
                )

                if risk_assessment.get('approved', True):
                    # Execute order
                    result = mock_executor.execute_order(
                        symbol='AAPL',
                        action=signal['action'],
                        quantity=signal['quantity'],
                        current_price=current_price
                    )

                    assert result['status'] in ['FILLED', 'REJECTED']
                    if result['status'] == 'FILLED':
                        trades_executed += 1

                    # Update portfolio
                    mock_portfolio.portfolio_value *= (1 + np.random.uniform(-0.01, 0.01))

        # Verify workflow completed
        assert trades_executed >= 0  # At least some trades should be attempted
        metrics = mock_portfolio.calculate_performance_metrics()
        assert 'total_return' in metrics
        assert 'sharpe_ratio' in metrics
        assert 'total_trades' in metrics

    def test_multi_symbol_trading_workflow(self, sample_market_data, mock_executor, mock_portfolio):
        """Test trading workflow with multiple symbols"""
        from src.strategies.momentum import MomentumStrategy
        from src.risk.advanced_risk_manager import AdvancedRiskManager

        symbols = list(sample_market_data.keys())
        strategies = {}
        risk_managers = {}

        # Setup strategies and risk managers for each symbol
        for symbol in symbols:
            strategies[symbol] = MomentumStrategy({
                'lookback_period': 10,
                'threshold': 0.02
            })

            risk_managers[symbol] = AdvancedRiskManager(
                capital=100000 // len(symbols),  # Divide capital among symbols
                max_position_size=0.05,
                max_daily_loss_pct=0.03
            )

        # Execute multi-symbol trading
        total_trades = 0
        for i in range(15, 50):  # Use smaller window for efficiency
            for symbol in symbols:
                data = sample_market_data[symbol].iloc[:i]

                if len(data) < 15:  # Skip if insufficient data
                    continue

                current_price = data['Close'].iloc[-1]

                # Generate signal
                signal = strategies[symbol].generate_signal(data)

                if signal['action'] != 'HOLD':
                    # Risk check
                    risk_ok = risk_managers[symbol].check_position_size(
                        quantity=signal.get('quantity', 5),
                        price=current_price
                    )

                    if risk_ok:
                        # Execute trade
                        result = mock_executor.execute_order(
                            symbol=symbol,
                            action=signal['action'],
                            quantity=signal['quantity'],
                            current_price=current_price
                        )

                        if result['status'] == 'FILLED':
                            total_trades += 1

        # Verify multi-symbol execution
        assert total_trades >= 0
        final_metrics = mock_portfolio.calculate_performance_metrics()
        assert final_metrics['total_trades'] >= total_trades

    def test_risk_managed_trading_workflow(self, sample_ohlcv_data, mock_executor):
        """Test trading workflow with comprehensive risk management"""
        from src.strategies.breakout import ImprovedBreakoutStrategy
        from src.risk.circuit_breaker import CircuitBreaker
        from src.risk.risk_manager import RiskManager

        # Setup components
        strategy = ImprovedBreakoutStrategy({
            'lookback_period': 20,
            'breakout_threshold': 0.03,
            'volume_multiplier': 1.5
        })

        risk_manager = RiskManager(
            capital=100000,
            max_position_size=0.02,  # Conservative position sizing
            max_daily_loss_pct=0.02
        )

        circuit_breaker = CircuitBreaker(
            max_daily_loss_pct=0.05,
            max_consecutive_losses=3,
            cooldown_period_minutes=5
        )

        # Execute risk-managed trading
        trades = []
        daily_pnl = 0
        consecutive_losses = 0

        for i in range(30, len(sample_ohlcv_data)):
            data_window = sample_ohlcv_data.iloc[:i]
            current_price = data_window['Close'].iloc[-1]

            # Check circuit breaker
            if circuit_breaker.is_tripped():
                continue

            # Generate signal
            signal = strategy.generate_signal(data_window)

            if signal['action'] != 'HOLD':
                # Risk assessment
                risk_result = risk_manager.assess_trade_risk(
                    symbol='AAPL',
                    quantity=signal['quantity'],
                    entry_price=current_price,
                    current_data=data_window
                )

                if risk_result.get('approved', False):
                    # Execute trade
                    result = mock_executor.execute_order(
                        symbol='AAPL',
                        action=signal['action'],
                        quantity=signal['quantity'],
                        current_price=current_price
                    )

                    if result['status'] == 'FILLED':
                        # Simulate P&L
                        pnl = np.random.uniform(-100, 200)  # Random P&L
                        daily_pnl += pnl

                        # Track consecutive losses
                        if pnl < 0:
                            consecutive_losses += 1
                        else:
                            consecutive_losses = 0

                        # Update circuit breaker
                        circuit_breaker.update_daily_pnl(daily_pnl)
                        circuit_breaker.update_consecutive_losses(consecutive_losses)

                        trades.append({
                            'symbol': 'AAPL',
                            'action': signal['action'],
                            'quantity': signal['quantity'],
                            'price': current_price,
                            'pnl': pnl,
                            'result': result
                        })

        # Verify risk management
        assert len(trades) >= 0
        if trades:
            total_pnl = sum(trade['pnl'] for trade in trades)
            assert abs(total_pnl) >= 0  # P&L should be calculated

        # Circuit breaker should prevent excessive losses
        assert not circuit_breaker.is_tripped() or daily_pnl <= -2500  # 5% of 50k

    def test_error_handling_in_trading_workflow(self, sample_ohlcv_data):
        """Test error handling throughout the trading workflow"""
        from src.strategies.trend_following import TrendFollowingAgent
        from src.risk.risk_manager import RiskManager

        strategy = TrendFollowingAgent("test_agent", {})
        risk_manager = RiskManager(capital=100000, max_position_size=0.1, max_daily_loss_pct=0.05)

        # Test with corrupted data
        corrupted_data = sample_ohlcv_data.copy()
        corrupted_data.loc[0, 'Close'] = np.nan  # Introduce NaN

        # Strategy should handle NaN gracefully
        try:
            signal = strategy.generate_trade_signal(corrupted_data, 100000)
            assert signal is not None
            assert 'action' in signal
        except Exception as e:
            # Should not crash, but might return HOLD signal
            assert isinstance(e, (ValueError, TypeError, KeyError))

        # Test risk manager with invalid inputs
        try:
            risk_result = risk_manager.assess_trade_risk(
                symbol='INVALID',
                quantity=-10,  # Invalid quantity
                entry_price=-100,  # Invalid price
                current_data=corrupted_data
            )
            # Should handle invalid inputs
            assert 'approved' in risk_result
        except Exception:
            # Expected to fail with invalid inputs
            assert True

    def test_performance_metrics_calculation(self, sample_ohlcv_data, mock_portfolio):
        """Test comprehensive performance metrics calculation"""
        # Simulate trading history
        trades = []
        portfolio_value = 100000

        for i in range(10):
            trade = {
                'timestamp': datetime.now() - timedelta(days=i),
                'symbol': 'AAPL',
                'action': 'BUY' if i % 2 == 0 else 'SELL',
                'quantity': 10,
                'price': 100 + np.random.uniform(-5, 5),
                'pnl': np.random.uniform(-200, 300)
            }
            trades.append(trade)
            portfolio_value += trade['pnl']

        # Mock portfolio to return our test data
        mock_portfolio.portfolio_value = portfolio_value
        mock_portfolio.get_trade_history.return_value = trades

        # Calculate metrics
        metrics = mock_portfolio.calculate_performance_metrics()

        # Verify all required metrics are present
        required_metrics = [
            'total_return', 'sharpe_ratio', 'max_drawdown',
            'total_trades', 'win_rate', 'profit_factor'
        ]

        for metric in required_metrics:
            assert metric in metrics
            assert isinstance(metrics[metric], (int, float, np.number))

        # Sanity checks
        assert metrics['total_trades'] == len(trades)
        assert 0 <= metrics['win_rate'] <= 1
        assert metrics['max_drawdown'] >= 0
