#!/usr/bin/env python3
"""
Comprehensive unit tests for risk management system.

Tests position sizing, risk assessment, portfolio optimization,
and circuit breaker functionality.
"""

import pytest
import pandas as pd
import numpy as np
from risk.advanced_risk_manager import AdvancedRiskManager, PortfolioMetrics
from risk.risk_manager import RiskManager


class TestBasicRiskManager:
    """Test basic risk manager functionality."""

    def test_initialization(self):
        """Test risk manager initialization."""
        rm = RiskManager(initial_capital=10000)

        assert rm.initial_capital == 10000
        assert rm.current_capital == 10000
        assert rm.stop_loss_pct == 0.02
        assert rm.take_profit_pct == 0.05

    def test_position_size_calculation(self):
        """Test position size calculation."""
        rm = RiskManager(initial_capital=10000, max_position_size=0.1)

        position_size = rm.calculate_position_size(
            entry_price=100.0,
            capital=10000,
            risk_pct=0.01
        )

        # Expected: (10000 * 0.01) / 100.0 = 1.0
        assert position_size == 1.0

    def test_stop_loss_check(self):
        """Test stop loss condition checking."""
        rm = RiskManager(stop_loss_pct=0.05)

        # Test long position stop loss
        assert rm.check_stop_loss(100.0, 95.0, 'long') == True   # 5% loss
        assert rm.check_stop_loss(100.0, 98.0, 'long') == False  # 2% loss

        # Test short position stop loss
        assert rm.check_stop_loss(100.0, 105.0, 'short') == True   # 5% loss
        assert rm.check_stop_loss(100.0, 102.0, 'short') == False  # 2% loss

    def test_take_profit_check(self):
        """Test take profit condition checking."""
        rm = RiskManager(take_profit_pct=0.10)

        # Test long position take profit
        assert rm.check_take_profit(100.0, 110.0, 'long') == True   # 10% profit
        assert rm.check_take_profit(100.0, 105.0, 'long') == False  # 5% profit

        # Test short position take profit
        assert rm.check_take_profit(100.0, 90.0, 'short') == True   # 10% profit
        assert rm.check_take_profit(95.0, 100.0, 'short') == False  # 5% profit


class TestAdvancedRiskManager:
    """Test advanced risk manager functionality."""

    def test_initialization(self):
        """Test advanced risk manager initialization."""
        arm = AdvancedRiskManager(initial_capital=10000)

        assert arm.initial_capital == 10000
        assert arm.current_capital == 10000
        assert arm.stop_loss_pct == 0.02
        assert arm.take_profit_pct == 0.05
        assert hasattr(arm, 'position_sizer')
        assert hasattr(arm, 'portfolio_optimizer')

    def test_trade_risk_assessment(self, sample_ohlcv_data):
        """Test comprehensive trade risk assessment."""
        arm = AdvancedRiskManager()

        assessment = arm.assess_trade_risk(
            symbol='ETHUSDT',
            signal=1,
            price=100.0,
            confidence=0.8,
            market_data=sample_ohlcv_data
        )

        # Check assessment structure
        assert isinstance(assessment, dict)
        assert 'approved' in assessment
        assert 'risk_score' in assessment
        assert 'recommended_size' in assessment
        assert 'warnings' in assessment
        assert 'reasons' in assessment

        # Check data types
        assert isinstance(assessment['approved'], bool)
        assert isinstance(assessment['risk_score'], float)
        assert isinstance(assessment['recommended_size'], float)
        assert isinstance(assessment['warnings'], list)
        assert isinstance(assessment['reasons'], list)

    def test_portfolio_metrics_calculation(self):
        """Test portfolio metrics calculation."""
        pm = PortfolioMetrics()

        # Create sample returns
        returns = pd.Series([0.01, 0.02, -0.005, 0.015, -0.01])

        pm.calculate_metrics(returns, {})

        # Check basic metrics
        assert isinstance(pm.daily_return, float)
        assert isinstance(pm.cumulative_return, float)
        assert isinstance(pm.volatility, float)
        assert isinstance(pm.sharpe_ratio, float)
        assert isinstance(pm.max_drawdown, float)

    def test_market_regime_detection(self, sample_ohlcv_data):
        """Test market regime detection."""
        arm = AdvancedRiskManager()

        # Test with sufficient data
        regime = arm._detect_market_regime(sample_ohlcv_data)
        assert regime in ['normal', 'volatile', 'crisis', 'unknown']

        # Test with insufficient data
        small_data = sample_ohlcv_data.head(10)
        regime_small = arm._detect_market_regime(small_data)
        assert regime_small == 'unknown'

    def test_volatility_calculation(self, sample_ohlcv_data):
        """Test volatility calculation."""
        arm = AdvancedRiskManager()

        volatility = arm._calculate_volatility('ETHUSDT', sample_ohlcv_data)
        assert isinstance(volatility, float)
        assert volatility >= 0

        # Test with insufficient data
        small_data = sample_ohlcv_data.head(5)
        volatility_small = arm._calculate_volatility('ETHUSDT', small_data)
        assert volatility_small == 0.02  # Default value

    def test_portfolio_rebalancing(self, sample_portfolio_state):
        """Test portfolio rebalancing calculations."""
        arm = AdvancedRiskManager()
        arm.update_portfolio(
            sample_portfolio_state['positions'],
            sample_portfolio_state['cash']
        )

        target_allocations = {'ETHUSDT': 0.6, 'BTCUSDT': 0.4}

        trades = arm.calculate_portfolio_rebalance(
            target_allocations,
            sample_portfolio_state['positions'],
            sample_portfolio_state['cash']
        )

        assert isinstance(trades, list)
        for trade in trades:
            assert 'symbol' in trade
            assert 'action' in trade
            assert 'quantity' in trade
            assert trade['action'] in ['BUY', 'SELL']
            assert trade['quantity'] > 0

    def test_risk_limits_checking(self):
        """Test portfolio risk limit enforcement."""
        arm = AdvancedRiskManager()

        # Test position that would exceed limits
        large_position = 100000  # Very large position
        price = 100.0

        exceeds_limit = arm._would_exceed_portfolio_limits(
            'ETHUSDT', large_position, price
        )

        assert isinstance(exceeds_limit, bool)

    def test_correlation_risk_assessment(self, sample_ohlcv_data):
        """Test correlation risk assessment."""
        arm = AdvancedRiskManager()

        correlation_risk = arm._check_correlation_risk('ETHUSDT', sample_ohlcv_data)

        assert isinstance(correlation_risk, float)
        assert 0.0 <= correlation_risk <= 1.0

    def test_stress_testing(self, sample_portfolio_state):
        """Test portfolio stress testing."""
        arm = AdvancedRiskManager()

        scenarios = [
            {'name': 'market_crash', 'type': 'price_shock', 'value': -0.20},
            {'name': 'recovery', 'type': 'price_shock', 'value': 0.10}
        ]

        results = arm.stress_test_portfolio(
            sample_portfolio_state['positions'],
            scenarios
        )

        assert isinstance(results, dict)
        assert len(results) == len(scenarios)

        for scenario_name, scenario_result in results.items():
            assert 'original_value' in scenario_result
            assert 'shocked_value' in scenario_result
            assert 'loss_pct' in scenario_result
            assert 'breach_warnings' in scenario_result

    def test_risk_report_generation(self):
        """Test comprehensive risk report generation."""
        arm = AdvancedRiskManager()

        # Add some positions for testing
        positions = {
            'ETHUSDT': {'quantity': 10, 'current_price': 100, 'entry_price': 95},
            'BTCUSDT': {'quantity': 2, 'current_price': 200, 'entry_price': 190}
        }
        arm.update_portfolio(positions, 8000)

        risk_report = arm.get_risk_report()

        assert isinstance(risk_report, dict)
        assert 'portfolio_metrics' in risk_report
        assert 'risk_limits' in risk_report
        assert 'current_exposure' in risk_report
        assert 'active_alerts' in risk_report

    def test_portfolio_metrics_edge_cases(self):
        """Test portfolio metrics with edge cases."""
        pm = PortfolioMetrics()

        # Test with insufficient data
        small_returns = pd.Series([0.01])
        pm.calculate_metrics(small_returns, {})

        # Should not crash, might return defaults
        assert hasattr(pm, 'daily_return')
        assert hasattr(pm, 'volatility')


class TestRiskManagerIntegration:
    """Test integration between risk managers."""

    def test_advanced_risk_manager_inheritance(self):
        """Test that AdvancedRiskManager properly integrates basic risk management."""
        arm = AdvancedRiskManager(initial_capital=10000)

        # Should have basic risk management attributes
        assert hasattr(arm, 'stop_loss_pct')
        assert hasattr(arm, 'take_profit_pct')

        # Should have advanced features
        assert hasattr(arm, 'position_sizer')
        assert hasattr(arm, 'portfolio_optimizer')

    def test_position_sizing_integration(self):
        """Test position sizing integration with market conditions."""
        arm = AdvancedRiskManager()

        position_size = arm.position_sizer.calculate_optimal_size(
            capital=10000,
            price=100.0,
            volatility=0.02,
            portfolio_volatility=0.015,
            symbol='ETHUSDT',
            current_positions={},
            market_regime='normal'
        )

        assert isinstance(position_size, float)
        assert position_size > 0
        assert position_size <= 10000 * 0.10 / 100.0  # Max 10% of capital per position


# Performance testing
@pytest.mark.slow
def test_risk_assessment_performance(sample_ohlcv_data):
    """Test risk assessment performance."""
    import time

    arm = AdvancedRiskManager()

    start_time = time.time()

    # Run multiple risk assessments
    for _ in range(100):
        assessment = arm.assess_trade_risk(
            symbol='ETHUSDT',
            signal=1,
            price=100.0,
            confidence=0.8,
            market_data=sample_ohlcv_data
        )

    end_time = time.time()

    total_time = end_time - start_time
    avg_time = total_time / 100

    # Should complete within reasonable time
    assert avg_time < 0.1  # Less than 100ms per assessment
    assert total_time < 10.0  # Less than 10 seconds total


# Edge case testing
def test_risk_manager_edge_cases():
    """Test risk manager behavior in edge cases."""

    # Test with zero capital
    arm = AdvancedRiskManager(initial_capital=0)
    assessment = arm.assess_trade_risk('ETHUSDT', 1, 100.0, 0.8, None)

    assert assessment['approved'] == False
    assert 'Invalid position size' in assessment['reasons']

    # Test with extreme volatility
    arm_normal = AdvancedRiskManager()
    position_size = arm_normal.position_sizer.calculate_optimal_size(
        capital=10000,
        price=100.0,
        volatility=1.0,  # 100% volatility
        portfolio_volatility=0.02,
        symbol='ETHUSDT',
        current_positions={},
        market_regime='crisis'
    )

    # Should be very conservative in crisis with high volatility
    assert position_size < 10  # Very small position


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
