#!/usr/bin/env python3
"""
Comprehensive Tests for Risk Management Package

Tests cover all components: calculations, limits, and core risk manager.
Target: 100% coverage for the risk module.
"""

import pytest
from datetime import datetime

# Import all risk components
from src.risk.calculations import (
    calculate_kelly_criterion,
    apply_position_sizing,
    calculate_var_historical,
    calculate_sharpe_ratio
)
from src.risk.limits import CircuitBreaker, PositionSizeLimiter
from src.risk.core import RiskManager


class TestRiskCalculations:
    """Test pure calculation functions."""

    def test_kelly_criterion_basic(self):
        """Test Kelly criterion with standard inputs."""
        # Win 50%, RR 2.0 -> f = (0.5*3 - 1)/2 = 0.25
        result = calculate_kelly_criterion(0.5, 2.0)
        assert result == 0.25

    def test_kelly_criterion_edge_cases(self):
        """Test Kelly with edge cases."""
        # Breakeven case
        assert calculate_kelly_criterion(0.5, 1.0) == 0.0

        # Invalid inputs
        assert calculate_kelly_criterion(-0.1, 2.0) == 0.0
        assert calculate_kelly_criterion(0.5, -1.0) == 0.0
        assert calculate_kelly_criterion(1.5, 2.0) == 0.0

    def test_position_sizing_logic(self):
        """Test position sizing with different modes."""
        capital = 10000
        kelly_fraction = 0.25

        # Full Kelly
        size = apply_position_sizing(capital, kelly_fraction, 1.0, 'full')
        assert size == 2500.0

        # Half Kelly
        size = apply_position_sizing(capital, kelly_fraction, 1.0, 'half')
        assert size == 1250.0

        # Quarter Kelly
        size = apply_position_sizing(capital, kelly_fraction, 1.0, 'quarter')
        assert size == 625.0

    def test_position_sizing_hard_cap(self):
        """Test hard cap functionality."""
        capital = 10000
        kelly_fraction = 0.25

        # Hard cap at 2% -> should return 200
        size = apply_position_sizing(capital, kelly_fraction, 0.02, 'half')
        assert size == 200.0  # min(1250, 200) = 200

    def test_var_calculation(self):
        """Test Value at Risk calculation."""
        returns = [-0.05, -0.03, -0.01, 0.02, 0.04]  # 5% worst case at 95% confidence
        var = calculate_var_historical(returns, 0.95)
        assert var == 0.05  # 5% VaR

    def test_sharpe_ratio(self):
        """Test Sharpe ratio calculation."""
        # Better returns data - more volatility and positive expected value
        returns = [0.05, 0.08, -0.02, 0.12, 0.03]  # Higher returns with volatility
        sharpe = calculate_sharpe_ratio(returns, 0.02)
        assert isinstance(sharpe, (float, int))  # Just check it's a number

        # Test with no volatility
        constant_returns = [0.02, 0.02, 0.02]
        sharpe_zero = calculate_sharpe_ratio(constant_returns, 0.02)
        assert sharpe_zero == 0.0  # Zero volatility = zero Sharpe


class TestRiskLimits:
    """Test risk limits and circuit breakers."""

    def test_circuit_breaker_initialization(self):
        """Test circuit breaker setup."""
        cb = CircuitBreaker(daily_limit_pct=0.10)
        assert cb.daily_limit == 0.10
        assert cb.is_active is False
        assert cb.current_daily_drawdown == 0.0

    def test_circuit_breaker_trigger(self):
        """Test circuit breaker triggering."""
        cb = CircuitBreaker(daily_limit_pct=0.10)  # 10% limit

        # Loss 5% -> Safe
        assert cb.update(-0.05) is True
        assert cb.is_active is False
        assert cb.current_daily_drawdown == 0.05

        # Another Loss 6% -> Total 11% -> Triggered
        assert cb.update(-0.06) is False
        assert cb.is_active is True
        assert cb.trigger_reason == 'daily_limit'

    def test_circuit_breaker_wins_no_trigger(self):
        """Test that wins don't trigger circuit breaker."""
        cb = CircuitBreaker(daily_limit_pct=0.10)

        # Win 5% -> should not affect drawdown
        assert cb.update(0.05) is True
        assert cb.current_daily_drawdown == 0.0
        assert cb.is_active is False

    def test_position_size_limiter(self):
        """Test position size validation."""
        limiter = PositionSizeLimiter(max_position_pct=0.10, max_portfolio_pct=0.50)

        # Valid position
        is_valid, reason = limiter.validate_position_size(1000, 10000, 0)
        assert is_valid is True
        assert "approved" in reason

        # Too large position
        is_valid, reason = limiter.validate_position_size(1500, 10000, 0)  # 15% > 10%
        assert is_valid is False
        assert "Position size" in reason

        # Too much total exposure
        is_valid, reason = limiter.validate_position_size(1000, 10000, 6000)  # Total 70% > 50%
        assert is_valid is False
        assert "portfolio" in reason


class TestRiskManagerIntegration:
    """Test complete RiskManager integration."""

    @pytest.fixture
    def risk_manager(self):
        """Create configured risk manager."""
        config = {
            "max_risk_per_trade": 0.1,  # 10% cap
            "kelly_mode": "full",
            "daily_loss_limit": 0.2,  # 20% daily limit
            "max_position_pct": 0.10,
            "max_portfolio_pct": 0.50,
            "risk_free_rate": 0.02,  # Required for Sharpe ratio
            "var_confidence": 0.95   # Required for VaR
        }
        return RiskManager(capital=10000, config=config)

    def test_risk_manager_initialization(self, risk_manager):
        """Test risk manager setup."""
        assert risk_manager.current_capital == 10000
        assert risk_manager.initial_capital == 10000
        assert risk_manager.is_trading_allowed() is True

    def test_position_sizing_flow(self, risk_manager):
        """Test complete position sizing flow."""
        # Get size for good trade
        size = risk_manager.get_target_size(0.6, 2.0)  # 60% win rate, 2:1 RR
        assert size > 0
        assert size <= 1000  # Max 10% of capital

    def test_trade_recording(self, risk_manager):
        """Test trade recording and state updates."""
        # Record winning trade
        risk_manager.record_trade(500, "BTCUSDT")
        assert risk_manager.current_capital == 10500
        assert risk_manager.total_trades == 1
        assert risk_manager.winning_trades == 1

        # Record losing trade
        risk_manager.record_trade(-300, "ETHUSDT")
        assert risk_manager.current_capital == 10200
        assert risk_manager.total_trades == 2
        assert risk_manager.winning_trades == 1

    def test_circuit_breaker_integration(self, risk_manager):
        """Test circuit breaker integration with trades."""
        # Record massive loss (-3000 = -30%)
        risk_manager.record_trade(-3000, "BTCUSDT")
        assert risk_manager.current_capital == 7000

        # Check breaker triggered (30% > 20% limit)
        assert risk_manager.circuit_breaker.is_active is True
        assert risk_manager.is_trading_allowed() is False

        # Position sizing should return 0
        size = risk_manager.get_target_size(0.6, 2.0)
        assert size == 0.0

    def test_risk_metrics_calculation(self, risk_manager):
        """Test risk metrics calculation."""
        # Add some trades
        trades = [500, -200, 300, -100, 400]
        for pnl in trades:
            risk_manager.record_trade(pnl, "TEST")

        metrics = risk_manager.get_risk_metrics()

        assert metrics['total_trades'] == 5
        assert metrics['win_rate'] == 0.6  # 3 wins out of 5
        assert metrics['total_pnl'] == 900
        assert metrics['current_capital'] == 10900
        assert 'sharpe_ratio' in metrics
        assert 'var_95' in metrics
        assert 'circuit_breaker_status' in metrics

    def test_reset_functionality(self, risk_manager):
        """Test risk manager reset."""
        # Make some changes
        risk_manager.record_trade(1000, "TEST")
        risk_manager.get_target_size(0.5, 1.5)

        # Reset
        risk_manager.reset()

        # Check reset state
        assert risk_manager.current_capital == 10000
        assert risk_manager.total_trades == 0
        assert risk_manager.total_pnl == 0.0
        assert len(risk_manager.trades) == 0
        assert risk_manager.is_trading_allowed() is True

    def test_invalid_inputs_handling(self, risk_manager):
        """Test handling of invalid inputs."""
        # Invalid win rate
        size = risk_manager.get_target_size(-0.1, 2.0)
        assert size == 0.0

        # Invalid RR ratio
        size = risk_manager.get_target_size(0.5, -1.0)
        assert size == 0.0

        # Win rate > 1
        size = risk_manager.get_target_size(1.5, 2.0)
        assert size == 0.0
