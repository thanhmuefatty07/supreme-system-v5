#!/usr/bin/env python3
"""
Unit tests for Circuit Breaker implementation.

Tests financial risk limits and regulatory compliance.
Based on SEC/ESMA requirements and circuit breaker patterns.
"""

import pytest
import time
from unittest.mock import Mock, MagicMock
from src.risk.circuit_breaker import (
    CircuitBreaker,
    CircuitBreakerState,
    CircuitBreakerOpen
)


class TestCircuitBreaker:
    """Test circuit breaker functionality."""

    @pytest.fixture
    def mock_portfolio(self):
        """Mock portfolio for testing."""
        portfolio = Mock()
        portfolio.total_value = 100000
        portfolio.get_daily_pnl.return_value = -2000  # -2%
        portfolio.get_hourly_pnl.return_value = -500   # -0.5%
        portfolio.get_current_drawdown.return_value = 0.05  # 5%
        portfolio.get_positions.return_value = []
        return portfolio

    @pytest.fixture
    def circuit_breaker_config(self):
        """Standard circuit breaker configuration."""
        return {
            'failure_threshold': 3,
            'timeout': 60,
            'success_threshold': 2,
            'max_daily_loss': 0.05,    # 5%
            'max_hourly_loss': 0.02,   # 2%
            'max_position_size': 0.10, # 10%
            'max_drawdown': 0.15       # 15%
        }

    def test_initialization(self, circuit_breaker_config):
        """Test circuit breaker initialization."""
        cb = CircuitBreaker(circuit_breaker_config)

        assert cb.state == CircuitBreakerState.CLOSED
        assert cb.failure_count == 0
        assert cb.success_count == 0
        assert cb.failure_threshold == 3
        assert cb.timeout == 60

    def test_call_success_closed_state(self, circuit_breaker_config):
        """Test successful call in closed state."""
        cb = CircuitBreaker(circuit_breaker_config)

        def mock_function():
            return "success"

        result = cb.call(mock_function)
        assert result == "success"
        assert cb.state == CircuitBreakerState.CLOSED
        assert cb.success_count_total == 1

    def test_call_failure_triggers_open(self, circuit_breaker_config):
        """Test that repeated failures trigger circuit open."""
        cb = CircuitBreaker(circuit_breaker_config)

        def failing_function():
            raise ValueError("Test failure")

        # First 3 calls should execute and fail, opening circuit on 3rd failure
        for i in range(3):
            with pytest.raises(ValueError):
                cb.call(failing_function)
            assert cb.failure_count == i + 1

        # Circuit should be open after 3 failures
        assert cb.state == CircuitBreakerState.OPEN

        # 4th call should not execute and raise CircuitBreakerOpen
        with pytest.raises(CircuitBreakerOpen):
            cb.call(failing_function)

    def test_daily_loss_limit(self, circuit_breaker_config, mock_portfolio):
        """Test daily loss limit triggers circuit breaker."""
        cb = CircuitBreaker(circuit_breaker_config)
        cb.portfolio = mock_portfolio

        # Set up portfolio to exceed daily loss limit
        mock_portfolio.get_daily_pnl.return_value = -6000  # -6% (exceeds 5% limit)

        def mock_function():
            return "should_not_execute"

        with pytest.raises(CircuitBreakerOpen):
            cb.call(mock_function)

        assert cb.state == CircuitBreakerState.OPEN

    def test_hourly_loss_limit(self, circuit_breaker_config, mock_portfolio):
        """Test hourly loss limit triggers circuit breaker."""
        cb = CircuitBreaker(circuit_breaker_config)
        cb.portfolio = mock_portfolio

        # Set up portfolio to exceed hourly loss limit
        mock_portfolio.get_hourly_pnl.return_value = -3000  # -3% (exceeds 2% limit)

        def mock_function():
            return "should_not_execute"

        with pytest.raises(CircuitBreakerOpen):
            cb.call(mock_function)

        assert cb.state == CircuitBreakerState.OPEN

    def test_half_open_recovery(self, circuit_breaker_config):
        """Test half-open state and recovery."""
        cb = CircuitBreaker(circuit_breaker_config)

        def failing_function():
            raise ValueError("Test failure")

        # Force circuit open (3 failures)
        for i in range(3):
            with pytest.raises(ValueError):
                cb.call(failing_function)

        assert cb.state == CircuitBreakerState.OPEN

        # Simulate timeout passage
        cb.last_failure_time = time.time() - 70  # 70 seconds ago

        def success_function():
            return "success"

        # First call in HALF_OPEN should succeed
        result = cb.call(success_function)
        assert result == "success"
        assert cb.state == CircuitBreakerState.HALF_OPEN
        assert cb.success_count == 1

        # Second successful call should close circuit
        result = cb.call(success_function)
        assert result == "success"
        assert cb.state == CircuitBreakerState.CLOSED
        assert cb.success_count == 0
        assert cb.failure_count == 0

    def test_half_open_failure_back_to_open(self, circuit_breaker_config):
        """Test failure in half-open state returns to open."""
        cb = CircuitBreaker(circuit_breaker_config)

        def failing_function():
            raise ValueError("Test failure")

        # Force circuit open (3 failures)
        for i in range(3):
            with pytest.raises(ValueError):
                cb.call(failing_function)

        assert cb.state == CircuitBreakerState.OPEN

        # Simulate timeout passage
        cb.last_failure_time = time.time() - 70

        # First call transitions to HALF_OPEN and succeeds
        def success_function():
            return "success"

        result = cb.call(success_function)
        assert cb.state == CircuitBreakerState.HALF_OPEN

        # Failure in HALF_OPEN should return to OPEN
        with pytest.raises(ValueError):
            cb.call(failing_function)

        assert cb.state == CircuitBreakerState.OPEN
        assert cb.success_count == 0

    def test_position_concentration_limit(self, circuit_breaker_config, mock_portfolio):
        """Test position concentration limit."""
        cb = CircuitBreaker(circuit_breaker_config)
        cb.portfolio = mock_portfolio

        # Mock large position
        large_position = Mock()
        large_position.value = 15000  # 15% of portfolio (exceeds 10% limit)
        mock_portfolio.get_positions.return_value = [large_position]

        def mock_function():
            return "should_not_execute"

        with pytest.raises(CircuitBreakerOpen):
            cb.call(mock_function)

        assert cb.state == CircuitBreakerState.OPEN

    def test_request_count_tracking(self, circuit_breaker_config):
        """Test request count tracking."""
        cb = CircuitBreaker(circuit_breaker_config)

        def mock_function():
            return "success"

        # Make several calls
        for i in range(5):
            cb.call(mock_function)

        assert cb.request_count == 5
        assert cb.success_count_total == 5

    def test_no_portfolio_no_risk_checks(self, circuit_breaker_config):
        """Test that without portfolio, risk checks are skipped."""
        cb = CircuitBreaker(circuit_breaker_config)
        # portfolio is None by default

        def mock_function():
            return "success"

        # Should work normally even with large "losses" since no portfolio
        result = cb.call(mock_function)
        assert result == "success"
        assert cb.state == CircuitBreakerState.CLOSED
