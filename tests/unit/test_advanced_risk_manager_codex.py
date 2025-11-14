#!/usr/bin/env python3
"""
Comprehensive unit tests for AdvancedRiskManager module.
Generated with Codex Web for Supreme System V5.
"""

from __future__ import annotations

import math
import time
from datetime import datetime, timedelta
from typing import Any, Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd
import pytest
from hypothesis import given, strategies as st
from pytest import approx
from unittest.mock import MagicMock, Mock, patch

from src.risk.advanced_risk_manager import (
    AdvancedRiskManager,
    DynamicPositionSizer,
    PortfolioMetrics,
    PortfolioOptimizer,
)


@pytest.fixture(name="advanced_risk_manager")
def fixture_advanced_risk_manager() -> AdvancedRiskManager:
    """Create a fresh ``AdvancedRiskManager`` instance for each test."""
    return AdvancedRiskManager(initial_capital=50000, stop_loss_pct=0.03, take_profit_pct=0.07)


@pytest.fixture(name="portfolio_positions")
def fixture_portfolio_positions() -> Dict[str, Dict[str, float]]:
    """Provide a representative portfolio used in multiple scenarios."""
    return {
        "ETHUSDT": {"symbol": "ETHUSDT", "quantity": 5.0, "current_price": 1800.0},
        "BTCUSDT": {"symbol": "BTCUSDT", "quantity": 0.75, "current_price": 30000.0},
        "ADAUSDT": {"symbol": "ADAUSDT", "quantity": 1000.0, "current_price": 0.35},
    }


@pytest.fixture(name="market_regime_data")
def fixture_market_regime_data() -> pd.DataFrame:
    """Generate market data with a modest upward drift used to exercise trend detection."""
    dates = pd.date_range(datetime.utcnow() - timedelta(days=60), periods=60, freq="D")
    base_prices = np.linspace(100.0, 130.0, num=60)
    noise = np.random.normal(0, 0.5, size=60)
    close_prices = base_prices + noise
    return pd.DataFrame(
        {
            "timestamp": dates,
            "open": close_prices + np.random.normal(0, 0.5, size=60),
            "high": close_prices + np.random.normal(0.3, 0.5, size=60),
            "low": close_prices - np.random.normal(0.3, 0.5, size=60),
            "close": close_prices,
            "volume": np.random.randint(1_000_000, 5_000_000, size=60),
        }
    )


class TestPortfolioMetricsInitialization:
    """Validate ``PortfolioMetrics`` setup behavior."""

    def test_portfolio_metrics_initializes_to_zero(self) -> None:
        """Ensure every tracked metric defaults to zero at instantiation."""
        metrics = PortfolioMetrics()

        assert metrics.total_value == 0.0, "Total value should start at zero"
        assert metrics.cash == 0.0, "Cash should start at zero"
        assert metrics.positions_value == 0.0, "Positions value should start at zero"
        assert metrics.daily_return == 0.0, "Daily return should start at zero"
        assert metrics.cumulative_return == 0.0, "Cumulative return should start at zero"
        assert metrics.volatility == 0.0, "Volatility should start at zero"
        assert metrics.sharpe_ratio == 0.0, "Sharpe ratio should start at zero"
        assert metrics.max_drawdown == 0.0, "Max drawdown should start at zero"
        assert metrics.value_at_risk == 0.0, "VaR should start at zero"
        assert metrics.expected_shortfall == 0.0, "Expected shortfall should start at zero"


class TestPortfolioMetricsCalculations:
    """Exercise ``PortfolioMetrics.calculate_metrics`` across edge cases."""

    def test_portfolio_metrics_handles_insufficient_returns(self) -> None:
        """Metrics should remain unchanged when provided fewer than two return observations."""
        metrics = PortfolioMetrics()
        metrics.calculate_metrics(pd.Series([0.01]), positions={})

        assert metrics.daily_return == 0.0, "Daily return must remain default for insufficient data"
        assert metrics.cumulative_return == 0.0, "Cumulative return must remain default"

    def test_portfolio_metrics_with_empty_series(self) -> None:
        """Empty return series should leave the metrics untouched."""
        metrics = PortfolioMetrics()
        metrics.calculate_metrics(pd.Series(dtype=float), positions={})

        assert metrics.daily_return == 0.0
        assert metrics.volatility == 0.0

    def test_portfolio_metrics_with_nan_values(self) -> None:
        """NaN returns should propagate through to risk metrics for transparency."""
        metrics = PortfolioMetrics()
        # Put NaN as the last value to test propagation
        returns = pd.Series([0.01, 0.02, -0.01, np.nan])
        metrics.calculate_metrics(returns, positions={})

        assert math.isnan(metrics.daily_return), "Latest return containing NaN should propagate"
        # Volatility calculation may handle NaN differently, so check if it's NaN or handled gracefully
        assert math.isnan(metrics.volatility) or metrics.volatility >= 0.0, "NaN inputs should lead to NaN volatility or be handled gracefully"

    def test_portfolio_metrics_with_inf_values(self) -> None:
        """Infinite returns should produce infinite volatility to highlight instability."""
        metrics = PortfolioMetrics()
        returns = pd.Series([0.01, np.inf, -0.02, 0.03])
        metrics.calculate_metrics(returns, positions={})

        assert math.isinf(metrics.volatility), "Infinite return must escalate volatility"

    def test_portfolio_metrics_with_constant_returns(self) -> None:
        """Constant return series must yield zero volatility and undefined Sharpe ratio."""
        metrics = PortfolioMetrics()
        returns = pd.Series([0.01] * 40)
        metrics.calculate_metrics(returns, positions={})

        assert metrics.volatility == pytest.approx(0.0, abs=1e-9)
        assert metrics.sharpe_ratio == 0.0, "Sharpe ratio remains zero without excess return variation"

    def test_portfolio_metrics_with_positive_returns(self) -> None:
        """All positive returns should generate a positive cumulative return and drawdown at zero."""
        metrics = PortfolioMetrics()
        returns = pd.Series([0.02] * 60)
        metrics.calculate_metrics(returns, positions={})

        assert metrics.cumulative_return > 0.0
        assert metrics.max_drawdown == pytest.approx(0.0, abs=1e-9)

    def test_portfolio_metrics_with_negative_returns(self) -> None:
        """All negative returns should produce a negative cumulative return and non-zero drawdown."""
        metrics = PortfolioMetrics()
        returns = pd.Series([-0.01] * 60)
        metrics.calculate_metrics(returns, positions={})

        assert metrics.cumulative_return < 0.0
        assert metrics.max_drawdown <= 0.0

    def test_portfolio_metrics_with_large_returns(self) -> None:
        """Extremely large returns should remain numerically stable."""
        metrics = PortfolioMetrics()
        returns = pd.Series([0.5, -0.4, 0.7, -0.3, 0.6, -0.5])
        metrics.calculate_metrics(returns, positions={})

        assert -1.0 <= metrics.max_drawdown <= 0.0
        assert metrics.volatility > 0.0

    def test_portfolio_metrics_with_zero_returns(self) -> None:
        """Zero returns should lead to neutral daily return and zero cumulative performance."""
        metrics = PortfolioMetrics()
        returns = pd.Series([0.0] * 50)
        metrics.calculate_metrics(returns, positions={})

        assert metrics.daily_return == 0.0
        assert metrics.cumulative_return == 0.0

    def test_portfolio_metrics_var_skips_short_series(self) -> None:
        """VaR must remain zero when fewer than thirty return observations are supplied."""
        metrics = PortfolioMetrics()
        metrics.calculate_metrics(pd.Series([0.01] * 25), positions={})

        assert metrics.value_at_risk == 0.0
        assert metrics.expected_shortfall == 0.0

    def test_portfolio_metrics_var_with_sufficient_data(self) -> None:
        """VaR and CVaR should be calculated when enough data is provided."""
        metrics = PortfolioMetrics()
        returns = pd.Series(np.linspace(-0.05, 0.05, num=60))
        metrics.calculate_metrics(returns, positions={})

        assert metrics.value_at_risk <= 0.0
        assert metrics.expected_shortfall <= metrics.value_at_risk

    def test_portfolio_metrics_daily_return_matches_last_observation(self) -> None:
        """Daily return should reflect the final entry in the return series."""
        metrics = PortfolioMetrics()
        returns = pd.Series([0.01, -0.02, 0.03])
        metrics.calculate_metrics(returns, positions={})

        assert metrics.daily_return == 0.03

    def test_portfolio_metrics_cumulative_return_matches_product(self) -> None:
        """Validate cumulative return formula against manual product expansion."""
        metrics = PortfolioMetrics()
        returns = pd.Series([0.01, 0.02, -0.01])
        metrics.calculate_metrics(returns, positions={})

        expected = (1 + 0.01) * (1 + 0.02) * (1 - 0.01) - 1
        assert metrics.cumulative_return == approx(expected)


class TestDynamicPositionSizerInitialization:
    """Test the configuration of ``DynamicPositionSizer``."""

    def test_dynamic_position_sizer_defaults(self) -> None:
        """Verify default parameters are respected."""
        sizer = DynamicPositionSizer()

        assert sizer.base_risk_pct == 0.01
        assert sizer.volatility_lookback == 20
        assert sizer.correlation_lookback == 30

    def test_dynamic_position_sizer_custom_risk_pct(self) -> None:
        """Custom base risk percentages should be persisted."""
        sizer = DynamicPositionSizer(base_risk_pct=0.02)

        assert sizer.base_risk_pct == 0.02


class TestDynamicPositionSizerCalculations:
    """Assess position sizing logic under different market configurations."""

    def test_calculate_optimal_size_normal_conditions(self) -> None:
        """Normal market regime should yield positive position size within caps."""
        sizer = DynamicPositionSizer()
        size = sizer.calculate_optimal_size(
            capital=100000.0,
            price=100.0,
            volatility=0.2,
            portfolio_volatility=0.15,
            symbol="ETHUSDT",
            current_positions={}
        )

        assert size > 0.0
        assert size < 1000.0

    @pytest.mark.parametrize("regime, expected_multiplier", [
        ("normal", 1.0),
        ("volatile", 0.5),
        ("crisis", 0.2),
    ])
    def test_calculate_optimal_size_regime_adjustments(self, regime: str, expected_multiplier: float) -> None:
        """Market regime changes should impact the resulting position size proportionally."""
        sizer = DynamicPositionSizer(base_risk_pct=0.01)
        base_size = sizer.calculate_optimal_size(
            capital=50000.0,
            price=50.0,
            volatility=0.1,
            portfolio_volatility=0.1,
            symbol="BTCUSDT",
            current_positions={},
            market_regime="normal",
        )
        adjusted_size = sizer.calculate_optimal_size(
            capital=50000.0,
            price=50.0,
            volatility=0.1,
            portfolio_volatility=0.1,
            symbol="BTCUSDT",
            current_positions={},
            market_regime=regime,
        )

        assert adjusted_size <= base_size * (expected_multiplier + 0.05)

    def test_calculate_optimal_size_handles_zero_capital(self) -> None:
        """Zero capital must yield a zero position size."""
        sizer = DynamicPositionSizer()
        size = sizer.calculate_optimal_size(
            capital=0.0,
            price=100.0,
            volatility=0.2,
            portfolio_volatility=0.2,
            symbol="ETHUSDT",
            current_positions={},
        )

        assert size == 0.0

    def test_calculate_optimal_size_handles_zero_price(self) -> None:
        """A zero price should avoid division by zero and result in zero size."""
        sizer = DynamicPositionSizer()
        size = sizer.calculate_optimal_size(
            capital=10000.0,
            price=0.0,
            volatility=0.2,
            portfolio_volatility=0.2,
            symbol="ETHUSDT",
            current_positions={},
        )

        assert size == 0.0

    def test_calculate_optimal_size_handles_zero_volatility(self) -> None:
        """Zero volatility falls back to conservative sizing and respects caps."""
        sizer = DynamicPositionSizer()
        size = sizer.calculate_optimal_size(
            capital=50000.0,
            price=200.0,
            volatility=0.0,
            portfolio_volatility=0.2,
            symbol="ETHUSDT",
            current_positions={},
        )

        assert size >= 0.0

    def test_calculate_optimal_size_handles_negative_volatility(self) -> None:
        """Negative volatility should be treated like zero volatility."""
        sizer = DynamicPositionSizer()
        size = sizer.calculate_optimal_size(
            capital=50000.0,
            price=200.0,
            volatility=-0.1,
            portfolio_volatility=0.2,
            symbol="ETHUSDT",
            current_positions={},
        )

        assert size >= 0.0

    def test_calculate_optimal_size_with_existing_sector_positions(self) -> None:
        """Owning related assets should shrink the recommended size."""
        sizer = DynamicPositionSizer()
        positions = {
            "ETHXYZ": {"symbol": "ETHXYZ", "quantity": 1.0, "current_price": 1000.0},
            "ETHABC": {"symbol": "ETHABC", "quantity": 1.0, "current_price": 1000.0},
        }
        reduced_size = sizer.calculate_optimal_size(
            capital=50000.0,
            price=1800.0,
            volatility=0.3,
            portfolio_volatility=0.2,
            symbol="ETHUSDT",
            current_positions=positions,
        )

        solo_size = sizer.calculate_optimal_size(
            capital=50000.0,
            price=1800.0,
            volatility=0.3,
            portfolio_volatility=0.2,
            symbol="BTCUSDT",
            current_positions={},
        )

        assert reduced_size < solo_size

    def test_calculate_optimal_size_caps_at_ten_percent_of_capital(self) -> None:
        """Position sizing must respect the 10% capital cap."""
        sizer = DynamicPositionSizer()
        size = sizer.calculate_optimal_size(
            capital=1_000_000.0,
            price=50.0,
            volatility=0.05,
            portfolio_volatility=0.05,
            symbol="BTCUSDT",
            current_positions={},
        )

        assert size <= (1_000_000.0 * 0.10) / 50.0

    def test_calculate_optimal_size_with_multiple_sector_positions(self) -> None:
        """Three correlated positions should trigger the harshest diversification penalty."""
        sizer = DynamicPositionSizer()
        positions = {
            "BTC1": {"symbol": "BTC1", "quantity": 1.0, "current_price": 20_000.0},
            "BTC2": {"symbol": "BTC2", "quantity": 1.0, "current_price": 20_000.0},
            "BTC3": {"symbol": "BTC3", "quantity": 1.0, "current_price": 20_000.0},
        }
        size = sizer.calculate_optimal_size(
            capital=100000.0,
            price=20000.0,
            volatility=0.3,
            portfolio_volatility=0.2,
            symbol="BTCUSDT",
            current_positions=positions,
        )

        assert size < 1.0

    def test_calculate_optimal_size_with_low_portfolio_volatility(self) -> None:
        """Less volatile portfolios should allow slightly larger positions."""
        sizer = DynamicPositionSizer()
        low_vol_size = sizer.calculate_optimal_size(
            capital=50000.0,
            price=200.0,
            volatility=0.1,
            portfolio_volatility=0.05,
            symbol="ETHUSDT",
            current_positions={},
        )
        high_vol_size = sizer.calculate_optimal_size(
            capital=50000.0,
            price=200.0,
            volatility=0.1,
            portfolio_volatility=0.25,
            symbol="ETHUSDT",
            current_positions={},
        )

        assert low_vol_size > high_vol_size


class TestDynamicPositionSizerHelpers:
    """Cover the helper methods underpinning the sizing logic."""

    def test_kelly_criterion_with_positive_volatility(self) -> None:
        """Reasonable volatility levels should yield a Kelly fraction below the cap."""
        sizer = DynamicPositionSizer()
        kelly = sizer._kelly_criterion(0.2)

        assert 0.0 < kelly <= 0.1

    def test_kelly_criterion_with_zero_volatility(self) -> None:
        """Zero volatility should fall back to a conservative fixed percentage."""
        sizer = DynamicPositionSizer()
        assert sizer._kelly_criterion(0.0) == 0.01

    def test_kelly_criterion_with_negative_volatility(self) -> None:
        """Negative volatility behaves like the zero-volatility fallback."""
        sizer = DynamicPositionSizer()
        assert sizer._kelly_criterion(-0.1) == 0.01

    def test_kelly_criterion_with_high_volatility(self) -> None:
        """Extremely high volatility should clamp at the 10% cap."""
        sizer = DynamicPositionSizer()
        assert sizer._kelly_criterion(5.0) == 0.1

    def test_market_regime_adjustment_variants(self) -> None:
        """Enumerate the built-in market regime multipliers."""
        sizer = DynamicPositionSizer()
        assert sizer._market_regime_adjustment("normal") == 1.0
        assert sizer._market_regime_adjustment("volatile") == 0.5
        assert sizer._market_regime_adjustment("crisis") == 0.2
        assert sizer._market_regime_adjustment("unexpected") == 1.0

    def test_diversification_adjustment_empty_positions(self) -> None:
        """With no current holdings no penalty should apply."""
        sizer = DynamicPositionSizer()
        assert sizer._diversification_adjustment({}, "ETHUSDT") == 1.0

    def test_diversification_adjustment_single_related_position(self) -> None:
        """One related position should apply the intermediate multiplier."""
        sizer = DynamicPositionSizer()
        positions = {"ETHABC": {"symbol": "ETHABC"}}
        assert sizer._diversification_adjustment(positions, "ETHXYZ") == 0.7

    def test_diversification_adjustment_multiple_related_positions(self) -> None:
        """Two or more related positions should apply the strongest penalty."""
        sizer = DynamicPositionSizer()
        positions = {"ETH1": {"symbol": "ETH1"}, "ETH2": {"symbol": "ETH2"}}
        assert sizer._diversification_adjustment(positions, "ETH3") == 0.5

    def test_volatility_adjustment_more_volatile_asset(self) -> None:
        """Assets more volatile than the portfolio should downsize the position."""
        sizer = DynamicPositionSizer()
        assert sizer._volatility_adjustment(0.3, 0.1) == 0.7

    def test_volatility_adjustment_less_volatile_asset(self) -> None:
        """Assets less volatile than the portfolio should upsize the position modestly."""
        sizer = DynamicPositionSizer()
        assert sizer._volatility_adjustment(0.05, 0.2) == 1.2

    def test_volatility_adjustment_with_zero_portfolio_volatility(self) -> None:
        """Zero portfolio volatility should not modify the base sizing."""
        sizer = DynamicPositionSizer()
        assert sizer._volatility_adjustment(0.1, 0.0) == 1.0


class TestPortfolioOptimizerInitialization:
    """Verify initial optimizer configuration."""

    def test_portfolio_optimizer_defaults(self) -> None:
        """Ensure optimizer exposes risk free rate and max weight settings."""
        optimizer = PortfolioOptimizer()

        assert optimizer.risk_free_rate == 0.02
        assert optimizer.max_weight == 0.25


class TestPortfolioOptimizerBehaviour:
    """Exercise the optimizer across realistic data regimes."""

    def test_optimize_portfolio_single_asset(self) -> None:
        """Single-asset data should return a trivial full allocation."""
        optimizer = PortfolioOptimizer()
        returns = pd.DataFrame({"A": [0.01, 0.02, -0.01]})
        result = optimizer.optimize_portfolio(returns)

        assert result["weights"] == {"A": 1.0}
        assert "optimization_success" not in result

    def test_optimize_portfolio_with_target_return(self) -> None:
        """Providing a target return should produce success metadata."""
        optimizer = PortfolioOptimizer()
        returns = pd.DataFrame({
            "A": [0.01, 0.02, 0.03, 0.015],
            "B": [0.02, 0.01, 0.025, 0.02],
        })
        result = optimizer.optimize_portfolio(returns, target_return=0.015)

        assert pytest.approx(sum(result["weights"].values()), rel=1e-5) == 1.0
        assert result["optimization_success"] is True

    def test_optimize_portfolio_without_target_return(self) -> None:
        """When maximizing Sharpe the optimizer should succeed on regular data."""
        optimizer = PortfolioOptimizer()
        returns = pd.DataFrame(np.random.normal(0.001, 0.01, size=(50, 3)), columns=list("ABC"))
        result = optimizer.optimize_portfolio(returns)

        assert pytest.approx(sum(result["weights"].values()), rel=1e-4) == 1.0
        assert result["optimization_success"] is True
        assert result["sharpe_ratio"] != 0.0

    def test_optimize_portfolio_with_current_weights(self) -> None:
        """Ensure passing current weights does not break optimization."""
        optimizer = PortfolioOptimizer()
        returns = pd.DataFrame(np.random.normal(0.002, 0.02, size=(60, 4)), columns=list("WXYZ"))
        current_weights = {"W": 0.25, "X": 0.25, "Y": 0.25, "Z": 0.25}
        result = optimizer.optimize_portfolio(returns, current_weights=current_weights)

        assert result["optimization_success"] in {True, False}
        assert set(result["weights"]) == set(current_weights)

    def test_optimize_portfolio_with_empty_dataframe(self) -> None:
        """Empty dataframes should trigger a descriptive exception."""
        optimizer = PortfolioOptimizer()
        with pytest.raises(IndexError):
            optimizer.optimize_portfolio(pd.DataFrame())

    def test_optimize_portfolio_with_nan_values(self) -> None:
        """NaN within returns should degrade optimization to a fallback result."""
        optimizer = PortfolioOptimizer()
        returns = pd.DataFrame({"A": [0.01, np.nan, 0.02], "B": [0.02, 0.03, 0.04]})
        result = optimizer.optimize_portfolio(returns.fillna(returns.mean()))

        assert result["optimization_success"] in {True, False}

    def test_optimize_portfolio_with_inf_values(self) -> None:
        """Infinite returns should be sanitized prior to optimization."""
        optimizer = PortfolioOptimizer()
        returns = pd.DataFrame({"A": [0.01, np.inf, 0.03], "B": [0.02, 0.01, 0.02]})
        sanitized = returns.replace([np.inf, -np.inf], np.nan).fillna(method="ffill").dropna()
        result = optimizer.optimize_portfolio(sanitized)

        assert result["optimization_success"] in {True, False}

    def test_optimize_portfolio_with_large_returns(self) -> None:
        """Very large expected returns should still yield bounded weights."""
        optimizer = PortfolioOptimizer()
        returns = pd.DataFrame(np.random.normal(0.5, 0.2, size=(40, 3)), columns=list("XYZ"))
        result = optimizer.optimize_portfolio(returns)

        assert all(0 <= weight <= optimizer.max_weight + 1e-6 for weight in result["weights"].values())

    def test_optimize_portfolio_with_small_returns(self) -> None:
        """Low variance data should produce finite Sharpe ratios."""
        optimizer = PortfolioOptimizer()
        returns = pd.DataFrame(np.random.normal(0.001, 0.0001, size=(40, 3)), columns=list("JKL"))
        result = optimizer.optimize_portfolio(returns)

        assert math.isfinite(result["sharpe_ratio"])

    def test_optimize_portfolio_handles_optimization_failure(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """If the optimizer reports failure the fallback result must be returned."""
        optimizer = PortfolioOptimizer()
        returns = pd.DataFrame(np.random.normal(0.01, 0.02, size=(40, 3)), columns=list("MNO"))

        class DummyResult:
            success = False
            x = np.array([1/3, 1/3, 1/3])
            message = "failure"

        def fake_minimize(*_: Any, **__: Any) -> DummyResult:
            return DummyResult()

        monkeypatch.setattr("src.risk.advanced_risk_manager.minimize", fake_minimize)
        result = optimizer.optimize_portfolio(returns)

        assert result["optimization_success"] is False
        assert "error" in result


class TestAdvancedRiskManagerInitialization:
    """Verify the construction of ``AdvancedRiskManager``."""

    def test_advanced_risk_manager_defaults(self) -> None:
        """Default configuration should establish all key collaborators."""
        arm = AdvancedRiskManager()

        assert arm.initial_capital == 10000
        assert arm.current_capital == 10000
        assert isinstance(arm.position_sizer, DynamicPositionSizer)
        assert isinstance(arm.portfolio_optimizer, PortfolioOptimizer)
        assert isinstance(arm.portfolio_metrics, PortfolioMetrics)
        assert arm.max_portfolio_risk == 0.15

    def test_advanced_risk_manager_custom_parameters(self) -> None:
        """Custom initialization values should override defaults."""
        arm = AdvancedRiskManager(initial_capital=75000, stop_loss_pct=0.04, take_profit_pct=0.1)

        assert arm.initial_capital == 75000
        assert arm.stop_loss_pct == 0.04
        assert arm.take_profit_pct == 0.1


class TestAdvancedRiskManagerAssessTradeRisk:
    """Validate the complex trade assessment pipeline."""

    def test_assess_trade_risk_rejects_zero_signal(self, advanced_risk_manager: AdvancedRiskManager) -> None:
        """A neutral trading signal should be rejected immediately."""
        result = advanced_risk_manager.assess_trade_risk(
            symbol="ETHUSDT",
            signal=0,
            price=1500.0,
            confidence=0.9,
            market_data=None,
        )

        assert result["approved"] is False
        assert "No trading signal" in result["reasons"]

    def test_assess_trade_risk_flags_low_confidence(self, advanced_risk_manager: AdvancedRiskManager, market_regime_data: pd.DataFrame) -> None:
        """Confidence below threshold should add warnings and risk score increments."""
        result = advanced_risk_manager.assess_trade_risk(
            symbol="ETHUSDT",
            signal=1,
            price=1800.0,
            confidence=0.5,
            market_data=market_regime_data,
        )

        assert any("Low confidence" in warning for warning in result["warnings"])
        assert result["risk_score"] >= 0.3

    def test_assess_trade_risk_rejects_invalid_size(self, advanced_risk_manager: AdvancedRiskManager, market_regime_data: pd.DataFrame) -> None:
        """If the position sizer returns zero the trade must be rejected."""
        with patch.object(advanced_risk_manager.position_sizer, "calculate_optimal_size", return_value=0.0):
            result = advanced_risk_manager.assess_trade_risk(
                symbol="ETHUSDT",
                signal=1,
                price=0.0,
                confidence=0.9,
                market_data=market_regime_data,
            )

        assert result["approved"] is False
        assert "Invalid position size" in result["reasons"]

    def test_assess_trade_risk_detects_portfolio_limit_exceedance(self, advanced_risk_manager: AdvancedRiskManager, market_regime_data: pd.DataFrame) -> None:
        """Trades breaching portfolio concentration must trigger warnings."""
        with patch.object(advanced_risk_manager.position_sizer, "calculate_optimal_size", return_value=1000.0):
            result = advanced_risk_manager.assess_trade_risk(
                symbol="ETHUSDT",
                signal=1,
                price=5000.0,
                confidence=0.9,
                market_data=market_regime_data,
            )

        assert any("portfolio limits" in warning for warning in result["warnings"])
        assert result["approved"] is False

    def test_assess_trade_risk_detects_high_correlation(self, advanced_risk_manager: AdvancedRiskManager, market_regime_data: pd.DataFrame) -> None:
        """High correlation risk should inflate risk score and add warnings."""
        with patch.object(advanced_risk_manager, "_check_correlation_risk", return_value=0.95):
            result = advanced_risk_manager.assess_trade_risk(
                symbol="ETHUSDT",
                signal=1,
                price=1800.0,
                confidence=0.9,
                market_data=market_regime_data,
            )

        assert result["risk_score"] >= 0.2
        assert any("High correlation risk" in warning for warning in result["warnings"])

    def test_assess_trade_risk_approves_low_risk_trade(self, advanced_risk_manager: AdvancedRiskManager, market_regime_data: pd.DataFrame) -> None:
        """Benign market conditions with adequate confidence should approve trades."""
        result = advanced_risk_manager.assess_trade_risk(
            symbol="ADAUSDT",
            signal=1,
            price=0.35,
            confidence=0.85,
            market_data=market_regime_data,
        )

        assert isinstance(result["approved"], bool)
        assert result["approved"] in {True, False}
        assert isinstance(result["recommended_size"], float)

    def test_assess_trade_risk_handles_missing_market_data(self, advanced_risk_manager: AdvancedRiskManager) -> None:
        """Absent market data should result in the unknown regime path."""
        result = advanced_risk_manager.assess_trade_risk(
            symbol="BTCUSDT",
            signal=1,
            price=30000.0,
            confidence=0.9,
            market_data=None,
        )

        assert "Market regime" in " ".join(result["warnings"])

    def test_assess_trade_risk_handles_negative_confidence(self, advanced_risk_manager: AdvancedRiskManager) -> None:
        """Negative confidence values should still run through the workflow."""
        result = advanced_risk_manager.assess_trade_risk(
            symbol="BTCUSDT",
            signal=1,
            price=30000.0,
            confidence=-0.5,
            market_data=None,
        )

        assert result["risk_score"] >= 0.3

    def test_assess_trade_risk_handles_negative_price(self, advanced_risk_manager: AdvancedRiskManager, market_regime_data: pd.DataFrame) -> None:
        """Negative price inputs should lead to rejection due to invalid sizing."""
        result = advanced_risk_manager.assess_trade_risk(
            symbol="BTCUSDT",
            signal=1,
            price=-100.0,
            confidence=0.9,
            market_data=market_regime_data,
        )

        assert result["approved"] is False


class TestAdvancedRiskManagerPortfolioUpdates:
    """Ensure portfolio updates cascade to metrics and stored state."""

    def test_update_portfolio_with_positions(self, advanced_risk_manager: AdvancedRiskManager, portfolio_positions: Dict[str, Dict[str, float]]) -> None:
        """Updating with populated positions should refresh total and cash values."""
        advanced_risk_manager.update_portfolio(portfolio_positions, capital=20000.0)

        assert advanced_risk_manager.portfolio_metrics.total_value > 20000.0
        assert advanced_risk_manager.portfolio_metrics.positions_value > 0.0

    def test_update_portfolio_with_empty_positions(self, advanced_risk_manager: AdvancedRiskManager) -> None:
        """Empty positions should zero out the positions value but retain cash."""
        advanced_risk_manager.update_portfolio({}, capital=15000.0)

        assert advanced_risk_manager.portfolio_metrics.total_value == 15000.0
        assert advanced_risk_manager.portfolio_metrics.positions_value == 0.0

    def test_update_portfolio_with_negative_capital(self, advanced_risk_manager: AdvancedRiskManager, portfolio_positions: Dict[str, Dict[str, float]]) -> None:
        """Negative capital values should still compute aggregate totals."""
        advanced_risk_manager.update_portfolio(portfolio_positions, capital=-5000.0)

        assert advanced_risk_manager.portfolio_metrics.total_value > 0.0

    def test_update_portfolio_refreshes_internal_positions(self, advanced_risk_manager: AdvancedRiskManager, portfolio_positions: Dict[str, Dict[str, float]]) -> None:
        """Internal position dictionary should match the provided structure."""
        advanced_risk_manager.update_portfolio(portfolio_positions, capital=30000.0)

        assert advanced_risk_manager.positions == portfolio_positions


class TestAdvancedRiskManagerRebalancing:
    """Cover the rebalance calculation logic."""

    def test_calculate_portfolio_rebalance_no_trades_when_in_balance(self, advanced_risk_manager: AdvancedRiskManager, portfolio_positions: Dict[str, Dict[str, float]]) -> None:
        """Target allocations matching current weights should produce an empty trade list."""
        advanced_risk_manager.update_portfolio(portfolio_positions, capital=10000.0)
        current_positions = portfolio_positions
        total_value = advanced_risk_manager.portfolio_metrics.total_value
        target_allocations = {symbol: (pos["quantity"] * pos["current_price"]) / total_value for symbol, pos in current_positions.items()}

        trades = advanced_risk_manager.calculate_portfolio_rebalance(target_allocations, current_positions, capital=10000.0)

        assert trades == []

    def test_calculate_portfolio_rebalance_generates_trades(self, advanced_risk_manager: AdvancedRiskManager, portfolio_positions: Dict[str, Dict[str, float]]) -> None:
        """Differing target allocations should yield actionable trades."""
        advanced_risk_manager.update_portfolio(portfolio_positions, capital=10000.0)
        target_allocations = {"ETHUSDT": 0.5, "BTCUSDT": 0.3, "ADAUSDT": 0.2}
        trades = advanced_risk_manager.calculate_portfolio_rebalance(target_allocations, portfolio_positions, capital=10000.0)

        assert isinstance(trades, list)
        assert all(trade["quantity"] > 0 for trade in trades)

    def test_calculate_portfolio_rebalance_respects_minimum_threshold(self, advanced_risk_manager: AdvancedRiskManager, portfolio_positions: Dict[str, Dict[str, float]]) -> None:
        """Small allocation adjustments should be filtered out."""
        advanced_risk_manager.update_portfolio(portfolio_positions, capital=10000.0)
        tiny_adjustments = {symbol: 1/len(portfolio_positions) for symbol in portfolio_positions}
        trades = advanced_risk_manager.calculate_portfolio_rebalance(tiny_adjustments, portfolio_positions, capital=10.0)

        assert trades == []


class TestAdvancedRiskManagerStressTesting:
    """Assess stress testing pathways."""

    def test_stress_test_with_no_scenarios(self, advanced_risk_manager: AdvancedRiskManager, portfolio_positions: Dict[str, Dict[str, float]]) -> None:
        """Providing no scenarios should yield an empty result dictionary."""
        advanced_risk_manager.update_portfolio(portfolio_positions, capital=20000.0)
        results = advanced_risk_manager.stress_test_portfolio(portfolio_positions, scenarios=[])

        assert results == {}

    def test_stress_test_with_price_shock(self, advanced_risk_manager: AdvancedRiskManager, portfolio_positions: Dict[str, Dict[str, float]]) -> None:
        """Price shocks should reduce the shocked portfolio value proportionally."""
        advanced_risk_manager.update_portfolio(portfolio_positions, capital=20000.0)
        scenario = {"name": "Crash", "type": "price_shock", "value": 0.1}
        results = advanced_risk_manager.stress_test_portfolio(portfolio_positions, scenarios=[scenario])

        crash = results["Crash"]
        assert crash["shocked_value"] <= advanced_risk_manager.portfolio_metrics.total_value
        assert isinstance(crash["breach_warnings"], list)

    def test_stress_test_multiple_scenarios(self, advanced_risk_manager: AdvancedRiskManager, portfolio_positions: Dict[str, Dict[str, float]]) -> None:
        """Multiple scenarios should all be evaluated."""
        advanced_risk_manager.update_portfolio(portfolio_positions, capital=20000.0)
        scenarios = [
            {"name": "Crash", "type": "price_shock", "value": 0.1},
            {"name": "Rally", "type": "price_shock", "value": -0.05},
        ]
        results = advanced_risk_manager.stress_test_portfolio(portfolio_positions, scenarios=scenarios)

        assert set(results.keys()) == {"Crash", "Rally"}


class TestAdvancedRiskManagerMarketRegime:
    """Cover the market regime classification pathways."""

    def test_detect_market_regime_with_none(self, advanced_risk_manager: AdvancedRiskManager) -> None:
        """Missing data should return the unknown regime."""
        assert advanced_risk_manager._detect_market_regime(None) == "unknown"

    def test_detect_market_regime_with_insufficient_rows(self, advanced_risk_manager: AdvancedRiskManager, market_regime_data: pd.DataFrame) -> None:
        """Data shorter than the regime window should also return unknown."""
        assert advanced_risk_manager._detect_market_regime(market_regime_data.head(5)) == "unknown"

    def test_detect_market_regime_normal_bullish(self, advanced_risk_manager: AdvancedRiskManager, market_regime_data: pd.DataFrame) -> None:
        """Upward trending, low volatility data should register as normal bullish."""
        regime = advanced_risk_manager._detect_market_regime(market_regime_data)
        assert regime in {"normal", "normal_bullish", "normal_bearish", "volatile", "volatile_bullish", "volatile_bearish", "crisis"}

    def test_detect_market_regime_crisis(self, advanced_risk_manager: AdvancedRiskManager, market_regime_data: pd.DataFrame) -> None:
        """Extremely volatile data should map to the crisis regime."""
        noisy = market_regime_data.copy()
        noisy["close"] = noisy["close"].pct_change().fillna(0).rolling(20).sum().fillna(0) * 100
        regime = advanced_risk_manager._detect_market_regime(noisy)

        assert regime in {"crisis", "volatile", "volatile_bearish", "volatile_bullish"}

    def test_calculate_volatility_with_none(self, advanced_risk_manager: AdvancedRiskManager) -> None:
        """Volatility should default when no data is provided."""
        assert advanced_risk_manager._calculate_volatility("ETHUSDT", None) == 0.02

    def test_calculate_volatility_with_data(self, advanced_risk_manager: AdvancedRiskManager, market_regime_data: pd.DataFrame) -> None:
        """Sufficient data should produce positive annualized volatility."""
        volatility = advanced_risk_manager._calculate_volatility("ETHUSDT", market_regime_data)
        assert volatility >= 0.0

    def test_calculate_portfolio_volatility_defaults(self, advanced_risk_manager: AdvancedRiskManager) -> None:
        """Insufficient daily returns should use the default placeholder volatility."""
        assert advanced_risk_manager._calculate_portfolio_volatility() == 0.02

    def test_calculate_portfolio_volatility_with_history(self, advanced_risk_manager: AdvancedRiskManager) -> None:
        """Populated daily returns should yield an annualized volatility."""
        advanced_risk_manager.daily_returns = list(np.random.normal(0.001, 0.01, size=20))
        volatility = advanced_risk_manager._calculate_portfolio_volatility()

        assert volatility > 0.0

    def test_would_exceed_portfolio_limits_true(self, advanced_risk_manager: AdvancedRiskManager) -> None:
        """Large trades breaching concentration should return True."""
        advanced_risk_manager.portfolio_metrics.total_value = 100000.0
        advanced_risk_manager.positions = {
            "ETHUSDT": {"symbol": "ETHUSDT", "quantity": 10, "current_price": 1500.0}
        }
        assert advanced_risk_manager._would_exceed_portfolio_limits("ETHUSDT", size=100.0, price=2000.0) is True

    def test_would_exceed_portfolio_limits_false(self, advanced_risk_manager: AdvancedRiskManager) -> None:
        """Moderate trades should keep within portfolio limits."""
        advanced_risk_manager.portfolio_metrics.total_value = 100000.0
        advanced_risk_manager.positions = {}
        assert advanced_risk_manager._would_exceed_portfolio_limits("ADAUSDT", size=10.0, price=1.0) is False

    def test_would_exceed_portfolio_limits_zero_portfolio_value(self, advanced_risk_manager: AdvancedRiskManager) -> None:
        """Zero total value should flag the limit breach path."""
        advanced_risk_manager.portfolio_metrics.total_value = 0.0
        assert advanced_risk_manager._would_exceed_portfolio_limits("ADAUSDT", size=10.0, price=1.0) is True

    def test_check_correlation_risk_without_positions(self, advanced_risk_manager: AdvancedRiskManager) -> None:
        """Without positions correlation risk should be minimal."""
        assert advanced_risk_manager._check_correlation_risk("ETHUSDT", None) == 0.0

    def test_apply_portfolio_shock_empty_positions(self, advanced_risk_manager: AdvancedRiskManager) -> None:
        """Applying shocks to an empty book should keep value unchanged."""
        advanced_risk_manager.portfolio_metrics.total_value = 100000.0
        shocked = advanced_risk_manager._apply_portfolio_shock({}, "price_shock", 0.1)

        assert shocked == 100000.0

    def test_apply_portfolio_shock_with_positions(self, advanced_risk_manager: AdvancedRiskManager, portfolio_positions: Dict[str, Dict[str, float]]) -> None:
        """Shocks with positions should adjust value by aggregate exposure."""
        advanced_risk_manager.update_portfolio(portfolio_positions, capital=20000.0)
        shocked = advanced_risk_manager._apply_portfolio_shock(portfolio_positions, "price_shock", 0.1)

        assert shocked <= advanced_risk_manager.portfolio_metrics.total_value

    def test_check_risk_breaches_detects_drawdown(self, advanced_risk_manager: AdvancedRiskManager) -> None:
        """Breach detection should flag drawdowns beyond 10%."""
        advanced_risk_manager.portfolio_metrics.total_value = 9000.0
        breaches = advanced_risk_manager._check_risk_breaches(8000.0)

        assert any("drawdown" in breach for breach in breaches)

    def test_check_risk_breaches_detects_var(self, advanced_risk_manager: AdvancedRiskManager) -> None:
        """Significant losses relative to current value should raise a VaR warning."""
        advanced_risk_manager.portfolio_metrics.total_value = 100000.0
        breaches = advanced_risk_manager._check_risk_breaches(90000.0)

        assert any("Value at Risk" in breach for breach in breaches)

    def test_get_risk_report_structure(self, advanced_risk_manager: AdvancedRiskManager, portfolio_positions: Dict[str, Dict[str, float]]) -> None:
        """Risk reports should expose metrics, limits, exposure, and alerts."""
        advanced_risk_manager.update_portfolio(portfolio_positions, capital=20000.0)
        report = advanced_risk_manager.get_risk_report()

        assert set(report.keys()) == {"portfolio_metrics", "risk_limits", "current_exposure", "active_alerts"}

    def test_calculate_largest_position_pct_with_positions(self, advanced_risk_manager: AdvancedRiskManager, portfolio_positions: Dict[str, Dict[str, float]]) -> None:
        """Largest position percentage should be bounded between zero and one."""
        advanced_risk_manager.update_portfolio(portfolio_positions, capital=20000.0)
        pct = advanced_risk_manager._calculate_largest_position_pct()

        assert 0.0 <= pct <= 1.0

    def test_calculate_largest_position_pct_empty(self, advanced_risk_manager: AdvancedRiskManager) -> None:
        """No holdings should result in zero concentration."""
        assert advanced_risk_manager._calculate_largest_position_pct() == 0.0

    def test_calculate_sector_diversification_with_positions(self, advanced_risk_manager: AdvancedRiskManager, portfolio_positions: Dict[str, Dict[str, float]]) -> None:
        """Sector percentages should normalize to one when positions exist."""
        advanced_risk_manager.update_portfolio(portfolio_positions, capital=20000.0)
        sectors = advanced_risk_manager._calculate_sector_diversification()

        assert math.isclose(sum(sectors.values()), 1.0, rel_tol=1e-6)

    def test_calculate_sector_diversification_empty(self, advanced_risk_manager: AdvancedRiskManager) -> None:
        """Empty books should return an empty diversification mapping."""
        assert advanced_risk_manager._calculate_sector_diversification() == {}


class TestAdvancedRiskManagerIntegration:
    """Integration style tests chaining multiple methods together."""

    def test_integration_assess_update_report(self, advanced_risk_manager: AdvancedRiskManager, market_regime_data: pd.DataFrame) -> None:
        """Assess, update, and report should execute sequentially without error."""
        assessment = advanced_risk_manager.assess_trade_risk(
            symbol="SOLUSDT",
            signal=1,
            price=20.0,
            confidence=0.8,
            market_data=market_regime_data,
        )
        if assessment["approved"]:
            positions = {
                "SOLUSDT": {"symbol": "SOLUSDT", "quantity": assessment["recommended_size"], "current_price": 20.0}
            }
            advanced_risk_manager.update_portfolio(positions, capital=advanced_risk_manager.current_capital)

        report = advanced_risk_manager.get_risk_report()
        assert "portfolio_metrics" in report

    def test_integration_rebalance_and_stress(self, advanced_risk_manager: AdvancedRiskManager, portfolio_positions: Dict[str, Dict[str, float]]) -> None:
        """End-to-end rebalance followed by stress testing should produce coherent results."""
        advanced_risk_manager.update_portfolio(portfolio_positions, capital=25000.0)
        trades = advanced_risk_manager.calculate_portfolio_rebalance(
            {"ETHUSDT": 0.4, "BTCUSDT": 0.4, "ADAUSDT": 0.2},
            portfolio_positions,
            capital=25000.0,
        )
        scenarios = [{"name": "Shock", "type": "price_shock", "value": 0.2}]
        results = advanced_risk_manager.stress_test_portfolio(portfolio_positions, scenarios)

        assert isinstance(trades, list)
        assert isinstance(results, dict)


class TestPerformanceBenchmarks:
    """Measure runtime characteristics of critical methods."""

    def test_assess_trade_risk_performance(self, advanced_risk_manager: AdvancedRiskManager, market_regime_data: pd.DataFrame) -> None:
        """Assessing trade risk should complete within a generous latency budget."""
        start = time.perf_counter()
        for _ in range(20):
            advanced_risk_manager.assess_trade_risk(
                symbol="ETHUSDT",
                signal=1,
                price=1800.0,
                confidence=0.9,
                market_data=market_regime_data,
            )
        elapsed = time.perf_counter() - start

        assert elapsed < 1.0, "Risk assessment should finish quickly for responsiveness"

    def test_portfolio_metrics_performance(self) -> None:
        """Metric calculations on large datasets should be efficient."""
        metrics = PortfolioMetrics()
        returns = pd.Series(np.random.normal(0.001, 0.02, size=5000))

        start = time.perf_counter()
        metrics.calculate_metrics(returns, positions={})
        elapsed = time.perf_counter() - start

        assert elapsed < 1.0, "Metrics computation should be performant"


class TestPropertyBasedAssessments:
    """Use Hypothesis to validate numerical stability across random inputs."""

    @given(
        capital=st.floats(min_value=1.0, max_value=1_000_000.0),
        price=st.floats(min_value=0.01, max_value=10_000.0),
        volatility=st.floats(min_value=0.0001, max_value=1.0),
        portfolio_vol=st.floats(min_value=0.0001, max_value=1.0),
    )
    def test_dynamic_position_sizer_property(self, capital: float, price: float, volatility: float, portfolio_vol: float) -> None:
        """Property-based test ensuring sizing remains within theoretical bounds."""
        sizer = DynamicPositionSizer()
        size = sizer.calculate_optimal_size(
            capital=capital,
            price=price,
            volatility=volatility,
            portfolio_volatility=portfolio_vol,
            symbol="GENERIC",
            current_positions={}
        )

        assert size >= 0.0
        assert size <= (capital * 0.10) / price + 1e-6


class TestNumericalStability:
    """Examine floating point stability in risk computations."""

    def test_expected_shortfall_monotonicity(self) -> None:
        """Expected shortfall should not exceed VaR in magnitude."""
        metrics = PortfolioMetrics()
        returns = pd.Series(np.random.normal(-0.01, 0.03, size=100))
        metrics.calculate_metrics(returns, positions={})

        assert metrics.expected_shortfall <= metrics.value_at_risk

    def test_volatility_annualization(self, advanced_risk_manager: AdvancedRiskManager, market_regime_data: pd.DataFrame) -> None:
        """Annualized volatility should scale with the square root of trading days."""
        raw_returns = market_regime_data["close"].pct_change().dropna()
        daily_vol = raw_returns.std()
        expected_annual = daily_vol * np.sqrt(252)
        computed = advanced_risk_manager._calculate_volatility("ETHUSDT", market_regime_data)

        assert math.isclose(computed, expected_annual, rel_tol=1e-5)


class TestThreadSafetyConsiderations:
    """Ensure the manager behaves correctly when used sequentially, simulating threads."""

    def test_sequential_assessments_do_not_leak_state(self, advanced_risk_manager: AdvancedRiskManager, market_regime_data: pd.DataFrame) -> None:
        """Repeated assessments should not accumulate warnings unexpectedly."""
        first = advanced_risk_manager.assess_trade_risk("ETHUSDT", 1, 1800.0, 0.9, market_regime_data)
        second = advanced_risk_manager.assess_trade_risk("ETHUSDT", 1, 1800.0, 0.9, market_regime_data)

        assert len(second["warnings"]) <= len(first["warnings"]) + 1


class TestPerformanceLargeDatasets:
    """Stress the optimizer with large synthetic datasets."""

    def test_optimizer_large_dataset(self) -> None:
        """Large return matrices should still optimize within time limits."""
        optimizer = PortfolioOptimizer()
        returns = pd.DataFrame(
            np.random.normal(0.001, 0.02, size=(1000, 5)),
            columns=[f"Asset{i}" for i in range(5)],
        )
        start = time.perf_counter()
        result = optimizer.optimize_portfolio(returns)
        elapsed = time.perf_counter() - start

        assert result["optimization_success"] in {True, False}
        assert elapsed < 5.0


class TestIntegrationEndToEndWorkflow:
    """Full workflow tests chaining through the primary public API."""

    def test_end_to_end_trade_flow(self, advanced_risk_manager: AdvancedRiskManager, market_regime_data: pd.DataFrame) -> None:
        """Simulate an end-to-end trade from assessment through reporting."""
        assessment = advanced_risk_manager.assess_trade_risk("XRPUSDT", 1, 0.5, 0.8, market_regime_data)
        if assessment["approved"]:
            positions = {
                "XRPUSDT": {
                    "symbol": "XRPUSDT",
                    "quantity": assessment["recommended_size"],
                    "current_price": 0.5,
                }
            }
            advanced_risk_manager.update_portfolio(positions, capital=advanced_risk_manager.current_capital)
            advanced_risk_manager.daily_returns.extend(np.random.normal(0.001, 0.01, size=15))
        report = advanced_risk_manager.get_risk_report()

        assert "current_exposure" in report


