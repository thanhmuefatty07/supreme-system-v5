"""

Property-Based Tests for Adaptive Kelly Risk Manager.

Uses Hypothesis to verify financial invariants.

"""

import pytest
from hypothesis import given, strategies as st, assume
from src.risk.adaptive_kelly import AdaptiveKellyRiskManager, RiskConfig


# Strategy Definitions
capital_strategy = st.floats(min_value=1000.0, max_value=1_000_000.0)
pnl_strategy = st.floats(min_value=-1000.0, max_value=1000.0)
bool_strategy = st.booleans()


@given(capital=capital_strategy)
def test_property_position_never_exceeds_capital(capital):
    """Property: Position size must never exceed current capital."""
    config = RiskConfig()
    manager = AdaptiveKellyRiskManager(config=config, current_capital=capital)

    for mode in ['full', 'half', 'quarter']:
        size = manager.get_target_size(mode=mode)
        assert 0.0 <= size <= capital, f"Position {size} exceeds capital {capital}"


@given(
    capital=capital_strategy,
    was_win=bool_strategy,
    pnl=pnl_strategy
)
def test_property_ewma_win_rate_bounded(capital, was_win, pnl):
    """Property: EWMA win rate must stay in [0, 1]."""
    config = RiskConfig()
    manager = AdaptiveKellyRiskManager(config=config, current_capital=capital)

    manager.update_performance(was_win, pnl)

    assert 0.0 <= manager.ewma_win_rate <= 1.0


@given(capital=capital_strategy)
def test_property_circuit_breaker_triggers_on_loss_limit(capital):
    """Property: Circuit breaker MUST trigger when daily loss > limit."""
    config = RiskConfig(max_daily_loss_pct=0.05)
    manager = AdaptiveKellyRiskManager(config=config, current_capital=capital)

    # Simulate losses totaling 6% (should trigger 5% limit)
    loss_amount = -0.02 * capital
    manager.update_performance(False, loss_amount)  # -2%
    manager.update_performance(False, loss_amount)  # -2%
    manager.update_performance(False, loss_amount)  # -2% (total -6%)

    assert manager.is_halted, "Circuit breaker should have triggered"
    assert "Daily Loss" in manager.halt_reason


@given(capital=capital_strategy)
def test_property_consecutive_loss_halt(capital):
    """Property: Must halt after N consecutive losses."""
    config = RiskConfig(max_consecutive_losses=3)
    manager = AdaptiveKellyRiskManager(config=config, current_capital=capital)

    # 3 small losses
    for _ in range(3):
        manager.update_performance(False, -10.0)

    assert manager.is_halted
    assert "Consecutive" in manager.halt_reason


@given(capital=capital_strategy)
def test_property_win_resets_consecutive_losses(capital):
    """Property: Win should reset consecutive loss counter."""
    config = RiskConfig()
    manager = AdaptiveKellyRiskManager(config=config, current_capital=capital)

    manager.update_performance(False, -10.0)
    manager.update_performance(False, -10.0)
    assert manager.consecutive_losses == 2

    manager.update_performance(True, 20.0)  # Win
    assert manager.consecutive_losses == 0


@given(capital=capital_strategy)
def test_property_halted_returns_zero_size(capital):
    """Property: When halted, position size must be 0."""
    config = RiskConfig(max_daily_loss_pct=0.01)
    manager = AdaptiveKellyRiskManager(config=config, current_capital=capital)

    # Trigger halt
    manager.update_performance(False, -0.02 * capital)

    assert manager.is_halted
    assert manager.get_target_size('full') == 0.0
