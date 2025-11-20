import pytest
from hypothesis import given, strategies as st
from src.risk.calculations import calculate_kelly_criterion, calculate_position_size, KellyInput



# --- UNIT TESTS (Classic) ---



def test_kelly_basic_scenario():
    # Win rate 50%, Reward/Risk 2.0
    # f = (0.5 * 3 - 1) / 2 = 0.5 / 2 = 0.25
    inp = KellyInput(win_rate=0.5, reward_risk_ratio=2.0)
    assert calculate_kelly_criterion(inp) == 0.25



def test_kelly_negative_expectation():
    # Win rate 10%, Reward/Risk 1.0 -> Expectation negative -> Should trade 0
    inp = KellyInput(win_rate=0.1, reward_risk_ratio=1.0)
    assert calculate_kelly_criterion(inp) == 0.0



def test_kelly_invalid_inputs():
    assert calculate_kelly_criterion(KellyInput(1.5, 2.0)) == 0.0 # Invalid win rate
    assert calculate_kelly_criterion(KellyInput(0.5, -1.0)) == 0.0 # Invalid RR



def test_position_sizing_modes():
    cap = 10000
    k = 0.5 # 50% Kelly

    # Full: 5000
    assert calculate_position_size(cap, k, 1.0, 'full') == 5000.0
    # Half: 2500
    assert calculate_position_size(cap, k, 1.0, 'half') == 2500.0
    # Cap at 10% (1000)
    assert calculate_position_size(cap, k, 0.1, 'full') == 1000.0



# --- PROPERTY-BASED TESTS (Hypothesis) ---



@given(
    win_rate=st.floats(min_value=0.0, max_value=1.0),
    rr_ratio=st.floats(min_value=0.1, max_value=100.0)
)
def test_kelly_always_between_0_and_1(win_rate, rr_ratio):
    inp = KellyInput(win_rate, rr_ratio)
    result = calculate_kelly_criterion(inp)
    assert 0.0 <= result <= 1.0



@given(
    capital=st.floats(min_value=1.0, max_value=1e9),
    kelly=st.floats(min_value=0.0, max_value=1.0),
    max_risk=st.floats(min_value=0.0, max_value=1.0)
)
def test_size_never_exceeds_capital_or_cap(capital, kelly, max_risk):
    size = calculate_position_size(capital, kelly, max_risk, 'full')

    # Invariant 1: Size <= Capital
    assert size <= capital

    # Invariant 2: Size <= Max Risk Cap * Capital
    # (Allowing for tiny floating point errors)
    assert size <= (capital * max_risk) + 1e-9
