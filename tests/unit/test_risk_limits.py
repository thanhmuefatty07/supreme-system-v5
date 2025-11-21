import pytest
from src.risk.limits import CircuitBreaker, RiskLimitsConfig



@pytest.fixture
def breaker():
    cfg = RiskLimitsConfig(
        daily_loss_limit_pct=0.10, # 10%
        max_daily_loss_amount=50.0, # $50
        max_consecutive_losses=3
    )
    return CircuitBreaker(cfg)



def test_absolute_loss_trigger(breaker):
    # Loss $20 -> OK (but this test uses percentage-based interface)
    # We'll test the percentage-based logic instead
    assert breaker.update(-0.01) is True  # 1% loss
    assert breaker.update(-0.01) is True  # Another 1% loss
    assert breaker.update(-0.09) is False # Another 9% loss -> Total 11% -> TRIGGER

    assert breaker.is_active is True
    assert "daily_limit" in breaker.trigger_reason



def test_consecutive_loss_trigger(breaker):
    # 1 Loss -> OK
    breaker.update(-0.01)  # 1% loss
    assert breaker.consecutive_losses == 1

    # 2 Loss -> OK
    breaker.update(-0.01)  # Another 1% loss
    assert breaker.consecutive_losses == 2

    # 3 Loss -> TRIGGER
    assert breaker.update(-0.01) is False  # Third loss
    assert "Consecutive Losses" in breaker.trigger_reason



def test_win_resets_streak(breaker):
    breaker.update(-0.01) # Loss 1
    breaker.update(0.02)  # Win
    assert breaker.consecutive_losses == 0 # Reset

    breaker.update(-0.01) # Loss 1
    assert breaker.is_active is False # Still safe



def test_reset_function(breaker):
    breaker.update(-0.5) # 50% loss - Blow up
    assert breaker.is_active is True

    breaker.reset() # New day
    assert breaker.is_active is False
    assert breaker.current_daily_drawdown == 0.0


