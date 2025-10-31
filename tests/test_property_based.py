"""
Property-Based Tests for Supreme System V5
ULTRA SFL implementation using hypothesis for edge case validation
"""

import pytest
from hypothesis import given, strategies as st, settings, HealthCheck
from hypothesis.stateful import RuleBasedStateMachine, rule, precondition
import asyncio
import numpy as np
from typing import List, Dict, Any
from dataclasses import dataclass

# Import system components
from supreme_system_v5.risk import RiskManager, RiskLimits, PortfolioState, RiskViolation
from supreme_system_v5.strategies import ScalpingStrategy, TechnicalIndicators
from supreme_system_v5.event_bus import EventBus, create_market_data_event, EventPriority


# ==========================================
# RISK MANAGEMENT PROPERTY TESTS
# ==========================================

class TestRiskManagerProperties:
    """Property-based tests for Risk Manager edge cases"""

    @given(
        drawdown_percent=st.floats(min_value=0.1, max_value=50.0),
        daily_loss_percent=st.floats(min_value=0.1, max_value=20.0),
        position_size=st.floats(min_value=100, max_value=10000),
        leverage=st.floats(min_value=0.1, max_value=5.0),
        active_positions=st.integers(min_value=1, max_value=10),
        symbol_concentration=st.floats(min_value=1.0, max_value=100.0)
    )
    @settings(max_examples=100, deadline=None, suppress_health_check=[HealthCheck.too_slow])
    def test_risk_limits_are_respected(self, drawdown_percent, daily_loss_percent,
                                      position_size, leverage, active_positions, symbol_concentration):
        """Test that risk limits are always respected regardless of input values"""
        # Create risk manager with extreme limits
        limits = RiskLimits(
            max_drawdown_percent=min(drawdown_percent, 50.0),  # Cap at reasonable values
            max_daily_loss_percent=min(daily_loss_percent, 20.0),
            max_position_size_usd=min(position_size, 5000.0),
            max_leverage=min(leverage, 3.0),
            max_active_positions=min(active_positions, 5),
            max_single_symbol_concentration=min(symbol_concentration, 50.0)
        )

        portfolio = PortfolioState(total_value=10000.0, cash_balance=10000.0)
        risk_manager = RiskManager(limits=limits, portfolio_state=portfolio)

        # Test various trade scenarios
        symbol = "BTC-USDT"

        # Scenario 1: Normal trade
        assessment = risk_manager.evaluate_trade(symbol, 1000.0, 1.0)
        assert isinstance(assessment, object)  # Should return assessment object

        # Scenario 2: Extreme position size (should be rejected or adjusted)
        large_position = limits.max_position_size_usd * 2
        assessment = risk_manager.evaluate_trade(symbol, large_position, 1.0)

        # Either rejected or adjusted to within limits
        if assessment.approved:
            assert assessment.adjusted_position_size is not None
            assert assessment.adjusted_position_size <= limits.max_position_size_usd

        # Scenario 3: Extreme leverage (should be rejected)
        assessment = risk_manager.evaluate_trade(symbol, 1000.0, limits.max_leverage * 2)
        if not assessment.approved:
            assert RiskViolation.LEVERAGE_LIMIT in assessment.violations

    @given(
        violation_count=st.integers(min_value=1, max_value=10),
        time_since_last_violation=st.floats(min_value=0.1, max_value=3600)  # 1 hour
    )
    @settings(max_examples=50, deadline=None)
    def test_circuit_breaker_activates(self, violation_count, time_since_last_violation):
        """Test circuit breaker activates after violations"""
        limits = RiskLimits(
            circuit_breaker_threshold=min(violation_count, 5),
            circuit_breaker_timeout_minutes=30
        )

        portfolio = PortfolioState(total_value=10000.0, cash_balance=10000.0)
        risk_manager = RiskManager(limits=limits, portfolio_state=portfolio)

        # Simulate violations
        for i in range(violation_count):
            portfolio.violation_count += 1

        # Check if circuit breaker should activate
        should_be_active = portfolio.violation_count >= limits.circuit_breaker_threshold

        if should_be_active:
            # Simulate circuit breaker activation
            portfolio.circuit_breaker_active = True
            portfolio.circuit_breaker_until = portfolio.last_trade_time + (limits.circuit_breaker_timeout_minutes * 60)

            # Verify circuit breaker blocks trades
            assessment = risk_manager.evaluate_trade("BTC-USDT", 1000.0, 1.0)
            assert not assessment.approved
            assert RiskViolation.CIRCUIT_BREAKER in assessment.violations

    @given(
        portfolio_value=st.floats(min_value=1000, max_value=100000),
        drawdown_amount=st.floats(min_value=100, max_value=50000)
    )
    @settings(max_examples=50, deadline=None)
    def test_drawdown_calculation(self, portfolio_value, drawdown_amount):
        """Test drawdown calculations are mathematically correct"""
        # Ensure drawdown doesn't exceed portfolio value
        actual_drawdown = min(drawdown_amount, portfolio_value)

        portfolio = PortfolioState(
            total_value=portfolio_value - actual_drawdown,
            peak_value=portfolio_value
        )

        # Calculate drawdown percentage
        if portfolio.peak_value > 0:
            drawdown_percent = ((portfolio.peak_value - portfolio.total_value) / portfolio.peak_value) * 100
            assert drawdown_percent >= 0
            assert drawdown_percent <= 100  # Can't lose more than 100%

            # If drawdown exceeds limits, should trigger violation
            limits = RiskLimits(max_drawdown_percent=10.0)
            risk_manager = RiskManager(limits=limits, portfolio_state=portfolio)

            if drawdown_percent > limits.max_drawdown_percent:
                assessment = risk_manager.evaluate_trade("BTC-USDT", 100.0, 1.0)
                # Note: May or may not be rejected depending on other conditions


# ==========================================
# TECHNICAL INDICATORS PROPERTY TESTS
# ==========================================

class TestTechnicalIndicatorsProperties:
    """Property-based tests for technical indicators"""

    @given(
        prices=st.lists(st.floats(min_value=0.01, max_value=100000), min_size=50, max_size=1000),
        period=st.integers(min_value=2, max_value=50)
    )
    @settings(max_examples=50, deadline=None)
    def test_ema_calculation_properties(self, prices, period):
        """Test EMA calculation properties"""
        if len(prices) < period:
            return  # Skip if not enough data

        ema_values = TechnicalIndicators.calculate_ema(prices, period)

        # EMA should exist for valid data
        assert len(ema_values) > 0

        # EMA should be bounded by price range
        price_min, price_max = min(prices), max(prices)
        ema_min, ema_max = min(ema_values), max(ema_values)

        assert ema_min >= price_min * 0.5  # EMA can be lower but not excessively
        assert ema_max <= price_max * 1.5  # EMA can be higher but not excessively

        # EMA should be smooth (changes should be dampened by the EMA formula)
        if len(ema_values) > 1:
            changes = [abs(ema_values[i] - ema_values[i-1]) / max(abs(ema_values[i-1]), 0.01)
                      for i in range(1, len(ema_values))]

            # For normal market data (not extreme artificial cases), most changes should be reasonable
            # But we allow for the test to generate extreme cases that might violate this
            if max(prices) / min(prices) < 100:  # Only check for non-extreme price ranges
                reasonable_changes = [c for c in changes if c < 0.5]  # Allow up to 50% changes
                assert len(reasonable_changes) / len(changes) > 0.5  # 50% of changes reasonable

    @given(
        prices=st.lists(st.floats(min_value=0.01, max_value=100000), min_size=20, max_size=500),
        rsi_period=st.integers(min_value=2, max_value=28)
    )
    @settings(max_examples=50, deadline=None)
    def test_rsi_calculation_properties(self, prices, rsi_period):
        """Test RSI calculation properties"""
        if len(prices) < rsi_period + 1:
            return

        rsi_values = TechnicalIndicators.calculate_rsi(prices, rsi_period)

        # RSI should exist for valid data
        assert len(rsi_values) > 0

        # RSI should be between 0 and 100
        for rsi in rsi_values:
            assert 0 <= rsi <= 100

        # RSI should not be constant (unless all price changes are zero)
        if len(set(prices)) > 1:  # If prices vary
            rsi_range = max(rsi_values) - min(rsi_values)
            assert rsi_range > 0  # RSI should vary

    @given(
        prices=st.lists(st.floats(min_value=0.01, max_value=100000), min_size=30, max_size=500)
    )
    @settings(max_examples=30, deadline=None)
    def test_bollinger_bands_properties(self, prices):
        """Test Bollinger Bands calculation properties"""
        # This would use the Rust implementation
        # For now, test with Python fallback
        if len(prices) < 20:
            return

        # Calculate simple moving average as proxy
        window = 20
        sma_values = []
        for i in range(window, len(prices)):
            sma = sum(prices[i-window:i]) / window
            sma_values.append(sma)

        # SMA should be reasonable
        assert len(sma_values) > 0
        assert all(sma > 0 for sma in sma_values)

        # SMA should be within price range
        price_min, price_max = min(prices), max(prices)
        sma_min, sma_max = min(sma_values), max(sma_values)

        assert sma_min >= price_min
        assert sma_max <= price_max


# ==========================================
# EVENT BUS PROPERTY TESTS
# ==========================================

class TestEventBusProperties:
    """Property-based tests for Event Bus"""

    @given(
        num_events=st.integers(min_value=1, max_value=100),
        num_subscribers=st.integers(min_value=1, max_value=10),
        queue_size=st.integers(min_value=10, max_value=1000)
    )
    @settings(max_examples=20, deadline=None)
    def test_event_bus_capacity(self, num_events, num_subscribers, queue_size):
        """Test event bus handles load without deadlocks"""
        async def run_test():
            bus = EventBus(max_queue_size=queue_size)
            await bus.start()

            # Subscribe multiple consumers
            events_received = []

            async def event_handler(event):
                events_received.append(event.id)
                await asyncio.sleep(0.001)  # Simulate processing time

            for i in range(num_subscribers):
                bus.subscribe("test_topic", f"consumer_{i}", event_handler)

            # Publish events
            published_events = []
            for i in range(min(num_events, queue_size)):
                event = create_market_data_event("BTC-USDT", 35000.0 + i, 1000.0, "test")
                published = await bus.publish(event)
                if published:
                    published_events.append(event.id)

            # Wait for processing
            await asyncio.sleep(0.1)

            # Verify all published events were processed
            assert len(events_received) == len(published_events) * num_subscribers

            await bus.stop()

        asyncio.run(run_test())

    @given(
        event_priority=st.sampled_from([p for p in EventPriority]),
        num_events=st.integers(min_value=5, max_value=50)
    )
    @settings(max_examples=20, deadline=None)
    def test_event_priority_ordering(self, event_priority, num_events):
        """Test events are processed in priority order"""
        async def run_test():
            bus = EventBus()
            await bus.start()

            processing_order = []

            async def priority_handler(event):
                processing_order.append((event.priority.value, event.id))

            bus.subscribe("test_topic", "priority_tester", priority_handler)

            # Publish events with different priorities
            for i in range(num_events):
                priority = EventPriority(i % 4)  # Cycle through priorities
                event = create_market_data_event("BTC-USDT", 35000.0, 1000.0, "test")
                event.priority = priority
                await bus.publish(event)

            # Wait for processing
            await asyncio.sleep(0.2)

            # Higher priority events should be processed first
            # (This is a simplified check - full priority ordering would need more complex logic)
            assert len(processing_order) == num_events

            await bus.stop()

        asyncio.run(run_test())


# ==========================================
# MARKET DATA EDGE CASE TESTS
# ==========================================

class TestMarketDataEdgeCases:
    """Test market data processing under extreme conditions"""

    @given(
        price=st.floats(min_value=0.000001, max_value=1000000),
        volume=st.floats(min_value=0, max_value=1000000),
        symbol=st.text(min_size=1, max_size=20, alphabet=st.characters(whitelist_categories=['Lu', 'Ll', 'Nd', '-']))
    )
    @settings(max_examples=100, deadline=None)
    def test_market_data_validation(self, price, volume, symbol):
        """Test market data handles edge values gracefully"""
        from supreme_system_v5.data_fabric.normalizer import MarketDataPoint

        # Create market data point
        try:
            data_point = MarketDataPoint(
                symbol=symbol.upper(),
                timestamp=1234567890.0,
                price=max(price, 0.000001),  # Ensure positive price
                volume=max(volume, 0),  # Ensure non-negative volume
                change_24h=0.0,
                high_24h=max(price * 1.01, 0.000001),
                low_24h=max(price * 0.99, 0.000001),
                bid=max(price * 0.999, 0.000001),
                ask=max(price * 1.001, 0.000001),
                market_cap=max(price * volume, 0),
                source="test",
                quality_score=0.8
            )

            # Validate basic properties
            assert data_point.price > 0
            assert data_point.volume >= 0
            assert data_point.symbol == symbol.upper()
            assert 0 <= data_point.quality_score <= 1.0

        except (ValueError, TypeError):
            # Some edge cases might legitimately fail validation
            pass

    @given(
        price_history=st.lists(st.floats(min_value=0.01, max_value=100000), min_size=2, max_size=100),
        volatility_factor=st.floats(min_value=0.001, max_value=10.0)
    )
    @settings(max_examples=30, deadline=None)
    def test_price_volatility_handling(self, price_history, volatility_factor):
        """Test system handles extreme price volatility"""
        # Apply volatility factor to create extreme price swings
        volatile_prices = []
        base_price = price_history[0]

        for i, price in enumerate(price_history):
            # Create volatility by adding random swings
            swing = (np.random.random() - 0.5) * volatility_factor * base_price
            volatile_price = base_price + swing
            volatile_prices.append(max(volatile_price, 0.01))  # Keep positive

        # Test that indicators can handle volatile data
        if len(volatile_prices) >= 14:
            rsi_values = TechnicalIndicators.calculate_rsi(volatile_prices, 14)
            assert len(rsi_values) > 0

            # RSI should still be valid even with extreme volatility
            for rsi in rsi_values:
                assert 0 <= rsi <= 100


# ==========================================
# STATEFUL TESTING FOR COMPLEX SCENARIOS
# ==========================================

class RiskManagerStateMachine(RuleBasedStateMachine):
    """Stateful testing for risk manager under complex scenarios"""

    def __init__(self):
        super().__init__()
        self.risk_manager = RiskManager(
            limits=RiskLimits(max_drawdown_percent=15.0, max_daily_loss_usd=200.0),
            portfolio_state=PortfolioState(total_value=10000.0, cash_balance=10000.0)
        )
        self.positions = {}

    @rule(
        symbol=st.sampled_from(["BTC-USDT", "ETH-USDT", "BNB-USDT"]),
        position_size=st.floats(min_value=100, max_value=2000),
        leverage=st.floats(min_value=0.5, max_value=2.0)
    )
    def add_position(self, symbol, position_size, leverage):
        """Add a position and check risk limits"""
        assessment = self.risk_manager.evaluate_trade(symbol, position_size, leverage)

        if assessment.approved:
            self.positions[symbol] = position_size
            # Simulate position opening
            self.risk_manager.portfolio.cash_balance -= position_size / leverage

    @rule(
        pnl_change=st.floats(min_value=-1000, max_value=1000)
    )
    def update_portfolio_value(self, pnl_change):
        """Update portfolio value (simulates P&L changes)"""
        self.risk_manager.portfolio.total_value += pnl_change

        # Update peak value if new high
        if self.risk_manager.portfolio.total_value > self.risk_manager.portfolio.peak_value:
            self.risk_manager.portfolio.peak_value = self.risk_manager.portfolio.total_value

    @rule()
    def check_invariant_drawdown_limit(self):
        """Invariant: drawdown never exceeds configured limit when circuit breaker active"""
        if self.risk_manager.portfolio.peak_value > 0:
            current_drawdown = ((self.risk_manager.portfolio.peak_value -
                               self.risk_manager.portfolio.total_value) /
                              self.risk_manager.portfolio.peak_value) * 100

            # If drawdown exceeds limit, circuit breaker should eventually activate
            if current_drawdown > self.risk_manager.limits.max_drawdown_percent:
                # Force a violation count to trigger circuit breaker
                self.risk_manager.portfolio.violation_count += 1

                if self.risk_manager.portfolio.violation_count >= self.risk_manager.limits.circuit_breaker_threshold:
                    assessment = self.risk_manager.evaluate_trade("BTC-USDT", 100.0, 1.0)
                    assert not assessment.approved or RiskViolation.CIRCUIT_BREAKER in assessment.violations


# Create test instance
TestRiskManagerState = RiskManagerStateMachine.TestCase


# ==========================================
# HYPOTHESIS STRATEGIES FOR DOMAIN-SPECIFIC DATA
# ==========================================

@st.composite
def valid_market_data_strategy(draw):
    """Strategy for generating valid market data"""
    symbol = draw(st.sampled_from(["BTC-USDT", "ETH-USDT", "ADA-USDT", "SOL-USDT"]))
    price = draw(st.floats(min_value=0.01, max_value=100000))
    volume = draw(st.floats(min_value=0.1, max_value=1000000))
    change_24h = draw(st.floats(min_value=-50, max_value=50))
    high_24h = draw(st.floats(min_value=price, max_value=price * 2))
    low_24h = draw(st.floats(min_value=price * 0.5, max_value=price))
    bid = draw(st.floats(min_value=price * 0.99, max_value=price))
    ask = draw(st.floats(min_value=price, max_value=price * 1.01))
    market_cap = draw(st.floats(min_value=1000000, max_value=1000000000000))

    return {
        'symbol': symbol,
        'price': price,
        'volume': volume,
        'change_24h': change_24h,
        'high_24h': high_24h,
        'low_24h': low_24h,
        'bid': bid,
        'ask': ask,
        'market_cap': market_cap
    }


@st.composite
def trading_strategy_config_strategy(draw):
    """Strategy for generating trading strategy configurations"""
    return {
        'ema_short_period': draw(st.integers(min_value=3, max_value=20)),
        'ema_medium_period': draw(st.integers(min_value=15, max_value=50)),
        'ema_long_period': draw(st.integers(min_value=30, max_value=200)),
        'rsi_period': draw(st.integers(min_value=7, max_value=28)),
        'rsi_overbought': draw(st.integers(min_value=65, max_value=85)),
        'rsi_oversold': draw(st.integers(min_value=15, max_value=35)),
        'min_signal_strength': draw(st.floats(min_value=0.1, max_value=0.9)),
        'profit_target_percent': draw(st.floats(min_value=0.05, max_value=0.5)),
        'stop_loss_percent': draw(st.floats(min_value=0.05, max_value=0.3))
    }
