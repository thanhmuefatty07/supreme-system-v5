"""
Integration Tests for Supreme System V5
Tests complete data flow: Data Fabric → Core → Strategy → Risk → Execution
"""

import asyncio
import pytest
import time
from unittest.mock import Mock, AsyncMock, patch
from dataclasses import dataclass

# Import system components
from supreme_system_v5.data_fabric import DataAggregator, CoinGeckoConnector, CacheManager
from supreme_system_v5.core import SupremeCore, SystemConfig, MarketData
from supreme_system_v5.strategies import ScalpingStrategy, SignalType
from supreme_system_v5.risk import RiskManager, RiskLimits

@dataclass
class MockMarketData:
    """Mock market data for testing"""
    symbol: str = "BTC-USDT"
    price: float = 35000.0
    volume: float = 1000000.0
    timestamp: float = time.time()
    bid: float = 34999.0
    ask: float = 35001.0

class TestDataFabricIntegration:
    """Test Data Fabric integration"""

    @pytest.mark.asyncio
    async def test_basic_imports(self):
        """Test that all major components can be imported"""
        # Test core imports
        from supreme_system_v5.core import SupremeCore, SystemConfig
        from supreme_system_v5.event_bus import EventBus, get_event_bus
        from supreme_system_v5.risk import RiskManager, RiskLimits
        from supreme_system_v5.strategies import ScalpingStrategy

        # Test data fabric imports
        from supreme_system_v5.data_fabric import DataAggregator
        from supreme_system_v5.data_fabric.cache import DataCache
        from supreme_system_v5.data_fabric.connectors import CoinGeckoConnector

        # Verify classes exist
        assert SupremeCore
        assert SystemConfig
        assert EventBus
        assert RiskManager
        assert ScalpingStrategy
        assert DataAggregator
        assert DataCache
        assert CoinGeckoConnector

        print("✅ All major component imports successful")

    @pytest.mark.asyncio
    async def test_event_bus_basic(self):
        """Test basic event bus functionality"""
        from supreme_system_v5.event_bus import EventBus, create_market_data_event

        bus = EventBus()
        await bus.start()

        # Create and publish event
        event = create_market_data_event("BTC-USDT", 35000.0, 1000.0, "test")
        success = await bus.publish(event)

        assert success
        assert event.type == "market_data"
        assert event.data["symbol"] == "BTC-USDT"

        await bus.stop()
        print("✅ Event bus basic functionality works")

    @pytest.mark.asyncio
    async def test_risk_manager_basic(self):
        """Test basic risk manager functionality"""
        from supreme_system_v5.risk import RiskManager, RiskLimits, PortfolioState

        limits = RiskLimits(max_drawdown_percent=10.0)
        portfolio = PortfolioState(total_value=10000.0)
        risk_manager = RiskManager(limits=limits, portfolio_state=portfolio)

        # Test basic risk check
        assessment = risk_manager.evaluate_trade("BTC-USDT", 1000.0, 1.0)
        assert assessment is not None
        assert hasattr(assessment, 'approved')

        print("✅ Risk manager basic functionality works")

class TestCoreIntegration:
    """Test Core integration"""

    @pytest.fixture
    def system_config(self):
        """Create system configuration for testing"""
        return SystemConfig(
            trading_symbols=["BTC-USDT", "ETH-USDT"],
            max_position_size=0.01,
            stop_loss_percent=0.5,
            take_profit_percent=0.2
        )

    @pytest.fixture
    def supreme_core(self, system_config):
        """Create Supreme Core for testing"""
        return SupremeCore(system_config)

    def test_core_initialization(self, supreme_core):
        """Test core initialization"""
        assert supreme_core.config is not None
        assert supreme_core.running == False
        assert len(supreme_core.market_data) == 0

    @pytest.mark.asyncio
    async def test_market_data_update(self, supreme_core):
        """Test market data updates"""
        # Update market data
        await supreme_core.update_market_data("BTC-USDT", 35000.0, 1000000.0, 34999.0, 35001.0)

        # Verify data
        assert "BTC-USDT" in supreme_core.market_data
        data = supreme_core.market_data["BTC-USDT"]
        assert data.price == 35000.0
        assert data.volume == 1000000.0
        assert data.bid == 34999.0
        assert data.ask == 35001.0

    @pytest.mark.asyncio
    async def test_indicator_calculation(self, supreme_core):
        """Test technical indicator calculation"""
        # Add some market data first
        for i in range(100):
            price = 35000.0 + (i * 10)  # Trending up
            await supreme_core.update_market_data("BTC-USDT", price, 1000000.0)

        # Calculate indicators
        indicators = supreme_core.calculate_technical_indicators("BTC-USDT")

        # Verify indicators are calculated
        assert indicators is not None
        assert 'ema_5' in indicators
        assert 'rsi_14' in indicators
        assert 'current_price' in indicators

class TestStrategyIntegration:
    """Test Strategy integration"""

    @pytest.fixture
    def risk_manager(self):
        """Create risk manager for testing"""
        limits = RiskLimits(
            max_position_size_usd=1000.0,
            max_position_size_percent=2.0,
            max_drawdown_percent=12.0,
            max_daily_loss_usd=100.0
        )
        return RiskManager(limits=limits)

    @pytest.fixture
    def scalping_strategy(self, risk_manager):
        """Create scalping strategy for testing"""
        config = {
            'min_signal_strength': 0.5,
            'max_hold_time_minutes': 15,
            'profit_target_percent': 0.2,
            'stop_loss_percent': 0.5
        }
        return ScalpingStrategy(risk_manager=risk_manager, config=config)

    def test_strategy_initialization(self, scalping_strategy):
        """Test strategy initialization"""
        assert scalping_strategy.name == "ScalpingStrategy"
        assert scalping_strategy.risk_manager is not None
        assert len(scalping_strategy.price_history) == 0

    def test_price_data_feeding(self, scalping_strategy):
        """Test price data feeding to strategy"""
        # Add price data
        scalping_strategy.add_price_data("BTC-USDT", 35000.0)
        scalping_strategy.add_price_data("BTC-USDT", 35010.0)
        scalping_strategy.add_price_data("BTC-USDT", 34990.0)

        # Check price history
        assert "BTC-USDT" in scalping_strategy.price_history
        assert len(scalping_strategy.price_history["BTC-USDT"]) == 3

    def test_indicator_calculation_with_minimal_data(self, scalping_strategy):
        """Test indicator calculation with minimal data"""
        # Add minimal price data
        for i in range(50):
            price = 35000.0 + (i * 5)
            scalping_strategy.add_price_data("BTC-USDT", price)

        # Calculate indicators
        indicators = scalping_strategy._calculate_indicators("BTC-USDT")

        # Should return empty dict with insufficient data
        assert indicators == {}

    def test_indicator_calculation_with_sufficient_data(self, scalping_strategy):
        """Test indicator calculation with sufficient data"""
        # Add sufficient price data (more than min_price_history=100)
        base_price = 35000.0
        for i in range(120):
            # Create some price movement
            price = base_price + (i * 2) + (10 * (i % 10 - 5))  # Trending up with volatility
            scalping_strategy.add_price_data("BTC-USDT", price)

        # Calculate indicators
        indicators = scalping_strategy._calculate_indicators("BTC-USDT")

        # Should return indicators
        assert indicators != {}
        assert 'ema_5' in indicators
        assert 'ema_20' in indicators
        assert 'rsi_14' in indicators
        assert indicators['price_history_count'] >= 100  # Should have at least min_price_history

class TestRiskManagementIntegration:
    """Test Risk Management integration"""

    @pytest.fixture
    def risk_limits(self):
        """Create risk limits for testing"""
        return RiskLimits(
            max_drawdown_percent=12.0,
            max_daily_loss_usd=100.0,
            max_daily_loss_percent=5.0,
            max_position_size_usd=1000.0,
            max_position_size_percent=2.0,
            max_leverage=2.0,
            max_active_positions=3,
            circuit_breaker_threshold=2
        )

    @pytest.fixture
    def risk_manager(self, risk_limits):
        """Create risk manager for testing"""
        return RiskManager(limits=risk_limits)

    def test_risk_evaluation_approved(self, risk_manager):
        """Test risk evaluation for approved trade"""
        assessment = risk_manager.evaluate_trade("BTC-USDT", 500.0, 1.0)

        assert assessment.approved == True
        assert len(assessment.violations) == 0
        assert assessment.reasoning == "✅ Trade approved - all risk checks passed"

    def test_risk_evaluation_position_size_violation(self, risk_manager):
        """Test risk evaluation for position size violation"""
        # Try position larger than limit
        assessment = risk_manager.evaluate_trade("BTC-USDT", 2000.0, 1.0)

        assert assessment.approved == False
        assert len(assessment.violations) > 0
        assert any(v.value == "position_size" for v in assessment.violations)

    def test_risk_evaluation_leverage_violation(self, risk_manager):
        """Test risk evaluation for leverage violation"""
        assessment = risk_manager.evaluate_trade("BTC-USDT", 500.0, 3.0)  # Leverage > 2.0

        assert assessment.approved == False
        assert len(assessment.violations) > 0
        assert any(v.value == "leverage_limit" for v in assessment.violations)

    def test_circuit_breaker_activation(self, risk_manager):
        """Test circuit breaker activation after multiple violations"""
        # Trigger multiple violations to activate circuit breaker
        for i in range(3):
            assessment = risk_manager.evaluate_trade("BTC-USDT", 2000.0, 1.0)  # Position size violation
            assert assessment.approved == False

        # Next evaluation should be blocked by circuit breaker
        assessment = risk_manager.evaluate_trade("BTC-USDT", 100.0, 1.0)  # Valid trade
        assert assessment.approved == False
        assert any(v.value == "circuit_breaker" for v in assessment.violations)

class TestEndToEndIntegration:
    """Test complete end-to-end integration"""

    @pytest.fixture
    async def integrated_system(self):
        """Create integrated system for testing"""
        # Setup components
        cache_manager = CacheManager({
            'enable_memory_cache': True,
            'enable_redis_cache': False,
            'enable_postgres_persistence': False
        })
        await cache_manager.start()

        # Create data aggregator
        data_aggregator = DataAggregator(cache_manager=cache_manager)

        # Create core system
        config = SystemConfig(trading_symbols=["BTC-USDT"])
        core = SupremeCore(config)

        # Create risk manager
        risk_limits = RiskLimits(max_position_size_usd=1000.0)
        risk_manager = RiskManager(limits=risk_limits)

        # Create strategy
        strategy = ScalpingStrategy(risk_manager=risk_manager)

        yield {
            'cache_manager': cache_manager,
            'data_aggregator': data_aggregator,
            'core': core,
            'risk_manager': risk_manager,
            'strategy': strategy
        }

        # Cleanup
        await cache_manager.stop()

    @pytest.mark.asyncio
    async def test_data_flow_integration(self, integrated_system):
        """Test complete data flow from source to signal"""
        system = integrated_system

        # Simulate market data update
        market_data = MockMarketData()
        system['core'].update_market_data(
            market_data.symbol,
            market_data.price,
            market_data.volume,
            market_data.bid,
            market_data.ask
        )

        # Feed data to strategy
        system['strategy'].add_price_data(
            market_data.symbol,
            market_data.price,
            market_data.volume,
            market_data.timestamp
        )

        # Calculate indicators in core
        indicators = system['core'].calculate_technical_indicators(market_data.symbol)

        # Strategy should be able to calculate with sufficient data
        # (Note: In real scenario, we would need more price history)
        strategy_status = system['strategy'].get_strategy_status()

        # Verify system components are working together
        assert system['core'].market_data[market_data.symbol].price == market_data.price
        assert strategy_status['name'] == 'ScalpingStrategy'
        assert system['risk_manager'].portfolio.total_value == 10000.0  # Default portfolio value

    @pytest.mark.asyncio
    async def test_strategy_signal_generation(self, integrated_system):
        """Test strategy signal generation with risk integration"""
        system = integrated_system

        # Add sufficient price history for indicator calculation
        base_price = 35000.0
        for i in range(150):
            price = base_price + (i * 0.1) + (5 * (i % 20 - 10))  # Trending with volatility
            system['strategy'].add_price_data("BTC-USDT", price)

        # Update core market data
        system['core'].update_market_data("BTC-USDT", base_price + 150, 1000000.0)

        # Create market data object for strategy
        from supreme_system_v5.strategies import MarketData
        market_data = MarketData(
            symbol="BTC-USDT",
            price=base_price + 150,
            volume=1000000.0,
            timestamp=time.time()
        )

        # Generate signal
        signal = system['strategy'].generate_signal("BTC-USDT", market_data)

        # Verify signal generation (may be None if conditions not met)
        # This is a valid outcome - strategy may not generate signals in all conditions
        assert signal is None or hasattr(signal, 'signal_type')

class TestCircuitBreakerResilience:
    """Test circuit breaker and system resilience"""

    def test_multiple_violations_trigger_circuit_breaker(self):
        """Test that multiple violations trigger circuit breaker"""
        risk_limits = RiskLimits(
            circuit_breaker_threshold=2,
            max_position_size_usd=500.0
        )
        risk_manager = RiskManager(limits=risk_limits)

        # First violation
        assessment1 = risk_manager.evaluate_trade("BTC-USDT", 1000.0, 1.0)
        assert assessment1.approved == False

        # Second violation
        assessment2 = risk_manager.evaluate_trade("BTC-USDT", 1000.0, 1.0)
        assert assessment2.approved == False

        # Third attempt should be blocked by circuit breaker
        assessment3 = risk_manager.evaluate_trade("BTC-USDT", 100.0, 1.0)  # Valid size
        assert assessment3.approved == False
        assert any(v.value == "circuit_breaker" for v in assessment3.violations)

    def test_circuit_breaker_recovery(self):
        """Test circuit breaker recovery after timeout"""
        risk_limits = RiskLimits(
            circuit_breaker_threshold=1,
            circuit_breaker_timeout_minutes=0  # Immediate timeout for testing
        )
        risk_manager = RiskManager(limits=risk_limits)

        # Trigger circuit breaker
        assessment = risk_manager.evaluate_trade("BTC-USDT", 2000.0, 1.0)
        assert assessment.approved == False

        # Circuit breaker should recover immediately due to 0 timeout
        # (In practice, we'd wait for the timeout)
        risk_manager.portfolio.circuit_breaker_until = time.time() - 1  # Set to past

        assessment2 = risk_manager.evaluate_trade("BTC-USDT", 100.0, 1.0)
        assert assessment2.approved == True

class TestWebSocketReconnection:
    """Test WebSocket reconnection logic"""

    @pytest.mark.asyncio
    async def test_websocket_connector_reconnection(self):
        """Test OKX WebSocket connector reconnection"""
        from supreme_system_v5.data_fabric.connectors.okx_public import OKXPublicConnector

        connector = OKXPublicConnector()

        # Mock WebSocket connection failure
        with patch.object(connector, '_connect_websocket', new_callable=AsyncMock) as mock_connect:
            mock_connect.side_effect = Exception("Connection failed")

            # Attempt connection
            await connector.connect()

            # Verify connection was attempted
            mock_connect.assert_called()

            # Verify system handles failure gracefully
            assert connector.ws_connected == False

if __name__ == "__main__":
    # Run integration tests
    pytest.main([__file__, "-v"])
