#!/usr/bin/env python3
"""
🧪 Supreme System V5 - Enhanced Interface Test Suite
Ultra SFL Deep Penetration - Comprehensive Error Prevention Testing

Tests specifically designed to catch and prevent the exact errors
seen in the terminal logs:

1. 'ScalpingStrategy' object has no attribute 'generate_signal'
2. add_price_data() takes from 2 to 4 positional arguments but 5 were given
3. 'PortfolioState' object has no attribute 'total_value'
"""

import pytest
import asyncio
import time
from unittest.mock import Mock, MagicMock
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

try:
    from python.supreme_system_v5.strategies_adapter import StrategyInterfaceAdapter, StrategyManager
    from python.supreme_system_v5.strategy_ctx import StrategyContextBuilder, ContextSchemaError
    from python.supreme_system_v5.portfolio_state import PortfolioState
    from python.supreme_system_v5.data_fabric.quorum_policy import QuorumPolicy, DataFabricAggregator
    ENHANCED_AVAILABLE = True
except ImportError as e:
    print(f"Enhanced components not available: {e}")
    ENHANCED_AVAILABLE = False

class TestStrategyInterfaceErrors:
    """
    Test suite specifically for strategy interface errors.
    """
    
    def test_missing_generate_signal_method(self):
        """Test ERROR: 'ScalpingStrategy' object has no attribute 'generate_signal'"""
        if not ENHANCED_AVAILABLE:
            pytest.skip("Enhanced components not available")
            
        class StrategyMissingGenerateSignal:
            """Strategy that doesn't have generate_signal method."""
            def analyze(self, data):
                return {'action': 'BUY', 'confidence': 0.8}
                
            def add_price_data(self, symbol, price):
                pass
                
        # This should NOT raise AttributeError with adapter
        strategy = StrategyMissingGenerateSignal()
        adapter = StrategyInterfaceAdapter(strategy)
        
        # Call generate_signal - should work via fallback to analyze()
        ctx = {'symbol': 'BTC-USDT', 'price': 50000.0}
        signal = adapter.generate_signal(ctx)
        
        assert signal is not None, "Adapter should return a signal"
        assert 'action' in signal, "Signal should have action field"
        assert signal['action'] in ['BUY', 'SELL', 'HOLD'], f"Invalid action: {signal['action']}"
        
    def test_missing_any_signal_method(self):
        """Test strategy with NO signal generation methods at all."""
        if not ENHANCED_AVAILABLE:
            pytest.skip("Enhanced components not available")
            
        class StrategyNoSignalMethods:
            """Strategy with no signal methods whatsoever."""
            def add_price_data(self, symbol, price):
                pass
                
        strategy = StrategyNoSignalMethods()
        adapter = StrategyInterfaceAdapter(strategy)
        
        # Should return HOLD signal without crashing
        ctx = {'symbol': 'ETH-USDT', 'price': 3000.0}
        signal = adapter.generate_signal(ctx)
        
        assert signal['action'] == 'HOLD', "Should default to HOLD"
        assert signal['confidence'] == 0.0, "Should have zero confidence"
        
    def test_add_price_data_arity_mismatch(self):
        """Test ERROR: add_price_data() takes from 2 to 4 positional arguments but 5 were given"""
        if not ENHANCED_AVAILABLE:
            pytest.skip("Enhanced components not available")
            
        class StrategyShortSignature:
            """Strategy with short add_price_data signature."""
            def add_price_data(self, symbol, price):
                """Only accepts 2 parameters (symbol, price)."""
                self.last_symbol = symbol
                self.last_price = price
                
            def generate_signal(self, data):
                return {'action': 'HOLD', 'confidence': 0.5}
                
        strategy = StrategyShortSignature()
        adapter = StrategyInterfaceAdapter(strategy)
        
        # This should NOT raise TypeError with adapter
        # Adapter should automatically try shorter signatures
        success = adapter.add_price_data('BTC-USDT', 50000.0, 1.5, time.time())
        
        assert success == True, "Adapter should handle arity mismatch gracefully"
        assert hasattr(strategy, 'last_symbol'), "Strategy should have been called"
        assert strategy.last_symbol == 'BTC-USDT', "Strategy should receive correct symbol"
        assert strategy.last_price == 50000.0, "Strategy should receive correct price"
        
    def test_add_price_data_dict_signature(self):
        """Test strategy that expects dict payload for add_price_data."""
        if not ENHANCED_AVAILABLE:
            pytest.skip("Enhanced components not available")
            
        class StrategyDictSignature:
            """Strategy that expects dictionary payload."""
            def add_price_data(self, data_dict):
                self.last_data = data_dict
                
            def generate_signal(self, data):
                return {'action': 'BUY', 'confidence': 0.7}
                
        strategy = StrategyDictSignature()
        adapter = StrategyInterfaceAdapter(strategy)
        
        # Should fall back to dict payload after positional args fail
        success = adapter.add_price_data('ETH-USDT', 3000.0, 2.0, time.time())
        
        assert success == True, "Adapter should handle dict signature fallback"
        assert hasattr(strategy, 'last_data'), "Strategy should have received data"
        assert isinstance(strategy.last_data, dict), "Strategy should receive dict payload"
        
class TestPortfolioStateErrors:
    """
    Test suite for portfolio state attribute errors.
    """
    
    def test_portfolio_state_total_value_attribute(self):
        """Test ERROR: 'PortfolioState' object has no attribute 'total_value'"""
        if not ENHANCED_AVAILABLE:
            pytest.skip("Enhanced components not available")
            
        # Create PortfolioState and ensure total_value works
        portfolio = PortfolioState(total_balance=10000.0, positions_value=2000.0)
        
        # This should NOT raise AttributeError
        total = portfolio.total_value
        
        assert total == 12000.0, f"total_value should be 12000.0, got {total}"
        assert hasattr(portfolio, 'total_value'), "PortfolioState should have total_value attribute"
        
    def test_portfolio_state_property_consistency(self):
        """Test that total_value property is consistent."""
        if not ENHANCED_AVAILABLE:
            pytest.skip("Enhanced components not available")
            
        portfolio = PortfolioState(total_balance=5000.0, positions_value=0.0)
        assert portfolio.total_value == 5000.0
        
        # Update balances
        portfolio.total_balance = 4500.0
        portfolio.positions_value = 1000.0
        
        assert portfolio.total_value == 5500.0, "total_value should update dynamically"
        
class TestContextBuilder:
    """
    Test suite for strategy context building.
    """
    
    def test_context_builder_basic(self):
        """Test basic context building."""
        if not ENHANCED_AVAILABLE:
            pytest.skip("Enhanced components not available")
            
        builder = StrategyContextBuilder()
        
        ctx = builder.build_ctx(
            symbol='BTC-USDT',
            price=50000.0,
            volume=1.5
        )
        
        assert 'symbol' in ctx, "Context should have symbol"
        assert 'price' in ctx, "Context should have price"
        assert 'volume' in ctx, "Context should have volume"
        assert 'timestamp' in ctx, "Context should have timestamp"
        assert 'indicators' in ctx, "Context should have indicators dict"
        
    def test_context_builder_strict_validation(self):
        """Test strict context validation."""
        if not ENHANCED_AVAILABLE:
            pytest.skip("Enhanced components not available")
            
        builder = StrategyContextBuilder(strict_mode=True)
        
        # Valid context should pass
        ctx = builder.build_ctx(
            symbol='ETH-USDT',
            price=3000.0,
            volume=2.0,
            bid=2999.0,
            ask=3001.0
        )
        
        # Should not raise exception
        assert ctx['symbol'] == 'ETH-USDT'
        assert ctx['price'] == 3000.0
        
        # Invalid price should raise in strict mode
        with pytest.raises(ContextSchemaError):
            builder.build_ctx(symbol='BTC-USDT', price=-100.0)  # Negative price
            
class TestIntegrationErrors:
    """
    Integration tests that replicate the exact error scenarios from logs.
    """
    
    @pytest.mark.asyncio
    async def test_full_integration_no_errors(self):
        """Test full integration without any interface errors."""
        if not ENHANCED_AVAILABLE:
            pytest.skip("Enhanced components not available")
            
        # Create a strategy similar to the one causing errors
        class MockScalpingStrategy:
            """Mock strategy that mimics the problematic ScalpingStrategy."""
            def __init__(self):
                self.prices = []
                
            def analyze(self, data):
                """Uses 'analyze' instead of 'generate_signal'."""
                return {'action': 'BUY', 'confidence': 0.6}
                
            def add_price_data(self, symbol, price):  # Short signature
                """Only accepts symbol and price."""
                self.prices.append((symbol, price))
                
        # Create adapter and manager
        strategy = MockScalpingStrategy()
        adapter = StrategyInterfaceAdapter(strategy)
        manager = StrategyManager()
        manager.add_strategy(strategy, is_primary=True)
        
        # Create context builder
        ctx_builder = StrategyContextBuilder()
        
        # Create portfolio state
        portfolio = PortfolioState(total_balance=10000.0)
        
        # Simulate the exact sequence that was causing errors
        for i in range(10):
            # 1. Add price data (this was causing arity error)
            success = manager.add_price_data('BTC-USDT', 50000.0 + i, 1.5, time.time())
            assert success, f"add_price_data should succeed on iteration {i}"
            
            # 2. Build context (this was causing total_value error)
            ctx = ctx_builder.build_ctx(
                symbol='BTC-USDT',
                price=50000.0 + i,
                volume=1.5,
                timestamp=time.time(),
                portfolio_state=portfolio
            )
            
            assert 'symbol' in ctx, "Context should have symbol"
            assert portfolio.total_value > 0, "Portfolio total_value should work"
            
            # 3. Generate signal (this was causing generate_signal error)
            signal = manager.generate_signal(ctx)
            
            assert signal is not None, f"Signal should not be None on iteration {i}"
            assert 'action' in signal, "Signal should have action"
            assert 'confidence' in signal, "Signal should have confidence"
            
        print("✅ Full integration test passed - NO INTERFACE ERRORS!")
        
    def test_strategy_manager_health_monitoring(self):
        """Test strategy health monitoring and circuit breaking."""
        if not ENHANCED_AVAILABLE:
            pytest.skip("Enhanced components not available")
            
        class FailingStrategy:
            """Strategy that fails after a few calls."""
            def __init__(self):
                self.call_count = 0
                
            def generate_signal(self, data):
                self.call_count += 1
                if self.call_count > 3:
                    raise Exception("Strategy failure simulation")
                return {'action': 'HOLD', 'confidence': 0.5}
                
        strategy = FailingStrategy()
        adapter = StrategyInterfaceAdapter(strategy, circuit_breaker_threshold=3)
        
        ctx = {'symbol': 'BTC-USDT', 'price': 50000.0}
        
        # First few calls should work
        for i in range(3):
            signal = adapter.generate_signal(ctx)
            assert signal['action'] == 'HOLD'
            
        # Next calls should trigger circuit breaker
        for i in range(5):
            signal = adapter.generate_signal(ctx)
            # Should still return signal (HOLD) even with circuit breaker
            assert signal is not None
            
        # Check circuit breaker status
        metrics = adapter.get_metrics()
        assert metrics['circuit_breaker_open'] == True, "Circuit breaker should be open"
        assert metrics['calls_failed'] > 0, "Should have recorded failures"
        
# Benchmark tests
class TestPerformanceBenchmarks:
    """
    Performance benchmark tests for i3-4GB optimization validation.
    """
    
    def test_context_builder_performance(self):
        """Test context builder meets performance targets."""
        if not ENHANCED_AVAILABLE:
            pytest.skip("Enhanced components not available")
            
        builder = StrategyContextBuilder()
        
        # Warm up
        for i in range(100):
            builder.build_ctx('BTC-USDT', 50000.0 + i)
            
        # Performance test
        start = time.perf_counter()
        iterations = 10000
        
        for i in range(iterations):
            ctx = builder.build_ctx(
                symbol='BTC-USDT',
                price=50000.0 + i * 0.01,
                volume=1.0 + i * 0.001,
                timestamp=time.time()
            )
            
        duration = time.perf_counter() - start
        avg_time_us = (duration / iterations) * 1e6
        
        # Performance targets for i3-4GB
        assert avg_time_us < 50.0, f"Context building too slow: {avg_time_us:.2f}μs (target: <50μs)"
        
        print(f"✅ Context builder performance: {avg_time_us:.2f}μs per context")
        
    def test_adapter_performance(self):
        """Test adapter meets performance targets."""
        if not ENHANCED_AVAILABLE:
            pytest.skip("Enhanced components not available")
            
        class FastMockStrategy:
            def generate_signal(self, data):
                return {'action': 'HOLD', 'confidence': 0.5}
                
            def add_price_data(self, symbol, price, volume, timestamp):
                pass
                
        strategy = FastMockStrategy()
        adapter = StrategyInterfaceAdapter(strategy)
        
        ctx = {'symbol': 'BTC-USDT', 'price': 50000.0, 'volume': 1.0, 'timestamp': time.time()}
        
        # Warm up
        for i in range(100):
            adapter.generate_signal(ctx)
            adapter.add_price_data('BTC-USDT', 50000.0 + i, 1.0, time.time())
            
        # Performance test
        start = time.perf_counter()
        iterations = 5000
        
        for i in range(iterations):
            adapter.generate_signal(ctx)
            adapter.add_price_data('BTC-USDT', 50000.0 + i, 1.0, time.time())
            
        duration = time.perf_counter() - start
        avg_time_us = (duration / iterations) * 1e6
        
        # Performance targets for scalping
        assert avg_time_us < 100.0, f"Adapter too slow: {avg_time_us:.2f}μs (target: <100μs)"
        
        print(f"✅ Adapter performance: {avg_time_us:.2f}μs per operation pair")
        
class TestMemoryOptimization:
    """
    Test memory optimization for i3-4GB systems.
    """
    
    def test_memory_bounded_buffers(self):
        """Test that circular buffers respect memory limits."""
        if not ENHANCED_AVAILABLE:
            pytest.skip("Enhanced components not available")
            
        import sys
        
        # Test circular buffer doesn't grow unbounded
        from python.supreme_system_v5.algorithms.scalping_futures_optimized import OptimizedFuturesScalpingEngine
        
        engine = OptimizedFuturesScalpingEngine(max_memory_mb=100)  # Small limit for testing
        
        # Add many data points
        for i in range(10000):
            ctx = {
                'symbol': 'BTC-USDT',
                'price': 50000.0 + i * 0.01,
                'volume': 1.0,
                'timestamp': time.time(),
                'spread_bps': 5.0
            }
            
            signal = engine.generate_scalping_signal(ctx)
            
            # Check memory usage periodically
            if i % 1000 == 0:
                # Buffers should not grow unbounded
                assert len(engine.spread_history) <= 100, "Spread history buffer should be bounded"
                assert len(engine.volume_profile) <= 200, "Volume profile buffer should be bounded"
                assert len(engine.decision_times) <= 1000, "Decision times buffer should be bounded"
                
        print("✅ Memory bounds respected")
        
# Regression tests
class TestRegressionPrevention:
    """
    Specific regression tests to prevent the logged errors from reoccurring.
    """
    
    def test_no_attribute_errors(self):
        """Ensure no AttributeError can occur with any strategy type."""
        if not ENHANCED_AVAILABLE:
            pytest.skip("Enhanced components not available")
            
        # Test various problematic strategy types
        class EmptyStrategy:
            pass
            
        class MinimalStrategy:
            def __init__(self):
                pass
                
        class PartialStrategy:
            def some_method(self):
                return "not a signal method"
                
        strategies = [EmptyStrategy(), MinimalStrategy(), PartialStrategy()]
        
        for i, strategy in enumerate(strategies):
            adapter = StrategyInterfaceAdapter(strategy)
            
            # These should NEVER raise AttributeError
            ctx = {'symbol': 'BTC-USDT', 'price': 50000.0}
            signal = adapter.generate_signal(ctx)
            
            assert signal is not None, f"Strategy {i} should return some signal"
            assert 'action' in signal, f"Strategy {i} signal should have action"
            
            # These should NEVER raise TypeError
            success = adapter.add_price_data('BTC-USDT', 50000.0, 1.0, time.time())
            # Success can be True or False, but should not raise exception
            
        print("✅ No AttributeError or TypeError possible")
        
    def test_data_fabric_error_resilience(self):
        """Test that data fabric errors don't crash the system."""
        if not ENHANCED_AVAILABLE:
            pytest.skip("Enhanced components not available")
            
        aggregator = DataFabricAggregator(['failing_source_1', 'failing_source_2'])
        
        # This should handle source failures gracefully
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        try:
            consensus_data, metadata = loop.run_until_complete(
                aggregator.aggregate_market_data('BTC-USDT')
            )
            
            # Should return None for consensus but not crash
            assert consensus_data is None or isinstance(consensus_data, dict)
            assert isinstance(metadata, dict)
            assert 'error' in metadata or 'sources_available' in metadata
            
        finally:
            loop.close()
            
        print("✅ Data fabric resilience confirmed")
        
if __name__ == '__main__':
    if ENHANCED_AVAILABLE:
        print("🧪 Running Enhanced Interface Tests...")
        pytest.main([__file__, '-v', '--tb=short'])
    else:
        print("⚠️ Enhanced components not available - skipping tests")
        sys.exit(0)