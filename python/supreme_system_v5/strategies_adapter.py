#!/usr/bin/env python3
"""
🚀 Supreme System V5 - Enhanced Strategy Interface Adapter
Ultra SFL Deep Penetration - Complete Interface Normalization

Eliminates ALL strategy interface mismatches:
- generate_signal method variations
- add_price_data arity mismatches  
- Exception handling and graceful degradation
- Performance tracking and circuit breaking
"""

from typing import Any, Dict, Optional, List, Callable
import time
import logging
from collections import defaultdict, deque
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class StrategyMetrics:
    """Track strategy performance and reliability."""
    calls_total: int = 0
    calls_successful: int = 0
    calls_failed: int = 0
    avg_latency_us: float = 0.0
    last_error: Optional[str] = None
    circuit_breaker_open: bool = False
    circuit_breaker_failures: int = 0
    
class StrategyInterfaceAdapter:
    """
    Ultra-robust adapter that normalizes ANY strategy implementation
    to work seamlessly with the Supreme System V5 engine.
    
    Features:
    - Automatic method discovery and fallback
    - Flexible parameter handling
    - Circuit breaker for failing strategies
    - Performance monitoring
    - Graceful degradation
    """
    
    def __init__(self, strategy: Any, circuit_breaker_threshold: int = 10):
        self._strategy = strategy
        self._metrics = StrategyMetrics()
        self._latency_history = deque(maxlen=1000)
        self._circuit_breaker_threshold = circuit_breaker_threshold
        self._method_cache = {}  # Cache discovered methods
        
        # Discover available methods at initialization
        self._discover_methods()
        
        logger.info(f"Strategy adapter initialized for {type(strategy).__name__}")
        logger.info(f"Available methods: {list(self._method_cache.keys())}")
        
    def _discover_methods(self):
        """Discover and cache available strategy methods."""
        # Methods for adding price data
        add_price_methods = [
            'add_price_data', 'update_price', 'update', 'add_tick', 
            'process_tick', 'on_price_update', 'feed_price'
        ]
        
        # Methods for signal generation
        signal_methods = [
            'generate_signal', 'analyze', 'analyze_signal', 'signal', 
            'create_signal', 'get_signal', 'predict', 'decision', 
            'evaluate', 'assess', 'compute_signal'
        ]
        
        # Cache discovered methods
        for method_name in add_price_methods:
            if hasattr(self._strategy, method_name):
                self._method_cache['add_price_data'] = getattr(self._strategy, method_name)
                break
                
        for method_name in signal_methods:
            if hasattr(self._strategy, method_name):
                self._method_cache['generate_signal'] = getattr(self._strategy, method_name)
                break
                
    def add_price_data(self, symbol: str, price: float, volume: Optional[float] = None, ts: Optional[float] = None) -> bool:
        """
        Add price data with automatic parameter adaptation.
        
        Returns:
            True if successful, False if failed (circuit breaker or error)
        """
        if self._metrics.circuit_breaker_open:
            return False
            
        start_time = time.perf_counter()
        
        try:
            if 'add_price_data' not in self._method_cache:
                # No suitable method found - strategy doesn't need price updates
                return True
                
            method = self._method_cache['add_price_data']
            
            # Try different parameter signatures
            signatures = [
                lambda: method(symbol, price, volume, ts),  # Full signature
                lambda: method(symbol, price, volume),      # Without timestamp  
                lambda: method(symbol, price),              # Minimal
                lambda: method({'symbol': symbol, 'price': price, 'volume': volume, 'timestamp': ts}),  # Dict payload
                lambda: method(price, volume),              # Price only
                lambda: method(price)                       # Just price
            ]
            
            for sig in signatures:
                try:
                    sig()
                    self._record_success(start_time)
                    return True
                except TypeError:
                    continue  # Try next signature
                except Exception as e:
                    logger.warning(f"Strategy add_price_data error: {e}")
                    self._record_failure(str(e))
                    return False
                    
            # All signatures failed
            logger.warning(f"Could not match any add_price_data signature for {type(self._strategy).__name__}")
            self._record_failure("signature_mismatch")
            return False
            
        except Exception as e:
            logger.error(f"Unexpected error in add_price_data: {e}")
            self._record_failure(str(e))
            return False
            
    def generate_signal(self, ctx: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Generate trading signal with comprehensive fallback handling.
        
        Returns:
            Signal dictionary or None if failed/circuit breaker open
        """
        if self._metrics.circuit_breaker_open:
            # Circuit breaker open - return safe HOLD signal
            return {'action': 'HOLD', 'confidence': 0.0, 'size': 0.0, 'reason': 'circuit_breaker_open'}
            
        start_time = time.perf_counter()
        
        try:
            if 'generate_signal' not in self._method_cache:
                # No signal method found - return HOLD
                return {'action': 'HOLD', 'confidence': 0.0, 'size': 0.0, 'reason': 'no_signal_method'}
                
            method = self._method_cache['generate_signal']
            
            # Prepare different payload formats
            payloads = [
                ctx,                                    # Full context
                {
                    'symbol': ctx.get('symbol'),
                    'price': ctx.get('price'),
                    'volume': ctx.get('volume'),
                    'timestamp': ctx.get('timestamp'),
                    'indicators': ctx.get('indicators', {})
                },                                      # Essential data only
                ctx.get('price', 0.0),                  # Just price
                (ctx.get('symbol'), ctx.get('price')),   # Tuple format
            ]
            
            for payload in payloads:
                try:
                    result = method(payload)
                    
                    # Normalize result to standard format
                    normalized = self._normalize_signal(result)
                    
                    self._record_success(start_time)
                    return normalized
                    
                except TypeError:
                    continue  # Try next payload format
                except Exception as e:
                    logger.warning(f"Strategy generate_signal error with payload {type(payload)}: {e}")
                    continue
                    
            # All payloads failed
            logger.warning(f"Could not call generate_signal with any payload format for {type(self._strategy).__name__}")
            self._record_failure("payload_mismatch")
            return {'action': 'HOLD', 'confidence': 0.0, 'size': 0.0, 'reason': 'payload_mismatch'}
            
        except Exception as e:
            logger.error(f"Unexpected error in generate_signal: {e}")
            self._record_failure(str(e))
            return {'action': 'HOLD', 'confidence': 0.0, 'size': 0.0, 'reason': f'error: {e}'}
            
    def _normalize_signal(self, raw_signal: Any) -> Dict[str, Any]:
        """
        Normalize strategy output to standard signal format.
        """
        if raw_signal is None:
            return {'action': 'HOLD', 'confidence': 0.0, 'size': 0.0}
            
        if isinstance(raw_signal, dict):
            # Already dictionary - normalize keys
            normalized = {
                'action': str(raw_signal.get('action', 'HOLD')).upper(),
                'confidence': float(raw_signal.get('confidence', 0.0)),
                'size': float(raw_signal.get('size', 0.0)),
            }
            
            # Optional fields
            for key in ['target_price', 'stop_loss', 'take_profit', 'reason', 'signal_strength']:
                if key in raw_signal:
                    normalized[key] = raw_signal[key]
                    
            return normalized
            
        elif isinstance(raw_signal, (int, float)):
            # Numeric signal - convert to buy/sell/hold
            if raw_signal > 0.3:
                return {'action': 'BUY', 'confidence': min(float(raw_signal), 1.0), 'size': 0.1}
            elif raw_signal < -0.3:
                return {'action': 'SELL', 'confidence': min(abs(float(raw_signal)), 1.0), 'size': 0.1}
            else:
                return {'action': 'HOLD', 'confidence': 1.0 - abs(float(raw_signal)), 'size': 0.0}
                
        elif isinstance(raw_signal, str):
            # String signal
            action = raw_signal.upper()
            if action in ['BUY', 'LONG']:
                return {'action': 'BUY', 'confidence': 0.7, 'size': 0.1}
            elif action in ['SELL', 'SHORT']:
                return {'action': 'SELL', 'confidence': 0.7, 'size': 0.1}
            else:
                return {'action': 'HOLD', 'confidence': 0.5, 'size': 0.0}
                
        else:
            # Unknown format - HOLD by default
            logger.warning(f"Unknown signal format: {type(raw_signal)} = {raw_signal}")
            return {'action': 'HOLD', 'confidence': 0.0, 'size': 0.0, 'reason': 'unknown_format'}
            
    def _record_success(self, start_time: float):
        """Record successful operation."""
        latency = time.perf_counter() - start_time
        self._latency_history.append(latency)
        
        self._metrics.calls_total += 1
        self._metrics.calls_successful += 1
        
        if len(self._latency_history) >= 100:
            self._metrics.avg_latency_us = sum(list(self._latency_history)[-100:]) / 100 * 1e6
            
        # Reset circuit breaker on success
        if self._metrics.circuit_breaker_open:
            logger.info(f"Circuit breaker reset for {type(self._strategy).__name__}")
            self._metrics.circuit_breaker_open = False
            self._metrics.circuit_breaker_failures = 0
            
    def _record_failure(self, error: str):
        """Record failed operation and manage circuit breaker."""
        self._metrics.calls_total += 1
        self._metrics.calls_failed += 1
        self._metrics.last_error = error
        self._metrics.circuit_breaker_failures += 1
        
        # Open circuit breaker if too many failures
        if (self._metrics.circuit_breaker_failures >= self._circuit_breaker_threshold and 
            not self._metrics.circuit_breaker_open):
            
            self._metrics.circuit_breaker_open = True
            logger.warning(f"Circuit breaker OPENED for {type(self._strategy).__name__} after {self._metrics.circuit_breaker_failures} failures")
            
    def get_metrics(self) -> Dict[str, Any]:
        """Get adapter performance metrics."""
        success_rate = (self._metrics.calls_successful / self._metrics.calls_total) if self._metrics.calls_total > 0 else 0.0
        
        latency_stats = {}
        if len(self._latency_history) >= 10:
            import numpy as np
            history = list(self._latency_history)
            latency_stats = {
                'p50_us': float(np.percentile(history, 50) * 1e6),
                'p95_us': float(np.percentile(history, 95) * 1e6),
                'p99_us': float(np.percentile(history, 99) * 1e6),
            }
            
        return {
            'strategy_class': type(self._strategy).__name__,
            'calls_total': self._metrics.calls_total,
            'calls_successful': self._metrics.calls_successful,
            'calls_failed': self._metrics.calls_failed,
            'success_rate': success_rate,
            'avg_latency_us': self._metrics.avg_latency_us,
            'circuit_breaker_open': self._metrics.circuit_breaker_open,
            'circuit_breaker_failures': self._metrics.circuit_breaker_failures,
            'last_error': self._metrics.last_error,
            'available_methods': list(self._method_cache.keys()),
            'latency_stats': latency_stats
        }
        
    def reset_circuit_breaker(self):
        """Manually reset circuit breaker."""
        self._metrics.circuit_breaker_open = False
        self._metrics.circuit_breaker_failures = 0
        logger.info(f"Circuit breaker manually reset for {type(self._strategy).__name__}")
        
    def is_healthy(self) -> bool:
        """Check if strategy adapter is healthy."""
        if self._metrics.calls_total == 0:
            return True  # No calls yet
            
        success_rate = self._metrics.calls_successful / self._metrics.calls_total
        return success_rate >= 0.8 and not self._metrics.circuit_breaker_open
        
class StrategyManager:
    """
    Manages multiple strategy adapters with load balancing and health monitoring.
    """
    
    def __init__(self):
        self.strategies: List[StrategyInterfaceAdapter] = []
        self.primary_strategy_index = 0
        self.fallback_enabled = True
        
    def add_strategy(self, strategy: Any, is_primary: bool = False) -> StrategyInterfaceAdapter:
        """Add a strategy with automatic adapter wrapping."""
        adapter = StrategyInterfaceAdapter(strategy)
        self.strategies.append(adapter)
        
        if is_primary:
            self.primary_strategy_index = len(self.strategies) - 1
            
        logger.info(f"Added strategy {type(strategy).__name__} (primary={is_primary})")
        return adapter
        
    def generate_signal(self, ctx: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Generate signal using primary strategy with automatic fallback.
        """
        if not self.strategies:
            return {'action': 'HOLD', 'confidence': 0.0, 'size': 0.0, 'reason': 'no_strategies'}
            
        # Try primary strategy first
        primary = self.strategies[self.primary_strategy_index]
        if primary.is_healthy():
            signal = primary.generate_signal(ctx)
            if signal and signal.get('action') != 'HOLD' or signal.get('confidence', 0) > 0.3:
                return signal
                
        # Fallback to other healthy strategies
        if self.fallback_enabled:
            for i, strategy in enumerate(self.strategies):
                if i != self.primary_strategy_index and strategy.is_healthy():
                    signal = strategy.generate_signal(ctx)
                    if signal:
                        signal['fallback_used'] = i
                        return signal
                        
        # All strategies failed or unhealthy
        return {'action': 'HOLD', 'confidence': 0.0, 'size': 0.0, 'reason': 'all_strategies_unhealthy'}
        
    def add_price_data(self, symbol: str, price: float, volume: Optional[float] = None, ts: Optional[float] = None):
        """Update price data for all strategies."""
        for adapter in self.strategies:
            if adapter.is_healthy():
                adapter.add_price_data(symbol, price, volume, ts)
                
    def get_health_report(self) -> Dict[str, Any]:
        """Get comprehensive health report for all strategies."""
        strategies_health = []
        for i, adapter in enumerate(self.strategies):
            metrics = adapter.get_metrics()
            metrics['index'] = i
            metrics['is_primary'] = (i == self.primary_strategy_index)
            metrics['is_healthy'] = adapter.is_healthy()
            strategies_health.append(metrics)
            
        healthy_count = sum(1 for s in strategies_health if s['is_healthy'])
        
        return {
            'total_strategies': len(self.strategies),
            'healthy_strategies': healthy_count,
            'primary_strategy_index': self.primary_strategy_index,
            'fallback_enabled': self.fallback_enabled,
            'overall_health': 'GOOD' if healthy_count > 0 else 'CRITICAL',
            'strategies': strategies_health
        }
        
# Convenience functions for backward compatibility
def create_strategy_adapter(strategy: Any) -> StrategyInterfaceAdapter:
    """Create a strategy adapter."""
    return StrategyInterfaceAdapter(strategy)
    
def create_strategy_manager() -> StrategyManager:
    """Create a strategy manager."""
    return StrategyManager()

# Performance testing
if __name__ == "__main__":
    # Mock strategy for testing
    class MockStrategy:
        def analyze(self, data):
            return {'action': 'BUY', 'confidence': 0.8}
            
        def update_price(self, symbol, price, volume):
            pass
            
    class BrokenStrategy:
        def broken_method(self, x, y, z, w, q):
            return "This will never be called"
            
    # Test adapter
    print("🧪 Testing Strategy Interface Adapter...")
    
    # Test working strategy
    adapter1 = StrategyInterfaceAdapter(MockStrategy())
    ctx = {'symbol': 'BTC-USDT', 'price': 50000.0, 'volume': 1.0}
    signal = adapter1.generate_signal(ctx)
    print(f"✅ Working strategy: {signal}")
    
    # Test broken strategy  
    adapter2 = StrategyInterfaceAdapter(BrokenStrategy())
    signal2 = adapter2.generate_signal(ctx)
    print(f"✅ Broken strategy (graceful): {signal2}")
    
    # Test manager
    manager = StrategyManager()
    manager.add_strategy(MockStrategy(), is_primary=True)
    manager.add_strategy(BrokenStrategy())
    
    signal3 = manager.generate_signal(ctx)
    print(f"✅ Manager signal: {signal3}")
    
    health = manager.get_health_report()
    print(f"✅ Health report: {health['overall_health']} ({health['healthy_strategies']}/{health['total_strategies']})")