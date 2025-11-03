#!/usr/bin/env python3
"""
🚀 Supreme System V5 - Strategy Context Builder & Validator
Ultra SFL Deep Penetration - Context Schema Standardization

Ensures 100% compatibility between engine and all strategy implementations.
Eliminates AttributeError, TypeError, and schema mismatches permanently.
"""

from typing import Dict, Any, Optional, List, Union
from dataclasses import dataclass, asdict
import time
import logging
import numpy as np

logger = logging.getLogger(__name__)

class ContextSchemaError(Exception):
    """Raised when strategy context doesn't meet required schema."""
    pass

@dataclass
class IndicatorSnapshot:
    """Standardized indicator values snapshot."""
    ema_short: Optional[float] = None
    ema_long: Optional[float] = None
    rsi: Optional[float] = None
    macd: Optional[float] = None
    macd_signal: Optional[float] = None
    macd_histogram: Optional[float] = None
    atr: Optional[float] = None
    bb_upper: Optional[float] = None
    bb_lower: Optional[float] = None
    bb_middle: Optional[float] = None
    volume_sma: Optional[float] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for backward compatibility."""
        return {k: v for k, v in asdict(self).items() if v is not None}

@dataclass
class MarketMicroStructure:
    """Market microstructure information for scalping."""
    spread_bps: float = 0.0
    mid_price: float = 0.0
    price_impact_bps: float = 0.0
    order_flow_imbalance: float = 0.0
    tick_direction: int = 0  # 1=uptick, -1=downtick, 0=neutral
    volatility_regime: str = "normal"  # low, normal, high
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

@dataclass
class RiskMetrics:
    """Risk management metrics."""
    portfolio_heat: float = 0.0  # Current risk exposure
    drawdown_current: float = 0.0
    drawdown_max: float = 0.0
    var_1d: Optional[float] = None
    sharpe_rolling: Optional[float] = None
    kelly_fraction: Optional[float] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {k: v for k, v in asdict(self).items() if v is not None}

class StrategyContextBuilder:
    """
    Ultra-robust context builder that ensures 100% compatibility
    with any strategy implementation.
    """
    
    def __init__(self, strict_mode: bool = False):
        self.strict_mode = strict_mode
        self.ctx_count = 0
        self.validation_errors = []
        self.build_times = []
        
    def build_ctx(self,
                  symbol: str,
                  price: float,
                  volume: Optional[float] = None,
                  timestamp: Optional[float] = None,
                  bid: Optional[float] = None,
                  ask: Optional[float] = None,
                  portfolio_state: Optional[Any] = None,
                  indicators: Optional[Dict[str, Any]] = None,
                  micro_trend: Optional[str] = None,
                  additional_data: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Build standardized strategy context with comprehensive fallbacks.
        
        Args:
            symbol: Trading symbol (required)
            price: Current price (required)
            volume: Trade volume
            timestamp: Unix timestamp
            bid: Best bid price
            ask: Best ask price
            portfolio_state: Current portfolio state
            indicators: Technical indicator values
            micro_trend: Current micro trend direction
            additional_data: Any additional context data
            
        Returns:
            Standardized context dictionary
        """
        start_time = time.perf_counter()
        
        try:
            # Default values with safety checks
            if volume is None:
                volume = 0.0
            if timestamp is None:
                timestamp = time.time()
            if bid is None:
                bid = price * 0.999  # Approximate bid
            if ask is None:
                ask = price * 1.001  # Approximate ask
                
            # Build indicator snapshot
            if indicators is None:
                indicators = {}
                
            indicator_snapshot = IndicatorSnapshot(
                ema_short=indicators.get('ema_short'),
                ema_long=indicators.get('ema_long'),
                rsi=indicators.get('rsi'),
                macd=indicators.get('macd'),
                macd_signal=indicators.get('macd_signal'),
                macd_histogram=indicators.get('macd_histogram'),
                atr=indicators.get('atr'),
                bb_upper=indicators.get('bb_upper'),
                bb_lower=indicators.get('bb_lower'),
                bb_middle=indicators.get('bb_middle'),
                volume_sma=indicators.get('volume_sma')
            )
            
            # Build market microstructure
            spread_bps = ((ask - bid) / price * 10000) if ask > bid else 0.0
            mid_price = (bid + ask) / 2 if ask > bid else price
            
            micro_structure = MarketMicroStructure(
                spread_bps=spread_bps,
                mid_price=mid_price,
                price_impact_bps=min(spread_bps * 0.5, 10.0),  # Estimate
                order_flow_imbalance=0.0,  # TODO: Calculate from book
                tick_direction=0,  # TODO: Calculate from price history
                volatility_regime=self._assess_volatility_regime(indicators.get('atr'))
            )
            
            # Build risk metrics
            risk_metrics = RiskMetrics()
            if portfolio_state:
                if hasattr(portfolio_state, 'total_value'):
                    risk_metrics.portfolio_heat = getattr(portfolio_state, 'heat', 0.0)
                if hasattr(portfolio_state, 'drawdown'):
                    risk_metrics.drawdown_current = getattr(portfolio_state, 'drawdown', 0.0)
            
            # Build comprehensive context
            ctx = {
                # Core market data
                'symbol': symbol,
                'price': float(price),
                'volume': float(volume),
                'timestamp': float(timestamp),
                'bid': float(bid),
                'ask': float(ask),
                'mid_price': float(mid_price),
                
                # Technical analysis
                'indicators': indicator_snapshot.to_dict(),
                'micro_trend': micro_trend or 'neutral',
                
                # Market microstructure
                'spread_bps': spread_bps,
                'market_microstructure': micro_structure.to_dict(),
                
                # Portfolio and risk
                'portfolio_state': portfolio_state,
                'risk_metrics': risk_metrics.to_dict(),
                
                # Metadata
                'ctx_version': '5.0.0',
                'build_timestamp': time.time(),
                'data_quality_score': 1.0,  # TODO: Calculate from source reliability
                
                # Additional data
                **(additional_data or {})
            }
            
            # Strict validation if enabled
            if self.strict_mode:
                self.validate_ctx(ctx)
                
            # Performance tracking
            build_time = time.perf_counter() - start_time
            self.build_times.append(build_time)
            self.ctx_count += 1
            
            # Log sample contexts for debugging
            if self.ctx_count % 1000 == 0:
                logger.debug(f"Sample ctx #{self.ctx_count}: {self._ctx_summary(ctx)}")
                
            return ctx
            
        except Exception as e:
            logger.error(f"Context building failed: {e}")
            if self.strict_mode:
                raise ContextSchemaError(f"Failed to build valid context: {e}")
            
            # Return minimal safe context
            return {
                'symbol': symbol,
                'price': float(price),
                'volume': float(volume or 0.0),
                'timestamp': float(timestamp or time.time()),
                'bid': float(price * 0.999),
                'ask': float(price * 1.001),
                'indicators': {},
                'micro_trend': 'neutral',
                'spread_bps': 10.0,  # Safe default
                'ctx_version': '5.0.0-fallback',
                'error': str(e)
            }
    
    def validate_ctx(self, ctx: Dict[str, Any]) -> None:
        """
        Validate context schema with comprehensive checks.
        """
        required_fields = [
            'symbol', 'price', 'volume', 'timestamp', 
            'bid', 'ask', 'indicators', 'micro_trend'
        ]
        
        # Check required fields
        for field in required_fields:
            if field not in ctx:
                raise ContextSchemaError(f"Missing required field: {field}")
                
        # Type validation
        numeric_fields = ['price', 'volume', 'timestamp', 'bid', 'ask']
        for field in numeric_fields:
            if not isinstance(ctx[field], (int, float)):
                raise ContextSchemaError(f"Field {field} must be numeric, got {type(ctx[field])}")
                
        # Value validation
        if ctx['price'] <= 0:
            raise ContextSchemaError("Price must be positive")
        if ctx['volume'] < 0:
            raise ContextSchemaError("Volume must be non-negative")
        if ctx['ask'] < ctx['bid']:
            raise ContextSchemaError("Ask price must be >= bid price")
            
        # Indicators validation
        if not isinstance(ctx['indicators'], dict):
            raise ContextSchemaError("Indicators must be a dictionary")
            
    def _assess_volatility_regime(self, atr: Optional[float]) -> str:
        """Assess current volatility regime."""
        if atr is None:
            return "normal"
        
        # Simple volatility classification
        if atr < 0.005:  # 0.5%
            return "low"
        elif atr > 0.02:  # 2%
            return "high"
        else:
            return "normal"
            
    def _ctx_summary(self, ctx: Dict[str, Any]) -> str:
        """Generate concise context summary for logging."""
        return f"symbol={ctx['symbol']}, price={ctx['price']:.2f}, spread={ctx.get('spread_bps', 0):.1f}bps"
        
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get context builder performance statistics."""
        if not self.build_times:
            return {'contexts_built': 0, 'avg_build_time_us': 0}
            
        avg_build_time = np.mean(self.build_times)
        p95_build_time = np.percentile(self.build_times, 95)
        
        return {
            'contexts_built': self.ctx_count,
            'avg_build_time_us': avg_build_time * 1e6,
            'p95_build_time_us': p95_build_time * 1e6,
            'validation_errors': len(self.validation_errors),
            'build_rate_per_sec': self.ctx_count / (sum(self.build_times) or 1)
        }

# Global context builder instance
_global_ctx_builder: Optional[StrategyContextBuilder] = None

def get_context_builder(strict_mode: bool = False) -> StrategyContextBuilder:
    """Get or create global context builder."""
    global _global_ctx_builder
    if _global_ctx_builder is None:
        _global_ctx_builder = StrategyContextBuilder(strict_mode=strict_mode)
    return _global_ctx_builder

def build_strategy_context(symbol: str, price: float, **kwargs) -> Dict[str, Any]:
    """Convenience function to build strategy context."""
    builder = get_context_builder()
    return builder.build_ctx(symbol, price, **kwargs)

def validate_strategy_context(ctx: Dict[str, Any]) -> None:
    """Convenience function to validate strategy context."""
    builder = get_context_builder(strict_mode=True)
    builder.validate_ctx(ctx)

# Performance testing
if __name__ == "__main__":
    import asyncio
    
    async def test_context_builder():
        """Test context builder performance."""
        builder = StrategyContextBuilder(strict_mode=True)
        
        print("🧪 Testing Strategy Context Builder...")
        
        # Test normal case
        ctx = builder.build_ctx(
            symbol="BTC-USDT",
            price=50000.0,
            volume=1.5,
            bid=49999.5,
            ask=50000.5,
            indicators={'ema_short': 49950.0, 'rsi': 65.0}
        )
        
        print(f"✅ Normal context built: {builder._ctx_summary(ctx)}")
        
        # Test edge cases
        try:
            builder.build_ctx("ETH-USDT", -100.0)  # Invalid price
            assert False, "Should have raised error"
        except ContextSchemaError:
            print("✅ Negative price validation works")
            
        # Performance test
        start = time.perf_counter()
        for i in range(10000):
            builder.build_ctx(f"TEST-{i % 10}", 50000 + i * 0.01, volume=1.0)
        duration = time.perf_counter() - start
        
        stats = builder.get_performance_stats()
        print(f"✅ Built 10,000 contexts in {duration:.3f}s")
        print(f"   Average: {stats['avg_build_time_us']:.2f}μs per context")
        print(f"   P95: {stats['p95_build_time_us']:.2f}μs")
        
    asyncio.run(test_context_builder())