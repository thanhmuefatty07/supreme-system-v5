#!/usr/bin/env python3
"""
🚀 Supreme System V5 - Data Fabric Quorum Policy
Ultra SFL Deep Penetration - Robust Multi-Source Data Aggregation

Features:
- Multi-source consensus with quality scoring
- Automatic failover and recovery
- Circuit breaker per data source
- Outlier detection and filtering
- Data quality validation
"""

import time
import logging
import statistics
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict, deque

logger = logging.getLogger(__name__)

class DataSourceStatus(Enum):
    """Data source status enumeration."""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    CIRCUIT_OPEN = "circuit_open"
    UNKNOWN = "unknown"

@dataclass
class DataSourceHealth:
    """Health metrics for a data source."""
    source_name: str
    status: DataSourceStatus = DataSourceStatus.UNKNOWN
    success_count: int = 0
    failure_count: int = 0
    last_success_time: Optional[float] = None
    last_failure_time: Optional[float] = None
    circuit_breaker_open_time: Optional[float] = None
    quality_scores: deque = field(default_factory=lambda: deque(maxlen=100))
    latency_history: deque = field(default_factory=lambda: deque(maxlen=100))
    
    @property
    def success_rate(self) -> float:
        """Calculate success rate."""
        total = self.success_count + self.failure_count
        return self.success_count / total if total > 0 else 0.0
    
    @property
    def avg_quality_score(self) -> float:
        """Calculate average quality score."""
        return statistics.mean(self.quality_scores) if self.quality_scores else 0.0
    
    @property
    def avg_latency_ms(self) -> float:
        """Calculate average latency in milliseconds."""
        return statistics.mean(self.latency_history) * 1000 if self.latency_history else 0.0

class CircuitBreaker:
    """
    Circuit breaker for individual data sources.
    """
    
    def __init__(self, failure_threshold: int = 5, recovery_timeout: float = 30.0):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.failure_count = 0
        self.last_failure_time = 0.0
        self.state = "CLOSED"  # CLOSED, OPEN, HALF_OPEN
        
    def can_attempt(self) -> bool:
        """Check if we can attempt to call the data source."""
        if self.state == "CLOSED":
            return True
        elif self.state == "OPEN":
            # Check if recovery timeout has passed
            if time.time() - self.last_failure_time > self.recovery_timeout:
                self.state = "HALF_OPEN"
                return True
            return False
        elif self.state == "HALF_OPEN":
            return True
        return False
    
    def record_success(self):
        """Record successful operation."""
        if self.state == "HALF_OPEN":
            self.state = "CLOSED"
            self.failure_count = 0
            logger.info("Circuit breaker reset to CLOSED")
        
    def record_failure(self):
        """Record failed operation."""
        self.failure_count += 1
        self.last_failure_time = time.time()
        
        if self.failure_count >= self.failure_threshold:
            if self.state != "OPEN":
                self.state = "OPEN"
                logger.warning(f"Circuit breaker OPENED after {self.failure_count} failures")

class QuorumPolicy:
    """
    Ultra-robust quorum policy for multi-source data aggregation.
    Handles source failures, outlier detection, and quality assessment.
    """
    
    def __init__(self, 
                 min_sources: int = 1,
                 outlier_threshold: float = 0.05,  # 5% price deviation
                 min_quality_score: float = 0.7):
        self.min_sources = min_sources
        self.outlier_threshold = outlier_threshold
        self.min_quality_score = min_quality_score
        
        # Health tracking per source
        self.source_health: Dict[str, DataSourceHealth] = defaultdict(
            lambda: DataSourceHealth(source_name="unknown")
        )
        
        # Circuit breakers per source
        self.circuit_breakers: Dict[str, CircuitBreaker] = defaultdict(
            lambda: CircuitBreaker(failure_threshold=3, recovery_timeout=30.0)
        )
        
        # Performance metrics
        self.quorum_decisions = deque(maxlen=1000)
        self.consensus_times = deque(maxlen=1000)
        
    def evaluate_consensus(self, 
                          source_data: Dict[str, Dict[str, Any]],
                          symbol: str) -> Tuple[Optional[Dict[str, Any]], Dict[str, Any]]:
        """
        Evaluate consensus from multiple data sources.
        
        Args:
            source_data: Dict mapping source_name -> data_dict
            symbol: Trading symbol
            
        Returns:
            (consensus_data, metadata) tuple
        """
        start_time = time.perf_counter()
        
        try:
            # Filter by circuit breaker status
            available_sources = {}
            for source_name, data in source_data.items():
                cb = self.circuit_breakers[source_name]
                if cb.can_attempt():
                    available_sources[source_name] = data
                else:
                    logger.debug(f"Circuit breaker OPEN for {source_name}")
                    
            if len(available_sources) < self.min_sources:
                logger.warning(f"Insufficient data sources: {len(available_sources)} < {self.min_sources}")
                return None, {
                    'error': 'insufficient_sources',
                    'available_sources': len(available_sources),
                    'min_required': self.min_sources
                }
                
            # Extract prices and assess quality
            source_prices = {}
            quality_scores = {}
            
            for source_name, data in available_sources.items():
                try:
                    price = float(data.get('price', 0))
                    if price <= 0:
                        raise ValueError("Invalid price")
                        
                    source_prices[source_name] = price
                    
                    # Calculate quality score
                    quality = self._calculate_quality_score(data)
                    quality_scores[source_name] = quality
                    
                    # Record success
                    self._record_source_success(source_name, quality, start_time)
                    
                except Exception as e:
                    logger.warning(f"Invalid data from {source_name}: {e}")
                    self._record_source_failure(source_name, str(e))
                    
            # Filter by quality
            high_quality_sources = {
                name: price for name, price in source_prices.items()
                if quality_scores.get(name, 0) >= self.min_quality_score
            }
            
            if not high_quality_sources:
                # Fallback to any available data if all quality is low
                logger.warning("All sources below quality threshold, using best available")
                if source_prices:
                    best_source = max(quality_scores.items(), key=lambda x: x[1])[0]
                    high_quality_sources = {best_source: source_prices[best_source]}
                    
            if not high_quality_sources:
                return None, {'error': 'no_quality_data', 'quality_scores': quality_scores}
                
            # Outlier detection
            prices = list(high_quality_sources.values())
            if len(prices) > 2:
                median_price = statistics.median(prices)
                filtered_prices = {}
                
                for source_name, price in high_quality_sources.items():
                    deviation = abs(price - median_price) / median_price
                    if deviation <= self.outlier_threshold:
                        filtered_prices[source_name] = price
                    else:
                        logger.warning(f"Outlier detected: {source_name} price={price:.2f} (deviation={deviation:.3f})")
                        
                if filtered_prices:
                    high_quality_sources = filtered_prices
                    
            # Generate consensus
            consensus_price = statistics.median(high_quality_sources.values())
            
            # Build consensus data
            representative_data = next(iter(available_sources.values()))
            consensus_data = {
                'symbol': symbol,
                'price': consensus_price,
                'volume': statistics.median([float(d.get('volume', 0)) for d in available_sources.values()]),
                'timestamp': time.time(),
                'bid': consensus_price * 0.9995,  # Approximate
                'ask': consensus_price * 1.0005,  # Approximate
                'spread_bps': 5.0,  # Default spread
                
                # Quality metadata
                'data_quality_score': statistics.mean(quality_scores.values()),
                'sources_used': list(high_quality_sources.keys()),
                'consensus_method': 'median',
                'outliers_removed': len(source_prices) - len(high_quality_sources)
            }
            
            # Performance tracking
            consensus_time = time.perf_counter() - start_time
            self.consensus_times.append(consensus_time)
            
            metadata = {
                'sources_available': len(available_sources),
                'sources_used': len(high_quality_sources),
                'quality_scores': quality_scores,
                'consensus_time_us': consensus_time * 1e6,
                'outliers_removed': len(source_prices) - len(high_quality_sources)
            }
            
            self.quorum_decisions.append({
                'timestamp': time.time(),
                'symbol': symbol,
                'sources_used': len(high_quality_sources),
                'consensus_price': consensus_price,
                'quality_score': consensus_data['data_quality_score']
            })
            
            return consensus_data, metadata
            
        except Exception as e:
            logger.error(f"Quorum evaluation failed: {e}")
            return None, {'error': f'quorum_failure: {e}'}
            
    def _calculate_quality_score(self, data: Dict[str, Any]) -> float:
        """
        Calculate data quality score (0.0 to 1.0).
        """
        score = 1.0
        
        # Check required fields
        required_fields = ['price', 'volume', 'timestamp']
        for field in required_fields:
            if field not in data or data[field] is None:
                score *= 0.8
                
        # Check data freshness
        if 'timestamp' in data:
            age_seconds = time.time() - float(data['timestamp'])
            if age_seconds > 60:  # Data older than 1 minute
                score *= 0.7
            elif age_seconds > 10:  # Data older than 10 seconds
                score *= 0.9
                
        # Check price reasonableness
        if 'price' in data:
            price = float(data['price'])
            if price <= 0 or price > 1e9:  # Unreasonable price
                score *= 0.1
                
        # Check volume
        if 'volume' in data:
            volume = float(data['volume'])
            if volume < 0:
                score *= 0.5
                
        return max(score, 0.0)
        
    def _record_source_success(self, source_name: str, quality_score: float, start_time: float):
        """Record successful data retrieval from source."""
        health = self.source_health[source_name]
        health.source_name = source_name
        health.success_count += 1
        health.last_success_time = time.time()
        health.quality_scores.append(quality_score)
        health.latency_history.append(time.perf_counter() - start_time)
        
        # Update status
        if health.success_rate >= 0.9:
            health.status = DataSourceStatus.HEALTHY
        elif health.success_rate >= 0.7:
            health.status = DataSourceStatus.DEGRADED
            
        # Record success in circuit breaker
        self.circuit_breakers[source_name].record_success()
        
    def _record_source_failure(self, source_name: str, error: str):
        """Record failed data retrieval from source."""
        health = self.source_health[source_name]
        health.source_name = source_name
        health.failure_count += 1
        health.last_failure_time = time.time()
        
        # Update status based on success rate
        if health.success_rate < 0.5:
            health.status = DataSourceStatus.CIRCUIT_OPEN
        elif health.success_rate < 0.8:
            health.status = DataSourceStatus.DEGRADED
            
        # Record failure in circuit breaker
        cb = self.circuit_breakers[source_name]
        cb.record_failure()
        
        if cb.state == "OPEN":
            health.status = DataSourceStatus.CIRCUIT_OPEN
            health.circuit_breaker_open_time = time.time()
            
    def get_source_health_report(self) -> Dict[str, Any]:
        """Generate comprehensive health report for all sources."""
        report = {
            'timestamp': time.time(),
            'sources': {},
            'overall_health': 'UNKNOWN',
            'healthy_sources': 0,
            'total_sources': len(self.source_health)
        }
        
        healthy_count = 0
        for source_name, health in self.source_health.items():
            cb = self.circuit_breakers[source_name]
            
            source_report = {
                'status': health.status.value,
                'success_rate': health.success_rate,
                'avg_quality_score': health.avg_quality_score,
                'avg_latency_ms': health.avg_latency_ms,
                'circuit_breaker_state': cb.state,
                'calls_total': health.success_count + health.failure_count,
                'last_success_time': health.last_success_time,
                'last_failure_time': health.last_failure_time
            }
            
            report['sources'][source_name] = source_report
            
            if health.status == DataSourceStatus.HEALTHY:
                healthy_count += 1
                
        report['healthy_sources'] = healthy_count
        
        # Overall health assessment
        if healthy_count >= self.min_sources:
            report['overall_health'] = 'GOOD'
        elif healthy_count > 0:
            report['overall_health'] = 'DEGRADED'
        else:
            report['overall_health'] = 'CRITICAL'
            
        return report
        
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get quorum policy performance metrics."""
        if not self.consensus_times:
            return {'decisions_made': 0, 'avg_consensus_time_us': 0}
            
        avg_time = statistics.mean(self.consensus_times)
        p95_time = statistics.quantiles(list(self.consensus_times), n=20)[18] if len(self.consensus_times) > 20 else avg_time
        
        return {
            'decisions_made': len(self.quorum_decisions),
            'avg_consensus_time_us': avg_time * 1e6,
            'p95_consensus_time_us': p95_time * 1e6,
            'recent_quality_avg': statistics.mean([d['quality_score'] for d in list(self.quorum_decisions)[-100:]]) if self.quorum_decisions else 0.0
        }

class DataFabricAggregator:
    """
    Main data fabric aggregator with quorum policy and circuit breaking.
    """
    
    def __init__(self, 
                 sources: List[str],
                 quorum_policy: Optional[QuorumPolicy] = None):
        self.sources = sources
        self.quorum_policy = quorum_policy or QuorumPolicy()
        
        # Source-specific configurations
        self.source_configs = {
            'binance': {'timeout': 5.0, 'retry_delay': 2.0},
            'coingecko': {'timeout': 10.0, 'retry_delay': 5.0},
            'okx': {'timeout': 7.0, 'retry_delay': 3.0}
        }
        
        self.aggregation_count = 0
        self.last_successful_data = {}
        
    async def aggregate_market_data(self, symbol: str) -> Tuple[Optional[Dict[str, Any]], Dict[str, Any]]:
        """
        Aggregate market data from multiple sources with quorum consensus.
        
        Returns:
            (aggregated_data, aggregation_metadata)
        """
        start_time = time.perf_counter()
        self.aggregation_count += 1
        
        # Collect data from available sources
        source_data = {}
        collection_errors = {}
        
        for source_name in self.sources:
            try:
                cb = self.quorum_policy.circuit_breakers[source_name]
                if not cb.can_attempt():
                    logger.debug(f"Skipping {source_name} - circuit breaker open")
                    continue
                    
                # Simulate data collection (replace with actual source calls)
                data = await self._fetch_from_source(source_name, symbol)
                if data:
                    source_data[source_name] = data
                    
            except Exception as e:
                collection_errors[source_name] = str(e)
                logger.warning(f"Failed to collect from {source_name}: {e}")
                
        # Evaluate consensus
        consensus_data, consensus_metadata = self.quorum_policy.evaluate_consensus(
            source_data, symbol
        )
        
        # Fallback to last known good data if consensus fails
        if consensus_data is None and symbol in self.last_successful_data:
            logger.info(f"Using last known good data for {symbol}")
            fallback_data = self.last_successful_data[symbol].copy()
            fallback_data['timestamp'] = time.time()
            fallback_data['data_quality_score'] = 0.5  # Degraded quality
            fallback_data['fallback_used'] = True
            
            consensus_data = fallback_data
            consensus_metadata['fallback_used'] = True
            
        # Cache successful data
        if consensus_data and consensus_data.get('data_quality_score', 0) > 0.6:
            self.last_successful_data[symbol] = consensus_data.copy()
            
        # Build aggregation metadata
        aggregation_time = time.perf_counter() - start_time
        metadata = {
            'aggregation_id': self.aggregation_count,
            'timestamp': time.time(),
            'sources_attempted': len(self.sources),
            'sources_available': len(source_data),
            'collection_errors': collection_errors,
            'aggregation_time_us': aggregation_time * 1e6,
            'consensus_metadata': consensus_metadata
        }
        
        return consensus_data, metadata
        
    async def _fetch_from_source(self, source_name: str, symbol: str) -> Optional[Dict[str, Any]]:
        """
        Fetch data from specific source (mock implementation).
        Replace with actual source connectors.
        """
        config = self.source_configs.get(source_name, {})
        timeout = config.get('timeout', 5.0)
        
        try:
            # Mock data generation (replace with real source calls)
            import random
            base_price = 50000.0
            
            if random.random() < 0.1:  # 10% chance of source error
                raise Exception(f"{source_name} temporary error")
                
            return {
                'price': base_price + random.uniform(-100, 100),
                'volume': random.uniform(0.1, 10.0),
                'timestamp': time.time(),
                'source': source_name,
                'bid': base_price - 0.5,
                'ask': base_price + 0.5
            }
            
        except Exception as e:
            logger.warning(f"Source {source_name} fetch failed: {e}")
            return None
            
    def get_health_dashboard(self) -> Dict[str, Any]:
        """Get comprehensive health dashboard data."""
        return {
            'quorum_policy': self.quorum_policy.get_source_health_report(),
            'performance': self.quorum_policy.get_performance_metrics(),
            'aggregations_total': self.aggregation_count,
            'cached_symbols': list(self.last_successful_data.keys())
        }

# Factory functions
def create_quorum_policy(min_sources: int = 1, 
                        outlier_threshold: float = 0.05, 
                        min_quality_score: float = 0.7) -> QuorumPolicy:
    """Create quorum policy with specified parameters."""
    return QuorumPolicy(min_sources, outlier_threshold, min_quality_score)
    
def create_data_fabric_aggregator(sources: List[str]) -> DataFabricAggregator:
    """Create data fabric aggregator."""
    return DataFabricAggregator(sources)

# Testing
if __name__ == "__main__":
    import asyncio
    
    async def test_quorum_policy():
        """Test quorum policy functionality."""
        print("🧪 Testing Data Fabric Quorum Policy...")
        
        aggregator = create_data_fabric_aggregator(['binance', 'coingecko', 'okx'])
        
        # Test normal operation
        for i in range(10):
            data, metadata = await aggregator.aggregate_market_data("BTC-USDT")
            if data:
                print(f"✅ Consensus {i+1}: price={data['price']:.2f}, quality={data['data_quality_score']:.2f}, sources={len(data['sources_used'])}")
            else:
                print(f"❌ Consensus {i+1}: failed - {metadata}")
                
        # Health report
        health = aggregator.get_health_dashboard()
        print(f"📊 Health: {health['quorum_policy']['overall_health']}")
        print(f"   Healthy sources: {health['quorum_policy']['healthy_sources']}/{health['quorum_policy']['total_sources']}")
        
    asyncio.run(test_quorum_policy())