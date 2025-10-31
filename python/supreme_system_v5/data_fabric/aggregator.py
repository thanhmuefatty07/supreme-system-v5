"""
Data Aggregator - Multi-source free API orchestration
Ultra SFL implementation for 99.9% uptime data feed
"""

import asyncio
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Callable, Dict, List, Optional

import polars as pl
from loguru import logger
from prometheus_client import Counter, Gauge, Histogram

from .normalizer import DataNormalizer, MarketDataPoint
from .quality import DataQualityScorer, QualityMetrics

# Metrics
API_CALLS = Counter("data_api_calls_total", "Total API calls", ["source", "endpoint"])
API_ERRORS = Counter("data_api_errors_total", "API errors", ["source", "error_type"])
DATA_QUALITY = Gauge("data_quality_score", "Data quality score", ["source", "symbol"])
LATENCY = Histogram("data_latency_seconds", "Data fetch latency", ["source"])


class DataSourceStatus(Enum):
    """Data source health status"""

    HEALTHY = "healthy"
    DEGRADED = "degraded"
    FAILED = "failed"
    CIRCUIT_OPEN = "circuit_open"


@dataclass
class DataSource:
    """Data source configuration and state"""

    name: str
    connector: Any
    priority: int  # Lower = higher priority
    weight: float = 1.0
    status: DataSourceStatus = DataSourceStatus.HEALTHY
    last_success: float = 0.0
    error_count: int = 0
    circuit_failures: int = 0
    circuit_open_until: float = 0.0
    quality_score: float = 1.0
    latency_p95: float = 0.0

    # Circuit breaker config
    max_errors: int = 5
    circuit_timeout: int = 300  # 5 minutes

    def is_available(self) -> bool:
        """Check if source is available for requests"""
        if self.status == DataSourceStatus.CIRCUIT_OPEN:
            if time.time() > self.circuit_open_until:
                self.status = DataSourceStatus.HEALTHY
                self.circuit_failures = 0
                logger.info(f"🔄 Circuit breaker reset for {self.name}")
                return True
            return False
        return self.status != DataSourceStatus.FAILED

    def record_success(self, latency: float):
        """Record successful API call"""
        self.last_success = time.time()
        self.error_count = 0
        self.latency_p95 = latency * 0.1 + self.latency_p95 * 0.9  # EMA

        if self.status != DataSourceStatus.HEALTHY:
            self.status = DataSourceStatus.HEALTHY
            logger.info(f"✅ {self.name} recovered")

    def record_error(self, error_type: str):
        """Record API error and manage circuit breaker"""
        self.error_count += 1
        API_ERRORS.labels(source=self.name, error_type=error_type).inc()

        if self.error_count >= self.max_errors:
            self.status = DataSourceStatus.CIRCUIT_OPEN
            self.circuit_open_until = time.time() + self.circuit_timeout
            self.circuit_failures += 1
            logger.error(
                f"🚨 Circuit breaker opened for {self.name} ({self.error_count} errors)"
            )
        elif self.error_count >= 2:
            self.status = DataSourceStatus.DEGRADED
            logger.warning(f"⚠️ {self.name} degraded ({self.error_count} errors)")


class DataAggregator:
    """
    Multi-source data aggregator with intelligent failover
    Achieves 99.9% uptime through source diversity and quality scoring
    """

    def __init__(self, cache_manager=None):
        """Initialize data aggregator"""
        self.sources: Dict[str, DataSource] = {}
        self.normalizer = DataNormalizer()
        self.quality_scorer = DataQualityScorer()

        # Initialize cache manager if not provided
        if cache_manager is None:
            from .cache import CacheManager

            self.cache_manager = CacheManager()
        else:
            self.cache_manager = cache_manager

        self.callbacks: Dict[str, List[Callable]] = {}
        self.running = False

        # State tracking
        self.last_data: Dict[str, MarketDataPoint] = {}
        self.data_timestamps: Dict[str, float] = {}

        logger.info("🏭 Data Fabric initialized")

    def add_source(self, source: DataSource):
        """Add data source to aggregator"""
        self.sources[source.name] = source
        logger.info(f"➕ Added data source: {source.name} (priority: {source.priority})")

    def add_callback(self, symbol: str, callback: Callable):
        """Add callback for symbol data updates"""
        if symbol not in self.callbacks:
            self.callbacks[symbol] = []
        self.callbacks[symbol].append(callback)
        logger.info(f"📡 Added callback for {symbol}")

    async def get_market_data(
        self, symbol: str, max_age_seconds: float = 60
    ) -> Optional[MarketDataPoint]:
        """
        Get latest market data with intelligent source selection
        Returns aggregated data from multiple sources with quality scoring
        """
        start_time = time.time()

        # Check cache first
        if hasattr(self.cache_manager, "get_market_data"):
            # Using DataCache interface
            cached_data = await self.cache_manager.get_market_data(symbol)
            if cached_data and (time.time() - cached_data.timestamp) < max_age_seconds:
                return cached_data
        elif self.cache_manager:
            # Using raw CacheManager interface
            cached_data = await self.cache_manager.get(f"market:{symbol}")
            if (
                cached_data
                and isinstance(cached_data, MarketDataPoint)
                and (time.time() - cached_data.timestamp) < max_age_seconds
            ):
                return cached_data

        # Get available sources sorted by priority and quality
        available_sources = [
            source for source in self.sources.values() if source.is_available()
        ]

        if not available_sources:
            logger.error(f"❌ No available sources for {symbol}")
            return None

        # Sort by priority (lower = better) and quality score (higher = better)
        available_sources.sort(key=lambda s: (s.priority, -s.quality_score))

        # Try sources until success
        data_points = []
        for source in available_sources[:3]:  # Try top 3 sources
            try:
                source_start = time.time()

                # Fetch data from source
                raw_data = await source.connector.get_price_data(symbol)

                if raw_data:
                    # Normalize data
                    normalized = self.normalizer.normalize(source.name, raw_data)

                    # Score quality
                    quality = self.quality_scorer.score(normalized, symbol)
                    source.quality_score = quality.overall_score

                    # Record metrics
                    latency = time.time() - source_start
                    source.record_success(latency)
                    LATENCY.labels(source=source.name).observe(latency)
                    DATA_QUALITY.labels(source=source.name, symbol=symbol).set(
                        quality.overall_score
                    )

                    data_points.append((normalized, quality, source.weight))
                    logger.debug(
                        f"✅ {source.name}: {symbol} = ${normalized.price:.4f} (quality: {quality.overall_score:.2f})"
                    )

            except Exception as e:
                source.record_error(type(e).__name__)
                logger.warning(f"⚠️ {source.name} failed for {symbol}: {e}")
                continue

        if not data_points:
            logger.error(f"❌ All sources failed for {symbol}")
            return None

        # Aggregate multiple data points
        aggregated_data = self._aggregate_data_points(symbol, data_points)

        # Cache result
        if self.cache_manager and aggregated_data:
            if hasattr(self.cache_manager, "set_market_data"):
                # Using DataCache interface
                await self.cache_manager.set_market_data(symbol, aggregated_data)
            else:
                # Using raw CacheManager interface
                await self.cache_manager.set(f"market:{symbol}", aggregated_data)

        # Update last data
        self.last_data[symbol] = aggregated_data
        self.data_timestamps[symbol] = time.time()

        total_latency = time.time() - start_time
        LATENCY.labels(source="aggregator").observe(total_latency)

        return aggregated_data

    def _aggregate_data_points(
        self, symbol: str, data_points: List[tuple]
    ) -> MarketDataPoint:
        """
        Aggregate multiple data points using weighted average based on quality
        """
        if len(data_points) == 1:
            return data_points[0][0]  # Single source, return as-is

        # Extract data for aggregation
        prices = []
        volumes = []
        weights = []

        for data_point, quality, source_weight in data_points:
            weight = quality.overall_score * source_weight
            prices.append(data_point.price * weight)
            volumes.append(data_point.volume * weight)
            weights.append(weight)

        total_weight = sum(weights)
        if total_weight == 0:
            # Fallback to first source
            return data_points[0][0]

        # Weighted average
        aggregated_price = sum(prices) / total_weight
        aggregated_volume = sum(volumes) / total_weight

        # Use best source's metadata
        best_data = max(data_points, key=lambda x: x[1].overall_score)[0]

        return MarketDataPoint(
            symbol=symbol,
            timestamp=time.time(),
            price=aggregated_price,
            volume=aggregated_volume,
            bid=best_data.bid,
            ask=best_data.ask,
            high_24h=best_data.high_24h,
            low_24h=best_data.low_24h,
            change_24h=best_data.change_24h,
            source="aggregated",
            quality_score=max(dp[1].overall_score for dp in data_points),
        )

    async def start_streaming(self, symbols: List[str], interval_seconds: float = 5.0):
        """
        Start streaming market data for symbols
        Maintains continuous feed with automatic failover
        """
        logger.info(f"📡 Starting data streaming for {len(symbols)} symbols")
        self.running = True

        # Create streaming tasks for each symbol
        tasks = []
        for symbol in symbols:
            task = asyncio.create_task(
                self._stream_symbol_data(symbol, interval_seconds)
            )
            tasks.append(task)

        # Start source health monitoring
        health_task = asyncio.create_task(self._monitor_source_health())
        tasks.append(health_task)

        # Wait for all tasks
        try:
            await asyncio.gather(*tasks)
        except Exception as e:
            logger.error(f"💥 Streaming error: {e}")
            self.running = False

    async def _stream_symbol_data(self, symbol: str, interval_seconds: float):
        """
        Stream data for a single symbol
        """
        logger.info(f"📈 Starting stream for {symbol}")

        while self.running:
            try:
                # Get latest market data
                data = await self.get_market_data(
                    symbol, max_age_seconds=interval_seconds
                )

                if data:
                    # Notify callbacks
                    if symbol in self.callbacks:
                        for callback in self.callbacks[symbol]:
                            try:
                                await callback(data)
                            except Exception as e:
                                logger.error(f"Callback error for {symbol}: {e}")

                # Wait for next interval
                await asyncio.sleep(interval_seconds)

            except Exception as e:
                logger.error(f"Stream error for {symbol}: {e}")
                await asyncio.sleep(min(interval_seconds * 2, 30))  # Backoff

    async def _monitor_source_health(self):
        """
        Monitor health of all data sources
        """
        while self.running:
            healthy_sources = 0
            total_sources = len(self.sources)

            for source in self.sources.values():
                if source.is_available():
                    healthy_sources += 1

            health_ratio = healthy_sources / total_sources if total_sources > 0 else 0

            if health_ratio < 0.5:
                logger.error(
                    f"🚨 Data fabric degraded: {healthy_sources}/{total_sources} sources healthy"
                )
            elif health_ratio < 0.8:
                logger.warning(
                    f"⚠️ Data fabric partial: {healthy_sources}/{total_sources} sources healthy"
                )

            # Wait 30 seconds between health checks
            await asyncio.sleep(30)

    async def stop(self):
        """Stop data streaming"""
        logger.info("🛑 Stopping data fabric...")
        self.running = False

        # Disconnect all sources
        for source in self.sources.values():
            try:
                if hasattr(source.connector, "disconnect"):
                    await source.connector.disconnect()
            except Exception as e:
                logger.warning(f"Error disconnecting {source.name}: {e}")

        logger.info("✅ Data fabric stopped")

    def get_source_health(self) -> Dict[str, Dict[str, Any]]:
        """
        Get health status of all sources
        """
        return {
            source_name: {
                "status": source.status.value,
                "quality_score": source.quality_score,
                "error_count": source.error_count,
                "last_success": source.last_success,
                "latency_p95": source.latency_p95,
                "circuit_failures": source.circuit_failures,
                "available": source.is_available(),
            }
            for source_name, source in self.sources.items()
        }
