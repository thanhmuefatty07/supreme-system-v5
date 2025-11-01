# python/supreme_system_v5/data_fabric/quality_tracker.py
"""
Data Fabric Quality Tracker - ULTRA SFL Implementation
Tracks and persists quality history and source health metrics
"""

import asyncio
import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta

import asyncpg
from loguru import logger


@dataclass
class QualityMetrics:
    """Quality metrics for a single data point"""
    symbol: str
    source: str
    quality_score: float
    response_time_ms: int
    data_freshness_seconds: int
    success: bool = True
    spread_bps: Optional[int] = None
    volume_score: Optional[float] = None
    error_message: Optional[str] = None


@dataclass
class SourceHealth:
    """Health status of a data source"""
    source: str
    status: str = "healthy"  # healthy, degraded, down
    uptime_percent: float = 1.0
    avg_response_time_ms: int = 0
    avg_quality_score: float = 1.0
    total_requests: int = 0
    successful_requests: int = 0
    consecutive_failures: int = 0
    last_failure_ts: Optional[datetime] = None
    last_success_ts: Optional[datetime] = None
    circuit_breaker_active: bool = False
    circuit_breaker_until: Optional[datetime] = None


@dataclass
class QualityTrackerConfig:
    """Configuration for quality tracking"""
    enable_persistence: bool = True
    batch_size: int = 100  # Batch size for bulk inserts
    flush_interval_seconds: int = 30  # How often to flush metrics to DB
    retention_days: int = 30  # How long to keep quality history
    health_check_interval_seconds: int = 60  # Health status update interval


class QualityTracker:
    """
    Data Fabric Quality Tracker
    Tracks quality metrics and source health with database persistence
    """

    def __init__(self, config: Optional[QualityTrackerConfig] = None, db_url: Optional[str] = None):
        self.config = config or QualityTrackerConfig()
        self.db_url = db_url or "postgresql://postgres:supreme_password@localhost:5432/supreme_trading"
        self.db_pool: Optional[asyncpg.Pool] = None

        # In-memory tracking
        self.quality_buffer: List[QualityMetrics] = []
        self.source_health: Dict[str, SourceHealth] = {}
        self.last_flush_time = time.time()
        self.last_health_check = time.time()

        # Circuit breaker settings
        self.circuit_breaker_threshold = 5  # Consecutive failures to trigger
        self.circuit_breaker_timeout_minutes = 5  # Minutes to wait before retry

        logger.info("🎯 Quality Tracker initialized")

    async def connect(self):
        """Connect to database"""
        try:
            self.db_pool = await asyncpg.create_pool(
                self.db_url,
                min_size=1,
                max_size=10,
                command_timeout=30
            )
            logger.info("✅ Quality Tracker database connected")

            # Initialize source health from database
            await self._load_source_health()

        except Exception as e:
            logger.error(f"❌ Quality Tracker database connection failed: {e}")
            self.db_pool = None

    async def disconnect(self):
        """Disconnect from database"""
        if self.db_pool:
            await self.db_pool.close()
            self.db_pool = None
            logger.info("✅ Quality Tracker database disconnected")

    async def _load_source_health(self):
        """Load source health from database"""
        if not self.db_pool:
            return

        try:
            rows = await self.db_pool.fetch("""
                SELECT source, status, uptime_percent, avg_response_time_ms,
                       avg_quality_score, total_requests, successful_requests,
                       consecutive_failures, last_failure_ts, last_success_ts,
                       circuit_breaker_active, circuit_breaker_until
                FROM source_health
            """)

            for row in rows:
                health = SourceHealth(
                    source=row['source'],
                    status=row['status'],
                    uptime_percent=float(row['uptime_percent'] or 1.0),
                    avg_response_time_ms=int(row['avg_response_time_ms'] or 0),
                    avg_quality_score=float(row['avg_quality_score'] or 1.0),
                    total_requests=int(row['total_requests'] or 0),
                    successful_requests=int(row['successful_requests'] or 0),
                    consecutive_failures=int(row['consecutive_failures'] or 0),
                    last_failure_ts=row['last_failure_ts'],
                    last_success_ts=row['last_success_ts'],
                    circuit_breaker_active=bool(row['circuit_breaker_active']),
                    circuit_breaker_until=row['circuit_breaker_until']
                )
                self.source_health[row['source']] = health

            logger.info(f"✅ Loaded health data for {len(self.source_health)} sources")

        except Exception as e:
            logger.warning(f"⚠️ Failed to load source health: {e}")

    async def record_quality_metrics(self, metrics: QualityMetrics):
        """Record quality metrics for a data point"""
        # Add to buffer
        self.quality_buffer.append(metrics)

        # Update source health in memory
        await self._update_source_health(metrics)

        # Flush if buffer is full or time threshold reached
        current_time = time.time()
        if (len(self.quality_buffer) >= self.config.batch_size or
            current_time - self.last_flush_time >= self.config.flush_interval_seconds):
            await self._flush_metrics()

        # Periodic health check
        if current_time - self.last_health_check >= self.config.health_check_interval_seconds:
            await self._update_health_statuses()
            self.last_health_check = current_time

    async def _update_source_health(self, metrics: QualityMetrics):
        """Update source health based on metrics"""
        source = metrics.source

        if source not in self.source_health:
            self.source_health[source] = SourceHealth(source=source)

        health = self.source_health[source]

        # Update counters
        health.total_requests += 1
        if metrics.success:
            health.successful_requests += 1
            health.consecutive_failures = 0
            health.last_success_ts = datetime.now()
        else:
            health.consecutive_failures += 1
            health.last_failure_ts = datetime.now()

        # Update averages (rolling)
        if metrics.success:
            # Update response time average
            if health.avg_response_time_ms == 0:
                health.avg_response_time_ms = metrics.response_time_ms
            else:
                health.avg_response_time_ms = int(
                    (health.avg_response_time_ms * (health.successful_requests - 1) +
                     metrics.response_time_ms) / health.successful_requests
                )

            # Update quality score average
            if health.avg_quality_score == 0:
                health.avg_quality_score = metrics.quality_score
            else:
                health.avg_quality_score = (
                    (health.avg_quality_score * (health.successful_requests - 1) +
                     metrics.quality_score) / health.successful_requests
                )

        # Calculate uptime percentage
        if health.total_requests > 0:
            health.uptime_percent = health.successful_requests / health.total_requests

        # Update status based on consecutive failures
        if health.consecutive_failures >= self.circuit_breaker_threshold:
            health.status = "down"
            health.circuit_breaker_active = True
            health.circuit_breaker_until = datetime.now() + timedelta(minutes=self.circuit_breaker_timeout_minutes)
        elif health.consecutive_failures >= 2:
            health.status = "degraded"
        else:
            health.status = "healthy"
            health.circuit_breaker_active = False
            health.circuit_breaker_until = None

    async def _flush_metrics(self):
        """Flush accumulated metrics to database"""
        if not self.quality_buffer or not self.db_pool:
            return

        try:
            # Bulk insert quality metrics
            await self.db_pool.executemany("""
                INSERT INTO quality_history (
                    symbol, source, quality_score, response_time_ms,
                    data_freshness_seconds, error_count, success_rate,
                    spread_bps, volume_score
                ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9)
            """, [
                (
                    m.symbol, m.source, m.quality_score, m.response_time_ms,
                    m.data_freshness_seconds,
                    0 if m.success else 1,  # error_count
                    1.0 if m.success else 0.0,  # success_rate
                    m.spread_bps, m.volume_score
                )
                for m in self.quality_buffer
            ])

            # Clear buffer
            flushed_count = len(self.quality_buffer)
            self.quality_buffer.clear()
            self.last_flush_time = time.time()

            logger.debug(f"✅ Flushed {flushed_count} quality metrics to database")

        except Exception as e:
            logger.error(f"❌ Failed to flush quality metrics: {e}")

    async def _update_health_statuses(self):
        """Update source health statuses in database"""
        if not self.db_pool:
            return

        try:
            # Bulk upsert source health
            for source, health in self.source_health.items():
                await self.db_pool.execute("""
                    INSERT INTO source_health (
                        source, status, uptime_percent, avg_response_time_ms,
                        avg_quality_score, total_requests, successful_requests,
                        consecutive_failures, last_failure_ts, last_success_ts,
                        circuit_breaker_active, circuit_breaker_until
                    ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12)
                    ON CONFLICT (source) DO UPDATE SET
                        status = EXCLUDED.status,
                        uptime_percent = EXCLUDED.uptime_percent,
                        avg_response_time_ms = EXCLUDED.avg_response_time_ms,
                        avg_quality_score = EXCLUDED.avg_quality_score,
                        total_requests = EXCLUDED.total_requests,
                        successful_requests = EXCLUDED.successful_requests,
                        consecutive_failures = EXCLUDED.consecutive_failures,
                        last_failure_ts = EXCLUDED.last_failure_ts,
                        last_success_ts = EXCLUDED.last_success_ts,
                        circuit_breaker_active = EXCLUDED.circuit_breaker_active,
                        circuit_breaker_until = EXCLUDED.circuit_breaker_until
                """, (
                    health.source, health.status, health.uptime_percent,
                    health.avg_response_time_ms, health.avg_quality_score,
                    health.total_requests, health.successful_requests,
                    health.consecutive_failures, health.last_failure_ts,
                    health.last_success_ts, health.circuit_breaker_active,
                    health.circuit_breaker_until
                ))

            logger.debug(f"✅ Updated health status for {len(self.source_health)} sources")

        except Exception as e:
            logger.error(f"❌ Failed to update source health: {e}")

    def get_source_health(self, source: str) -> Optional[SourceHealth]:
        """Get current health status for a source"""
        return self.source_health.get(source)

    def is_source_healthy(self, source: str) -> bool:
        """Check if a source is healthy (not down and circuit breaker not active)"""
        health = self.source_health.get(source)
        if not health:
            return True  # Unknown sources default to healthy

        if health.status == "down":
            return False

        if health.circuit_breaker_active:
            if health.circuit_breaker_until and datetime.now() > health.circuit_breaker_until:
                # Circuit breaker expired, reset
                health.circuit_breaker_active = False
                health.consecutive_failures = 0
                health.status = "healthy"
                return True
            return False

        return True

    async def get_quality_history(self, symbol: str, source: Optional[str] = None,
                                hours: int = 24) -> List[Dict]:
        """Get quality history for symbol/source combination"""
        if not self.db_pool:
            return []

        try:
            if source:
                rows = await self.db_pool.fetch("""
                    SELECT symbol, source, quality_score, response_time_ms,
                           data_freshness_seconds, success_rate, spread_bps,
                           volume_score, ts
                    FROM quality_history
                    WHERE symbol = $1 AND source = $2
                      AND ts > NOW() - INTERVAL '%s hours'
                    ORDER BY ts DESC
                    LIMIT 1000
                """ % hours, symbol, source)
            else:
                rows = await self.db_pool.fetch("""
                    SELECT symbol, source, quality_score, response_time_ms,
                           data_freshness_seconds, success_rate, spread_bps,
                           volume_score, ts
                    FROM quality_history
                    WHERE symbol = $1
                      AND ts > NOW() - INTERVAL '%s hours'
                    ORDER BY ts DESC
                    LIMIT 1000
                """ % hours, symbol)

            return [dict(row) for row in rows]

        except Exception as e:
            logger.error(f"❌ Failed to get quality history: {e}")
            return []

    async def cleanup_old_data(self):
        """Clean up old quality data beyond retention period"""
        if not self.db_pool:
            return

        try:
            # Delete old quality history
            deleted_history = await self.db_pool.execute("""
                DELETE FROM quality_history
                WHERE ts < NOW() - INTERVAL '%s days'
            """ % self.config.retention_days)

            # Delete old metrics
            deleted_metrics = await self.db_pool.execute("""
                DELETE FROM data_fabric_metrics
                WHERE ts < NOW() - INTERVAL '%s days'
            """ % self.config.retention_days)

            logger.info(f"🧹 Cleaned up {deleted_history} quality history and {deleted_metrics} metric records")

        except Exception as e:
            logger.error(f"❌ Failed to cleanup old data: {e}")


# Global quality tracker instance
_quality_tracker: Optional[QualityTracker] = None


async def get_quality_tracker() -> QualityTracker:
    """Get global quality tracker instance"""
    global _quality_tracker
    if _quality_tracker is None:
        _quality_tracker = QualityTracker()
        await _quality_tracker.connect()
    return _quality_tracker


async def record_data_quality(symbol: str, source: str, quality_score: float,
                            response_time_ms: int, data_freshness_seconds: int,
                            success: bool = True, spread_bps: Optional[int] = None,
                            volume_score: Optional[float] = None,
                            error_message: Optional[str] = None):
    """Convenience function to record quality metrics"""
    tracker = await get_quality_tracker()
    metrics = QualityMetrics(
        symbol=symbol,
        source=source,
        quality_score=quality_score,
        response_time_ms=response_time_ms,
        data_freshness_seconds=data_freshness_seconds,
        success=success,
        spread_bps=spread_bps,
        volume_score=volume_score,
        error_message=error_message
    )
    await tracker.record_quality_metrics(metrics)


async def check_source_health(source: str) -> bool:
    """Check if a source is healthy"""
    tracker = await get_quality_tracker()
    return tracker.is_source_healthy(source)
