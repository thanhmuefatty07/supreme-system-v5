"""
Data Cache Layer - High-performance caching with Redis + PostgreSQL persistence
ULTRA SFL implementation for enterprise-grade data caching
"""

import asyncio
import json
import time
from datetime import datetime, timezone
from typing import Dict, Optional, Any, List
from dataclasses import dataclass, field

import aioredis
import asyncpg
from loguru import logger
from prometheus_client import Counter, Gauge, Histogram

from .normalizer import MarketDataPoint

# Metrics
CACHE_HITS = Counter('cache_hits_total', 'Cache hit count', ['cache_type'])
CACHE_MISSES = Counter('cache_misses_total', 'Cache miss count', ['cache_type'])
CACHE_SIZE = Gauge('cache_size_bytes', 'Cache size in bytes', ['cache_type'])
CACHE_LATENCY = Histogram('cache_latency_seconds', 'Cache operation latency', ['operation'])

@dataclass
class CacheConfig:
    """Cache configuration"""
    redis_url: str = "redis://localhost:6379/0"
    postgres_url: str = "postgresql://postgres:supreme_password@localhost:5432/supreme_trading"

    # Redis settings
    redis_max_connections: int = 20
    redis_ttl_seconds: int = 300  # 5 minutes default TTL

    # PostgreSQL settings
    pg_min_connections: int = 5
    pg_max_connections: int = 20

    # Cache settings
    enable_memory_cache: bool = True
    enable_redis_cache: bool = True
    enable_postgres_persistence: bool = True

    # Performance settings
    max_memory_items: int = 10000
    cleanup_interval_seconds: int = 300  # 5 minutes

class MemoryCache:
    """In-memory LRU cache for ultra-fast access"""

    def __init__(self, max_items: int = 10000):
        self.max_items = max_items
        self.cache: Dict[str, Dict[str, Any]] = {}
        self.access_times: Dict[str, float] = {}
        self._lock = asyncio.Lock()

    async def get(self, key: str) -> Optional[Any]:
        """Get item from memory cache"""
        async with self._lock:
            if key in self.cache:
                # Update access time for LRU
                self.access_times[key] = time.time()
                CACHE_HITS.labels(cache_type='memory').inc()
                return self.cache[key]['data']
            else:
                CACHE_MISSES.labels(cache_type='memory').inc()
                return None

    async def set(self, key: str, value: Any, ttl_seconds: int = 300):
        """Set item in memory cache"""
        async with self._lock:
            # Evict LRU items if at capacity
            if len(self.cache) >= self.max_items:
                await self._evict_lru()

            expire_time = time.time() + ttl_seconds
            self.cache[key] = {
                'data': value,
                'expire_time': expire_time
            }
            self.access_times[key] = time.time()

            CACHE_SIZE.labels(cache_type='memory').set(len(self.cache))

    async def delete(self, key: str):
        """Delete item from memory cache"""
        async with self._lock:
            if key in self.cache:
                del self.cache[key]
                del self.access_times[key]
                CACHE_SIZE.labels(cache_type='memory').set(len(self.cache))

    async def cleanup_expired(self):
        """Remove expired items"""
        async with self._lock:
            current_time = time.time()
            expired_keys = [
                key for key, data in self.cache.items()
                if data['expire_time'] <= current_time
            ]

            for key in expired_keys:
                del self.cache[key]
                del self.access_times[key]

            if expired_keys:
                logger.debug(f"🧹 Memory cache cleanup: removed {len(expired_keys)} expired items")
                CACHE_SIZE.labels(cache_type='memory').set(len(self.cache))

    async def _evict_lru(self):
        """Evict least recently used items"""
        if not self.access_times:
            return

        # Find oldest access time
        oldest_key = min(self.access_times.keys(), key=lambda k: self.access_times[k])

        # Remove oldest item
        del self.cache[oldest_key]
        del self.access_times[oldest_key]

        logger.debug(f"🗑️ Memory cache LRU eviction: removed {oldest_key}")

    def stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        return {
            'items': len(self.cache),
            'max_items': self.max_items,
            'utilization_percent': (len(self.cache) / self.max_items) * 100,
            'oldest_access': min(self.access_times.values()) if self.access_times else None,
            'newest_access': max(self.access_times.values()) if self.access_times else None
        }

class RedisCache:
    """Redis-based distributed cache"""

    def __init__(self, redis_url: str, max_connections: int = 20):
        self.redis_url = redis_url
        self.max_connections = max_connections
        self.redis: Optional[aioredis.Redis] = None
        self._lock = asyncio.Lock()

    async def connect(self):
        """Connect to Redis"""
        try:
            self.redis = aioredis.from_url(
                self.redis_url,
                max_connections=self.max_connections,
                encoding='utf-8',
                decode_responses=True
            )
            await self.redis.ping()
            logger.info("✅ Redis cache connected")
        except Exception as e:
            logger.error(f"❌ Redis cache connection failed: {e}")
            self.redis = None

    async def disconnect(self):
        """Disconnect from Redis"""
        if self.redis:
            await self.redis.close()
            self.redis = None
            logger.info("✅ Redis cache disconnected")

    async def get(self, key: str) -> Optional[Any]:
        """Get item from Redis cache"""
        if not self.redis:
            return None

        try:
            start_time = time.time()
            data = await self.redis.get(key)

            if data:
                # Deserialize JSON data
                parsed_data = json.loads(data)
                CACHE_HITS.labels(cache_type='redis').inc()
                CACHE_LATENCY.labels(operation='get').observe(time.time() - start_time)
                return parsed_data
            else:
                CACHE_MISSES.labels(cache_type='redis').inc()
                CACHE_LATENCY.labels(operation='get').observe(time.time() - start_time)
                return None

        except Exception as e:
            logger.error(f"❌ Redis cache get error for {key}: {e}")
            return None

    async def set(self, key: str, value: Any, ttl_seconds: int = 300):
        """Set item in Redis cache"""
        if not self.redis:
            return

        try:
            start_time = time.time()
            # Serialize data to JSON
            data = json.dumps(value, default=str)
            await self.redis.setex(key, ttl_seconds, data)
            CACHE_LATENCY.labels(operation='set').observe(time.time() - start_time)

        except Exception as e:
            logger.error(f"❌ Redis cache set error for {key}: {e}")

    async def delete(self, key: str):
        """Delete item from Redis cache"""
        if not self.redis:
            return

        try:
            await self.redis.delete(key)
        except Exception as e:
            logger.error(f"❌ Redis cache delete error for {key}: {e}")

    async def get_size(self) -> int:
        """Get approximate cache size in bytes"""
        if not self.redis:
            return 0

        try:
            info = await self.redis.info('memory')
            return info.get('used_memory', 0)
        except Exception:
            return 0

class PostgreSQLPersistence:
    """PostgreSQL-based persistent storage for historical data"""

    def __init__(self, postgres_url: str, min_connections: int = 5, max_connections: int = 20):
        self.postgres_url = postgres_url
        self.min_connections = min_connections
        self.max_connections = max_connections
        self.pool: Optional[asyncpg.Pool] = None

    async def connect(self):
        """Connect to PostgreSQL"""
        try:
            self.pool = await asyncpg.create_pool(
                self.postgres_url,
                min_size=self.min_connections,
                max_size=self.max_connections,
                command_timeout=30
            )

            # Ensure tables exist
            await self._create_tables()
            logger.info("✅ PostgreSQL persistence connected")

        except Exception as e:
            logger.error(f"❌ PostgreSQL connection failed: {e}")
            self.pool = None

    async def disconnect(self):
        """Disconnect from PostgreSQL"""
        if self.pool:
            await self.pool.close()
            self.pool = None
            logger.info("✅ PostgreSQL persistence disconnected")

    async def _create_tables(self):
        """Create necessary tables if they don't exist"""
        if not self.pool:
            return

        try:
            async with self.pool.acquire() as conn:
                # Market data table
                await conn.execute("""
                    CREATE TABLE IF NOT EXISTS market_data (
                        id SERIAL PRIMARY KEY,
                        symbol VARCHAR(20) NOT NULL,
                        timestamp TIMESTAMP WITH TIME ZONE NOT NULL,
                        price DECIMAL(20, 10),
                        volume_24h DECIMAL(30, 10),
                        change_24h DECIMAL(10, 5),
                        high_24h DECIMAL(20, 10),
                        low_24h DECIMAL(20, 10),
                        bid DECIMAL(20, 10),
                        ask DECIMAL(20, 10),
                        market_cap DECIMAL(30, 10),
                        source VARCHAR(50) NOT NULL,
                        quality_score DECIMAL(3, 2) DEFAULT 1.0,
                        created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),

                        UNIQUE(symbol, timestamp, source)
                    )
                """)

                # Create indexes for performance
                await conn.execute("""
                    CREATE INDEX IF NOT EXISTS idx_market_data_symbol_timestamp
                    ON market_data (symbol, timestamp DESC)
                """)

                await conn.execute("""
                    CREATE INDEX IF NOT EXISTS idx_market_data_source_timestamp
                    ON market_data (source, timestamp DESC)
                """)

                logger.debug("✅ PostgreSQL tables created/verified")

        except Exception as e:
            logger.error(f"❌ PostgreSQL table creation failed: {e}")

    async def save_market_data(self, data_point: MarketDataPoint):
        """Save market data point to PostgreSQL"""
        if not self.pool or not data_point:
            return

        try:
            async with self.pool.acquire() as conn:
                await conn.execute("""
                    INSERT INTO market_data (
                        symbol, timestamp, price, volume_24h, change_24h,
                        high_24h, low_24h, bid, ask, market_cap,
                        source, quality_score
                    ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12)
                    ON CONFLICT (symbol, timestamp, source)
                    DO UPDATE SET
                        price = EXCLUDED.price,
                        volume_24h = EXCLUDED.volume_24h,
                        change_24h = EXCLUDED.change_24h,
                        high_24h = EXCLUDED.high_24h,
                        low_24h = EXCLUDED.low_24h,
                        bid = EXCLUDED.bid,
                        ask = EXCLUDED.ask,
                        market_cap = EXCLUDED.market_cap,
                        quality_score = EXCLUDED.quality_score
                """,
                data_point.symbol,
                datetime.fromtimestamp(data_point.timestamp, tz=timezone.utc),
                data_point.price,
                data_point.volume_24h,
                data_point.change_24h,
                data_point.high_24h,
                data_point.low_24h,
                data_point.bid,
                data_point.ask,
                data_point.market_cap,
                data_point.source,
                data_point.quality_score
                )

                logger.debug(f"💾 PostgreSQL saved: {data_point.symbol} @ {data_point.price}")

        except Exception as e:
            logger.error(f"❌ PostgreSQL save error: {e}")

    async def get_historical_data(self, symbol: str, limit: int = 100) -> List[Dict[str, Any]]:
        """Get historical data for symbol"""
        if not self.pool:
            return []

        try:
            async with self.pool.acquire() as conn:
                rows = await conn.fetch("""
                    SELECT * FROM market_data
                    WHERE symbol = $1
                    ORDER BY timestamp DESC
                    LIMIT $2
                """, symbol, limit)

                return [dict(row) for row in rows]

        except Exception as e:
            logger.error(f"❌ PostgreSQL historical data error for {symbol}: {e}")
            return []

class CacheManager:
    """
    Unified cache manager with multi-layer caching
    Memory -> Redis -> PostgreSQL hierarchy
    """

    def __init__(self, config: Optional[CacheConfig] = None):
        self.config = config or CacheConfig()

        # Initialize cache layers
        self.memory_cache = MemoryCache(self.config.max_memory_items) if self.config.enable_memory_cache else None
        self.redis_cache = RedisCache(self.config.redis_url, self.config.redis_max_connections) if self.config.enable_redis_cache else None
        self.postgres = PostgreSQLPersistence(self.config.postgres_url, self.config.pg_min_connections, self.config.pg_max_connections) if self.config.enable_postgres_persistence else None

        # Background cleanup task
        self.cleanup_task: Optional[asyncio.Task] = None
        self.running = False

        logger.info("🏗️ Cache manager initialized")

    async def start(self):
        """Start cache manager and connect to backends"""
        logger.info("🚀 Starting cache manager...")

        # Connect to Redis if enabled
        if self.redis_cache:
            await self.redis_cache.connect()

        # Connect to PostgreSQL if enabled
        if self.postgres:
            await self.postgres.connect()

        # Start cleanup task
        self.running = True
        self.cleanup_task = asyncio.create_task(self._cleanup_loop())

        logger.info("✅ Cache manager started")

    async def stop(self):
        """Stop cache manager and disconnect from backends"""
        logger.info("🛑 Stopping cache manager...")

        self.running = False

        if self.cleanup_task:
            self.cleanup_task.cancel()
            try:
                await self.cleanup_task
            except asyncio.CancelledError:
                pass

        if self.redis_cache:
            await self.redis_cache.disconnect()

        if self.postgres:
            await self.postgres.disconnect()

        logger.info("✅ Cache manager stopped")

    async def get(self, key: str) -> Optional[Any]:
        """
        Get data from cache hierarchy
        Memory -> Redis -> PostgreSQL (for historical data)
        """
        # Try memory cache first
        if self.memory_cache:
            data = await self.memory_cache.get(key)
            if data:
                return data

        # Try Redis cache
        if self.redis_cache:
            data = await self.redis_cache.get(key)
            if data:
                # Populate memory cache for faster future access
                if self.memory_cache:
                    await self.memory_cache.set(key, data, self.config.redis_ttl_seconds)
                return data

        # For historical data, try PostgreSQL (different key format)
        if self.postgres and key.startswith('historical:'):
            symbol = key.replace('historical:', '')
            historical_data = await self.postgres.get_historical_data(symbol, 1000)
            if historical_data:
                # Cache in Redis for future requests
                if self.redis_cache:
                    await self.redis_cache.set(key, historical_data, 3600)  # 1 hour TTL
                return historical_data

        return None

    async def set(self, key: str, value: Any, ttl_seconds: Optional[int] = None):
        """
        Set data in cache hierarchy
        Memory + Redis (for current data)
        PostgreSQL (for persistent storage)
        """
        ttl = ttl_seconds or self.config.redis_ttl_seconds

        # Set in memory cache
        if self.memory_cache:
            await self.memory_cache.set(key, value, ttl)

        # Set in Redis cache
        if self.redis_cache:
            await self.redis_cache.set(key, value, ttl)

        # Persist to PostgreSQL if it's market data
        if self.postgres and isinstance(value, MarketDataPoint):
            await self.postgres.save_market_data(value)

    async def delete(self, key: str):
        """Delete data from all cache layers"""
        if self.memory_cache:
            await self.memory_cache.delete(key)

        if self.redis_cache:
            await self.redis_cache.delete(key)

        # Note: PostgreSQL data is kept for historical purposes

    async def _cleanup_loop(self):
        """Background cleanup loop"""
        while self.running:
            try:
                # Memory cache cleanup
                if self.memory_cache:
                    await self.memory_cache.cleanup_expired()

                # Wait for next cleanup cycle
                await asyncio.sleep(self.config.cleanup_interval_seconds)

            except Exception as e:
                logger.error(f"❌ Cache cleanup error: {e}")
                await asyncio.sleep(60)  # Wait before retrying

    def get_stats(self) -> Dict[str, Any]:
        """Get comprehensive cache statistics"""
        stats = {
            'config': {
                'memory_enabled': self.config.enable_memory_cache,
                'redis_enabled': self.config.enable_redis_cache,
                'postgres_enabled': self.config.enable_postgres_persistence,
                'redis_ttl_seconds': self.config.redis_ttl_seconds,
                'max_memory_items': self.config.max_memory_items
            }
        }

        # Memory cache stats
        if self.memory_cache:
            stats['memory'] = self.memory_cache.stats()

        # Redis cache stats
        if self.redis_cache:
            stats['redis'] = {
                'connected': self.redis_cache.redis is not None,
                'size_bytes': self.redis_cache.get_size() if self.redis_cache.redis else 0
            }

        # PostgreSQL stats
        stats['postgres'] = {
            'connected': self.postgres.pool is not None if self.postgres else False
        }

        return stats

# DataCache class for backward compatibility
class DataCache:
    """
    Simplified cache interface for data operations
    Wraps CacheManager for easier usage
    """

    def __init__(self, cache_manager: CacheManager):
        self.cache_manager = cache_manager

    async def get_market_data(self, symbol: str) -> Optional[MarketDataPoint]:
        """Get cached market data for symbol"""
        data = await self.cache_manager.get(f"market:{symbol}")
        if data and isinstance(data, dict):
            # Reconstruct MarketDataPoint from dict
            try:
                return MarketDataPoint(**data)
            except Exception:
                return None
        return data

    async def set_market_data(self, symbol: str, data_point: MarketDataPoint):
        """Cache market data for symbol"""
        await self.cache_manager.set(f"market:{symbol}", data_point)

    async def get_historical_data(self, symbol: str, limit: int = 100) -> List[Dict[str, Any]]:
        """Get historical data for symbol"""
        data = await self.cache_manager.get(f"historical:{symbol}")
        if data and isinstance(data, list):
            return data[:limit]
        return []

    async def invalidate_symbol(self, symbol: str):
        """Invalidate all cached data for symbol"""
        await self.cache_manager.delete(f"market:{symbol}")
        await self.cache_manager.delete(f"historical:{symbol}")

    async def get_cache_health(self) -> Dict[str, Any]:
        """Get cache health status"""
        return self.cache_manager.get_stats()
