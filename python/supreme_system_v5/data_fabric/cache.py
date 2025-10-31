"""
Data Cache - Multi-tier caching system
Memory -> Redis -> PostgreSQL with intelligent TTL management
"""

import asyncio
import json
import time
from typing import Dict, Optional, Any, List
from datetime import datetime, timedelta

import aioredis
import asyncpg
from loguru import logger

from .normalizer import MarketDataPoint

class DataCache:
    """
    Multi-tier cache for market data
    Layer 1: Memory (sub-ms access)
    Layer 2: Redis (ms access) 
    Layer 3: PostgreSQL (persistence)
    """
    
    def __init__(self, redis_url: str = None, postgres_url: str = None):
        """Initialize multi-tier cache"""
        self.redis_url = redis_url or "redis://localhost:6379/0"
        self.postgres_url = postgres_url or "postgresql://postgres:supreme_password@localhost:5432/supreme_trading"
        
        # Memory cache (L1)
        self.memory_cache: Dict[str, MarketDataPoint] = {}
        self.memory_ttl: Dict[str, float] = {}
        self.max_memory_items = 1000  # Limit for i3-4GB systems
        
        # Redis connection (L2)
        self.redis: Optional[aioredis.Redis] = None
        
        # PostgreSQL connection (L3)
        self.postgres: Optional[asyncpg.Connection] = None
        
        # Cache statistics
        self.stats = {
            'hits_memory': 0,
            'hits_redis': 0,
            'hits_postgres': 0,
            'misses': 0,
            'writes': 0,
            'errors': 0
        }
    
    async def connect(self):
        """Connect to Redis and PostgreSQL"""
        try:
            # Connect to Redis
            self.redis = aioredis.from_url(
                self.redis_url,
                encoding="utf-8",
                decode_responses=True,
                max_connections=10
            )
            await self.redis.ping()
            logger.info("✅ Redis cache connected")
            
        except Exception as e:
            logger.error(f"❌ Redis connection failed: {e}")
            self.redis = None
        
        try:
            # Connect to PostgreSQL
            self.postgres = await asyncpg.connect(self.postgres_url)
            logger.info("✅ PostgreSQL cache connected")
            
            # Ensure market_data table exists
            await self._ensure_tables()
            
        except Exception as e:
            logger.error(f"❌ PostgreSQL connection failed: {e}")
            self.postgres = None
    
    async def disconnect(self):
        """Disconnect from cache backends"""
        if self.redis:
            await self.redis.close()
            self.redis = None
        
        if self.postgres:
            await self.postgres.close()
            self.postgres = None
            
        self.memory_cache.clear()
        self.memory_ttl.clear()
        
        logger.info("✅ Data cache disconnected")
    
    async def get(self, symbol: str, max_age_seconds: float = 60) -> Optional[MarketDataPoint]:
        """
        Get market data with intelligent cache hierarchy
        """
        cache_key = f"market:{symbol}"
        current_time = time.time()
        
        # L1: Memory cache (fastest)
        if cache_key in self.memory_cache:
            if current_time - self.memory_ttl.get(cache_key, 0) < max_age_seconds:
                self.stats['hits_memory'] += 1
                return self.memory_cache[cache_key]
            else:
                # Expired from memory
                del self.memory_cache[cache_key]
                del self.memory_ttl[cache_key]
        
        # L2: Redis cache (fast)
        if self.redis:
            try:
                cached_json = await self.redis.get(cache_key)
                if cached_json:
                    cached_data = json.loads(cached_json)
                    
                    # Check if data is fresh enough
                    if current_time - cached_data.get('timestamp', 0) < max_age_seconds:
                        # Reconstruct MarketDataPoint
                        market_data = MarketDataPoint(**cached_data)
                        
                        # Store in memory cache for next access
                        self._set_memory(cache_key, market_data)
                        
                        self.stats['hits_redis'] += 1
                        return market_data
                        
            except Exception as e:
                logger.warning(f"⚠️ Redis get error: {e}")
                self.stats['errors'] += 1
        
        # L3: PostgreSQL (persistence)
        if self.postgres:
            try:
                query = """
                    SELECT symbol, price, volume_24h, change_24h, high_24h, low_24h,
                           bid, ask, source, quality_score, 
                           EXTRACT(EPOCH FROM ts) as timestamp
                    FROM market_data 
                    WHERE symbol = $1 
                    AND ts > NOW() - INTERVAL '%s seconds'
                    ORDER BY ts DESC 
                    LIMIT 1
                """
                
                row = await self.postgres.fetchrow(query % max_age_seconds, symbol)
                if row:
                    market_data = MarketDataPoint(
                        symbol=row['symbol'],
                        timestamp=float(row['timestamp']),
                        price=float(row['price']),
                        volume_24h=float(row['volume_24h']) if row['volume_24h'] else 0.0,
                        change_24h=float(row['change_24h']) if row['change_24h'] else 0.0,
                        high_24h=float(row['high_24h']) if row['high_24h'] else 0.0,
                        low_24h=float(row['low_24h']) if row['low_24h'] else 0.0,
                        bid=float(row['bid']) if row['bid'] else 0.0,
                        ask=float(row['ask']) if row['ask'] else 0.0,
                        source=row['source'],
                        quality_score=float(row['quality_score']) if row['quality_score'] else 1.0
                    )
                    
                    # Cache in Redis and memory
                    await self._set_redis(cache_key, market_data)
                    self._set_memory(cache_key, market_data)
                    
                    self.stats['hits_postgres'] += 1
                    return market_data
                    
            except Exception as e:
                logger.warning(f"⚠️ PostgreSQL get error: {e}")
                self.stats['errors'] += 1
        
        # Cache miss
        self.stats['misses'] += 1
        return None
    
    async def set(self, symbol: str, data: MarketDataPoint, ttl_seconds: int = 300):
        """
        Set data in all cache layers
        """
        cache_key = f"market:{symbol}"
        
        try:
            # L1: Memory cache
            self._set_memory(cache_key, data)
            
            # L2: Redis cache
            await self._set_redis(cache_key, data, ttl_seconds)
            
            # L3: PostgreSQL (persistence)
            await self._set_postgres(data)
            
            self.stats['writes'] += 1
            
        except Exception as e:
            logger.error(f"❌ Cache set error for {symbol}: {e}")
            self.stats['errors'] += 1
    
    def _set_memory(self, cache_key: str, data: MarketDataPoint):
        """Set data in memory cache with size limits"""
        # Evict old items if at capacity
        if len(self.memory_cache) >= self.max_memory_items:
            # Remove oldest item
            oldest_key = min(self.memory_ttl.keys(), key=lambda k: self.memory_ttl[k])
            del self.memory_cache[oldest_key]
            del self.memory_ttl[oldest_key]
        
        self.memory_cache[cache_key] = data
        self.memory_ttl[cache_key] = time.time()
    
    async def _set_redis(self, cache_key: str, data: MarketDataPoint, ttl_seconds: int = 300):
        """Set data in Redis cache"""
        if not self.redis:
            return
            
        try:
            # Convert MarketDataPoint to JSON
            data_dict = {
                'symbol': data.symbol,
                'timestamp': data.timestamp,
                'price': data.price,
                'volume_24h': data.volume_24h,
                'change_24h': data.change_24h,
                'high_24h': data.high_24h,
                'low_24h': data.low_24h,
                'bid': data.bid,
                'ask': data.ask,
                'source': data.source,
                'quality_score': data.quality_score
            }
            
            await self.redis.setex(
                cache_key,
                ttl_seconds,
                json.dumps(data_dict)
            )
            
        except Exception as e:
            logger.warning(f"⚠️ Redis set error: {e}")
    
    async def _set_postgres(self, data: MarketDataPoint):
        """Persist data to PostgreSQL"""
        if not self.postgres:
            return
            
        try:
            query = """
                INSERT INTO market_data 
                (symbol, price, volume_24h, change_24h, high_24h, low_24h, 
                 bid, ask, source, quality_score, ts)
                VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, TO_TIMESTAMP($11))
            """
            
            await self.postgres.execute(
                query,
                data.symbol, data.price, data.volume_24h, data.change_24h,
                data.high_24h, data.low_24h, data.bid, data.ask,
                data.source, data.quality_score, data.timestamp
            )
            
        except Exception as e:
            logger.warning(f"⚠️ PostgreSQL insert error: {e}")
    
    async def _ensure_tables(self):
        """Ensure required tables exist in PostgreSQL"""
        if not self.postgres:
            return
            
        try:
            # Create market_data table if not exists (matches sql/init.sql)
            await self.postgres.execute("""
                CREATE TABLE IF NOT EXISTS market_data (
                    id SERIAL PRIMARY KEY,
                    symbol VARCHAR(32) NOT NULL,
                    price NUMERIC(18,8) NOT NULL,
                    volume_24h NUMERIC(24,8),
                    change_24h NUMERIC(8,4),
                    high_24h NUMERIC(18,8),
                    low_24h NUMERIC(18,8),
                    bid NUMERIC(18,8),
                    ask NUMERIC(18,8),
                    source VARCHAR(32),
                    quality_score NUMERIC(6,4),
                    ts TIMESTAMP WITH TIME ZONE DEFAULT NOW()
                )
            """)
            
            # Create index for fast lookups
            await self.postgres.execute("""
                CREATE INDEX IF NOT EXISTS idx_market_data_symbol_ts 
                ON market_data(symbol, ts DESC)
            """)
            
            logger.info("✅ PostgreSQL tables ensured")
            
        except Exception as e:
            logger.error(f"❌ PostgreSQL table creation error: {e}")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache performance statistics"""
        total_requests = sum([
            self.stats['hits_memory'],
            self.stats['hits_redis'], 
            self.stats['hits_postgres'],
            self.stats['misses']
        ])
        
        return {
            'total_requests': total_requests,
            'hit_rate': (total_requests - self.stats['misses']) / max(1, total_requests),
            'memory_hit_rate': self.stats['hits_memory'] / max(1, total_requests),
            'redis_hit_rate': self.stats['hits_redis'] / max(1, total_requests),
            'postgres_hit_rate': self.stats['hits_postgres'] / max(1, total_requests),
            'miss_rate': self.stats['misses'] / max(1, total_requests),
            'error_rate': self.stats['errors'] / max(1, self.stats['writes']),
            'memory_items': len(self.memory_cache),
            **self.stats
        }

class CacheManager:
    """
    Cache manager orchestrating multiple cache instances
    Handles cache warming, cleanup, and health monitoring
    """
    
    def __init__(self, cache: DataCache):
        """Initialize cache manager"""
        self.cache = cache
        self.warming_symbols: List[str] = []
        self.cleanup_task: Optional[asyncio.Task] = None
        self.running = False
    
    async def start(self, symbols: List[str]):
        """Start cache manager with symbol warming"""
        await self.cache.connect()
        
        self.warming_symbols = symbols
        self.running = True
        
        # Start background tasks
        self.cleanup_task = asyncio.create_task(self._cleanup_loop())
        
        logger.info(f"🔥 Cache manager started with {len(symbols)} symbols")
    
    async def stop(self):
        """Stop cache manager"""
        self.running = False
        
        if self.cleanup_task:
            self.cleanup_task.cancel()
            try:
                await self.cleanup_task
            except asyncio.CancelledError:
                pass
        
        await self.cache.disconnect()
        logger.info("✅ Cache manager stopped")
    
    async def _cleanup_loop(self):
        """Background cleanup of stale cache entries"""
        while self.running:
            try:
                # Clean memory cache
                current_time = time.time()
                expired_keys = [
                    key for key, ttl in self.cache.memory_ttl.items()
                    if current_time - ttl > 300  # 5 minutes
                ]
                
                for key in expired_keys:
                    del self.cache.memory_cache[key]
                    del self.cache.memory_ttl[key]
                
                if expired_keys:
                    logger.debug(f"🧹 Cleaned {len(expired_keys)} stale cache entries")
                
                # Wait 60 seconds between cleanup runs
                await asyncio.sleep(60)
                
            except Exception as e:
                logger.error(f"❌ Cache cleanup error: {e}")
                await asyncio.sleep(60)  # Continue trying
    
    def get_health(self) -> Dict[str, Any]:
        """Get cache health status"""
        stats = self.cache.get_stats()
        
        return {
            'status': 'healthy' if stats['error_rate'] < 0.1 else 'degraded',
            'redis_connected': self.cache.redis is not None,
            'postgres_connected': self.cache.postgres is not None,
            'memory_usage_percent': (len(self.cache.memory_cache) / self.cache.max_memory_items) * 100,
            **stats
        }
