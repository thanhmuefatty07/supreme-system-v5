"""
Supreme System V5 - Advanced Multi-Source Data Aggregator
Handles free data sources with circuit breakers, quality scoring, and failover
"""

import asyncio
import aiohttp
import websockets
import json
import time
import statistics
import re
from typing import Dict, List, Optional, Callable, Any
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime, timedelta
import redis.asyncio as redis
import psutil

class DataSourceStatus(Enum):
    ACTIVE = "active"
    DEGRADED = "degraded"
    FAILED = "failed"
    CIRCUIT_BREAKER = "circuit_breaker"

class QualityScore(Enum):
    EXCELLENT = 1.0
    GOOD = 0.8
    FAIR = 0.6
    POOR = 0.3
    UNUSABLE = 0.0

@dataclass
class QualityMetrics:
    latency_ms: float
    freshness_seconds: float
    completeness_score: float
    consistency_score: float
    reliability_score: float
    overall_score: float
    timestamp: float = field(default_factory=time.time)
    data_points: int = 0
    error_count: int = 0

@dataclass
class MarketData:
    symbol: str
    price: float
    volume: float
    change_24h: float
    timestamp: float
    source: str
    quality_metrics: QualityMetrics
    raw_data: Dict = field(default_factory=dict)
    bid: float = 0.0
    ask: float = 0.0
    high_24h: float = 0.0
    low_24h: float = 0.0

class CircuitBreaker:
    """Circuit breaker pattern for data sources"""

    def __init__(self, failure_threshold: int = 5, recovery_timeout: int = 300,
                 success_threshold: int = 3):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.success_threshold = success_threshold
        self.failure_count = 0
        self.success_count = 0
        self.last_failure_time = 0
        self.state = DataSourceStatus.ACTIVE

    def call_succeeded(self):
        """Record successful call"""
        self.failure_count = 0
        self.success_count += 1

        if self.state == DataSourceStatus.CIRCUIT_BREAKER:
            if self.success_count >= self.success_threshold:
                self.state = DataSourceStatus.ACTIVE
                self.success_count = 0
                print(f"✅ Circuit breaker reset - service recovered")

    def call_failed(self):
        """Record failed call"""
        self.failure_count += 1
        self.success_count = 0
        self.last_failure_time = time.time()

        if self.failure_count >= self.failure_threshold:
            self.state = DataSourceStatus.CIRCUIT_BREAKER
            print(f"🚨 Circuit breaker activated - {self.failure_count} failures")

    def can_execute(self) -> bool:
        """Check if calls are allowed"""
        if self.state == DataSourceStatus.CIRCUIT_BREAKER:
            # Check if recovery timeout has passed
            if time.time() - self.last_failure_time > self.recovery_timeout:
                self.state = DataSourceStatus.ACTIVE
                self.failure_count = 0
                print(f"🔄 Circuit breaker attempting recovery")
                return True
            return False
        return True

    def get_status(self) -> Dict:
        """Get circuit breaker status"""
        return {
            'state': self.state.value,
            'failure_count': self.failure_count,
            'success_count': self.success_count,
            'last_failure': self.last_failure_time
        }

class RateLimiter:
    """Token bucket rate limiter"""

    def __init__(self, rate_per_minute: int = 60):
        self.rate_per_minute = rate_per_minute
        self.tokens = rate_per_minute
        self.last_refill = time.time()
        self.max_tokens = rate_per_minute

    def acquire(self, tokens: int = 1) -> bool:
        """Acquire tokens from bucket"""
        now = time.time()
        time_passed = now - self.last_refill
        refill_amount = time_passed * (self.rate_per_minute / 60)  # tokens per second

        self.tokens = min(self.max_tokens, self.tokens + refill_amount)
        self.last_refill = now

        if self.tokens >= tokens:
            self.tokens -= tokens
            return True

        return False

    def get_remaining_tokens(self) -> float:
        """Get remaining tokens"""
        now = time.time()
        time_passed = now - self.last_refill
        refill_amount = time_passed * (self.rate_per_minute / 60)
        current_tokens = min(self.max_tokens, self.tokens + refill_amount)
        return current_tokens

class AdvancedDataAggregator:
    """
    Advanced multi-source data aggregator with enterprise-grade reliability
    Features: Circuit breakers, quality scoring, rate limiting, failover
    """

    def __init__(self):
        self.sources = {}
        self.circuit_breakers = {}
        self.rate_limiters = {}
        self.quality_history = {}
        self.redis_client = None
        self.callbacks: List[Callable] = []
        self.running = False
        self.stats = {
            'total_requests': 0,
            'successful_requests': 0,
            'failed_requests': 0,
            'data_points_processed': 0,
            'quality_score_avg': 0.0
        }

        # Free data source configurations
        self.source_configs = {
            'binance_ws': {
                'url': 'wss://stream.binance.com:9443/ws',
                'type': 'websocket',
                'priority': 1,
                'rate_limit': None,  # Unlimited for WebSocket
                'parser': self._parse_binance_ws
            },
            'coingecko': {
                'url': 'https://api.coingecko.com/api/v3',
                'type': 'rest',
                'priority': 2,
                'rate_limit': 50,  # 50 calls/minute
                'parser': self._parse_coingecko
            },
            'okx': {
                'url': 'https://www.okx.com/api/v5',
                'type': 'rest',
                'priority': 1,
                'rate_limit': 20,  # Conservative limit
                'parser': self._parse_okx
            },
            'cryptocompare': {
                'url': 'https://min-api.cryptocompare.com/data',
                'type': 'rest',
                'priority': 3,
                'rate_limit': 100,  # 100 calls/minute
                'parser': self._parse_cryptocompare
            },
            'coinmarketcap': {
                'url': 'https://pro-api.coinmarketcap.com/v1',
                'type': 'rest',
                'priority': 4,
                'rate_limit': 10,  # Very conservative for free tier
                'parser': self._parse_coinmarketcap,
                'api_key_required': True
            }
        }

    async def initialize(self, redis_url: str = "redis://localhost:6379/0"):
        """Initialize aggregator with Redis connection"""
        try:
            self.redis_client = redis.Redis.from_url(redis_url, decode_responses=True)
            await self.redis_client.ping()
            print("✅ Redis connection established")

            # Initialize circuit breakers and rate limiters
            for source_name, config in self.source_configs.items():
                self.circuit_breakers[source_name] = CircuitBreaker()
                if config.get('rate_limit'):
                    self.rate_limiters[source_name] = RateLimiter(config['rate_limit'])

            print(f"✅ Initialized {len(self.source_configs)} data sources")

        except Exception as e:
            print(f"⚠️ Redis connection failed: {e}")
            print("🔄 Operating without Redis caching")

    async def start_aggregation(self, symbols: List[str]):
        """Start multi-source data aggregation"""
        self.running = True
        print(f"🚀 Starting data aggregation for {len(symbols)} symbols")

        # Start individual source tasks
        tasks = []

        # WebSocket sources
        if 'binance_ws' in self.source_configs:
            task = asyncio.create_task(
                self._websocket_stream('binance_ws', symbols)
            )
            tasks.append(task)

        # REST API sources
        for source_name in ['coingecko', 'okx', 'cryptocompare', 'coinmarketcap']:
            if source_name in self.source_configs:
                task = asyncio.create_task(
                    self._rest_api_poller(source_name, symbols)
                )
                tasks.append(task)

        # Quality monitoring
        quality_task = asyncio.create_task(self._quality_monitor())
        tasks.append(task)

        # Stats reporting
        stats_task = asyncio.create_task(self._stats_reporter())
        tasks.append(stats_task)

        # Wait for all tasks
        try:
            await asyncio.gather(*tasks, return_exceptions=True)
        except Exception as e:
            print(f"❌ Aggregation error: {e}")
        finally:
            self.running = False

    async def _websocket_stream(self, source_name: str, symbols: List[str]):
        """Handle WebSocket stream for real-time data"""
        config = self.source_configs[source_name]
        circuit_breaker = self.circuit_breakers[source_name]

        while self.running:
            if not circuit_breaker.can_execute():
                await asyncio.sleep(60)
                continue

            try:
                # Create stream configuration
                streams = []
                for symbol in symbols:
                    # Convert symbol format (BTC-USDT -> btcusdt)
                    ws_symbol = symbol.lower().replace('-', '')
                    streams.append(f"{ws_symbol}@ticker")

                stream_url = f"{config['url']}/stream?streams={'/'.join(streams)}"

                async with aiohttp.ClientSession() as session:
                    async with session.ws_connect(stream_url) as ws:
                        print(f"✅ {source_name} WebSocket connected")

                        circuit_breaker.call_succeeded()

                        async for msg in ws:
                            if msg.type == aiohttp.WSMsgType.TEXT:
                                try:
                                    data = json.loads(msg.data)
                                    await self._process_websocket_data(source_name, data)

                                except json.JSONDecodeError:
                                    continue
                                except Exception as e:
                                    print(f"⚠️ {source_name} data processing error: {e}")

                            elif msg.type == aiohttp.WSMsgType.ERROR:
                                print(f"❌ {source_name} WebSocket error: {msg}")
                                break

            except Exception as e:
                print(f"❌ {source_name} WebSocket error: {e}")
                circuit_breaker.call_failed()
                await asyncio.sleep(30)

    async def _rest_api_poller(self, source_name: str, symbols: List[str]):
        """Poll REST API sources"""
        config = self.source_configs[source_name]
        circuit_breaker = self.circuit_breakers[source_name]
        rate_limiter = self.rate_limiters.get(source_name)

        # Dynamic interval based on rate limits
        base_interval = 60  # 1 minute base
        if rate_limiter:
            calls_per_minute = config.get('rate_limit', 60)
            base_interval = max(10, 60 // calls_per_minute)  # At least 10 seconds

        while self.running:
            if not circuit_breaker.can_execute():
                await asyncio.sleep(60)
                continue

            # Rate limiting
            if rate_limiter and not rate_limiter.acquire():
                await asyncio.sleep(5)  # Wait for tokens
                continue

            try:
                async with aiohttp.ClientSession() as session:
                    start_time = time.time()

                    # Batch requests for efficiency
                    batch_size = min(len(symbols), 10)  # Max 10 symbols per batch

                    for i in range(0, len(symbols), batch_size):
                        batch_symbols = symbols[i:i+batch_size]

                        data = await self._fetch_batch_data(session, source_name, batch_symbols)

                        # Process batch results
                        for symbol_data in data:
                            quality = await self._assess_data_quality(
                                source_name, symbol_data,
                                response_time=(time.time() - start_time) * 1000
                            )

                            symbol_data.quality_metrics = quality
                            await self._cache_and_broadcast(symbol_data)

                        self.stats['data_points_processed'] += len(data)

                    circuit_breaker.call_succeeded()
                    self.stats['successful_requests'] += 1

            except Exception as e:
                print(f"❌ {source_name} polling error: {e}")
                circuit_breaker.call_failed()
                self.stats['failed_requests'] += 1

            await asyncio.sleep(base_interval)

    async def _fetch_batch_data(self, session: aiohttp.ClientSession,
                               source_name: str, symbols: List[str]) -> List[MarketData]:
        """Fetch data for batch of symbols"""
        config = self.source_configs[source_name]
        parser = config['parser']

        try:
            # Call source-specific parser
            return await parser(session, symbols)
        except Exception as e:
            print(f"⚠️ {source_name} batch fetch error: {e}")
            return []

    async def _process_websocket_data(self, source_name: str, raw_data: Dict):
        """Process WebSocket data"""
        try:
            config = self.source_configs[source_name]
            parser = config['parser']

            # Parse WebSocket message
            market_data = await parser(None, [], raw_data)

            if market_data:
                quality = await self._assess_data_quality(source_name, market_data)
                market_data.quality_metrics = quality

                await self._cache_and_broadcast(market_data)
                self.stats['data_points_processed'] += 1

        except Exception as e:
            print(f"⚠️ {source_name} WebSocket processing error: {e}")

    async def _assess_data_quality(self, source: str, data: MarketData,
                                 response_time: float = 0) -> QualityMetrics:
        """Comprehensive data quality assessment"""
        self.stats['total_requests'] += 1

        # Latency score
        latency_ms = response_time or 50  # Default 50ms for WebSocket
        if latency_ms < 100:
            latency_score = 1.0
        elif latency_ms < 500:
            latency_score = 0.8
        elif latency_ms < 1000:
            latency_score = 0.6
        else:
            latency_score = 0.3

        # Freshness score
        age_seconds = time.time() - data.timestamp
        if age_seconds < 5:
            freshness_score = 1.0
        elif age_seconds < 30:
            freshness_score = 0.8
        elif age_seconds < 60:
            freshness_score = 0.6
        else:
            freshness_score = 0.3

        # Completeness score
        required_fields = ['price', 'volume', 'change_24h']
        present_fields = sum(1 for field in required_fields
                           if hasattr(data, field) and getattr(data, field) is not None)
        completeness_score = present_fields / len(required_fields)

        # Consistency score
        consistency_score = await self._calculate_consistency_score(data)

        # Reliability score
        cb = self.circuit_breakers.get(source)
        if cb and cb.state == DataSourceStatus.ACTIVE:
            reliability_score = 1.0
        elif cb and cb.state == DataSourceStatus.DEGRADED:
            reliability_score = 0.7
        else:
            reliability_score = 0.3

        # Overall score
        overall_score = (
            latency_score * 0.2 +
            freshness_score * 0.3 +
            completeness_score * 0.2 +
            consistency_score * 0.2 +
            reliability_score * 0.1
        )

        # Update rolling average
        if self.stats['total_requests'] > 0:
            self.stats['quality_score_avg'] = (
                self.stats['quality_score_avg'] * (self.stats['total_requests'] - 1) +
                overall_score
            ) / self.stats['total_requests']

        return QualityMetrics(
            latency_ms=latency_ms,
            freshness_seconds=age_seconds,
            completeness_score=completeness_score,
            consistency_score=consistency_score,
            reliability_score=reliability_score,
            overall_score=overall_score
        )

    async def _calculate_consistency_score(self, data: MarketData) -> float:
        """Calculate price consistency with other sources"""
        if not self.redis_client:
            return 1.0

        try:
            pattern = f"market_data:{data.symbol}:*"
            keys = await self.redis_client.keys(pattern)

            other_prices = []
            for key in keys:
                if data.source not in key:
                    cached_data = await self.redis_client.get(key)
                    if cached_data:
                        try:
                            data_dict = json.loads(cached_data)
                            if time.time() - data_dict['timestamp'] < 300:  # 5 min
                                other_prices.append(data_dict['price'])
                        except:
                            continue

            if not other_prices:
                return 1.0

            avg_price = statistics.mean(other_prices)
            deviation = abs(data.price - avg_price) / avg_price if avg_price > 0 else 0

            if deviation <= 0.005:  # Within 0.5%
                return 1.0
            elif deviation <= 0.01:  # Within 1%
                return 0.8
            elif deviation <= 0.02:  # Within 2%
                return 0.6
            else:
                return 0.3

        except Exception as e:
            return 0.5

    async def _cache_and_broadcast(self, data: MarketData):
        """Cache data and broadcast to subscribers"""
        # Cache in Redis
        if self.redis_client:
            try:
                cache_key = f"market_data:{data.symbol}:{data.source}"
                cache_data = {
                    'symbol': data.symbol,
                    'price': data.price,
                    'volume': data.volume,
                    'change_24h': data.change_24h,
                    'timestamp': data.timestamp,
                    'source': data.source,
                    'quality_score': data.quality_metrics.overall_score,
                    'bid': data.bid,
                    'ask': data.ask,
                    'high_24h': data.high_24h,
                    'low_24h': data.low_24h
                }

                await self.redis_client.setex(cache_key, 300, json.dumps(cache_data))

            except Exception as e:
                print(f"⚠️ Cache error: {e}")

        # Broadcast to callbacks
        for callback in self.callbacks:
            try:
                await callback(data)
            except Exception as e:
                print(f"⚠️ Callback error: {e}")

    def add_callback(self, callback: Callable):
        """Add data callback"""
        self.callbacks.append(callback)

    async def get_best_data(self, symbol: str, min_quality: float = 0.6) -> Optional[MarketData]:
        """Get best quality data for symbol"""
        if not self.redis_client:
            return None

        try:
            pattern = f"market_data:{symbol}:*"
            keys = await self.redis_client.keys(pattern)

            best_data = None
            best_score = min_quality

            for key in keys:
                cached_data = await self.redis_client.get(key)
                if cached_data:
                    try:
                        data_dict = json.loads(cached_data)

                        # Check if data is recent (within 2 minutes)
                        if time.time() - data_dict['timestamp'] < 120:
                            score = data_dict.get('quality_score', 0)
                            if score > best_score:
                                best_score = score
                                best_data = MarketData(
                                    symbol=data_dict['symbol'],
                                    price=data_dict['price'],
                                    volume=data_dict['volume'],
                                    change_24h=data_dict['change_24h'],
                                    timestamp=data_dict['timestamp'],
                                    source=data_dict['source'],
                                    bid=data_dict.get('bid', 0),
                                    ask=data_dict.get('ask', 0),
                                    high_24h=data_dict.get('high_24h', 0),
                                    low_24h=data_dict.get('low_24h', 0),
                                    quality_metrics=QualityMetrics(
                                        latency_ms=0, freshness_seconds=0,
                                        completeness_score=1.0, consistency_score=1.0,
                                        reliability_score=1.0, overall_score=score
                                    )
                                )
                    except:
                        continue

            return best_data

        except Exception as e:
            print(f"❌ Get best data error: {e}")
            return None

    async def _quality_monitor(self):
        """Monitor and report data quality metrics"""
        while self.running:
            try:
                # Generate quality report every 5 minutes
                await asyncio.sleep(300)

                if self.stats['total_requests'] > 0:
                    success_rate = self.stats['successful_requests'] / self.stats['total_requests']
                    quality_avg = self.stats['quality_score_avg']

                    print(f"📊 Quality Report:")
                    print(f"   Success Rate: {success_rate:.1%}")
                    print(f"   Avg Quality: {quality_avg:.3f}")
                    print(f"   Data Points: {self.stats['data_points_processed']}")
                    print(f"   Active Sources: {len([cb for cb in self.circuit_breakers.values() if cb.state == DataSourceStatus.ACTIVE])}")

            except Exception as e:
                print(f"⚠️ Quality monitoring error: {e}")

    async def _stats_reporter(self):
        """Report system statistics"""
        while self.running:
            try:
                await asyncio.sleep(600)  # Every 10 minutes

                memory = psutil.virtual_memory()
                print(f"💻 System Stats:")
                print(f"   Memory: {memory.percent:.1f}% used")
                print(f"   CPU: {psutil.cpu_percent():.1f}% used")
                print(f"   Network: {psutil.net_io_counters().bytes_sent // 1024}KB sent")

            except Exception as e:
                print(f"⚠️ Stats reporting error: {e}")

    # Source-specific parsers

    async def _parse_binance_ws(self, session, symbols, ws_data=None):
        """Parse Binance WebSocket data"""
        if ws_data and 'stream' in ws_data and 'data' in ws_data:
            ticker_data = ws_data['data']
            symbol = ticker_data['s'].replace('USDT', '-USDT')

            return MarketData(
                symbol=symbol,
                price=float(ticker_data['c']),
                volume=float(ticker_data['v']),
                change_24h=float(ticker_data['P']),
                timestamp=time.time(),
                source='binance_ws',
                bid=float(ticker_data.get('b', 0)),
                ask=float(ticker_data.get('a', 0)),
                high_24h=float(ticker_data.get('h', 0)),
                low_24h=float(ticker_data.get('l', 0))
            )
        return []

    async def _parse_coingecko(self, session, symbols):
        """Parse CoinGecko API data"""
        results = []

        for symbol in symbols:
            try:
                coin_id = self._symbol_to_coingecko_id(symbol)
                url = f"{self.source_configs['coingecko']['url']}/simple/price"
                params = {
                    'ids': coin_id,
                    'vs_currencies': 'usd',
                    'include_24hr_change': 'true',
                    'include_24hr_vol': 'true'
                }

                async with session.get(url, params=params) as response:
                    if response.status == 200:
                        data = await response.json()
                        if coin_id in data:
                            coin_data = data[coin_id]
                            results.append(MarketData(
                                symbol=symbol,
                                price=coin_data['usd'],
                                volume=coin_data.get('usd_24h_vol', 0),
                                change_24h=coin_data.get('usd_24h_change', 0),
                                timestamp=time.time(),
                                source='coingecko'
                            ))

            except Exception as e:
                print(f"⚠️ CoinGecko parsing error for {symbol}: {e}")

        return results

    async def _parse_okx(self, session, symbols):
        """Parse OKX API data"""
        results = []

        for symbol in symbols:
            try:
                okx_symbol = symbol.replace('-', '-')
                url = f"{self.source_configs['okx']['url']}/market/ticker"
                params = {'instId': okx_symbol}

                async with session.get(url, params=params) as response:
                    if response.status == 200:
                        data = await response.json()
                        if data.get('code') == '0' and data.get('data'):
                            ticker = data['data'][0]
                            results.append(MarketData(
                                symbol=symbol,
                                price=float(ticker['last']),
                                volume=float(ticker.get('vol24h', 0)),
                                change_24h=float(ticker.get('change24h', 0)) * 100,
                                timestamp=time.time(),
                                source='okx'
                            ))

            except Exception as e:
                print(f"⚠️ OKX parsing error for {symbol}: {e}")

        return results

    async def _parse_cryptocompare(self, session, symbols):
        """Parse CryptoCompare API data"""
        results = []

        for symbol in symbols:
            try:
                base_symbol = symbol.split('-')[0]
                url = f"{self.source_configs['cryptocompare']['url']}/price"
                params = {
                    'fsym': base_symbol,
                    'tsyms': 'USD',
                    'api_key': os.getenv('CRYPTOCOMPARE_API_KEY', '')
                }

                async with session.get(url, params=params) as response:
                    if response.status == 200:
                        data = await response.json()
                        if 'USD' in data:
                            # Note: CryptoCompare free tier has limited data
                            results.append(MarketData(
                                symbol=symbol,
                                price=data['USD'],
                                volume=0,  # Limited in free tier
                                change_24h=0,  # Limited in free tier
                                timestamp=time.time(),
                                source='cryptocompare'
                            ))

            except Exception as e:
                print(f"⚠️ CryptoCompare parsing error for {symbol}: {e}")

        return results

    async def _parse_coinmarketcap(self, session, symbols):
        """Parse CoinMarketCap API data (requires API key)"""
        results = []

        api_key = os.getenv('COINMARKETCAP_API_KEY')
        if not api_key:
            return results

        try:
            # CMC requires symbol mapping
            symbol_ids = []
            for symbol in symbols:
                coin_id = self._symbol_to_cmc_id(symbol)
                if coin_id:
                    symbol_ids.append(coin_id)

            if symbol_ids:
                url = f"{self.source_configs['coinmarketcap']['url']}/cryptocurrency/quotes/latest"
                headers = {'X-CMC_PRO_API_KEY': api_key}
                params = {'id': ','.join(symbol_ids)}

                async with session.get(url, headers=headers, params=params) as response:
                    if response.status == 200:
                        data = await response.json()
                        if data.get('data'):
                            for coin_id, coin_data in data['data'].items():
                                quote = coin_data['quote']['USD']
                                symbol = self._cmc_id_to_symbol(coin_id)
                                if symbol:
                                    results.append(MarketData(
                                        symbol=symbol,
                                        price=quote['price'],
                                        volume=quote.get('volume_24h', 0),
                                        change_24h=quote.get('percent_change_24h', 0),
                                        timestamp=time.time(),
                                        source='coinmarketcap'
                                    ))

        except Exception as e:
            print(f"⚠️ CoinMarketCap parsing error: {e}")

        return results

    def _symbol_to_coingecko_id(self, symbol: str) -> str:
        """Convert symbol to CoinGecko ID"""
        mapping = {
            'BTC-USDT': 'bitcoin',
            'ETH-USDT': 'ethereum',
            'BNB-USDT': 'binancecoin',
            'ADA-USDT': 'cardano',
            'SOL-USDT': 'solana'
        }
        return mapping.get(symbol, symbol.lower().split('-')[0])

    def _symbol_to_cmc_id(self, symbol: str) -> str:
        """Convert symbol to CoinMarketCap ID"""
        # This would require a mapping table
        return None

    def _cmc_id_to_symbol(self, cmc_id: str) -> str:
        """Convert CoinMarketCap ID back to symbol"""
        # Reverse mapping
        return None

    async def stop(self):
        """Stop aggregation"""
        self.running = False
        if self.redis_client:
            await self.redis_client.close()

    def get_stats(self) -> Dict:
        """Get aggregator statistics"""
        return self.stats.copy()

# Global aggregator instance
data_aggregator = AdvancedDataAggregator()
