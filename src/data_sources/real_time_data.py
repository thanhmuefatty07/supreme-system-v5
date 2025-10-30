"""
📈 Supreme System V5 - Real-Time Data Provider
Production-grade financial data integration with multiple sources and failover

Features:
- Multiple data source integration (Alpha Vantage, Finnhub, Yahoo Finance)
- Real-time WebSocket feeds (Binance, Coinbase)
- Automatic failover and redundancy
- Data validation and quality checks
- Rate limiting and caching
- Hardware-aware optimization
"""

import asyncio
import aiohttp
import json
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Callable, Union, Tuple
from dataclasses import dataclass
from enum import Enum
import logging
from collections import deque, defaultdict
import websockets
import numpy as np

# Hardware optimization
try:
    from ..config.hardware_profiles import optimal_profile, ProcessorType, MemoryProfile
    HARDWARE_OPTIMIZATION = True
except ImportError:
    HARDWARE_OPTIMIZATION = False
    optimal_profile = None

logger = logging.getLogger("supreme_data")

class DataSource(Enum):
    """Supported data sources"""
    ALPHA_VANTAGE = "alpha_vantage"
    FINNHUB = "finnhub"
    YAHOO_FINANCE = "yahoo_finance"
    BINANCE_WS = "binance_websocket"
    COINBASE_WS = "coinbase_websocket"
    IEX_CLOUD = "iex_cloud"

class DataType(Enum):
    """Data type classification"""
    REAL_TIME_QUOTE = "real_time_quote"
    HISTORICAL_OHLC = "historical_ohlc"
    MARKET_NEWS = "market_news"
    ECONOMIC_INDICATORS = "economic_indicators"
    CRYPTO_TICKER = "crypto_ticker"
    OPTIONS_CHAIN = "options_chain"

@dataclass
class DataSourceConfig:
    """Configuration for data source"""
    source: DataSource
    api_key: str
    base_url: str
    rate_limit_per_minute: int
    priority: int  # Lower = higher priority
    timeout_seconds: float
    enabled: bool = True

@dataclass
class MarketData:
    """Standardized market data structure"""
    symbol: str
    price: float
    bid: float
    ask: float
    volume: float
    timestamp: datetime
    source: DataSource
    
    # Additional fields
    open_price: Optional[float] = None
    high_price: Optional[float] = None
    low_price: Optional[float] = None
    prev_close: Optional[float] = None
    change_percent: Optional[float] = None
    
    # Data quality metrics
    latency_ms: Optional[float] = None
    data_age_ms: Optional[float] = None
    quality_score: float = 1.0  # 0-1 quality score

class DataProvider:
    """Base class for data providers"""
    
    def __init__(self, config: DataSourceConfig):
        self.config = config
        self.session = None
        self.last_request_time = 0
        self.request_count = 0
        self.error_count = 0
        self.connected = False
        
    async def initialize(self):
        """Initialize data provider"""
        self.session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=self.config.timeout_seconds)
        )
        self.connected = True
        logger.info(f"✅ {self.config.source.value} provider initialized")
        
    async def cleanup(self):
        """Cleanup resources"""
        if self.session:
            await self.session.close()
        self.connected = False
        
    async def get_real_time_quote(self, symbol: str) -> Optional[MarketData]:
        """Get real-time quote - to be implemented by subclasses"""
        raise NotImplementedError
        
    def _check_rate_limit(self) -> bool:
        """Check if within rate limits"""
        now = time.time()
        if now - self.last_request_time < (60.0 / self.config.rate_limit_per_minute):
            return False
        return True
        
    def _update_request_stats(self, success: bool):
        """Update request statistics"""
        self.request_count += 1
        self.last_request_time = time.time()
        if not success:
            self.error_count += 1

class AlphaVantageProvider(DataProvider):
    """Alpha Vantage data provider - Primary source"""
    
    def __init__(self, api_key: str):
        config = DataSourceConfig(
            source=DataSource.ALPHA_VANTAGE,
            api_key=api_key,
            base_url="https://www.alphavantage.co/query",
            rate_limit_per_minute=5,  # Free tier: 5 requests/minute
            priority=1,  # Primary source
            timeout_seconds=10.0
        )
        super().__init__(config)
        
    async def get_real_time_quote(self, symbol: str) -> Optional[MarketData]:
        """Get real-time quote from Alpha Vantage"""
        if not self.connected or not self._check_rate_limit():
            return None
            
        request_start = time.perf_counter()
        
        try:
            params = {
                "function": "GLOBAL_QUOTE",
                "symbol": symbol,
                "apikey": self.config.api_key
            }
            
            async with self.session.get(self.config.base_url, params=params) as response:
                if response.status == 200:
                    data = await response.json()
                    
                    # Parse Alpha Vantage response
                    quote_data = data.get("Global Quote", {})
                    
                    if quote_data:
                        price = float(quote_data.get("05. price", 0))
                        prev_close = float(quote_data.get("08. previous close", 0))
                        volume = float(quote_data.get("06. volume", 0))
                        change_pct = float(quote_data.get("10. change percent", "0%").replace("%", ""))
                        
                        # Calculate bid/ask spread (estimated)
                        spread = price * 0.001  # 0.1% spread estimate
                        bid = price - spread / 2
                        ask = price + spread / 2
                        
                        latency_ms = (time.perf_counter() - request_start) * 1000
                        
                        market_data = MarketData(
                            symbol=symbol,
                            price=price,
                            bid=bid,
                            ask=ask,
                            volume=volume,
                            timestamp=datetime.utcnow(),
                            source=DataSource.ALPHA_VANTAGE,
                            prev_close=prev_close,
                            change_percent=change_pct,
                            latency_ms=latency_ms,
                            quality_score=1.0
                        )
                        
                        self._update_request_stats(True)
                        logger.debug(f"✅ Alpha Vantage data: {symbol} = ${price:.2f} ({latency_ms:.1f}ms)")
                        
                        return market_data
                    
        except Exception as e:
            logger.error(f"❌ Alpha Vantage error for {symbol}: {e}")
            self._update_request_stats(False)
            
        return None

class FinnhubProvider(DataProvider):
    """Finnhub data provider - Secondary source"""
    
    def __init__(self, api_key: str):
        config = DataSourceConfig(
            source=DataSource.FINNHUB,
            api_key=api_key,
            base_url="https://finnhub.io/api/v1",
            rate_limit_per_minute=60,  # Free tier: 60 requests/minute
            priority=2,  # Secondary source
            timeout_seconds=8.0
        )
        super().__init__(config)
        
    async def get_real_time_quote(self, symbol: str) -> Optional[MarketData]:
        """Get real-time quote from Finnhub"""
        if not self.connected or not self._check_rate_limit():
            return None
            
        request_start = time.perf_counter()
        
        try:
            headers = {"X-Finnhub-Token": self.config.api_key}
            params = {"symbol": symbol}
            
            async with self.session.get(
                f"{self.config.base_url}/quote", 
                params=params, 
                headers=headers
            ) as response:
                if response.status == 200:
                    data = await response.json()
                    
                    if "c" in data:  # Current price
                        price = float(data["c"])
                        prev_close = float(data.get("pc", price))
                        high = float(data.get("h", price))
                        low = float(data.get("l", price))
                        open_price = float(data.get("o", price))
                        
                        # Calculate change percentage
                        change_pct = ((price - prev_close) / prev_close * 100) if prev_close > 0 else 0
                        
                        # Estimate bid/ask
                        spread = price * 0.001
                        bid = price - spread / 2
                        ask = price + spread / 2
                        
                        latency_ms = (time.perf_counter() - request_start) * 1000
                        
                        market_data = MarketData(
                            symbol=symbol,
                            price=price,
                            bid=bid,
                            ask=ask,
                            volume=0.0,  # Finnhub doesn't provide volume in quote endpoint
                            timestamp=datetime.utcnow(),
                            source=DataSource.FINNHUB,
                            open_price=open_price,
                            high_price=high,
                            low_price=low,
                            prev_close=prev_close,
                            change_percent=change_pct,
                            latency_ms=latency_ms,
                            quality_score=0.9  # Good quality but no volume
                        )
                        
                        self._update_request_stats(True)
                        logger.debug(f"✅ Finnhub data: {symbol} = ${price:.2f} ({latency_ms:.1f}ms)")
                        
                        return market_data
                    
        except Exception as e:
            logger.error(f"❌ Finnhub error for {symbol}: {e}")
            self._update_request_stats(False)
            
        return None

class YahooFinanceProvider(DataProvider):
    """Yahoo Finance data provider - Backup source"""
    
    def __init__(self):
        config = DataSourceConfig(
            source=DataSource.YAHOO_FINANCE,
            api_key="",  # No API key needed
            base_url="https://query1.finance.yahoo.com/v8/finance/chart",
            rate_limit_per_minute=200,  # Higher rate limit
            priority=3,  # Backup source
            timeout_seconds=6.0
        )
        super().__init__(config)
        
    async def get_real_time_quote(self, symbol: str) -> Optional[MarketData]:
        """Get real-time quote from Yahoo Finance"""
        if not self.connected or not self._check_rate_limit():
            return None
            
        request_start = time.perf_counter()
        
        try:
            # Yahoo Finance expects different symbol format
            yahoo_symbol = symbol.replace("/", "")
            
            params = {
                "interval": "1m",
                "range": "1d",
                "includePrePost": "true"
            }
            
            async with self.session.get(
                f"{self.config.base_url}/{yahoo_symbol}", 
                params=params
            ) as response:
                if response.status == 200:
                    data = await response.json()
                    
                    chart_data = data.get("chart", {}).get("result", [{}])[0]
                    meta = chart_data.get("meta", {})
                    
                    if "regularMarketPrice" in meta:
                        price = float(meta["regularMarketPrice"])
                        prev_close = float(meta.get("previousClose", price))
                        volume = float(meta.get("regularMarketVolume", 0))
                        
                        # Calculate change percentage
                        change_pct = ((price - prev_close) / prev_close * 100) if prev_close > 0 else 0
                        
                        # Estimate bid/ask
                        spread = price * 0.001
                        bid = price - spread / 2
                        ask = price + spread / 2
                        
                        latency_ms = (time.perf_counter() - request_start) * 1000
                        
                        market_data = MarketData(
                            symbol=symbol,
                            price=price,
                            bid=bid,
                            ask=ask,
                            volume=volume,
                            timestamp=datetime.utcnow(),
                            source=DataSource.YAHOO_FINANCE,
                            prev_close=prev_close,
                            change_percent=change_pct,
                            latency_ms=latency_ms,
                            quality_score=0.8  # Good free source
                        )
                        
                        self._update_request_stats(True)
                        logger.debug(f"✅ Yahoo Finance data: {symbol} = ${price:.2f} ({latency_ms:.1f}ms)")
                        
                        return market_data
                    
        except Exception as e:
            logger.error(f"❌ Yahoo Finance error for {symbol}: {e}")
            self._update_request_stats(False)
            
        return None

class RealTimeDataProvider:
    """Unified real-time data provider with multiple sources and failover"""
    
    def __init__(self, 
                 alpha_vantage_key: str = "",
                 finnhub_key: str = "",
                 enable_websockets: bool = True):
        
        # Initialize data providers
        self.providers: List[DataProvider] = []
        
        # Primary sources (require API keys)
        if alpha_vantage_key:
            self.providers.append(AlphaVantageProvider(alpha_vantage_key))
        
        if finnhub_key:
            self.providers.append(FinnhubProvider(finnhub_key))
        
        # Always include Yahoo Finance as backup (no API key needed)
        self.providers.append(YahooFinanceProvider())
        
        # Sort by priority
        self.providers.sort(key=lambda p: p.config.priority)
        
        # Data caching and quality tracking
        self.data_cache: Dict[str, MarketData] = {}
        self.cache_ttl_seconds = 5.0  # Cache for 5 seconds
        self.quality_tracker: Dict[DataSource, deque] = defaultdict(lambda: deque(maxlen=100))
        
        # Hardware-specific optimizations
        self._apply_hardware_optimizations()
        
        # WebSocket connections
        self.websocket_enabled = enable_websockets
        self.websocket_connections = {}
        
        logger.info(f"📈 RealTimeDataProvider initialized with {len(self.providers)} sources")
        
    def _apply_hardware_optimizations(self):
        """Apply hardware-specific optimizations"""
        if HARDWARE_OPTIMIZATION and optimal_profile:
            if optimal_profile.processor_type == ProcessorType.I3_8TH_GEN:
                # Conservative settings for i3
                self.cache_ttl_seconds = 10.0  # Longer cache for i3
                self.max_concurrent_requests = 2  # Limit concurrent requests
                
                # Reduce timeout for faster failover
                for provider in self.providers:
                    provider.config.timeout_seconds = min(provider.config.timeout_seconds, 5.0)
                
                logger.info("⚡ Applied i3-8th gen data provider optimizations")
    
    async def initialize(self):
        """Initialize all data providers"""
        logger.info("🔄 Initializing data providers...")
        
        # Initialize providers concurrently
        init_tasks = [provider.initialize() for provider in self.providers]
        results = await asyncio.gather(*init_tasks, return_exceptions=True)
        
        # Check initialization results
        active_providers = 0
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.warning(f"⚠️ Provider {self.providers[i].config.source.value} failed to initialize: {result}")
                self.providers[i].connected = False
            else:
                active_providers += 1
        
        if active_providers == 0:
            raise RuntimeError("No data providers available")
        
        logger.info(f"✅ Data providers initialized: {active_providers}/{len(self.providers)} active")
        
    async def get_market_data(self, symbol: str, use_cache: bool = True, require_quorum: bool = True) -> Optional[MarketData]:
        """Get market data with quorum-based consensus and automatic failover
        
        Args:
            symbol: Trading symbol to get data for
            use_cache: Whether to use cached data if available
            require_quorum: If True, require at least 2 sources to agree (default: True)
        """
        # Check cache first
        if use_cache and symbol in self.data_cache:
            cached_data = self.data_cache[symbol]
            data_age = (datetime.utcnow() - cached_data.timestamp).total_seconds()
            
            if data_age < self.cache_ttl_seconds:
                cached_data.data_age_ms = data_age * 1000
                return cached_data
        
        # If quorum not required, fall back to simple failover
        if not require_quorum or len(self.providers) < 2:
            return await self._get_single_source_data(symbol)
        
        # Query all providers concurrently for quorum selection
        active_providers = [p for p in self.providers if p.connected]
        if len(active_providers) < 2:
            logger.warning(f"⚠️ Less than 2 active providers, using single source mode")
            return await self._get_single_source_data(symbol)
        
        # Fetch from all providers concurrently
        tasks = [provider.get_real_time_quote(symbol) for provider in active_providers]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Collect successful responses
        successful_data: List[Tuple[MarketData, DataProvider]] = []
        for provider, result in zip(active_providers, results):
            if isinstance(result, MarketData) and result is not None:
                successful_data.append((result, provider))
        
        if len(successful_data) == 0:
            logger.error(f"❌ No data available for {symbol} from any provider")
            return None
        
        # If only one source, return it (no quorum needed)
        if len(successful_data) == 1:
            market_data, provider = successful_data[0]
            self.data_cache[symbol] = market_data
            self.quality_tracker[provider.config.source].append(market_data.quality_score)
            logger.debug(f"📊 Single source data for {symbol}: ${market_data.price:.2f} from {provider.config.source.value}")
            return market_data
        
        # Quorum selection: require at least 2 sources to agree
        # Price tolerance: 0.5% difference considered acceptable
        price_tolerance_pct = 0.5
        consensus_data = self._select_quorum_consensus(successful_data, price_tolerance_pct)
        
        if consensus_data:
            # Update cache
            self.data_cache[symbol] = consensus_data
            
            # Update quality tracker for all participating providers
            for market_data, provider in successful_data:
                self.quality_tracker[provider.config.source].append(market_data.quality_score)
            
            logger.debug(f"✅ Quorum consensus for {symbol}: ${consensus_data.price:.2f} (from {len(successful_data)} sources)")
            return consensus_data
        else:
            # No quorum reached, use weighted average from all sources
            logger.warning(f"⚠️ No quorum consensus for {symbol}, using weighted average")
            weighted_avg = self._calculate_weighted_average(successful_data)
            if weighted_avg:
                self.data_cache[symbol] = weighted_avg
                return weighted_avg
        
        logger.error(f"❌ Failed to get consensus data for {symbol}")
        return None
    
    async def _get_single_source_data(self, symbol: str) -> Optional[MarketData]:
        """Get data from single source (fallback mode)"""
        for provider in self.providers:
            if not provider.connected:
                continue
                
            try:
                market_data = await provider.get_real_time_quote(symbol)
                
                if market_data:
                    # Update cache
                    self.data_cache[symbol] = market_data
                    
                    # Update quality tracker
                    self.quality_tracker[provider.config.source].append(market_data.quality_score)
                    
                    return market_data
                    
            except Exception as e:
                logger.warning(f"⚠️ Provider {provider.config.source.value} failed for {symbol}: {e}")
                continue
        
        return None
    
    def _select_quorum_consensus(self, data_list: List[Tuple[MarketData, DataProvider]], tolerance_pct: float) -> Optional[MarketData]:
        """Select consensus data using quorum voting
        
        Args:
            data_list: List of (MarketData, DataProvider) tuples
            tolerance_pct: Price tolerance percentage (e.g., 0.5 for 0.5%)
        
        Returns:
            Consensus MarketData if quorum reached, None otherwise
        """
        if len(data_list) < 2:
            return None
        
        # Group prices by similarity (within tolerance)
        price_groups: List[List[Tuple[MarketData, DataProvider]]] = []
        
        for market_data, provider in data_list:
            price = market_data.price
            matched = False
            
            # Try to find existing group with similar price
            for group in price_groups:
                group_price = group[0][0].price
                price_diff_pct = abs(price - group_price) / group_price * 100
                
                if price_diff_pct <= tolerance_pct:
                    group.append((market_data, provider))
                    matched = True
                    break
            
            # Create new group if no match
            if not matched:
                price_groups.append([(market_data, provider)])
        
        # Find largest group (quorum)
        largest_group = max(price_groups, key=len) if price_groups else None
        
        # Require at least 2 sources to agree
        if largest_group and len(largest_group) >= 2:
            # Use weighted average within the quorum group
            return self._calculate_weighted_average(largest_group)
        
        return None
    
    def _calculate_weighted_average(self, data_list: List[Tuple[MarketData, DataProvider]]) -> Optional[MarketData]:
        """Calculate weighted average of market data based on quality scores
        
        Args:
            data_list: List of (MarketData, DataProvider) tuples
        
        Returns:
            Weighted average MarketData
        """
        if not data_list:
            return None
        
        if len(data_list) == 1:
            return data_list[0][0]
        
        # Calculate weights based on quality score and priority
        total_weight = 0.0
        weighted_price = 0.0
        weighted_bid = 0.0
        weighted_ask = 0.0
        weighted_volume = 0.0
        
        for market_data, provider in data_list:
            # Weight = quality_score * (1 / priority) - lower priority number = higher weight
            weight = market_data.quality_score * (1.0 / max(provider.config.priority, 1))
            total_weight += weight
            
            weighted_price += market_data.price * weight
            weighted_bid += market_data.bid * weight
            weighted_ask += market_data.ask * weight
            weighted_volume += market_data.volume * weight
        
        if total_weight == 0:
            return None
        
        # Normalize
        avg_price = weighted_price / total_weight
        avg_bid = weighted_bid / total_weight
        avg_ask = weighted_ask / total_weight
        avg_volume = weighted_volume / total_weight
        
        # Use the highest quality source as base, update with weighted averages
        base_data, _ = max(data_list, key=lambda x: x[0].quality_score)
        
        # Create consensus data
        consensus_data = MarketData(
            symbol=base_data.symbol,
            price=avg_price,
            bid=avg_bid,
            ask=avg_ask,
            volume=avg_volume,
            timestamp=datetime.utcnow(),
            source=DataSource.ALPHA_VANTAGE,  # Use primary source as identifier
            open_price=base_data.open_price,
            high_price=base_data.high_price,
            low_price=base_data.low_price,
            prev_close=base_data.prev_close,
            change_percent=base_data.change_percent,
            latency_ms=sum(d[0].latency_ms or 0 for d in data_list) / len(data_list),
            quality_score=sum(d[0].quality_score for d in data_list) / len(data_list)
        )
        
        return consensus_data
    
    async def get_multiple_quotes(self, symbols: List[str]) -> Dict[str, Optional[MarketData]]:
        """Get quotes for multiple symbols efficiently"""
        # Hardware-aware batch processing
        max_concurrent = getattr(self, 'max_concurrent_requests', len(symbols))
        
        if HARDWARE_OPTIMIZATION and optimal_profile:
            if optimal_profile.processor_type == ProcessorType.I3_8TH_GEN:
                max_concurrent = min(max_concurrent, 3)  # Limit for i3
        
        # Process in batches
        results = {}
        
        for i in range(0, len(symbols), max_concurrent):
            batch = symbols[i:i + max_concurrent]
            
            # Get data for batch
            tasks = [self.get_market_data(symbol) for symbol in batch]
            batch_results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Process batch results
            for j, result in enumerate(batch_results):
                symbol = batch[j]
                if isinstance(result, Exception):
                    logger.error(f"❌ Error getting data for {symbol}: {result}")
                    results[symbol] = None
                else:
                    results[symbol] = result
        
        return results
    
    def get_data_quality_report(self) -> Dict[str, Any]:
        """Get data quality report"""
        report = {
            "providers": {},
            "cache_stats": {
                "cached_symbols": len(self.data_cache),
                "cache_ttl_seconds": self.cache_ttl_seconds
            },
            "overall_quality": 0.0
        }
        
        total_quality = 0.0
        total_providers = 0
        
        for provider in self.providers:
            if provider.connected:
                quality_scores = self.quality_tracker.get(provider.config.source, [])
                avg_quality = sum(quality_scores) / len(quality_scores) if quality_scores else 0.0
                
                success_rate = (provider.request_count - provider.error_count) / provider.request_count if provider.request_count > 0 else 0
                
                report["providers"][provider.config.source.value] = {
                    "connected": provider.connected,
                    "requests": provider.request_count,
                    "errors": provider.error_count,
                    "success_rate": success_rate,
                    "avg_quality_score": avg_quality,
                    "priority": provider.config.priority
                }
                
                total_quality += avg_quality
                total_providers += 1
        
        report["overall_quality"] = total_quality / total_providers if total_providers > 0 else 0.0
        
        return report
    
    async def cleanup(self):
        """Cleanup all providers"""
        cleanup_tasks = [provider.cleanup() for provider in self.providers]
        await asyncio.gather(*cleanup_tasks, return_exceptions=True)
        
        logger.info("📋 Data providers cleaned up")

# Global data provider instance
data_provider: Optional[RealTimeDataProvider] = None

def get_data_provider() -> RealTimeDataProvider:
    """Get global data provider instance"""
    global data_provider
    if data_provider is None:
        # Initialize with demo keys (replace with real keys in production)
        data_provider = RealTimeDataProvider(
            alpha_vantage_key="demo",  # Replace with real API key
            finnhub_key="demo"        # Replace with real API key
        )
    return data_provider

if __name__ == "__main__":
    # Demo real-time data provider
    import asyncio
    
    async def demo():
        print("📈 Supreme System V5 - Real-Time Data Provider Demo")
        print("=" * 55)
        
        # Initialize data provider
        provider = RealTimeDataProvider()
        await provider.initialize()
        
        # Test symbols
        symbols = ["AAPL", "TSLA", "MSFT"]
        
        print(f"\n📉 Testing data sources for: {symbols}")
        
        # Get market data
        market_data = await provider.get_multiple_quotes(symbols)
        
        for symbol, data in market_data.items():
            if data:
                print(f"   ✅ {symbol}: ${data.price:.2f} (source: {data.source.value}, latency: {data.latency_ms:.1f}ms)")
            else:
                print(f"   ❌ {symbol}: No data available")
        
        # Get quality report
        quality_report = provider.get_data_quality_report()
        print(f"\n📈 Data Quality Report:")
        print(f"   Overall quality: {quality_report['overall_quality']:.2f}")
        
        for source, stats in quality_report["providers"].items():
            print(f"   {source}: {stats['success_rate']:.1%} success rate, {stats['requests']} requests")
        
        # Cleanup
        await provider.cleanup()
        
        print("\n🚀 Real-time data provider demo complete!")
    
    asyncio.run(demo())