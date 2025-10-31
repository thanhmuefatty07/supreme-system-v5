"""
Supreme System V5 - Data Fabric Module
Multi-source free API aggregation with enterprise-grade quality
"""

from .aggregator import DataAggregator, DataSource
from .connectors import (
    CoinGeckoConnector,
    CoinMarketCapConnector,
    CryptoCompareConnector,
    AlphaVantageConnector,
    BinancePublicConnector,
    OKXPublicConnector
)
from .cache import DataCache, CacheManager, CacheConfig, MemoryCache, RedisCache, PostgreSQLPersistence
from .quality import DataQualityScorer, QualityMetrics
from .normalizer import DataNormalizer, MarketDataPoint

__all__ = [
    'DataAggregator',
    'DataSource',
    'CoinGeckoConnector',
    'CoinMarketCapConnector',
    'CryptoCompareConnector',
    'AlphaVantageConnector',
    'BinancePublicConnector',
    'OKXPublicConnector',
    'DataCache',
    'CacheManager',
    'CacheConfig',
    'MemoryCache',
    'RedisCache',
    'PostgreSQLPersistence',
    'DataQualityScorer',
    'QualityMetrics',
    'DataNormalizer',
    'MarketDataPoint'
]
