"""
Supreme System V5 - Data Fabric Module
Multi-source free API aggregation with enterprise-grade quality
"""

from .aggregator import DataAggregator, DataSource
from .cache import CacheManager, DataCache
from .connectors import (
    AlphaVantageConnector,
    BinancePublicConnector,
    CoinGeckoConnector,
    CoinMarketCapConnector,
    CryptoCompareConnector,
    OKXPublicConnector,
)
from .normalizer import DataNormalizer, MarketDataPoint
from .quality import DataQualityScorer, QualityMetrics
from .quality_tracker import (
    QualityTracker,
    QualityTrackerConfig,
    SourceHealth,
    QualityMetrics as QualityTrackerMetrics,
    record_data_quality,
    check_source_health,
    get_quality_tracker
)

__all__ = [
    "DataAggregator",
    "DataSource",
    "CoinGeckoConnector",
    "CoinMarketCapConnector",
    "CryptoCompareConnector",
    "AlphaVantageConnector",
    "BinancePublicConnector",
    "OKXPublicConnector",
    "DataCache",
    "CacheManager",
    "DataQualityScorer",
    "QualityMetrics",
    "DataNormalizer",
    "MarketDataPoint",
    "QualityTracker",
    "QualityTrackerConfig",
    "SourceHealth",
    "QualityTrackerMetrics",
    "record_data_quality",
    "check_source_health",
    "get_quality_tracker",
]
