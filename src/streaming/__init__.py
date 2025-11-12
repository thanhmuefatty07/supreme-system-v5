"""
Real-time Streaming Analytics for Supreme System V5.

This module contains high-performance streaming implementations:
- Apache Kafka + Flink + ClickHouse integration
- Real-time market data processing
- Streaming ML inference
- High-throughput data pipelines
"""

from .realtime_analytics import (
    RealTimeStreamingAnalytics,
    KafkaStreamingProducer,
    FlinkStreamingProcessor,
    ClickHouseDataSink,
    StreamingQueryEngine,
    MarketData,
    StreamingAnalyticsResult
)

__all__ = [
    'RealTimeStreamingAnalytics',
    'KafkaStreamingProducer',
    'FlinkStreamingProcessor',
    'ClickHouseDataSink',
    'StreamingQueryEngine',
    'MarketData',
    'StreamingAnalyticsResult'
]
