"""
Data Quality Scorer - Intelligent source selection
Evaluates latency, consistency, completeness for optimal data routing
"""

import statistics
import time
from collections import defaultdict, deque
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import numpy as np
from loguru import logger


@dataclass
class QualityMetrics:
    """Data quality metrics for a single data point"""

    latency_score: float  # 0.0-1.0 (lower latency = higher score)
    completeness_score: float  # 0.0-1.0 (more fields = higher score)
    consistency_score: float  # 0.0-1.0 (vs historical = higher score)
    freshness_score: float  # 0.0-1.0 (newer data = higher score)
    overall_score: float  # Weighted composite score

    def __post_init__(self):
        """Calculate overall score"""
        if self.overall_score == 0.0:  # Not manually set
            self.overall_score = (
                self.latency_score * 0.3
                + self.completeness_score * 0.2
                + self.consistency_score * 0.3
                + self.freshness_score * 0.2
            )


class DataQualityScorer:
    """
    Intelligent data quality scorer
    Maintains rolling statistics for each source and symbol
    """

    def __init__(self, history_size: int = 100):
        """Initialize quality scorer"""
        self.history_size = history_size

        # Rolling statistics per source+symbol
        self.price_history: Dict[str, deque] = defaultdict(
            lambda: deque(maxlen=history_size)
        )
        self.latency_history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=50))
        self.last_update: Dict[str, float] = {}

        # Field completeness weights
        self.required_fields = {
            "price": 1.0,  # Critical
            "volume_24h": 0.8,  # Important
            "change_24h": 0.6,  # Useful
            "high_24h": 0.4,  # Nice to have
            "low_24h": 0.4,  # Nice to have
            "bid": 0.5,  # Important for spread
            "ask": 0.5,  # Important for spread
            "market_cap": 0.3,  # Optional
        }

    def score(self, data_point: Any, symbol: str, source: str = None) -> QualityMetrics:
        """
        Score data quality for a market data point
        """
        if not data_point:
            return QualityMetrics(0, 0, 0, 0, 0)

        # Handle both dict (raw data) and MarketDataPoint (normalized data)
        if hasattr(data_point, 'source'):
            # MarketDataPoint object
            data_source = source or data_point.source
        else:
            # Dict object
            data_source = source or data_point.get('source', 'unknown')

        source_key = f"{data_source}:{symbol}"

        # Score components
        latency_score = self._score_latency(data_point, source_key)
        completeness_score = self._score_completeness(data_point)
        consistency_score = self._score_consistency(data_point, source_key)
        freshness_score = self._score_freshness(data_point)

        # Update history
        self._update_history(data_point, source_key)

        return QualityMetrics(
            latency_score=latency_score,
            completeness_score=completeness_score,
            consistency_score=consistency_score,
            freshness_score=freshness_score,
            overall_score=0.0,  # Will be calculated in __post_init__
        )

    def _score_latency(self, data_point: Any, source_key: str) -> float:
        """
        Score latency (0.0 = slow, 1.0 = fast)
        """
        current_time = time.time()
        if hasattr(data_point, 'timestamp'):
            data_timestamp = data_point.timestamp
        else:
            data_timestamp = data_point.get("timestamp", current_time)

        latency = current_time - data_timestamp

        # Record latency history
        self.latency_history[source_key].append(latency)

        # Score based on latency (exponential decay)
        # 0s = 1.0, 1s = 0.6, 5s = 0.2, 10s+ = 0.1
        if latency < 0.5:
            score = 1.0
        elif latency < 2.0:
            score = 0.8
        elif latency < 5.0:
            score = 0.5
        elif latency < 10.0:
            score = 0.3
        else:
            score = 0.1

        return score

    def _score_completeness(self, data_point: Any) -> float:
        """
        Score data completeness based on available fields
        """
        total_weight = sum(self.required_fields.values())
        available_weight = 0.0

        for field, weight in self.required_fields.items():
            try:
                if hasattr(data_point, field):
                    value = getattr(data_point, field)
                else:
                    value = data_point.get(field) if hasattr(data_point, 'get') else None

                if value is not None:
                    # Check for non-zero values (many APIs return 0 for missing data)
                    if isinstance(value, (int, float)) and value > 0:
                        available_weight += weight
                    elif isinstance(value, str) and value:
                        available_weight += weight
            except (AttributeError, KeyError, TypeError):
                continue

        return available_weight / total_weight if total_weight > 0 else 0.0

    def _score_consistency(self, data_point: Any, source_key: str) -> float:
        """
        Score consistency against historical data
        """
        if hasattr(data_point, 'price'):
            price = data_point.price
        else:
            price = data_point.get("price", 0)
        if not price or price <= 0:
            return 0.0

        price_hist = self.price_history[source_key]
        if len(price_hist) < 5:
            return 0.8  # Default score for new sources

        # Calculate recent price statistics
        recent_prices = list(price_hist)[-10:]  # Last 10 prices
        median_price = statistics.median(recent_prices)
        price_std = statistics.stdev(recent_prices) if len(recent_prices) > 1 else 0

        # Score based on deviation from median
        if price_std == 0:
            deviation = 0
        else:
            deviation = abs(price - median_price) / price_std

        # Consistency score (higher = more consistent)
        if deviation < 1.0:  # Within 1 std dev
            score = 1.0
        elif deviation < 2.0:  # Within 2 std dev
            score = 0.8
        elif deviation < 3.0:  # Within 3 std dev
            score = 0.5
        else:  # Outlier
            score = 0.1
            logger.warning(
                f"📊 Price outlier detected: {source_key} price={price:.4f} vs median={median_price:.4f}"
            )

        return score

    def _score_freshness(self, data_point: Any) -> float:
        """
        Score data freshness (0.0 = stale, 1.0 = fresh)
        """
        if hasattr(data_point, 'timestamp'):
            data_timestamp = data_point.timestamp
        else:
            data_timestamp = data_point.get("timestamp", time.time())
        current_time = time.time()

        age_seconds = current_time - data_timestamp

        # Freshness score (exponential decay)
        if age_seconds < 5:  # <5s = perfect
            score = 1.0
        elif age_seconds < 30:  # <30s = excellent
            score = 0.9
        elif age_seconds < 60:  # <1min = good
            score = 0.7
        elif age_seconds < 300:  # <5min = acceptable
            score = 0.4
        else:  # >5min = stale
            score = 0.1

        return score

    def _update_history(self, data_point: Any, source_key: str):
        """
        Update rolling history for statistical analysis
        """
        if hasattr(data_point, 'price'):
            price = data_point.price
        else:
            price = data_point.get("price")

        if price and price > 0:
            self.price_history[source_key].append(price)

        self.last_update[source_key] = time.time()

    def get_source_stats(self, source: str, symbol: str) -> Dict[str, Any]:
        """
        Get statistical summary for source+symbol
        """
        source_key = f"{source}:{symbol}"

        price_hist = self.price_history[source_key]
        latency_hist = self.latency_history[source_key]

        stats = {
            "source": source,
            "symbol": symbol,
            "data_points": len(price_hist),
            "last_update": self.last_update.get(source_key, 0),
        }

        if price_hist:
            prices = list(price_hist)
            stats.update(
                {
                    "price_mean": statistics.mean(prices),
                    "price_median": statistics.median(prices),
                    "price_std": statistics.stdev(prices) if len(prices) > 1 else 0,
                    "price_min": min(prices),
                    "price_max": max(prices),
                }
            )

        if latency_hist:
            latencies = list(latency_hist)
            stats.update(
                {
                    "latency_mean": statistics.mean(latencies),
                    "latency_p95": np.percentile(latencies, 95),
                    "latency_p99": np.percentile(latencies, 99),
                }
            )

        return stats
