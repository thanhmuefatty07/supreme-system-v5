"""
Data Normalizer - Standardize data from multiple sources
Ensures consistent format across all free API sources
"""

import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, Optional, Union

from loguru import logger


@dataclass
class MarketDataPoint:
    """
    Normalized market data structure
    Standard format for all data sources
    """

    symbol: str
    timestamp: float  # Unix timestamp in seconds
    price: float  # Current/last price in USD
    volume_24h: float  # 24h trading volume
    change_24h: float  # 24h price change percentage
    high_24h: float  # 24h high price
    low_24h: float  # 24h low price
    bid: float  # Best bid price
    ask: float  # Best ask price
    market_cap: Optional[float] = None  # Market capitalization
    source: str = "unknown"  # Data source name
    quality_score: float = 1.0  # Quality score from scorer

    @property
    def spread(self) -> float:
        """Calculate bid-ask spread"""
        return abs(self.ask - self.bid) if self.ask and self.bid else 0.0

    @property
    def spread_bps(self) -> float:
        """Calculate spread in basis points"""
        if self.price <= 0:
            return 0.0
        return (self.spread / self.price) * 10000

    @property
    def age_seconds(self) -> float:
        """Data age in seconds"""
        return time.time() - self.timestamp

    def is_stale(self, max_age_seconds: float = 60) -> bool:
        """Check if data is stale"""
        return self.age_seconds > max_age_seconds


class DataNormalizer:
    """
    Data normalizer for multi-source API aggregation
    Handles different API formats and ensures consistency
    """

    def __init__(self):
        """Initialize normalizer"""
        self.normalization_stats = {
            "total_normalized": 0,
            "errors": 0,
            "sources": set(),
        }

    def normalize(
        self, source: str, raw_data: Dict[str, Any]
    ) -> Optional[MarketDataPoint]:
        """
        Normalize raw API data to standard MarketDataPoint format
        """
        try:
            if source == "coingecko":
                return self._normalize_coingecko(raw_data)
            elif source == "coinmarketcap":
                return self._normalize_coinmarketcap(raw_data)
            elif source == "cryptocompare":
                return self._normalize_cryptocompare(raw_data)
            elif source == "alpha_vantage":
                return self._normalize_alpha_vantage(raw_data)
            elif source == "binance":
                return self._normalize_binance(raw_data)
            elif source == "okx":
                return self._normalize_okx(raw_data)
            else:
                logger.warning(f"⚠️ Unknown source for normalization: {source}")
                return self._normalize_generic(raw_data)

        except Exception as e:
            logger.error(f"❌ Normalization error for {source}: {e}")
            self.normalization_stats["errors"] += 1
            return None

    def _normalize_coingecko(self, data: Dict) -> MarketDataPoint:
        """Normalize CoinGecko API response"""
        return MarketDataPoint(
            symbol=data.get("symbol", ""),
            timestamp=data.get("timestamp", time.time()),
            price=float(data.get("price", 0)),
            volume_24h=float(data.get("volume_24h", 0)),
            change_24h=float(data.get("change_24h", 0)),
            high_24h=float(
                data.get("price", 0)
            ),  # CG simple API doesn't include high/low
            low_24h=float(data.get("price", 0)),
            bid=float(data.get("price", 0)) * 0.9999,  # Estimated spread
            ask=float(data.get("price", 0)) * 1.0001,
            market_cap=data.get("market_cap"),
            source="coingecko",
        )

    def _normalize_coinmarketcap(self, data: Dict) -> MarketDataPoint:
        """Normalize CoinMarketCap API response"""
        return MarketDataPoint(
            symbol=data.get("symbol", ""),
            timestamp=data.get("timestamp", time.time()),
            price=float(data.get("price", 0)),
            volume_24h=float(data.get("volume_24h", 0)),
            change_24h=float(data.get("change_24h", 0)),
            high_24h=float(data.get("price", 0)),
            low_24h=float(data.get("price", 0)),
            bid=float(data.get("price", 0)) * 0.9999,
            ask=float(data.get("price", 0)) * 1.0001,
            market_cap=data.get("market_cap"),
            source="coinmarketcap",
        )

    def _normalize_cryptocompare(self, data: Dict) -> MarketDataPoint:
        """Normalize CryptoCompare API response"""
        return MarketDataPoint(
            symbol=data.get("symbol", ""),
            timestamp=data.get("timestamp", time.time()),
            price=float(data.get("price", 0)),
            volume_24h=float(data.get("volume_24h", 0)),
            change_24h=float(data.get("change_24h", 0)),
            high_24h=float(data.get("high_24h", 0)),
            low_24h=float(data.get("low_24h", 0)),
            bid=float(data.get("price", 0)) * 0.9999,
            ask=float(data.get("price", 0)) * 1.0001,
            market_cap=data.get("market_cap"),
            source="cryptocompare",
        )

    def _normalize_binance(self, data: Dict) -> MarketDataPoint:
        """Normalize Binance API response"""
        return MarketDataPoint(
            symbol=data.get("symbol", ""),
            timestamp=data.get("timestamp", time.time()),
            price=float(data.get("price", 0)),
            volume_24h=float(data.get("volume_24h", 0)),
            change_24h=float(data.get("change_24h", 0)),
            high_24h=float(data.get("high_24h", 0)),
            low_24h=float(data.get("low_24h", 0)),
            bid=float(data.get("bid", data.get("price", 0))),
            ask=float(data.get("ask", data.get("price", 0))),
            source="binance",
        )

    def _normalize_okx(self, data: Dict) -> MarketDataPoint:
        """Normalize OKX API response"""
        return MarketDataPoint(
            symbol=data.get("symbol", ""),
            timestamp=data.get("timestamp", time.time()),
            price=float(data.get("price", 0)),
            volume_24h=float(data.get("volume_24h", 0)),
            change_24h=float(data.get("change_24h", 0)),
            high_24h=float(data.get("high_24h", 0)),
            low_24h=float(data.get("low_24h", 0)),
            bid=float(data.get("bid", data.get("price", 0))),
            ask=float(data.get("ask", data.get("price", 0))),
            source="okx",
        )

    def _normalize_alpha_vantage(self, data: Dict) -> MarketDataPoint:
        """Normalize Alpha Vantage API response"""
        return MarketDataPoint(
            symbol=data.get("symbol", ""),
            timestamp=data.get("timestamp", time.time()),
            price=float(data.get("price", 0)),
            volume_24h=float(data.get("volume_24h", 0)),
            change_24h=float(data.get("change_24h", 0)),
            high_24h=float(data.get("high_24h", 0)),
            low_24h=float(data.get("low_24h", 0)),
            bid=float(data.get("price", 0)) * 0.9999,
            ask=float(data.get("price", 0)) * 1.0001,
            source="alpha_vantage",
        )

    def _normalize_generic(self, data: Dict) -> MarketDataPoint:
        """Generic normalization for unknown sources"""
        return MarketDataPoint(
            symbol=data.get("symbol", data.get("pair", "")),
            timestamp=data.get("timestamp", data.get("time", time.time())),
            price=float(data.get("price", data.get("last", data.get("close", 0)))),
            volume_24h=float(
                data.get("volume_24h", data.get("volume", data.get("vol", 0)))
            ),
            change_24h=float(data.get("change_24h", data.get("change", 0))),
            high_24h=float(data.get("high_24h", data.get("high", 0))),
            low_24h=float(data.get("low_24h", data.get("low", 0))),
            bid=float(data.get("bid", data.get("price", 0))),
            ask=float(data.get("ask", data.get("price", 0))),
            source=data.get("source", "unknown"),
        )

    def _update_history(self, data_point: Dict, source_key: str):
        """Update normalization history"""
        self.normalization_stats["total_normalized"] += 1
        self.normalization_stats["sources"].add(source_key.split(":")[0])

    def get_stats(self) -> Dict[str, Any]:
        """Get normalization statistics"""
        return {
            "total_normalized": self.normalization_stats["total_normalized"],
            "total_errors": self.normalization_stats["errors"],
            "error_rate": self.normalization_stats["errors"]
            / max(1, self.normalization_stats["total_normalized"]),
            "active_sources": len(self.normalization_stats["sources"]),
            "sources": list(self.normalization_stats["sources"]),
        }
