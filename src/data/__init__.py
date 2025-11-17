"""
Supreme System V5 - Data Management Module

Handles market data collection, processing, and storage.
"""

from .binance_client import BinanceClient
from .preprocessing import ZScoreNormalizer, VarianceThreshold, safe_divide, normalize_features

__all__ = ["BinanceClient", "ZScoreNormalizer", "VarianceThreshold", "safe_divide", "normalize_features"]
