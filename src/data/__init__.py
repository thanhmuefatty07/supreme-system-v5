"""
Supreme System V5 - Data Management Module

Handles market data collection, processing, and storage.
"""

from .binance_client import BinanceClient
from .preprocessing import ZScoreNormalizer, safe_divide

__all__ = ["BinanceClient", "ZScoreNormalizer", "safe_divide"]
