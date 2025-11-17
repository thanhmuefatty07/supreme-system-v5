"""
Supreme System V5 - Data Management Module

Handles market data collection, processing, and storage.
"""

from .binance_client import BinanceClient
from .preprocessing import ZScoreNormalizer, VarianceThreshold, safe_divide, normalize_features
from .validation import WalkForwardValidator, plot_walk_forward_splits, compare_cv_methods

__all__ = [
    "BinanceClient",
    "ZScoreNormalizer",
    "VarianceThreshold",
    "safe_divide",
    "normalize_features",
    "WalkForwardValidator",
    "plot_walk_forward_splits",
    "compare_cv_methods"
]
