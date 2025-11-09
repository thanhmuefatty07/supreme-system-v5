"""
Supreme System V5 - Trading Strategies Module

Contains various trading strategies for algorithmic trading.
"""

from .base_strategy import BaseStrategy
from .moving_average import MovingAverageStrategy
from .mean_reversion import MeanReversionStrategy
from .momentum import MomentumStrategy
from .breakout import BreakoutStrategy

__all__ = [
    "BaseStrategy",
    "MovingAverageStrategy",
    "MeanReversionStrategy",
    "MomentumStrategy",
    "BreakoutStrategy"
]
