"""
Supreme System V5 - Trading Strategies Module

Contains various trading strategies for algorithmic trading.
"""

from .base_strategy import BaseStrategy
from .moving_average import MovingAverageStrategy

__all__ = ["BaseStrategy", "MovingAverageStrategy"]
