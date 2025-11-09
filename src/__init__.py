"""
Supreme System V5 - Real Algorithmic Trading System

A production-ready trading system for ETH-USDT scalping built from the ground up.
"""

__version__ = "0.1.0"
__author__ = "Supreme Trading Team"
__description__ = "Real algorithmic trading system for ETH-USDT scalping"

# Import main components for easy access
from .data.binance_client import BinanceClient
from .strategies.base_strategy import BaseStrategy
from .strategies.moving_average import MovingAverageStrategy
from .risk.risk_manager import RiskManager

__all__ = [
    "BinanceClient",
    "BaseStrategy",
    "MovingAverageStrategy",
    "RiskManager",
]
