"""
Exchange connectors for Supreme System V5
Real-time market data and order execution
"""

# Exchange connector imports
try:
    from .okx_connector import OKXConnector

    OKX_AVAILABLE = True
except ImportError:
    OKX_AVAILABLE = False

try:
    from .binance_connector import BinanceConnector

    BINANCE_AVAILABLE = True
except ImportError:
    BINANCE_AVAILABLE = False

# Base exchange interface
from .base import BaseExchange, ExchangeConfig

__all__ = [
    "BaseExchange",
    "ExchangeConfig",
    "OKXConnector",
    "BinanceConnector",
    "OKX_AVAILABLE",
    "BINANCE_AVAILABLE",
]
