"""
🔧 Supreme System V5 - Real-Time Data Sources
Production-grade financial data integration

Supported Data Sources:
- Alpha Vantage (Primary)
- Finnhub (Secondary)
- Yahoo Finance (Backup)
- Binance WebSocket (Crypto)
- Real-time news feeds
- Economic indicators
"""

__version__ = "5.0.0"
__author__ = "Supreme System V5 Team"

from .real_time_data import RealTimeDataProvider
from .alpha_vantage import AlphaVantageProvider  
from .finnhub_provider import FinnhubProvider
from .yahoo_finance import YahooFinanceProvider
from .binance_websocket import BinanceWebSocketProvider
from .economic_indicators import EconomicDataProvider
from .news_feeds import NewsDataProvider

__all__ = [
    "RealTimeDataProvider",
    "AlphaVantageProvider", 
    "FinnhubProvider",
    "YahooFinanceProvider",
    "BinanceWebSocketProvider", 
    "EconomicDataProvider",
    "NewsDataProvider"
]