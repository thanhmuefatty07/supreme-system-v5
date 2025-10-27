#!/usr/bin/env python3
"""
🌐 Supreme System V5 - Data Connectors Abstraction Layer
Free-tier sources with enterprise upgrade path

Features:
- Universal data schema standardization  
- Multi-provider failover & health checks
- Rate limiting & quota management
- Free-tier optimized with paid upgrade adapters
- Real-time + historical data integration
- Quality scoring & validation
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Optional, Union, Any
from enum import Enum

# Core interfaces for all data types
__all__ = [
    'DataQuality',
    'IPriceFeed', 
    'IOrderBookFeed',
    'INewsFeed',
    'IOnChainFeed', 
    'IMacroFeed',
    'ISocialFeed',
    'DataConnectorManager',
    'StandardizedData'
]

class DataQuality(Enum):
    """Data quality levels"""
    EXCELLENT = 1.0
    GOOD = 0.8
    ACCEPTABLE = 0.6  
    POOR = 0.4
    UNRELIABLE = 0.2

@dataclass
class StandardizedData:
    """Universal data container with quality metrics"""
    timestamp: datetime
    symbol: str
    data: Dict[str, Any]
    source: str
    quality: float
    latency_ms: float
    metadata: Dict[str, Any] = None

class IPriceFeed(ABC):
    """Price/OHLCV data interface"""
    
    @abstractmethod
    async def get_real_time_price(self, symbol: str) -> Optional[StandardizedData]:
        """Get current price data"""
        pass
    
    @abstractmethod  
    async def get_historical_ohlcv(self, symbol: str, period: str, limit: int = 100) -> List[StandardizedData]:
        """Get historical OHLCV data"""
        pass
    
    @abstractmethod
    async def health_check(self) -> Dict[str, Any]:
        """Provider health and quota status"""
        pass

class IOrderBookFeed(ABC):
    """Order book depth data interface"""
    
    @abstractmethod
    async def get_order_book(self, symbol: str, depth: int = 20) -> Optional[StandardizedData]:
        """Get order book snapshot"""
        pass
    
    @abstractmethod
    async def subscribe_order_book(self, symbol: str, callback: callable) -> None:
        """Subscribe to order book updates"""
        pass

class INewsFeed(ABC):
    """News and sentiment data interface"""
    
    @abstractmethod  
    async def get_latest_news(self, symbols: List[str] = None, limit: int = 10) -> List[StandardizedData]:
        """Get latest news with sentiment"""
        pass
    
    @abstractmethod
    async def get_sentiment_score(self, symbol: str) -> Optional[StandardizedData]:
        """Get aggregated sentiment score"""
        pass

class IOnChainFeed(ABC):
    """On-chain/whale tracking interface"""
    
    @abstractmethod
    async def get_whale_transactions(self, min_value_usd: float = 1000000) -> List[StandardizedData]:
        """Get large transactions/whale activity"""
        pass
    
    @abstractmethod
    async def get_exchange_flows(self, symbol: str) -> Optional[StandardizedData]:
        """Get exchange inflow/outflow data"""
        pass

class IMacroFeed(ABC):
    """Macro economic data interface"""
    
    @abstractmethod
    async def get_economic_indicators(self, indicators: List[str]) -> List[StandardizedData]:
        """Get economic indicators (GDP, CPI, etc.)"""
        pass
    
    @abstractmethod
    async def get_economic_calendar(self, days_ahead: int = 7) -> List[StandardizedData]:
        """Get upcoming economic events"""
        pass

class ISocialFeed(ABC):
    """Social media sentiment interface"""
    
    @abstractmethod
    async def get_social_sentiment(self, symbol: str) -> Optional[StandardizedData]:
        """Get aggregated social sentiment"""  
        pass
    
    @abstractmethod
    async def get_trending_topics(self, limit: int = 10) -> List[StandardizedData]:
        """Get trending financial topics"""
        pass

class DataConnectorManager:
    """Central manager for all data connectors with failover"""
    
    def __init__(self):
        self.price_providers: List[IPriceFeed] = []
        self.orderbook_providers: List[IOrderBookFeed] = []
        self.news_providers: List[INewsFeed] = []  
        self.onchain_providers: List[IOnChainFeed] = []
        self.macro_providers: List[IMacroFeed] = []
        self.social_providers: List[ISocialFeed] = []
        
        self.health_status = {}
        self.quota_status = {}
    
    def register_price_provider(self, provider: IPriceFeed, priority: int = 0):
        """Register price data provider"""
        self.price_providers.append((priority, provider))
        self.price_providers.sort(key=lambda x: x[0])  # Sort by priority
    
    def register_orderbook_provider(self, provider: IOrderBookFeed, priority: int = 0):
        """Register order book provider"""
        self.orderbook_providers.append((priority, provider))
        self.orderbook_providers.sort(key=lambda x: x[0])
    
    def register_news_provider(self, provider: INewsFeed, priority: int = 0):
        """Register news provider"""
        self.news_providers.append((priority, provider))
        self.news_providers.sort(key=lambda x: x[0])
    
    def register_onchain_provider(self, provider: IOnChainFeed, priority: int = 0):
        """Register on-chain provider"""
        self.onchain_providers.append((priority, provider))
        self.onchain_providers.sort(key=lambda x: x[0])
    
    def register_macro_provider(self, provider: IMacroFeed, priority: int = 0):
        """Register macro data provider"""
        self.macro_providers.append((priority, provider))
        self.macro_providers.sort(key=lambda x: x[0])
    
    def register_social_provider(self, provider: ISocialFeed, priority: int = 0):
        """Register social sentiment provider"""
        self.social_providers.append((priority, provider))
        self.social_providers.sort(key=lambda x: x[0])
    
    async def get_price_with_failover(self, symbol: str) -> Optional[StandardizedData]:
        """Get price data with automatic failover"""
        for priority, provider in self.price_providers:
            try:
                health = await provider.health_check()
                if health.get('healthy', False) and health.get('quota_remaining', 0) > 0:
                    return await provider.get_real_time_price(symbol)
            except Exception as e:
                continue
        return None
    
    async def get_orderbook_with_failover(self, symbol: str) -> Optional[StandardizedData]:
        """Get order book with automatic failover"""
        for priority, provider in self.orderbook_providers:
            try:
                return await provider.get_order_book(symbol)
            except Exception:
                continue
        return None
    
    async def get_news_with_failover(self, symbols: List[str]) -> List[StandardizedData]:
        """Get news with automatic failover"""
        for priority, provider in self.news_providers:
            try:
                return await provider.get_latest_news(symbols)
            except Exception:
                continue
        return []
    
    async def get_whale_activity_with_failover(self) -> List[StandardizedData]:
        """Get whale activity with automatic failover"""
        for priority, provider in self.onchain_providers:
            try:
                return await provider.get_whale_transactions()
            except Exception:
                continue
        return []
    
    async def get_macro_data_with_failover(self, indicators: List[str]) -> List[StandardizedData]:
        """Get macro data with automatic failover"""
        for priority, provider in self.macro_providers:
            try:
                return await provider.get_economic_indicators(indicators)
            except Exception:
                continue
        return []
    
    async def get_social_sentiment_with_failover(self, symbol: str) -> Optional[StandardizedData]:
        """Get social sentiment with automatic failover"""
        for priority, provider in self.social_providers:
            try:
                return await provider.get_social_sentiment(symbol)
            except Exception:
                continue
        return None
    
    async def health_check_all(self) -> Dict[str, Any]:
        """Check health of all providers"""
        health_report = {
            'price_providers': [],
            'orderbook_providers': [],
            'news_providers': [],
            'onchain_providers': [],
            'macro_providers': [],
            'social_providers': []
        }
        
        # Check price providers
        for priority, provider in self.price_providers:
            try:
                health = await provider.health_check()
                health_report['price_providers'].append({
                    'provider': provider.__class__.__name__,
                    'priority': priority,
                    'health': health
                })
            except Exception as e:
                health_report['price_providers'].append({
                    'provider': provider.__class__.__name__, 
                    'priority': priority,
                    'health': {'healthy': False, 'error': str(e)}
                })
        
        return health_report

# Global connector manager instance
connector_manager = DataConnectorManager()