"""
Alpha Vantage Connector - Multi-asset data provider
Free tier: 5 calls/min for stocks, forex, and crypto
Supports technical indicators and fundamental data
"""

import asyncio
import time
from typing import Dict, Optional, List, Any

import aiohttp
from loguru import logger

class AlphaVantageConnector:
    """
    Alpha Vantage API connector - Multi-asset data provider
    Free tier: 5 calls/min, supports stocks/forex/crypto
    Best for technical indicators and cross-asset analysis
    """
    
    def __init__(self, api_key: str):
        """Initialize Alpha Vantage connector"""
        self.base_url = "https://www.alphavantage.co/query"
        self.api_key = api_key
        self.session: Optional[aiohttp.ClientSession] = None
        
        # Rate limiting for free tier
        self.rate_limit_delay = 12.0  # 5 calls/min = 12s between calls
        self.last_call = 0.0
        
        # Symbol mapping
        self.crypto_symbols = {
            'BTC-USDT': 'BTC',
            'ETH-USDT': 'ETH',
            'BNB-USDT': 'BNB',
            'XRP-USDT': 'XRP',
            'ADA-USDT': 'ADA',
            'DOGE-USDT': 'DOGE',
            'SOL-USDT': 'SOL',
            'MATIC-USDT': 'MATIC',
            'DOT-USDT': 'DOT',
            'AVAX-USDT': 'AVAX'
        }
        
        if not api_key:
            logger.warning("⚠️ Alpha Vantage API key not provided - limited functionality")
    
    async def connect(self):
        """Initialize HTTP session"""
        if not self.session:
            timeout = aiohttp.ClientTimeout(total=60, connect=15)  # Longer timeout for AV
            self.session = aiohttp.ClientSession(
                timeout=timeout,
                headers={
                    'User-Agent': 'Supreme-System-V5/1.0',
                    'Accept': 'application/json'
                }
            )
        
        logger.info("✅ Alpha Vantage connector initialized")
    
    async def disconnect(self):
        """Close HTTP session"""
        if self.session:
            await self.session.close()
            self.session = None
        logger.info("✅ Alpha Vantage disconnected")
    
    async def _rate_limit(self):
        """Enforce strict rate limiting for free tier"""
        elapsed = time.time() - self.last_call
        if elapsed < self.rate_limit_delay:
            sleep_time = self.rate_limit_delay - elapsed
            logger.info(f"⏳ Alpha Vantage rate limit: sleeping {sleep_time:.1f}s")
            await asyncio.sleep(sleep_time)
        self.last_call = time.time()
    
    async def get_price_data(self, symbol: str) -> Optional[Dict[str, Any]]:
        """
        Get current price data for crypto symbol
        Uses CURRENCY_EXCHANGE_RATE function for real-time rates
        """
        if not self.session:
            await self.connect()
            
        if not self.api_key:
            logger.error("❌ Alpha Vantage API key required")
            return None
        
        base_symbol = self.crypto_symbols.get(symbol)
        if not base_symbol:
            logger.warning(f"⚠️ Alpha Vantage: Unknown symbol {symbol}")
            return None
        
        try:
            await self._rate_limit()
            
            # Use CURRENCY_EXCHANGE_RATE for real-time crypto rates
            params = {
                'function': 'CURRENCY_EXCHANGE_RATE',
                'from_currency': base_symbol,
                'to_currency': 'USD',
                'apikey': self.api_key
            }
            
            async with self.session.get(self.base_url, params=params) as response:
                if response.status == 200:
                    data = await response.json()
                    
                    if 'Realtime Currency Exchange Rate' in data:
                        rate_data = data['Realtime Currency Exchange Rate']
                        
                        return {
                            'symbol': symbol,
                            'price': float(rate_data['5. Exchange Rate']),
                            'volume_24h': 0.0,  # AV doesn't provide volume for CURRENCY_EXCHANGE_RATE
                            'change_24h': 0.0,  # Would need separate call
                            'bid': float(rate_data['8. Bid Price']) if '8. Bid Price' in rate_data else 0.0,
                            'ask': float(rate_data['9. Ask Price']) if '9. Ask Price' in rate_data else 0.0,
                            'last_updated': rate_data['6. Last Refreshed'],
                            'source': 'alpha_vantage',
                            'timestamp': time.time()
                        }
                    elif 'Error Message' in data:
                        logger.error(f"❌ Alpha Vantage error: {data['Error Message']}")
                        return None
                    elif 'Note' in data:
                        # Rate limit or API limit reached
                        logger.warning(f"⚠️ Alpha Vantage limit: {data['Note']}")
                        raise aiohttp.ClientResponseError(
                            request_info=response.request_info,
                            history=response.history,
                            status=429,
                            message="API limit reached"
                        )
                else:
                    logger.error(f"❌ Alpha Vantage HTTP error: {response.status}")
                    return None
                    
        except asyncio.TimeoutError:
            logger.error(f"❌ Alpha Vantage timeout for {symbol}")
            raise
        except Exception as e:
            logger.error(f"❌ Alpha Vantage error for {symbol}: {e}")
            raise
        
        return None
    
    async def get_technical_indicators(self, symbol: str, indicator: str = 'RSI') -> Optional[Dict[str, Any]]:
        """
        Get technical indicators from Alpha Vantage
        Available indicators: RSI, EMA, SMA, MACD, BBANDS, etc.
        """
        if not self.api_key:
            return None
            
        base_symbol = self.crypto_symbols.get(symbol)
        if not base_symbol:
            return None
        
        try:
            await self._rate_limit()
            
            # Get RSI indicator
            params = {
                'function': indicator,
                'symbol': f"{base_symbol}USD",  # Alpha Vantage format
                'interval': '5min',
                'time_period': 14,
                'series_type': 'close',
                'apikey': self.api_key
            }
            
            async with self.session.get(self.base_url, params=params) as response:
                if response.status == 200:
                    data = await response.json()
                    
                    # Alpha Vantage returns time series data
                    indicator_key = f"Technical Analysis: {indicator}"
                    if indicator_key in data:
                        indicator_data = data[indicator_key]
                        
                        # Get most recent value
                        if indicator_data:
                            latest_timestamp = max(indicator_data.keys())
                            latest_value = indicator_data[latest_timestamp]
                            
                            return {
                                'symbol': symbol,
                                'indicator': indicator,
                                'value': float(latest_value[indicator]),
                                'timestamp': latest_timestamp,
                                'source': 'alpha_vantage'
                            }
                            
        except Exception as e:
            logger.error(f"❌ Alpha Vantage indicator error: {e}")
            raise
        
        return None
    
    async def get_supported_symbols(self) -> List[str]:
        """Get supported crypto symbols"""
        return list(self.crypto_symbols.keys())
