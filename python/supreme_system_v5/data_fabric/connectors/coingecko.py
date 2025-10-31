"""
CoinGecko Connector - Primary free data source
Rated #1 crypto API with 99.9% uptime and 13M+ tokens coverage
"""

import asyncio
import time
from typing import Dict, Optional, List, Any

import aiohttp
from loguru import logger

class CoinGeckoConnector:
    """
    CoinGecko API connector - Primary data source
    Free tier: 10-30 calls/min, 99.9% uptime, 13M tokens, 240+ networks
    """
    
    def __init__(self, api_key: Optional[str] = None):
        """Initialize CoinGecko connector"""
        self.base_url = "https://api.coingecko.com/api/v3"
        self.api_key = api_key  # Optional for higher limits
        self.session: Optional[aiohttp.ClientSession] = None
        self.rate_limit_delay = 2.0  # 30 calls/min = 2s between calls
        self.last_call = 0.0
        
        # Symbol mapping (CoinGecko uses different IDs)
        self.symbol_map = {
            'BTC-USDT': 'bitcoin',
            'ETH-USDT': 'ethereum', 
            'BNB-USDT': 'binancecoin',
            'XRP-USDT': 'ripple',
            'ADA-USDT': 'cardano',
            'DOGE-USDT': 'dogecoin',
            'SOL-USDT': 'solana',
            'MATIC-USDT': 'matic-network',
            'DOT-USDT': 'polkadot',
            'AVAX-USDT': 'avalanche-2'
        }
    
    async def connect(self):
        """Initialize HTTP session"""
        if not self.session:
            timeout = aiohttp.ClientTimeout(total=30, connect=10)
            self.session = aiohttp.ClientSession(
                timeout=timeout,
                headers={
                    'User-Agent': 'Supreme-System-V5/1.0',
                    'Accept': 'application/json'
                }
            )
            
            if self.api_key:
                self.session.headers['x-cg-demo-api-key'] = self.api_key
                self.rate_limit_delay = 1.2  # Higher limits with API key
            
        logger.info("✅ CoinGecko connector initialized")
    
    async def disconnect(self):
        """Close HTTP session"""
        if self.session:
            await self.session.close()
            self.session = None
        logger.info("✅ CoinGecko disconnected")
    
    async def _rate_limit(self):
        """Enforce rate limiting"""
        elapsed = time.time() - self.last_call
        if elapsed < self.rate_limit_delay:
            await asyncio.sleep(self.rate_limit_delay - elapsed)
        self.last_call = time.time()
    
    async def get_price_data(self, symbol: str) -> Optional[Dict[str, Any]]:
        """
        Get current price data for symbol from CoinGecko
        """
        if not self.session:
            await self.connect()
        
        # Map symbol to CoinGecko ID
        coin_id = self.symbol_map.get(symbol)
        if not coin_id:
            logger.warning(f"⚠️ Unknown symbol mapping for {symbol}")
            return None
        
        try:
            await self._rate_limit()
            
            # Use simple price endpoint for speed
            url = f"{self.base_url}/simple/price"
            params = {
                'ids': coin_id,
                'vs_currencies': 'usd',
                'include_market_cap': 'true',
                'include_24hr_vol': 'true', 
                'include_24hr_change': 'true',
                'include_last_updated_at': 'true'
            }
            
            async with self.session.get(url, params=params) as response:
                if response.status == 200:
                    data = await response.json()
                    
                    if coin_id in data:
                        coin_data = data[coin_id]
                        
                        return {
                            'symbol': symbol,
                            'price': coin_data.get('usd', 0.0),
                            'volume_24h': coin_data.get('usd_24h_vol', 0.0),
                            'change_24h': coin_data.get('usd_24h_change', 0.0),
                            'market_cap': coin_data.get('usd_market_cap', 0.0),
                            'last_updated': coin_data.get('last_updated_at', int(time.time())),
                            'source': 'coingecko',
                            'timestamp': time.time()
                        }
                elif response.status == 429:
                    logger.warning("⚠️ CoinGecko rate limit exceeded")
                    raise aiohttp.ClientResponseError(
                        request_info=response.request_info,
                        history=response.history,
                        status=429,
                        message="Rate limit exceeded"
                    )
                else:
                    logger.error(f"❌ CoinGecko API error: {response.status}")
                    return None
                    
        except asyncio.TimeoutError:
            logger.error(f"❌ CoinGecko timeout for {symbol}")
            raise
        except Exception as e:
            logger.error(f"❌ CoinGecko error for {symbol}: {e}")
            raise
        
        return None
    
    async def get_historical_data(self, symbol: str, days: int = 30) -> Optional[List[Dict]]:
        """
        Get historical price data
        """
        if not self.session:
            await self.connect()
            
        coin_id = self.symbol_map.get(symbol)
        if not coin_id:
            return None
        
        try:
            await self._rate_limit()
            
            url = f"{self.base_url}/coins/{coin_id}/market_chart"
            params = {
                'vs_currency': 'usd',
                'days': min(days, 365),  # CoinGecko free limit
                'interval': 'hourly' if days <= 90 else 'daily'
            }
            
            async with self.session.get(url, params=params) as response:
                if response.status == 200:
                    data = await response.json()
                    
                    # Convert to standard format
                    historical = []
                    prices = data.get('prices', [])
                    volumes = data.get('total_volumes', [])
                    
                    for i, (timestamp, price) in enumerate(prices):
                        volume = volumes[i][1] if i < len(volumes) else 0.0
                        
                        historical.append({
                            'timestamp': timestamp / 1000,  # Convert to seconds
                            'price': price,
                            'volume': volume,
                            'source': 'coingecko'
                        })
                    
                    logger.info(f"📊 Got {len(historical)} historical points for {symbol}")
                    return historical
                    
        except Exception as e:
            logger.error(f"❌ CoinGecko historical error for {symbol}: {e}")
            raise
        
        return None
    
    async def get_supported_symbols(self) -> List[str]:
        """
        Get list of supported trading symbols
        """
        return list(self.symbol_map.keys())
