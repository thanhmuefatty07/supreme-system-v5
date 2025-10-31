"""
CoinMarketCap Connector - Secondary data source
#2 most popular crypto API - backup for CoinGecko
"""

import asyncio
import time
from typing import Any, Dict, List, Optional

import aiohttp
from loguru import logger


class CoinMarketCapConnector:
    """
    CoinMarketCap API connector - Secondary data source
    Free tier: 10,000 calls/month, good for backup/validation
    """

    def __init__(self, api_key: Optional[str] = None):
        """Initialize CMC connector"""
        self.base_url = "https://pro-api.coinmarketcap.com/v1"
        self.sandbox_url = "https://sandbox-api.coinmarketcap.com/v1"
        self.api_key = api_key
        self.session: Optional[aiohttp.ClientSession] = None
        self.use_sandbox = not api_key  # Use sandbox if no API key

        # Rate limiting
        self.rate_limit_delay = 6.0  # 10 calls/min free = 6s between calls
        self.last_call = 0.0

        # Symbol mapping to CMC IDs
        self.symbol_map = {
            "BTC-USDT": "1",  # Bitcoin
            "ETH-USDT": "1027",  # Ethereum
            "BNB-USDT": "1839",  # BNB
            "XRP-USDT": "52",  # XRP
            "ADA-USDT": "2010",  # Cardano
            "DOGE-USDT": "74",  # Dogecoin
            "SOL-USDT": "5426",  # Solana
            "MATIC-USDT": "3890",  # Polygon
            "DOT-USDT": "6636",  # Polkadot
            "AVAX-USDT": "5805",  # Avalanche
        }

    async def connect(self):
        """Initialize HTTP session"""
        if not self.session:
            timeout = aiohttp.ClientTimeout(total=30, connect=10)
            headers = {
                "User-Agent": "Supreme-System-V5/1.0",
                "Accept": "application/json",
            }

            if self.api_key:
                headers["X-CMC_PRO_API_KEY"] = self.api_key
                self.rate_limit_delay = 1.0  # Higher limits with API key

            self.session = aiohttp.ClientSession(timeout=timeout, headers=headers)

        base_url = self.base_url if not self.use_sandbox else self.sandbox_url
        logger.info(
            f"✅ CMC connector initialized ({'Sandbox' if self.use_sandbox else 'Live'})"
        )

    async def disconnect(self):
        """Close HTTP session"""
        if self.session:
            await self.session.close()
            self.session = None
        logger.info("✅ CoinMarketCap disconnected")

    async def _rate_limit(self):
        """Enforce rate limiting"""
        elapsed = time.time() - self.last_call
        if elapsed < self.rate_limit_delay:
            await asyncio.sleep(self.rate_limit_delay - elapsed)
        self.last_call = time.time()

    async def get_price_data(self, symbol: str) -> Optional[Dict[str, Any]]:
        """
        Get current price data from CoinMarketCap
        """
        if not self.session:
            await self.connect()

        cmc_id = self.symbol_map.get(symbol)
        if not cmc_id:
            logger.warning(f"⚠️ Unknown CMC mapping for {symbol}")
            return None

        try:
            await self._rate_limit()

            base_url = self.base_url if not self.use_sandbox else self.sandbox_url
            url = f"{base_url}/cryptocurrency/quotes/latest"
            params = {"id": cmc_id, "convert": "USD"}

            async with self.session.get(url, params=params) as response:
                if response.status == 200:
                    data = await response.json()

                    if "data" in data and cmc_id in data["data"]:
                        coin_data = data["data"][cmc_id]
                        quote = coin_data["quote"]["USD"]

                        return {
                            "symbol": symbol,
                            "price": quote["price"],
                            "volume_24h": quote["volume_24h"],
                            "change_24h": quote["percent_change_24h"],
                            "market_cap": quote["market_cap"],
                            "last_updated": quote["last_updated"],
                            "source": "coinmarketcap",
                            "timestamp": time.time(),
                        }
                elif response.status == 429:
                    logger.warning("⚠️ CMC rate limit exceeded")
                    raise aiohttp.ClientResponseError(
                        request_info=response.request_info,
                        history=response.history,
                        status=429,
                    )
                else:
                    logger.error(f"❌ CMC API error: {response.status}")
                    return None

        except Exception as e:
            logger.error(f"❌ CMC error for {symbol}: {e}")
            raise

        return None

    async def get_supported_symbols(self) -> List[str]:
        """Get supported trading symbols"""
        return list(self.symbol_map.keys())
