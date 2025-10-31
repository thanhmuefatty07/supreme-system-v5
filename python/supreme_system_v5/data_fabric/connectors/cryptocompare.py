"""
CryptoCompare Connector - Multi-exchange aggregated data
Free tier: 100,000 calls/month - excellent for validation
"""

import asyncio
import time
from typing import Any, Dict, List, Optional

import aiohttp
from loguru import logger


class CryptoCompareConnector:
    """
    CryptoCompare API connector - Multi-exchange aggregated data
    Free tier: 100,000 calls/month, multi-source normalized feed
    """

    def __init__(self, api_key: Optional[str] = None):
        """Initialize CryptoCompare connector"""
        self.base_url = "https://min-api.cryptocompare.com/data"
        self.api_key = api_key
        self.session: Optional[aiohttp.ClientSession] = None

        # Rate limiting
        self.rate_limit_delay = 1.0  # More generous limits
        self.last_call = 0.0

        # Symbol mapping (CryptoCompare uses base symbols)
        self.symbol_map = {
            "BTC-USDT": ("BTC", "USDT"),
            "ETH-USDT": ("ETH", "USDT"),
            "BNB-USDT": ("BNB", "USDT"),
            "XRP-USDT": ("XRP", "USDT"),
            "ADA-USDT": ("ADA", "USDT"),
            "DOGE-USDT": ("DOGE", "USDT"),
            "SOL-USDT": ("SOL", "USDT"),
            "MATIC-USDT": ("MATIC", "USDT"),
            "DOT-USDT": ("DOT", "USDT"),
            "AVAX-USDT": ("AVAX", "USDT"),
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
                headers["authorization"] = f"Apikey {self.api_key}"
                self.rate_limit_delay = 0.5  # Higher limits

            self.session = aiohttp.ClientSession(timeout=timeout, headers=headers)

        logger.info("✅ CryptoCompare connector initialized")

    async def disconnect(self):
        """Close HTTP session"""
        if self.session:
            await self.session.close()
            self.session = None
        logger.info("✅ CryptoCompare disconnected")

    async def _rate_limit(self):
        """Enforce rate limiting"""
        elapsed = time.time() - self.last_call
        if elapsed < self.rate_limit_delay:
            await asyncio.sleep(self.rate_limit_delay - elapsed)
        self.last_call = time.time()

    async def get_price_data(self, symbol: str) -> Optional[Dict[str, Any]]:
        """
        Get current price data from CryptoCompare
        """
        if not self.session:
            await self.connect()

        symbol_pair = self.symbol_map.get(symbol)
        if not symbol_pair:
            logger.warning(f"⚠️ Unknown CryptoCompare mapping for {symbol}")
            return None

        from_symbol, to_symbol = symbol_pair

        try:
            await self._rate_limit()

            # Get current price
            url = f"{self.base_url}/pricemultifull"
            params = {"fsyms": from_symbol, "tsyms": to_symbol}

            async with self.session.get(url, params=params) as response:
                if response.status == 200:
                    data = await response.json()

                    if (
                        "RAW" in data
                        and from_symbol in data["RAW"]
                        and to_symbol in data["RAW"][from_symbol]
                    ):
                        raw_data = data["RAW"][from_symbol][to_symbol]

                        return {
                            "symbol": symbol,
                            "price": raw_data["PRICE"],
                            "volume_24h": raw_data["VOLUME24HOUR"],
                            "change_24h": raw_data["CHANGEPCT24HOUR"],
                            "high_24h": raw_data["HIGH24HOUR"],
                            "low_24h": raw_data["LOW24HOUR"],
                            "market_cap": raw_data.get("MKTCAP", 0),
                            "source": "cryptocompare",
                            "timestamp": time.time(),
                            "exchange_volume": raw_data.get("TOTALVOLUME24H", 0),
                        }
                elif response.status == 429:
                    logger.warning("⚠️ CryptoCompare rate limit")
                    raise aiohttp.ClientResponseError(
                        request_info=response.request_info,
                        history=response.history,
                        status=429,
                    )

        except Exception as e:
            logger.error(f"❌ CryptoCompare error for {symbol}: {e}")
            raise

        return None

    async def get_supported_symbols(self) -> List[str]:
        """Get supported symbols"""
        return list(self.symbol_map.keys())
