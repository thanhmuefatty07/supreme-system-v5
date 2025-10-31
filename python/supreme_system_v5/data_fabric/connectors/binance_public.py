"""
Binance Public API Connector - High-frequency WebSocket data
Free unlimited WebSocket feeds for real-time market data
"""

import asyncio
import json
import time
from typing import Any, Callable, Dict, List, Optional

import aiohttp
import websockets
from loguru import logger


class BinancePublicConnector:
    """
    Binance Public API - Free unlimited WebSocket data
    Best for real-time price feeds and order book data
    """

    def __init__(self):
        """Initialize Binance public connector"""
        self.base_url = "https://api.binance.com/api/v3"
        self.ws_url = "wss://stream.binance.com:9443/ws"
        self.session: Optional[aiohttp.ClientSession] = None
        self.websocket: Optional[websockets.WebSocketServerProtocol] = None
        self.connected = False
        self.subscriptions: Dict[str, Callable] = {}

        # Symbol mapping
        self.symbol_map = {
            "BTC-USDT": "BTCUSDT",
            "ETH-USDT": "ETHUSDT",
            "BNB-USDT": "BNBUSDT",
            "XRP-USDT": "XRPUSDT",
            "ADA-USDT": "ADAUSDT",
            "DOGE-USDT": "DOGEUSDT",
            "SOL-USDT": "SOLUSDT",
            "MATIC-USDT": "MATICUSDT",
            "DOT-USDT": "DOTUSDT",
            "AVAX-USDT": "AVAXUSDT",
        }

    async def connect(self):
        """Connect to Binance WebSocket"""
        try:
            logger.info("🔗 Connecting to Binance WebSocket...")

            # HTTP session for REST calls
            if not self.session:
                timeout = aiohttp.ClientTimeout(total=30)
                self.session = aiohttp.ClientSession(
                    timeout=timeout, headers={"User-Agent": "Supreme-System-V5/1.0"}
                )

            # WebSocket connection
            self.websocket = await websockets.connect(
                self.ws_url,
                max_size=2**20,
                max_queue=200,
                ping_interval=20,
                ping_timeout=10,
            )

            self.connected = True
            logger.info("✅ Binance WebSocket connected")

            # Start message handler
            asyncio.create_task(self._handle_messages())

        except Exception as e:
            logger.error(f"❌ Binance connection failed: {e}")
            raise

    async def disconnect(self):
        """Disconnect from Binance"""
        logger.info("🔌 Disconnecting from Binance...")

        self.connected = False

        if self.websocket:
            await self.websocket.close()
            self.websocket = None

        if self.session:
            await self.session.close()
            self.session = None

        logger.info("✅ Binance disconnected")

    async def get_price_data(self, symbol: str) -> Optional[Dict[str, Any]]:
        """
        Get current price via REST API (for initial data)
        """
        if not self.session:
            await self.connect()

        binance_symbol = self.symbol_map.get(symbol)
        if not binance_symbol:
            return None

        try:
            url = f"{self.base_url}/ticker/24hr"
            params = {"symbol": binance_symbol}

            async with self.session.get(url, params=params) as response:
                if response.status == 200:
                    data = await response.json()

                    return {
                        "symbol": symbol,
                        "price": float(data["lastPrice"]),
                        "volume_24h": float(data["volume"]),
                        "change_24h": float(data["priceChangePercent"]),
                        "high_24h": float(data["highPrice"]),
                        "low_24h": float(data["lowPrice"]),
                        "bid": float(data["bidPrice"]),
                        "ask": float(data["askPrice"]),
                        "source": "binance",
                        "timestamp": time.time(),
                    }

        except Exception as e:
            logger.error(f"❌ Binance REST error for {symbol}: {e}")
            raise

        return None

    async def subscribe_ticker(self, symbol: str, callback: Callable):
        """
        Subscribe to real-time ticker updates via WebSocket
        """
        if not self.websocket:
            await self.connect()

        binance_symbol = self.symbol_map.get(symbol)
        if not binance_symbol:
            logger.error(f"❌ Unknown Binance symbol: {symbol}")
            return

        # Subscribe to ticker stream
        stream_name = f"{binance_symbol.lower()}@ticker"

        subscribe_msg = {
            "method": "SUBSCRIBE",
            "params": [stream_name],
            "id": int(time.time()),
        }

        await self.websocket.send(json.dumps(subscribe_msg))
        self.subscriptions[stream_name] = callback

        logger.info(f"📡 Subscribed to Binance {symbol} ticker")

    async def _handle_messages(self):
        """
        Handle incoming WebSocket messages
        """
        logger.info("👂 Binance message handler started")

        try:
            async for message in self.websocket:
                data = json.loads(message)

                # Handle ticker updates
                if "s" in data and "c" in data:  # Ticker data
                    binance_symbol = data["s"]
                    stream_name = f"{binance_symbol.lower()}@ticker"

                    if stream_name in self.subscriptions:
                        # Convert to standard format
                        symbol = next(
                            (
                                k
                                for k, v in self.symbol_map.items()
                                if v == binance_symbol
                            ),
                            binance_symbol,
                        )

                        market_data = {
                            "symbol": symbol,
                            "price": float(data["c"]),  # Last price
                            "volume_24h": float(data["v"]),  # 24h volume
                            "change_24h": float(data["P"]),  # 24h change %
                            "high_24h": float(data["h"]),
                            "low_24h": float(data["l"]),
                            "bid": float(data.get("b", data["c"])),
                            "ask": float(data.get("a", data["c"])),
                            "source": "binance",
                            "timestamp": time.time(),
                        }

                        # Call registered callback
                        try:
                            await self.subscriptions[stream_name](market_data)
                        except Exception as e:
                            logger.error(f"Callback error: {e}")

        except websockets.exceptions.ConnectionClosed:
            logger.warning("⚠️ Binance WebSocket connection closed")
            self.connected = False
        except Exception as e:
            logger.error(f"❌ Binance message handler error: {e}")
            self.connected = False

    async def get_supported_symbols(self) -> List[str]:
        """Get supported symbols"""
        return list(self.symbol_map.keys())
