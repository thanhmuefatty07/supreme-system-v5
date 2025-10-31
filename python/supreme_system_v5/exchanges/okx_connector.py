"""
OKX Exchange Connector
High-performance WebSocket implementation for futures scalping
"""

import asyncio
import base64
import hashlib
import hmac
import json
import time
from datetime import datetime
from typing import Any, Dict, List, Optional
from urllib.parse import urlencode

import aiohttp
import websockets
from loguru import logger

from .base import BaseExchange, ExchangeConfig, OrderResult


class OKXConnector(BaseExchange):
    """
    OKX exchange connector with WebSocket real-time data
    Optimized for futures scalping with minimal latency
    """

    def __init__(self, config: ExchangeConfig):
        super().__init__(config)

        # OKX endpoints
        self.base_url = (
            "https://www.okx.com" if not config.sandbox else "https://www.okx.com"
        )
        self.ws_url = "wss://ws.okx.com:8443/ws/v5/public"
        self.ws_private_url = "wss://ws.okx.com:8443/ws/v5/private"

        # Connection state
        self.websocket = None
        self.session = None
        self.heartbeat_task = None

    def _generate_signature(
        self, timestamp: str, method: str, request_path: str, body: str = ""
    ) -> str:
        """
        Generate OKX API signature
        """
        if not self.config.secret_key:
            raise ValueError("Secret key required for OKX authentication")

        message = timestamp + method + request_path + body
        signature = base64.b64encode(
            hmac.new(
                self.config.secret_key.encode(), message.encode(), hashlib.sha256
            ).digest()
        ).decode()
        return signature

    def _get_headers(
        self, method: str, request_path: str, body: str = ""
    ) -> Dict[str, str]:
        """
        Generate OKX API headers with authentication
        """
        timestamp = datetime.utcnow().isoformat()[:-3] + "Z"
        signature = self._generate_signature(timestamp, method, request_path, body)

        return {
            "OK-ACCESS-KEY": self.config.api_key,
            "OK-ACCESS-SIGN": signature,
            "OK-ACCESS-TIMESTAMP": timestamp,
            "OK-ACCESS-PASSPHRASE": self.config.passphrase,
            "Content-Type": "application/json",
        }

    async def connect(self) -> bool:
        """
        Connect to OKX WebSocket
        """
        try:
            logger.info("🔗 Connecting to OKX WebSocket...")

            # Create HTTP session
            self.session = aiohttp.ClientSession()

            # Connect to WebSocket
            self.websocket = await websockets.connect(
                self.ws_url,
                max_size=2**20,  # 1MB buffer
                max_queue=100,
                ping_interval=20,
                ping_timeout=10,
            )

            self.connected = True
            logger.info("✅ OKX WebSocket connected")

            # Start heartbeat
            self.heartbeat_task = asyncio.create_task(self._heartbeat())

            return True

        except Exception as e:
            logger.error(f"❌ OKX connection failed: {e}")
            return False

    async def disconnect(self):
        """
        Disconnect from OKX
        """
        logger.info("🔌 Disconnecting from OKX...")

        self.connected = False

        if self.heartbeat_task:
            self.heartbeat_task.cancel()

        if self.websocket:
            await self.websocket.close()

        if self.session:
            await self.session.close()

        logger.info("✅ OKX disconnected")

    async def _heartbeat(self):
        """
        Maintain WebSocket connection with heartbeat
        """
        while self.connected:
            try:
                if self.websocket:
                    await self.websocket.send("ping")
                await asyncio.sleep(20)
            except Exception as e:
                logger.error(f"Heartbeat failed: {e}")
                break

    async def subscribe_market_data(self, symbols: List[str]):
        """
        Subscribe to real-time ticker data
        """
        if not self.websocket:
            raise RuntimeError("Not connected to OKX WebSocket")

        # Convert symbols to OKX format
        okx_symbols = [symbol.replace("-", "-") for symbol in symbols]

        subscribe_msg = {
            "op": "subscribe",
            "args": [
                {"channel": "tickers", "instId": symbol} for symbol in okx_symbols
            ],
        }

        await self.websocket.send(json.dumps(subscribe_msg))
        logger.info(f"📈 Subscribed to market data: {okx_symbols}")

        # Start data handler
        asyncio.create_task(self._handle_market_data())

    async def _handle_market_data(self):
        """
        Handle incoming market data
        """
        try:
            async for message in self.websocket:
                data = json.loads(message)

                if "data" in data:
                    for ticker in data["data"]:
                        symbol = ticker.get("instId", "")
                        price = float(ticker.get("last", 0))
                        volume = float(ticker.get("vol24h", 0))
                        bid = float(ticker.get("bidPx", price * 0.9999))
                        ask = float(ticker.get("askPx", price * 1.0001))

                        # Call market data callback
                        if self.market_data_callback:
                            await self.market_data_callback(
                                symbol, price, volume, bid, ask
                            )

        except Exception as e:
            logger.error(f"Market data handler error: {e}")

    async def place_market_order(
        self, symbol: str, side: str, amount: float
    ) -> OrderResult:
        """
        Place market order via REST API
        """
        # Mock implementation for development
        logger.info(f"🔄 Mock market order: {side} {amount} {symbol}")

        return OrderResult(
            success=True,
            order_id=f"okx_{int(time.time())}",
            symbol=symbol,
            side=side,
            amount=amount,
            price=35000.0 if "BTC" in symbol else 1800.0,  # Mock prices
            fee=amount * 0.0005,  # 0.05% fee
            timestamp=time.time(),
        )

    async def place_limit_order(
        self, symbol: str, side: str, amount: float, price: float
    ) -> OrderResult:
        """
        Place limit order via REST API
        """
        # Mock implementation
        logger.info(f"🎯 Mock limit order: {side} {amount} {symbol} @ {price}")

        return OrderResult(
            success=True,
            order_id=f"okx_limit_{int(time.time())}",
            symbol=symbol,
            side=side,
            amount=amount,
            price=price,
            fee=amount * price * 0.0005,
            timestamp=time.time(),
        )

    async def cancel_order(self, order_id: str, symbol: str) -> bool:
        """
        Cancel order
        """
        logger.info(f"❌ Mock cancel order: {order_id}")
        return True

    async def get_balance(self) -> Dict[str, float]:
        """
        Get account balance
        """
        # Mock implementation
        return {"USDT": 10000.0, "BTC": 0.0, "ETH": 0.0}  # Mock balance

    async def get_positions(self) -> List[Dict[str, Any]]:
        """
        Get current positions
        """
        # Mock implementation
        return []
