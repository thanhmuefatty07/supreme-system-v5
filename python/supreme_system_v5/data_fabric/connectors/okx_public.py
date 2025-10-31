"""
OKX Public Connector - WebSocket and REST for real-time crypto data
Free tier: Unlimited public endpoints, 300 requests/min with API key
"""

import asyncio
import json
import time
from typing import Dict, Optional, List, Any, Callable

import aiohttp
import websockets
from loguru import logger

class OKXPublicConnector:
    """
    OKX Public API connector - WebSocket + REST for real-time crypto data
    Free tier: Unlimited public endpoints, 300 requests/min with API key
    """

    def __init__(self, api_key: Optional[str] = None, api_secret: Optional[str] = None):
        """Initialize OKX connector"""
        self.rest_url = "https://www.okx.com/api/v5"
        self.ws_url = "wss://ws.okx.com:8443/ws/v5/public"
        self.api_key = api_key
        self.api_secret = api_secret

        self.session: Optional[aiohttp.ClientSession] = None
        self.ws: Optional[websockets.WebSocketServerProtocol] = None
        self.ws_connected = False

        # Rate limiting (300 calls/min with API key, 10 calls/min without)
        self.rate_limit_delay = 0.2 if api_key else 6.0  # 300/min vs 10/min
        self.last_call = 0.0

        # WebSocket subscriptions
        self.subscriptions: Dict[str, List[Callable]] = {}
        self.ws_task: Optional[asyncio.Task] = None

        # Symbol mapping (OKX uses BTC-USDT format)
        self.symbol_map = {
            'BTC-USDT': 'BTC-USDT',
            'ETH-USDT': 'ETH-USDT',
            'BNB-USDT': 'BNB-USDT',
            'XRP-USDT': 'XRP-USDT',
            'ADA-USDT': 'ADA-USDT',
            'DOGE-USDT': 'DOGE-USDT',
            'SOL-USDT': 'SOL-USDT',
            'MATIC-USDT': 'MATIC-USDT',
            'DOT-USDT': 'DOT-USDT',
            'AVAX-USDT': 'AVAX-USDT'
        }

    async def connect(self):
        """Initialize HTTP session and WebSocket connection"""
        if not self.session:
            timeout = aiohttp.ClientTimeout(total=30, connect=10)
            headers = {
                'User-Agent': 'Supreme-System-V5/1.0',
                'Accept': 'application/json'
            }

            if self.api_key:
                headers.update({
                    'OK-ACCESS-KEY': self.api_key,
                    'OK-ACCESS-TIMESTAMP': str(int(time.time())),
                    'OK-ACCESS-PASSPHRASE': self.api_secret or '',
                })

            self.session = aiohttp.ClientSession(
                timeout=timeout,
                headers=headers
            )

            logger.info("✅ OKX REST connector initialized")

        # Connect WebSocket for real-time data
        await self._connect_websocket()
        logger.info("✅ OKX connector ready")

    async def disconnect(self):
        """Close HTTP session and WebSocket connection"""
        if self.ws_task:
            self.ws_task.cancel()
            try:
                await self.ws_task
            except asyncio.CancelledError:
                pass

        if self.ws and self.ws_connected:
            await self.ws.close()
            self.ws_connected = False

        if self.session:
            await self.session.close()
            self.session = None

        logger.info("✅ OKX connector disconnected")

    async def _connect_websocket(self):
        """Establish WebSocket connection"""
        try:
            self.ws = await websockets.connect(
                self.ws_url,
                extra_headers={
                    'User-Agent': 'Supreme-System-V5/1.0'
                },
                ping_interval=20,
                ping_timeout=10,
                close_timeout=5
            )
            self.ws_connected = True

            # Start WebSocket message handler
            self.ws_task = asyncio.create_task(self._handle_websocket_messages())

            logger.info("✅ OKX WebSocket connected")

        except Exception as e:
            logger.error(f"❌ OKX WebSocket connection failed: {e}")
            self.ws_connected = False

    async def _handle_websocket_messages(self):
        """Handle incoming WebSocket messages"""
        try:
            while self.ws_connected and self.ws:
                try:
                    message = await asyncio.wait_for(self.ws.recv(), timeout=30)
                    await self._process_websocket_message(message)
                except asyncio.TimeoutError:
                    # Send ping to keep connection alive
                    await self.ws.ping()
                except websockets.exceptions.ConnectionClosed:
                    logger.warning("⚠️ OKX WebSocket connection closed")
                    self.ws_connected = False
                    break

        except Exception as e:
            logger.error(f"❌ OKX WebSocket handler error: {e}")
            self.ws_connected = False

        # Attempt reconnection
        if not self.ws_connected:
            logger.info("🔄 Attempting OKX WebSocket reconnection...")
            await asyncio.sleep(5)
            await self._connect_websocket()

    async def _process_websocket_message(self, message: str):
        """Process incoming WebSocket message"""
        try:
            data = json.loads(message)

            if 'event' in data:
                event = data['event']
                if event == 'subscribe':
                    logger.debug(f"✅ OKX subscription confirmed: {data}")
                elif event == 'error':
                    logger.error(f"❌ OKX subscription error: {data}")
                return

            # Process market data
            if 'arg' in data and 'data' in data:
                channel = data['arg'].get('channel', '')
                symbol = data['arg'].get('instId', '')

                if symbol in self.subscriptions:
                    for callback in self.subscriptions[symbol]:
                        try:
                            await callback(data['data'], channel)
                        except Exception as e:
                            logger.error(f"❌ OKX callback error for {symbol}: {e}")

        except json.JSONDecodeError as e:
            logger.error(f"❌ OKX message parsing error: {e}")

    async def _enforce_rate_limit(self):
        """Enforce API rate limits"""
        current_time = time.time()
        elapsed = current_time - self.last_call

        if elapsed < self.rate_limit_delay:
            await asyncio.sleep(self.rate_limit_delay - elapsed)

        self.last_call = time.time()

    async def get_price_data(self, symbol: str) -> Optional[Dict[str, Any]]:
        """
        Get latest price data via REST API
        """
        await self._enforce_rate_limit()

        if symbol not in self.symbol_map:
            logger.warning(f"⚠️ Symbol {symbol} not mapped for OKX")
            return None

        okx_symbol = self.symbol_map[symbol]

        try:
            params = {
                'instId': okx_symbol,
                'sz': '1'  # Only latest trade
            }

            async with self.session.get(f"{self.rest_url}/market/trades", params=params) as response:
                if response.status == 200:
                    data = await response.json()
                    return self._parse_price_response(data, symbol)
                else:
                    logger.error(f"❌ OKX REST API error: {response.status}")
                    return None

        except Exception as e:
            logger.error(f"❌ OKX REST error for {symbol}: {e}")
            return None

    def _parse_price_response(self, data: Dict, symbol: str) -> Dict[str, Any]:
        """Parse OKX REST price response"""
        result = {
            'symbol': symbol,
            'source': 'okx',
            'timestamp': time.time()
        }

        try:
            if data.get('code') == '0' and 'data' in data:
                trades = data['data']
                if trades:
                    latest_trade = trades[0]  # Most recent trade
                    result.update({
                        'price': float(latest_trade.get('px', 0)),
                        'volume': float(latest_trade.get('sz', 0)),
                        'side': latest_trade.get('side', ''),
                        'timestamp': int(latest_trade.get('ts', 0)) / 1000,  # Convert ms to s
                        'trade_id': latest_trade.get('tradeId', ''),
                        'success': True
                    })

                    logger.debug(f"✅ OKX REST: {symbol} = ${result.get('price', 0):.4f}")
                    return result

            result['success'] = False
            result['error'] = data.get('msg', 'Unknown error')
            return result

        except (KeyError, ValueError, TypeError) as e:
            logger.error(f"❌ OKX REST parsing error: {e}")
            result['success'] = False
            result['error'] = str(e)
            return result

    async def subscribe_price_stream(self, symbol: str, callback: Callable):
        """
        Subscribe to real-time price updates via WebSocket
        """
        if symbol not in self.symbol_map:
            logger.warning(f"⚠️ Symbol {symbol} not mapped for OKX WebSocket")
            return

        okx_symbol = self.symbol_map[symbol]

        if symbol not in self.subscriptions:
            self.subscriptions[symbol] = []

        self.subscriptions[symbol].append(callback)

        if self.ws_connected and self.ws:
            # Send subscription message
            subscription_msg = {
                "op": "subscribe",
                "args": [{
                    "channel": "trades",
                    "instId": okx_symbol
                }]
            }

            try:
                await self.ws.send(json.dumps(subscription_msg))
                logger.info(f"✅ OKX WebSocket subscribed to {symbol}")
            except Exception as e:
                logger.error(f"❌ OKX WebSocket subscription failed for {symbol}: {e}")

    async def unsubscribe_price_stream(self, symbol: str, callback: Callable = None):
        """
        Unsubscribe from real-time price updates
        """
        if symbol not in self.subscriptions:
            return

        if callback:
            if callback in self.subscriptions[symbol]:
                self.subscriptions[symbol].remove(callback)
        else:
            self.subscriptions[symbol].clear()

        if not self.subscriptions[symbol]:
            del self.subscriptions[symbol]

            if self.ws_connected and self.ws:
                # Send unsubscribe message
                okx_symbol = self.symbol_map[symbol]
                unsubscribe_msg = {
                    "op": "unsubscribe",
                    "args": [{
                        "channel": "trades",
                        "instId": okx_symbol
                    }]
                }

                try:
                    await self.ws.send(json.dumps(unsubscribe_msg))
                    logger.info(f"✅ OKX WebSocket unsubscribed from {symbol}")
                except Exception as e:
                    logger.error(f"❌ OKX WebSocket unsubscription failed for {symbol}: {e}")

    async def get_orderbook(self, symbol: str, depth: int = 20) -> Optional[Dict[str, Any]]:
        """
        Get order book snapshot
        """
        await self._enforce_rate_limit()

        if symbol not in self.symbol_map:
            logger.warning(f"⚠️ Symbol {symbol} not mapped for OKX")
            return None

        okx_symbol = self.symbol_map[symbol]

        try:
            params = {
                'instId': okx_symbol,
                'sz': str(depth)
            }

            async with self.session.get(f"{self.rest_url}/market/books", params=params) as response:
                if response.status == 200:
                    data = await response.json()
                    return self._parse_orderbook_response(data, symbol)
                else:
                    logger.error(f"❌ OKX orderbook API error: {response.status}")
                    return None

        except Exception as e:
            logger.error(f"❌ OKX orderbook error for {symbol}: {e}")
            return None

    def _parse_orderbook_response(self, data: Dict, symbol: str) -> Dict[str, Any]:
        """Parse OKX orderbook response"""
        result = {
            'symbol': symbol,
            'source': 'okx',
            'timestamp': time.time()
        }

        try:
            if data.get('code') == '0' and 'data' in data:
                books = data['data']
                if books:
                    book = books[0]
                    timestamp = int(book.get('ts', 0)) / 1000

                    # Parse bids and asks
                    bids = []
                    asks = []

                    for bid in book.get('bids', []):
                        bids.append({
                            'price': float(bid[0]),
                            'volume': float(bid[1]),
                            'orders': int(bid[3]) if len(bid) > 3 else 1
                        })

                    for ask in book.get('asks', []):
                        asks.append({
                            'price': float(ask[0]),
                            'volume': float(ask[1]),
                            'orders': int(ask[3]) if len(ask) > 3 else 1
                        })

                    result.update({
                        'bids': bids,
                        'asks': asks,
                        'timestamp': timestamp,
                        'success': True
                    })

                    if bids and asks:
                        result.update({
                            'bid': bids[0]['price'],
                            'ask': asks[0]['price'],
                            'spread': asks[0]['price'] - bids[0]['price'],
                            'spread_bps': ((asks[0]['price'] - bids[0]['price']) / bids[0]['price']) * 10000
                        })

                    logger.debug(f"✅ OKX orderbook: {symbol} bid={result.get('bid', 0):.4f} ask={result.get('ask', 0):.4f}")
                    return result

            result['success'] = False
            result['error'] = data.get('msg', 'Unknown error')
            return result

        except (KeyError, ValueError, TypeError) as e:
            logger.error(f"❌ OKX orderbook parsing error: {e}")
            result['success'] = False
            result['error'] = str(e)
            return result

    async def get_24h_stats(self, symbol: str) -> Optional[Dict[str, Any]]:
        """
        Get 24-hour statistics
        """
        await self._enforce_rate_limit()

        if symbol not in self.symbol_map:
            logger.warning(f"⚠️ Symbol {symbol} not mapped for OKX")
            return None

        okx_symbol = self.symbol_map[symbol]

        try:
            params = {
                'instId': okx_symbol
            }

            async with self.session.get(f"{self.rest_url}/market/ticker", params=params) as response:
                if response.status == 200:
                    data = await response.json()
                    return self._parse_24h_response(data, symbol)
                else:
                    logger.error(f"❌ OKX 24h stats API error: {response.status}")
                    return None

        except Exception as e:
            logger.error(f"❌ OKX 24h stats error for {symbol}: {e}")
            return None

    def _parse_24h_response(self, data: Dict, symbol: str) -> Dict[str, Any]:
        """Parse OKX 24h statistics response"""
        result = {
            'symbol': symbol,
            'source': 'okx',
            'timestamp': time.time()
        }

        try:
            if data.get('code') == '0' and 'data' in data:
                stats = data['data']
                if stats:
                    stat = stats[0]

                    result.update({
                        'price': float(stat.get('last', 0)),
                        'high_24h': float(stat.get('high24h', 0)),
                        'low_24h': float(stat.get('low24h', 0)),
                        'open_24h': float(stat.get('open24h', 0)),
                        'volume_24h': float(stat.get('vol24h', 0)),
                        'change_24h': float(stat.get('change24h', 0)),
                        'change_percent_24h': float(stat.get('changePercent24h', '0').rstrip('%')),
                        'timestamp': int(stat.get('ts', 0)) / 1000,
                        'success': True
                    })

                    logger.debug(f"✅ OKX 24h: {symbol} price=${result.get('price', 0):.4f} change={result.get('change_percent_24h', 0):.2f}%")
                    return result

            result['success'] = False
            result['error'] = data.get('msg', 'Unknown error')
            return result

        except (KeyError, ValueError, TypeError) as e:
            logger.error(f"❌ OKX 24h parsing error: {e}")
            result['success'] = False
            result['error'] = str(e)
            return result
