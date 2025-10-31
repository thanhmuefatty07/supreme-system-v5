"""
OKX Public API Connector - Real-time WebSocket feeds
Unlimited public endpoints with high-frequency market data
Optimized for futures scalping with minimal latency
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
    OKX Public API connector - Real-time market data
    Free unlimited public WebSocket streams
    Optimized for high-frequency trading
    """
    
    def __init__(self):
        """Initialize OKX public connector"""
        self.base_url = "https://www.okx.com/api/v5"
        self.ws_url = "wss://ws.okx.com:8443/ws/v5/public"
        self.session: Optional[aiohttp.ClientSession] = None
        self.websocket: Optional[websockets.WebSocketServerProtocol] = None
        self.connected = False
        self.subscriptions: Dict[str, Callable] = {}
        self.heartbeat_task: Optional[asyncio.Task] = None
        
        # Symbol mapping to OKX format
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
        """Connect to OKX public WebSocket"""
        try:
            logger.info("🔗 Connecting to OKX public WebSocket...")
            
            # HTTP session for REST calls
            if not self.session:
                timeout = aiohttp.ClientTimeout(total=30, connect=10)
                self.session = aiohttp.ClientSession(
                    timeout=timeout,
                    headers={
                        'User-Agent': 'Supreme-System-V5/1.0',
                        'Accept': 'application/json'
                    }
                )
            
            # WebSocket connection with optimized settings
            self.websocket = await websockets.connect(
                self.ws_url,
                max_size=2**20,  # 1MB buffer
                max_queue=500,   # High-frequency queue
                ping_interval=20,
                ping_timeout=10,
                close_timeout=10
            )
            
            self.connected = True
            logger.info("✅ OKX public WebSocket connected")
            
            # Start message handler and heartbeat
            asyncio.create_task(self._handle_messages())
            self.heartbeat_task = asyncio.create_task(self._heartbeat())
            
        except Exception as e:
            logger.error(f"❌ OKX connection failed: {e}")
            self.connected = False
            raise
    
    async def disconnect(self):
        """Disconnect from OKX WebSocket"""
        logger.info("🔌 Disconnecting from OKX...")
        
        self.connected = False
        
        # Cancel heartbeat
        if self.heartbeat_task:
            self.heartbeat_task.cancel()
            try:
                await self.heartbeat_task
            except asyncio.CancelledError:
                pass
        
        # Close WebSocket
        if self.websocket:
            try:
                await self.websocket.close()
            except Exception as e:
                logger.warning(f"WebSocket close error: {e}")
            self.websocket = None
            
        # Close HTTP session
        if self.session:
            await self.session.close()
            self.session = None
            
        logger.info("✅ OKX disconnected")
    
    async def _heartbeat(self):
        """Maintain WebSocket connection with heartbeat"""
        while self.connected and self.websocket:
            try:
                # OKX uses ping frames for heartbeat
                await self.websocket.ping()
                await asyncio.sleep(20)
            except Exception as e:
                logger.error(f"❌ OKX heartbeat failed: {e}")
                self.connected = False
                break
    
    async def get_price_data(self, symbol: str) -> Optional[Dict[str, Any]]:
        """
        Get current price data via REST API (for initial data)
        """
        if not self.session:
            await self.connect()
        
        okx_symbol = self.symbol_map.get(symbol)
        if not okx_symbol:
            logger.warning(f"⚠️ Unknown OKX symbol: {symbol}")
            return None
        
        try:
            url = f"{self.base_url}/market/ticker"
            params = {'instId': okx_symbol}
            
            async with self.session.get(url, params=params) as response:
                if response.status == 200:
                    data = await response.json()
                    
                    if 'data' in data and data['data']:
                        ticker = data['data'][0]
                        
                        return {
                            'symbol': symbol,
                            'price': float(ticker['last']),
                            'volume_24h': float(ticker['vol24h']),
                            'change_24h': float(ticker['change24h']) * 100,  # Convert to percentage
                            'high_24h': float(ticker['high24h']),
                            'low_24h': float(ticker['low24h']),
                            'bid': float(ticker['bidPx']) if ticker['bidPx'] else 0.0,
                            'ask': float(ticker['askPx']) if ticker['askPx'] else 0.0,
                            'open_24h': float(ticker['open24h']),
                            'source': 'okx_public',
                            'timestamp': time.time()
                        }
                else:
                    logger.error(f"❌ OKX REST API error: {response.status}")
                    return None
                    
        except Exception as e:
            logger.error(f"❌ OKX REST error for {symbol}: {e}")
            raise
        
        return None
    
    async def subscribe_ticker(self, symbol: str, callback: Callable):
        """
        Subscribe to real-time ticker updates via WebSocket
        """
        if not self.websocket or not self.connected:
            await self.connect()
            
        okx_symbol = self.symbol_map.get(symbol)
        if not okx_symbol:
            logger.error(f"❌ Unknown OKX symbol: {symbol}")
            return
        
        # Subscribe to tickers channel
        subscribe_msg = {
            "op": "subscribe",
            "args": [
                {
                    "channel": "tickers",
                    "instId": okx_symbol
                }
            ]
        }
        
        await self.websocket.send(json.dumps(subscribe_msg))
        
        # Register callback
        callback_key = f"tickers:{okx_symbol}"
        self.subscriptions[callback_key] = callback
        
        logger.info(f"📡 Subscribed to OKX ticker: {symbol}")
    
    async def subscribe_orderbook(self, symbol: str, callback: Callable, depth: str = "5"):
        """
        Subscribe to order book updates (for spread analysis)
        """
        if not self.websocket or not self.connected:
            await self.connect()
            
        okx_symbol = self.symbol_map.get(symbol)
        if not okx_symbol:
            return
        
        subscribe_msg = {
            "op": "subscribe",
            "args": [
                {
                    "channel": "books",
                    "instId": okx_symbol
                }
            ]
        }
        
        await self.websocket.send(json.dumps(subscribe_msg))
        
        callback_key = f"books:{okx_symbol}"
        self.subscriptions[callback_key] = callback
        
        logger.info(f"📊 Subscribed to OKX orderbook: {symbol}")
    
    async def _handle_messages(self):
        """
        Handle incoming WebSocket messages with error recovery
        """
        logger.info("👂 OKX message handler started")
        
        try:
            async for message in self.websocket:
                try:
                    data = json.loads(message)
                    
                    # Handle subscription confirmations
                    if data.get('event') == 'subscribe':
                        logger.info(f"✅ OKX subscription confirmed: {data.get('arg', {})}")
                        continue
                    
                    # Handle data messages
                    if 'data' in data and 'arg' in data:
                        channel = data['arg'].get('channel')
                        inst_id = data['arg'].get('instId')
                        callback_key = f"{channel}:{inst_id}"
                        
                        if callback_key in self.subscriptions:
                            # Convert symbol back to our format
                            symbol = next(
                                (k for k, v in self.symbol_map.items() if v == inst_id),
                                inst_id
                            )
                            
                            # Process based on channel type
                            if channel == 'tickers':
                                await self._process_ticker_data(symbol, data['data'][0])
                            elif channel == 'books':
                                await self._process_orderbook_data(symbol, data['data'][0])
                            
                            # Call registered callback
                            await self.subscriptions[callback_key](data['data'][0])
                            
                except json.JSONDecodeError as e:
                    logger.warning(f"⚠️ OKX invalid JSON: {e}")
                except Exception as e:
                    logger.error(f"❌ OKX message processing error: {e}")
                    
        except websockets.exceptions.ConnectionClosed:
            logger.warning("⚠️ OKX WebSocket connection closed")
            self.connected = False
        except Exception as e:
            logger.error(f"❌ OKX message handler error: {e}")
            self.connected = False
    
    async def _process_ticker_data(self, symbol: str, ticker_data: Dict):
        """
        Process ticker data and emit normalized format
        """
        try:
            normalized_data = {
                'symbol': symbol,
                'price': float(ticker_data['last']),
                'volume_24h': float(ticker_data['vol24h']),
                'change_24h': float(ticker_data['change24h']) * 100,
                'high_24h': float(ticker_data['high24h']),
                'low_24h': float(ticker_data['low24h']),
                'bid': float(ticker_data['bidPx']) if ticker_data['bidPx'] else 0.0,
                'ask': float(ticker_data['askPx']) if ticker_data['askPx'] else 0.0,
                'open_24h': float(ticker_data['open24h']),
                'source': 'okx_public',
                'timestamp': time.time(),
                'ws_timestamp': int(ticker_data['ts']) / 1000  # OKX timestamp in ms
            }
            
            logger.debug(f"📈 OKX {symbol}: ${normalized_data['price']:.4f} vol={normalized_data['volume_24h']:.0f}")
            
        except Exception as e:
            logger.error(f"❌ OKX ticker processing error: {e}")
    
    async def _process_orderbook_data(self, symbol: str, book_data: Dict):
        """
        Process order book data for spread analysis
        """
        try:
            bids = book_data.get('bids', [])
            asks = book_data.get('asks', [])
            
            if bids and asks:
                best_bid = float(bids[0][0]) if bids[0] else 0.0
                best_ask = float(asks[0][0]) if asks[0] else 0.0
                
                spread = abs(best_ask - best_bid)
                spread_bps = (spread / best_bid * 10000) if best_bid > 0 else 0
                
                logger.debug(f"📊 OKX {symbol} spread: {spread:.4f} ({spread_bps:.1f} bps)")
                
        except Exception as e:
            logger.error(f"❌ OKX orderbook processing error: {e}")
    
    async def get_supported_symbols(self) -> List[str]:
        """Get supported trading symbols"""
        return list(self.symbol_map.keys())
