#!/usr/bin/env python3
"""
Supreme System V5 - MEXC Exchange Connector
Ultra-optimized connector for MEXC Global exchange
Agent Mode: Production-ready with error handling and rate limiting

Features:
- WebSocket real-time data streams
- REST API for order execution
- Built-in rate limiting and error recovery
- Ultra-constrained memory profile
- Comprehensive logging and monitoring
"""

import asyncio
import json
import time
from typing import Dict, List, Optional, Any, Callable
from datetime import datetime, timezone
from dataclasses import dataclass, field

try:
    import ccxt
    import websockets
    import aiohttp
    from loguru import logger
except ImportError as e:
    print(f"⚠️ Missing dependencies for MEXC connector: {e}")
    print("Install with: pip install ccxt websockets aiohttp loguru")
    raise

from .base import BaseExchangeConnector
from ..data_fabric.quality import DataQualityScorer
from ..optimized.circular_buffer import CircularBuffer


@dataclass
class MEXCMarketData:
    """MEXC-specific market data structure"""
    symbol: str
    price: float
    volume_24h: float
    change_24h: float
    timestamp: float
    bid: Optional[float] = None
    ask: Optional[float] = None
    high_24h: Optional[float] = None
    low_24h: Optional[float] = None
    volume_quote: Optional[float] = None
    count: Optional[int] = None  # Number of trades
    
    def to_standard_format(self) -> Dict[str, Any]:
        """Convert to standard market data format"""
        return {
            'symbol': self.symbol,
            'price': self.price,
            'volume': self.volume_24h,
            'timestamp': self.timestamp,
            'bid': self.bid,
            'ask': self.ask,
            'high_24h': self.high_24h,
            'low_24h': self.low_24h,
            'change_24h': self.change_24h,
            'source': 'mexc',
            'quality_score': self._calculate_quality()
        }
        
    def _calculate_quality(self) -> float:
        """Calculate data quality score for this data point"""
        score = 0.8  # Base score for MEXC
        
        # Price validity
        if self.price > 0:
            score += 0.1
            
        # Volume presence
        if self.volume_24h > 0:
            score += 0.05
            
        # Bid/ask spread available
        if self.bid and self.ask and self.bid > 0 and self.ask > 0:
            spread = (self.ask - self.bid) / self.bid
            if spread < 0.01:  # <1% spread is good
                score += 0.05
                
        return min(1.0, score)


class MEXCConnector(BaseExchangeConnector):
    """Ultra-optimized MEXC exchange connector"""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize MEXC connector with ultra-constrained optimization
        
        Args:
            config: Configuration dictionary with MEXC settings
        """
        super().__init__(config)
        
        self.exchange_name = "MEXC"
        self.config = config
        
        # MEXC API configuration
        self.api_key = config.get('mexc_api_key', '')
        self.api_secret = config.get('mexc_api_secret', '')
        self.passphrase = config.get('mexc_passphrase', '')  # MEXC may require passphrase
        self.sandbox_mode = config.get('mexc_sandbox', True)  # Default to sandbox
        
        # Connection settings
        self.base_url = config.get('mexc_base_url', 'https://api.mexc.com')
        self.ws_url = config.get('mexc_ws_url', 'wss://wbs.mexc.com/ws')
        
        # Rate limiting (MEXC specific limits)
        self.rate_limit_per_second = config.get('mexc_rate_limit', 10)  # Conservative
        self.last_request_times = CircularBuffer(size=100)  # Track recent requests
        
        # Ultra-constrained settings
        self.max_symbols = config.get('max_symbols', 3)  # Limit concurrent symbols
        self.buffer_size = config.get('buffer_size', 200)
        self.reconnect_delay = config.get('reconnect_delay', 5)
        self.max_reconnect_attempts = config.get('max_reconnect_attempts', 5)
        
        # State management
        self.connected = False
        self.ws_connection = None
        self.session = None
        self.subscribed_symbols = set()
        self.data_buffer = CircularBuffer(size=self.buffer_size)
        
        # Callbacks
        self.on_tick_callback: Optional[Callable] = None
        self.on_error_callback: Optional[Callable] = None
        
        # Quality scoring
        self.quality_scorer = DataQualityScorer({
            'min_quality_threshold': 0.7,
            'price_validation': True,
            'volume_validation': True,
            'timestamp_validation': True
        })
        
        # Initialize CCXT client (for REST API)
        try:
            self.ccxt_client = ccxt.mexc({
                'apiKey': self.api_key,
                'secret': self.api_secret,
                'password': self.passphrase,
                'sandbox': self.sandbox_mode,
                'enableRateLimit': True,
                'rateLimit': 1000 // self.rate_limit_per_second,  # ms between requests
                'options': {
                    'adjustForTimeDifference': True,
                    'recvWindow': 10000,
                }
            })
            logger.info(f"✅ MEXC CCXT client initialized (sandbox: {self.sandbox_mode})")
        except Exception as e:
            logger.error(f"❌ Failed to initialize MEXC CCXT client: {e}")
            self.ccxt_client = None
            
        logger.info(f"🚀 MEXCConnector initialized - Ultra-constrained mode")
        
    async def connect(self) -> bool:
        """Establish connection to MEXC"""
        try:
            # Test REST API connection
            if self.ccxt_client:
                await asyncio.get_event_loop().run_in_executor(
                    None, self.ccxt_client.load_markets
                )
                logger.info("✅ MEXC REST API connection established")
                
            # Initialize aiohttp session for WebSocket
            self.session = aiohttp.ClientSession()
            
            self.connected = True
            logger.info("✅ MEXC connector connected")
            return True
            
        except Exception as e:
            logger.error(f"❌ MEXC connection failed: {e}")
            self.connected = False
            return False
            
    async def disconnect(self):
        """Disconnect from MEXC"""
        try:
            if self.ws_connection:
                await self.ws_connection.close()
                self.ws_connection = None
                
            if self.session:
                await self.session.close()
                self.session = None
                
            self.connected = False
            self.subscribed_symbols.clear()
            
            logger.info("✅ MEXC connector disconnected")
            
        except Exception as e:
            logger.error(f"❌ MEXC disconnect error: {e}")
            
    async def subscribe_symbol(self, symbol: str) -> bool:
        """Subscribe to real-time data for a symbol"""
        if symbol in self.subscribed_symbols:
            logger.debug(f"📊 Already subscribed to {symbol}")
            return True
            
        if len(self.subscribed_symbols) >= self.max_symbols:
            logger.warning(f"⚠️ Max symbols ({self.max_symbols}) reached, cannot subscribe to {symbol}")
            return False
            
        try:
            # Convert symbol to MEXC format (e.g., ETH-USDT -> ETHUSDT)
            mexc_symbol = symbol.replace('-', '')
            
            # Subscribe via WebSocket
            success = await self._subscribe_websocket(mexc_symbol)
            
            if success:
                self.subscribed_symbols.add(symbol)
                logger.info(f"✅ Subscribed to {symbol} on MEXC")
                return True
            else:
                logger.error(f"❌ Failed to subscribe to {symbol} on MEXC")
                return False
                
        except Exception as e:
            logger.error(f"❌ MEXC subscription error for {symbol}: {e}")
            return False
            
    async def _subscribe_websocket(self, mexc_symbol: str) -> bool:
        """Subscribe to MEXC WebSocket for real-time data"""
        try:
            if not self.ws_connection:
                # Connect to MEXC WebSocket
                self.ws_connection = await websockets.connect(
                    self.ws_url,
                    ping_interval=20,
                    ping_timeout=10,
                    close_timeout=10
                )
                
                # Start WebSocket message handler
                asyncio.create_task(self._handle_websocket_messages())
                
            # Subscribe to ticker data
            subscribe_message = {
                "method": "SUBSCRIPTION",
                "params": [f"spot@public.bookTicker.v3.api@{mexc_symbol}"],
                "id": int(time.time())
            }
            
            await self.ws_connection.send(json.dumps(subscribe_message))
            
            # Also subscribe to 24hr ticker for additional data
            ticker_subscribe = {
                "method": "SUBSCRIPTION",
                "params": [f"spot@public.deals.v3.api@{mexc_symbol}"],
                "id": int(time.time()) + 1
            }
            
            await self.ws_connection.send(json.dumps(ticker_subscribe))
            
            logger.info(f"📡 WebSocket subscription sent for {mexc_symbol}")
            return True
            
        except Exception as e:
            logger.error(f"❌ WebSocket subscription failed for {mexc_symbol}: {e}")
            return False
            
    async def _handle_websocket_messages(self):
        """Handle incoming WebSocket messages from MEXC"""
        try:
            async for message in self.ws_connection:
                try:
                    data = json.loads(message)
                    await self._process_websocket_data(data)
                    
                except json.JSONDecodeError as e:
                    logger.warning(f"⚠️ Invalid JSON from MEXC WebSocket: {e}")
                except Exception as e:
                    logger.error(f"❌ Error processing MEXC WebSocket message: {e}")
                    
        except websockets.exceptions.ConnectionClosed:
            logger.warning("⚠️ MEXC WebSocket connection closed")
            await self._handle_reconnection()
        except Exception as e:
            logger.error(f"❌ MEXC WebSocket handler error: {e}")
            await self._handle_reconnection()
            
    async def _process_websocket_data(self, data: Dict[str, Any]):
        """Process WebSocket data and convert to standard format"""
        try:
            # Handle MEXC WebSocket data format
            if 'c' in data and 'd' in data:  # MEXC ticker format
                channel = data['c']
                ticker_data = data['d']
                
                if 'bookTicker' in channel:
                    # Process book ticker (bid/ask prices)
                    await self._process_book_ticker(ticker_data, channel)
                elif 'deals' in channel:
                    # Process trade data
                    await self._process_trade_data(ticker_data, channel)
                    
        except Exception as e:
            logger.error(f"❌ Error processing MEXC WebSocket data: {e}")
            
    async def _process_book_ticker(self, data: Dict[str, Any], channel: str):
        """Process MEXC book ticker data"""
        try:
            # Extract symbol from channel
            symbol_part = channel.split('@')[-1]
            mexc_symbol = symbol_part.upper()
            
            # Convert back to standard format
            if mexc_symbol.endswith('USDT'):
                standard_symbol = f"{mexc_symbol[:-4]}-USDT"
            else:
                standard_symbol = mexc_symbol  # Fallback
                
            # Create market data object
            market_data = MEXCMarketData(
                symbol=standard_symbol,
                price=(float(data.get('a', 0)) + float(data.get('b', 0))) / 2,  # Mid price
                volume_24h=0.0,  # Will be updated from trade data
                change_24h=0.0,  # Will be calculated
                timestamp=time.time(),
                bid=float(data.get('b', 0)),
                ask=float(data.get('a', 0)),
                volume_quote=float(data.get('B', 0)) if 'B' in data else None
            )
            
            # Store in buffer
            self.data_buffer.append(market_data)
            
            # Trigger callback if available
            if self.on_tick_callback:
                await self.on_tick_callback(market_data.to_standard_format())
                
        except Exception as e:
            logger.error(f"❌ Error processing MEXC book ticker: {e}")
            
    async def _process_trade_data(self, data: Dict[str, Any], channel: str):
        """Process MEXC trade data for volume and price confirmation"""
        try:
            # Extract symbol and update volume information
            # This is complementary to book ticker data
            
            if isinstance(data, list) and len(data) > 0:
                latest_trade = data[0] if isinstance(data[0], dict) else {}
                
                # Update volume information if available
                if 'v' in latest_trade:  # Volume
                    volume = float(latest_trade['v'])
                    
                    # Update latest market data in buffer if exists
                    if len(self.data_buffer) > 0:
                        latest_data = self.data_buffer[-1]
                        if hasattr(latest_data, 'volume_24h'):
                            latest_data.volume_24h = volume
                            
        except Exception as e:
            logger.error(f"❌ Error processing MEXC trade data: {e}")
            
    async def _handle_reconnection(self):
        """Handle WebSocket reconnection with exponential backoff"""
        for attempt in range(self.max_reconnect_attempts):
            try:
                delay = min(self.reconnect_delay * (2 ** attempt), 60)  # Max 60s delay
                logger.info(f"🔄 Reconnecting to MEXC WebSocket in {delay}s (attempt {attempt + 1}/{self.max_reconnect_attempts})")
                
                await asyncio.sleep(delay)
                
                # Reconnect
                self.ws_connection = await websockets.connect(
                    self.ws_url,
                    ping_interval=20,
                    ping_timeout=10,
                    close_timeout=10
                )
                
                # Re-subscribe to all symbols
                for symbol in list(self.subscribed_symbols):
                    mexc_symbol = symbol.replace('-', '')
                    await self._subscribe_websocket(mexc_symbol)
                    
                logger.info(f"✅ MEXC WebSocket reconnected successfully")
                
                # Restart message handler
                asyncio.create_task(self._handle_websocket_messages())
                return
                
            except Exception as e:
                logger.error(f"❌ MEXC reconnection attempt {attempt + 1} failed: {e}")
                
        # All reconnection attempts failed
        logger.error(f"❌ MEXC WebSocket reconnection failed after {self.max_reconnect_attempts} attempts")
        self.connected = False
        
        if self.on_error_callback:
            await self.on_error_callback("mexc_websocket_disconnected")
            
    async def get_ticker(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Get latest ticker data for symbol using REST API"""
        if not self.ccxt_client:
            logger.error("❌ MEXC CCXT client not available")
            return None
            
        try:
            # Rate limiting check
            if not self._check_rate_limit():
                logger.warning("⚠️ MEXC rate limit exceeded, skipping request")
                return None
                
            # Get ticker data
            mexc_symbol = symbol.replace('-', '/')  # CCXT format
            ticker = await asyncio.get_event_loop().run_in_executor(
                None, self.ccxt_client.fetch_ticker, mexc_symbol
            )
            
            # Convert to standard format
            market_data = MEXCMarketData(
                symbol=symbol,
                price=float(ticker['last']),
                volume_24h=float(ticker['baseVolume']),
                change_24h=float(ticker['percentage'] or 0),
                timestamp=ticker['timestamp'] / 1000 if ticker['timestamp'] else time.time(),
                bid=float(ticker['bid']) if ticker['bid'] else None,
                ask=float(ticker['ask']) if ticker['ask'] else None,
                high_24h=float(ticker['high']) if ticker['high'] else None,
                low_24h=float(ticker['low']) if ticker['low'] else None
            )
            
            return market_data.to_standard_format()
            
        except Exception as e:
            logger.error(f"❌ MEXC ticker fetch error for {symbol}: {e}")
            return None
            
    def _check_rate_limit(self) -> bool:
        """Check if we can make a request without exceeding rate limits"""
        current_time = time.time()
        
        # Add current request time
        self.last_request_times.append(current_time)
        
        # Count requests in last second
        recent_requests = sum(
            1 for req_time in self.last_request_times 
            if current_time - req_time <= 1.0
        )
        
        return recent_requests <= self.rate_limit_per_second
        
    async def place_order(self, symbol: str, side: str, amount: float, price: Optional[float] = None) -> Optional[Dict[str, Any]]:
        """Place order on MEXC exchange"""
        if not self.ccxt_client:
            logger.error("❌ MEXC CCXT client not available for order placement")
            return None
            
        if not self.api_key or not self.api_secret:
            logger.error("❌ MEXC API credentials not configured")
            return None
            
        try:
            # Rate limiting
            if not self._check_rate_limit():
                logger.warning("⚠️ MEXC rate limit exceeded, cannot place order")
                return None
                
            # Convert symbol format
            mexc_symbol = symbol.replace('-', '/')
            
            # Determine order type
            if price is not None:
                order_type = 'limit'
            else:
                order_type = 'market'
                
            # Place order
            order_result = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: self.ccxt_client.create_order(
                    mexc_symbol, order_type, side.lower(), amount, price
                )
            )
            
            logger.info(f"✅ MEXC order placed: {side} {amount} {symbol} at {price or 'market'}")
            
            return {
                'order_id': order_result['id'],
                'symbol': symbol,
                'side': side,
                'amount': amount,
                'price': price,
                'type': order_type,
                'status': order_result.get('status', 'unknown'),
                'timestamp': time.time(),
                'exchange': 'mexc',
                'raw_response': order_result
            }
            
        except Exception as e:
            logger.error(f"❌ MEXC order placement failed: {e}")
            return None
            
    async def get_balance(self) -> Optional[Dict[str, Any]]:
        """Get account balance from MEXC"""
        if not self.ccxt_client:
            logger.error("❌ MEXC CCXT client not available")
            return None
            
        try:
            if not self._check_rate_limit():
                return None
                
            balance = await asyncio.get_event_loop().run_in_executor(
                None, self.ccxt_client.fetch_balance
            )
            
            # Convert to standard format
            return {
                'total_usd': balance.get('USDT', {}).get('total', 0.0),
                'available_usd': balance.get('USDT', {}).get('free', 0.0),
                'balances': {
                    asset: {
                        'total': info.get('total', 0.0),
                        'available': info.get('free', 0.0),
                        'locked': info.get('used', 0.0)
                    }
                    for asset, info in balance.items()
                    if isinstance(info, dict) and info.get('total', 0) > 0
                },
                'timestamp': time.time(),
                'exchange': 'mexc'
            }
            
        except Exception as e:
            logger.error(f"❌ MEXC balance fetch error: {e}")
            return None
            
    async def get_order_status(self, order_id: str, symbol: str) -> Optional[Dict[str, Any]]:
        """Get order status from MEXC"""
        if not self.ccxt_client:
            return None
            
        try:
            if not self._check_rate_limit():
                return None
                
            mexc_symbol = symbol.replace('-', '/')
            order = await asyncio.get_event_loop().run_in_executor(
                None, self.ccxt_client.fetch_order, order_id, mexc_symbol
            )
            
            return {
                'order_id': order['id'],
                'symbol': symbol,
                'status': order['status'],
                'side': order['side'],
                'amount': order['amount'],
                'filled': order['filled'],
                'price': order['price'],
                'average': order['average'],
                'timestamp': order['timestamp'] / 1000 if order['timestamp'] else None,
                'exchange': 'mexc'
            }
            
        except Exception as e:
            logger.error(f"❌ MEXC order status error: {e}")
            return None
            
    def set_callbacks(self, on_tick: Optional[Callable] = None, on_error: Optional[Callable] = None):
        """Set callback functions for real-time data and errors"""
        self.on_tick_callback = on_tick
        self.on_error_callback = on_error
        logger.info("✅ MEXC callbacks configured")
        
    def get_connection_status(self) -> Dict[str, Any]:
        """Get current connection status and statistics"""
        return {
            'connected': self.connected,
            'exchange': 'mexc',
            'subscribed_symbols': list(self.subscribed_symbols),
            'ws_connected': self.ws_connection is not None,
            'rest_available': self.ccxt_client is not None,
            'buffer_size': len(self.data_buffer),
            'buffer_max': self.buffer_size,
            'rate_limit_per_second': self.rate_limit_per_second,
            'sandbox_mode': self.sandbox_mode,
            'last_activity': time.time()
        }
        
    async def start_monitoring(self) -> Dict[str, Any]:
        """Start monitoring and return status"""
        try:
            await self.connect()
            
            # Subscribe to default symbols if configured
            default_symbols = self.config.get('default_symbols', ['ETH-USDT'])
            for symbol in default_symbols:
                await self.subscribe_symbol(symbol)
                
            return {
                'status': 'running',
                'connected': self.connected,
                'subscribed_count': len(self.subscribed_symbols),
                'monitoring_started': True
            }
            
        except Exception as e:
            logger.error(f"❌ MEXC monitoring start failed: {e}")
            return {
                'status': 'error',
                'error': str(e),
                'monitoring_started': False
            }
            
    def stop_monitoring(self):
        """Stop monitoring and cleanup resources"""
        asyncio.create_task(self.disconnect())
        logger.info("🛑 MEXC monitoring stopped")
        
    def get_supported_symbols(self) -> List[str]:
        """Get list of supported trading symbols"""
        # Common MEXC trading pairs
        return [
            'BTC-USDT', 'ETH-USDT', 'BNB-USDT', 'ADA-USDT',
            'SOL-USDT', 'DOT-USDT', 'MATIC-USDT', 'AVAX-USDT',
            'ATOM-USDT', 'LINK-USDT'
        ]
        
    def get_latest_data(self, symbol: str, count: int = 1) -> List[Dict[str, Any]]:
        """Get latest market data from buffer"""
        try:
            # Find data for the requested symbol
            symbol_data = [
                item.to_standard_format() 
                for item in self.data_buffer 
                if hasattr(item, 'symbol') and item.symbol == symbol
            ]
            
            # Return latest 'count' items
            return symbol_data[-count:] if symbol_data else []
            
        except Exception as e:
            logger.error(f"❌ Error retrieving latest MEXC data: {e}")
            return []
            
    async def __aenter__(self):
        """Async context manager entry"""
        await self.connect()
        return self
        
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        await self.disconnect()


# Factory function for easy initialization
async def create_mexc_connector(config: Dict[str, Any]) -> MEXCConnector:
    """Create and initialize MEXC connector"""
    connector = MEXCConnector(config)
    
    # Test connection
    connected = await connector.connect()
    if not connected:
        logger.warning("⚠️ MEXC connector created but connection failed")
        
    return connector


# Configuration template
DEFAULT_MEXC_CONFIG = {
    'mexc_api_key': '',  # Set from environment
    'mexc_api_secret': '',  # Set from environment
    'mexc_passphrase': '',  # Set from environment if required
    'mexc_sandbox': True,  # Use sandbox by default
    'max_symbols': 3,  # Ultra-constrained limit
    'buffer_size': 200,  # Memory limit
    'rate_limit_per_second': 10,  # Conservative rate limiting
    'reconnect_delay': 5,
    'max_reconnect_attempts': 5,
    'default_symbols': ['ETH-USDT']
}


if __name__ == "__main__":
    """Test MEXC connector functionality"""
    import os
    
    async def test_mexc_connector():
        """Basic functionality test"""
        print("🧪 Testing MEXC Connector...")
        
        config = DEFAULT_MEXC_CONFIG.copy()
        config.update({
            'mexc_api_key': os.getenv('MEXC_API_KEY', ''),
            'mexc_api_secret': os.getenv('MEXC_API_SECRET', ''),
            'mexc_sandbox': True  # Always sandbox for tests
        })
        
        async with MEXCConnector(config) as connector:
            print(f"✅ Connector created: {connector.get_connection_status()}")
            
            # Test ticker fetch
            ticker = await connector.get_ticker('ETH-USDT')
            if ticker:
                print(f"📊 ETH-USDT ticker: ${ticker['price']:.2f}")
            else:
                print("⚠️ Ticker fetch failed (API keys may be missing)")
                
            # Test symbol subscription (WebSocket)
            success = await connector.subscribe_symbol('ETH-USDT')
            print(f"📡 WebSocket subscription: {'✅' if success else '❌'}")
            
            # Wait for some data
            await asyncio.sleep(3)
            
            # Check latest data
            latest = connector.get_latest_data('ETH-USDT', count=1)
            print(f"📈 Latest data: {len(latest)} points")
            
        print("✅ MEXC connector test completed")
        
    # Run test
    asyncio.run(test_mexc_connector())