"""
Binance Exchange Connector
Backup connector for market data and trading
"""

import asyncio
import json
import hmac
import hashlib
import time
from typing import Dict, List, Optional, Any

import aiohttp
import websockets
from loguru import logger

from .base import BaseExchange, ExchangeConfig, OrderResult

class BinanceConnector(BaseExchange):
    """
    Binance exchange connector
    Backup exchange for OKX
    """
    
    def __init__(self, config: ExchangeConfig):
        super().__init__(config)
        
        # Binance endpoints
        self.base_url = "https://api.binance.com" if not config.sandbox else "https://testnet.binance.vision"
        self.ws_url = "wss://stream.binance.com:9443/ws"
        
        # Connection state
        self.websocket = None
        self.session = None
    
    def _generate_signature(self, query_string: str) -> str:
        """
        Generate Binance API signature
        """
        return hmac.new(
            self.config.secret_key.encode(),
            query_string.encode(),
            hashlib.sha256
        ).hexdigest()
    
    async def connect(self) -> bool:
        """
        Connect to Binance WebSocket
        """
        try:
            logger.info("🔗 Connecting to Binance WebSocket...")
            
            # Create HTTP session
            self.session = aiohttp.ClientSession()
            
            # Connect to WebSocket
            self.websocket = await websockets.connect(self.ws_url)
            
            self.connected = True
            logger.info("✅ Binance WebSocket connected")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Binance connection failed: {e}")
            return False
    
    async def disconnect(self):
        """
        Disconnect from Binance
        """
        logger.info("🔌 Disconnecting from Binance...")
        
        self.connected = False
        
        if self.websocket:
            await self.websocket.close()
            
        if self.session:
            await self.session.close()
            
        logger.info("✅ Binance disconnected")
    
    async def subscribe_market_data(self, symbols: List[str]):
        """
        Subscribe to Binance ticker stream
        """
        if not self.websocket:
            raise RuntimeError("Not connected to Binance WebSocket")
            
        # Convert symbols to Binance format
        binance_symbols = [symbol.replace("-", "").lower() + "@ticker" for symbol in symbols]
        
        subscribe_msg = {
            "method": "SUBSCRIBE",
            "params": binance_symbols,
            "id": 1
        }
        
        await self.websocket.send(json.dumps(subscribe_msg))
        logger.info(f"📈 Subscribed to Binance market data: {symbols}")
        
        # Start data handler
        asyncio.create_task(self._handle_market_data())
    
    async def _handle_market_data(self):
        """
        Handle Binance market data
        """
        try:
            async for message in self.websocket:
                data = json.loads(message)
                
                if 'c' in data:  # Ticker data
                    symbol = data.get('s', '').replace('USDT', '-USDT')
                    price = float(data.get('c', 0))  # Current price
                    volume = float(data.get('v', 0))  # 24h volume
                    bid = float(data.get('b', price * 0.9999))
                    ask = float(data.get('a', price * 1.0001))
                    
                    # Call market data callback
                    if self.market_data_callback:
                        await self.market_data_callback(symbol, price, volume, bid, ask)
                        
        except Exception as e:
            logger.error(f"Binance market data handler error: {e}")
    
    async def place_market_order(self, symbol: str, side: str, amount: float) -> OrderResult:
        """
        Place market order on Binance
        """
        # Mock implementation for development
        logger.info(f"🔄 Mock Binance market order: {side} {amount} {symbol}")
        
        return OrderResult(
            success=True,
            order_id=f"binance_{int(time.time())}",
            symbol=symbol,
            side=side,
            amount=amount,
            price=35000.0 if "BTC" in symbol else 1800.0,
            fee=amount * 0.001,  # 0.1% fee
            timestamp=time.time()
        )
    
    async def place_limit_order(self, symbol: str, side: str, amount: float, price: float) -> OrderResult:
        """
        Place limit order on Binance
        """
        # Mock implementation
        logger.info(f"🎯 Mock Binance limit order: {side} {amount} {symbol} @ {price}")
        
        return OrderResult(
            success=True,
            order_id=f"binance_limit_{int(time.time())}",
            symbol=symbol,
            side=side,
            amount=amount,
            price=price,
            fee=amount * price * 0.001,
            timestamp=time.time()
        )
    
    async def cancel_order(self, order_id: str, symbol: str) -> bool:
        """
        Cancel order on Binance
        """
        logger.info(f"❌ Mock cancel Binance order: {order_id}")
        return True
    
    async def get_balance(self) -> Dict[str, float]:
        """
        Get Binance account balance
        """
        return {
            'USDT': 10000.0,
            'BTC': 0.0,
            'ETH': 0.0
        }
    
    async def get_positions(self) -> List[Dict[str, Any]]:
        """
        Get Binance futures positions
        """
        return []
