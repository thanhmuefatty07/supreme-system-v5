#!/usr/bin/env python3
"""
Supreme System V5 - Live Data Manager

Real-time market data integration with Binance WebSocket streams.
Provides enterprise-grade data pipeline with auto-reconnect, validation,
and high availability for live trading systems.
"""

import asyncio
import json
import logging
from typing import Dict, List, Optional, Callable, Any
from dataclasses import dataclass
from datetime import datetime, timedelta
import websockets
import gzip

from .data_validator import validate_live_market_data
from ..utils.validators import validate_market_data as legacy_validate

logger = logging.getLogger(__name__)


@dataclass
class StreamConfig:
    """Configuration for WebSocket data streams."""
    symbol: str
    interval: str = "1m"  # 1m, 3m, 5m, 15m, 30m, 1h, 2h, 4h, 6h, 8h, 12h, 1d
    stream_type: str = "kline"  # kline, ticker, trade, depth


@dataclass
class ConnectionStatus:
    """Real-time connection status information."""
    connected: bool = False
    last_message_time: Optional[datetime] = None
    reconnect_count: int = 0
    messages_received: int = 0
    errors_count: int = 0
    uptime_seconds: float = 0.0


class LiveDataManager:
    """
    Enterprise-grade live data manager for real-time market data.

    Features:
    - Binance WebSocket integration (public streams, no API key required)
    - Auto-reconnect with exponential backoff
    - Data validation and quality monitoring
    - Multiple symbol support with configurable intervals
    - Event-driven architecture for high performance
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the live data manager.

        Args:
            config: Configuration dictionary with connection settings
        """
        self.config = config or {
            "reconnect_delay": 5.0,        # Initial reconnect delay
            "max_reconnect_delay": 300.0,  # Maximum reconnect delay
            "max_reconnect_attempts": 10,  # Maximum reconnect attempts
            "ping_interval": 30.0,         # WebSocket ping interval
            "timeout": 10.0,               # Connection timeout
            "validate_data": True,         # Enable data validation
            "buffer_size": 1000            # Data buffer size
        }

        # Connection management
        self.websocket: Optional[websockets.WebSocketServerProtocol] = None
        self.connection_status = ConnectionStatus()
        self.stream_configs: List[StreamConfig] = []
        self.data_callbacks: List[Callable] = []

        # Data processing
        self.data_buffer: List[Dict[str, Any]] = []
        self.last_data_time: Optional[datetime] = None

        # Control flags
        self.running = False
        self._reconnect_task: Optional[asyncio.Task] = None
        self._ping_task: Optional[asyncio.Task] = None

        logger.info("LiveDataManager initialized with enterprise configuration")

    def add_stream(self, symbol: str, interval: str = "1m", stream_type: str = "kline"):
        """
        Add a data stream to monitor.

        Args:
            symbol: Trading symbol (e.g., 'BTCUSDT')
            interval: Kline interval (1m, 5m, 15m, 1h, 4h, 1d)
            stream_type: Type of stream (kline, ticker, trade)
        """
        config = StreamConfig(symbol=symbol, interval=interval, stream_type=stream_type)
        self.stream_configs.append(config)
        logger.info(f"Added stream: {symbol}@{interval} ({stream_type})")

    def add_data_callback(self, callback: Callable):
        """
        Add a callback function to receive processed market data.

        Args:
            callback: Function that receives market data dict
        """
        self.data_callbacks.append(callback)
        logger.info(f"Added data callback: {callback.__name__}")

    def _build_websocket_url(self) -> str:
        """
        Build WebSocket URL for subscribed streams.

        Returns:
            Complete WebSocket URL for Binance streams
        """
        if not self.stream_configs:
            raise ValueError("No streams configured")

        # Build stream names
        stream_names = []
        for config in self.stream_configs:
            if config.stream_type == "kline":
                stream_names.append(f"{config.symbol.lower()}@kline_{config.interval}")
            elif config.stream_type == "ticker":
                stream_names.append(f"{config.symbol.lower()}@ticker")
            elif config.stream_type == "trade":
                stream_names.append(f"{config.symbol.lower()}@trade")

        # Combine streams into single WebSocket connection
        streams_param = "/".join(stream_names)
        url = f"wss://stream.binance.com:9443/ws/{streams_param}"

        logger.debug(f"Built WebSocket URL: {url}")
        return url

    async def connect(self) -> bool:
        """
        Establish WebSocket connection to Binance.

        Returns:
            True if connection successful, False otherwise
        """
        try:
            url = self._build_websocket_url()

            # Connect with timeout
            self.websocket = await asyncio.wait_for(
                websockets.connect(url, extra_headers={"Accept-Encoding": "gzip"}),
                timeout=self.config["timeout"]
            )

            self.connection_status.connected = True
            self.connection_status.reconnect_count = 0
            self.connection_status.uptime_seconds = 0.0

            logger.info("Successfully connected to Binance WebSocket")

            # Start ping task
            self._ping_task = asyncio.create_task(self._ping_loop())

            return True

        except Exception as e:
            logger.error(f"Failed to connect to WebSocket: {e}")
            self.connection_status.errors_count += 1
            return False

    async def disconnect(self):
        """Close WebSocket connection gracefully."""
        self.running = False

        if self._ping_task and not self._ping_task.done():
            self._ping_task.cancel()

        if self._reconnect_task and not self._reconnect_task.done():
            self._reconnect_task.cancel()

        if self.websocket:
            await self.websocket.close()
            self.websocket = None

        self.connection_status.connected = False
        logger.info("Disconnected from Binance WebSocket")

    async def _ping_loop(self):
        """Maintain connection with periodic pings."""
        try:
            while self.running and self.websocket:
                await asyncio.sleep(self.config["ping_interval"])

                if self.websocket and self.websocket.open:
                    # Send ping to keep connection alive
                    ping_data = {"method": "ping", "params": [], "id": 1}
                    await self.websocket.send(json.dumps(ping_data))

        except Exception as e:
            logger.warning(f"Ping loop error: {e}")

    async def _handle_reconnect(self):
        """Handle reconnection with exponential backoff."""
        delay = self.config["reconnect_delay"]
        max_delay = self.config["max_reconnect_delay"]
        max_attempts = self.config["max_reconnect_attempts"]

        while self.running and not self.connection_status.connected:
            if self.connection_status.reconnect_count >= max_attempts:
                logger.error(f"Max reconnection attempts ({max_attempts}) reached")
                break

            logger.info(f"Attempting reconnection in {delay:.1f} seconds...")
            await asyncio.sleep(delay)

            if await self.connect():
                logger.info("Reconnected successfully")
                break

            self.connection_status.reconnect_count += 1
            delay = min(delay * 1.5, max_delay)  # Exponential backoff

    def _parse_kline_message(self, data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Parse Binance kline WebSocket message into OHLCV format.

        Args:
            data: Raw WebSocket message data

        Returns:
            Parsed market data or None if invalid
        """
        try:
            k = data.get("k", {})
            if not k:
                return None

            # Extract OHLCV data from kline
            market_data = {
                "symbol": data.get("s", "").upper(),
                "timestamp": int(k.get("t", 0)),  # Kline start time
                "open": float(k.get("o", 0)),
                "high": float(k.get("h", 0)),
                "low": float(k.get("l", 0)),
                "close": float(k.get("c", 0)),
                "volume": float(k.get("v", 0)),
                "close_time": int(k.get("T", 0)),  # Kline close time
                "quote_volume": float(k.get("q", 0)),
                "trades_count": int(k.get("n", 0)),
                "taker_buy_volume": float(k.get("V", 0)),
                "taker_buy_quote_volume": float(k.get("Q", 0)),
                "is_closed": k.get("x", False)  # Is kline closed?
            }

            # Validate data
            if self.config["validate_data"]:
                try:
                    # Use our live market data validator
                    validated = validate_live_market_data(market_data)
                    if validated:
                        return validated.model_dump()
                except Exception as e:
                    logger.warning(f"Data validation failed: {e}")
                    return None
            else:
                return market_data

        except Exception as e:
            logger.error(f"Error parsing kline message: {e}")
            return None

    async def _process_message(self, message: str):
        """
        Process incoming WebSocket message.

        Args:
            message: Raw JSON message from WebSocket
        """
        try:
            # Handle compressed messages
            if isinstance(message, bytes):
                message = gzip.decompress(message).decode('utf-8')

            data = json.loads(message)

            # Update connection status
            self.connection_status.last_message_time = datetime.now()
            self.connection_status.messages_received += 1

            # Process different message types
            if "stream" in data and "data" in data:
                # Multi-stream format
                stream_data = data["data"]
                processed_data = self._parse_kline_message(stream_data)

            elif "e" in data and data["e"] == "kline":
                # Direct kline message
                processed_data = self._parse_kline_message(data)

            else:
                # Other message types (ping, subscription confirm, etc.)
                logger.debug(f"Received non-data message: {data}")
                return

            if processed_data:
                # Add to buffer
                self.data_buffer.append(processed_data)
                if len(self.data_buffer) > self.config["buffer_size"]:
                    self.data_buffer.pop(0)

                # Update timestamp
                self.last_data_time = datetime.now()

                # Notify callbacks
                for callback in self.data_callbacks:
                    try:
                        await callback(processed_data)
                    except Exception as e:
                        logger.error(f"Error in data callback {callback.__name__}: {e}")

        except json.JSONDecodeError as e:
            logger.warning(f"Invalid JSON message: {e}")
        except Exception as e:
            logger.error(f"Error processing message: {e}")
            self.connection_status.errors_count += 1

    async def start_streaming(self):
        """
        Start the data streaming loop.

        This method will run indefinitely until stopped.
        """
        self.running = True
        logger.info("Starting live data streaming...")

        while self.running:
            try:
                if not self.connection_status.connected:
                    logger.info("Not connected, attempting to connect...")
                    if not await self.connect():
                        # Start reconnection task
                        if not self._reconnect_task or self._reconnect_task.done():
                            self._reconnect_task = asyncio.create_task(self._handle_reconnect())
                        await asyncio.sleep(1)
                        continue

                # Receive and process messages
                if self.websocket and self.websocket.open:
                    try:
                        message = await asyncio.wait_for(
                            self.websocket.recv(),
                            timeout=self.config["timeout"]
                        )
                        await self._process_message(message)

                    except asyncio.TimeoutError:
                        logger.warning("WebSocket receive timeout")
                        self.connection_status.connected = False
                        continue

                    except websockets.exceptions.ConnectionClosed:
                        logger.warning("WebSocket connection closed")
                        self.connection_status.connected = False
                        continue

                else:
                    logger.warning("WebSocket not available")
                    self.connection_status.connected = False
                    await asyncio.sleep(1)

            except Exception as e:
                logger.error(f"Error in streaming loop: {e}")
                self.connection_status.errors_count += 1
                self.connection_status.connected = False
                await asyncio.sleep(1)

        logger.info("Data streaming stopped")

    async def get_latest_data(self, symbol: str) -> Optional[Dict[str, Any]]:
        """
        Get the latest data for a specific symbol.

        Args:
            symbol: Trading symbol

        Returns:
            Latest market data for the symbol or None
        """
        # Find most recent data for symbol
        for data in reversed(self.data_buffer):
            if data.get("symbol", "").upper() == symbol.upper():
                return data
        return None

    def get_connection_status(self) -> ConnectionStatus:
        """Get current connection status."""
        # Update uptime
        if self.connection_status.connected and self.connection_status.last_message_time:
            self.connection_status.uptime_seconds = (
                datetime.now() - self.connection_status.last_message_time
            ).total_seconds()

        return self.connection_status

    def get_buffered_data(self, symbol: Optional[str] = None, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Get buffered data, optionally filtered by symbol.

        Args:
            symbol: Filter by symbol (optional)
            limit: Maximum number of records to return (optional)

        Returns:
            List of market data records
        """
        data = self.data_buffer

        if symbol:
            data = [d for d in data if d.get("symbol", "").upper() == symbol.upper()]

        if limit:
            data = data[-limit:]

        return data

    async def subscribe_symbol(self, symbol: str, interval: str = "1m"):
        """
        Subscribe to additional symbol streams dynamically.

        Args:
            symbol: Trading symbol to subscribe to
            interval: Kline interval
        """
        # This would require sending subscription messages to WebSocket
        # For now, we'll log that this feature needs to be implemented
        logger.info(f"Dynamic subscription for {symbol}@{interval} requested - feature not yet implemented")

    async def unsubscribe_symbol(self, symbol: str):
        """
        Unsubscribe from symbol streams dynamically.

        Args:
            symbol: Trading symbol to unsubscribe from
        """
        logger.info(f"Dynamic unsubscription for {symbol} requested - feature not yet implemented")
