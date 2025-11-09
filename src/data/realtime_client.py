#!/usr/bin/env python3
"""
Supreme System V5 - Real-time WebSocket Client

Real-time data streaming from Binance WebSocket API.
Handles live market data, order book updates, and trade streams.
"""

import asyncio
import json
import websockets
import threading
import time
from typing import Dict, List, Optional, Callable, Any
from datetime import datetime
import logging

try:
    from ..config.config import get_config
except ImportError:
    from config.config import get_config


class BinanceWebSocketClient:
    """
    Real-time WebSocket client for Binance market data.

    Features:
    - Live price updates
    - Order book depth
    - Trade streams
    - Automatic reconnection
    - Message buffering
    - Performance monitoring
    """

    def __init__(self, config_file: Optional[str] = None):
        """
        Initialize WebSocket client.

        Args:
            config_file: Optional configuration file
        """
        self.config = get_config()
        if config_file:
            from config.config import load_config
            self.config = load_config(config_file)

        self.logger = logging.getLogger(__name__)

        # WebSocket configuration
        self.ws_url = "wss://stream.binance.com:9443/ws"
        self.testnet_ws_url = "wss://testnet.binance.vision/ws"

        # Connection settings
        self.is_testnet = self.config.get('binance.testnet', True)
        self.base_url = self.testnet_ws_url if self.is_testnet else self.ws_url

        self.max_reconnect_attempts = 5
        self.reconnect_delay = 1.0
        self.ping_interval = 30  # seconds

        # Connection state
        self.websocket = None
        self.is_connected = False
        self.is_running = False
        self.connection_thread = None

        # Data streams
        self.active_streams = set()
        self.message_handlers = {}
        self.data_buffers = {}

        # Performance metrics
        self.metrics = {
            'messages_received': 0,
            'messages_processed': 0,
            'connection_attempts': 0,
            'successful_connections': 0,
            'reconnections': 0,
            'errors': 0,
            'start_time': None,
            'last_message_time': None
        }

        # Callbacks
        self.on_connect_callbacks = []
        self.on_disconnect_callbacks = []
        self.on_error_callbacks = []
        self.on_message_callbacks = []

    def add_stream(self, stream_name: str, callback: Optional[Callable] = None):
        """
        Add a data stream to subscribe to.

        Args:
            stream_name: Stream name (e.g., 'btcusdt@trade', 'ethusdt@depth')
            callback: Optional callback function for stream messages
        """
        self.active_streams.add(stream_name)
        if callback:
            self.message_handlers[stream_name] = callback

        # Initialize buffer for this stream
        if stream_name not in self.data_buffers:
            self.data_buffers[stream_name] = []

        self.logger.info(f"Added stream: {stream_name}")

    def remove_stream(self, stream_name: str):
        """Remove a data stream."""
        self.active_streams.discard(stream_name)
        self.message_handlers.pop(stream_name, None)
        self.data_buffers.pop(stream_name, None)
        self.logger.info(f"Removed stream: {stream_name}")

    def subscribe_price_stream(self, symbol: str, callback: Optional[Callable] = None):
        """Subscribe to price ticker stream."""
        stream = f"{symbol.lower()}@ticker"
        self.add_stream(stream, callback)

    def subscribe_trade_stream(self, symbol: str, callback: Optional[Callable] = None):
        """Subscribe to trade stream."""
        stream = f"{symbol.lower()}@trade"
        self.add_stream(stream, callback)

    def subscribe_kline_stream(self, symbol: str, interval: str, callback: Optional[Callable] = None):
        """Subscribe to kline/candlestick stream."""
        stream = f"{symbol.lower()}@kline_{interval}"
        self.add_stream(stream, callback)

    def subscribe_depth_stream(self, symbol: str, level: int = 10, callback: Optional[Callable] = None):
        """Subscribe to order book depth stream."""
        stream = f"{symbol.lower()}@depth{level}"
        self.add_stream(stream, callback)

    def start(self):
        """Start the WebSocket client in a background thread."""
        if self.is_running:
            self.logger.warning("WebSocket client is already running")
            return

        self.is_running = True
        self.metrics['start_time'] = datetime.now()

        self.connection_thread = threading.Thread(target=self._run_client, daemon=True)
        self.connection_thread.start()

        self.logger.info("WebSocket client started")

    def stop(self):
        """Stop the WebSocket client."""
        if not self.is_running:
            return

        self.is_running = False

        # Wait for thread to finish
        if self.connection_thread and self.connection_thread.is_alive():
            self.connection_thread.join(timeout=5.0)

        self.logger.info("WebSocket client stopped")

    def _run_client(self):
        """Run the WebSocket client in an event loop."""
        try:
            # Create new event loop for this thread
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

            # Run the async client
            loop.run_until_complete(self._async_client())

        except Exception as e:
            self.logger.error(f"WebSocket client error: {e}")
            self._trigger_error_callbacks(e)

    async def _async_client(self):
        """Async WebSocket client implementation."""
        while self.is_running:
            try:
                await self._connect_and_listen()
            except Exception as e:
                self.logger.error(f"Connection error: {e}")
                self.metrics['errors'] += 1

                if self.is_running:
                    await self._handle_reconnection()

    async def _connect_and_listen(self):
        """Connect to WebSocket and listen for messages."""
        try:
            self.metrics['connection_attempts'] += 1

            async with websockets.connect(self.base_url) as websocket:
                self.websocket = websocket
                self.is_connected = True
                self.metrics['successful_connections'] += 1

                self.logger.info(f"Connected to Binance WebSocket ({'testnet' if self.is_testnet else 'live'})")
                self._trigger_connect_callbacks()

                # Subscribe to streams
                await self._subscribe_to_streams()

                # Start ping task
                ping_task = asyncio.create_task(self._ping_loop())

                try:
                    async for message in websocket:
                        await self._handle_message(message)
                finally:
                    ping_task.cancel()

        except websockets.exceptions.ConnectionClosed:
            self.logger.warning("WebSocket connection closed")
        except Exception as e:
            self.logger.error(f"WebSocket error: {e}")
            raise
        finally:
            self.is_connected = False
            self.websocket = None
            self._trigger_disconnect_callbacks()

    async def _subscribe_to_streams(self):
        """Subscribe to all active streams."""
        if not self.active_streams:
            return

        subscription_message = {
            "method": "SUBSCRIBE",
            "params": list(self.active_streams),
            "id": 1
        }

        await self.websocket.send(json.dumps(subscription_message))
        self.logger.info(f"Subscribed to {len(self.active_streams)} streams")

    async def _handle_message(self, message: str):
        """Handle incoming WebSocket message."""
        try:
            self.metrics['messages_received'] += 1
            self.metrics['last_message_time'] = datetime.now()

            data = json.loads(message)

            # Handle subscription confirmation
            if 'result' in data and data.get('id') == 1:
                self.logger.info("Stream subscription confirmed")
                return

            # Extract stream name
            stream_name = data.get('stream')
            if not stream_name:
                # Single stream message
                if 'e' in data:  # Event type
                    stream_name = f"{data.get('s', 'unknown').lower()}@{data['e'].lower()}"

            if stream_name:
                # Buffer the message
                if stream_name not in self.data_buffers:
                    self.data_buffers[stream_name] = []

                self.data_buffers[stream_name].append({
                    'data': data,
                    'timestamp': datetime.now()
                })

                # Keep buffer size manageable (last 1000 messages per stream)
                if len(self.data_buffers[stream_name]) > 1000:
                    self.data_buffers[stream_name] = self.data_buffers[stream_name][-500:]

                # Handle message with specific callback
                if stream_name in self.message_handlers:
                    try:
                        self.message_handlers[stream_name](data)
                    except Exception as e:
                        self.logger.error(f"Error in message handler for {stream_name}: {e}")

                self.metrics['messages_processed'] += 1

            # Trigger general message callbacks
            for callback in self.on_message_callbacks:
                try:
                    callback(data)
                except Exception as e:
                    self.logger.error(f"Error in message callback: {e}")

        except json.JSONDecodeError:
            self.logger.warning(f"Invalid JSON message: {message}")
        except Exception as e:
            self.logger.error(f"Error handling message: {e}")

    async def _ping_loop(self):
        """Send periodic ping to keep connection alive."""
        while self.is_connected and self.is_running:
            try:
                await self.websocket.ping()
                await asyncio.sleep(self.ping_interval)
            except Exception as e:
                self.logger.error(f"Ping error: {e}")
                break

    async def _handle_reconnection(self):
        """Handle reconnection logic."""
        if not self.is_running:
            return

        self.metrics['reconnections'] += 1

        for attempt in range(self.max_reconnect_attempts):
            if not self.is_running:
                break

            delay = self.reconnect_delay * (2 ** attempt)  # Exponential backoff
            self.logger.info(f"Reconnecting in {delay:.1f} seconds (attempt {attempt + 1})")

            await asyncio.sleep(delay)

            if self.is_running:
                break

    def get_stream_data(self, stream_name: str, limit: Optional[int] = None) -> List[Dict]:
        """
        Get buffered data for a specific stream.

        Args:
            stream_name: Stream name
            limit: Maximum number of messages to return

        Returns:
            List of message data
        """
        if stream_name not in self.data_buffers:
            return []

        data = self.data_buffers[stream_name]
        if limit:
            data = data[-limit:]

        return [msg['data'] for msg in data]

    def get_latest_price(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Get latest price data for a symbol."""
        stream_name = f"{symbol.lower()}@ticker"
        data = self.get_stream_data(stream_name, limit=1)

        if data:
            return data[0]
        return None

    def get_recent_trades(self, symbol: str, limit: int = 10) -> List[Dict[str, Any]]:
        """Get recent trades for a symbol."""
        stream_name = f"{symbol.lower()}@trade"
        return self.get_stream_data(stream_name, limit=limit)

    def get_order_book(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Get current order book for a symbol."""
        stream_name = f"{symbol.lower()}@depth10"
        data = self.get_stream_data(stream_name, limit=1)

        if data:
            return data[0]
        return None

    def get_metrics(self) -> Dict[str, Any]:
        """Get client performance metrics."""
        metrics = self.metrics.copy()

        # Calculate uptime
        if metrics['start_time']:
            uptime = (datetime.now() - metrics['start_time']).total_seconds()
            metrics['uptime_seconds'] = uptime

        # Calculate message rate
        if metrics['start_time'] and metrics['messages_received'] > 0:
            elapsed = (datetime.now() - metrics['start_time']).total_seconds()
            if elapsed > 0:
                metrics['messages_per_second'] = metrics['messages_received'] / elapsed

        # Connection health
        metrics['connection_healthy'] = self.is_connected
        metrics['active_streams'] = len(self.active_streams)

        return metrics

    def add_connect_callback(self, callback: Callable):
        """Add callback for connection events."""
        self.on_connect_callbacks.append(callback)

    def add_disconnect_callback(self, callback: Callable):
        """Add callback for disconnection events."""
        self.on_disconnect_callbacks.append(callback)

    def add_error_callback(self, callback: Callable):
        """Add callback for error events."""
        self.on_error_callbacks.append(callback)

    def add_message_callback(self, callback: Callable):
        """Add callback for all messages."""
        self.on_message_callbacks.append(callback)

    def _trigger_connect_callbacks(self):
        """Trigger connection callbacks."""
        for callback in self.on_connect_callbacks:
            try:
                callback()
            except Exception as e:
                self.logger.error(f"Error in connect callback: {e}")

    def _trigger_disconnect_callbacks(self):
        """Trigger disconnection callbacks."""
        for callback in self.on_disconnect_callbacks:
            try:
                callback()
            except Exception as e:
                self.logger.error(f"Error in disconnect callback: {e}")

    def _trigger_error_callbacks(self, error: Exception):
        """Trigger error callbacks."""
        for callback in self.on_error_callbacks:
            try:
                callback(error)
            except Exception as e:
                self.logger.error(f"Error in error callback: {e}")

    def __enter__(self):
        """Context manager entry."""
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.stop()
