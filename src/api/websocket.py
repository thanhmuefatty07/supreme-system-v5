"""
🔌 Supreme System V5 - Advanced WebSocket Handler
Real-time data streaming and communication hub with enhanced features

Features:
- Ultra-low latency WebSocket streaming (<500ms target)
- Multi-client connection management with subscription control
- 6 message types with separate frequencies
- Backpressure control for high-frequency data
- Anonymous read-only access (JWT upgrade path ready)
- Real-time trading data broadcast
- Performance metrics streaming with throttling
- Component status updates
- Event-driven notifications
- Connection health monitoring
- Message queuing and prioritization
"""

import asyncio
import json
import time
from datetime import datetime
from typing import Dict, List, Set, Any, Optional, Callable
from dataclasses import dataclass, field
from enum import Enum
from collections import deque

from fastapi import WebSocket, WebSocketDisconnect
import logging

# Configure logging
logger = logging.getLogger("supreme_websocket")


class MessageType(Enum):
    """WebSocket message types with different frequencies"""
    # High frequency (100ms)
    PERFORMANCE = "performance"
    MARKET_DATA = "market_data"
    
    # Medium frequency (500ms)
    TRADING_STATUS = "trading_status"
    PORTFOLIO_UPDATE = "portfolio_update"
    
    # Low frequency (2s)
    COMPONENT_STATUS = "component_status"
    SYSTEM_ALERT = "system_alert"
    
    # Event-driven (immediate)
    ORDER_UPDATE = "order_update"
    ERROR = "error"
    HEARTBEAT = "heartbeat"


class MessagePriority(Enum):
    """Message priority levels for backpressure control"""
    CRITICAL = 1  # Errors, alerts - never drop
    HIGH = 2      # Orders, trading status - preserve under pressure
    NORMAL = 3    # Performance, portfolio - can be throttled
    LOW = 4       # Component status - first to drop


@dataclass
class MessageFrequencyConfig:
    """Configuration for message frequencies"""
    high_frequency_ms: int = 100     # 10 times per second
    medium_frequency_ms: int = 500   # 2 times per second
    low_frequency_ms: int = 2000     # Every 2 seconds
    heartbeat_frequency_ms: int = 10000  # Every 10 seconds


@dataclass
class BackpressureConfig:
    """Backpressure control configuration"""
    max_queue_size: int = 1000
    drop_threshold: float = 0.8  # Drop messages when queue is 80% full
    priority_preservation: bool = True  # Always preserve high-priority messages


@dataclass
class WebSocketMessage:
    """Enhanced WebSocket message structure"""
    type: MessageType
    timestamp: datetime
    data: Dict[str, Any]
    client_id: Optional[str] = None
    sequence_id: Optional[int] = None
    priority: MessagePriority = MessagePriority.NORMAL
    
    def to_json(self) -> str:
        """Convert message to JSON string"""
        return json.dumps({
            "type": self.type.value,
            "timestamp": self.timestamp.isoformat(),
            "data": self.data,
            "client_id": self.client_id,
            "sequence_id": self.sequence_id,
            "priority": self.priority.value
        })


@dataclass
class ClientConnection:
    """Enhanced WebSocket client connection info"""
    websocket: WebSocket
    client_id: str
    connected_at: float
    last_heartbeat: float
    subscriptions: Set[MessageType]
    message_count: int = 0
    
    # Backpressure management
    message_queue: deque = field(default_factory=deque)
    max_queue_size: int = 1000
    dropped_messages: int = 0
    
    # Performance tracking
    bytes_sent: int = 0
    avg_latency_ms: float = 0.0
    connection_quality: float = 1.0  # 0.0 to 1.0
    
    def is_alive(self, heartbeat_timeout: float = 30.0) -> bool:
        """Check if connection is alive based on heartbeat"""
        return time.time() - self.last_heartbeat < heartbeat_timeout
    
    def can_receive_message(self, priority: MessagePriority) -> bool:
        """Check if client can receive message based on backpressure"""
        queue_size = len(self.message_queue)
        
        # Always allow critical messages
        if priority == MessagePriority.CRITICAL:
            return True
        
        # Check queue capacity
        if queue_size >= self.max_queue_size:
            return False
        
        # Apply backpressure based on queue fullness
        queue_ratio = queue_size / self.max_queue_size
        
        if priority == MessagePriority.HIGH and queue_ratio < 0.8:
            return True
        elif priority == MessagePriority.NORMAL and queue_ratio < 0.6:
            return True
        elif priority == MessagePriority.LOW and queue_ratio < 0.4:
            return True
        
        return False
    
    def add_message(self, message: WebSocketMessage) -> bool:
        """Add message to client queue with backpressure control"""
        if not self.can_receive_message(message.priority):
            self.dropped_messages += 1
            logger.debug(f"Dropped message for client {self.client_id} due to backpressure")
            return False
        
        self.message_queue.append(message)
        return True


class WebSocketHandler:
    """Advanced WebSocket connection manager with enhanced features"""
    
    def __init__(self, 
                 freq_config: MessageFrequencyConfig = None,
                 backpressure_config: BackpressureConfig = None):
        self.connections: Dict[str, ClientConnection] = {}
        self.freq_config = freq_config or MessageFrequencyConfig()
        self.backpressure_config = backpressure_config or BackpressureConfig()
        
        # Background tasks
        self.broadcast_tasks: Dict[str, Optional[asyncio.Task]] = {}
        self.heartbeat_task: Optional[asyncio.Task] = None
        
        # State management
        self.sequence_counter = 0
        self.is_running = False
        
        # Performance tracking
        self.messages_sent = 0
        self.messages_dropped = 0
        self.start_time = time.time()
        
        # Message type to priority mapping
        self.message_priorities = {
            MessageType.ERROR: MessagePriority.CRITICAL,
            MessageType.SYSTEM_ALERT: MessagePriority.CRITICAL,
            MessageType.ORDER_UPDATE: MessagePriority.HIGH,
            MessageType.TRADING_STATUS: MessagePriority.HIGH,
            MessageType.PORTFOLIO_UPDATE: MessagePriority.NORMAL,
            MessageType.PERFORMANCE: MessagePriority.NORMAL,
            MessageType.MARKET_DATA: MessagePriority.NORMAL,
            MessageType.COMPONENT_STATUS: MessagePriority.LOW,
            MessageType.HEARTBEAT: MessagePriority.LOW
        }
    
    async def start(self):
        """Start WebSocket handler"""
        if self.is_running:
            return
        
        self.is_running = True
        
        # Start heartbeat task
        self.heartbeat_task = asyncio.create_task(self._heartbeat_loop())
        
        logger.info("🔌 WebSocket handler started")
        logger.info(f"   Message frequencies: High={self.freq_config.high_frequency_ms}ms, Medium={self.freq_config.medium_frequency_ms}ms")
        logger.info(f"   Backpressure threshold: {self.backpressure_config.drop_threshold}")
    
    async def stop(self):
        """Stop WebSocket handler and close all connections"""
        self.is_running = False
        
        # Cancel heartbeat task
        if self.heartbeat_task:
            self.heartbeat_task.cancel()
        
        # Close all connections
        for client_id in list(self.connections.keys()):
            await self.disconnect_client(client_id)
        
        logger.info("🔌 WebSocket handler stopped")
    
    async def connect_client(self, websocket: WebSocket, client_id: Optional[str] = None) -> str:
        """Connect a new WebSocket client with anonymous access"""
        await websocket.accept()
        
        if not client_id:
            client_id = f"client_{len(self.connections)}_{int(time.time() * 1000)}"
        
        # Create enhanced client connection
        client = ClientConnection(
            websocket=websocket,
            client_id=client_id,
            connected_at=time.time(),
            last_heartbeat=time.time(),
            subscriptions=set(MessageType),  # Subscribe to all by default
            max_queue_size=self.backpressure_config.max_queue_size
        )
        
        self.connections[client_id] = client
        
        # Send enhanced welcome message
        welcome_msg = WebSocketMessage(
            type=MessageType.SYSTEM_ALERT,
            timestamp=datetime.utcnow(),
            data={
                "message": "Connected to Supreme System V5",
                "client_id": client_id,
                "available_subscriptions": [t.value for t in MessageType],
                "frequency_config": {
                    "high_frequency_ms": self.freq_config.high_frequency_ms,
                    "medium_frequency_ms": self.freq_config.medium_frequency_ms,
                    "low_frequency_ms": self.freq_config.low_frequency_ms
                },
                "access_mode": "anonymous_read_only",
                "backpressure_enabled": True
            },
            client_id=client_id,
            priority=MessagePriority.HIGH
        )
        
        await self._send_to_client_immediately(client_id, welcome_msg)
        
        logger.info(f"🔌 Client {client_id} connected (anonymous). Total: {len(self.connections)}")
        return client_id
    
    async def disconnect_client(self, client_id: str):
        """Disconnect a WebSocket client"""
        if client_id not in self.connections:
            return
        
        client = self.connections[client_id]
        
        try:
            await client.websocket.close()
        except:
            pass
        
        # Log connection statistics
        logger.info(f"🔌 Client {client_id} disconnected:")
        logger.info(f"   Messages sent: {client.message_count}")
        logger.info(f"   Messages dropped: {client.dropped_messages}")
        logger.info(f"   Connection quality: {client.connection_quality:.2f}")
        
        del self.connections[client_id]
        
        logger.info(f"🔌 Client disconnected. Remaining: {len(self.connections)}")
    
    async def handle_client_message(self, client_id: str, message: str):
        """Handle incoming message from client"""
        try:
            data = json.loads(message)
            msg_type = data.get("type")
            
            if msg_type == "heartbeat":
                await self._handle_heartbeat(client_id)
            elif msg_type == "subscribe":
                await self._handle_subscription(client_id, data.get("subscriptions", []))
            elif msg_type == "unsubscribe":
                await self._handle_unsubscription(client_id, data.get("unsubscriptions", []))
        
        except json.JSONDecodeError:
            logger.error(f"Invalid JSON from client {client_id}: {message}")
        except Exception as e:
            logger.error(f"Error handling message from {client_id}: {e}")
    
    async def broadcast_message(self, message_type: MessageType, data: Dict[str, Any], 
                              target_clients: Set[str] = None):
        """Broadcast message with enhanced routing and backpressure"""
        message = WebSocketMessage(
            type=message_type,
            timestamp=datetime.utcnow(),
            data=data,
            sequence_id=self._get_next_sequence(),
            priority=self.message_priorities.get(message_type, MessagePriority.NORMAL)
        )
        
        if target_clients is None:
            target_clients = set(self.connections.keys())
        
        # Filter clients by subscription and backpressure
        eligible_clients = []
        for client_id in target_clients:
            if client_id in self.connections:
                client = self.connections[client_id]
                if message.type in client.subscriptions:
                    # Apply special handling for market_data under backpressure
                    if message_type == MessageType.MARKET_DATA:
                        if client.can_receive_message(message.priority):
                            eligible_clients.append(client_id)
                        else:
                            # Drop market_data first under pressure
                            client.dropped_messages += 1
                            self.messages_dropped += 1
                    else:
                        if client.can_receive_message(message.priority):
                            eligible_clients.append(client_id)
        
        # Send to eligible clients
        if eligible_clients:
            tasks = [self._send_to_client_queued(client_id, message) 
                    for client_id in eligible_clients]
            await asyncio.gather(*tasks, return_exceptions=True)
    
    async def _send_to_client_queued(self, client_id: str, message: WebSocketMessage):
        """Send message to client through queue"""
        if client_id not in self.connections:
            return
        
        client = self.connections[client_id]
        if client.add_message(message):
            # Process client's message queue
            await self._process_client_queue(client_id)
    
    async def _send_to_client_immediately(self, client_id: str, message: WebSocketMessage):
        """Send message to client immediately (bypass queue)"""
        if client_id not in self.connections:
            return
        
        client = self.connections[client_id]
        
        try:
            start_time = time.perf_counter()
            await client.websocket.send_text(message.to_json())
            latency_ms = (time.perf_counter() - start_time) * 1000
            
            # Update client metrics
            client.message_count += 1
            client.avg_latency_ms = (client.avg_latency_ms * 0.9) + (latency_ms * 0.1)
            
            self.messages_sent += 1
            
        except WebSocketDisconnect:
            await self.disconnect_client(client_id)
        except Exception as e:
            logger.error(f"Error sending to client {client_id}: {e}")
            await self.disconnect_client(client_id)
    
    async def _process_client_queue(self, client_id: str):
        """Process a client's message queue"""
        if client_id not in self.connections:
            return
        
        client = self.connections[client_id]
        
        # Process up to 5 messages per batch to prevent blocking
        batch_size = min(5, len(client.message_queue))
        for _ in range(batch_size):
            if not client.message_queue:
                break
                
            message = client.message_queue.popleft()
            await self._send_to_client_immediately(client_id, message)
            
            # Small delay for connection quality management
            if client.connection_quality < 0.8:
                await asyncio.sleep(0.001)  # 1ms delay for poor connections
    
    async def _handle_heartbeat(self, client_id: str):
        """Handle heartbeat from client"""
        if client_id in self.connections:
            client = self.connections[client_id]
            client.last_heartbeat = time.time()
            
            # Send heartbeat response
            response = WebSocketMessage(
                type=MessageType.HEARTBEAT,
                timestamp=datetime.utcnow(),
                data={
                    "status": "alive",
                    "server_time": datetime.utcnow().isoformat(),
                    "connection_quality": client.connection_quality,
                    "queue_size": len(client.message_queue),
                    "messages_dropped": client.dropped_messages,
                    "avg_latency_ms": client.avg_latency_ms
                },
                client_id=client_id,
                priority=MessagePriority.LOW
            )
            await self._send_to_client_immediately(client_id, response)
    
    async def _handle_subscription(self, client_id: str, subscriptions: List[str]):
        """Handle subscription changes"""
        if client_id in self.connections:
            client = self.connections[client_id]
            try:
                client.subscriptions = {MessageType(sub) for sub in subscriptions}
                logger.info(f"🔔 Client {client_id} updated subscriptions: {subscriptions}")
            except ValueError as e:
                logger.warning(f"Invalid subscription from {client_id}: {e}")
    
    async def _handle_unsubscription(self, client_id: str, unsubscriptions: List[str]):
        """Handle unsubscriptions"""
        if client_id in self.connections:
            client = self.connections[client_id]
            try:
                for unsub in unsubscriptions:
                    client.subscriptions.discard(MessageType(unsub))
                logger.info(f"🔕 Client {client_id} unsubscribed from: {unsubscriptions}")
            except ValueError as e:
                logger.warning(f"Invalid unsubscription from {client_id}: {e}")
    
    async def _heartbeat_loop(self):
        """Background task for connection health monitoring"""
        while self.is_running:
            try:
                # Check for dead connections
                dead_clients = []
                for client_id, client in self.connections.items():
                    if not client.is_alive():
                        dead_clients.append(client_id)
                
                # Remove dead connections
                for client_id in dead_clients:
                    await self.disconnect_client(client_id)
                
                # Send server heartbeat to all clients (if any)
                if self.connections:
                    await self.broadcast_message(
                        MessageType.HEARTBEAT,
                        {
                            "server_status": "healthy", 
                            "total_clients": len(self.connections),
                            "messages_sent": self.messages_sent,
                            "messages_dropped": self.messages_dropped
                        }
                    )
                
                await asyncio.sleep(self.freq_config.heartbeat_frequency_ms / 1000.0)
                
            except Exception as e:
                logger.error(f"Heartbeat loop error: {e}")
    
    def _get_next_sequence(self) -> int:
        """Get next sequence ID"""
        self.sequence_counter += 1
        return self.sequence_counter
    
    def get_connection_stats(self) -> Dict[str, Any]:
        """Get comprehensive WebSocket connection statistics"""
        uptime = time.time() - self.start_time
        total_dropped = sum(client.dropped_messages for client in self.connections.values())
        avg_quality = sum(client.connection_quality for client in self.connections.values()) / len(self.connections) if self.connections else 0
        
        return {
            "active_connections": len(self.connections),
            "messages_sent": self.messages_sent,
            "messages_dropped": self.messages_dropped + total_dropped,
            "uptime_seconds": uptime,
            "messages_per_second": self.messages_sent / uptime if uptime > 0 else 0,
            "average_connection_quality": avg_quality,
            "frequency_config": {
                "high_frequency_ms": self.freq_config.high_frequency_ms,
                "medium_frequency_ms": self.freq_config.medium_frequency_ms,
                "low_frequency_ms": self.freq_config.low_frequency_ms
            },
            "clients": {
                client_id: {
                    "connected_at": client.connected_at,
                    "messages_sent": client.message_count,
                    "messages_dropped": client.dropped_messages,
                    "queue_size": len(client.message_queue),
                    "connection_quality": client.connection_quality,
                    "avg_latency_ms": client.avg_latency_ms,
                    "subscriptions": [sub.value for sub in client.subscriptions],
                    "is_alive": client.is_alive()
                } for client_id, client in self.connections.items()
            }
        }


# Global WebSocket handler instance with production configuration
websocket_handler = WebSocketHandler(
    MessageFrequencyConfig(
        high_frequency_ms=100,   # 10 FPS for performance/market data
        medium_frequency_ms=500, # 2 FPS for trading/portfolio updates
        low_frequency_ms=2000,   # 0.5 FPS for component status
        heartbeat_frequency_ms=10000  # Every 10 seconds
    ),
    BackpressureConfig(
        max_queue_size=1000,
        drop_threshold=0.8,
        priority_preservation=True
    )
)


# Convenience functions for easy broadcasting of the 6 message types
async def broadcast_performance(performance_data: Dict[str, Any]):
    """Broadcast performance data (high frequency)"""
    await websocket_handler.broadcast_message(
        MessageType.PERFORMANCE, performance_data
    )


async def broadcast_trading_status(trading_data: Dict[str, Any]):
    """Broadcast trading status (medium frequency)"""
    await websocket_handler.broadcast_message(
        MessageType.TRADING_STATUS, trading_data
    )


async def broadcast_portfolio_update(portfolio_data: Dict[str, Any]):
    """Broadcast portfolio update (medium frequency)"""
    await websocket_handler.broadcast_message(
        MessageType.PORTFOLIO_UPDATE, portfolio_data
    )


async def broadcast_order_update(order_data: Dict[str, Any]):
    """Broadcast order update (immediate)"""
    await websocket_handler.broadcast_message(
        MessageType.ORDER_UPDATE, order_data
    )


async def broadcast_system_alert(message: str, severity: str = "info", **kwargs):
    """Broadcast system alert (immediate)"""
    alert_data = {"message": message, "severity": severity, **kwargs}
    await websocket_handler.broadcast_message(
        MessageType.SYSTEM_ALERT, alert_data
    )


async def broadcast_component_status(component_data: Dict[str, Any]):
    """Broadcast component status (low frequency)"""
    await websocket_handler.broadcast_message(
        MessageType.COMPONENT_STATUS, component_data
    )


# Handler lifecycle functions
async def start_websocket_handler():
    """Start the global WebSocket handler"""
    await websocket_handler.start()


async def stop_websocket_handler():
    """Stop the global WebSocket handler"""
    await websocket_handler.stop()


# Connection management
async def handle_websocket_connection(websocket: WebSocket, client_id: Optional[str] = None):
    """Handle WebSocket connection lifecycle with enhanced error handling"""
    client_id = await websocket_handler.connect_client(websocket, client_id)
    
    try:
        while True:
            # Receive messages from client
            message = await websocket.receive_text()
            await websocket_handler.handle_client_message(client_id, message)
            
    except WebSocketDisconnect:
        await websocket_handler.disconnect_client(client_id)
    except Exception as e:
        logger.error(f"WebSocket connection error for {client_id}: {e}")
        await websocket_handler.disconnect_client(client_id)


if __name__ == "__main__":
    # Enhanced demo with performance testing
    import asyncio
    import random
    
    async def demo():
        print("🔌 Supreme System V5 WebSocket Demo")
        print("=" * 50)
        
        handler = WebSocketHandler()
        await handler.start()
        
        # Simulate high-frequency performance data
        for i in range(10):
            await handler.broadcast_message(
                MessageType.PERFORMANCE,
                {
                    "latency_us": 0.26 + random.uniform(-0.05, 0.05),
                    "throughput_tps": 486000 + random.randint(-1000, 1000),
                    "accuracy_percent": 92.7 + random.uniform(-1.0, 1.0),
                    "timestamp": datetime.utcnow().isoformat()
                }
            )
            await asyncio.sleep(0.1)
        
        # Simulate trading status update
        await handler.broadcast_message(
            MessageType.TRADING_STATUS,
            {
                "state": "running",
                "active_pairs": ["BTC/USDT", "ETH/USDT"],
                "orders_executed": 42,
                "current_pnl": 125.50
            }
        )
        
        # Simulate order update
        await handler.broadcast_message(
            MessageType.ORDER_UPDATE,
            {
                "order_id": "ORD_1729123456789",
                "symbol": "BTC/USDT",
                "side": "buy",
                "quantity": 0.001,
                "price": 67500.00,
                "status": "filled"
            }
        )
        
        # Get statistics
        stats = handler.get_connection_stats()
        print(f"📊 WebSocket Statistics:")
        print(f"   Messages sent: {stats['messages_sent']}")
        print(f"   Messages dropped: {stats['messages_dropped']}")
        print(f"   Active connections: {stats['active_connections']}")
        print(f"   Messages/second: {stats['messages_per_second']:.1f}")
        
        await handler.stop()
        print("🚀 WebSocket demo completed")
    
    asyncio.run(demo())