"""
Event Bus - Async Message Bus for Supreme System V5
ULTRA SFL implementation for scalable event-driven architecture
"""

import asyncio
import json
import time
from typing import Dict, List, Any, Optional, Callable, Awaitable
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict

from loguru import logger
from prometheus_client import Counter, Gauge, Histogram

# Metrics
EVENTS_PUBLISHED = Counter('events_published_total', 'Events published to bus', ['event_type'])
EVENTS_CONSUMED = Counter('events_consumed_total', 'Events consumed from bus', ['event_type', 'consumer'])
EVENT_BUS_LATENCY = Histogram('event_bus_latency_seconds', 'Event processing latency', ['operation'])
ACTIVE_SUBSCRIBERS = Gauge('active_subscribers', 'Active subscribers per topic', ['topic'])

class EventPriority(Enum):
    """Event priority levels"""
    LOW = 0
    NORMAL = 1
    HIGH = 2
    CRITICAL = 3

@dataclass
class Event:
    """Event structure for message bus"""
    id: str
    type: str
    timestamp: float
    priority: EventPriority
    data: Dict[str, Any]
    source: str
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """Validate event data"""
        if not self.id or not self.type:
            raise ValueError("Event must have id and type")

        # Ensure timestamp is set
        if not self.timestamp:
            self.timestamp = time.time()

@dataclass
class Subscription:
    """Subscription configuration"""
    consumer_id: str
    callback: Callable[[Event], Awaitable[None]]
    filter_func: Optional[Callable[[Event], bool]] = None
    priority: int = 0  # Higher priority processed first

class EventBus:
    """
    Async Event Bus for Supreme System V5
    ULTRA SFL implementation with priority queues and filtering
    """

    def __init__(self, max_queue_size: int = 10000):
        self.max_queue_size = max_queue_size

        # Subscription management
        self.subscriptions: Dict[str, List[Subscription]] = defaultdict(list)

        # Priority queues for different event types
        self.queues: Dict[EventPriority, asyncio.Queue] = {
            priority: asyncio.Queue(maxsize=max_queue_size)
            for priority in EventPriority
        }

        # Processing state
        self.running = False
        self.tasks: List[asyncio.Task] = []
        self.processing_lock = asyncio.Lock()

        logger.info("🎯 Event Bus initialized with priority queues")

    async def start(self):
        """Start the event bus processing"""
        if self.running:
            return

        self.running = True
        logger.info("🚀 Starting Event Bus processing")

        # Create processing tasks for each priority level
        for priority in EventPriority:
            task = asyncio.create_task(self._process_queue(priority))
            self.tasks.append(task)

        # Create metrics update task
        metrics_task = asyncio.create_task(self._update_metrics())
        self.tasks.append(metrics_task)

        logger.info(f"✅ Event Bus started with {len(self.tasks)} processing tasks")

    async def stop(self):
        """Stop the event bus processing"""
        if not self.running:
            return

        logger.info("🛑 Stopping Event Bus...")
        self.running = False

        # Cancel all tasks
        for task in self.tasks:
            task.cancel()

        # Wait for tasks to complete
        try:
            await asyncio.gather(*self.tasks, return_exceptions=True)
        except asyncio.CancelledError:
            pass

        # Clear queues
        for queue in self.queues.values():
            while not queue.empty():
                try:
                    queue.get_nowait()
                except asyncio.QueueEmpty:
                    break

        self.tasks.clear()
        logger.info("✅ Event Bus stopped")

    async def publish(self, event: Event) -> bool:
        """
        Publish an event to the bus
        Returns True if published successfully
        """
        start_time = time.time()

        try:
            # Add to appropriate priority queue
            await asyncio.wait_for(
                self.queues[event.priority].put(event),
                timeout=5.0  # 5 second timeout
            )

            EVENTS_PUBLISHED.labels(event_type=event.type).inc()
            EVENT_BUS_LATENCY.labels(operation='publish').observe(time.time() - start_time)

            logger.debug(f"📤 Published event: {event.type} ({event.id}) priority={event.priority.value}")
            return True

        except asyncio.TimeoutError:
            logger.warning(f"⚠️ Event bus queue full for priority {event.priority.value}, dropping event {event.id}")
            return False
        except Exception as e:
            logger.error(f"❌ Failed to publish event {event.id}: {e}")
            return False

    def subscribe(self, topic: str, consumer_id: str, callback: Callable[[Event], Awaitable[None]],
                  filter_func: Optional[Callable[[Event], bool]] = None, priority: int = 0):
        """
        Subscribe to a topic with filtering capability
        """
        subscription = Subscription(
            consumer_id=consumer_id,
            callback=callback,
            filter_func=filter_func,
            priority=priority
        )

        self.subscriptions[topic].append(subscription)

        # Sort subscriptions by priority (higher first)
        self.subscriptions[topic].sort(key=lambda s: s.priority, reverse=True)

        ACTIVE_SUBSCRIBERS.labels(topic=topic).set(len(self.subscriptions[topic]))

        logger.info(f"📡 {consumer_id} subscribed to topic '{topic}' (priority: {priority})")

    def unsubscribe(self, topic: str, consumer_id: str):
        """Unsubscribe from a topic"""
        if topic in self.subscriptions:
            original_count = len(self.subscriptions[topic])
            self.subscriptions[topic] = [
                sub for sub in self.subscriptions[topic]
                if sub.consumer_id != consumer_id
            ]

            removed = original_count - len(self.subscriptions[topic])
            if removed > 0:
                ACTIVE_SUBSCRIBERS.labels(topic=topic).set(len(self.subscriptions[topic]))
                logger.info(f"📴 {consumer_id} unsubscribed from topic '{topic}' ({removed} subscriptions removed)")

    async def _process_queue(self, priority: EventPriority):
        """Process events from a specific priority queue"""
        logger.info(f"⚙️ Started processing queue for priority {priority.value}")

        while self.running:
            try:
                # Get event from queue
                event = await self.queues[priority].get()

                # Process the event
                await self._process_event(event)

                # Mark task as done
                self.queues[priority].task_done()

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"❌ Error processing event from priority {priority.value} queue: {e}")
                await asyncio.sleep(1)  # Brief pause before retry

        logger.info(f"⚙️ Stopped processing queue for priority {priority.value}")

    async def _process_event(self, event: Event):
        """Process a single event by delivering to subscribers"""
        start_time = time.time()

        # Find matching topics (event.type can be used as topic)
        topics_to_check = [event.type]

        # Also check for wildcard subscriptions
        if '*' in self.subscriptions:
            topics_to_check.append('*')

        delivered_count = 0

        for topic in topics_to_check:
            if topic not in self.subscriptions:
                continue

            # Get subscriptions sorted by priority
            subscriptions = self.subscriptions[topic]

            for subscription in subscriptions:
                try:
                    # Apply filter if specified
                    if subscription.filter_func and not subscription.filter_func(event):
                        continue

                    # Deliver event to subscriber
                    await subscription.callback(event)
                    delivered_count += 1

                    EVENTS_CONSUMED.labels(
                        event_type=event.type,
                        consumer=subscription.consumer_id
                    ).inc()

                except Exception as e:
                    logger.error(f"❌ Subscriber {subscription.consumer_id} failed to process event {event.id}: {e}")
                    continue

        processing_time = time.time() - start_time
        EVENT_BUS_LATENCY.labels(operation='process').observe(processing_time)

        if delivered_count > 0:
            logger.debug(f"✅ Event {event.id} processed in {processing_time:.4f}s, delivered to {delivered_count} subscribers")
        else:
            logger.debug(f"⚠️ Event {event.id} processed but no subscribers matched")

    async def _update_metrics(self):
        """Update metrics periodically"""
        while self.running:
            try:
                # Update queue sizes
                for priority in EventPriority:
                    queue_size = self.queues[priority].qsize()
                    # Could add gauge metric here if needed

                await asyncio.sleep(10)  # Update every 10 seconds

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"❌ Error updating metrics: {e}")
                await asyncio.sleep(5)

    def get_stats(self) -> Dict[str, Any]:
        """Get event bus statistics"""
        stats = {
            'running': self.running,
            'topics': len(self.subscriptions),
            'total_subscriptions': sum(len(subs) for subs in self.subscriptions.values()),
            'queue_sizes': {
                priority.value: self.queues[priority].qsize()
                for priority in EventPriority
            }
        }

        # Add topic details
        topic_details = {}
        for topic, subscriptions in self.subscriptions.items():
            topic_details[topic] = {
                'subscribers': len(subscriptions),
                'subscriber_ids': [sub.consumer_id for sub in subscriptions]
            }
        stats['topics_detail'] = topic_details

        return stats

# Convenience functions for common event types

def create_market_data_event(symbol: str, price: float, volume: float,
                           source: str, **kwargs) -> Event:
    """Create a market data event"""
    return Event(
        id=f"market_{symbol}_{int(time.time() * 1000)}",
        type="market_data",
        timestamp=time.time(),
        priority=EventPriority.NORMAL,
        data={
            'symbol': symbol,
            'price': price,
            'volume': volume,
            **kwargs
        },
        source=source
    )

def create_signal_event(symbol: str, signal_type: str, confidence: float,
                       entry_price: float, **kwargs) -> Event:
    """Create a trading signal event"""
    return Event(
        id=f"signal_{symbol}_{int(time.time() * 1000)}",
        type="trading_signal",
        timestamp=time.time(),
        priority=EventPriority.HIGH,
        data={
            'symbol': symbol,
            'signal_type': signal_type,
            'confidence': confidence,
            'entry_price': entry_price,
            **kwargs
        },
        source="strategy"
    )

def create_risk_event(violation_type: str, symbol: str, details: Dict[str, Any]) -> Event:
    """Create a risk violation event"""
    return Event(
        id=f"risk_{violation_type}_{symbol}_{int(time.time() * 1000)}",
        type="risk_violation",
        timestamp=time.time(),
        priority=EventPriority.CRITICAL,
        data={
            'violation_type': violation_type,
            'symbol': symbol,
            **details
        },
        source="risk_manager"
    )

def create_execution_event(order_id: str, symbol: str, side: str, quantity: float,
                          price: float, **kwargs) -> Event:
    """Create an order execution event"""
    return Event(
        id=f"execution_{order_id}",
        type="order_execution",
        timestamp=time.time(),
        priority=EventPriority.HIGH,
        data={
            'order_id': order_id,
            'symbol': symbol,
            'side': side,
            'quantity': quantity,
            'price': price,
            **kwargs
        },
        source="execution"
    )

# Global event bus instance
_event_bus: Optional[EventBus] = None

def get_event_bus() -> EventBus:
    """Get the global event bus instance"""
    global _event_bus
    if _event_bus is None:
        _event_bus = EventBus()
    return _event_bus

async def init_event_bus():
    """Initialize the global event bus"""
    global _event_bus
    if _event_bus is None:
        _event_bus = EventBus()
    await _event_bus.start()

async def shutdown_event_bus():
    """Shutdown the global event bus"""
    global _event_bus
    if _event_bus:
        await _event_bus.stop()
        _event_bus = None
