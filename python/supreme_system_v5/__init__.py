# python/supreme_system_v5/__init__.py
"""
Supreme System V5 - Minimal Hybrid Python + Rust Architecture
"""

__version__ = "5.0.0"
__author__ = "Supreme System Team"

# Note: Rust engine integration will be added later
# For now, we provide minimal Python stubs
RUST_AVAILABLE = False
RustEngine = None

# Import Python components
from .utils import get_logger
from .core import SupremeCore, SupremeSystem, SystemConfig
from .strategies import ScalpingStrategy
from .risk import RiskManager
from .event_bus import (
    EventBus, Event, EventPriority, Subscription,
    get_event_bus, create_market_data_event, create_signal_event,
    create_risk_event, create_execution_event
)

__all__ = [
    "get_logger",
    "SupremeCore",
    "SupremeSystem",
    "SystemConfig",
    "ScalpingStrategy",
    "RiskManager",
    "EventBus",
    "Event",
    "EventPriority",
    "Subscription",
    "get_event_bus",
    "create_market_data_event",
    "create_signal_event",
    "create_risk_event",
    "create_execution_event",
    "RUST_AVAILABLE",
    "__version__",
    "__author__",
]
