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

__all__ = [
    "get_logger",
    "SupremeCore",
    "SupremeSystem",
    "SystemConfig",
    "ScalpingStrategy",
    "RiskManager",
    "RUST_AVAILABLE",
    "__version__",
    "__author__",
]
