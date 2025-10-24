"""
⚡ Supreme System V5 - Ultra-Low Latency Module
Sub-microsecond processing for high-frequency trading

Features:
- Lock-free algorithms
- Zero-copy memory management
- Hardware optimization
- FPGA-ready implementation
"""

__version__ = "5.0.0"
__author__ = "Supreme System V5 Team"

from .processor import UltraLowLatencyProcessor, LatencyConfig
from .algorithms import LockFreeQueue, CircularBuffer
from .memory_manager import ZeroCopyManager

# Export main classes
UltraLowLatencyEngine = UltraLowLatencyProcessor  # Alias

__all__ = [
    "UltraLowLatencyProcessor",
    "UltraLowLatencyEngine",
    "LatencyConfig",
    "LockFreeQueue",
    "CircularBuffer",
    "ZeroCopyManager"
]