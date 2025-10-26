"""
⚙️ Supreme System V5 - Configuration Module
Hardware-aware configuration management
"""

from .hardware_profiles import (
    HardwareDetector,
    HardwareProfile,
    MemoryProfile,
    PerformanceOptimizer,
    ProcessorType,
    hardware_detector,
    optimal_profile,
    performance_optimizer,
)

__version__ = "5.0.0"
__all__ = [
    "HardwareDetector",
    "HardwareProfile",
    "PerformanceOptimizer",
    "ProcessorType",
    "MemoryProfile",
    "hardware_detector",
    "optimal_profile",
    "performance_optimizer",
]
