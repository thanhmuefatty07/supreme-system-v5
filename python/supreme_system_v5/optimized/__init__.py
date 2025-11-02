"""
Ultra-optimized trading indicators and data structures.
Memory-efficient, CPU-optimized for i3-4GB systems.
"""

from .circular_buffer import CircularBuffer, RollingAverage
from .ema import UltraOptimizedEMA
from .rsi import UltraOptimizedRSI
from .macd import UltraOptimizedMACD
from .smart_events import SmartEventProcessor
from .analyzer import OptimizedTechnicalAnalyzer

__all__ = [
    'CircularBuffer', 'RollingAverage',
    'UltraOptimizedEMA', 'UltraOptimizedRSI', 'UltraOptimizedMACD',
    'SmartEventProcessor', 'OptimizedTechnicalAnalyzer'
]
