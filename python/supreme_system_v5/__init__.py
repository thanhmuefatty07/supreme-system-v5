"""
🚀 Supreme System V5 - Hybrid Python + Rust Trading System
World's First Neuromorphic Trading System

This package provides a high-performance trading system that combines:
- Python for orchestration, ML, and strategy development
- Rust for ultra-fast computational kernels
- Real-time data integration
- Advanced backtesting capabilities
- Neuromorphic computing algorithms
"""

__version__ = "5.0.0"
__author__ = "Supreme Trading Team"
__email__ = "info@supremetrading.ai"
__description__ = "World's First Neuromorphic Trading System - Hybrid Python + Rust"

# Import Rust engine (compiled from Rust code)
try:
    import supreme_engine_rs as rust_engine
    RUST_ENGINE_AVAILABLE = True
    __rust_version__ = getattr(rust_engine, '__version__', 'unknown')
except ImportError as e:
    RUST_ENGINE_AVAILABLE = False
    __rust_version__ = None
    import warnings
    warnings.warn(
        f"Rust engine not available: {e}. "
        "Using Python fallbacks. Performance will be reduced."
    )

# Core modules
from .core import (
    TradingSystem,
    HardwareDetector,
    PerformanceMonitor,
)

from .data import (
    DataConnector,
    RealTimeProvider,
    HistoricalProvider,
)

from .strategies import (
    StrategyBase,
    NeuromorphicStrategy,
    TechnicalStrategy,
    HybridStrategy,
)

from .backtesting import (
    BacktestEngine,
    PerformanceAnalyzer,
    RiskAnalyzer,
)

from .risk import (
    RiskManager,
    PositionSizer,
    PortfolioOptimizer,
)

from .utils import (
    Logger,
    Config,
    Validators,
)

# Export main components
__all__ = [
    # Core
    'TradingSystem',
    'HardwareDetector', 
    'PerformanceMonitor',
    
    # Data
    'DataConnector',
    'RealTimeProvider',
    'HistoricalProvider',
    
    # Strategies
    'StrategyBase',
    'NeuromorphicStrategy',
    'TechnicalStrategy',
    'HybridStrategy',
    
    # Backtesting
    'BacktestEngine',
    'PerformanceAnalyzer',
    'RiskAnalyzer',
    
    # Risk Management
    'RiskManager',
    'PositionSizer',
    'PortfolioOptimizer',
    
    # Utils
    'Logger',
    'Config',
    'Validators',
    
    # Constants
    'RUST_ENGINE_AVAILABLE',
    '__rust_version__',
]

# Hardware detection at import time
_hardware_detector = HardwareDetector()
HARDWARE_PROFILE = _hardware_detector.detect_hardware()
OPTIMIZED_FOR_HARDWARE = True

# Performance monitoring
_perf_monitor = PerformanceMonitor(HARDWARE_PROFILE)

def get_system_info():
    """Get comprehensive system information"""
    return {
        'version': __version__,
        'rust_engine_available': RUST_ENGINE_AVAILABLE,
        'rust_version': __rust_version__,
        'hardware_profile': HARDWARE_PROFILE,
        'optimized_for_hardware': OPTIMIZED_FOR_HARDWARE,
        'performance_targets': _perf_monitor.get_targets(),
    }

def benchmark_system(duration_seconds: int = 10):
    """Run system benchmark"""
    return _perf_monitor.run_benchmark(duration_seconds)

# Version compatibility checks
if RUST_ENGINE_AVAILABLE:
    rust_target_latency = getattr(rust_engine, 'TARGET_LATENCY_US', 10.0)
    if rust_target_latency > HARDWARE_PROFILE.target_latency_us:
        warnings.warn(
            f"Rust engine target latency ({rust_target_latency}μs) exceeds "
            f"hardware profile target ({HARDWARE_PROFILE.target_latency_us}μs)"
        )

# Initialize logging
logger = Logger(__name__)
logger.info(f"Supreme System V5 {__version__} initialized")
logger.info(f"Hardware: {HARDWARE_PROFILE.processor_type} + {HARDWARE_PROFILE.memory_profile}")
logger.info(f"Rust engine: {'✅ Available' if RUST_ENGINE_AVAILABLE else '❌ Not available'}")
logger.info(f"Target latency: {HARDWARE_PROFILE.target_latency_us}μs")
