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

from .core import SupremeCore, SupremeSystem, SystemConfig, MarketData
from .event_bus import (
    Event,
    EventBus,
    EventPriority,
    Subscription,
    create_execution_event,
    create_market_data_event,
    create_risk_event,
    create_signal_event,
    get_event_bus,
)
from .risk import RiskManager
from .strategies import ScalpingStrategy

# Import backtest components
from .backtest import run_realtime_backtest, BacktestConfig, BacktestMetrics

# Import optimized algorithms
from .algorithms.ultra_optimized_indicators import (
    UltraOptimizedEMA,
    UltraOptimizedRSI,
    UltraOptimizedMACD,
    CircularBuffer,
    SmartEventProcessor,
    IndicatorResult,
    benchmark_indicators,
)

# Import symbol analysis
from .symbol_analysis import (
    SymbolAnalyzer,
    SingleSymbolConfig,
    OPTIMAL_TRADING_SYMBOL,
    SYMBOL_CONFIG,
    analyze_all_symbols,
)

# Import resource allocation
from .resource_allocation import (
    ResourceAllocator,
    SystemResourceMonitor,
    RESOURCE_CONFIG,
    create_single_symbol_resource_config,
    validate_system_resources,
    print_resource_allocation_report,
)

# Import news APIs
from .news_apis import (
    NewsAggregator,
    EconomicDataFetcher,
    WhaleTracker,
    APIManager,
    NewsItem,
    test_all_apis,
    demo_news_fetching,
)

# Import news classifier
from .news_classifier import (
    AdvancedNewsClassifier,
    NewsImpactPredictor,
    ClassifiedNews,
    ImpactCategory,
    NewsSentiment,
    demo_news_classification,
)

# Import whale tracking
from .whale_tracking import (
    WhaleTrackingSystem,
    WhaleTransaction,
    WhaleActivityMetrics,
    demo_whale_tracking,
)

# Import dynamic risk manager
from .dynamic_risk_manager import (
    DynamicRiskManager,
    OptimalPosition,
    RiskLevel,
    LeverageLevel,
    PortfolioState,
    SignalConfidence,
    demo_dynamic_risk_management,
)

# Import pattern recognition
from .pattern_recognition import (
    AdvancedPatternRecognition,
    PatternResult,
    PatternType,
    PatternDirection,
    Candlestick,
    demo_pattern_recognition,
)

# Import multi-timeframe engine
from .multi_timeframe_engine import (
    MultiTimeframeEngine,
    ConsensusSignal,
    Timeframe,
    TimeframeData,
    demo_multi_timeframe_engine,
)

# Import master orchestrator
from .master_orchestrator import (
    MasterTradingOrchestrator,
    OrchestrationResult,
    TradingDecision,
    ComponentPriority,
    ComponentStatus,
    demo_master_orchestrator,
)

# Import resource monitor
from .resource_monitor import (
    AdvancedResourceMonitor,
    ResourceMetrics,
    ResourceThreshold,
    OptimizationResult,
    PerformanceProfile,
    demo_resource_monitor,
)

# Import integration testing
from .integration_test import (
    IntegrationTestSuite,
    run_integration_tests,
)

# Import Python components
from .utils import get_logger

__all__ = [
    "get_logger",
    "SupremeCore",
    "SupremeSystem",
    "SystemConfig",
    "MarketData",
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
    "run_realtime_backtest",
    "BacktestConfig",
    "BacktestMetrics",
    # Optimized Indicators
    "UltraOptimizedEMA",
    "UltraOptimizedRSI",
    "UltraOptimizedMACD",
    "CircularBuffer",
    "SmartEventProcessor",
    "IndicatorResult",
    "benchmark_indicators",
    # Symbol Analysis
    "SymbolAnalyzer",
    "SingleSymbolConfig",
    "OPTIMAL_TRADING_SYMBOL",
    "SYMBOL_CONFIG",
    "analyze_all_symbols",
    # Resource Allocation
    "ResourceAllocator",
    "SystemResourceMonitor",
    "RESOURCE_CONFIG",
    "create_single_symbol_resource_config",
    "validate_system_resources",
    "print_resource_allocation_report",
    # News APIs
    "NewsAggregator",
    "EconomicDataFetcher",
    "WhaleTracker",
    "APIManager",
    "NewsItem",
    "test_all_apis",
    "demo_news_fetching",
    # News Classifier
    "AdvancedNewsClassifier",
    "NewsImpactPredictor",
    "ClassifiedNews",
    "ImpactCategory",
    "NewsSentiment",
    "demo_news_classification",
    # Whale Tracking
    "WhaleTrackingSystem",
    "WhaleTransaction",
    "WhaleActivityMetrics",
    "demo_whale_tracking",
    # Dynamic Risk Manager
    "DynamicRiskManager",
    "OptimalPosition",
    "RiskLevel",
    "LeverageLevel",
    "PortfolioState",
    "SignalConfidence",
    "demo_dynamic_risk_management",
    # Pattern Recognition
    "AdvancedPatternRecognition",
    "PatternResult",
    "PatternType",
    "PatternDirection",
    "Candlestick",
    "demo_pattern_recognition",
    # Multi-Timeframe Engine
    "MultiTimeframeEngine",
    "ConsensusSignal",
    "Timeframe",
    "TimeframeData",
    "demo_multi_timeframe_engine",
    # Master Orchestrator
    "MasterTradingOrchestrator",
    "OrchestrationResult",
    "TradingDecision",
    "ComponentPriority",
    "ComponentStatus",
    "demo_master_orchestrator",
    # Resource Monitor
    "AdvancedResourceMonitor",
    "ResourceMetrics",
    "ResourceThreshold",
    "OptimizationResult",
    "PerformanceProfile",
    "demo_resource_monitor",
    # Integration Testing
    "IntegrationTestSuite",
    "run_integration_tests",
    "RUST_AVAILABLE",
    "__version__",
    "__author__",
]
