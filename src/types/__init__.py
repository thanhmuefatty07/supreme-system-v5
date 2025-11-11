"""
Type definitions for Supreme System V5.

Provides comprehensive type hints and TypedDict definitions for type safety.
"""

from .trading_types import (
    # Basic types
    Symbol, Price, Volume, Timestamp, OrderAction, OrderStatus, OrderType,
    SignalStrength, Timeframe,

    # Data structures
    OHLCVData, OrderData, TradeResult, SignalData, PortfolioData,
    RiskMetrics, PerformanceMetrics, StrategyConfig, BacktestConfig,
    BacktestResult, ValidationResult, IndicatorResult, DataSourceConfig,

    # Classes
    MarketData, TradingSignal, Position,

    # Protocols
    TradingStrategyProtocol, DataProviderProtocol, RiskManagerProtocol,
    OrderExecutorProtocol,

    # Type aliases
    IndicatorValues, StrategyParameters, ConfigurationDict, MetricsDict,
    ResultType, ErrorContext, ErrorCode,

    # Utility functions
    validate_ohlcv_data, validate_signal_data, validate_order_data,
    create_market_data, create_trading_signal,
)

__all__ = [
    # Basic types
    'Symbol', 'Price', 'Volume', 'Timestamp', 'OrderAction', 'OrderStatus', 'OrderType',
    'SignalStrength', 'Timeframe',

    # Data structures
    'OHLCVData', 'OrderData', 'TradeResult', 'SignalData', 'PortfolioData',
    'RiskMetrics', 'PerformanceMetrics', 'StrategyConfig', 'BacktestConfig',
    'BacktestResult', 'ValidationResult', 'IndicatorResult', 'DataSourceConfig',

    # Classes
    'MarketData', 'TradingSignal', 'Position',

    # Protocols
    'TradingStrategyProtocol', 'DataProviderProtocol', 'RiskManagerProtocol',
    'OrderExecutorProtocol',

    # Type aliases
    'IndicatorValues', 'StrategyParameters', 'ConfigurationDict', 'MetricsDict',
    'ResultType', 'ErrorContext', 'ErrorCode',

    # Utility functions
    'validate_ohlcv_data', 'validate_signal_data', 'validate_order_data',
    'create_market_data', 'create_trading_signal',
]
