#!/usr/bin/env python3
"""
Supreme System V5 - Utilities Module

Shared utilities, helpers, and common functionality.
"""

from .async_utils import (
    batch_process_async,
    gather_with_exception_handling,
    retry_async,
    run_async,
    run_concurrent_with_limit,
    run_periodic_task,
    run_with_deadline,
    run_with_timeout,
)
from .constants import (
    DEFAULT_ATR_PERIOD,
    DEFAULT_BB_PERIOD,
    DEFAULT_BB_STD,
    DEFAULT_CONFIG,
    DEFAULT_MACD_FAST,
    DEFAULT_MACD_SIGNAL,
    DEFAULT_MACD_SLOW,
    DEFAULT_MAX_DAILY_LOSS_PCT,
    DEFAULT_MAX_PORTFOLIO_DRAWDOWN,
    DEFAULT_MAX_POSITION_SIZE_PCT,
    DEFAULT_RSI_PERIOD,
    DEFAULT_STOP_LOSS_PCT,
    DEFAULT_TAKE_PROFIT_PCT,
    LOGGING_CONFIG,
    REQUIRED_OHLCV_COLUMNS,
    SUPPORTED_INTERVALS,
    SUPPORTED_SYMBOLS,
)
from .data_utils import (
    calculate_alpha,
    calculate_beta,
    calculate_correlation_matrix,
    calculate_drawdowns,
    calculate_returns,
    calculate_roll_max_drawdown,
    calculate_technical_indicators,
    chunk_dataframe,
    detect_outliers,
    find_highly_correlated_pairs,
    get_memory_usage_mb,
    handle_missing_data,
    normalize_data,
    optimize_dataframe_memory,
    resample_ohlcv,
    split_data_by_date,
    validate_and_clean_data,
)
from .exceptions import (
    BacktestDataError,
    BacktestError,
    CircuitBreakerError,
    ConfigurationError,
    ConnectionError,
    DataError,
    FileOperationError,
    InsufficientDataError,
    InsufficientFundsError,
    InvalidOrderError,
    MarketDataError,
    MaxDrawdownError,
    MonitoringError,
    NetworkError,
    PositionSizeError,
    RateLimitError,
    RiskError,
    SerializationError,
    StopLossError,
    StrategyError,
    StrategyTimeoutError,
    SupremeSystemError,
    TradingError,
    ValidationError,
)
from .helpers import (
    calculate_max_drawdown,
    calculate_moving_average,
    calculate_percentage_change,
    calculate_sharpe_ratio,
    calculate_volatility,
    detect_data_gaps,
    ensure_directory_exists,
    format_file_size,
    normalize_symbol,
    round_to_significant_digits,
    safe_divide,
    setup_logging,
    validate_config,
    validate_ohlcv_columns,
)

__all__ = [
    # Helper functions
    "setup_logging",
    "validate_config",
    "safe_divide",
    "calculate_percentage_change",

    # Constants
    "DEFAULT_CONFIG",
    "SUPPORTED_INTERVALS",
    "REQUIRED_OHLCV_COLUMNS",

    # Async utilities
    "run_async",
    "gather_with_exception_handling",

    # Data utilities
    "resample_ohlcv",
    "calculate_returns",
    "detect_outliers",

    # Exceptions
    "SupremeSystemError",
    "ConfigurationError",
    "DataError",
    "ValidationError",
    "TradingError",
    "RiskError",
    "NetworkError",
    "StrategyError",
    "BacktestError",
    "CircuitBreakerError",
    "InsufficientFundsError",
    "InvalidOrderError",
    "MarketDataError",
    "ConnectionError",
    "RateLimitError",
    "PositionSizeError",
    "StopLossError",
    "MaxDrawdownError",
    "StrategyTimeoutError",
    "BacktestDataError",
    "InsufficientDataError",
    "FileOperationError",
    "SerializationError",
    "MonitoringError"
]
