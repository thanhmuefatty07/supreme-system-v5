#!/usr/bin/env python3
"""
Supreme System V5 - Utilities Module

Shared utilities, helpers, and common functionality.
"""

from .helpers import (
    setup_logging, validate_config, safe_divide, calculate_percentage_change,
    calculate_moving_average, calculate_volatility, detect_data_gaps,
    ensure_directory_exists, format_file_size, validate_ohlcv_columns,
    normalize_symbol, calculate_sharpe_ratio, calculate_max_drawdown,
    round_to_significant_digits
)
from .constants import (
    REQUIRED_OHLCV_COLUMNS, SUPPORTED_INTERVALS, SUPPORTED_SYMBOLS,
    DEFAULT_STOP_LOSS_PCT, DEFAULT_TAKE_PROFIT_PCT, DEFAULT_MAX_POSITION_SIZE_PCT,
    DEFAULT_MAX_DAILY_LOSS_PCT, DEFAULT_MAX_PORTFOLIO_DRAWDOWN, DEFAULT_MACD_FAST,
    DEFAULT_MACD_SLOW, DEFAULT_MACD_SIGNAL, DEFAULT_RSI_PERIOD, DEFAULT_BB_PERIOD,
    DEFAULT_BB_STD, DEFAULT_ATR_PERIOD, DEFAULT_CONFIG, LOGGING_CONFIG
)
from .async_utils import (
    run_async, gather_with_exception_handling, run_concurrent_with_limit,
    retry_async, run_with_timeout, batch_process_async, run_periodic_task,
    run_with_deadline
)
from .data_utils import (
    optimize_dataframe_memory, chunk_dataframe, get_memory_usage_mb,
    validate_and_clean_data, resample_ohlcv, calculate_returns, detect_outliers,
    calculate_technical_indicators, normalize_data, handle_missing_data,
    calculate_correlation_matrix, find_highly_correlated_pairs, calculate_drawdowns,
    calculate_roll_max_drawdown, split_data_by_date, calculate_beta, calculate_alpha
)
from .exceptions import (
    SupremeSystemError, ConfigurationError, DataError, ValidationError, TradingError,
    RiskError, NetworkError, StrategyError, BacktestError, CircuitBreakerError,
    InsufficientFundsError, InvalidOrderError, MarketDataError, ConnectionError,
    RateLimitError, PositionSizeError, StopLossError, MaxDrawdownError,
    StrategyTimeoutError, BacktestDataError, InsufficientDataError, FileOperationError,
    SerializationError, MonitoringError
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
