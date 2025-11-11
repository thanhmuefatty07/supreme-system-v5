#!/usr/bin/env python3
"""
Supreme System V5 - Custom Exceptions

Comprehensive exception hierarchy for the trading system.
"""

from typing import Dict, Any, Optional


class SupremeSystemError(Exception):
    """Base exception for Supreme System V5."""

    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(message)
        self.message = message
        self.details = details or {}

    def __str__(self) -> str:
        if self.details:
            return f"{self.message} - Details: {self.details}"
        return self.message


class ConfigurationError(SupremeSystemError):
    """Raised when there are configuration-related errors."""
    pass


class DataError(SupremeSystemError):
    """Raised when there are data-related errors."""
    pass


class ValidationError(SupremeSystemError):
    """Raised when data validation fails."""
    pass


class TradingError(SupremeSystemError):
    """Raised when trading operations fail."""
    pass


class RiskError(SupremeSystemError):
    """Raised when risk management violations occur."""
    pass


class NetworkError(SupremeSystemError):
    """Raised when network/API operations fail."""
    pass


class StrategyError(SupremeSystemError):
    """Raised when strategy execution fails."""
    pass


class BacktestError(SupremeSystemError):
    """Raised when backtesting operations fail."""
    pass


class CircuitBreakerError(SupremeSystemError):
    """Raised when circuit breaker is triggered."""
    pass


class InsufficientFundsError(TradingError):
    """Raised when insufficient funds for trading."""
    pass


class InvalidOrderError(TradingError):
    """Raised when order parameters are invalid."""
    pass


class MarketDataError(DataError):
    """Raised when market data is unavailable or invalid."""
    pass


class ConnectionError(NetworkError):
    """Raised when connection to external services fails."""
    pass


class RateLimitError(NetworkError):
    """Raised when API rate limits are exceeded."""
    pass


class PositionSizeError(RiskError):
    """Raised when position size violates risk limits."""
    pass


class StopLossError(RiskError):
    """Raised when stop loss conditions are triggered."""
    pass


class MaxDrawdownError(RiskError):
    """Raised when maximum drawdown limits are exceeded."""
    pass


class StrategyTimeoutError(StrategyError):
    """Raised when strategy execution times out."""
    pass


class BacktestDataError(BacktestError):
    """Raised when backtest data is insufficient or invalid."""
    pass


class InsufficientDataError(DataError):
    """Raised when insufficient data for operations."""
    pass


class FileOperationError(SupremeSystemError):
    """Raised when file operations fail."""
    pass


class SerializationError(SupremeSystemError):
    """Raised when serialization/deserialization fails."""
    pass


class MonitoringError(SupremeSystemError):
    """Raised when monitoring operations fail."""
    pass


# Exception mapping for error handling
EXCEPTION_MAPPING = {
    'configuration': ConfigurationError,
    'data': DataError,
    'validation': ValidationError,
    'trading': TradingError,
    'risk': RiskError,
    'network': NetworkError,
    'strategy': StrategyError,
    'backtest': BacktestError,
    'circuit_breaker': CircuitBreakerError,
    'insufficient_funds': InsufficientFundsError,
    'invalid_order': InvalidOrderError,
    'market_data': MarketDataError,
    'connection': ConnectionError,
    'rate_limit': RateLimitError,
    'position_size': PositionSizeError,
    'stop_loss': StopLossError,
    'max_drawdown': MaxDrawdownError,
    'strategy_timeout': StrategyTimeoutError,
    'backtest_data': BacktestDataError,
    'insufficient_data': InsufficientDataError,
    'file_operation': FileOperationError,
    'serialization': SerializationError,
    'monitoring': MonitoringError
}


def create_exception(error_type: str, message: str, details: Optional[Dict[str, Any]] = None) -> SupremeSystemError:
    """
    Create an exception of the appropriate type.

    Args:
        error_type: Type of error (key from EXCEPTION_MAPPING)
        message: Error message
        details: Additional error details

    Returns:
        Appropriate exception instance

    Raises:
        ValueError: If error_type is not recognized
    """
    exception_class = EXCEPTION_MAPPING.get(error_type)
    if exception_class is None:
        raise ValueError(f"Unknown error type: {error_type}")

    return exception_class(message, details)


def handle_exception(exc: Exception, logger: Optional[Any] = None, re_raise: bool = True) -> None:
    """
    Handle an exception with proper logging and re-raising.

    Args:
        exc: Exception to handle
        logger: Logger instance for logging
        re_raise: Whether to re-raise the exception
    """
    if logger:
        if isinstance(exc, SupremeSystemError):
            logger.error(f"Supreme System Error: {exc}")
        else:
            logger.error(f"Unexpected error: {exc}", exc_info=True)

    if re_raise:
        raise exc


__all__ = [
    'SupremeSystemError',
    'ConfigurationError',
    'DataError',
    'ValidationError',
    'TradingError',
    'RiskError',
    'NetworkError',
    'StrategyError',
    'BacktestError',
    'CircuitBreakerError',
    'InsufficientFundsError',
    'InvalidOrderError',
    'MarketDataError',
    'ConnectionError',
    'RateLimitError',
    'PositionSizeError',
    'StopLossError',
    'MaxDrawdownError',
    'StrategyTimeoutError',
    'BacktestDataError',
    'InsufficientDataError',
    'FileOperationError',
    'SerializationError',
    'MonitoringError',
    'EXCEPTION_MAPPING',
    'create_exception',
    'handle_exception'
]

