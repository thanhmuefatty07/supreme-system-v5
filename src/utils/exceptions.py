#!/usr/bin/env python3
"""
Supreme System V5 - Custom Exceptions

Comprehensive exception hierarchy with advanced error handling,
decorators, and context management for production-grade reliability.
"""

import logging
from typing import Any, Dict, Optional, List
from datetime import datetime
from functools import wraps


class SupremeSystemError(Exception):
    """
    Base exception for Supreme System V5 with advanced context and logging.

    Provides structured error information, automatic logging, and recovery suggestions
    for production-grade error handling.
    """

    def __init__(
        self,
        message: str,
        error_code: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None,
        recovery_suggestions: Optional[List[str]] = None,
        log_level: str = "ERROR",
        details: Optional[Dict[str, Any]] = None  # Backward compatibility
    ):
        super().__init__(message)

        self.message = message
        self.error_code = error_code or self._generate_error_code()
        self.context = context or details or {}  # Support both context and details
        self.recovery_suggestions = recovery_suggestions or []
        self.log_level = log_level
        self.timestamp = datetime.now()
        self.details = self.context  # Backward compatibility

        # Auto-log the error
        self._log_error()

    def _generate_error_code(self) -> str:
        """Generate a unique error code based on exception type."""
        class_name = self.__class__.__name__
        timestamp = self.timestamp.strftime("%H%M%S")
        return f"{class_name[:3].upper()}{timestamp}"

    def _log_error(self):
        """Log the error with context."""
        logger = logging.getLogger(self.__class__.__module__)

        log_message = f"[{self.error_code}] {self.message}"
        if self.context:
            log_message += f" | Context: {self.context}"

        if self.recovery_suggestions:
            log_message += f" | Suggestions: {self.recovery_suggestions}"

        logger.log(getattr(logging, self.log_level), log_message)

    def to_dict(self) -> Dict[str, Any]:
        """Convert exception to dictionary for serialization."""
        return {
            "error_code": self.error_code,
            "message": self.message,
            "exception_type": self.__class__.__name__,
            "timestamp": self.timestamp.isoformat(),
            "context": self.context,
            "recovery_suggestions": self.recovery_suggestions,
            "log_level": self.log_level
        }

    def __str__(self) -> str:
        if self.context:
            return f"[{self.error_code}] {self.message} - Context: {self.context}"
        return f"[{self.error_code}] {self.message}"


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


# Advanced Error Handling Decorators

def handle_errors(operation: str = None, log_level: str = "ERROR", reraise: bool = True):
    """
    Decorator for comprehensive error handling.

    Args:
        operation: Description of the operation
        log_level: Logging level for errors
        reraise: Whether to re-raise exceptions after handling

    Returns:
        Decorated function with error handling
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            op_name = operation or f"{func.__module__}.{func.__qualname__}"
            logger = logging.getLogger(func.__module__)

            try:
                logger.debug(f"Starting operation: {op_name}")
                result = func(*args, **kwargs)
                logger.debug(f"Operation completed: {op_name}")
                return result

            except SupremeSystemError as e:
                # Already logged by the exception itself
                if reraise:
                    raise
                return None

            except Exception as e:
                # Wrap unexpected exceptions
                error = SupremeSystemError(
                    f"Unexpected error in {op_name}: {str(e)}",
                    context={
                        "function": func.__qualname__,
                        "args_count": len(args),
                        "kwargs_keys": list(kwargs.keys())
                    },
                    log_level=log_level
                )

                if reraise:
                    raise error from e
                return None

        return wrapper
    return decorator


def retry_on_failure(max_attempts: int = 3, delay: float = 1.0, backoff: float = 2.0,
                    exceptions: tuple = (NetworkError, ConnectionError, RateLimitError)):
    """
    Decorator for retrying operations on failure.

    Args:
        max_attempts: Maximum number of retry attempts
        delay: Initial delay between retries
        backoff: Backoff multiplier for delay
        exceptions: Tuple of exceptions to retry on

    Returns:
        Decorated function with retry logic
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            logger = logging.getLogger(func.__module__)
            current_delay = delay
            last_exception = None

            for attempt in range(max_attempts):
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    last_exception = e
                    if attempt < max_attempts - 1:
                        logger.warning(
                            f"Attempt {attempt + 1} failed for {func.__qualname__}: {e}. "
                            f"Retrying in {current_delay}s..."
                        )
                        import time
                        time.sleep(current_delay)
                        current_delay *= backoff
                    else:
                        logger.error(f"All {max_attempts} attempts failed for {func.__qualname__}")

            raise last_exception

        return wrapper
    return decorator


class ErrorHandler:
    """
    Context manager for comprehensive error handling.

    Usage:
        with ErrorHandler("database operation", logger=my_logger):
            # risky code here
            pass
    """

    def __init__(self, operation: str, logger: Optional[logging.Logger] = None, reraise: bool = True):
        """
        Initialize error handler.

        Args:
            operation: Description of the operation being performed
            logger: Logger instance (defaults to module logger)
            reraise: Whether to re-raise exceptions after handling
        """
        self.operation = operation
        self.logger = logger or logging.getLogger(__name__)
        self.reraise = reraise

    def __enter__(self):
        self.logger.debug(f"Starting operation: {self.operation}")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is not None:
            # Log the error with context
            if isinstance(exc_val, SupremeSystemError):
                self.logger.log(
                    getattr(logging, exc_val.log_level),
                    f"Operation failed: {self.operation} - {exc_val}"
                )
            else:
                self.logger.exception(f"Operation failed: {self.operation} - Unexpected error: {exc_val}")

            return not self.reraise  # Suppress exception if not reraising

        self.logger.debug(f"Operation completed successfully: {self.operation}")
        return False


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
    'handle_exception',
    # Advanced error handling
    'handle_errors',
    'retry_on_failure',
    'ErrorHandler'
]

