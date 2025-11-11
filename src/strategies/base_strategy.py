#!/usr/bin/env python3
"""
Supreme System V5 - Base Trading Strategy

Abstract base class for all trading strategies.
"""

import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Protocol, Union, List, Tuple

import pandas as pd

# Import typed structures
try:
    from ..types.trading_types import (
        OHLCVData, SignalData, StrategyConfig, IndicatorResult,
        validate_ohlcv_data, SignalStrength, OrderAction
    )
except ImportError:
    # Fallback for missing types
    OHLCVData = Dict[str, Any]
    SignalData = Dict[str, Any]
    StrategyConfig = Dict[str, Any]
    IndicatorResult = Dict[str, Any]
    SignalStrength = str
    OrderAction = str

    def validate_ohlcv_data(data: pd.DataFrame) -> bool:
        """Fallback validation."""
        required = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
        return all(col in data.columns for col in required)

# Import with fallback for different execution contexts
try:
    from ..utils.data_utils import optimize_dataframe_memory, validate_and_clean_data
except ImportError:
    try:
        from utils.data_utils import optimize_dataframe_memory, validate_and_clean_data
    except ImportError:
        # Fallback implementation
        def validate_and_clean_data(df, required_columns=None):
            return df

        def optimize_dataframe_memory(df, copy=True):
            return df.copy() if copy else df


class SignalType:
    """Enumeration of possible trading signals."""
    HOLD = 0
    BUY = 1
    SELL = -1


class MarketDataProtocol(Protocol):
    """Protocol for market data structure."""
    @property
    def columns(self) -> pd.Index: ...

    @property
    def index(self) -> pd.Index: ...

    def __getitem__(self, key: str) -> pd.Series: ...


class BaseStrategy(ABC):
    """
    Abstract base class for trading strategies.

    All trading strategies must inherit from this class and implement
    the required methods.
    """

    def __init__(self, name: str = "BaseStrategy") -> None:
        """
        Initialize the strategy.

        Args:
            name: Strategy name for identification
        """
        self.name: str = name
        self.logger: logging.Logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        self.parameters: Dict[str, Any] = {}
        self.is_initialized: bool = False

    @abstractmethod
    def generate_signal(self, data: pd.DataFrame, portfolio_value: Optional[float] = None) -> SignalData:
        """
        Generate trading signal based on market data.

        Args:
            data: DataFrame with OHLCV data (timestamp, open, high, low, close, volume)
            portfolio_value: Current portfolio value (optional)

        Returns:
            SignalData dict with action, symbol, strength, confidence, etc.
        """
        pass

    def validate_data(self, data: pd.DataFrame) -> bool:
        """
        Validate that the input data has required columns and format.

        Args:
            data: DataFrame to validate

        Returns:
            True if data is valid, False otherwise
        """
        required_columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume']

        # Check for required columns
        missing_columns = [col for col in required_columns if col not in data.columns]
        if missing_columns:
            self.logger.error(f"Missing required columns: {missing_columns}")
            return False

        # Use optimized validation and memory management
        try:
            # Use optimized validation and cleaning
            validated_data = validate_and_clean_data(data)

            # Update original data with validated and optimized version
            # Ensure dtype compatibility to avoid FutureWarning and SettingWithCopyWarning
            for col in data.columns:
                if col in validated_data.columns:
                    # Convert to compatible dtype if needed
                    if data[col].dtype != validated_data[col].dtype:
                        data.loc[:, col] = validated_data[col].astype(data[col].dtype)
                    else:
                        data.loc[:, col] = validated_data[col]

            return True

        except Exception as e:
            self.logger.error(f"Data validation error: {e}")
            return False

    def set_parameters(self, **kwargs: Any) -> None:
        """Set strategy parameters."""
        self.parameters.update(kwargs)

        # Also update instance attributes if they exist
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)

        self.logger.info(f"Updated parameters: {kwargs}")

    def get_parameters(self) -> Dict[str, Any]:
        """Get current strategy parameters."""
        return self.parameters.copy()

    def reset(self) -> None:
        """Reset strategy state."""
        self.parameters.clear()
        self.is_initialized = False
        self.logger.info("Strategy reset")

    def get_strategy_info(self) -> Dict[str, Any]:
        """Get strategy information."""
        return {
            'name': self.name,
            'type': self.__class__.__name__,
            'parameters': self.get_parameters(),
            'is_initialized': self.is_initialized
        }


    def __str__(self) -> str:
        return f"{self.name}(parameters={self.parameters})"

    def __repr__(self) -> str:
        return self.__str__()
