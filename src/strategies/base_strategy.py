#!/usr/bin/env python3
"""
Supreme System V5 - Base Trading Strategy

Abstract base class for all trading strategies.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
import pandas as pd
import logging


class BaseStrategy(ABC):
    """
    Abstract base class for trading strategies.

    All trading strategies must inherit from this class and implement
    the required methods.
    """

    def __init__(self, name: str = "BaseStrategy"):
        """
        Initialize the strategy.

        Args:
            name: Strategy name for identification
        """
        self.name = name
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        self.parameters: Dict[str, Any] = {}

    @abstractmethod
    def generate_signal(self, data: pd.DataFrame) -> int:
        """
        Generate trading signal based on market data.

        Args:
            data: DataFrame with OHLCV data

        Returns:
            1 for buy signal, -1 for sell signal, 0 for hold
        """
        pass

    def set_parameters(self, **kwargs):
        """Set strategy parameters."""
        self.parameters.update(kwargs)
        self.logger.info(f"Updated parameters: {kwargs}")

    def get_parameters(self) -> Dict[str, Any]:
        """Get current strategy parameters."""
        return self.parameters.copy()

    def validate_data(self, data: pd.DataFrame) -> bool:
        """
        Validate that data has required columns.

        Args:
            data: DataFrame to validate

        Returns:
            True if data is valid, False otherwise
        """
        required_columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume']

        for col in required_columns:
            if col not in data.columns:
                self.logger.error(f"Missing required column: {col}")
                return False

        if len(data) == 0:
            self.logger.error("Data is empty")
            return False

        return True

    def __str__(self) -> str:
        return f"{self.name}(parameters={self.parameters})"

    def __repr__(self) -> str:
        return self.__str__()
