"""
Base exchange connector interface
Standardized interface for all exchange implementations
"""

import asyncio
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional

from loguru import logger


@dataclass
class ExchangeConfig:
    """
    Exchange configuration parameters
    """

    api_key: str = ""
    secret_key: str = ""
    passphrase: str = ""  # For OKX
    sandbox: bool = True  # Default to sandbox for safety
    rate_limit_ms: int = 100  # Rate limiting
    timeout_seconds: int = 30
    max_retries: int = 3


@dataclass
class OrderResult:
    """
    Result of order execution
    """

    success: bool
    order_id: str
    symbol: str
    side: str  # "buy" or "sell"
    amount: float
    price: float
    fee: float = 0.0
    timestamp: float = 0.0
    error_message: str = ""


class BaseExchange(ABC):
    """
    Abstract base class for exchange connectors
    Ensures consistent interface across all exchanges
    """

    def __init__(self, config: ExchangeConfig):
        self.config = config
        self.connected = False
        self.market_data_callback: Optional[Callable] = None
        self.order_update_callback: Optional[Callable] = None

        # Validate configuration
        if not config.api_key or not config.secret_key:
            logger.warning("⚠️ Exchange API credentials not provided")

    @abstractmethod
    async def connect(self) -> bool:
        """
        Connect to exchange
        Returns True if successful
        """
        pass

    @abstractmethod
    async def disconnect(self):
        """
        Disconnect from exchange
        """
        pass

    @abstractmethod
    async def subscribe_market_data(self, symbols: List[str]):
        """
        Subscribe to real-time market data
        """
        pass

    @abstractmethod
    async def place_market_order(
        self, symbol: str, side: str, amount: float
    ) -> OrderResult:
        """
        Place market order
        """
        pass

    @abstractmethod
    async def place_limit_order(
        self, symbol: str, side: str, amount: float, price: float
    ) -> OrderResult:
        """
        Place limit order
        """
        pass

    @abstractmethod
    async def cancel_order(self, order_id: str, symbol: str) -> bool:
        """
        Cancel existing order
        """
        pass

    @abstractmethod
    async def get_balance(self) -> Dict[str, float]:
        """
        Get account balance
        """
        pass

    @abstractmethod
    async def get_positions(self) -> List[Dict[str, Any]]:
        """
        Get current positions
        """
        pass

    def set_market_data_callback(self, callback: Callable):
        """
        Set callback for market data updates
        """
        self.market_data_callback = callback

    def set_order_update_callback(self, callback: Callable):
        """
        Set callback for order updates
        """
        self.order_update_callback = callback

    async def health_check(self) -> bool:
        """
        Check if exchange connection is healthy
        """
        return self.connected
