#!/usr/bin/env python3
"""
Base Strategy Interface 2.0 - Enterprise Strategy Framework

Advanced strategy framework with risk integration, portfolio awareness,
and production-grade signal generation with validation and lifecycle management.
"""

import logging
from abc import ABC, abstractmethod
from collections import deque
from dataclasses import dataclass
from typing import Optional, Dict, Any, List
from datetime import datetime

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class Signal:
    """
    Standardized signal format for all strategies.

    Provides consistent interface for signal communication between
    strategies, risk managers, and execution engines.
    """
    symbol: str
    side: str  # 'buy' or 'sell'
    price: float
    strength: float = 1.0  # 0.0 to 1.0 (signal confidence/strength)
    metadata: Optional[Dict[str, Any]] = None
    timestamp: Optional[datetime] = None

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()

    def to_dict(self) -> Dict[str, Any]:
        """Convert signal to dictionary format."""
        return {
            'symbol': self.symbol,
            'side': self.side,
            'price': self.price,
            'strength': self.strength,
            'metadata': self.metadata or {},
            'timestamp': self.timestamp.isoformat() if self.timestamp else None
        }


class BaseStrategy(ABC):
    """
    Abstract base class for all trading strategies v2.0.

    Enterprise-grade strategy framework with:
    - Portfolio awareness and risk integration
    - Standardized signal format
    - Comprehensive validation and error handling
    - Performance tracking and analytics
    """

    def __init__(self, name: str, config: Dict[str, Any]):
        """
        Initialize the strategy with enterprise configuration.

        Args:
            name: Unique strategy identifier
            config: Comprehensive configuration dictionary
        """
        self.name = name
        self.config = config
        self.logger = logging.getLogger(f"Strategy.{name}")

        # Core state
        self.is_active = True
        self.creation_time = datetime.now()
        self.last_signal_time = None

        # Position tracking (for strategy awareness)
        self.current_position = None  # Current position info
        self.portfolio_value = config.get('initial_capital', 10000.0)

        # Risk integration
        self.risk_manager = None  # Will be injected by framework
        self.max_position_size = config.get('max_position_size', 0.1)
        self.max_daily_loss = config.get('max_daily_loss', 0.05)

        # OPTIMIZATION: Centralized price buffer with maxlen (memory-safe)
        buffer_size = config.get('buffer_size', 100)
        self.prices = deque(maxlen=buffer_size)

        # Performance tracking
        self.total_signals = 0
        self.executed_signals = 0
        self.successful_signals = 0
        self.total_pnl = 0.0

        # Strategy-specific state (subclasses can override)
        self._initialize_state()

        logger.info(f"Strategy {name} v2.0 initialized with enterprise config")

    def _initialize_state(self):
        """Initialize strategy-specific state. Override in subclasses."""
        pass

    def update_price(self, price: float):
        """
        Helper method to safely update the price buffer.

        Args:
            price: Current price to add to buffer
        """
        self.prices.append(price)

    @abstractmethod
    def generate_signal(self, market_data: Dict[str, Any]) -> Optional[Signal]:
        """
        Core strategy logic: Analyze market data and generate signals.

        Args:
            market_data: Current market data with price, volume, etc.

        Returns:
            Signal object or None if no action needed
        """
        pass

    def validate_signal(self, signal: Signal) -> bool:
        """
        Pre-validate signal before sending to risk manager.

        Args:
            signal: Signal to validate

        Returns:
            True if signal passes validation
        """
        if not signal:
            return False

        # Basic validation
        if signal.side not in ['buy', 'sell']:
            self.logger.error(f"Invalid signal side: {signal.side}")
            return False

        if signal.price <= 0:
            self.logger.error(f"Invalid signal price: {signal.price}")
            return False

        if not (0.0 <= signal.strength <= 1.0):
            self.logger.warning(f"Signal strength out of range: {signal.strength}")
            signal.strength = max(0.0, min(1.0, signal.strength))  # Clamp to valid range

        # Risk-aware validation
        if self.risk_manager:
            # Check if we can afford this position
            position_value = signal.price * self._estimate_position_size(signal)
            if position_value > self.portfolio_value * self.max_position_size:
                self.logger.warning(f"Signal exceeds max position size: {position_value}")
                return False

        return True

    def _estimate_position_size(self, signal: Signal) -> float:
        """
        Estimate position size for signal (simplified).

        Returns:
            Estimated number of shares/contracts
        """
        # Simplified: assume we risk 1% of portfolio per trade
        risk_amount = self.portfolio_value * 0.01
        stop_distance = signal.price * 0.02  # Assume 2% stop loss

        if stop_distance > 0:
            return risk_amount / stop_distance
        return 0.0

    def on_order_filled(self, trade_info: Dict[str, Any]):
        """
        Callback when order is executed. Update internal state.

        Args:
            trade_info: Details about the executed trade
        """
        self.executed_signals += 1
        self.last_signal_time = datetime.now()

        # Update portfolio value (simplified)
        if 'pnl' in trade_info:
            self.total_pnl += trade_info['pnl']
            self.portfolio_value += trade_info['pnl']

        # Update position tracking
        if trade_info.get('side') == 'buy':
            self.current_position = {
                'symbol': trade_info.get('symbol'),
                'quantity': trade_info.get('quantity', 0),
                'avg_price': trade_info.get('price', 0),
                'timestamp': datetime.now()
            }
        elif trade_info.get('side') == 'sell':
            self.current_position = None  # Closed position

        self.logger.info(f"Order filled: {trade_info}")

    def update_portfolio_state(self, portfolio_info: Dict[str, Any]):
        """
        Update strategy with current portfolio state.

        Args:
            portfolio_info: Current portfolio metrics and positions
        """
        self.portfolio_value = portfolio_info.get('total_value', self.portfolio_value)
        self.current_position = portfolio_info.get('current_position')

    def get_status(self) -> Dict[str, Any]:
        """
        Get comprehensive strategy status and performance metrics.

        Returns:
            Detailed status dictionary
        """
        win_rate = self.successful_signals / self.executed_signals if self.executed_signals > 0 else 0

        return {
            'name': self.name,
            'version': '2.0',
            'is_active': self.is_active,
            'config': self.config,
            'creation_time': self.creation_time.isoformat(),
            'last_signal_time': self.last_signal_time.isoformat() if self.last_signal_time else None,

            # Performance metrics
            'total_signals': self.total_signals,
            'executed_signals': self.executed_signals,
            'successful_signals': self.successful_signals,
            'win_rate': win_rate,
            'total_pnl': self.total_pnl,
            'portfolio_value': self.portfolio_value,

            # Position info
            'current_position': self.current_position,

            # Risk settings
            'max_position_size': self.max_position_size,
            'max_daily_loss': self.max_daily_loss
        }

    def reset(self):
        """Reset strategy state for fresh start."""
        self.total_signals = 0
        self.executed_signals = 0
        self.successful_signals = 0
        self.total_pnl = 0.0
        self.last_signal_time = None
        self.current_position = None

        # CRITICAL FIX: Clear centralized price buffer
        self.prices.clear()

        self._initialize_state()
        self.logger.info(f"Strategy {self.name} reset to initial state")
