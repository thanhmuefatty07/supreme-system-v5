"""
Type Definitions for Supreme System V5 Trading Components

Comprehensive type hints and TypedDict definitions for type safety.
"""

from typing import Dict, List, Optional, Union, Literal, Any, Protocol, TypeVar, TypedDict, Tuple
from datetime import datetime
from dataclasses import dataclass
from enum import Enum
import pandas as pd
import numpy as np


# Basic data types
Symbol = str
Price = float
Volume = Union[int, float]
Timestamp = Union[datetime, str, float]

# Order types
OrderAction = Literal['BUY', 'SELL', 'HOLD']
OrderStatus = Literal['PENDING', 'FILLED', 'REJECTED', 'CANCELLED', 'ERROR']
OrderType = Literal['MARKET', 'LIMIT', 'STOP_LOSS', 'TAKE_PROFIT']

# Strategy signals
SignalStrength = Literal['WEAK', 'MODERATE', 'STRONG', 'VERY_STRONG']

# Timeframe definitions
Timeframe = Literal['1m', '5m', '15m', '30m', '1h', '4h', '1d', '1w']

# Generic type variables
T = TypeVar('T')
DataFrameType = TypeVar('DataFrameType', bound=pd.DataFrame)
SeriesType = TypeVar('SeriesType', bound=pd.Series)


class OHLCVData(TypedDict, total=False):
    """OHLCV market data structure."""
    timestamp: Timestamp
    open: Price
    high: Price
    low: Price
    close: Price
    volume: Volume
    symbol: Symbol


class OrderData(TypedDict, total=False):
    """Order execution data structure."""
    order_id: str
    symbol: Symbol
    action: OrderAction
    quantity: Union[int, float]
    price: Price
    order_type: OrderType
    stop_loss: Optional[Price]
    take_profit: Optional[Price]
    timestamp: Timestamp
    status: OrderStatus


class TradeResult(TypedDict, total=False):
    """Trade execution result structure."""
    success: bool
    order_id: Optional[str]
    executed_quantity: Optional[Union[int, float]]
    executed_price: Optional[Price]
    fee: Optional[float]
    timestamp: Timestamp
    error_message: Optional[str]


class SignalData(TypedDict, total=False):
    """Trading signal data structure."""
    action: OrderAction
    symbol: Symbol
    strength: SignalStrength
    confidence: float
    indicators: Dict[str, Any]
    timestamp: Timestamp
    metadata: Optional[Dict[str, Any]]


class PortfolioData(TypedDict, total=False):
    """Portfolio status data structure."""
    total_value: float
    cash_balance: float
    positions: Dict[Symbol, Dict[str, Any]]
    unrealized_pnl: float
    realized_pnl: float
    timestamp: Timestamp


class RiskMetrics(TypedDict, total=False):
    """Risk management metrics structure."""
    max_drawdown: float
    sharpe_ratio: float
    sortino_ratio: float
    calmar_ratio: float
    var_95: float
    cvar_95: float
    beta: float
    alpha: float


class PerformanceMetrics(TypedDict, total=False):
    """Performance metrics structure."""
    total_return: float
    annualized_return: float
    volatility: float
    max_drawdown: float
    sharpe_ratio: float
    sortino_ratio: float
    win_rate: float
    profit_factor: float
    total_trades: int
    avg_trade_duration: Optional[float]


class StrategyConfig(TypedDict, total=False):
    """Strategy configuration structure."""
    name: str
    parameters: Dict[str, Any]
    enabled: bool
    risk_limits: Dict[str, float]
    timeframe: Timeframe
    symbols: List[Symbol]


class BacktestConfig(TypedDict, total=False):
    """Backtesting configuration structure."""
    start_date: Union[str, datetime]
    end_date: Union[str, datetime]
    initial_capital: float
    commission: float
    slippage: float
    symbols: List[Symbol]
    strategies: List[StrategyConfig]


class BacktestResult(TypedDict, total=False):
    """Backtesting result structure."""
    strategy_name: str
    total_return: float
    sharpe_ratio: float
    max_drawdown: float
    win_rate: float
    total_trades: int
    start_date: datetime
    end_date: datetime
    execution_time: float
    trades: List[Dict[str, Any]]
    equity_curve: List[Tuple[datetime, float]]


class ValidationResult(TypedDict, total=False):
    """Data validation result structure."""
    is_valid: bool
    errors: List[str]
    warnings: List[str]
    cleaned_data: Optional[pd.DataFrame]


class IndicatorResult(TypedDict, total=False):
    """Technical indicator calculation result."""
    name: str
    values: Union[pd.Series, np.ndarray]
    metadata: Optional[Dict[str, Any]]


class DataSourceConfig(TypedDict, total=False):
    """Data source configuration structure."""
    name: str
    type: Literal['binance', 'yahoo', 'alpha_vantage', 'local']
    api_key: Optional[str]
    api_secret: Optional[str]
    base_url: Optional[str]
    timeout: Optional[int]
    rate_limit: Optional[int]


@dataclass(slots=True)
class MarketData:
    """Typed market data container."""
    symbol: Symbol
    timeframe: Timeframe
    data: pd.DataFrame
    last_updated: datetime
    source: str

    def __post_init__(self):
        """Validate data structure after initialization."""
        required_columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
        if not all(col in self.data.columns for col in required_columns):
            raise ValueError(f"Market data missing required columns: {required_columns}")


@dataclass(slots=True)
class TradingSignal:
    """Typed trading signal container."""
    symbol: Symbol
    action: OrderAction
    strength: SignalStrength
    confidence: float
    price: Optional[Price] = None
    quantity: Optional[Union[int, float]] = None
    timestamp: Optional[datetime] = None
    metadata: Optional[Dict[str, Any]] = None

    def __post_init__(self):
        """Set timestamp if not provided."""
        if self.timestamp is None:
            self.timestamp = datetime.now()


@dataclass(slots=True)
class Position:
    """Typed position container."""
    symbol: Symbol
    quantity: Union[int, float]
    entry_price: Price
    current_price: Price
    unrealized_pnl: float
    stop_loss: Optional[Price] = None
    take_profit: Optional[Price] = None
    entry_time: Optional[datetime] = None

    def __post_init__(self):
        """Set entry time if not provided."""
        if self.entry_time is None:
            self.entry_time = datetime.now()


# Protocol definitions for type safety
class TradingStrategyProtocol(Protocol):
    """Protocol for trading strategy implementations."""

    def generate_signal(self, data: pd.DataFrame, portfolio_value: float) -> SignalData:
        """Generate trading signal from market data."""
        ...

    def validate_data(self, data: pd.DataFrame) -> bool:
        """Validate input data format."""
        ...

    def get_parameters(self) -> Dict[str, Any]:
        """Get strategy parameters."""
        ...


class DataProviderProtocol(Protocol):
    """Protocol for data provider implementations."""

    def fetch_historical_data(self, symbol: Symbol, start_date: datetime,
                            end_date: datetime, timeframe: Timeframe) -> pd.DataFrame:
        """Fetch historical market data."""
        ...

    def get_real_time_data(self, symbol: Symbol) -> OHLCVData:
        """Get real-time market data."""
        ...


class RiskManagerProtocol(Protocol):
    """Protocol for risk manager implementations."""

    def assess_trade_risk(self, symbol: Symbol, quantity: Union[int, float],
                         entry_price: Price, current_data: pd.DataFrame) -> Dict[str, Any]:
        """Assess risk for potential trade."""
        ...

    def calculate_position_size(self, capital: float, risk_per_trade: float,
                              stop_loss_pct: float) -> Union[int, float]:
        """Calculate appropriate position size."""
        ...


class OrderExecutorProtocol(Protocol):
    """Protocol for order executor implementations."""

    def execute_order(self, order: OrderData) -> TradeResult:
        """Execute trading order."""
        ...

    def get_order_status(self, order_id: str) -> OrderStatus:
        """Get status of specific order."""
        ...


# Type aliases for complex types
IndicatorValues = Dict[str, Union[pd.Series, np.ndarray]]
StrategyParameters = Dict[str, Union[int, float, str, bool]]
ConfigurationDict = Dict[str, Any]
MetricsDict = Dict[str, Union[int, float, str]]

# Generic result types
ResultType = Union[Dict[str, Any], pd.DataFrame, pd.Series, List[Dict[str, Any]], None]

# Error types
ErrorContext = Dict[str, Any]
ErrorCode = str


# Utility functions for type validation
def validate_ohlcv_data(data: pd.DataFrame) -> bool:
    """Validate OHLCV data structure."""
    required_columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
    return all(col in data.columns for col in required_columns)


def validate_signal_data(signal: SignalData) -> bool:
    """Validate signal data structure."""
    required_keys = ['action', 'symbol', 'strength', 'confidence']
    return all(key in signal for key in required_keys)


def validate_order_data(order: OrderData) -> bool:
    """Validate order data structure."""
    required_keys = ['symbol', 'action', 'quantity', 'price']
    return all(key in order for key in required_keys)


# Type-safe factory functions
def create_market_data(symbol: Symbol, timeframe: Timeframe,
                      data: pd.DataFrame, source: str) -> MarketData:
    """Create validated MarketData instance."""
    if not validate_ohlcv_data(data):
        raise ValueError(f"Invalid OHLCV data for {symbol}")

    return MarketData(
        symbol=symbol,
        timeframe=timeframe,
        data=data,
        last_updated=datetime.now(),
        source=source
    )


def create_trading_signal(symbol: Symbol, action: OrderAction,
                         strength: SignalStrength, confidence: float,
                         **kwargs) -> TradingSignal:
    """Create validated TradingSignal instance."""
    if not 0.0 <= confidence <= 1.0:
        raise ValueError(f"Confidence must be between 0.0 and 1.0, got {confidence}")

    return TradingSignal(
        symbol=symbol,
        action=action,
        strength=strength,
        confidence=confidence,
        **kwargs
    )
