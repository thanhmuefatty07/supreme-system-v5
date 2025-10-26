"""
📋 Supreme System V5 - API Data Models
Pydantic models for request/response validation and documentation

Features:
- Comprehensive request/response models
- Automatic data validation
- OpenAPI schema generation
- Type safety and documentation
- Standardized error responses
- Portfolio and trading data structures
"""

from datetime import datetime
from decimal import Decimal
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, root_validator, validator


class ResponseStatus(str, Enum):
    """Standard response status values"""

    SUCCESS = "success"
    ERROR = "error"
    WARNING = "warning"
    PENDING = "pending"


class TradingState(str, Enum):
    """Trading engine states"""

    IDLE = "idle"
    STARTING = "starting"
    RUNNING = "running"
    STOPPING = "stopping"
    STOPPED = "stopped"
    ERROR = "error"


class OrderSide(str, Enum):
    """Order side enumeration"""

    BUY = "buy"
    SELL = "sell"


class OrderType(str, Enum):
    """Order type enumeration"""

    MARKET = "market"
    LIMIT = "limit"
    STOP_LOSS = "stop_loss"
    TAKE_PROFIT = "take_profit"


class OrderStatus(str, Enum):
    """Order status enumeration"""

    PENDING = "pending"
    FILLED = "filled"
    PARTIALLY_FILLED = "partially_filled"
    CANCELLED = "cancelled"
    REJECTED = "rejected"


class ComponentStatus(str, Enum):
    """System component status"""

    ACTIVE = "active"
    INACTIVE = "inactive"
    ERROR = "error"
    INITIALIZING = "initializing"
    STOPPING = "stopping"


# Base response models
class BaseResponse(BaseModel):
    """Base response model"""

    status: ResponseStatus
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    message: Optional[str] = None


class ErrorResponse(BaseResponse):
    """Error response model"""

    status: ResponseStatus = ResponseStatus.ERROR
    error_code: Optional[str] = None
    details: Optional[Dict[str, Any]] = None


class SuccessResponse(BaseResponse):
    """Success response model"""

    status: ResponseStatus = ResponseStatus.SUCCESS
    data: Optional[Dict[str, Any]] = None


# Authentication models
class LoginRequest(BaseModel):
    """User login request"""

    username: str = Field(..., min_length=3, max_length=50)
    password: str = Field(..., min_length=8, max_length=128)


class LoginResponse(BaseModel):
    """User login response"""

    access_token: str
    refresh_token: str
    token_type: str = "bearer"
    expires_in: int
    user: Dict[str, Any]


class RefreshTokenRequest(BaseModel):
    """Token refresh request"""

    refresh_token: str


class TokenResponse(BaseModel):
    """Token response"""

    access_token: str
    token_type: str = "bearer"
    expires_in: int


# System status models
class PerformanceMetrics(BaseModel):
    """Performance metrics data"""

    timestamp: datetime
    latency_us: float = Field(..., description="Latency in microseconds")
    throughput_tps: float = Field(..., description="Throughput in transactions per second")
    accuracy_percent: float = Field(
        ..., ge=0, le=100, description="Prediction accuracy percentage"
    )
    memory_mb: float = Field(..., ge=0, description="Memory usage in MB")
    cpu_percent: float = Field(..., ge=0, le=100, description="CPU usage percentage")
    gpu_utilization: float = Field(..., ge=0, le=100, description="GPU utilization percentage")


class ComponentInfo(BaseModel):
    """System component information"""

    status: ComponentStatus
    initialized: bool
    last_activity: datetime
    metadata: Optional[Dict[str, Any]] = None


class SystemStatus(BaseModel):
    """System status response"""

    status: str
    version: str
    uptime_seconds: float
    timestamp: datetime
    components: Dict[str, ComponentInfo]
    performance: PerformanceMetrics


# Trading models
class TradingConfig(BaseModel):
    """Trading configuration"""

    exchange: str
    testnet: bool = True
    trading_pairs: List[str]
    max_position_size: float = Field(..., gt=0)
    max_daily_loss: float = Field(..., gt=0)
    stop_loss_pct: float = Field(..., gt=0, le=100)
    take_profit_pct: float = Field(..., gt=0, le=100)
    max_open_positions: int = Field(..., gt=0)

    # AI component flags
    use_neuromorphic: bool = True
    use_ultra_low_latency: bool = True
    use_foundation_models: bool = True
    use_mamba_ssm: bool = True


class MarketData(BaseModel):
    """Market data structure"""

    symbol: str
    price: Decimal = Field(..., decimal_places=8)
    bid: Decimal = Field(..., decimal_places=8)
    ask: Decimal = Field(..., decimal_places=8)
    volume: Decimal = Field(..., decimal_places=4)
    timestamp: int
    exchange: str


class Position(BaseModel):
    """Trading position"""

    symbol: str
    quantity: Decimal = Field(..., decimal_places=8)
    average_price: Decimal = Field(..., decimal_places=8)
    current_price: Optional[Decimal] = Field(None, decimal_places=8)
    unrealized_pnl: Optional[Decimal] = Field(None, decimal_places=2)
    side: OrderSide
    opened_at: datetime


class Order(BaseModel):
    """Trading order"""

    order_id: str
    symbol: str
    side: OrderSide
    type: OrderType
    quantity: Decimal = Field(..., decimal_places=8)
    price: Optional[Decimal] = Field(None, decimal_places=8)
    status: OrderStatus
    filled_quantity: Decimal = Field(default=Decimal("0"), decimal_places=8)
    timestamp: datetime
    execution_time_ms: Optional[float] = None
    exchange: str


class PortfolioBalance(BaseModel):
    """Portfolio balance for a currency"""

    currency: str
    available: Decimal = Field(..., decimal_places=8)
    locked: Decimal = Field(default=Decimal("0"), decimal_places=8)
    total: Decimal = Field(..., decimal_places=8)


class PnLSummary(BaseModel):
    """Profit and Loss summary"""

    realized: Decimal = Field(..., decimal_places=2)
    unrealized: Decimal = Field(..., decimal_places=2)
    total: Decimal = Field(..., decimal_places=2)
    daily: Decimal = Field(..., decimal_places=2)

    @root_validator
    def calculate_total(cls, values):
        """Calculate total PnL"""
        realized = values.get("realized", Decimal("0"))
        unrealized = values.get("unrealized", Decimal("0"))
        values["total"] = realized + unrealized
        return values


class PortfolioStatus(BaseModel):
    """Portfolio status response"""

    total_value: Decimal = Field(
        ..., decimal_places=2, description="Total portfolio value in USD"
    )
    available_balance: Decimal = Field(
        ..., decimal_places=2, description="Available cash balance"
    )
    positions: List[Position] = Field(default=[])
    balances: List[PortfolioBalance] = Field(default=[])
    pnl: PnLSummary
    last_updated: datetime
    open_positions: int = Field(default=0, ge=0)
    max_positions: int = Field(default=5, ge=1)


class TradingResponse(BaseModel):
    """Trading operation response"""

    success: bool
    message: str
    timestamp: datetime
    data: Optional[Dict[str, Any]] = None


class OrderRequest(BaseModel):
    """Order placement request"""

    symbol: str = Field(..., description="Trading pair symbol")
    side: OrderSide
    type: OrderType
    quantity: Decimal = Field(..., gt=0, decimal_places=8)
    price: Optional[Decimal] = Field(None, decimal_places=8)
    stop_price: Optional[Decimal] = Field(None, decimal_places=8)
    time_in_force: Optional[str] = Field("GTC", description="Time in force")

    @validator("price")
    def price_required_for_limit_orders(cls, v, values):
        """Validate price for limit orders"""
        order_type = values.get("type")
        if order_type == OrderType.LIMIT and v is None:
            msg = "Price is required for limit orders"
            raise ValueError(msg)
        return v


class SignalData(BaseModel):
    """AI signal data"""

    symbol: str
    signal_strength: float = Field(..., ge=-1.0, le=1.0)
    confidence: float = Field(..., ge=0.0, le=1.0)
    components: Dict[str, float] = Field(default={})
    metadata: Dict[str, Any] = Field(default={})
    timestamp: datetime = Field(default_factory=datetime.utcnow)


# Performance and analytics models
class AIComponentMetrics(BaseModel):
    """AI component performance metrics"""

    component_name: str
    processing_time_us: float
    accuracy: Optional[float] = Field(None, ge=0, le=100)
    patterns_detected: int = Field(default=0, ge=0)
    predictions_made: int = Field(default=0, ge=0)
    success_rate: Optional[float] = Field(None, ge=0, le=100)


class TradingMetrics(BaseModel):
    """Trading performance metrics"""

    total_trades: int = Field(default=0, ge=0)
    successful_trades: int = Field(default=0, ge=0)
    win_rate: float = Field(default=0.0, ge=0, le=100)
    total_pnl: Decimal = Field(default=Decimal("0"), decimal_places=2)
    average_trade_pnl: Decimal = Field(default=Decimal("0"), decimal_places=2)
    max_drawdown: Decimal = Field(default=Decimal("0"), decimal_places=2)
    sharpe_ratio: Optional[float] = None

    @validator("win_rate", pre=True)
    def calculate_win_rate(cls, v, values):
        """Calculate win rate from successful and total trades"""
        total = values.get("total_trades", 0)
        successful = values.get("successful_trades", 0)
        return (successful / total * 100) if total > 0 else 0.0


class SessionMetrics(BaseModel):
    """Trading session metrics"""

    session_id: str
    start_time: datetime
    end_time: Optional[datetime] = None
    duration_minutes: float = Field(default=0.0, ge=0)
    status: TradingState
    initial_balance: Decimal = Field(..., decimal_places=2)
    final_balance: Optional[Decimal] = Field(None, decimal_places=2)
    total_return: Optional[Decimal] = Field(None, decimal_places=4)


# WebSocket models
class WebSocketMessage(BaseModel):
    """WebSocket message structure"""

    type: str
    timestamp: datetime
    data: Dict[str, Any]
    client_id: Optional[str] = None
    sequence_id: Optional[int] = None


class SubscriptionRequest(BaseModel):
    """WebSocket subscription request"""

    subscriptions: List[str] = Field(default=[])
    client_id: Optional[str] = None


class HeartbeatMessage(BaseModel):
    """WebSocket heartbeat message"""

    type: str = "heartbeat"
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    status: str = "alive"


# Configuration models
class APIConfig(BaseModel):
    """API configuration"""

    host: str = "0.0.0.0"
    port: int = Field(default=8000, ge=1024, le=65535)
    debug: bool = False
    workers: int = Field(default=1, ge=1, le=16)
    log_level: str = Field(default="info", regex=r"^(debug|info|warning|error|critical)$")
    cors_origins: List[str] = Field(default=["*"])
    max_connections: int = Field(default=1000, ge=1)

    # Rate limiting
    rate_limit_enabled: bool = True
    rate_limit_per_minute: int = Field(default=60, ge=1)

    # Security
    jwt_secret_key: str
    jwt_expiry_minutes: int = Field(default=60, ge=5)


class MonitoringConfig(BaseModel):
    """Monitoring system configuration"""

    prometheus_enabled: bool = True
    prometheus_port: int = Field(default=9090, ge=1024, le=65535)
    metrics_retention_days: int = Field(default=30, ge=1)
    alert_thresholds: Dict[str, float] = Field(default={})

    # Health check settings
    health_check_interval_seconds: int = Field(default=30, ge=1)
    component_timeout_seconds: int = Field(default=10, ge=1)


# Error handling models
class ValidationError(BaseModel):
    """Validation error details"""

    field: str
    message: str
    rejected_value: Any


class APIError(BaseModel):
    """API error response"""

    error: str
    message: str
    status_code: int
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    request_id: Optional[str] = None
    validation_errors: Optional[List[ValidationError]] = None


# Batch operation models
class BatchOrderRequest(BaseModel):
    """Batch order request"""

    orders: List[OrderRequest] = Field(..., max_items=10)

    @validator("orders")
    def validate_orders(cls, v):
        """Validate order list"""
        if len(v) == 0:
            msg = "At least one order is required"
            raise ValueError(msg)
        return v


class BatchOrderResponse(BaseModel):
    """Batch order response"""

    submitted_orders: List[Order] = Field(default=[])
    failed_orders: List[Dict[str, Any]] = Field(default=[])
    success_count: int = Field(default=0, ge=0)
    failure_count: int = Field(default=0, ge=0)

    @root_validator
    def calculate_counts(cls, values):
        """Calculate success and failure counts"""
        values["success_count"] = len(values.get("submitted_orders", []))
        values["failure_count"] = len(values.get("failed_orders", []))
        return values


# Advanced analytics models
class BacktestRequest(BaseModel):
    """Backtesting request"""

    strategy_id: str
    start_date: datetime
    end_date: datetime
    initial_balance: Decimal = Field(..., gt=0, decimal_places=2)
    symbols: List[str] = Field(..., min_items=1)
    parameters: Dict[str, Any] = Field(default={})


class BacktestResult(BaseModel):
    """Backtesting result"""

    strategy_id: str
    start_date: datetime
    end_date: datetime
    total_return: Decimal = Field(..., decimal_places=4)
    max_drawdown: Decimal = Field(..., decimal_places=4)
    sharpe_ratio: float
    total_trades: int = Field(..., ge=0)
    win_rate: float = Field(..., ge=0, le=100)
    profit_factor: float
    final_balance: Decimal = Field(..., decimal_places=2)


# Health check models
class HealthCheck(BaseModel):
    """Health check response"""

    status: str = "healthy"
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    version: str = "5.0.0"
    uptime_seconds: float
    checks: Dict[str, bool] = Field(default={})


class ComponentHealth(BaseModel):
    """Individual component health"""

    name: str
    status: ComponentStatus
    last_check: datetime
    response_time_ms: float
    error_message: Optional[str] = None


# Export all models for easy importing
__all__ = [
    # Enums
    "ResponseStatus",
    "TradingState",
    "OrderSide",
    "OrderType",
    "OrderStatus",
    "ComponentStatus",
    # Base models
    "BaseResponse",
    "ErrorResponse",
    "SuccessResponse",
    # Authentication
    "LoginRequest",
    "LoginResponse",
    "RefreshTokenRequest",
    "TokenResponse",
    # System status
    "PerformanceMetrics",
    "ComponentInfo",
    "SystemStatus",
    # Trading
    "TradingConfig",
    "MarketData",
    "Position",
    "Order",
    "PortfolioBalance",
    "PnLSummary",
    "PortfolioStatus",
    "TradingResponse",
    "OrderRequest",
    "SignalData",
    # Performance
    "AIComponentMetrics",
    "TradingMetrics",
    "SessionMetrics",
    # WebSocket
    "WebSocketMessage",
    "SubscriptionRequest",
    "HeartbeatMessage",
    # Configuration
    "APIConfig",
    "MonitoringConfig",
    # Error handling
    "ValidationError",
    "APIError",
    # Batch operations
    "BatchOrderRequest",
    "BatchOrderResponse",
    # Analytics
    "BacktestRequest",
    "BacktestResult",
    # Health
    "HealthCheck",
    "ComponentHealth",
]
