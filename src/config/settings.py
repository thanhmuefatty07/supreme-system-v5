"""
Production-ready configuration management for Supreme System V5.

Uses Pydantic for type-safe, validated configuration with environment variable support.
"""

import os
import json
from pathlib import Path
from typing import Dict, List, Optional, Union, Any
from decimal import Decimal

from pydantic import BaseSettings, Field, validator, root_validator
from pydantic.env_settings import BaseEnvSettings


class TradingConfig(BaseSettings):
    """Trading-related configuration."""

    # Capital and position sizing
    initial_capital: float = Field(100000.0, gt=0, description="Initial trading capital")
    max_position_size_pct: float = Field(0.02, gt=0, le=1, description="Max position size as % of capital")
    max_daily_loss_pct: float = Field(0.05, gt=0, le=1, description="Max daily loss as % of capital")

    # Trading parameters
    symbols: List[str] = Field(['AAPL', 'MSFT', 'GOOGL'], min_items=1, description="Trading symbols")
    default_timeframe: str = Field('1h', description="Default trading timeframe")

    # Risk management
    stop_loss_pct: float = Field(0.02, gt=0, le=1, description="Default stop loss percentage")
    take_profit_pct: float = Field(0.05, gt=0, le=1, description="Default take profit percentage")
    max_holding_period_hours: int = Field(24, gt=0, description="Max holding period in hours")

    # Transaction costs
    commission_per_trade: float = Field(0.001, ge=0, description="Commission per trade")
    slippage_pct: float = Field(0.0005, ge=0, description="Slippage percentage")

    class Config:
        env_prefix = "TRADING_"


class DataConfig(BaseSettings):
    """Data management configuration."""

    # Data sources
    primary_source: str = Field('binance', description="Primary data source")
    backup_sources: List[str] = Field([], description="Backup data sources")

    # Data storage
    data_directory: str = Field('./data', description="Data storage directory")
    cache_enabled: bool = Field(True, description="Enable data caching")
    cache_ttl_hours: int = Field(24, gt=0, description="Cache TTL in hours")

    # Data validation
    max_data_age_days: int = Field(1, gt=0, description="Maximum data age in days")
    require_volume_validation: bool = Field(True, description="Require volume data validation")

    # Historical data
    historical_data_path: str = Field('./data/historical', description="Historical data path")
    preload_symbols: List[str] = Field([], description="Symbols to preload")

    class Config:
        env_prefix = "DATA_"


class APIConfig(BaseSettings):
    """External API configuration."""

    # Binance API (required for live trading)
    binance_api_key: Optional[str] = Field(None, description="Binance API key")
    binance_api_secret: Optional[str] = Field(None, description="Binance API secret")
    binance_testnet: bool = Field(True, description="Use Binance testnet")

    # Bybit API (alternative exchange)
    bybit_api_key: Optional[str] = Field(None, description="Bybit API key")
    bybit_api_secret: Optional[str] = Field(None, description="Bybit API secret")
    bybit_testnet: bool = Field(True, description="Use Bybit testnet")

    # Exchange selection
    primary_exchange: str = Field("binance", description="Primary exchange: 'binance' or 'bybit'")

    # Alternative data providers
    alpha_vantage_api_key: Optional[str] = Field(None, description="Alpha Vantage API key")
    yahoo_finance_enabled: bool = Field(True, description="Enable Yahoo Finance fallback")

    # API rate limiting
    api_timeout_seconds: int = Field(30, gt=0, description="API timeout in seconds")
    max_retries: int = Field(3, ge=0, description="Maximum API retries")
    retry_backoff_seconds: float = Field(1.0, gt=0, description="Retry backoff seconds")

    @validator('binance_api_key', 'binance_api_secret')
    def validate_api_credentials(cls, v, field):
        """Validate API credentials are provided for live trading."""
        if not cls().binance_testnet and not v:
            raise ValueError(f"{field.name} is required when not using testnet")
        return v

    class Config:
        env_prefix = "API_"


class RiskConfig(BaseSettings):
    """Risk management configuration."""

    # Circuit breaker
    circuit_breaker_enabled: bool = Field(True, description="Enable circuit breaker")
    max_daily_loss_pct: float = Field(0.05, gt=0, le=1, description="Circuit breaker threshold")
    circuit_breaker_cooldown_minutes: int = Field(60, gt=0, description="Circuit breaker cooldown")

    # Position limits
    max_positions: int = Field(5, gt=0, description="Maximum concurrent positions")
    max_symbol_exposure_pct: float = Field(0.1, gt=0, le=1, description="Max exposure per symbol")

    # Risk metrics
    var_confidence_level: float = Field(0.95, gt=0, lt=1, description="VaR confidence level")
    stress_test_enabled: bool = Field(True, description="Enable stress testing")

    # Advanced risk controls
    correlation_limits: Dict[str, float] = Field({}, description="Symbol correlation limits")
    sector_exposure_limits: Dict[str, float] = Field({}, description="Sector exposure limits")

    class Config:
        env_prefix = "RISK_"


class MonitoringConfig(BaseSettings):
    """Monitoring and alerting configuration."""

    # Logging
    log_level: str = Field('INFO', description="Logging level")
    log_file: Optional[str] = Field('./logs/supreme_system.log', description="Log file path")
    log_max_size_mb: int = Field(100, gt=0, description="Max log file size in MB")
    log_backup_count: int = Field(5, ge=0, description="Number of log backups")

    # Metrics
    metrics_enabled: bool = Field(True, description="Enable metrics collection")
    metrics_port: int = Field(8001, gt=0, description="Metrics server port")

    # Alerts
    alerts_enabled: bool = Field(True, description="Enable alerting")
    alert_email_recipients: List[str] = Field([], description="Alert email recipients")
    alert_sms_numbers: List[str] = Field([], description="Alert SMS numbers")

    # Health checks
    health_check_interval_seconds: int = Field(60, gt=0, description="Health check interval")
    health_check_timeout_seconds: int = Field(10, gt=0, description="Health check timeout")

    class Config:
        env_prefix = "MONITORING_"


class SystemConfig(BaseSettings):
    """System-level configuration."""

    # Environment
    environment: str = Field('development', description="Deployment environment")
    debug_mode: bool = Field(False, description="Enable debug mode")

    # Performance
    max_workers: int = Field(4, gt=0, description="Maximum worker threads")
    chunk_size: int = Field(10000, gt=0, description="Data processing chunk size")
    memory_limit_mb: int = Field(2048, gt=0, description="Memory limit in MB")

    # Security
    secret_key: str = Field(..., description="Application secret key")
    allowed_hosts: List[str] = Field(['localhost'], description="Allowed hosts")
    cors_origins: List[str] = Field(['http://localhost:3000'], description="CORS origins")

    # Database
    database_url: Optional[str] = Field(None, description="Database connection URL")
    database_pool_size: int = Field(10, gt=0, description="Database connection pool size")

    @validator('environment')
    def validate_environment(cls, v):
        """Validate environment value."""
        valid_environments = ['development', 'staging', 'production']
        if v not in valid_environments:
            raise ValueError(f"Environment must be one of: {valid_environments}")
        return v

    @validator('secret_key')
    def validate_secret_key(cls, v):
        """Validate secret key strength."""
        if len(v) < 32:
            raise ValueError("Secret key must be at least 32 characters long")
        return v

    class Config:
        env_prefix = "SYSTEM_"


class SupremeSystemSettings(BaseSettings):
    """
    Main configuration class for Supreme System V5.

    Combines all configuration sections with validation and environment variable support.
    """

    # Configuration sections
    trading: TradingConfig = Field(default_factory=TradingConfig)
    data: DataConfig = Field(default_factory=DataConfig)
    api: APIConfig = Field(default_factory=APIConfig)
    risk: RiskConfig = Field(default_factory=RiskConfig)
    monitoring: MonitoringConfig = Field(default_factory=MonitoringConfig)
    system: SystemConfig

    # Global settings
    version: str = Field("5.0.0", description="Application version")
    config_file: Optional[str] = Field(None, description="Configuration file path")

    @root_validator
    def validate_configuration(cls, values):
        """Validate entire configuration for consistency."""
        # Validate trading capital vs position sizing
        trading = values.get('trading')
        if trading:
            max_position_value = trading.initial_capital * trading.max_position_size_pct
            if max_position_value < 100:  # Minimum position size
                raise ValueError("Maximum position size too small for effective trading")

        # Validate API configuration for production
        system = values.get('system')
        api = values.get('api')
        if system and system.environment == 'production':
            if not api or not api.binance_api_key:
                raise ValueError("API credentials required for production environment")

        return values

    @classmethod
    def from_env_file(cls, env_file: str = '.env') -> 'SupremeSystemSettings':
        """Load configuration from environment file."""
        if os.path.exists(env_file):
            from dotenv import load_dotenv
            load_dotenv(env_file)

        return cls()

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'SupremeSystemSettings':
        """Load configuration from dictionary."""
        return cls(**config_dict)

    def to_dict(self) -> Dict[str, Any]:
        """Export configuration to dictionary."""
        return self.dict()

    def save_to_file(self, file_path: str) -> None:
        """Save configuration to JSON file."""
        config_dict = self.to_dict()

        # Convert non-serializable objects
        for key, value in config_dict.items():
            if hasattr(value, '__dict__'):
                config_dict[key] = value.__dict__

        with open(file_path, 'w') as f:
            json.dump(config_dict, f, indent=2, default=str)

    @classmethod
    def load_from_file(cls, file_path: str) -> 'SupremeSystemSettings':
        """Load configuration from JSON file."""
        with open(file_path, 'r') as f:
            config_dict = json.load(f)

        return cls(**config_dict)

    def get_database_url(self) -> Optional[str]:
        """Get database URL with fallback logic."""
        return self.system.database_url or os.getenv('DATABASE_URL')

    def is_production(self) -> bool:
        """Check if running in production environment."""
        return self.system.environment == 'production'

    def is_development(self) -> bool:
        """Check if running in development environment."""
        return self.system.environment == 'development'

    class Config:
        env_file = '.env'
        case_sensitive = True


# Global settings instance
settings = None

def get_settings() -> SupremeSystemSettings:
    """Get global settings instance (singleton pattern)."""
    global settings
    if settings is None:
        settings = SupremeSystemSettings.from_env_file()
    return settings

def reload_settings() -> SupremeSystemSettings:
    """Reload settings from environment."""
    global settings
    settings = SupremeSystemSettings.from_env_file()
    return settings

def validate_configuration() -> bool:
    """Validate current configuration."""
    try:
        settings = get_settings()
        # Additional validation logic can be added here
        return True
    except Exception as e:
        print(f"Configuration validation failed: {e}")
        return False
