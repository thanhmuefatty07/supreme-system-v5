"""
Dependency Injection Container for Supreme System V5

Provides centralized dependency management with proper lifecycle,
configuration, and testing support using dependency-injector.
"""

from dependency_injector import containers, providers
from dependency_injector.containers import DeclarativeContainer

# Import core components
from ..data.binance_client import AsyncBinanceClient
from ..data.data_pipeline import DataPipeline
from ..data.data_storage import DataStorage
from ..data.data_validator import DataValidator

# Static imports to eliminate security vulnerabilities (CRITICAL FIX)
try:
    from ..trading.live_trading_engine import LiveTradingEngine
except ImportError:
    LiveTradingEngine = None

try:
    from ..backtesting.production_backtester import ProductionBacktester
except ImportError:
    ProductionBacktester = None

try:
    from ..monitoring.alert_manager import AlertManager
except ImportError:
    AlertManager = None

try:
    from ..supreme_system_app import SupremeSystemApp
except ImportError:
    SupremeSystemApp = None

# ENTERPRISE SERVICE MESH COMPONENTS
try:
    from ..security.service_mesh import ServiceMeshManager
except ImportError:
    ServiceMeshManager = None

try:
    from ..security.zero_trust import ZeroTrustSecurity
except ImportError:
    ZeroTrustSecurity = None

try:
    from ..security.quantum_crypto import QuantumCryptography
except ImportError:
    QuantumCryptography = None

# AI-POWERED AUTONOMOUS SRE
try:
    from ..ai.autonomous_sre import AutonomousSREPlatform
except ImportError:
    AutonomousSREPlatform = None

# REAL-TIME STREAMING ANALYTICS
try:
    from ..streaming.realtime_analytics import RealTimeStreamingAnalytics
except ImportError:
    RealTimeStreamingAnalytics = None
from ..strategies.strategy_registry import StrategyFactory
from ..risk.risk_manager import RiskManager
from ..trading.portfolio_manager import PortfolioManager
from ..monitoring.prometheus_metrics import SupremeSystemMetrics
from ..utils.secrets_manager import SecretsManager
from ..utils.exceptions import ResilienceManager, ErrorRecoveryManager
from ..utils.logger import setup_logging


class CoreContainer(DeclarativeContainer):
    """Core system components container."""

    # Configuration
    config = providers.Configuration()

    # Logging
    logger = providers.Singleton(
        setup_logging,
        level=config.logging.level,
        log_file=config.logging.file,
        console_level=config.logging.console_level
    )

    # Secrets Management
    secrets_manager = providers.Singleton(SecretsManager)

    # Data Components
    data_validator = providers.Singleton(DataValidator)

    binance_client = providers.Singleton(
        AsyncBinanceClient,
        api_key=config.binance.api_key,
        api_secret=config.binance.api_secret,
        testnet=config.binance.testnet,
        secrets_manager=secrets_manager
    )

    data_storage = providers.Singleton(
        DataStorage,
        base_path=config.storage.base_path,
        compression=config.storage.compression
    )

    data_pipeline = providers.Singleton(
        DataPipeline,
        config=config.data_pipeline,
        use_async=config.data_pipeline.use_async,
        client=binance_client,
        storage=data_storage,
        validator=data_validator
    )

    # Strategy Components
    strategy_factory = providers.Singleton(StrategyFactory)

    # Risk Management
    risk_manager = providers.Singleton(
        RiskManager,
        initial_capital=config.risk.initial_capital,
        max_position_size=config.risk.max_position_size,
        stop_loss_pct=config.risk.stop_loss_pct,
        take_profit_pct=config.risk.take_profit_pct
    )

    # Portfolio Management
    portfolio_manager = providers.Singleton(
        PortfolioManager,
        initial_capital=config.portfolio.initial_capital,
        fee_structure=config.portfolio.fee_structure
    )

    # Monitoring
    metrics = providers.Singleton(SupremeSystemMetrics)

    # Resilience
    resilience_manager = providers.Singleton(ResilienceManager)
    error_recovery_manager = providers.Singleton(ErrorRecoveryManager)


class TradingContainer(DeclarativeContainer):
    """Trading-specific components container."""

    # Core dependency
    core = providers.DependenciesContainer()

    # Trading Engine (SECURITY FIX: Static import)
    live_trading_engine = providers.Factory(
        lambda config, portfolio, risk, strategies, metrics, logger:
            LiveTradingEngine(
                config=config,
                portfolio_manager=portfolio,
                risk_manager=risk,
                strategy_factory=strategies,
                metrics=metrics,
                logger=logger
            ) if LiveTradingEngine else None,
        config=core.config.trading,
        portfolio=core.portfolio_manager,
        risk=core.risk_manager,
        strategies=core.strategy_factory,
        metrics=core.metrics,
        logger=core.logger
    )

    # Backtesting Engine (SECURITY FIX: Static import)
    backtester = providers.Factory(
        lambda config, metrics, logger:
            ProductionBacktester(
                config=config,
                metrics=metrics,
                logger=logger
            ) if ProductionBacktester else None,
        config=core.config.backtesting,
        metrics=core.metrics,
        logger=core.logger
    )


class MonitoringContainer(DeclarativeContainer):
    """Monitoring and observability components."""

    core = providers.DependenciesContainer()

    # Prometheus metrics
    prometheus_metrics = providers.Singleton(
        SupremeSystemMetrics,
        service_name=core.config.monitoring.service_name
    )

    # Alert manager (SECURITY FIX: Static import)
    alert_manager = providers.Singleton(
        lambda config, logger:
            AlertManager(
                config=config,
                logger=logger
            ) if AlertManager else None,
        config=core.config.monitoring.alerts,
        logger=core.logger
    )


class SecurityContainer(DeclarativeContainer):
    """Security and compliance components."""

    core = providers.DependenciesContainer()

    # Zero Trust Security
    zero_trust_security = providers.Singleton(
        lambda logger:
            ZeroTrustSecurity() if ZeroTrustSecurity else None,
        logger=core.logger
    )

    # Quantum Cryptography
    quantum_cryptography = providers.Singleton(
        lambda logger:
            QuantumCryptography() if QuantumCryptography else None,
        logger=core.logger
    )

    # Service Mesh (if implemented)
    service_mesh = providers.Singleton(
        lambda config, logger:
            ServiceMeshManager(config=config, logger=logger) if ServiceMeshManager else None,
        config=core.config.security.service_mesh,
        logger=core.logger
    )


class ApplicationContainer(DeclarativeContainer):
    """Main application container combining all components."""

    # Configuration provider
    config = providers.Configuration()

    # Sub-containers
    core = providers.Container(CoreContainer, config=config)
    trading = providers.Container(TradingContainer, core=core)
    monitoring = providers.Container(MonitoringContainer, core=core)
    security = providers.Container(SecurityContainer, core=core)

    # Main application (SECURITY FIX: Static import)
    supreme_system_app = providers.Factory(
        lambda config, core, trading, monitoring, security, logger:
            SupremeSystemApp(
                config=config,
                core_container=core,
                trading_container=trading,
                monitoring_container=monitoring,
                security_container=security,
                logger=logger
            ) if SupremeSystemApp else None,
        config=config,
        core=core,
        trading=trading,
        monitoring=monitoring,
        security=security,
        logger=core.logger
    )


# Global container instance
container = ApplicationContainer()


def init_container(config_dict: dict = None) -> ApplicationContainer:
    """
    Initialize the dependency injection container.

    Args:
        config_dict: Configuration dictionary to override defaults

    Returns:
        Initialized container
    """
    # Set default configuration
    default_config = {
        'logging': {
            'level': 'INFO',
            'file': 'logs/supreme_system.log',
            'console_level': 'WARNING'
        },
        'binance': {
            'api_key': None,
            'api_secret': None,
            'testnet': True
        },
        'storage': {
            'base_path': 'data/',
            'compression': 'parquet'
        },
        'data_pipeline': {
            'use_async': True,
            'batch_size': 1000,
            'timeout': 30
        },
        'risk': {
            'initial_capital': 10000.0,
            'max_position_size': 0.1,
            'stop_loss_pct': 0.02,
            'take_profit_pct': 0.05
        },
        'portfolio': {
            'initial_capital': 10000.0,
            'fee_structure': {'maker': 0.001, 'taker': 0.001}
        },
        'trading': {
            'symbols': ['BTCUSDT', 'ETHUSDT'],
            'strategies': ['momentum', 'trend_following'],
            'risk_limits': {
                'max_position_size': 0.1,
                'max_daily_loss': 0.05,
                'max_drawdown': 0.1
            }
        },
        'backtesting': {
            'initial_capital': 10000.0,
            'commission': 0.001,
            'slippage': 0.001
        },
        'monitoring': {
            'service_name': 'supreme_system_v5',
            'alerts': {
                'enabled': True,
                'channels': ['log', 'email']
            }
        }
    }

    # Override with provided config
    if config_dict:
        def update_dict(base, update):
            for key, value in update.items():
                if isinstance(value, dict) and key in base:
                    update_dict(base[key], value)
                else:
                    base[key] = value

        update_dict(default_config, config_dict)

    # Wire configuration
    container.config.from_dict(default_config)

    return container


def get_container() -> ApplicationContainer:
    """Get the global container instance."""
    return container


# Convenience functions for accessing components
def get_data_pipeline():
    """Get data pipeline instance."""
    return container.core.data_pipeline()


def get_risk_manager():
    """Get risk manager instance."""
    return container.core.risk_manager()


def get_portfolio_manager():
    """Get portfolio manager instance."""
    return container.core.portfolio_manager()


def get_strategy_factory():
    """Get strategy factory instance."""
    return container.core.strategy_factory()


def get_trading_engine():
    """Get live trading engine instance."""
    return container.trading.live_trading_engine()


def get_backtester():
    """Get backtester instance."""
    return container.trading.backtester()


def get_metrics():
    """Get metrics instance."""
    return container.core.metrics()


if __name__ == "__main__":
    # Example usage
    container = init_container()

    # Access components
    pipeline = container.core.data_pipeline()
    risk_mgr = container.core.risk_manager()
    portfolio = container.core.portfolio_manager()

    print("✅ Dependency injection container initialized successfully")
    print(f"Data pipeline: {type(pipeline)}")
    print(f"Risk manager: {type(risk_mgr)}")
    print(f"Portfolio manager: {type(portfolio)}")
