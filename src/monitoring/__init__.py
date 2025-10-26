"""
📈 Supreme System V5 - Monitoring Module
Prometheus-based monitoring and alerting system

Features:
- Prometheus metrics export
- 8 Tier-1 core metrics tracking
- Alert threshold management
- Grafana dashboard compatibility
- Real-time performance monitoring
- Component health tracking
"""

from .alerts import AlertManager, AlertSeverity, AlertThreshold, configure_alert_thresholds
from .health import ComponentHealth, HealthChecker, quick_health_check
from .metrics import (
    MetricsCollector,
    PrometheusExporter,
    get_metrics_summary,
    start_metrics_server,
)

__version__ = "5.0.0"
__author__ = "Supreme System V5 Team"

__all__ = [
    "PrometheusExporter",
    "MetricsCollector",
    "HealthChecker",
    "AlertManager",
    "start_metrics_server",
    "get_metrics_summary",
    "quick_health_check",
    "perform_health_check",  # Alias for backward compatibility
    "configure_alert_thresholds",
    "ComponentHealth",
    "AlertThreshold",
    "AlertSeverity",
]

# Monitoring specifications
MONITORING_SPECS = {
    "prometheus_port": 9090,
    "metrics_endpoint": "/metrics",
    "health_endpoint": "/health",
    "alerts_enabled": True,
    "retention_days": 30,
    "scrape_interval_seconds": 15,
}

# Tier-1 Core Metrics (8 metrics as per requirements)
CORE_METRICS = {
    "api_latency_ms": "API endpoint response latency in milliseconds",
    "websocket_clients": "Number of active WebSocket connections",
    "trading_loop_ms": "Trading loop execution time in milliseconds",
    "orders_executed": "Total number of orders executed",
    "pnl_daily": "Daily profit and loss in USD",
    "exchange_connectivity": "Exchange connection status (1=connected, 0=disconnected)",
    "gross_exposure_usd": "Total gross exposure in USD",
    "max_drawdown_pct": "Maximum drawdown percentage",
}

# Default alert thresholds as per requirements
DEFAULT_ALERT_THRESHOLDS = {
    "api_latency_ms": {"warning": 50.0, "critical": 100.0, "duration_minutes": 3},
    "trading_loop_ms": {"warning": 50.0, "critical": 100.0, "duration_minutes": 1},
    "exchange_connectivity": {
        "critical": 0,  # Alert if disconnected
        "duration_seconds": 10,
    },
    "pnl_daily": {
        "warning": -500.0,  # -$500 daily loss
        "critical": -1000.0,  # -$1000 daily loss
        "duration_minutes": 5,
    },
    "websocket_clients": {"warning": 1000, "critical": 1500},  # High connection count
    "max_drawdown_pct": {
        "warning": 5.0,  # 5% drawdown
        "critical": 10.0,  # 10% drawdown
    },
}

# Backward compatibility alias
perform_health_check = quick_health_check
