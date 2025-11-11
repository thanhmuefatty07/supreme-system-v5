"""
Prometheus Metrics for Supreme System V5

Comprehensive monitoring and metrics collection for production trading system.
Provides real-time insights into system health, performance, and trading activity.
"""

import time
import uuid
from typing import Dict, Any, Optional, List
from prometheus_client import (
    Counter, Gauge, Histogram, Summary, CollectorRegistry,
    generate_latest, CONTENT_TYPE_LATEST
)
from prometheus_client.core import GaugeMetricFamily, CounterMetricFamily
import psutil
import threading
from datetime import datetime, timedelta
from collections import defaultdict, deque

# OpenTelemetry imports (optional)
try:
    from opentelemetry import trace, metrics
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.metrics import MeterProvider
    from opentelemetry.sdk.trace.export import BatchSpanProcessor
    from opentelemetry.exporter.jaeger.thrift import JaegerExporter
    from opentelemetry.sdk.resources import Resource
    from opentelemetry.trace import Status, StatusCode
    from opentelemetry.metrics import CallbackOptions, Observation
    OPENTELEMETRY_AVAILABLE = True
except ImportError:
    OPENTELEMETRY_AVAILABLE = False
    trace = None
    metrics = None


class SupremeSystemMetrics:
    """
    Comprehensive metrics collection for Supreme System V5.

    Provides Prometheus-compatible metrics for monitoring trading system health,
    performance, and business metrics.
    """

    def __init__(self, registry: Optional[CollectorRegistry] = None):
        """
        Initialize metrics collection.

        Args:
            registry: Optional custom Prometheus registry
        """
        self.registry = registry or CollectorRegistry()

        # Initialize OpenTelemetry if available
        self._setup_opentelemetry()

        # Correlation ID tracking
        self._correlation_ids = defaultdict(lambda: str(uuid.uuid4()))
        self._active_spans = {}

        # Latency tracking
        self._latency_buffers = defaultdict(lambda: deque(maxlen=1000))
        self._performance_history = deque(maxlen=10000)

    def _setup_opentelemetry(self):
        """Setup OpenTelemetry tracing and metrics."""
        if not OPENTELEMETRY_AVAILABLE:
            self.tracer = None
            self.meter = None
            return

        # Setup tracing
        resource = Resource.create({"service.name": "supreme-system-v5"})
        trace.set_tracer_provider(TracerProvider(resource=resource))

        # Jaeger exporter for distributed tracing
        jaeger_exporter = JaegerExporter(
            agent_host_name="localhost",
            agent_port=6831,
        )
        span_processor = BatchSpanProcessor(jaeger_exporter)
        trace.get_tracer_provider().add_span_processor(span_processor)

        self.tracer = trace.get_tracer(__name__)

        # Setup metrics
        metrics.set_meter_provider(MeterProvider(resource=resource))
        self.meter = metrics.get_meter(__name__)

        # Create OpenTelemetry metrics
        self.ot_trading_operations = self.meter.create_counter(
            "trading_operations_total",
            description="Total number of trading operations"
        )

        self.ot_operation_duration = self.meter.create_histogram(
            "operation_duration_seconds",
            description="Duration of operations in seconds"
        )

        self.ot_portfolio_value = self.meter.create_gauge(
            "portfolio_value",
            description="Current portfolio value",
            unit="USD"
        )

        logger.info("OpenTelemetry initialized with Jaeger tracing")

    # ===== DISTRIBUTED TRACING METHODS =====

    def start_trace(self, operation: str, correlation_id: Optional[str] = None,
                   attributes: Optional[Dict[str, Any]] = None) -> Optional[str]:
        """
        Start a distributed trace span.

        Args:
            operation: Operation name
            correlation_id: Optional correlation ID
            attributes: Span attributes

        Returns:
            Span ID or None if tracing not available
        """
        if not self.tracer:
            return None

        span_id = str(uuid.uuid4())
        correlation_id = correlation_id or self._correlation_ids[operation]

        with self.tracer.start_as_span(operation) as span:
            span.set_attribute("correlation_id", correlation_id)
            span.set_attribute("operation", operation)

            if attributes:
                for key, value in attributes.items():
                    span.set_attribute(key, str(value))

            self._active_spans[span_id] = span
            return span_id

    def end_trace(self, span_id: str, status: str = "success",
                  error: Optional[str] = None):
        """
        End a distributed trace span.

        Args:
            span_id: Span ID to end
            status: Operation status
            error: Error message if failed
        """
        if span_id not in self._active_spans:
            return

        span = self._active_spans[span_id]

        if status == "error" and error:
            span.set_status(Status(StatusCode.ERROR, error))
        elif status == "success":
            span.set_status(Status(StatusCode.OK))

        span.end()
        del self._active_spans[span_id]

    def get_correlation_id(self, operation: str) -> str:
        """Get correlation ID for an operation."""
        return self._correlation_ids[operation]

    # ===== LATENCY HISTOGRAMS =====

    def record_latency(self, operation: str, duration: float,
                      attributes: Optional[Dict[str, Any]] = None):
        """
        Record operation latency for histogram analysis.

        Args:
            operation: Operation name
            duration: Duration in seconds
            attributes: Additional attributes
        """
        # Store in latency buffer
        self._latency_buffers[operation].append({
            'duration': duration,
            'timestamp': time.time(),
            'attributes': attributes or {}
        })

        # Record in OpenTelemetry if available
        if self.meter:
            self.ot_operation_duration.record(
                duration,
                {
                    "operation": operation,
                    **(attributes or {})
                }
            )

    def get_latency_percentiles(self, operation: str,
                               percentiles: List[float] = [50, 95, 99]) -> Dict[str, float]:
        """
        Get latency percentiles for an operation.

        Args:
            operation: Operation name
            percentiles: Percentiles to calculate

        Returns:
            Dict of percentile -> latency mapping
        """
        latencies = [entry['duration'] for entry in self._latency_buffers[operation]]

        if not latencies:
            return {f'p{p}': 0.0 for p in percentiles}

        sorted_latencies = sorted(latencies)
        n = len(sorted_latencies)

        result = {}
        for p in percentiles:
            index = int((p / 100) * (n - 1))
            result[f'p{p}'] = sorted_latencies[index]

        return result

    # ===== BUSINESS METRICS =====

    def record_trading_operation(self, operation_type: str, symbol: str,
                               quantity: float, price: float, pnl: Optional[float] = None):
        """
        Record business-level trading metrics.

        Args:
            operation_type: Type of operation (buy, sell, etc.)
            symbol: Trading symbol
            quantity: Quantity traded
            price: Execution price
            pnl: Profit/loss if applicable
        """
        # Record in Prometheus
        self.trading_operations_total.labels(
            operation=operation_type,
            symbol=symbol
        ).inc()

        self.last_trade_price.labels(symbol=symbol).set(price)

        if pnl is not None:
            self.total_pnl.inc(pnl)

        # Record in OpenTelemetry
        if self.meter:
            self.ot_trading_operations.add(
                1,
                {
                    "operation_type": operation_type,
                    "symbol": symbol,
                    "quantity": quantity,
                    "price": price,
                    "pnl": pnl or 0.0
                }
            )

    def update_business_metrics(self, portfolio_value: float,
                               daily_pnl: float, win_rate: float):
        """
        Update high-level business metrics.

        Args:
            portfolio_value: Current portfolio value
            daily_pnl: Daily profit/loss
            win_rate: Trading win rate
        """
        self.portfolio_value.set(portfolio_value)
        self.daily_pnl.set(daily_pnl)
        self.win_rate.set(win_rate)

        # Record in OpenTelemetry
        if self.meter:
            self.ot_portfolio_value.set(portfolio_value)

    # ===== RESOURCE USAGE TRACKING =====

    def record_resource_usage(self, operation: str):
        """
        Record system resource usage during an operation.

        Args:
            operation: Operation name
        """
        cpu_percent = psutil.cpu_percent(interval=0.1)
        memory_percent = psutil.virtual_memory().percent
        disk_usage = psutil.disk_usage('/').percent

        self.cpu_usage_during_operation.labels(operation=operation).observe(cpu_percent)
        self.memory_usage_during_operation.labels(operation=operation).observe(memory_percent)
        self.disk_usage_during_operation.labels(operation=operation).observe(disk_usage)

    def get_resource_usage_summary(self) -> Dict[str, Any]:
        """
        Get summary of resource usage across all operations.

        Returns:
            Resource usage summary
        """
        return {
            'cpu_percent': psutil.cpu_percent(),
            'memory_percent': psutil.virtual_memory().percent,
            'memory_used_gb': psutil.virtual_memory().used / (1024**3),
            'disk_percent': psutil.disk_usage('/').percent,
            'network_connections': len(psutil.net_connections()),
            'open_files': len(psutil.Process().open_files()) if psutil.Process().open_files() else 0
        }


# ===== COMPLETED ENHANCED MONITORING =====

# System Health Metrics
        self.system_uptime = Gauge(
            'supreme_system_uptime_seconds',
            'System uptime in seconds',
            registry=self.registry
        )

        self.system_memory_usage = Gauge(
            'supreme_system_memory_usage_bytes',
            'Current memory usage in bytes',
            registry=self.registry
        )

        self.system_cpu_usage = Gauge(
            'supreme_system_cpu_usage_percent',
            'Current CPU usage percentage',
            registry=self.registry
        )

        # Trading Activity Metrics
        self.trades_total = Counter(
            'supreme_trades_total',
            'Total number of trades executed',
            ['symbol', 'action', 'strategy', 'result'],
            registry=self.registry
        )

        self.trades_value_total = Counter(
            'supreme_trades_value_total',
            'Total value of trades executed in base currency',
            ['symbol', 'action'],
            registry=self.registry
        )

        self.positions_active = Gauge(
            'supreme_positions_active',
            'Number of currently active positions',
            ['symbol'],
            registry=self.registry
        )

        self.portfolio_value = Gauge(
            'supreme_portfolio_value',
            'Current portfolio total value',
            registry=self.registry
        )

        # Strategy Performance Metrics
        self.strategy_signals_generated = Counter(
            'supreme_strategy_signals_total',
            'Total trading signals generated by strategies',
            ['strategy_name', 'signal_type', 'symbol'],
            registry=self.registry
        )

        self.strategy_execution_time = Histogram(
            'supreme_strategy_execution_time_seconds',
            'Time taken to execute strategy logic',
            ['strategy_name'],
            buckets=(0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0, 5.0),
            registry=self.registry
        )

        # Risk Management Metrics
        self.risk_checks_total = Counter(
            'supreme_risk_checks_total',
            'Total risk assessment checks performed',
            ['check_type', 'result'],
            registry=self.registry
        )

        self.circuit_breaker_trips = Counter(
            'supreme_circuit_breaker_trips_total',
            'Total circuit breaker activations',
            ['reason'],
            registry=self.registry
        )

        self.max_drawdown_current = Gauge(
            'supreme_max_drawdown_percent',
            'Current maximum drawdown percentage',
            registry=self.registry
        )

        # Data Pipeline Metrics
        self.data_requests_total = Counter(
            'supreme_data_requests_total',
            'Total data requests made to external APIs',
            ['source', 'endpoint', 'result'],
            registry=self.registry
        )

        self.data_request_duration = Histogram(
            'supreme_data_request_duration_seconds',
            'Time taken for data API requests',
            ['source', 'endpoint'],
            buckets=(0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0),
            registry=self.registry
        )

        self.data_cache_hits = Counter(
            'supreme_data_cache_hits_total',
            'Total data cache hits',
            ['cache_type'],
            registry=self.registry
        )

        # Error and Exception Metrics
        self.errors_total = Counter(
            'supreme_errors_total',
            'Total errors and exceptions',
            ['error_type', 'component', 'severity'],
            registry=self.registry
        )

        self.api_errors_total = Counter(
            'supreme_api_errors_total',
            'Total API-related errors',
            ['api_name', 'error_type'],
            registry=self.registry
        )

        # Performance Metrics
        self.request_processing_time = Histogram(
            'supreme_request_processing_time_seconds',
            'Time taken to process trading requests',
            ['request_type'],
            buckets=(0.001, 0.005, 0.01, 0.05, 0.1, 0.5),
            registry=self.registry
        )

        self.database_query_time = Histogram(
            'supreme_database_query_time_seconds',
            'Time taken for database queries',
            ['query_type', 'table'],
            buckets=(0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0),
            registry=self.registry
        )

        # Business Metrics
        self.pnl_realized = Counter(
            'supreme_pnl_realized_total',
            'Total realized profit and loss',
            ['symbol', 'strategy'],
            registry=self.registry
        )

        self.pnl_unrealized = Gauge(
            'supreme_pnl_unrealized_current',
            'Current unrealized profit and loss',
            ['symbol'],
            registry=self.registry
        )

        self.sharpe_ratio_current = Gauge(
            'supreme_sharpe_ratio_current',
            'Current Sharpe ratio',
            registry=self.registry
        )

        # Initialize system metrics
        self._start_time = time.time()
        self._update_system_metrics()

        # Start background metrics updater
        self._stop_event = threading.Event()
        self._metrics_thread = threading.Thread(
            target=self._background_metrics_updater,
            daemon=True
        )
        self._metrics_thread.start()

    def _update_system_metrics(self):
        """Update system-level metrics."""
        try:
            # System uptime
            self.system_uptime.set(time.time() - self._start_time)

            # Memory usage
            memory = psutil.virtual_memory()
            self.system_memory_usage.set(memory.used)

            # CPU usage
            cpu_percent = psutil.cpu_percent(interval=1)
            self.system_cpu_usage.set(cpu_percent)

        except Exception as e:
            self.errors_total.labels(
                error_type='system_metrics_update',
                component='monitoring',
                severity='warning'
            ).inc()

    def _background_metrics_updater(self):
        """Background thread to update system metrics periodically."""
        while not self._stop_event.is_set():
            self._update_system_metrics()
            time.sleep(30)  # Update every 30 seconds

    def record_trade(self, symbol: str, action: str, quantity: float,
                    price: float, strategy: str, result: str):
        """
        Record a trade execution.

        Args:
            symbol: Trading symbol
            action: Trade action (BUY/SELL)
            quantity: Trade quantity
            price: Trade price
            strategy: Strategy that generated the trade
            result: Trade result (FILLED/REJECTED/ERROR)
        """
        # Record trade count
        self.trades_total.labels(
            symbol=symbol,
            action=action,
            strategy=strategy,
            result=result
        ).inc()

        # Record trade value
        trade_value = quantity * price
        self.trades_value_total.labels(
            symbol=symbol,
            action=action
        ).inc(trade_value)

    def record_strategy_signal(self, strategy_name: str, signal_type: str,
                              symbol: str, execution_time: Optional[float] = None):
        """
        Record strategy signal generation.

        Args:
            strategy_name: Name of the strategy
            signal_type: Type of signal generated
            symbol: Target symbol
            execution_time: Time taken to generate signal
        """
        self.strategy_signals_generated.labels(
            strategy_name=strategy_name,
            signal_type=signal_type,
            symbol=symbol
        ).inc()

        if execution_time is not None:
            self.strategy_execution_time.labels(
                strategy_name=strategy_name
            ).observe(execution_time)

    def record_risk_check(self, check_type: str, result: str):
        """
        Record risk management check.

        Args:
            check_type: Type of risk check performed
            result: Result of the check (PASSED/FAILED)
        """
        self.risk_checks_total.labels(
            check_type=check_type,
            result=result
        ).inc()

    def record_data_request(self, source: str, endpoint: str,
                          result: str, duration: Optional[float] = None):
        """
        Record data API request.

        Args:
            source: Data source name
            endpoint: API endpoint
            result: Request result (SUCCESS/ERROR)
            duration: Request duration in seconds
        """
        self.data_requests_total.labels(
            source=source,
            endpoint=endpoint,
            result=result
        ).inc()

        if duration is not None:
            self.data_request_duration.labels(
                source=source,
                endpoint=endpoint
            ).observe(duration)

    def record_error(self, error_type: str, component: str, severity: str = "error"):
        """
        Record application error.

        Args:
            error_type: Type of error
            component: Component where error occurred
            severity: Error severity level
        """
        self.errors_total.labels(
            error_type=error_type,
            component=component,
            severity=severity
        ).inc()

    def update_portfolio_metrics(self, portfolio_value: float,
                               positions_count: Dict[str, int],
                               unrealized_pnl: Dict[str, float],
                               sharpe_ratio: Optional[float] = None,
                               max_drawdown: Optional[float] = None):
        """
        Update portfolio-level metrics.

        Args:
            portfolio_value: Current total portfolio value
            positions_count: Active positions by symbol
            unrealized_pnl: Unrealized P&L by symbol
            sharpe_ratio: Current Sharpe ratio
            max_drawdown: Current maximum drawdown
        """
        # Portfolio value
        self.portfolio_value.set(portfolio_value)

        # Positions count
        for symbol, count in positions_count.items():
            self.positions_active.labels(symbol=symbol).set(count)

        # Unrealized P&L
        for symbol, pnl in unrealized_pnl.items():
            self.pnl_unrealized.labels(symbol=symbol).set(pnl)

        # Risk metrics
        if sharpe_ratio is not None:
            self.sharpe_ratio_current.set(sharpe_ratio)

        if max_drawdown is not None:
            self.max_drawdown_current.set(max_drawdown)

    def record_pnl_realized(self, symbol: str, strategy: str, pnl_amount: float):
        """
        Record realized profit and loss.

        Args:
            symbol: Trading symbol
            strategy: Strategy that generated the P&L
            pnl_amount: Realized P&L amount
        """
        self.pnl_realized.labels(
            symbol=symbol,
            strategy=strategy
        ).inc(pnl_amount)

    def get_metrics_text(self) -> str:
        """
        Get all metrics in Prometheus text format.

        Returns:
            Metrics in Prometheus exposition format
        """
        return generate_latest(self.registry).decode('utf-8')

    def shutdown(self):
        """Shutdown metrics collection and background threads."""
        self._stop_event.set()
        if self._metrics_thread.is_alive():
            self._metrics_thread.join(timeout=5)


# Global metrics instance
_metrics_instance = None

def get_metrics() -> SupremeSystemMetrics:
    """Get global metrics instance (singleton pattern)."""
    global _metrics_instance
    if _metrics_instance is None:
        _metrics_instance = SupremeSystemMetrics()
    return _metrics_instance


def record_trade_execution(symbol: str, action: str, quantity: float,
                          price: float, strategy: str, result: str):
    """Convenience function to record trade execution."""
    get_metrics().record_trade(symbol, action, quantity, price, strategy, result)


def record_api_error(api_name: str, error_type: str):
    """Convenience function to record API errors."""
    get_metrics().api_errors_total.labels(
        api_name=api_name,
        error_type=error_type
    ).inc()


def time_request_processing(request_type: str):
    """Decorator to time request processing."""
    def decorator(func):
        def wrapper(*args, **kwargs):
            start_time = time.time()
            try:
                result = func(*args, **kwargs)
                processing_time = time.time() - start_time
                get_metrics().request_processing_time.labels(
                    request_type=request_type
                ).observe(processing_time)
                return result
            except Exception as e:
                processing_time = time.time() - start_time
                get_metrics().request_processing_time.labels(
                    request_type=request_type
                ).observe(processing_time)
                raise
        return wrapper
    return decorator


# Custom collector for system information
class SystemInfoCollector:
    """Custom Prometheus collector for system information."""

    def collect(self):
        """Collect system information metrics."""
        # CPU information
        cpu_count = psutil.cpu_count()
        cpu_freq = psutil.cpu_freq()
        if cpu_freq:
            yield GaugeMetricFamily(
                'supreme_system_cpu_count',
                'Total number of CPU cores',
                value=cpu_count
            )
            yield GaugeMetricFamily(
                'supreme_system_cpu_frequency_mhz',
                'Current CPU frequency in MHz',
                value=cpu_freq.current
            )

        # Memory information
        memory = psutil.virtual_memory()
        yield GaugeMetricFamily(
            'supreme_system_memory_total_bytes',
            'Total system memory in bytes',
            value=memory.total
        )

        # Disk information
        disk = psutil.disk_usage('/')
        yield GaugeMetricFamily(
            'supreme_system_disk_total_bytes',
            'Total disk space in bytes',
            value=disk.total
        )
        yield GaugeMetricFamily(
            'supreme_system_disk_free_bytes',
            'Free disk space in bytes',
            value=disk.free
        )

        # Network information
        net = psutil.net_io_counters()
        if net:
            yield CounterMetricFamily(
                'supreme_system_network_bytes_sent_total',
                'Total bytes sent over network',
                value=net.bytes_sent
            )
            yield CounterMetricFamily(
                'supreme_system_network_bytes_recv_total',
                'Total bytes received over network',
                value=net.bytes_recv
            )


# Initialize system info collector
_system_collector = SystemInfoCollector()

def register_system_collector(registry: CollectorRegistry):
    """Register system information collector."""
    registry.register(_system_collector)


if __name__ == "__main__":
    # Example usage
    metrics = get_metrics()

    # Simulate some activity
    metrics.record_trade("AAPL", "BUY", 100, 150.0, "momentum", "FILLED")
    metrics.record_strategy_signal("momentum", "BUY", "AAPL", 0.023)
    metrics.record_data_request("binance", "klines", "SUCCESS", 0.15)

    # Print metrics
    print("Current Metrics:")
    print(metrics.get_metrics_text())
