"""
🚀 Supreme System V5 - Health Checks & Circuit Breakers

Implements comprehensive health monitoring and circuit breaker patterns:
- Automated health checks for all system components
- Circuit breaker pattern for fault tolerance
- Service discovery and dependency health validation
- Graceful degradation under load
"""

import asyncio
import time
import threading
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Any, Optional, Callable, Awaitable
from datetime import datetime, timedelta
import json
import logging

try:
    import aiohttp
    import redis
    import psutil
except ImportError:
    # Graceful degradation
    aiohttp = None
    redis = None
    psutil = None

logger = logging.getLogger(__name__)


class HealthStatus(Enum):
    """Health check status enumeration"""
    HEALTHY = "healthy"
    UNHEALTHY = "unhealthy"
    DEGRADED = "degraded"
    UNKNOWN = "unknown"


class CircuitBreakerState(Enum):
    """Circuit breaker states"""
    CLOSED = "closed"      # Normal operation
    OPEN = "open"          # Failing, requests blocked
    HALF_OPEN = "half_open"  # Testing recovery


@dataclass
class HealthCheckResult:
    """Health check result data structure"""
    service_name: str
    status: HealthStatus
    response_time: float
    timestamp: datetime
    details: Dict[str, Any] = field(default_factory=dict)
    error_message: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            'service_name': self.service_name,
            'status': self.status.value,
            'response_time': self.response_time,
            'timestamp': self.timestamp.isoformat(),
            'details': self.details,
            'error_message': self.error_message
        }


class HealthCheck(ABC):
    """Abstract base class for health checks"""

    def __init__(self, name: str, timeout: float = 5.0, interval: float = 30.0):
        self.name = name
        self.timeout = timeout
        self.interval = interval
        self.last_check: Optional[datetime] = None
        self.last_result: Optional[HealthCheckResult] = None
        self.consecutive_failures = 0
        self.total_checks = 0
        self.successful_checks = 0

    @abstractmethod
    async def check_health(self) -> HealthCheckResult:
        """Perform the actual health check"""
        pass

    async def execute(self) -> HealthCheckResult:
        """Execute health check with timing and error handling"""
        start_time = time.time()
        self.total_checks += 1

        try:
            result = await asyncio.wait_for(
                self.check_health(),
                timeout=self.timeout
            )
            result.response_time = time.time() - start_time

            if result.status == HealthStatus.HEALTHY:
                self.successful_checks += 1
                self.consecutive_failures = 0
            else:
                self.consecutive_failures += 1

            self.last_check = datetime.now()
            self.last_result = result

            return result

        except asyncio.TimeoutError:
            response_time = time.time() - start_time
            self.consecutive_failures += 1
            result = HealthCheckResult(
                service_name=self.name,
                status=HealthStatus.UNHEALTHY,
                response_time=response_time,
                timestamp=datetime.now(),
                error_message=f"Health check timed out after {self.timeout}s"
            )
            self.last_result = result
            return result

        except Exception as e:
            response_time = time.time() - start_time
            self.consecutive_failures += 1
            result = HealthCheckResult(
                service_name=self.name,
                status=HealthStatus.UNHEALTHY,
                response_time=response_time,
                timestamp=datetime.now(),
                error_message=str(e)
            )
            self.last_result = result
            return result

    def get_health_score(self) -> float:
        """Calculate health score (0-100)"""
        if self.total_checks == 0:
            return 100.0

        success_rate = self.successful_checks / self.total_checks
        recency_penalty = 0.0

        # Penalize based on recency of failures
        if self.consecutive_failures > 0:
            recency_penalty = min(self.consecutive_failures * 10, 50)

        return max(0, (success_rate * 100) - recency_penalty)


class HTTPHealthCheck(HealthCheck):
    """HTTP endpoint health check"""

    def __init__(self, name: str, url: str, expected_status: int = 200,
                 headers: Optional[Dict[str, str]] = None, **kwargs):
        super().__init__(name, **kwargs)
        self.url = url
        self.expected_status = expected_status
        self.headers = headers or {}

    async def check_health(self) -> HealthCheckResult:
        """Check HTTP endpoint health"""
        if not aiohttp:
            return HealthCheckResult(
                service_name=self.name,
                status=HealthStatus.UNKNOWN,
                response_time=0.0,
                timestamp=datetime.now(),
                error_message="aiohttp not available"
            )

        async with aiohttp.ClientSession() as session:
            try:
                async with session.get(self.url, headers=self.headers) as response:
                    status_code = response.status

                    # Check status
                    if status_code == self.expected_status:
                        status = HealthStatus.HEALTHY
                    elif 200 <= status_code < 400:
                        status = HealthStatus.DEGRADED
                    else:
                        status = HealthStatus.UNHEALTHY

                    # Try to parse response
                    details = {}
                    try:
                        response_text = await response.text()
                        if response_text:
                            details = json.loads(response_text)
                    except:
                        details = {"status_code": status_code}

                    return HealthCheckResult(
                        service_name=self.name,
                        status=status,
                        response_time=0.0,  # Will be set by execute()
                        timestamp=datetime.now(),
                        details=details
                    )

            except aiohttp.ClientError as e:
                return HealthCheckResult(
                    service_name=self.name,
                    status=HealthStatus.UNHEALTHY,
                    response_time=0.0,
                    timestamp=datetime.now(),
                    error_message=f"HTTP request failed: {e}"
                )


class RedisHealthCheck(HealthCheck):
    """Redis connectivity health check"""

    def __init__(self, name: str, host: str = "localhost", port: int = 6379,
                 password: Optional[str] = None, db: int = 0, **kwargs):
        super().__init__(name, **kwargs)
        self.host = host
        self.port = port
        self.password = password
        self.db = db

    async def check_health(self) -> HealthCheckResult:
        """Check Redis connectivity"""
        if not redis:
            return HealthCheckResult(
                service_name=self.name,
                status=HealthStatus.UNKNOWN,
                response_time=0.0,
                timestamp=datetime.now(),
                error_message="redis library not available"
            )

        # Redis operations are blocking, run in thread pool
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(None, self._check_redis_sync)

        return result

    def _check_redis_sync(self) -> HealthCheckResult:
        """Synchronous Redis health check"""
        try:
            client = redis.Redis(
                host=self.host,
                port=self.port,
                password=self.password,
                db=self.db,
                socket_connect_timeout=5,
                socket_timeout=5
            )

            # Test basic operations
            client.ping()

            # Get some stats
            info = client.info()
            details = {
                "connected_clients": info.get("connected_clients", 0),
                "used_memory_human": info.get("used_memory_human", "unknown"),
                "total_connections_received": info.get("total_connections_received", 0),
                "uptime_in_seconds": info.get("uptime_in_seconds", 0)
            }

            client.close()

            return HealthCheckResult(
                service_name=self.name,
                status=HealthStatus.HEALTHY,
                response_time=0.0,
                timestamp=datetime.now(),
                details=details
            )

        except redis.ConnectionError as e:
            return HealthCheckResult(
                service_name=self.name,
                status=HealthStatus.UNHEALTHY,
                response_time=0.0,
                timestamp=datetime.now(),
                error_message=f"Redis connection failed: {e}"
            )
        except Exception as e:
            return HealthCheckResult(
                service_name=self.name,
                status=HealthStatus.UNHEALTHY,
                response_time=0.0,
                timestamp=datetime.now(),
                error_message=f"Redis health check failed: {e}"
            )


class SystemHealthCheck(HealthCheck):
    """System resource health check"""

    def __init__(self, name: str, cpu_threshold: float = 85.0,
                 memory_threshold: float = 90.0, disk_threshold: float = 95.0, **kwargs):
        super().__init__(name, **kwargs)
        self.cpu_threshold = cpu_threshold
        self.memory_threshold = memory_threshold
        self.disk_threshold = disk_threshold

    async def check_health(self) -> HealthCheckResult:
        """Check system resource usage"""
        if not psutil:
            return HealthCheckResult(
                service_name=self.name,
                status=HealthStatus.UNKNOWN,
                response_time=0.0,
                timestamp=datetime.now(),
                error_message="psutil not available"
            )

        # Get system metrics
        cpu_percent = psutil.cpu_percent(interval=1)
        memory = psutil.virtual_memory()
        disk = psutil.disk_usage('/')

        details = {
            "cpu_percent": cpu_percent,
            "memory_percent": memory.percent,
            "memory_used_mb": memory.used / 1024 / 1024,
            "memory_total_mb": memory.total / 1024 / 1024,
            "disk_percent": disk.percent,
            "disk_used_gb": disk.used / 1024 / 1024 / 1024,
            "disk_total_gb": disk.total / 1024 / 1024 / 1024
        }

        # Determine health status
        issues = []
        status = HealthStatus.HEALTHY

        if cpu_percent > self.cpu_threshold:
            issues.append(f"High CPU usage: {cpu_percent:.1f}%")
            status = HealthStatus.DEGRADED

        if memory.percent > self.memory_threshold:
            issues.append(f"High memory usage: {memory.percent:.1f}%")
            status = HealthStatus.DEGRADED

        if disk.percent > self.disk_threshold:
            issues.append(f"High disk usage: {disk.percent:.1f}%")
            status = HealthStatus.UNHEALTHY

        if issues:
            details["issues"] = issues

        return HealthCheckResult(
            service_name=self.name,
            status=status,
            response_time=0.0,
            timestamp=datetime.now(),
            details=details
        )


class CircuitBreaker:
    """Circuit breaker implementation for fault tolerance"""

    def __init__(self, name: str, failure_threshold: int = 5,
                 recovery_timeout: float = 60.0, success_threshold: int = 3):
        self.name = name
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.success_threshold = success_threshold

        self.state = CircuitBreakerState.CLOSED
        self.failure_count = 0
        self.success_count = 0
        self.last_failure_time: Optional[float] = None

    async def call(self, func: Callable[[], Awaitable[Any]], *args, **kwargs) -> Any:
        """Execute function through circuit breaker"""
        if self.state == CircuitBreakerState.OPEN:
            if self._should_attempt_reset():
                self.state = CircuitBreakerState.HALF_OPEN
            else:
                raise CircuitBreakerOpenException(f"Circuit breaker {self.name} is OPEN")

        try:
            result = await func(*args, **kwargs)

            if self.state == CircuitBreakerState.HALF_OPEN:
                self.success_count += 1
                if self.success_count >= self.success_threshold:
                    self._reset()
            else:
                self._reset()

            return result

        except Exception as e:
            self._record_failure()
            raise e

    def _record_failure(self):
        """Record a failure and potentially open the circuit"""
        self.failure_count += 1
        self.last_failure_time = time.time()

        if self.failure_count >= self.failure_threshold:
            self.state = CircuitBreakerState.OPEN
            logger.warning(f"Circuit breaker {self.name} opened after {self.failure_count} failures")

    def _should_attempt_reset(self) -> bool:
        """Check if we should attempt to reset the circuit"""
        if self.last_failure_time is None:
            return True

        return (time.time() - self.last_failure_time) >= self.recovery_timeout

    def _reset(self):
        """Reset the circuit breaker to closed state"""
        self.state = CircuitBreakerState.CLOSED
        self.failure_count = 0
        self.success_count = 0
        logger.info(f"Circuit breaker {self.name} reset to CLOSED")


class CircuitBreakerOpenException(Exception):
    """Exception raised when circuit breaker is open"""
    pass


class HealthMonitor:
    """Central health monitoring and circuit breaker management"""

    def __init__(self):
        self.health_checks: Dict[str, HealthCheck] = {}
        self.circuit_breakers: Dict[str, CircuitBreaker] = {}
        self.monitoring_active = False
        self.monitor_thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()

    def add_health_check(self, health_check: HealthCheck):
        """Add a health check to monitor"""
        with self._lock:
            self.health_checks[health_check.name] = health_check

    def add_circuit_breaker(self, circuit_breaker: CircuitBreaker):
        """Add a circuit breaker"""
        with self._lock:
            self.circuit_breakers[circuit_breaker.name] = circuit_breaker

    def start_monitoring(self):
        """Start background health monitoring"""
        if self.monitoring_active:
            return

        self.monitoring_active = True
        self.monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.monitor_thread.start()
        logger.info("🏥 Health monitoring started")

    def stop_monitoring(self):
        """Stop background monitoring"""
        self.monitoring_active = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=5)
        logger.info("🏥 Health monitoring stopped")

    def _monitor_loop(self):
        """Background monitoring loop"""
        while self.monitoring_active:
            try:
                # Run all health checks
                tasks = []
                for health_check in self.health_checks.values():
                    # Only check if enough time has passed
                    if (health_check.last_check is None or
                        (datetime.now() - health_check.last_check).total_seconds() >= health_check.interval):
                        tasks.append(self._run_health_check(health_check))

                # Run tasks concurrently
                if tasks:
                    asyncio.run(asyncio.gather(*tasks, return_exceptions=True))

            except Exception as e:
                logger.error(f"Health monitoring error: {e}")

            time.sleep(10)  # Check every 10 seconds

    async def _run_health_check(self, health_check: HealthCheck):
        """Run a single health check"""
        try:
            result = await health_check.execute()

            # Log based on status
            if result.status == HealthStatus.UNHEALTHY:
                logger.error(f"❌ Health check failed: {result.service_name} - {result.error_message}")
            elif result.status == HealthStatus.DEGRADED:
                logger.warning(f"⚠️ Health check degraded: {result.service_name}")
            elif result.status == HealthStatus.HEALTHY and result.response_time > 1.0:
                logger.info(f"✅ Health check passed: {result.service_name} ({result.response_time:.3f}s)")

        except Exception as e:
            logger.error(f"Health check execution error for {health_check.name}: {e}")

    def get_health_status(self) -> Dict[str, Any]:
        """Get overall health status"""
        with self._lock:
            health_status = {}

            # Individual service health
            for name, health_check in self.health_checks.items():
                if health_check.last_result:
                    health_status[name] = {
                        'status': health_check.last_result.status.value,
                        'response_time': health_check.last_result.response_time,
                        'last_check': health_check.last_result.timestamp.isoformat(),
                        'health_score': health_check.get_health_score(),
                        'consecutive_failures': health_check.consecutive_failures
                    }
                else:
                    health_status[name] = {
                        'status': 'unknown',
                        'health_score': 0.0
                    }

            # Circuit breaker status
            circuit_status = {}
            for name, circuit_breaker in self.circuit_breakers.items():
                circuit_status[name] = {
                    'state': circuit_breaker.state.value,
                    'failure_count': circuit_breaker.failure_count,
                    'last_failure_time': circuit_breaker.last_failure_time
                }

            # Overall system health
            healthy_services = sum(1 for s in health_status.values() if s['status'] == 'healthy')
            total_services = len(health_status)

            overall_status = "healthy" if healthy_services == total_services else \
                           "degraded" if healthy_services >= total_services * 0.7 else "unhealthy"

            return {
                'timestamp': datetime.now().isoformat(),
                'overall_status': overall_status,
                'services_healthy': healthy_services,
                'services_total': total_services,
                'services': health_status,
                'circuit_breakers': circuit_status
            }

    async def call_with_circuit_breaker(self, circuit_breaker_name: str,
                                      func: Callable[[], Awaitable[Any]],
                                      *args, **kwargs) -> Any:
        """Execute function with circuit breaker protection"""
        circuit_breaker = self.circuit_breakers.get(circuit_breaker_name)
        if circuit_breaker:
            return await circuit_breaker.call(func, *args, **kwargs)
        else:
            # No circuit breaker, call directly
            return await func(*args, **kwargs)


# Global health monitor instance
health_monitor = HealthMonitor()


def create_default_health_checks() -> List[HealthCheck]:
    """Create default health checks for Supreme System V5"""
    checks = []

    # Application health checks
    checks.append(HTTPHealthCheck(
        name="api_health",
        url="http://localhost:8000/api/v1/health",
        interval=30.0
    ))

    checks.append(HTTPHealthCheck(
        name="api_status",
        url="http://localhost:8000/api/v1/status",
        interval=60.0
    ))

    checks.append(HTTPHealthCheck(
        name="metrics_endpoint",
        url="http://localhost:8000/metrics",
        interval=30.0
    ))

    # Infrastructure health checks
    checks.append(RedisHealthCheck(
        name="redis_cache",
        host="localhost",
        port=6379,
        interval=30.0
    ))

    checks.append(SystemHealthCheck(
        name="system_resources",
        cpu_threshold=85.0,
        memory_threshold=90.0,
        disk_threshold=95.0,
        interval=60.0
    ))

    # External service checks (if applicable)
    checks.append(HTTPHealthCheck(
        name="prometheus_health",
        url="http://localhost:9090/-/healthy",
        interval=60.0
    ))

    return checks


def create_default_circuit_breakers() -> List[CircuitBreaker]:
    """Create default circuit breakers"""
    breakers = []

    # API circuit breakers
    breakers.append(CircuitBreaker(
        name="trading_api",
        failure_threshold=5,
        recovery_timeout=60.0,
        success_threshold=3
    ))

    breakers.append(CircuitBreaker(
        name="market_data",
        failure_threshold=3,
        recovery_timeout=30.0,
        success_threshold=2
    ))

    breakers.append(CircuitBreaker(
        name="redis_cache",
        failure_threshold=5,
        recovery_timeout=120.0,
        success_threshold=3
    ))

    return breakers


def initialize_health_monitoring():
    """Initialize the global health monitoring system"""
    # Add default health checks
    for check in create_default_health_checks():
        health_monitor.add_health_check(check)

    # Add default circuit breakers
    for breaker in create_default_circuit_breakers():
        health_monitor.add_circuit_breaker(breaker)

    # Start monitoring
    health_monitor.start_monitoring()

    logger.info("🏥 Health monitoring system initialized")


if __name__ == "__main__":
    # Quick health check test
    async def test_health():
        monitor = HealthMonitor()

        # Add a simple health check
        check = HTTPHealthCheck("test_api", "http://httpbin.org/status/200")
        monitor.add_health_check(check)

        # Run check
        result = await check.execute()
        print(f"Health check result: {result.status.value}")
        print(f"Response time: {result.response_time:.3f}s")

    asyncio.run(test_health())
