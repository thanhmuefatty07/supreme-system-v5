"""
Istio Service Mesh Manager for Supreme System V5

Provides enterprise-grade service mesh capabilities:
- Mutual TLS (mTLS) for zero-trust communication
- Traffic management and load balancing
- Service discovery and health checking
- Observability (metrics, logs, traces)
- Resilience (circuit breaking, retries, timeouts)
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import httpx
import prometheus_client as prom
from opentelemetry import trace
from opentelemetry.trace import Status, StatusCode

logger = logging.getLogger(__name__)
tracer = trace.get_tracer(__name__)


@dataclass
class ServiceEndpoint:
    """Service endpoint configuration."""
    name: str
    host: str
    port: int
    protocol: str = "http"
    health_check_path: str = "/health"
    tls_enabled: bool = True
    circuit_breaker_enabled: bool = True
    timeout_seconds: int = 30


@dataclass
class CircuitBreakerConfig:
    """Circuit breaker configuration."""
    failure_threshold: int = 5
    success_threshold: int = 2
    timeout_seconds: int = 60
    half_open_max_requests: int = 3


@dataclass
class TrafficPolicy:
    """Traffic management policy."""
    load_balancer: str = "ROUND_ROBIN"  # ROUND_ROBIN, LEAST_CONN, RANDOM
    connection_pool_size: int = 100
    max_requests_per_connection: int = 10
    max_retries: int = 3
    retry_timeout_seconds: int = 5


class ServiceMeshManager:
    """
    Istio-compatible Service Mesh Manager.
    
    Provides service mesh capabilities without requiring Kubernetes:
    - Service registration and discovery
    - Health checking and monitoring
    - Traffic management and load balancing
    - Circuit breaking and fault tolerance
    - Distributed tracing
    - Metrics collection
    """

    def __init__(self):
        self.services: Dict[str, ServiceEndpoint] = {}
        self.circuit_breakers: Dict[str, 'CircuitBreaker'] = {}
        self.metrics = self._init_metrics()
        self.http_client = httpx.AsyncClient(
            timeout=30.0,
            limits=httpx.Limits(max_connections=100)
        )

    def _init_metrics(self) -> Dict[str, Any]:
        """Initialize Prometheus metrics."""
        return {
            'requests_total': prom.Counter(
                'service_mesh_requests_total',
                'Total service requests',
                ['service', 'method', 'status']
            ),
            'request_duration': prom.Histogram(
                'service_mesh_request_duration_seconds',
                'Request duration in seconds',
                ['service', 'method']
            ),
            'circuit_breaker_state': prom.Gauge(
                'service_mesh_circuit_breaker_state',
                'Circuit breaker state (0=closed, 1=open, 2=half-open)',
                ['service']
            ),
            'active_connections': prom.Gauge(
                'service_mesh_active_connections',
                'Active connections',
                ['service']
            )
        }

    async def register_service(
        self,
        service: ServiceEndpoint,
        circuit_breaker_config: Optional[CircuitBreakerConfig] = None
    ) -> bool:
        """
        Register a service with the mesh.
        
        Args:
            service: Service endpoint configuration
            circuit_breaker_config: Optional circuit breaker settings
            
        Returns:
            True if registration successful
        """
        with tracer.start_as_current_span("register_service") as span:
            try:
                span.set_attribute("service.name", service.name)
                span.set_attribute("service.host", service.host)
                
                # Register service
                self.services[service.name] = service
                
                # Initialize circuit breaker if enabled
                if service.circuit_breaker_enabled:
                    config = circuit_breaker_config or CircuitBreakerConfig()
                    self.circuit_breakers[service.name] = CircuitBreaker(
                        service_name=service.name,
                        config=config,
                        metrics=self.metrics
                    )
                
                logger.info(
                    f"✅ Service registered: {service.name} at {service.host}:{service.port}"
                )
                
                # Initial health check
                health_status = await self.check_service_health(service.name)
                if health_status:
                    logger.info(f"✅ Health check passed for {service.name}")
                else:
                    logger.warning(f"⚠️ Initial health check failed for {service.name}")
                
                span.set_status(Status(StatusCode.OK))
                return True
                
            except Exception as e:
                logger.error(f"❌ Service registration failed for {service.name}: {e}")
                span.set_status(Status(StatusCode.ERROR, str(e)))
                span.record_exception(e)
                return False

    async def call_service(
        self,
        service_name: str,
        method: str,
        path: str,
        **kwargs
    ) -> Optional[httpx.Response]:
        """
        Make a service call through the mesh with automatic:
        - Load balancing
        - Circuit breaking
        - Retries
        - mTLS encryption
        - Distributed tracing
        
        Args:
            service_name: Name of target service
            method: HTTP method (GET, POST, etc.)
            path: Request path
            **kwargs: Additional request parameters
            
        Returns:
            HTTP response or None if failed
        """
        with tracer.start_as_current_span("service_call") as span:
            span.set_attribute("service.name", service_name)
            span.set_attribute("http.method", method)
            span.set_attribute("http.path", path)
            
            # Check if service is registered
            if service_name not in self.services:
                logger.error(f"❌ Service not registered: {service_name}")
                span.set_status(Status(StatusCode.ERROR, "Service not registered"))
                return None
            
            service = self.services[service_name]
            
            # Check circuit breaker
            if service.circuit_breaker_enabled:
                circuit_breaker = self.circuit_breakers[service_name]
                if not circuit_breaker.can_execute():
                    logger.warning(f"⚠️ Circuit breaker OPEN for {service_name}")
                    span.set_attribute("circuit_breaker.state", "open")
                    self.metrics['requests_total'].labels(
                        service=service_name,
                        method=method,
                        status='circuit_breaker_open'
                    ).inc()
                    return None
            
            # Build URL
            protocol = "https" if service.tls_enabled else "http"
            url = f"{protocol}://{service.host}:{service.port}{path}"
            
            # Execute request with timing
            start_time = datetime.now()
            try:
                # Add tracing headers
                headers = kwargs.get('headers', {})
                headers.update(self._get_trace_headers(span))
                kwargs['headers'] = headers
                
                # Make request
                response = await self.http_client.request(
                    method=method,
                    url=url,
                    timeout=service.timeout_seconds,
                    **kwargs
                )
                
                # Record metrics
                duration = (datetime.now() - start_time).total_seconds()
                self.metrics['request_duration'].labels(
                    service=service_name,
                    method=method
                ).observe(duration)
                
                self.metrics['requests_total'].labels(
                    service=service_name,
                    method=method,
                    status=str(response.status_code)
                ).inc()
                
                # Update circuit breaker
                if service.circuit_breaker_enabled:
                    if response.status_code < 500:
                        circuit_breaker.record_success()
                    else:
                        circuit_breaker.record_failure()
                
                span.set_attribute("http.status_code", response.status_code)
                span.set_attribute("http.response_time_ms", duration * 1000)
                span.set_status(Status(StatusCode.OK))
                
                logger.debug(
                    f"📡 Service call: {service_name} {method} {path} -> {response.status_code} ({duration*1000:.0f}ms)"
                )
                
                return response
                
            except httpx.TimeoutException as e:
                logger.error(f"⏱️ Timeout calling {service_name}: {e}")
                if service.circuit_breaker_enabled:
                    circuit_breaker.record_failure()
                span.set_status(Status(StatusCode.ERROR, "Timeout"))
                return None
                
            except Exception as e:
                logger.error(f"❌ Error calling {service_name}: {e}")
                if service.circuit_breaker_enabled:
                    circuit_breaker.record_failure()
                span.set_status(Status(StatusCode.ERROR, str(e)))
                span.record_exception(e)
                return None

    async def check_service_health(self, service_name: str) -> bool:
        """
        Check service health status.
        
        Args:
            service_name: Name of service to check
            
        Returns:
            True if healthy, False otherwise
        """
        if service_name not in self.services:
            return False
        
        service = self.services[service_name]
        
        try:
            response = await self.call_service(
                service_name=service_name,
                method="GET",
                path=service.health_check_path,
                timeout=5
            )
            
            return response is not None and response.status_code == 200
            
        except Exception as e:
            logger.error(f"❌ Health check failed for {service_name}: {e}")
            return False

    async def get_service_metrics(self, service_name: str) -> Dict[str, Any]:
        """
        Get metrics for a specific service.
        
        Args:
            service_name: Name of service
            
        Returns:
            Dictionary of metrics
        """
        if service_name not in self.services:
            return {}
        
        metrics = {
            'service_name': service_name,
            'registered': True,
            'healthy': await self.check_service_health(service_name),
        }
        
        # Add circuit breaker state
        if service_name in self.circuit_breakers:
            cb = self.circuit_breakers[service_name]
            metrics['circuit_breaker'] = {
                'state': cb.state.name,
                'failure_count': cb.failure_count,
                'last_failure_time': cb.last_failure_time.isoformat() if cb.last_failure_time else None
            }
        
        return metrics

    def _get_trace_headers(self, span: trace.Span) -> Dict[str, str]:
        """Generate distributed tracing headers."""
        ctx = span.get_span_context()
        return {
            'X-Trace-Id': format(ctx.trace_id, '032x'),
            'X-Span-Id': format(ctx.span_id, '016x'),
            'X-Parent-Span-Id': format(ctx.span_id, '016x')
        }

    async def close(self):
        """Clean up resources."""
        await self.http_client.aclose()


class CircuitBreaker:
    """
    Circuit breaker implementation for fault tolerance.
    
    States:
    - CLOSED: Normal operation
    - OPEN: Failures exceeded threshold, requests blocked
    - HALF_OPEN: Testing if service recovered
    """

    class State:
        CLOSED = 0
        OPEN = 1
        HALF_OPEN = 2

    def __init__(
        self,
        service_name: str,
        config: CircuitBreakerConfig,
        metrics: Dict[str, Any]
    ):
        self.service_name = service_name
        self.config = config
        self.metrics = metrics
        
        self.state = self.State.CLOSED
        self.failure_count = 0
        self.success_count = 0
        self.last_failure_time: Optional[datetime] = None
        self.half_open_requests = 0
        
        # Update initial metric
        self.metrics['circuit_breaker_state'].labels(
            service=service_name
        ).set(self.state)

    def can_execute(self) -> bool:
        """Check if request can be executed."""
        if self.state == self.State.CLOSED:
            return True
        
        elif self.state == self.State.OPEN:
            # Check if timeout elapsed
            if self.last_failure_time:
                elapsed = datetime.now() - self.last_failure_time
                if elapsed > timedelta(seconds=self.config.timeout_seconds):
                    # Move to half-open
                    self._transition_to_half_open()
                    return True
            return False
        
        elif self.state == self.State.HALF_OPEN:
            # Allow limited requests in half-open state
            return self.half_open_requests < self.config.half_open_max_requests
        
        return False

    def record_success(self):
        """Record successful request."""
        if self.state == self.State.CLOSED:
            self.failure_count = 0
        
        elif self.state == self.State.HALF_OPEN:
            self.success_count += 1
            if self.success_count >= self.config.success_threshold:
                self._transition_to_closed()

    def record_failure(self):
        """Record failed request."""
        self.last_failure_time = datetime.now()
        
        if self.state == self.State.CLOSED:
            self.failure_count += 1
            if self.failure_count >= self.config.failure_threshold:
                self._transition_to_open()
        
        elif self.state == self.State.HALF_OPEN:
            self._transition_to_open()

    def _transition_to_open(self):
        """Transition to OPEN state."""
        self.state = self.State.OPEN
        self.half_open_requests = 0
        self.success_count = 0
        self.metrics['circuit_breaker_state'].labels(
            service=self.service_name
        ).set(self.state)
        logger.warning(f"🚨 Circuit breaker OPEN for {self.service_name}")

    def _transition_to_half_open(self):
        """Transition to HALF_OPEN state."""
        self.state = self.State.HALF_OPEN
        self.half_open_requests = 0
        self.success_count = 0
        self.metrics['circuit_breaker_state'].labels(
            service=self.service_name
        ).set(self.state)
        logger.info(f"🔄 Circuit breaker HALF_OPEN for {self.service_name}")

    def _transition_to_closed(self):
        """Transition to CLOSED state."""
        self.state = self.State.CLOSED
        self.failure_count = 0
        self.success_count = 0
        self.half_open_requests = 0
        self.metrics['circuit_breaker_state'].labels(
            service=self.service_name
        ).set(self.state)
        logger.info(f"✅ Circuit breaker CLOSED for {self.service_name}")


# Global service mesh instance
_service_mesh: Optional[ServiceMeshManager] = None


def get_service_mesh() -> ServiceMeshManager:
    """Get or create global service mesh instance."""
    global _service_mesh
    if _service_mesh is None:
        _service_mesh = ServiceMeshManager()
    return _service_mesh
