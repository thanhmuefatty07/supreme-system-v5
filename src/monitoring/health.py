"""
🏥 SUPREME SYSTEM V5 - HEALTH CHECK SYSTEM

Advanced health monitoring and system status verification for production deployment.
Performs comprehensive checks across all system components with real-time status.

Author: Supreme Team
Date: 2025-10-25
Version: 5.0 Production
"""

import asyncio
import time
import psutil
import logging
from typing import Dict, List, Optional, Any, Union
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from enum import Enum
import json
import aioredis
from contextlib import asynccontextmanager

logger = logging.getLogger(__name__)


class HealthStatus(Enum):
    """Health status enumeration"""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    UNKNOWN = "unknown"


@dataclass
class ComponentHealth:
    """Individual component health status"""
    name: str
    status: HealthStatus
    response_time_ms: float
    last_check: datetime
    message: str
    metadata: Dict[str, Any] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        return {
            'name': self.name,
            'status': self.status.value,
            'response_time_ms': self.response_time_ms,
            'last_check': self.last_check.isoformat(),
            'message': self.message,
            'metadata': self.metadata or {}
        }


@dataclass
class SystemHealth:
    """Overall system health aggregation"""
    overall_status: HealthStatus
    components: List[ComponentHealth]
    timestamp: datetime
    uptime_seconds: float
    system_load: Dict[str, float]
    memory_usage: Dict[str, float]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        return {
            'overall_status': self.overall_status.value,
            'components': [comp.to_dict() for comp in self.components],
            'timestamp': self.timestamp.isoformat(),
            'uptime_seconds': self.uptime_seconds,
            'system_load': self.system_load,
            'memory_usage': self.memory_usage
        }


class HealthChecker:
    """
    🏥 Production-grade health checking system
    
    Features:
    - Multi-component health verification
    - Real-time system resource monitoring  
    - Intelligent status aggregation
    - Performance threshold management
    - Automated recovery detection
    """
    
    def __init__(self, redis_url: str = "redis://localhost:6379/0"):
        self.redis_url = redis_url
        self.start_time = time.time()
        self.redis_pool: Optional[aioredis.ConnectionPool] = None
        
        # Health thresholds (configurable)
        self.thresholds = {
            'response_time_ms': 1000,  # 1 second max response
            'cpu_percent': 80.0,       # 80% CPU usage warning
            'memory_percent': 85.0,    # 85% memory usage warning
            'disk_percent': 90.0,      # 90% disk usage critical
        }
        
        # Component checkers registry
        self.checkers = {
            'neuromorphic': self._check_neuromorphic_engine,
            'ultra_latency': self._check_ultra_latency_engine,
            'foundation_models': self._check_foundation_models,
            'mamba_ssm': self._check_mamba_engine,
            'trading_engine': self._check_trading_engine,
            'api_server': self._check_api_server,
            'redis_cache': self._check_redis_connection,
            'system_resources': self._check_system_resources
        }
        
    async def __aenter__(self):
        """Async context manager entry"""
        await self.initialize()
        return self
        
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        await self.cleanup()
        
    async def initialize(self):
        """Initialize health checker resources"""
        try:
            # Initialize Redis connection pool
            self.redis_pool = aioredis.ConnectionPool.from_url(
                self.redis_url,
                max_connections=10,
                retry_on_timeout=True
            )
            logger.info("Health checker initialized successfully")
        except Exception as e:
            logger.warning(f"Redis connection failed during init: {e}")
            self.redis_pool = None
    
    async def cleanup(self):
        """Cleanup resources"""
        if self.redis_pool:
            await self.redis_pool.disconnect()
            logger.info("Health checker cleaned up")
    
    async def check_system_health(self) -> SystemHealth:
        """
        🔍 Perform comprehensive system health check
        
        Returns:
            SystemHealth: Complete system status with all components
        """
        start_time = time.time()
        components = []
        
        # Check all registered components
        for component_name, checker in self.checkers.items():
            try:
                health = await checker()
                components.append(health)
                logger.debug(f"Health check completed for {component_name}: {health.status.value}")
            except Exception as e:
                logger.error(f"Health check failed for {component_name}: {e}")
                components.append(ComponentHealth(
                    name=component_name,
                    status=HealthStatus.UNHEALTHY,
                    response_time_ms=0.0,
                    last_check=datetime.utcnow(),
                    message=f"Check failed: {str(e)}"
                ))
        
        # Calculate overall status
        overall_status = self._calculate_overall_status(components)
        
        # Get system metrics
        system_load = self._get_system_load()
        memory_usage = self._get_memory_usage()
        uptime = time.time() - self.start_time
        
        total_check_time = (time.time() - start_time) * 1000
        logger.info(f"System health check completed in {total_check_time:.2f}ms - Status: {overall_status.value}")
        
        return SystemHealth(
            overall_status=overall_status,
            components=components,
            timestamp=datetime.utcnow(),
            uptime_seconds=uptime,
            system_load=system_load,
            memory_usage=memory_usage
        )
    
    async def _check_neuromorphic_engine(self) -> ComponentHealth:
        """Check neuromorphic computing engine health"""
        start_time = time.time()
        
        try:
            from ..neuromorphic import NeuromorphicEngine
            
            # Perform lightweight health check
            engine = NeuromorphicEngine()
            # Simple validation without heavy computation
            is_ready = hasattr(engine, 'process_spike_train')
            
            response_time = (time.time() - start_time) * 1000
            
            if is_ready and response_time < self.thresholds['response_time_ms']:
                status = HealthStatus.HEALTHY
                message = "Neuromorphic engine operational"
            else:
                status = HealthStatus.DEGRADED
                message = f"High response time: {response_time:.2f}ms"
                
        except ImportError:
            status = HealthStatus.UNHEALTHY
            message = "Neuromorphic engine module not available"
            response_time = 0.0
        except Exception as e:
            status = HealthStatus.UNHEALTHY
            message = f"Engine error: {str(e)}"
            response_time = (time.time() - start_time) * 1000
        
        return ComponentHealth(
            name="neuromorphic_engine",
            status=status,
            response_time_ms=response_time,
            last_check=datetime.utcnow(),
            message=message
        )
    
    async def _check_ultra_latency_engine(self) -> ComponentHealth:
        """Check ultra-low latency engine health"""
        start_time = time.time()
        
        try:
            from ..ultra_low_latency import UltraLowLatencyEngine
            
            engine = UltraLowLatencyEngine()
            # Quick capability check
            is_ready = hasattr(engine, 'process_market_data')
            
            response_time = (time.time() - start_time) * 1000
            
            # Ultra-low latency should be sub-millisecond
            if is_ready and response_time < 1.0:  # <1ms
                status = HealthStatus.HEALTHY
                message = f"Ultra-latency engine optimal: {response_time:.3f}ms"
            elif is_ready:
                status = HealthStatus.DEGRADED
                message = f"Latency degraded: {response_time:.3f}ms"
            else:
                status = HealthStatus.UNHEALTHY
                message = "Engine not ready"
                
        except Exception as e:
            status = HealthStatus.UNHEALTHY
            message = f"Ultra-latency engine error: {str(e)}"
            response_time = (time.time() - start_time) * 1000
        
        return ComponentHealth(
            name="ultra_latency_engine",
            status=status,
            response_time_ms=response_time,
            last_check=datetime.utcnow(),
            message=message
        )
    
    async def _check_foundation_models(self) -> ComponentHealth:
        """Check foundation models health"""
        start_time = time.time()
        
        try:
            from ..foundation_models import FoundationModelEngine
            
            engine = FoundationModelEngine()
            is_ready = hasattr(engine, 'zero_shot_forecast')
            
            response_time = (time.time() - start_time) * 1000
            
            if is_ready and response_time < self.thresholds['response_time_ms']:
                status = HealthStatus.HEALTHY
                message = "Foundation models ready"
            else:
                status = HealthStatus.DEGRADED
                message = f"Models loading: {response_time:.2f}ms"
                
        except Exception as e:
            status = HealthStatus.UNHEALTHY
            message = f"Foundation models error: {str(e)}"
            response_time = (time.time() - start_time) * 1000
        
        return ComponentHealth(
            name="foundation_models",
            status=status,
            response_time_ms=response_time,
            last_check=datetime.utcnow(),
            message=message
        )
    
    async def _check_mamba_engine(self) -> ComponentHealth:
        """Check Mamba SSM engine health"""
        start_time = time.time()
        
        try:
            from ..mamba_ssm import MambaEngine
            
            engine = MambaEngine()
            is_ready = hasattr(engine, 'process_sequence')
            
            response_time = (time.time() - start_time) * 1000
            
            if is_ready and response_time < self.thresholds['response_time_ms']:
                status = HealthStatus.HEALTHY
                message = "Mamba SSM engine operational"
            else:
                status = HealthStatus.DEGRADED
                message = f"Mamba engine slow: {response_time:.2f}ms"
                
        except Exception as e:
            status = HealthStatus.UNHEALTHY
            message = f"Mamba engine error: {str(e)}"
            response_time = (time.time() - start_time) * 1000
        
        return ComponentHealth(
            name="mamba_engine",
            status=status,
            response_time_ms=response_time,
            last_check=datetime.utcnow(),
            message=message
        )
    
    async def _check_trading_engine(self) -> ComponentHealth:
        """Check trading engine health"""
        start_time = time.time()
        
        try:
            from ..trading import TradingEngine
            
            engine = TradingEngine()
            is_ready = hasattr(engine, 'execute_trade')
            
            response_time = (time.time() - start_time) * 1000
            
            if is_ready and response_time < self.thresholds['response_time_ms']:
                status = HealthStatus.HEALTHY
                message = "Trading engine ready"
            else:
                status = HealthStatus.DEGRADED
                message = f"Trading engine slow: {response_time:.2f}ms"
                
        except Exception as e:
            status = HealthStatus.UNHEALTHY
            message = f"Trading engine error: {str(e)}"
            response_time = (time.time() - start_time) * 1000
        
        return ComponentHealth(
            name="trading_engine",
            status=status,
            response_time_ms=response_time,
            last_check=datetime.utcnow(),
            message=message
        )
    
    async def _check_api_server(self) -> ComponentHealth:
        """Check API server health"""
        start_time = time.time()
        
        try:
            # Check if FastAPI components are importable
            from ..api import server
            is_ready = hasattr(server, 'app')
            
            response_time = (time.time() - start_time) * 1000
            
            if is_ready and response_time < 100:  # API should be very fast
                status = HealthStatus.HEALTHY
                message = "API server ready"
            else:
                status = HealthStatus.DEGRADED
                message = f"API server slow: {response_time:.2f}ms"
                
        except Exception as e:
            status = HealthStatus.UNHEALTHY
            message = f"API server error: {str(e)}"
            response_time = (time.time() - start_time) * 1000
        
        return ComponentHealth(
            name="api_server",
            status=status,
            response_time_ms=response_time,
            last_check=datetime.utcnow(),
            message=message
        )
    
    async def _check_redis_connection(self) -> ComponentHealth:
        """Check Redis connection health"""
        start_time = time.time()
        
        try:
            if not self.redis_pool:
                raise Exception("Redis pool not initialized")
                
            redis = aioredis.Redis(connection_pool=self.redis_pool)
            
            # Test Redis with ping
            await asyncio.wait_for(redis.ping(), timeout=2.0)
            
            response_time = (time.time() - start_time) * 1000
            
            if response_time < 50:  # Redis should be very fast
                status = HealthStatus.HEALTHY
                message = f"Redis operational: {response_time:.2f}ms"
            else:
                status = HealthStatus.DEGRADED
                message = f"Redis slow: {response_time:.2f}ms"
                
        except asyncio.TimeoutError:
            status = HealthStatus.UNHEALTHY
            message = "Redis timeout (>2s)"
            response_time = 2000.0
        except Exception as e:
            status = HealthStatus.UNHEALTHY
            message = f"Redis error: {str(e)}"
            response_time = (time.time() - start_time) * 1000
        
        return ComponentHealth(
            name="redis_cache",
            status=status,
            response_time_ms=response_time,
            last_check=datetime.utcnow(),
            message=message
        )
    
    async def _check_system_resources(self) -> ComponentHealth:
        """Check system resources (CPU, Memory, Disk)"""
        start_time = time.time()
        
        try:
            # Get system metrics
            cpu_percent = psutil.cpu_percent(interval=0.1)
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage('/')
            
            response_time = (time.time() - start_time) * 1000
            
            # Determine status based on thresholds
            issues = []
            if cpu_percent > self.thresholds['cpu_percent']:
                issues.append(f"CPU: {cpu_percent:.1f}%")
            if memory.percent > self.thresholds['memory_percent']:
                issues.append(f"Memory: {memory.percent:.1f}%")
            if disk.percent > self.thresholds['disk_percent']:
                issues.append(f"Disk: {disk.percent:.1f}%")
            
            if not issues:
                status = HealthStatus.HEALTHY
                message = f"Resources optimal - CPU: {cpu_percent:.1f}%, Memory: {memory.percent:.1f}%"
            elif len(issues) <= 1:
                status = HealthStatus.DEGRADED
                message = f"Resource warning: {', '.join(issues)}"
            else:
                status = HealthStatus.UNHEALTHY
                message = f"Resource critical: {', '.join(issues)}"
            
            metadata = {
                'cpu_percent': cpu_percent,
                'memory_percent': memory.percent,
                'disk_percent': disk.percent,
                'memory_available_gb': memory.available / (1024**3),
                'disk_free_gb': disk.free / (1024**3)
            }
                
        except Exception as e:
            status = HealthStatus.UNHEALTHY
            message = f"System resources check failed: {str(e)}"
            response_time = (time.time() - start_time) * 1000
            metadata = {}
        
        return ComponentHealth(
            name="system_resources",
            status=status,
            response_time_ms=response_time,
            last_check=datetime.utcnow(),
            message=message,
            metadata=metadata
        )
    
    def _calculate_overall_status(self, components: List[ComponentHealth]) -> HealthStatus:
        """
        Calculate overall system status based on component health
        
        Logic:
        - If any component is UNHEALTHY -> UNHEALTHY
        - If any component is DEGRADED -> DEGRADED  
        - If all components are HEALTHY -> HEALTHY
        - If mix of UNKNOWN -> UNKNOWN
        """
        if not components:
            return HealthStatus.UNKNOWN
        
        statuses = [comp.status for comp in components]
        
        if HealthStatus.UNHEALTHY in statuses:
            return HealthStatus.UNHEALTHY
        elif HealthStatus.DEGRADED in statuses:
            return HealthStatus.DEGRADED
        elif all(status == HealthStatus.HEALTHY for status in statuses):
            return HealthStatus.HEALTHY
        else:
            return HealthStatus.UNKNOWN
    
    def _get_system_load(self) -> Dict[str, float]:
        """Get system load averages"""
        try:
            load_avg = psutil.getloadavg()
            return {
                '1min': load_avg[0],
                '5min': load_avg[1],
                '15min': load_avg[2]
            }
        except (AttributeError, OSError):
            # Fallback for systems without getloadavg
            return {
                '1min': psutil.cpu_percent(interval=1),
                '5min': 0.0,
                '15min': 0.0
            }
    
    def _get_memory_usage(self) -> Dict[str, float]:
        """Get memory usage statistics"""
        try:
            memory = psutil.virtual_memory()
            return {
                'total_gb': memory.total / (1024**3),
                'used_gb': memory.used / (1024**3),
                'available_gb': memory.available / (1024**3),
                'percent': memory.percent
            }
        except Exception:
            return {
                'total_gb': 0.0,
                'used_gb': 0.0,
                'available_gb': 0.0,
                'percent': 0.0
            }


# Singleton health checker instance
_health_checker: Optional[HealthChecker] = None


async def get_health_checker() -> HealthChecker:
    """Get global health checker instance"""
    global _health_checker
    if _health_checker is None:
        _health_checker = HealthChecker()
        await _health_checker.initialize()
    return _health_checker


async def quick_health_check() -> Dict[str, Any]:
    """
    🚀 Quick health check for API endpoints
    
    Returns minimal health status for fast API responses
    """
    try:
        checker = await get_health_checker()
        health = await checker.check_system_health()
        
        return {
            'status': health.overall_status.value,
            'timestamp': health.timestamp.isoformat(),
            'uptime_seconds': health.uptime_seconds,
            'component_count': len(health.components),
            'healthy_components': len([c for c in health.components if c.status == HealthStatus.HEALTHY])
        }
    except Exception as e:
        logger.error(f"Quick health check failed: {e}")
        return {
            'status': 'unhealthy',
            'timestamp': datetime.utcnow().isoformat(),
            'error': str(e)
        }


if __name__ == "__main__":
    # Demo health check
    async def demo():
        async with HealthChecker() as checker:
            health = await checker.check_system_health()
            print("\n🏥 SUPREME SYSTEM V5 - HEALTH STATUS")
            print(f"Overall Status: {health.overall_status.value.upper()}")
            print(f"Uptime: {health.uptime_seconds:.1f}s")
            print(f"Timestamp: {health.timestamp}")
            
            print("\nComponent Health:")
            for comp in health.components:
                status_icon = {
                    HealthStatus.HEALTHY: "✅",
                    HealthStatus.DEGRADED: "⚠️",
                    HealthStatus.UNHEALTHY: "❌",
                    HealthStatus.UNKNOWN: "❓"
                }[comp.status]
                print(f"{status_icon} {comp.name}: {comp.message} ({comp.response_time_ms:.2f}ms)")
    
    asyncio.run(demo())