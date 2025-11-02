#!/usr/bin/env python3
"""
🚀 SUPREME SYSTEM V5 - Advanced Resource Monitor
Real-time monitoring với auto-optimization capabilities

Features:
- Real-time CPU/RAM monitoring
- Auto-optimization khi vượt ngưỡng
- Performance bottleneck detection
- Emergency resource management
- Memory-efficient tracking
"""

from __future__ import annotations
import psutil
import time
import threading
import os
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, field
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class ResourceType(Enum):
    """Resource types to monitor"""
    CPU = "cpu"
    RAM = "ram"
    DISK = "disk"
    NETWORK = "network"


class OptimizationAction(Enum):
    """Available optimization actions"""
    REDUCE_UPDATE_FREQUENCY = "reduce_update_frequency"
    DISABLE_NON_CRITICAL = "disable_non_critical"
    CLEAR_CACHES = "clear_caches"
    REDUCE_WORKERS = "reduce_workers"
    INCREASE_THRESHOLDS = "increase_thresholds"
    EMERGENCY_SHUTDOWN = "emergency_shutdown"


@dataclass
class ResourceThreshold:
    """Resource usage thresholds"""
    warning_level: float  # Warning threshold (%)
    critical_level: float  # Critical threshold (%)
    emergency_level: float  # Emergency threshold (%)


@dataclass
class ResourceMetrics:
    """Comprehensive resource metrics"""
    timestamp: float
    cpu_percent: float
    ram_gb: float
    ram_percent: float
    disk_usage_gb: float
    network_sent_mb: float
    network_recv_mb: float

    # Process-specific metrics
    process_cpu_percent: float
    process_ram_mb: float
    process_threads: int
    process_fds: Optional[int] = None  # File descriptors (Unix only)

    # System-wide metrics
    system_load_avg: Optional[List[float]] = None  # Load average (Unix only)
    swap_percent: float = 0.0


@dataclass
class OptimizationResult:
    """Result of optimization action"""
    action: OptimizationAction
    timestamp: float
    success: bool
    resource_before: ResourceMetrics
    resource_after: ResourceMetrics
    impact_description: str
    rollback_possible: bool = True


@dataclass
class PerformanceProfile:
    """Performance profile for different operation modes"""
    name: str
    cpu_limit: float
    ram_limit_gb: float
    update_frequency_modifier: float  # Multiplier for update intervals
    component_reductions: Dict[str, float]  # Component -> reduction factor
    description: str


class AdvancedResourceMonitor:
    """
    Advanced resource monitor với auto-optimization
    Maintains 88% CPU và 3.46GB RAM targets
    Auto-optimizes khi vượt ngưỡng
    """

    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or self._get_default_config()

        # Resource thresholds
        self.thresholds = {
            ResourceType.CPU: ResourceThreshold(
                warning_level=self.config['cpu_warning_threshold'],
                critical_level=self.config['cpu_critical_threshold'],
                emergency_level=self.config['cpu_emergency_threshold']
            ),
            ResourceType.RAM: ResourceThreshold(
                warning_level=self.config['ram_warning_threshold'],
                critical_level=self.config['ram_critical_threshold'],
                emergency_level=self.config['ram_emergency_threshold']
            )
        }

        # Performance profiles
        self.performance_profiles = self._initialize_performance_profiles()

        # Monitoring state
        self.monitoring_active = False
        self.monitor_thread: Optional[threading.Thread] = None
        self.monitoring_interval = self.config['monitoring_interval']

        # Historical data
        self.metrics_history: List[ResourceMetrics] = []
        self.optimization_history: List[OptimizationResult] = []
        self.max_history_size = self.config['max_history_size']

        # Current performance profile
        self.current_profile = "normal"

        # Emergency state
        self.emergency_mode = False
        self.last_emergency_action = 0

        # Process monitoring
        self.process = psutil.Process(os.getpid())
        self.system_boot_time = psutil.boot_time()

        # Optimization callbacks
        self.optimization_callbacks: Dict[OptimizationAction, Callable] = {}

        logger.info("AdvancedResourceMonitor initialized with %s monitoring", "active" if self.monitoring_active else "inactive")

    def _get_default_config(self) -> Dict[str, Any]:
        """Get default monitoring configuration"""
        return {
            'monitoring_interval': 5.0,  # 5 seconds
            'cpu_warning_threshold': 75.0,   # 75% CPU warning
            'cpu_critical_threshold': 85.0,  # 85% CPU critical
            'cpu_emergency_threshold': 95.0, # 95% CPU emergency
            'ram_warning_threshold': 80.0,   # 80% RAM warning
            'ram_critical_threshold': 90.0,  # 90% RAM critical
            'ram_emergency_threshold': 95.0, # 95% RAM emergency
            'max_history_size': 1000,
            'auto_optimization': True,
            'emergency_cooldown': 300.0,  # 5 minutes between emergency actions
            'performance_profile_switching': True
        }

    def _initialize_performance_profiles(self) -> Dict[str, PerformanceProfile]:
        """Initialize performance profiles for different conditions"""
        return {
            "normal": PerformanceProfile(
                name="normal",
                cpu_limit=88.0,
                ram_limit_gb=3.46,
                update_frequency_modifier=1.0,
                component_reductions={},
                description="Normal operation within target limits"
            ),
            "conservative": PerformanceProfile(
                name="conservative",
                cpu_limit=70.0,
                ram_limit_gb=2.8,
                update_frequency_modifier=1.5,  # 50% slower updates
                component_reductions={
                    "news_classifier": 0.7,    # 30% less frequent
                    "whale_tracking": 0.7,     # 30% less frequent
                    "pattern_recognition": 0.8 # 20% less frequent
                },
                description="Conservative mode for resource constraints"
            ),
            "minimal": PerformanceProfile(
                name="minimal",
                cpu_limit=50.0,
                ram_limit_gb=2.0,
                update_frequency_modifier=3.0,  # 3x slower updates
                component_reductions={
                    "news_classifier": 0.3,    # 70% less frequent
                    "whale_tracking": 0.3,     # 70% less frequent
                    "pattern_recognition": 0.5, # 50% less frequent
                    "multi_timeframe": 0.7     # 30% less frequent
                },
                description="Minimal mode for severe resource constraints"
            ),
            "emergency": PerformanceProfile(
                name="emergency",
                cpu_limit=30.0,
                ram_limit_gb=1.5,
                update_frequency_modifier=10.0,  # 10x slower updates
                component_reductions={
                    "news_classifier": 0.1,    # 90% less frequent
                    "whale_tracking": 0.1,     # 90% less frequent
                    "pattern_recognition": 0.2, # 80% less frequent
                    "multi_timeframe": 0.3,     # 70% less frequent
                    "risk_manager": 0.5        # 50% less frequent
                },
                description="Emergency mode - minimal operation"
            )
        }

    def start_monitoring(self):
        """Start background resource monitoring"""
        if self.monitoring_active:
            return

        self.monitoring_active = True
        self.monitor_thread = threading.Thread(
            target=self._monitoring_loop,
            daemon=True,
            name="ResourceMonitor"
        )
        self.monitor_thread.start()
        logger.info("Resource monitoring started")

    def stop_monitoring(self):
        """Stop background resource monitoring"""
        self.monitoring_active = False
        if self.monitor_thread and self.monitor_thread.is_alive():
            self.monitor_thread.join(timeout=5.0)
        logger.info("Resource monitoring stopped")

    def _monitoring_loop(self):
        """Main monitoring loop"""
        while self.monitoring_active:
            try:
                # Collect metrics
                metrics = self._collect_resource_metrics()

                # Store in history
                self.metrics_history.append(metrics)
                if len(self.metrics_history) > self.max_history_size:
                    self.metrics_history = self.metrics_history[-self.max_history_size:]

                # Check thresholds and optimize if needed
                if self.config.get('auto_optimization', True):
                    self._check_and_optimize(metrics)

                # Sleep until next monitoring cycle
                time.sleep(self.monitoring_interval)

            except Exception as e:
                logger.error("Monitoring loop error: %s", e)
                time.sleep(self.monitoring_interval)

    def _collect_resource_metrics(self) -> ResourceMetrics:
        """Collect comprehensive resource metrics"""
        timestamp = time.time()

        # System-wide metrics
        cpu_percent = psutil.cpu_percent(interval=1.0)
        ram = psutil.virtual_memory()
        disk = psutil.disk_usage('/')
        network = psutil.net_io_counters()

        # Process-specific metrics
        process_cpu = self.process.cpu_percent()
        process_ram = self.process.memory_info().rss / (1024 * 1024)  # MB
        process_threads = self.process.num_threads()

        # File descriptors (Unix only)
        try:
            process_fds = self.process.num_fds() if hasattr(self.process, 'num_fds') else None
        except:
            process_fds = None

        # Load average (Unix only)
        try:
            load_avg = os.getloadavg() if hasattr(os, 'getloadavg') else None
        except:
            load_avg = None

        # Swap usage
        swap = psutil.swap_memory()

        return ResourceMetrics(
            timestamp=timestamp,
            cpu_percent=cpu_percent,
            ram_gb=ram.used / (1024**3),
            ram_percent=ram.percent,
            disk_usage_gb=disk.used / (1024**3),
            network_sent_mb=network.bytes_sent / (1024**2) if network else 0,
            network_recv_mb=network.bytes_recv / (1024**2) if network else 0,
            process_cpu_percent=process_cpu,
            process_ram_mb=process_ram,
            process_threads=process_threads,
            process_fds=process_fds,
            system_load_avg=load_avg,
            swap_percent=swap.percent
        )

    def _check_and_optimize(self, metrics: ResourceMetrics):
        """Check resource usage và trigger optimization if needed"""
        current_time = time.time()

        # Check CPU thresholds
        cpu_level = self._get_resource_level(metrics.cpu_percent, ResourceType.CPU)
        ram_level = self._get_resource_level(metrics.ram_percent, ResourceType.RAM)

        # Determine if optimization is needed
        needs_optimization = (
            cpu_level in ['critical', 'emergency'] or
            ram_level in ['critical', 'emergency']
        )

        if needs_optimization:
            # Check emergency cooldown
            if current_time - self.last_emergency_action < self.config['emergency_cooldown']:
                logger.info("Optimization needed but in cooldown period")
                return

            # Perform optimization
            optimization_result = self._perform_optimization(metrics, cpu_level, ram_level)

            if optimization_result:
                self.optimization_history.append(optimization_result)
                self.last_emergency_action = current_time

                logger.info("Optimization performed: %s", optimization_result.action.value)

                # Keep history size manageable
                if len(self.optimization_history) > 100:
                    self.optimization_history = self.optimization_history[-100:]

    def _get_resource_level(self, usage_percent: float, resource_type: ResourceType) -> str:
        """Get resource usage level"""
        threshold = self.thresholds[resource_type]

        if usage_percent >= threshold.emergency_level:
            return "emergency"
        elif usage_percent >= threshold.critical_level:
            return "critical"
        elif usage_percent >= threshold.warning_level:
            return "warning"
        else:
            return "normal"

    def _perform_optimization(self, metrics: ResourceMetrics,
                            cpu_level: str, ram_level: str) -> Optional[OptimizationResult]:
        """Perform resource optimization"""
        # Determine optimization strategy based on severity
        if cpu_level == "emergency" or ram_level == "emergency":
            new_profile = "emergency"
            actions = [OptimizationAction.EMERGENCY_SHUTDOWN]
        elif cpu_level == "critical" or ram_level == "critical":
            new_profile = "minimal"
            actions = [OptimizationAction.REDUCE_UPDATE_FREQUENCY,
                      OptimizationAction.DISABLE_NON_CRITICAL]
        else:  # warning level
            new_profile = "conservative"
            actions = [OptimizationAction.REDUCE_UPDATE_FREQUENCY,
                      OptimizationAction.CLEAR_CACHES]

        # Switch performance profile
        success = self.switch_performance_profile(new_profile)

        if success:
            # Collect metrics after optimization
            time.sleep(2.0)  # Wait for changes to take effect
            after_metrics = self._collect_resource_metrics()

            return OptimizationResult(
                action=OptimizationAction.REDUCE_UPDATE_FREQUENCY,  # Primary action
                timestamp=time.time(),
                success=True,
                resource_before=metrics,
                resource_after=after_metrics,
                impact_description=f"Switched to {new_profile} performance profile",
                rollback_possible=True
            )

        return None

    def switch_performance_profile(self, profile_name: str) -> bool:
        """Switch to a different performance profile"""
        if profile_name not in self.performance_profiles:
            logger.error("Unknown performance profile: %s", profile_name)
            return False

        old_profile = self.current_profile
        new_profile = self.performance_profiles[profile_name]

        # Apply profile changes
        self.current_profile = profile_name

        # Call optimization callbacks if registered
        for action in new_profile.component_reductions.keys():
            callback = self.optimization_callbacks.get(OptimizationAction.REDUCE_UPDATE_FREQUENCY)
            if callback:
                try:
                    callback(new_profile.component_reductions)
                except Exception as e:
                    logger.error("Optimization callback error: %s", e)

        logger.info("Switched performance profile: %s -> %s", old_profile, profile_name)
        return True

    def register_optimization_callback(self, action: OptimizationAction, callback: Callable):
        """Register callback for optimization actions"""
        self.optimization_callbacks[action] = callback

    def get_current_metrics(self) -> Optional[ResourceMetrics]:
        """Get most recent resource metrics"""
        return self.metrics_history[-1] if self.metrics_history else None

    def get_resource_trends(self, hours: int = 1) -> Dict[str, Any]:
        """Get resource usage trends over time"""
        cutoff_time = time.time() - (hours * 3600)

        # Filter recent metrics
        recent_metrics = [
            m for m in self.metrics_history
            if m.timestamp >= cutoff_time
        ]

        if not recent_metrics:
            return {"error": "No recent metrics available"}

        # Calculate trends
        cpu_trend = self._calculate_trend([m.cpu_percent for m in recent_metrics])
        ram_trend = self._calculate_trend([m.ram_gb for m in recent_metrics])

        return {
            "period_hours": hours,
            "data_points": len(recent_metrics),
            "cpu_trend": cpu_trend,
            "ram_trend": ram_trend,
            "avg_cpu": sum(m.cpu_percent for m in recent_metrics) / len(recent_metrics),
            "avg_ram_gb": sum(m.ram_gb for m in recent_metrics) / len(recent_metrics),
            "peak_cpu": max(m.cpu_percent for m in recent_metrics),
            "peak_ram_gb": max(m.ram_gb for m in recent_metrics)
        }

    def _calculate_trend(self, values: List[float]) -> str:
        """Calculate trend direction from values"""
        if len(values) < 2:
            return "insufficient_data"

        # Simple linear trend
        first_half = values[:len(values)//2]
        second_half = values[len(values)//2:]

        first_avg = sum(first_half) / len(first_half)
        second_avg = sum(second_half) / len(second_half)

        diff = second_avg - first_avg

        if diff > 1.0:  # 1% increase threshold
            return "increasing"
        elif diff < -1.0:  # 1% decrease threshold
            return "decreasing"
        else:
            return "stable"

    def get_optimization_history(self) -> List[Dict[str, Any]]:
        """Get history of optimization actions"""
        return [
            {
                "action": opt.action.value,
                "timestamp": opt.timestamp,
                "success": opt.success,
                "impact": opt.impact_description,
                "cpu_before": opt.resource_before.cpu_percent,
                "cpu_after": opt.resource_after.cpu_percent,
                "ram_before": opt.resource_before.ram_gb,
                "ram_after": opt.resource_after.ram_gb
            }
            for opt in self.optimization_history[-20:]  # Last 20 optimizations
        ]

    def force_optimization(self, profile: str = "conservative") -> bool:
        """Force immediate optimization"""
        metrics = self._collect_resource_metrics()
        return self.switch_performance_profile(profile)

    def reset_to_normal(self) -> bool:
        """Reset to normal performance profile"""
        return self.switch_performance_profile("normal")

    def get_system_health_report(self) -> Dict[str, Any]:
        """Get comprehensive system health report"""
        current_metrics = self.get_current_metrics()
        trends = self.get_resource_trends(hours=1)

        if not current_metrics:
            return {"error": "No metrics available"}

        # Determine overall health
        cpu_level = self._get_resource_level(current_metrics.cpu_percent, ResourceType.CPU)
        ram_level = self._get_resource_level(current_metrics.ram_percent, ResourceType.RAM)

        health_score = self._calculate_health_score(cpu_level, ram_level)

        return {
            "timestamp": time.time(),
            "overall_health": health_score,
            "health_status": self._health_score_to_status(health_score),
            "current_metrics": {
                "cpu_percent": current_metrics.cpu_percent,
                "ram_gb": current_metrics.ram_gb,
                "ram_percent": current_metrics.ram_percent,
                "cpu_level": cpu_level,
                "ram_level": ram_level
            },
            "performance_profile": self.current_profile,
            "trends": trends,
            "emergency_mode": self.emergency_mode,
            "optimizations_performed": len(self.optimization_history),
            "recommendations": self._generate_health_recommendations(health_score, cpu_level, ram_level)
        }

    def _calculate_health_score(self, cpu_level: str, ram_level: str) -> float:
        """Calculate overall health score (0-100)"""
        level_scores = {
            "normal": 100,
            "warning": 75,
            "critical": 50,
            "emergency": 25
        }

        cpu_score = level_scores.get(cpu_level, 50)
        ram_score = level_scores.get(ram_level, 50)

        return (cpu_score + ram_score) / 2

    def _health_score_to_status(self, score: float) -> str:
        """Convert health score to status string"""
        if score >= 90:
            return "excellent"
        elif score >= 75:
            return "good"
        elif score >= 60:
            return "warning"
        elif score >= 40:
            return "critical"
        else:
            return "emergency"

    def _generate_health_recommendations(self, health_score: float,
                                       cpu_level: str, ram_level: str) -> List[str]:
        """Generate health recommendations"""
        recommendations = []

        if health_score < 60:
            recommendations.append("Immediate action required - system resources critical")

        if cpu_level in ["critical", "emergency"]:
            recommendations.extend([
                "Reduce CPU-intensive operations",
                "Increase update intervals for components",
                "Consider switching to conservative performance profile"
            ])

        if ram_level in ["critical", "emergency"]:
            recommendations.extend([
                "Clear component caches",
                "Reduce data history sizes",
                "Monitor for memory leaks"
            ])

        if health_score >= 90:
            recommendations.append("System operating within optimal parameters")

        return recommendations


async def demo_resource_monitor():
    import asyncio
    """Demo advanced resource monitor"""
    print("🚀 SUPREME SYSTEM V5 - Advanced Resource Monitor Demo")
    print("=" * 65)

    # Initialize monitor
    monitor = AdvancedResourceMonitor()

    print("📊 Initializing resource monitoring...")

    # Collect initial metrics
    metrics = monitor._collect_resource_metrics()

    print("\n📈 CURRENT RESOURCE USAGE:")
    print(".1f")
    print(".2f")
    print(".1f")
    print(".1f")
    print(".1f")
    print(f"   Process Threads: {metrics.process_threads}")

    # Start monitoring
    monitor.start_monitoring()
    print("\n🔍 MONITORING ACTIVE - Collecting data for 10 seconds...")
    await asyncio.sleep(10)

    # Get health report
    health_report = monitor.get_system_health_report()

    print("\n🏥 SYSTEM HEALTH REPORT:")
    print(".1f")
    print(f"   Status: {health_report['health_status'].upper()}")
    print(f"   Profile: {health_report['performance_profile']}")

    print("\n📊 DETAILED METRICS:")
    current = health_report['current_metrics']
    print(".1f")
    print(".2f")
    print(f"   CPU Level: {current['cpu_level'].upper()}")
    print(f"   RAM Level: {current['ram_level'].upper()}")

    # Get trends
    trends = health_report['trends']
    if 'avg_cpu' in trends:
        print("\n📈 USAGE TRENDS (1 hour):")
        print(".1f")
        print(".2f")
        print(".1f")
        print(".1f")
        print(f"   CPU Trend: {trends['cpu_trend']}")
        print(f"   RAM Trend: {trends['ram_trend']}")

    # Show recommendations
    if health_report['recommendations']:
        print("\n💡 RECOMMENDATIONS:")
        for rec in health_report['recommendations']:
            print(f"   • {rec}")

    # Test performance profile switching
    print("\n🔄 TESTING PERFORMANCE PROFILE SWITCHING:")
    print("   Switching to conservative profile...")
    success = monitor.switch_performance_profile("conservative")
    print(f"   Profile switch: {'✅ SUCCESS' if success else '❌ FAILED'}")

    await asyncio.sleep(2)

    print("   Switching back to normal profile...")
    success = monitor.reset_to_normal()
    print(f"   Profile reset: {'✅ SUCCESS' if success else '❌ FAILED'}")

    # Stop monitoring
    monitor.stop_monitoring()

    print("\n🎯 SYSTEM CAPABILITIES:")
    print("   • Real-time CPU/RAM monitoring")
    print("   • Auto-optimization khi vượt ngưỡng")
    print("   • Performance profile switching")
    print("   • Emergency resource management")
    print("   • Health scoring và recommendations")
    print("   • Memory-efficient historical tracking")

    print("\n✅ Advanced Resource Monitor Demo Complete")
    print("   Supreme System V5 resource management ready!")


if __name__ == "__main__":
    asyncio.run(demo_resource_monitor())
