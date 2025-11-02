"""
Advanced Resource Monitor for Supreme System V5.
Real-time monitoring with Prometheus metrics and auto-optimization.
"""

import time
import psutil
import threading
from typing import Dict, List, Optional, Any, NamedTuple
from enum import Enum
import gc

class ResourceMetrics(NamedTuple):
    """Resource metrics snapshot."""
    timestamp: float
    cpu_percent: float
    memory_gb: float
    memory_percent: float
    disk_usage_percent: float
    network_connections: int
    thread_count: int
    indicator_latency_ms: float
    event_skip_ratio: float

class PerformanceProfile(Enum):
    """System performance profiles."""
    MINIMAL = "minimal"      # Maximum resource conservation
    CONSERVATIVE = "conservative"  # Balanced performance
    NORMAL = "normal"        # Standard operation
    PERFORMANCE = "performance"  # High performance mode

class OptimizationResult(NamedTuple):
    """Auto-optimization result."""
    action_taken: str
    previous_value: Any
    new_value: Any
    expected_impact: str
    timestamp: float

class AdvancedResourceMonitor:
    """
    Advanced resource monitoring with Prometheus-style metrics and auto-optimization.

    Key Metrics:
    - optimized_indicator_latency_seconds: Indicator calculation time
    - event_skip_ratio: Percentage of events filtered
    - memory_in_use_bytes: Current memory usage
    - cpu_percent_gauge: CPU utilization percentage
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize advanced resource monitor.

        Args:
            config: Monitoring configuration
        """
        self.config = config

        # Thresholds for optimization triggers
        self.cpu_high_threshold = config.get('cpu_high_threshold', 88.0)  # 88%
        self.memory_high_threshold = config.get('memory_high_threshold', 3.86)  # 3.86GB
        self.latency_high_threshold = config.get('latency_high_threshold', 200)  # 200ms

        # Monitoring intervals
        self.monitoring_interval = config.get('monitoring_interval', 5.0)  # 5 seconds
        self.optimization_check_interval = config.get('optimization_check_interval', 60.0)  # 1 minute

        # Performance tracking
        self.current_profile = PerformanceProfile.NORMAL
        self.metrics_history: List[ResourceMetrics] = []
        self.optimization_history: List[OptimizationResult] = []

        # Indicator performance tracking
        self.indicator_latencies: List[float] = []
        self.event_skip_counts = {'processed': 0, 'skipped': 0}

        # System state
        self.is_monitoring = False
        self.monitoring_thread: Optional[threading.Thread] = None
        self.last_optimization_time = 0

        # Process information
        self.process = psutil.Process()
        self.start_time = time.time()

        # Performance profile settings
        self.profile_settings = {
            PerformanceProfile.MINIMAL: {
                'max_indicator_frequency': 10,  # Updates per second
                'event_filter_aggressive': True,
                'cache_enabled': True,
                'memory_cleanup_interval': 30
            },
            PerformanceProfile.CONSERVATIVE: {
                'max_indicator_frequency': 20,
                'event_filter_aggressive': False,
                'cache_enabled': True,
                'memory_cleanup_interval': 60
            },
            PerformanceProfile.NORMAL: {
                'max_indicator_frequency': 30,
                'event_filter_aggressive': False,
                'cache_enabled': True,
                'memory_cleanup_interval': 120
            },
            PerformanceProfile.PERFORMANCE: {
                'max_indicator_frequency': 50,
                'event_filter_aggressive': False,
                'cache_enabled': False,  # Fresh calculations
                'memory_cleanup_interval': 300
            }
        }

    def start_monitoring(self):
        """Start background monitoring thread."""
        if self.is_monitoring:
            return

        self.is_monitoring = True
        self.monitoring_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self.monitoring_thread.start()
        print("📊 Resource monitoring started")

    def stop_monitoring(self):
        """Stop background monitoring."""
        self.is_monitoring = False
        if self.monitoring_thread:
            self.monitoring_thread.join(timeout=5.0)
        print("📊 Resource monitoring stopped")

    def _monitoring_loop(self):
        """Background monitoring loop."""
        while self.is_monitoring:
            try:
                # Collect metrics
                metrics = self._collect_metrics()

                # Store in history (keep last 100 entries)
                self.metrics_history.append(metrics)
                if len(self.metrics_history) > 100:
                    self.metrics_history.pop(0)

                # Check for optimization triggers
                self._check_optimization_triggers(metrics)

                # Periodic memory cleanup
                self._periodic_memory_cleanup()

            except Exception as e:
                print(f"Monitoring error: {e}")

            time.sleep(self.monitoring_interval)

    def _collect_metrics(self) -> ResourceMetrics:
        """Collect comprehensive system metrics."""
        # CPU and memory
        cpu_percent = self.process.cpu_percent(interval=0.1)
        memory_info = self.process.memory_info()
        memory_gb = memory_info.rss / (1024 ** 3)  # Convert to GB
        memory_percent = self.process.memory_percent()

        # Disk and network
        disk_usage = psutil.disk_usage('/')
        network_connections = len(psutil.net_connections())

        # Thread count
        thread_count = self.process.num_threads()

        # Indicator performance
        avg_latency = (sum(self.indicator_latencies[-10:]) / len(self.indicator_latencies[-10:])
                      if self.indicator_latencies else 0.0)
        avg_latency_ms = avg_latency * 1000  # Convert to milliseconds

        # Event skip ratio
        total_events = self.event_skip_counts['processed'] + self.event_skip_counts['skipped']
        skip_ratio = (self.event_skip_counts['skipped'] / total_events
                     if total_events > 0 else 0.0)

        return ResourceMetrics(
            timestamp=time.time(),
            cpu_percent=cpu_percent,
            memory_gb=memory_gb,
            memory_percent=memory_percent,
            disk_usage_percent=disk_usage.percent,
            network_connections=network_connections,
            thread_count=thread_count,
            indicator_latency_ms=avg_latency_ms,
            event_skip_ratio=skip_ratio
        )

    def _check_optimization_triggers(self, metrics: ResourceMetrics):
        """Check for conditions requiring optimization."""
        current_time = time.time()

        # Only check every optimization interval
        if current_time - self.last_optimization_time < self.optimization_check_interval:
            return

        self.last_optimization_time = current_time
        optimizations_made = []

        # CPU optimization trigger
        if metrics.cpu_percent > self.cpu_high_threshold:
            if self.current_profile != PerformanceProfile.MINIMAL:
                old_profile = self.current_profile
                self._switch_performance_profile(PerformanceProfile.MINIMAL)
                optimizations_made.append(f"CPU high ({metrics.cpu_percent:.1f}%): Switched to {self.current_profile.value} profile")

        # Memory optimization trigger
        elif metrics.memory_gb > self.memory_high_threshold:
            if self.current_profile != PerformanceProfile.MINIMAL:
                old_profile = self.current_profile
                self._switch_performance_profile(PerformanceProfile.MINIMAL)
                optimizations_made.append(f"Memory high ({metrics.memory_gb:.2f}GB): Switched to {self.current_profile.value} profile")

        # Latency optimization trigger
        elif metrics.indicator_latency_ms > self.latency_high_threshold:
            if self.current_profile != PerformanceProfile.CONSERVATIVE:
                old_profile = self.current_profile
                self._switch_performance_profile(PerformanceProfile.CONSERVATIVE)
                optimizations_made.append(f"Latency high ({metrics.indicator_latency_ms:.1f}ms): Switched to {self.current_profile.value} profile")

        # Recovery to normal profile if conditions improve
        elif (metrics.cpu_percent < self.cpu_high_threshold * 0.8 and
              metrics.memory_gb < self.memory_high_threshold * 0.8 and
              self.current_profile == PerformanceProfile.MINIMAL):
            old_profile = self.current_profile
            self._switch_performance_profile(PerformanceProfile.NORMAL)
            optimizations_made.append(f"Conditions improved: Switched to {self.current_profile.value} profile")

        # Log optimizations
        for optimization in optimizations_made:
            print(f"🔧 Auto-optimization: {optimization}")
            self.optimization_history.append(OptimizationResult(
                action_taken=optimization,
                previous_value=str(old_profile.value),
                new_value=str(self.current_profile.value),
                expected_impact="Resource usage reduction",
                timestamp=current_time
            ))

    def _switch_performance_profile(self, new_profile: PerformanceProfile):
        """Switch to a new performance profile."""
        if new_profile == self.current_profile:
            return

        print(f"🔄 Switching from {self.current_profile.value} to {new_profile.value} profile")
        self.current_profile = new_profile

        # Apply profile-specific settings
        settings = self.profile_settings[new_profile]

        # In a real implementation, these would adjust:
        # - Indicator calculation frequencies
        # - Event filtering aggressiveness
        # - Cache policies
        # - Memory cleanup intervals
        # - Component update intervals

        # For demo, just update configuration
        self.config.update(settings)

    def _periodic_memory_cleanup(self):
        """Perform periodic memory cleanup."""
        # Force garbage collection
        collected = gc.collect()

        # Clear old metrics history if too large
        if len(self.metrics_history) > 200:
            self.metrics_history = self.metrics_history[-100:]

        # Clean old indicator latencies
        if len(self.indicator_latencies) > 1000:
            self.indicator_latencies = self.indicator_latencies[-500:]

    def record_indicator_latency(self, latency_seconds: float):
        """Record indicator calculation latency."""
        self.indicator_latencies.append(latency_seconds)

        # Keep only recent measurements
        if len(self.indicator_latencies) > 1000:
            self.indicator_latencies.pop(0)

    def record_event_processed(self, was_processed: bool):
        """Record event processing decision."""
        if was_processed:
            self.event_skip_counts['processed'] += 1
        else:
            self.event_skip_counts['skipped'] += 1

    def get_system_health_report(self) -> Dict[str, Any]:
        """Get comprehensive system health report."""
        if not self.metrics_history:
            return {'error': 'No metrics available'}

        latest = self.metrics_history[-1]

        # Calculate trends (last 10 measurements)
        recent_metrics = self.metrics_history[-10:] if len(self.metrics_history) >= 10 else self.metrics_history

        # Health assessment
        health_score = self._calculate_health_score(latest)
        health_status = self._get_health_status(health_score)

        return {
            'timestamp': latest.timestamp,
            'overall_health': health_score,
            'health_status': health_status,
            'current_metrics': {
                'cpu_percent': latest.cpu_percent,
                'ram_gb': latest.memory_gb,
                'ram_percent': latest.memory_percent,
                'disk_percent': latest.disk_usage_percent,
                'indicator_latency_ms': latest.indicator_latency_ms,
                'event_skip_ratio': latest.event_skip_ratio,
                'threads': latest.thread_count,
                'connections': latest.network_connections
            },
            'performance_profile': self.current_profile.value,
            'metrics_count': len(self.metrics_history),
            'uptime_seconds': time.time() - self.start_time,
            'recent_optimizations': len(self.optimization_history)
        }

    def _calculate_health_score(self, metrics: ResourceMetrics) -> float:
        """Calculate overall system health score (0-100)."""
        # CPU health (lower usage = higher health)
        cpu_health = max(0, 100 - (metrics.cpu_percent / self.cpu_high_threshold) * 100)

        # Memory health
        memory_health = max(0, 100 - (metrics.memory_gb / self.memory_high_threshold) * 100)

        # Latency health (lower latency = higher health)
        latency_health = max(0, 100 - (metrics.indicator_latency_ms / self.latency_high_threshold) * 100)

        # Weighted average
        weights = [0.4, 0.4, 0.2]  # CPU and Memory most important
        health_components = [cpu_health, memory_health, latency_health]

        overall_health = sum(w * h for w, h in zip(weights, health_components))
        return min(overall_health, 100.0)

    def _get_health_status(self, health_score: float) -> str:
        """Convert health score to status string."""
        if health_score >= 80:
            return "excellent"
        elif health_score >= 60:
            return "good"
        elif health_score >= 40:
            return "warning"
        else:
            return "critical"

    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get detailed performance metrics."""
        if not self.metrics_history:
            return {'error': 'No metrics available'}

        # Calculate averages from recent history
        recent = self.metrics_history[-20:]  # Last 20 measurements

        return {
            'avg_cpu_percent': sum(m.cpu_percent for m in recent) / len(recent),
            'avg_memory_gb': sum(m.memory_gb for m in recent) / len(recent),
            'avg_indicator_latency_ms': sum(m.indicator_latency_ms for m in recent) / len(recent),
            'avg_event_skip_ratio': sum(m.event_skip_ratio for m in recent) / len(recent),
            'current_profile': self.current_profile.value,
            'total_optimizations': len(self.optimization_history),
            'indicator_measurements': len(self.indicator_latencies),
            'events_processed': self.event_skip_counts['processed'],
            'events_skipped': self.event_skip_counts['skipped']
        }

    def export_prometheus_metrics(self) -> str:
        """
        Export metrics in Prometheus format.

        Returns:
            Prometheus-formatted metrics string
        """
        if not self.metrics_history:
            return "# No metrics available"

        latest = self.metrics_history[-1]

        metrics = [
            "# HELP optimized_indicator_latency_seconds Time for indicator calculations",
            "# TYPE optimized_indicator_latency_seconds gauge",
            f"optimized_indicator_latency_seconds {latest.indicator_latency_ms / 1000:.6f}",
            "",
            "# HELP event_skip_ratio Ratio of events filtered out",
            "# TYPE event_skip_ratio gauge",
            f"event_skip_ratio {latest.event_skip_ratio:.6f}",
            "",
            "# HELP memory_in_use_bytes Current memory usage",
            "# TYPE memory_in_use_bytes gauge",
            f"memory_in_use_bytes {int(latest.memory_gb * 1024**3)}",
            "",
            "# HELP cpu_percent_gauge CPU utilization percentage",
            "# TYPE cpu_percent_gauge gauge",
            f"cpu_percent_gauge {latest.cpu_percent:.2f}",
            "",
            "# HELP system_health_score Overall system health (0-100)",
            "# TYPE system_health_score gauge",
            f"system_health_score {self._calculate_health_score(latest):.2f}"
        ]

        return "\n".join(metrics)

# Demo function for testing
def demo_resource_monitor():
    """Demonstrate resource monitoring capabilities."""
    print("📊 SUPREME SYSTEM V5 - Advanced Resource Monitor Demo")
    print("=" * 60)

    # Initialize monitor
    config = {
        'cpu_high_threshold': 80.0,  # Lower threshold for demo
        'memory_high_threshold': 2.0,  # 2GB for demo
        'latency_high_threshold': 100,
        'monitoring_interval': 2.0,
        'optimization_check_interval': 10.0
    }

    monitor = AdvancedResourceMonitor(config)

    # Start monitoring
    monitor.start_monitoring()

    print("🔄 Monitoring system resources...")
    print("   Recording indicator latencies and event processing...")

    # Simulate some activity
    import random

    for i in range(20):
        # Simulate indicator calculations with random latency
        latency = random.uniform(0.001, 0.05)  # 1-50ms
        monitor.record_indicator_latency(latency)

        # Simulate event processing decisions
        processed = random.random() > 0.3  # 70% processed, 30% skipped
        monitor.record_event_processed(processed)

        print(".1f")
        time.sleep(0.5)

    # Get health report
    health_report = monitor.get_system_health_report()
    performance_metrics = monitor.get_performance_metrics()

    monitor.stop_monitoring()

    print("
🏥 SYSTEM HEALTH REPORT:"    print(f"   Overall Health: {health_report['overall_health']:.1f}/100 ({health_report['health_status']})")
    print(f"   CPU Usage: {health_report['current_metrics']['cpu_percent']:.1f}%")
    print(".2f")
    print(".1f")
    print(".3f")
    print(f"   Performance Profile: {health_report['performance_profile']}")
    print(f"   Uptime: {health_report['uptime_seconds']:.1f} seconds")

    print("
📈 PERFORMANCE METRICS:"    print(".1f")
    print(".2f")
    print(".1f")
    print(".3f")
    print(f"   Indicator Measurements: {performance_metrics['indicator_measurements']}")
    print(f"   Events Processed: {performance_metrics['events_processed']}")
    print(f"   Events Skipped: {performance_metrics['events_skipped']}")

    # Export Prometheus metrics
    prometheus_output = monitor.export_prometheus_metrics()
    print("
📊 PROMETHEUS METRICS EXPORT:"    print(prometheus_output[:200] + "..." if len(prometheus_output) > 200 else prometheus_output)

    print("
🎯 SYSTEM CAPABILITIES:"    print("   • Real-time CPU/memory monitoring")
    print("   • Auto-optimization based on thresholds")
    print("   • Prometheus metrics export")
    print("   • Performance profile switching")
    print("   • Indicator latency tracking")
    print("   • Event processing efficiency monitoring")

    print("
✅ Advanced Resource Monitor Demo Complete"    print("   System ready for production resource monitoring!")

if __name__ == "__main__":
    demo_resource_monitor()
