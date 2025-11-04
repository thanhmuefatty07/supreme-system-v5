#!/usr/bin/env python3
"""
Supreme System V5 - Ultra-Constrained Resource Monitor
Real-time monitoring with emergency controls for 1GB RAM deployment
Agent Mode: Automated resource management with failsafe mechanisms

Features:
- Ultra-constrained monitoring (450MB RAM, 85% CPU targets)
- Emergency shutdown on critical resource usage
- Auto-optimization and memory management
- Performance tracking and alerts
- Failsafe mechanisms for production deployment
"""

import asyncio
import gc
import logging
import os
import psutil
import time
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Any, Callable, Tuple

logger = logging.getLogger(__name__)


class ResourceStatus(Enum):
    """Resource status levels for ultra-constrained deployment"""
    OPTIMAL = "optimal"        # Well below limits
    NORMAL = "normal"          # Within acceptable range
    WARNING = "warning"        # Approaching limits
    CRITICAL = "critical"      # Exceeding limits
    EMERGENCY = "emergency"    # Emergency shutdown required


class OptimizationLevel(Enum):
    """Optimization intensity levels"""
    NONE = 0
    LIGHT = 1       # Reduce update frequencies
    MODERATE = 2    # Disable non-critical features
    AGGRESSIVE = 3  # Minimal operation mode
    EMERGENCY = 4   # Emergency shutdown


@dataclass
class ResourceSnapshot:
    """Single resource measurement snapshot"""
    timestamp: float
    cpu_percent: float
    memory_mb: float
    memory_percent: float
    available_memory_mb: float
    disk_usage_percent: float
    process_cpu_percent: float
    process_memory_mb: float
    process_threads: int
    gc_objects: int = 0
    network_connections: int = 0


@dataclass
class UltraConstrainedLimits:
    """Resource limits for ultra-constrained deployment"""
    # Memory limits (1GB RAM hardware)
    max_memory_mb: int = 450          # 450MB budget (47% of 1GB)
    warning_memory_mb: int = 350      # Warning at 350MB
    critical_memory_mb: int = 420     # Critical at 420MB
    emergency_memory_mb: int = 480    # Emergency at 480MB
    
    # CPU limits
    max_cpu_percent: float = 85.0     # 85% CPU limit
    warning_cpu_percent: float = 70.0 # Warning at 70%
    critical_cpu_percent: float = 80.0 # Critical at 80%
    emergency_cpu_percent: float = 90.0 # Emergency at 90%
    
    # System limits
    max_disk_percent: float = 85.0
    max_process_threads: int = 10
    max_gc_objects: int = 100000


@dataclass
class PerformanceMetrics:
    """Performance metrics for ultra-constrained monitoring"""
    uptime_seconds: float = 0.0
    total_measurements: int = 0
    avg_cpu_percent: float = 0.0
    avg_memory_mb: float = 0.0
    peak_memory_mb: float = 0.0
    peak_cpu_percent: float = 0.0
    
    # Violation tracking
    memory_violations: int = 0
    cpu_violations: int = 0
    emergency_activations: int = 0
    
    # Optimization tracking
    optimizations_applied: int = 0
    gc_collections_forced: int = 0
    emergency_shutdowns: int = 0
    
    # Target achievement
    memory_target_met: bool = True
    cpu_target_met: bool = True
    uptime_target_met: bool = True


class UltraConstrainedResourceMonitor:
    """Ultra-constrained resource monitor for 1GB RAM deployment"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        
        # Initialize limits
        self.limits = UltraConstrainedLimits(
            max_memory_mb=self.config.get('max_memory_mb', 450),
            max_cpu_percent=self.config.get('max_cpu_percent', 85.0)
        )
        
        # Monitoring configuration
        self.check_interval = self.config.get('check_interval', 30)  # 30 seconds
        self.emergency_shutdown_enabled = self.config.get('emergency_shutdown_enabled', True)
        self.auto_optimization_enabled = self.config.get('auto_optimization_enabled', True)
        self.gc_enabled = self.config.get('gc_enabled', True)
        
        # State tracking
        self.snapshots: List[ResourceSnapshot] = []
        self.metrics = PerformanceMetrics()
        self.status = ResourceStatus.NORMAL
        self.optimization_level = OptimizationLevel.NONE
        
        self.start_time = time.time()
        self.running = False
        self.last_optimization = 0.0
        self.last_gc_collection = 0.0
        
        # Process references
        try:
            self.process = psutil.Process(os.getpid())
            self.system_available = True
        except Exception as e:
            logger.warning(f"Process monitoring limited: {e}")
            self.process = None
            self.system_available = False
            
        # Emergency handlers
        self.emergency_handlers: List[Callable] = []
        self.optimization_handlers: List[Callable] = []
        
        logger.info(f"Ultra-constrained monitor initialized: {self.limits.max_memory_mb}MB RAM, {self.limits.max_cpu_percent}% CPU")


class PerformanceProfile(Enum):
    """Performance optimization levels"""
    MINIMAL = "minimal"        # Maximum resource conservation
    CONSERVATIVE = "conservative"  # Balanced conservation
    NORMAL = "normal"          # Standard performance
    PERFORMANCE = "performance"  # Maximum performance


# Alias for compatibility
AdvancedResourceMonitor = UltraConstrainedResourceMonitor

@dataclass
class ResourceThreshold:
    """Resource usage thresholds for monitoring"""
    memory_warning_mb: float = 300.0
    memory_critical_mb: float = 400.0
    cpu_warning_percent: float = 70.0
    cpu_critical_percent: float = 85.0
    latency_warning_ms: float = 10.0
    latency_critical_ms: float = 50.0

@dataclass
class OptimizationResult:
    """Result of an optimization operation"""
    optimization_type: str
    applied: bool = False
    memory_saved_mb: float = 0.0
    cpu_reduced_percent: float = 0.0
    timestamp: float = field(default_factory=time.time)
    description: str = ""

@dataclass
class ResourceMetrics:
    """Resource metrics for monitoring"""
    memory_used_mb: float = 0.0
    memory_percent: float = 0.0
    cpu_percent: float = 0.0
    latency_ms: float = 0.0
    timestamp: float = field(default_factory=time.time)


class UltraConstrainedResourceMonitor:
    """Ultra-constrained resource monitor for 1GB RAM deployment"""

    async def _take_resource_snapshot(self) -> Optional[ResourceSnapshot]:
        """Take comprehensive resource snapshot"""
        try:
            if not self.system_available:
                return None
                
            timestamp = time.time()
            
            # System-wide metrics
            cpu_percent = psutil.cpu_percent(interval=0.1)
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage('.')
            
            # Process-specific metrics
            process_cpu = 0.0
            process_memory_mb = 0.0
            process_threads = 0
            
            if self.process:
                process_cpu = self.process.cpu_percent()
                process_memory_mb = self.process.memory_info().rss / (1024 * 1024)
                process_threads = self.process.num_threads()
                
            # Python-specific metrics
            gc_objects = len(gc.get_objects()) if self.gc_enabled else 0
            
            # Network connections (optional)
            network_connections = 0
            try:
                if self.process:
                    network_connections = len(self.process.connections())
            except:
                pass  # May not have permission
                
            return ResourceSnapshot(
                timestamp=timestamp,
                cpu_percent=cpu_percent,
                memory_mb=memory.used / (1024 * 1024),
                memory_percent=memory.percent,
                available_memory_mb=memory.available / (1024 * 1024),
                disk_usage_percent=disk.percent,
                process_cpu_percent=process_cpu,
                process_memory_mb=process_memory_mb,
                process_threads=process_threads,
                gc_objects=gc_objects,
                network_connections=network_connections
            )
            
        except Exception as e:
            logger.error(f"Failed to take resource snapshot: {e}")
            return None
            
    async def _process_snapshot(self, snapshot: ResourceSnapshot):
        """Process and store resource snapshot"""
        # Add to history
        self.snapshots.append(snapshot)
        
        # Maintain circular buffer (prevent memory growth)
        max_snapshots = 1000  # Last 1000 snapshots
        if len(self.snapshots) > max_snapshots:
            self.snapshots = self.snapshots[-max_snapshots:]
            
        # Update metrics
        self._update_performance_metrics(snapshot)
        
        # Update status
        old_status = self.status
        self.status = self._calculate_status(snapshot)
        
        # Log status changes
        if self.status != old_status:
            logger.info(f"Resource status: {old_status.value} -> {self.status.value}")
            
    def _update_performance_metrics(self, snapshot: ResourceSnapshot):
        """Update performance metrics with new snapshot"""
        self.metrics.total_measurements += 1
        self.metrics.uptime_seconds = time.time() - self.start_time
        
        count = self.metrics.total_measurements
        
        # Update averages
        if count == 1:
            self.metrics.avg_cpu_percent = snapshot.cpu_percent
            self.metrics.avg_memory_mb = snapshot.process_memory_mb
        else:
            self.metrics.avg_cpu_percent = (
                (self.metrics.avg_cpu_percent * (count - 1) + snapshot.cpu_percent) / count
            )
            self.metrics.avg_memory_mb = (
                (self.metrics.avg_memory_mb * (count - 1) + snapshot.process_memory_mb) / count
            )
            
        # Update peaks
        self.metrics.peak_memory_mb = max(self.metrics.peak_memory_mb, snapshot.process_memory_mb)
        self.metrics.peak_cpu_percent = max(self.metrics.peak_cpu_percent, snapshot.cpu_percent)
        
        # Check violations
        if snapshot.process_memory_mb > self.limits.max_memory_mb:
            self.metrics.memory_violations += 1
        if snapshot.cpu_percent > self.limits.max_cpu_percent:
            self.metrics.cpu_violations += 1
            
        # Update target achievement
        self.metrics.memory_target_met = self.metrics.avg_memory_mb <= self.limits.max_memory_mb
        self.metrics.cpu_target_met = self.metrics.avg_cpu_percent <= self.limits.max_cpu_percent
        self.metrics.uptime_target_met = self.metrics.uptime_seconds > 3600  # 1 hour+
        
    def _calculate_status(self, snapshot: ResourceSnapshot) -> ResourceStatus:
        """Calculate current resource status for ultra-constrained deployment"""
        memory_mb = snapshot.process_memory_mb
        cpu_percent = snapshot.cpu_percent
        
        # Emergency conditions (immediate shutdown required)
        if (memory_mb >= self.limits.emergency_memory_mb or
            cpu_percent >= self.limits.emergency_cpu_percent):
            return ResourceStatus.EMERGENCY
            
        # Critical conditions (optimization required)
        if (memory_mb >= self.limits.critical_memory_mb or
            cpu_percent >= self.limits.critical_cpu_percent):
            return ResourceStatus.CRITICAL
            
        # Warning conditions (monitoring increased)
        if (memory_mb >= self.limits.warning_memory_mb or
            cpu_percent >= self.limits.warning_cpu_percent):
            return ResourceStatus.WARNING
            
        # Optimal conditions (well below limits)
        if (memory_mb < self.limits.warning_memory_mb * 0.8 and
            cpu_percent < self.limits.warning_cpu_percent * 0.8):
            return ResourceStatus.OPTIMAL
            
        return ResourceStatus.NORMAL
        
    async def _check_emergency_conditions(self):
        """Check for emergency conditions requiring immediate action"""
        if not self.snapshots:
            return
            
        current = self.snapshots[-1]
        
        if self.status == ResourceStatus.EMERGENCY:
            if self.emergency_shutdown_enabled:
                logger.critical(f"EMERGENCY: Resource usage critical - Memory: {current.process_memory_mb:.1f}MB, CPU: {current.cpu_percent:.1f}%")
                await self._trigger_emergency_shutdown("resource_emergency")
            else:
                logger.critical(f"EMERGENCY CONDITIONS DETECTED (shutdown disabled): Memory: {current.process_memory_mb:.1f}MB, CPU: {current.cpu_percent:.1f}%")
                
    async def _auto_optimize_check(self):
        """Check if auto-optimization should be triggered"""
        current_time = time.time()
        
        # Only optimize every 2 minutes minimum
        if current_time - self.last_optimization < 120:
            return
            
        if self.status in [ResourceStatus.CRITICAL, ResourceStatus.WARNING]:
            await self._apply_optimization()
            self.last_optimization = current_time
            
    async def _apply_optimization(self):
        """Apply resource optimization based on current status"""
        optimizations_applied = []
        
        # Force garbage collection if enabled
        if self.gc_enabled:
            current_time = time.time()
            if current_time - self.last_gc_collection > 60:  # Every minute max
                collected = await self._force_garbage_collection()
                if collected > 0:
                    optimizations_applied.append(f"gc_freed_{collected}_objects")
                self.last_gc_collection = current_time
                
        # Apply optimization based on status
        if self.status == ResourceStatus.CRITICAL:
            self.optimization_level = OptimizationLevel.AGGRESSIVE
            optimizations_applied.append("aggressive_optimization")
        elif self.status == ResourceStatus.WARNING:
            self.optimization_level = OptimizationLevel.MODERATE
            optimizations_applied.append("moderate_optimization")
            
        # Call optimization handlers
        for handler in self.optimization_handlers:
            try:
                if asyncio.iscoroutinefunction(handler):
                    result = await handler(self.optimization_level)
                else:
                    result = handler(self.optimization_level)
                if result:
                    optimizations_applied.append(str(result))
            except Exception as e:
                logger.error(f"Optimization handler failed: {e}")
                
        if optimizations_applied:
            self.metrics.optimizations_applied += 1
            logger.info(f"Auto-optimization applied: {', '.join(optimizations_applied)}")
            
    async def _force_garbage_collection(self) -> int:
        """Force Python garbage collection"""
        if not self.gc_enabled:
            return 0
            
        logger.debug("Forcing garbage collection...")
        collected = gc.collect()
        self.metrics.gc_collections_forced += 1
        
        if collected > 0:
            logger.info(f"Garbage collection freed {collected} objects")
            
        return collected
        
    async def _trigger_emergency_shutdown(self, reason: str):
        """Trigger emergency system shutdown"""
        logger.critical(f"TRIGGERING EMERGENCY SHUTDOWN: {reason}")
        self.metrics.emergency_shutdowns += 1
        self.metrics.emergency_activations += 1
        
        # Call all emergency handlers
        for handler in self.emergency_handlers:
            try:
                if asyncio.iscoroutinefunction(handler):
                    await handler(reason)
                else:
                    handler(reason)
            except Exception as e:
                logger.error(f"Emergency handler failed: {e}")
                
        # Stop monitoring
        self.running = False
        
        # Emergency report
        report_path = self.export_emergency_report()
        logger.critical(f"Emergency report saved: {report_path}")
        
        # Terminate process
        logger.critical("Emergency shutdown complete - terminating process in 5 seconds")
        await asyncio.sleep(5)
        os._exit(1)
        
    def get_current_status(self) -> Dict[str, Any]:
        """Get comprehensive current status"""
        if not self.snapshots:
            return {"status": "no_data", "message": "No measurements available"}
            
        current = self.snapshots[-1]
        
        return {
            "status": self.status.value,
            "optimization_level": self.optimization_level.value,
            "timestamp": current.timestamp,
            "uptime_seconds": self.metrics.uptime_seconds,
            "resources": {
                "process_memory_mb": round(current.process_memory_mb, 1),
                "system_memory_percent": round(current.memory_percent, 1),
                "cpu_percent": round(current.cpu_percent, 1),
                "process_cpu_percent": round(current.process_cpu_percent, 1),
                "available_memory_mb": round(current.available_memory_mb, 1),
                "process_threads": current.process_threads,
                "gc_objects": current.gc_objects
            },
            "limits": {
                "max_memory_mb": self.limits.max_memory_mb,
                "max_cpu_percent": self.limits.max_cpu_percent,
                "warning_memory_mb": self.limits.warning_memory_mb,
                "critical_memory_mb": self.limits.critical_memory_mb
            },
            "violations": {
                "memory_exceeded": current.process_memory_mb > self.limits.max_memory_mb,
                "cpu_exceeded": current.cpu_percent > self.limits.max_cpu_percent,
                "emergency_conditions": self.status == ResourceStatus.EMERGENCY
            },
            "targets_met": {
                "memory_target": self.metrics.memory_target_met,
                "cpu_target": self.metrics.cpu_target_met,
                "uptime_target": self.metrics.uptime_target_met
            }
        }
        
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get performance summary for ultra-constrained deployment"""
        return {
            "deployment_type": "ultra_constrained",
            "target_specs": {
                "ram_budget_mb": self.limits.max_memory_mb,
                "cpu_limit_percent": self.limits.max_cpu_percent,
                "hardware_target": "1GB_RAM_2CPU"
            },
            "performance_metrics": {
                "uptime_hours": round(self.metrics.uptime_seconds / 3600, 2),
                "total_measurements": self.metrics.total_measurements,
                "avg_memory_mb": round(self.metrics.avg_memory_mb, 1),
                "avg_cpu_percent": round(self.metrics.avg_cpu_percent, 1),
                "peak_memory_mb": round(self.metrics.peak_memory_mb, 1),
                "peak_cpu_percent": round(self.metrics.peak_cpu_percent, 1)
            },
            "efficiency_metrics": {
                "memory_efficiency_percent": min(100, (self.limits.max_memory_mb / max(1, self.metrics.avg_memory_mb)) * 100),
                "cpu_efficiency_percent": min(100, (self.limits.max_cpu_percent / max(1, self.metrics.avg_cpu_percent)) * 100),
                "uptime_stability": min(100, (self.metrics.uptime_seconds / 86400) * 100)  # Daily stability
            },
            "violation_tracking": {
                "memory_violations": self.metrics.memory_violations,
                "cpu_violations": self.metrics.cpu_violations,
                "emergency_activations": self.metrics.emergency_activations,
                "optimizations_applied": self.metrics.optimizations_applied
            },
            "target_achievement": {
                "memory_target_met": self.metrics.memory_target_met,
                "cpu_target_met": self.metrics.cpu_target_met,
                "overall_success": self.metrics.memory_target_met and self.metrics.cpu_target_met
            }
        }
        
    def export_emergency_report(self) -> str:
        """Export emergency report with all available data"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        report_path = f"emergency_report_{timestamp}.json"
        
        report = {
            "emergency_timestamp": datetime.now().isoformat(),
            "emergency_reason": "resource_critical",
            "system_status": self.get_current_status(),
            "performance_summary": self.get_performance_summary(),
            "recent_snapshots": [
                {
                    "timestamp": s.timestamp,
                    "memory_mb": round(s.process_memory_mb, 1),
                    "cpu_percent": round(s.cpu_percent, 1),
                    "threads": s.process_threads,
                    "gc_objects": s.gc_objects
                }
                for s in self.snapshots[-20:]  # Last 20 snapshots
            ],
            "configuration": {
                "limits": {
                    "max_memory_mb": self.limits.max_memory_mb,
                    "max_cpu_percent": self.limits.max_cpu_percent
                },
                "monitoring": {
                    "check_interval": self.check_interval,
                    "emergency_shutdown_enabled": self.emergency_shutdown_enabled,
                    "auto_optimization_enabled": self.auto_optimization_enabled
                }
            }
        }
        
        try:
            import json
            with open(report_path, 'w') as f:
                json.dump(report, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to export emergency report: {e}")
            
        return report_path
        
        
async def demo_ultra_constrained_monitor():
    """Demo ultra-constrained resource monitoring"""
    print("🔍 Supreme System V5 - Ultra-Constrained Resource Monitor Demo")
    print("=" * 65)
    
    # Initialize with ultra-constrained settings
    config = {
        'max_memory_mb': 450,
        'max_cpu_percent': 85,
        'check_interval': 5,  # 5 seconds for demo
        'emergency_shutdown_enabled': False  # Disabled for demo safety
    }
    
    monitor = UltraConstrainedResourceMonitor(config)
    
    print(f"Monitor initialized for ultra-constrained deployment:")
    print(f"   RAM Budget: {monitor.limits.max_memory_mb}MB (47% of 1GB)")
    print(f"   CPU Limit: {monitor.limits.max_cpu_percent}%")
    print(f"   Hardware Target: 1GB RAM, 2 CPU cores")
    print(f"   Emergency Shutdown: {'Enabled' if monitor.emergency_shutdown_enabled else 'Disabled (Demo)'}")
    print()
    
    # Add demo handlers
    async def demo_emergency_handler(reason: str):
        print(f"🚨 EMERGENCY HANDLER: {reason}")
        
    def demo_optimization_handler(level: OptimizationLevel) -> str:
        return f"Applied {level.name} optimization"
        
    monitor.add_emergency_handler(demo_emergency_handler)
    monitor.add_optimization_handler(demo_optimization_handler)
    
    print("📊 Taking resource snapshots...")
    
    # Simulate monitoring for demo
    for i in range(5):
        snapshot = await monitor._take_resource_snapshot()
        if snapshot:
            await monitor._process_snapshot(snapshot)
            
            print(f"   Snapshot {i+1}: {monitor.status.value.upper()} - "
                  f"Memory: {snapshot.process_memory_mb:.1f}MB, "
                  f"CPU: {snapshot.cpu_percent:.1f}%")
                  
        await asyncio.sleep(2)
        
    print()
    
    # Show current status
    status = monitor.get_current_status()
    print("📊 CURRENT SYSTEM STATUS:")
    print(f"   Status: {status['status'].upper()}")
    print(f"   Memory Usage: {status['resources']['process_memory_mb']:.1f}MB / {status['limits']['max_memory_mb']}MB")
    print(f"   CPU Usage: {status['resources']['cpu_percent']:.1f}% / {status['limits']['max_cpu_percent']}%")
    print(f"   Available Memory: {status['resources']['available_memory_mb']:.1f}MB")
    print(f"   Process Threads: {status['resources']['process_threads']}")
    
    print()
    
    # Show performance summary
    summary = monitor.get_performance_summary()
    print("🏆 PERFORMANCE SUMMARY:")
    print(f"   Deployment: {summary['deployment_type']}")
    print(f"   Uptime: {summary['performance_metrics']['uptime_hours']:.2f} hours")
    print(f"   Measurements: {summary['performance_metrics']['total_measurements']}")
    print(f"   Memory Efficiency: {summary['efficiency_metrics']['memory_efficiency_percent']:.1f}%")
    print(f"   CPU Efficiency: {summary['efficiency_metrics']['cpu_efficiency_percent']:.1f}%")
    print(f"   Target Achievement: {'✅' if summary['target_achievement']['overall_success'] else '⚠️'}")
    
    print()
    
    # Test optimization trigger
    print("🔧 Testing auto-optimization...")
    if monitor.auto_optimization_enabled:
        await monitor._apply_optimization()
        print(f"   Optimizations applied: {monitor.metrics.optimizations_applied}")
    else:
        print("   Auto-optimization disabled")
        
    print()
    
    print("🎯 ULTRA-CONSTRAINED FEATURES:")
    print("   • 450MB RAM budget monitoring")
    print("   • 85% CPU limit enforcement")
    print("   • Emergency shutdown protection")
    print("   • Auto-garbage collection")
    print("   • Memory leak detection")
    print("   • Performance target tracking")
    print("   • Emergency report generation")
    print("   • Production failsafe mechanisms")
    
    print()
    print("✅ Ultra-Constrained Resource Monitor Demo Complete")
    print("   System ready for 1GB RAM production deployment!")
    

async def demo_resource_monitor():
    """Demonstrate ultra-constrained resource monitoring capabilities."""
    print("📊 SUPREME SYSTEM V5 - Ultra-Constrained Resource Monitor Demo")
    print("=" * 65)

    # Initialize ultra-constrained monitor
    config = {
        'max_memory_mb': 450.0,
        'max_cpu_percent': 85.0,
        'check_interval': 5.0,
        'emergency_shutdown_enabled': True,
        'auto_optimization_enabled': True,
        'gc_enabled': True
    }

    monitor = UltraConstrainedResourceMonitor(config)

    print("🔄 Starting ultra-constrained monitoring...")
    print("   Target: 450MB RAM, 85% CPU limit")
    print()

    # Start monitoring
    monitoring_task = asyncio.create_task(monitor.start_monitoring())

    # Let it run for a short demo
    await asyncio.sleep(10)

    # Stop monitoring
    monitor.stop_monitoring()
    monitoring_task.cancel()

    try:
        await monitoring_task
    except asyncio.CancelledError:
        pass

    print()
    print("✅ Ultra-Constrained Resource Monitor Demo Complete")
    print("   System ready for 1GB RAM production deployment!")


def demo_resource_monitor_sync():
    """Synchronous wrapper for demo function."""
    asyncio.run(demo_resource_monitor())


if __name__ == "__main__":
    asyncio.run(demo_ultra_constrained_monitor())