"""
Supreme System V5 - Advanced Process Optimizer
Manages CPU priorities, memory locking, and process isolation for i3-4GB systems
"""

import os
import sys
import psutil
import threading
import time
import signal
from typing import Dict, Optional, Callable
from dataclasses import dataclass
from enum import Enum

class ProcessPriority(Enum):
    REALTIME = -20
    HIGH = -10
    NORMAL = 0
    LOW = 10
    IDLE = 19

class ProcessType(Enum):
    TRADING = "trading"
    RISK = "risk"
    DATA = "data"
    DASHBOARD = "dashboard"
    AI = "ai"
    MONITORING = "monitoring"

@dataclass
class ProcessConfig:
    name: str
    pid: int
    priority: ProcessPriority
    cpu_affinity: list
    memory_limit_mb: int
    io_priority: int
    memory_lock: bool = False

@dataclass
class SystemResources:
    total_memory_mb: int
    available_memory_mb: int
    cpu_count: int
    cpu_usage_percent: float
    memory_usage_percent: float

class AdvancedProcessOptimizer:
    """
    Advanced process optimizer for i3-4GB systems
    Provides real-time process management, memory locking, and CPU affinity
    """

    def __init__(self, max_memory_mb: int = 3584):  # 3.5GB safe limit
        self.max_memory_mb = max_memory_mb
        self.processes: Dict[str, ProcessConfig] = {}
        self.monitoring_active = False
        self.monitor_thread: Optional[threading.Thread] = None
        self.alert_callbacks: list[Callable] = []
        self.performance_history = []

        # i3-4GB specific optimizations
        self.cpu_cores = min(psutil.cpu_count(), 4)  # Max 4 cores for i3
        self.memory_guard_mb = 512  # Keep 512MB free

        # Performance thresholds
        self.thresholds = {
            'memory_usage_percent': 85,
            'cpu_usage_percent': 90,
            'memory_available_mb': 256,
            'process_restart_threshold': 95  # Restart if >95% usage
        }

    def register_process(self, name: str, pid: int, process_type: ProcessType) -> bool:
        """Register a process for optimization"""
        try:
            # Create process configuration based on type
            config = self._create_process_config(name, pid, process_type)

            # Apply initial optimizations
            if not self._apply_process_optimizations(config):
                return False

            self.processes[name] = config

            print(f"✅ Process {name} (PID: {pid}) registered and optimized")
            return True

        except Exception as e:
            print(f"❌ Failed to register process {name}: {e}")
            return False

    def _create_process_config(self, name: str, pid: int, process_type: ProcessType) -> ProcessConfig:
        """Create optimized configuration for process type"""

        if process_type == ProcessType.TRADING:
            return ProcessConfig(
                name=name,
                pid=pid,
                priority=ProcessPriority.REALTIME,
                cpu_affinity=[0, 1],  # Cores 0,1 for trading
                memory_limit_mb=1536,  # 1.5GB for trading
                io_priority=0,  # Real-time I/O
                memory_lock=True
            )

        elif process_type == ProcessType.RISK:
            return ProcessConfig(
                name=name,
                pid=pid,
                priority=ProcessPriority.HIGH,
                cpu_affinity=[2],  # Core 2 for risk
                memory_limit_mb=256,
                io_priority=2,  # Best effort I/O
                memory_lock=False
            )

        elif process_type == ProcessType.DATA:
            return ProcessConfig(
                name=name,
                pid=pid,
                priority=ProcessPriority.NORMAL,
                cpu_affinity=[3],  # Core 3 for data
                memory_limit_mb=512,
                io_priority=2,
                memory_lock=False
            )

        elif process_type == ProcessType.DASHBOARD:
            return ProcessConfig(
                name=name,
                pid=pid,
                priority=ProcessPriority.LOW,
                cpu_affinity=[3],  # Core 3 (shared)
                memory_limit_mb=128,
                io_priority=3,  # Idle I/O
                memory_lock=False
            )

        else:  # Default configuration
            return ProcessConfig(
                name=name,
                pid=pid,
                priority=ProcessPriority.NORMAL,
                cpu_affinity=list(range(self.cpu_cores)),
                memory_limit_mb=256,
                io_priority=2,
                memory_lock=False
            )

    def _apply_process_optimizations(self, config: ProcessConfig) -> bool:
        """Apply all optimizations to a process"""
        try:
            process = psutil.Process(config.pid)

            # 1. Set process priority
            self._set_process_priority(config.pid, config.priority)

            # 2. Set CPU affinity
            if config.cpu_affinity:
                process.cpu_affinity(config.cpu_affinity)

            # 3. Set I/O priority
            self._set_io_priority(config.pid, config.io_priority)

            # 4. Memory locking (if requested and possible)
            if config.memory_lock:
                self._lock_process_memory(config.pid)

            # 5. Set memory limits
            if config.memory_limit_mb > 0:
                self._set_memory_limit(config.pid, config.memory_limit_mb)

            return True

        except Exception as e:
            print(f"⚠️ Failed to apply optimizations to PID {config.pid}: {e}")
            return False

    def _set_process_priority(self, pid: int, priority: ProcessPriority) -> bool:
        """Set process nice priority"""
        try:
            # Use os.setpriority for nice values
            os.setpriority(os.PRIO_PROCESS, pid, priority.value)

            # For real-time priority, use chrt if available
            if priority == ProcessPriority.REALTIME:
                try:
                    import subprocess
                    subprocess.run(['chrt', '-f', '-p', '50', str(pid)],
                                 capture_output=True, check=True)
                except (subprocess.CalledProcessError, FileNotFoundError):
                    pass  # chrt not available, nice priority will work

            return True
        except Exception as e:
            print(f"⚠️ Failed to set priority for PID {pid}: {e}")
            return False

    def _set_io_priority(self, pid: int, io_priority: int) -> bool:
        """Set I/O scheduling priority"""
        try:
            import subprocess

            # ionice command: class (1=realtime, 2=best-effort, 3=idle), priority
            if io_priority == 0:  # Real-time
                subprocess.run(['ionice', '-c', '1', '-n', '4', '-p', str(pid)],
                             capture_output=True, check=True)
            elif io_priority == 3:  # Idle
                subprocess.run(['ionice', '-c', '3', '-p', str(pid)],
                             capture_output=True, check=True)
            else:  # Best effort
                subprocess.run(['ionice', '-c', '2', '-n', str(io_priority), '-p', str(pid)],
                             capture_output=True, check=True)

            return True
        except (subprocess.CalledProcessError, FileNotFoundError):
            # ionice not available on this system
            return False
        except Exception as e:
            print(f"⚠️ Failed to set I/O priority for PID {pid}: {e}")
            return False

    def _lock_process_memory(self, pid: int) -> bool:
        """Lock process memory to prevent swapping"""
        try:
            # Try to lock memory using mlockall
            import mlock
            # Note: This requires root privileges and appropriate capabilities
            mlock.mlockall(mlock.MCL_CURRENT | mlock.MCL_FUTURE)
            print(f"✅ Memory locked for PID {pid}")
            return True
        except ImportError:
            print("⚠️ mlock module not available, memory locking disabled")
            return False
        except OSError as e:
            print(f"⚠️ Memory locking failed for PID {pid}: {e}")
            print("   (Requires root privileges or CAP_IPC_LOCK capability)")
            return False

    def _set_memory_limit(self, pid: int, limit_mb: int) -> bool:
        """Set memory limit for process"""
        try:
            # Use prlimit or cgroups to set memory limits
            limit_bytes = limit_mb * 1024 * 1024

            # Try using prlimit command
            import subprocess
            result = subprocess.run([
                'prlimit', '--pid', str(pid),
                '--as=soft:' + str(limit_bytes),
                '--as=hard:' + str(limit_bytes)
            ], capture_output=True)

            if result.returncode == 0:
                print(f"✅ Memory limit set to {limit_mb}MB for PID {pid}")
                return True
            else:
                # Fallback: try cgcreate/cgset if cgroups available
                self._set_cgroups_memory_limit(pid, limit_mb)
                return True

        except Exception as e:
            print(f"⚠️ Failed to set memory limit for PID {pid}: {e}")
            return False

    def _set_cgroups_memory_limit(self, pid: int, limit_mb: int) -> bool:
        """Set memory limit using cgroups"""
        try:
            import subprocess

            # Create cgroup
            cgroup_name = f"supreme_{pid}"
            subprocess.run(['cgcreate', '-g', f'memory:{cgroup_name}'],
                         capture_output=True, check=True)

            # Set memory limit
            limit_bytes = limit_mb * 1024 * 1024
            subprocess.run(['cgset', '-r', f'memory.limit_in_bytes={limit_bytes}', cgroup_name],
                         capture_output=True, check=True)

            # Add process to cgroup
            subprocess.run(['cgclassify', '-g', f'memory:{cgroup_name}', str(pid)],
                         capture_output=True, check=True)

            return True
        except Exception as e:
            return False

    def start_monitoring(self, interval_seconds: int = 5):
        """Start continuous performance monitoring"""
        if self.monitoring_active:
            return

        self.monitoring_active = True

        def monitor_loop():
            while self.monitoring_active:
                try:
                    self._monitor_processes()
                    time.sleep(interval_seconds)
                except Exception as e:
                    print(f"⚠️ Monitoring error: {e}")
                    time.sleep(10)

        self.monitor_thread = threading.Thread(target=monitor_loop, daemon=True)
        self.monitor_thread.start()

        print("✅ Process monitoring started")

    def stop_monitoring(self):
        """Stop performance monitoring"""
        self.monitoring_active = False
        if self.monitor_thread and self.monitor_thread.is_alive():
            self.monitor_thread.join(timeout=10)
        print("✅ Process monitoring stopped")

    def _monitor_processes(self):
        """Monitor all registered processes"""
        current_resources = self._get_system_resources()
        alerts = []

        for name, config in self.processes.items():
            try:
                process = psutil.Process(config.pid)

                # Check if process is still running
                if not process.is_running():
                    alerts.append(f"Process {name} (PID {config.pid}) stopped")
                    continue

                # Monitor resource usage
                cpu_percent = process.cpu_percent()
                memory_mb = process.memory_info().rss / (1024 * 1024)

                # Check thresholds
                if memory_mb > config.memory_limit_mb * 1.2:  # 20% over limit
                    alerts.append(f"Process {name} memory usage high: {memory_mb:.1f}MB")

                if cpu_percent > 90:
                    alerts.append(f"Process {name} CPU usage high: {cpu_percent:.1f}%")

                # Check for memory pressure
                if current_resources.memory_usage_percent > self.thresholds['memory_usage_percent']:
                    alerts.append(f"System memory pressure: {current_resources.memory_usage_percent:.1f}%")

            except psutil.NoSuchProcess:
                alerts.append(f"Process {name} (PID {config.pid}) no longer exists")
            except Exception as e:
                alerts.append(f"Error monitoring process {name}: {e}")

        # Store performance data
        self.performance_history.append({
            'timestamp': time.time(),
            'resources': current_resources,
            'alerts': alerts
        })

        # Keep only recent history (last 1000 entries)
        if len(self.performance_history) > 1000:
            self.performance_history = self.performance_history[-500:]

        # Trigger alerts
        for alert in alerts:
            print(f"🚨 {alert}")
            for callback in self.alert_callbacks:
                try:
                    callback(alert)
                except Exception as e:
                    print(f"⚠️ Alert callback error: {e}")

    def _get_system_resources(self) -> SystemResources:
        """Get current system resource usage"""
        memory = psutil.virtual_memory()
        cpu_percent = psutil.cpu_percent(interval=1)

        return SystemResources(
            total_memory_mb=int(memory.total / (1024 * 1024)),
            available_memory_mb=int(memory.available / (1024 * 1024)),
            cpu_count=self.cpu_cores,
            cpu_usage_percent=cpu_percent,
            memory_usage_percent=memory.percent
        )

    def get_performance_report(self) -> Dict:
        """Generate comprehensive performance report"""
        if not self.performance_history:
            return {'error': 'No performance data available'}

        # Calculate averages
        recent_data = self.performance_history[-100:]  # Last 100 measurements

        avg_cpu = sum(d['resources'].cpu_usage_percent for d in recent_data) / len(recent_data)
        avg_memory = sum(d['resources'].memory_usage_percent for d in recent_data) / len(recent_data)

        # Count alerts
        total_alerts = sum(len(d['alerts']) for d in recent_data)

        # Process-specific stats
        process_stats = {}
        for name, config in self.processes.items():
            try:
                process = psutil.Process(config.pid)
                process_stats[name] = {
                    'cpu_percent': process.cpu_percent(),
                    'memory_mb': process.memory_info().rss / (1024 * 1024),
                    'status': 'running' if process.is_running() else 'stopped'
                }
            except:
                process_stats[name] = {'status': 'error'}

        return {
            'timestamp': time.time(),
            'system': {
                'avg_cpu_percent': round(avg_cpu, 2),
                'avg_memory_percent': round(avg_memory, 2),
                'total_alerts': total_alerts,
                'monitoring_active': self.monitoring_active
            },
            'processes': process_stats,
            'recommendations': self._generate_recommendations(avg_cpu, avg_memory, total_alerts)
        }

    def _generate_recommendations(self, avg_cpu: float, avg_memory: float, alert_count: int) -> list:
        """Generate performance recommendations"""
        recommendations = []

        if avg_cpu > 80:
            recommendations.append("High CPU usage detected - consider reducing process priorities")

        if avg_memory > 85:
            recommendations.append("High memory usage - consider reducing memory limits or adding more RAM")

        if alert_count > 10:
            recommendations.append("High alert frequency - check for resource conflicts")

        if avg_cpu < 30 and avg_memory < 60:
            recommendations.append("System has spare capacity - can increase performance limits")

        return recommendations

    def add_alert_callback(self, callback: Callable):
        """Add callback for performance alerts"""
        self.alert_callbacks.append(callback)

    def emergency_restart(self, process_name: str) -> bool:
        """Emergency restart of a process if it's consuming too many resources"""
        if process_name not in self.processes:
            return False

        try:
            config = self.processes[process_name]
            process = psutil.Process(config.pid)

            # Force kill if using too many resources
            if process.memory_info().rss > config.memory_limit_mb * 1024 * 1024 * 1.5:  # 150% limit
                print(f"🚨 Emergency restart of {process_name} due to high memory usage")
                process.kill()
                return True

        except Exception as e:
            print(f"⚠️ Emergency restart failed for {process_name}: {e}")

        return False

    def optimize_system_resources(self):
        """Apply system-wide optimizations"""
        try:
            # Disable unnecessary services
            services_to_stop = [
                'bluetooth.service',
                'cups.service',
                'avahi-daemon.service'
            ]

            for service in services_to_stop:
                try:
                    import subprocess
                    subprocess.run(['sudo', 'systemctl', 'stop', service],
                                 capture_output=True)
                    subprocess.run(['sudo', 'systemctl', 'disable', service],
                                 capture_output=True)
                except:
                    pass

            # Optimize kernel parameters
            kernel_params = {
                'vm.swappiness': '1',
                'vm.dirty_ratio': '5',
                'vm.vfs_cache_pressure': '50'
            }

            for param, value in kernel_params.items():
                try:
                    with open(f'/proc/sys/{param.replace(".", "/")}', 'w') as f:
                        f.write(value)
                except:
                    pass

            print("✅ System resource optimization applied")

        except Exception as e:
            print(f"⚠️ System optimization failed: {e}")

# Global optimizer instance
process_optimizer = AdvancedProcessOptimizer()

def setup_signal_handlers():
    """Setup signal handlers for graceful shutdown"""
    def signal_handler(signum, frame):
        print(f"\n🛑 Received signal {signum}, initiating graceful shutdown...")
        process_optimizer.stop_monitoring()
        sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

# Auto-setup signal handlers
setup_signal_handlers()
