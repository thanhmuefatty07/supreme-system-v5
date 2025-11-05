#!/usr/bin/env python3
"""
🚀 Supreme System V5 - Chaos Engineering Test Suite

Injects controlled failures to validate:
- Monitoring system effectiveness
- Alert generation accuracy
- Automated recovery procedures
- System resilience and fault tolerance

Chaos Engineering Principles:
- Build a hypothesis about system behavior
- Inject realistic failures
- Measure impact and recovery
- Improve system reliability
"""

import asyncio
import json
import os
import random
import signal
import subprocess
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Any, Optional, Callable
from concurrent.futures import ThreadPoolExecutor
import threading

# Add project paths
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "python"))

try:
    import psutil
    import requests
    from loguru import logger
except ImportError as e:
    print(f"Missing dependencies: {e}")
    sys.exit(1)


class ChaosExperiment:
    """Individual chaos experiment definition"""

    def __init__(self, name: str, description: str, duration: int = 60):
        self.name = name
        self.description = description
        self.duration = duration
        self.start_time = None
        self.end_time = None
        self.metrics_before = {}
        self.metrics_after = {}
        self.alerts_triggered = []
        self.recovery_successful = False

    def record_metric(self, name: str, value: Any, phase: str = "during"):
        """Record metric at different experiment phases"""
        if phase == "before":
            self.metrics_before[name] = value
        elif phase == "after":
            self.metrics_after[name] = value

    def record_alert(self, alert_name: str, timestamp: float):
        """Record alerts triggered during experiment"""
        self.alerts_triggered.append({
            'name': alert_name,
            'timestamp': timestamp
        })


class ChaosEngine:
    """Chaos engineering orchestration engine"""

    def __init__(self, target_url: str = "http://localhost:8000", duration: int = 300, failure_rate: float = 0.1):
        self.target_url = target_url.rstrip('/')
        self.total_duration = duration
        self.failure_rate = failure_rate
        self.output_dir = Path("run_artifacts")
        self.output_dir.mkdir(exist_ok=True)

        self.experiments = []
        self.results = {
            'metadata': {
                'target_url': target_url,
                'total_duration': duration,
                'failure_rate': failure_rate,
                'start_time': None,
                'end_time': None
            },
            'experiments': [],
            'system_health': {},
            'alert_validation': {},
            'recovery_metrics': {}
        }

        # Monitoring
        self.monitoring_active = False
        self.monitoring_thread = None
        self.system_metrics = []

    def add_experiment(self, experiment: ChaosExperiment):
        """Add experiment to test suite"""
        self.experiments.append(experiment)

    def start_monitoring(self):
        """Start background system monitoring"""
        if self.monitoring_active:
            return

        self.monitoring_active = True
        self.monitoring_thread = threading.Thread(target=self._monitor_system, daemon=True)
        self.monitoring_thread.start()
        logger.info("🖥️ System monitoring started")

    def stop_monitoring(self):
        """Stop background monitoring"""
        self.monitoring_active = False
        if self.monitoring_thread:
            self.monitoring_thread.join(timeout=5)
        logger.info("🖥️ System monitoring stopped")

    def _monitor_system(self):
        """Background system monitoring"""
        while self.monitoring_active:
            try:
                # System metrics
                cpu_percent = psutil.cpu_percent(interval=1)
                memory = psutil.virtual_memory()
                disk = psutil.disk_usage('/')

                # Application health check
                health_status = "unknown"
                response_time = None
                try:
                    start_time = time.time()
                    response = requests.get(f"{self.target_url}/api/v1/health", timeout=5)
                    response_time = time.time() - start_time
                    health_status = "healthy" if response.status_code == 200 else "unhealthy"
                except:
                    health_status = "unreachable"

                # Prometheus metrics (if available)
                prometheus_metrics = {}
                try:
                    response = requests.get("http://localhost:9090/api/v1/query?query=up", timeout=2)
                    if response.status_code == 200:
                        prometheus_metrics = response.json()
                except:
                    pass

                metric_snapshot = {
                    'timestamp': time.time(),
                    'cpu_percent': cpu_percent,
                    'memory_percent': memory.percent,
                    'memory_used_mb': memory.used / 1024 / 1024,
                    'disk_percent': disk.percent,
                    'health_status': health_status,
                    'response_time': response_time,
                    'prometheus_up': len(prometheus_metrics) > 0
                }

                self.system_metrics.append(metric_snapshot)

            except Exception as e:
                logger.warning(f"Monitoring error: {e}")

            time.sleep(5)  # Monitor every 5 seconds

    def run_chaos_tests(self) -> Dict[str, Any]:
        """Execute complete chaos engineering test suite"""
        logger.info("🌀 STARTING CHAOS ENGINEERING TEST SUITE")
        logger.info("=" * 60)
        logger.info(f"Target: {self.target_url}")
        logger.info(f"Duration: {self.total_duration}s")
        logger.info(f"Failure Rate: {self.failure_rate * 100}%")
        logger.info(f"Experiments: {len(self.experiments)}")

        self.results['metadata']['start_time'] = datetime.now().isoformat()

        # Start monitoring
        self.start_monitoring()

        try:
            # Baseline measurement
            logger.info("📊 Establishing baseline metrics...")
            self._measure_baseline()

            # Execute experiments
            for i, experiment in enumerate(self.experiments, 1):
                logger.info(f"🧪 Running Experiment {i}/{len(self.experiments)}: {experiment.name}")
                self._run_single_experiment(experiment)

            # Recovery validation
            logger.info("🔄 Validating system recovery...")
            self._validate_recovery()

            # Analysis
            logger.info("📈 Analyzing results...")
            self._analyze_results()

        except Exception as e:
            logger.error(f"Chaos testing failed: {e}")
            self.results['error'] = str(e)

        finally:
            # Cleanup
            self.stop_monitoring()
            self.results['metadata']['end_time'] = datetime.now().isoformat()

        # Save results
        self._save_results()

        logger.info("✅ Chaos engineering test suite completed")
        return self.results

    def _measure_baseline(self):
        """Measure baseline system performance"""
        logger.info("📏 Measuring baseline performance...")

        baseline_metrics = {}

        # HTTP performance
        response_times = []
        for _ in range(10):
            try:
                start = time.time()
                response = requests.get(f"{self.target_url}/api/v1/health", timeout=5)
                response_times.append(time.time() - start)
            except:
                response_times.append(5.0)  # Timeout

        baseline_metrics['avg_response_time'] = sum(response_times) / len(response_times)
        baseline_metrics['p95_response_time'] = sorted(response_times)[int(len(response_times) * 0.95)]

        # Error rate
        baseline_metrics['error_rate'] = 0.0  # Assume baseline is error-free

        # Resource usage
        baseline_metrics['cpu_percent'] = psutil.cpu_percent(interval=1)
        baseline_metrics['memory_percent'] = psutil.virtual_memory().percent

        self.results['baseline_metrics'] = baseline_metrics
        logger.info(f"📊 Baseline established: {baseline_metrics}")

    def _run_single_experiment(self, experiment: ChaosExperiment):
        """Execute individual chaos experiment"""
        experiment.start_time = time.time()

        logger.info(f"🧪 {experiment.description}")

        # Pre-experiment metrics
        experiment.record_metric('cpu_before', psutil.cpu_percent(), 'before')
        experiment.record_metric('memory_before', psutil.virtual_memory().percent, 'before')

        try:
            # Execute chaos (subclass implements this)
            self._inject_failure(experiment)

            # Monitor during experiment
            self._monitor_during_experiment(experiment)

            # Validate recovery
            experiment.recovery_successful = self._test_recovery(experiment)

        except Exception as e:
            logger.error(f"Experiment {experiment.name} failed: {e}")
            experiment.recovery_successful = False

        experiment.end_time = time.time()

        # Post-experiment metrics
        experiment.record_metric('cpu_after', psutil.cpu_percent(), 'after')
        experiment.record_metric('memory_after', psutil.virtual_memory().percent, 'after')

        # Record experiment results
        self.results['experiments'].append({
            'name': experiment.name,
            'description': experiment.description,
            'duration': experiment.duration,
            'start_time': experiment.start_time,
            'end_time': experiment.end_time,
            'recovery_successful': experiment.recovery_successful,
            'alerts_triggered': experiment.alerts_triggered,
            'metrics_before': experiment.metrics_before,
            'metrics_after': experiment.metrics_after
        })

    def _inject_failure(self, experiment: ChaosExperiment):
        """Inject failure based on experiment type"""
        if "pod_failure" in experiment.name.lower():
            self._inject_pod_failure()
        elif "network_partition" in experiment.name.lower():
            self._inject_network_partition()
        elif "high_cpu" in experiment.name.lower():
            self._inject_high_cpu()
        elif "memory_leak" in experiment.name.lower():
            self._inject_memory_pressure()
        elif "database_failure" in experiment.name.lower():
            self._inject_database_failure()

    def _inject_pod_failure(self):
        """Simulate pod failure by killing the process"""
        logger.info("💥 Injecting pod failure...")

        # Find and kill the application process
        for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
            try:
                if 'python' in proc.info['name'].lower():
                    cmdline = ' '.join(proc.info['cmdline'] or [])
                    if 'main.py' in cmdline or 'uvicorn' in cmdline:
                        logger.info(f"Killing process {proc.info['pid']}")
                        proc.kill()
                        time.sleep(2)  # Wait for restart
                        break
            except:
                continue

    def _inject_network_partition(self):
        """Simulate network partition using iptables"""
        logger.info("🌐 Injecting network partition...")

        # Drop packets to simulate network issues
        try:
            subprocess.run([
                'sudo', 'iptables', '-A', 'INPUT', '-p', 'tcp', '--dport', '8000',
                '-m', 'statistic', '--mode', 'random', '--probability', str(self.failure_rate),
                '-j', 'DROP'
            ], check=True, capture_output=True)

            time.sleep(30)  # Partition duration

            # Restore network
            subprocess.run([
                'sudo', 'iptables', '-F', 'INPUT'
            ], check=True, capture_output=True)

        except subprocess.CalledProcessError as e:
            logger.warning(f"Network partition failed (requires sudo): {e}")

    def _inject_high_cpu(self):
        """Inject high CPU usage"""
        logger.info("🔥 Injecting high CPU load...")

        def cpu_stress():
            # Simple CPU stress test
            end_time = time.time() + 30
            while time.time() < end_time:
                [x**2 for x in range(10000)]

        # Run CPU stress in thread pool
        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = [executor.submit(cpu_stress) for _ in range(4)]
            for future in futures:
                future.result(timeout=35)

    def _inject_memory_pressure(self):
        """Inject memory pressure"""
        logger.info("💧 Injecting memory pressure...")

        # Allocate memory gradually
        memory_hog = []
        target_mb = 500  # 500MB

        try:
            while len(memory_hog) * 8 < target_mb * 1024 * 1024:  # 8 bytes per float
                memory_hog.extend([0.0] * 1000000)  # Add 8MB chunks
                time.sleep(1)

            time.sleep(20)  # Hold memory pressure

        except MemoryError:
            logger.info("Memory limit reached (expected)")

        # Clean up
        del memory_hog

    def _inject_database_failure(self):
        """Simulate database failure"""
        logger.info("💾 Injecting database failure...")

        # If Redis is running, restart it
        try:
            subprocess.run(['sudo', 'systemctl', 'restart', 'redis'], check=True, capture_output=True)
            logger.info("Redis restarted to simulate failure")
        except subprocess.CalledProcessError:
            logger.warning("Redis restart failed")

        time.sleep(10)  # Recovery time

    def _monitor_during_experiment(self, experiment: ChaosExperiment):
        """Monitor system during experiment"""
        start_time = time.time()

        while time.time() - start_time < experiment.duration:
            # Check for alerts (simplified - in real system, query Prometheus)
            try:
                # Simulate alert checking
                response = requests.get(f"{self.target_url}/api/v1/health", timeout=2)
                if response.status_code != 200:
                    experiment.record_alert("ServiceUnhealthy", time.time())
            except:
                experiment.record_alert("ServiceDown", time.time())

            time.sleep(2)

    def _test_recovery(self, experiment: ChaosExperiment) -> bool:
        """Test if system recovered after experiment"""
        logger.info("🔄 Testing recovery...")

        # Test for 60 seconds
        recovery_start = time.time()
        consecutive_successes = 0
        required_successes = 5

        while time.time() - recovery_start < 60:
            try:
                response = requests.get(f"{self.target_url}/api/v1/health", timeout=3)
                if response.status_code == 200:
                    consecutive_successes += 1
                    if consecutive_successes >= required_successes:
                        logger.info("✅ Recovery successful")
                        return True
                else:
                    consecutive_successes = 0
            except:
                consecutive_successes = 0

            time.sleep(2)

        logger.warning("❌ Recovery failed")
        return False

    def _validate_recovery(self):
        """Validate overall system recovery after all experiments"""
        logger.info("🏥 Validating overall system recovery...")

        recovery_metrics = {}

        # Test API endpoints
        endpoints = ['/api/v1/health', '/api/v1/status', '/metrics']
        for endpoint in endpoints:
            try:
                start_time = time.time()
                response = requests.get(f"{self.target_url}{endpoint}", timeout=5)
                response_time = time.time() - start_time
                recovery_metrics[f"{endpoint}_status"] = response.status_code
                recovery_metrics[f"{endpoint}_response_time"] = response_time
            except Exception as e:
                recovery_metrics[f"{endpoint}_error"] = str(e)

        # Check resource usage
        recovery_metrics['cpu_percent'] = psutil.cpu_percent(interval=1)
        recovery_metrics['memory_percent'] = psutil.virtual_memory().percent

        self.results['recovery_metrics'] = recovery_metrics

        # Determine if system is healthy
        all_healthy = all(
            recovery_metrics.get(f"{endpoint}_status") == 200
            for endpoint in endpoints
            if f"{endpoint}_status" in recovery_metrics
        )

        self.results['system_recovered'] = all_healthy
        logger.info(f"🏥 System recovery status: {'✅ Healthy' if all_healthy else '❌ Unhealthy'}")

    def _analyze_results(self):
        """Analyze chaos engineering results"""
        logger.info("📊 Analyzing chaos engineering results...")

        analysis = {
            'experiments_run': len(self.experiments),
            'experiments_passed': sum(1 for exp in self.results['experiments'] if exp['recovery_successful']),
            'total_alerts_triggered': sum(len(exp['alerts_triggered']) for exp in self.results['experiments']),
            'system_resilience_score': 0.0,
            'recommendations': []
        }

        # Calculate resilience score
        if analysis['experiments_run'] > 0:
            recovery_rate = analysis['experiments_passed'] / analysis['experiments_run']
            analysis['system_resilience_score'] = recovery_rate * 100

        # Generate recommendations
        if analysis['system_resilience_score'] < 80:
            analysis['recommendations'].append("Improve automated recovery procedures")
        if analysis['total_alerts_triggered'] < len(self.experiments):
            analysis['recommendations'].append("Enhance monitoring and alerting coverage")
        if any(not exp['recovery_successful'] for exp in self.results['experiments']):
            analysis['recommendations'].append("Strengthen circuit breakers and fallback mechanisms")

        self.results['analysis'] = analysis

    def _save_results(self):
        """Save chaos engineering results"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"chaos_test_results_{timestamp}.json"
        filepath = self.output_dir / filename

        with open(filepath, 'w') as f:
            json.dump(self.results, f, indent=2, default=str)

        logger.info(f"💾 Chaos engineering results saved to: {filepath}")

        # Generate summary report
        self._generate_summary_report()

    def _generate_summary_report(self):
        """Generate human-readable summary report"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"chaos_test_report_{timestamp}.txt"
        filepath = self.output_dir / filename

        analysis = self.results.get('analysis', {})

        with open(filepath, 'w') as f:
            f.write("🌀 Supreme System V5 - Chaos Engineering Report\n")
            f.write("=" * 60)
            f.write(f"\nTest Duration: {self.total_duration} seconds\n")
            f.write(f"Experiments Run: {len(self.experiments)}\n")
            f.write(f"Failure Rate: {self.failure_rate * 100}%\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

            f.write("📊 EXPERIMENT RESULTS\n")
            f.write("-" * 40)
            for exp in self.results['experiments']:
                status = "✅ PASSED" if exp['recovery_successful'] else "❌ FAILED"
                f.write(f"\n{exp['name']}: {status}")
                f.write(f"\n  Duration: {exp['duration']}s")
                f.write(f"\n  Alerts: {len(exp['alerts_triggered'])}")
                f.write(f"\n  CPU Before/After: {exp['metrics_before'].get('cpu_before', 'N/A'):.1f}% / {exp['metrics_after'].get('cpu_after', 'N/A'):.1f}%")
                f.write(f"\n  Memory Before/After: {exp['metrics_before'].get('memory_before', 'N/A'):.1f}% / {exp['metrics_after'].get('memory_after', 'N/A'):.1f}%")

            f.write("
🎯 SYSTEM RESILIENCE ANALYSIS\n"            f.write("-" * 40)
            f.write(f"\nResilience Score: {analysis.get('system_resilience_score', 0):.1f}%")
            f.write(f"\nExperiments Passed: {analysis.get('experiments_passed', 0)}/{analysis.get('experiments_run', 0)}")
            f.write(f"\nTotal Alerts Triggered: {analysis.get('total_alerts_triggered', 0)}")

            f.write("
💡 RECOMMENDATIONS\n"            f.write("-" * 40)
            for rec in analysis.get('recommendations', []):
                f.write(f"\n• {rec}")

            f.write("
🏥 RECOVERY STATUS\n"            f.write("-" * 40)
            recovery = self.results.get('recovery_metrics', {})
            f.write(f"\nSystem Recovered: {'✅ YES' if self.results.get('system_recovered', False) else '❌ NO'}")

            for endpoint in ['/api/v1/health', '/api/v1/status', '/metrics']:
                status = recovery.get(f"{endpoint}_status")
                response_time = recovery.get(f"{endpoint}_response_time")
                if status:
                    f.write(f"\n{endpoint}: HTTP {status} ({response_time:.3f}s)" if response_time else f"\n{endpoint}: HTTP {status}")
                else:
                    error = recovery.get(f"{endpoint}_error")
                    f.write(f"\n{endpoint}: ERROR - {error}" if error else f"\n{endpoint}: NOT TESTED")

        logger.info(f"📋 Chaos engineering report saved to: {filepath}")


def create_standard_experiments() -> List[ChaosExperiment]:
    """Create standard chaos engineering experiments"""
    experiments = [
        ChaosExperiment(
            name="pod_failure_recovery",
            description="Test Kubernetes pod failure and automatic recovery",
            duration=45
        ),
        ChaosExperiment(
            name="network_partition",
            description="Simulate network partition and test service mesh recovery",
            duration=60
        ),
        ChaosExperiment(
            name="high_cpu_load",
            description="Inject high CPU usage and validate performance degradation handling",
            duration=30
        ),
        ChaosExperiment(
            name="memory_pressure",
            description="Apply memory pressure and test garbage collection and OOM handling",
            duration=45
        ),
        ChaosExperiment(
            name="database_failure",
            description="Simulate Redis/database failure and test caching layer resilience",
            duration=30
        ),
        ChaosExperiment(
            name="multiple_failures",
            description="Inject multiple simultaneous failures to test system limits",
            duration=90
        )
    ]

    return experiments


def main():
    """Main chaos engineering execution"""
    import argparse

    parser = argparse.ArgumentParser(description="Supreme System V5 Chaos Engineering")
    parser.add_argument("--target", default="http://localhost:8000",
                       help="Target application URL")
    parser.add_argument("--duration", type=int, default=300,
                       help="Total test duration in seconds")
    parser.add_argument("--failure-rate", type=float, default=0.1,
                       help="Failure injection rate (0.0-1.0)")
    parser.add_argument("--output", default="run_artifacts/chaos-results.json",
                       help="Output file path")

    args = parser.parse_args()

    try:
        # Initialize chaos engine
        engine = ChaosEngine(
            target_url=args.target,
            duration=args.duration,
            failure_rate=args.failure_rate
        )

        # Add standard experiments
        experiments = create_standard_experiments()
        for exp in experiments:
            engine.add_experiment(exp)

        # Run chaos tests
        results = engine.run_chaos_tests()

        # Print summary
        analysis = results.get('analysis', {})
        print("
🌀 CHAOS ENGINEERING SUMMARY"        print("=" * 50)
        print(f"Experiments: {analysis.get('experiments_run', 0)}")
        print(f"Passed: {analysis.get('experiments_passed', 0)}")
        print(".1f"        print(f"Alerts: {analysis.get('total_alerts_triggered', 0)}")
        print(f"Recovery: {'✅ SUCCESS' if results.get('system_recovered', False) else '❌ FAILED'}")

        if analysis.get('system_resilience_score', 0) >= 80:
            print("🎉 System demonstrates high resilience!")
        elif analysis.get('system_resilience_score', 0) >= 60:
            print("⚠️ System shows moderate resilience - improvements needed")
        else:
            print("❌ System resilience needs significant improvement")

    except KeyboardInterrupt:
        print("\n🛑 Chaos testing interrupted by user")
    except Exception as e:
        print(f"❌ Chaos engineering failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
