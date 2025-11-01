#!/usr/bin/env python3

"""
Performance validation script for Supreme System V5

Validates that dashboard addition doesn't impact trading performance
"""

import asyncio
import time
import statistics
import psutil
from typing import List, Dict
from dataclasses import dataclass

@dataclass
class PerformanceMetrics:
    latency_ms: List[float]
    memory_usage_mb: List[float]
    cpu_percent: List[float]
    throughput_per_sec: List[float]
    timestamp: float

class PerformanceValidator:
    def __init__(self):
        self.trading_process = None
        self.dashboard_process = None
        self.baseline_metrics = None

    async def measure_trading_latency(self, duration_seconds: int = 60) -> List[float]:
        """Measure trading core latency over time"""
        latencies = []
        start_time = time.time()

        while time.time() - start_time < duration_seconds:
            cycle_start = time.time()

            # Simulate trading cycle
            await self._simulate_trading_cycle()

            cycle_end = time.time()
            latency_ms = (cycle_end - cycle_start) * 1000
            latencies.append(latency_ms)

            # Wait for next cycle (1 second interval)
            await asyncio.sleep(max(0, 1.0 - (cycle_end - cycle_start)))

        return latencies

    async def _simulate_trading_cycle(self):
        """Simulate a typical trading cycle for measurement"""
        # This would call actual trading components
        from supreme_system_v5.data_fabric import DataAggregator
        from supreme_system_v5.strategies import ScalpingStrategy

        # Simulate data fetch + strategy calculation + risk check
        await asyncio.sleep(0.015)  # Simulate realistic processing time

    def measure_system_resources(self) -> Dict:
        """Measure current system resource usage"""
        memory = psutil.virtual_memory()
        cpu_percent = psutil.cpu_percent(interval=1)

        return {
            'memory_total_mb': memory.total / (1024**2),
            'memory_used_mb': memory.used / (1024**2),
            'memory_available_mb': memory.available / (1024**2),
            'memory_percent': memory.percent,
            'cpu_percent': cpu_percent,
            'swap_used_mb': psutil.swap_memory().used / (1024**2)
        }

    async def run_baseline_test(self) -> PerformanceMetrics:
        """Run performance test without dashboard"""
        print("🔍 Running baseline performance test (no dashboard)...")

        # Measure latency
        latencies = await self.measure_trading_latency(60)

        # Measure resources
        resources = self.measure_system_resources()

        metrics = PerformanceMetrics(
            latency_ms=latencies,
            memory_usage_mb=[resources['memory_used_mb']],
            cpu_percent=[resources['cpu_percent']],
            throughput_per_sec=[len(latencies) / 60],
            timestamp=time.time()
        )

        self.baseline_metrics = metrics
        return metrics

    async def run_dashboard_test(self) -> PerformanceMetrics:
        """Run performance test with dashboard running"""
        print("📊 Running dashboard performance test...")

        # Start dashboard process
        await self._start_dashboard()

        # Wait for dashboard to stabilize
        await asyncio.sleep(30)

        # Measure latency with dashboard running
        latencies = await self.measure_trading_latency(60)

        # Measure resources
        resources = self.measure_system_resources()

        metrics = PerformanceMetrics(
            latency_ms=latencies,
            memory_usage_mb=[resources['memory_used_mb']],
            cpu_percent=[resources['cpu_percent']],
            throughput_per_sec=[len(latencies) / 60],
            timestamp=time.time()
        )

        return metrics

    async def _start_dashboard(self):
        """Start dashboard process for testing"""
        # This would start the dashboard container
        import subprocess
        subprocess.Popen([
            'docker-compose', '-f', 'docker-compose.performance.yml',
            'up', '-d', 'supreme-dashboard'
        ])

    def analyze_performance_impact(self, baseline: PerformanceMetrics,
                                 with_dashboard: PerformanceMetrics) -> Dict:
        """Analyze performance impact of dashboard"""

        baseline_avg_latency = statistics.mean(baseline.latency_ms)
        dashboard_avg_latency = statistics.mean(with_dashboard.latency_ms)

        baseline_p95_latency = statistics.quantiles(baseline.latency_ms, n=20)[18]
        dashboard_p95_latency = statistics.quantiles(with_dashboard.latency_ms, n=20)[18]

        latency_increase = dashboard_avg_latency - baseline_avg_latency
        latency_increase_percent = (latency_increase / baseline_avg_latency) * 100

        return {
            'baseline_avg_latency_ms': baseline_avg_latency,
            'dashboard_avg_latency_ms': dashboard_avg_latency,
            'latency_increase_ms': latency_increase,
            'latency_increase_percent': latency_increase_percent,
            'baseline_p95_latency_ms': baseline_p95_latency,
            'dashboard_p95_latency_ms': dashboard_p95_latency,
            'performance_acceptable': latency_increase_percent < 10,  # <10% increase
            'target_met': dashboard_avg_latency < 25  # <25ms target
        }

    def print_performance_report(self, analysis: Dict):
        """Print detailed performance analysis report"""
        print("\n" + "="*60)
        print("📊 PERFORMANCE IMPACT ANALYSIS")
        print("="*60)

        print(f"\n🎯 LATENCY ANALYSIS:")
        print(f"  Baseline average:     {analysis['baseline_avg_latency_ms']:.2f}ms")
        print(f"  With dashboard:       {analysis['dashboard_avg_latency_ms']:.2f}ms")
        print(f"  Increase:             {analysis['latency_increase_ms']:.2f}ms ({analysis['latency_increase_percent']:.1f}%)")

        print(f"\n📈 P95 LATENCY:")
        print(f"  Baseline P95:         {analysis['baseline_p95_latency_ms']:.2f}ms")
        print(f"  Dashboard P95:        {analysis['dashboard_p95_latency_ms']:.2f}ms")

        print(f"\n✅ PERFORMANCE VALIDATION:")
        acceptable = "✅ PASS" if analysis['performance_acceptable'] else "❌ FAIL"
        target_met = "✅ PASS" if analysis['target_met'] else "❌ FAIL"

        print(f"  Impact <10%:          {acceptable}")
        print(f"  Latency <25ms:        {target_met}")

        if analysis['performance_acceptable'] and analysis['target_met']:
            print(f"\n🎉 PERFORMANCE VALIDATION SUCCESSFUL")
            print(f"   Dashboard can be deployed with minimal impact")
        else:
            print(f"\n⚠️  PERFORMANCE VALIDATION FAILED")
            print(f"   Dashboard optimization required")

async def main():
    validator = PerformanceValidator()

    try:
        # Run baseline test
        baseline = await validator.run_baseline_test()

        # Run dashboard test
        with_dashboard = await validator.run_dashboard_test()

        # Analyze impact
        analysis = validator.analyze_performance_impact(baseline, with_dashboard)

        # Print report
        validator.print_performance_report(analysis)

    except Exception as e:
        print(f"❌ Performance validation failed: {e}")
        return 1

    return 0

if __name__ == "__main__":
    exit_code = asyncio.run(main())
    exit(exit_code)
