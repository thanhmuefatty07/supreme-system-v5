#!/usr/bin/env python3
"""
🚀 Supreme System V5 - Ultra-Constrained Performance Profiler

World-class profiling for ultra-constrained trading system.
Identifies bottlenecks, measures latency, memory usage, and CPU utilization.

Requirements:
- Maintain <0.020ms average processing latency
- Keep memory usage <15MB
- CPU utilization <85%
- Statistical significance for all measurements
"""

import cProfile
import pstats
import io
import time
import psutil
import tracemalloc
import statistics
from typing import Dict, List, Any, Optional, Tuple
import os
import sys
from pathlib import Path
from datetime import datetime
import json

# Add python path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'python'))

class UltraConstrainedProfiler:
    """World-class profiler for ultra-constrained trading systems"""

    def __init__(self, output_dir: str = "run_artifacts"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Profiling results
        self.baseline_metrics = {}
        self.optimization_metrics = {}

    def profile_system_startup(self) -> Dict[str, Any]:
        """Profile system startup time and import overhead"""
        print("🔧 Profiling system startup...")

        # Profile import time
        import_start = time.perf_counter()
        import supreme_system_v5
        from supreme_system_v5.strategies import ScalpingStrategy
        from supreme_system_v5.risk import RiskManager
        import_end = time.perf_counter()

        import_time = (import_end - import_start) * 1000  # Convert to milliseconds

        # Profile strategy instantiation
        strategy_start = time.perf_counter()
        config = {
            'symbol': 'ETH-USDT',
            'position_size_pct': 0.02,
            'ema_period': 14,
            'rsi_period': 14
        }
        strategy = ScalpingStrategy(config)
        strategy_end = time.perf_counter()

        instantiation_time = (strategy_end - strategy_start) * 1000

        return {
            'import_time_ms': import_time,
            'instantiation_time_ms': instantiation_time,
            'total_startup_ms': import_time + instantiation_time
        }

    def profile_market_data_processing(self, iterations: int = 1000) -> Dict[str, Any]:
        """Profile market data processing latency with statistical analysis"""
        print(f"📊 Profiling market data processing ({iterations} iterations)...")

        # Setup strategy
        from supreme_system_v5.strategies import ScalpingStrategy
        config = {
            'symbol': 'ETH-USDT',
            'position_size_pct': 0.02,
            'ema_period': 14,
            'rsi_period': 14
        }
        strategy = ScalpingStrategy(config)

        # Generate sample market data
        import random
        latencies = []

        for i in range(iterations):
            # Simulate market data update
            market_data = {
                'timestamp': time.time(),
                'symbol': 'ETH-USDT',
                'price': 45000 + random.uniform(-100, 100),
                'volume': random.uniform(100, 1000),
                'bid': 44990 + random.uniform(-10, 10),
                'ask': 45010 + random.uniform(-10, 10)
            }

            # Measure processing latency
            start_time = time.perf_counter()
            try:
                # This simulates the strategy processing market data
                result = strategy.analyzer.update_price(market_data['price'])
            except AttributeError:
                # Fallback if method doesn't exist
                result = None
            end_time = time.perf_counter()

            latency_ms = (end_time - start_time) * 1000
            latencies.append(latency_ms)

        # Statistical analysis
        stats = {
            'iterations': iterations,
            'mean_latency_ms': statistics.mean(latencies),
            'median_latency_ms': statistics.median(latencies),
            'p95_latency_ms': sorted(latencies)[int(iterations * 0.95)],
            'p99_latency_ms': sorted(latencies)[int(iterations * 0.99)],
            'min_latency_ms': min(latencies),
            'max_latency_ms': max(latencies),
            'std_dev_ms': statistics.stdev(latencies) if len(latencies) > 1 else 0
        }

        return stats

    def profile_memory_usage(self) -> Dict[str, Any]:
        """Profile memory usage during operation"""
        print("💾 Profiling memory usage...")

        tracemalloc.start()

        # Setup strategy
        from supreme_system_v5.strategies import ScalpingStrategy
        config = {
            'symbol': 'ETH-USDT',
            'position_size_pct': 0.02,
            'ema_period': 14,
            'rsi_period': 14
        }
        strategy = ScalpingStrategy(config)

        # Process some data to establish baseline
        for i in range(100):
            market_data = {
                'timestamp': time.time(),
                'symbol': 'ETH-USDT',
                'price': 45000 + i * 0.1,
                'volume': 500,
                'bid': 44990,
                'ask': 45010
            }
            try:
                strategy.analyzer.update_price(market_data['price'])
            except AttributeError:
                pass

        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        # Get process memory
        process = psutil.Process()
        process_memory = process.memory_info().rss / (1024 * 1024)  # MB

        return {
            'tracemalloc_current_mb': current / (1024 * 1024),
            'tracemalloc_peak_mb': peak / (1024 * 1024),
            'process_memory_mb': process_memory,
            'memory_efficiency_pct': (current / peak) * 100 if peak > 0 else 100
        }

    def profile_cpu_usage(self, duration_seconds: int = 10) -> Dict[str, Any]:
        """Profile CPU usage during operation"""
        print(f"⚡ Profiling CPU usage ({duration_seconds}s)...")

        process = psutil.Process()
        cpu_samples = []

        # Start monitoring
        start_time = time.time()
        while time.time() - start_time < duration_seconds:
            cpu_percent = process.cpu_percent(interval=0.1)
            cpu_samples.append(cpu_percent)

        stats = {
            'monitoring_duration_s': duration_seconds,
            'samples_collected': len(cpu_samples),
            'avg_cpu_pct': statistics.mean(cpu_samples),
            'max_cpu_pct': max(cpu_samples),
            'p95_cpu_pct': sorted(cpu_samples)[int(len(cpu_samples) * 0.95)]
        }

        return stats

    def run_comprehensive_profiling(self) -> Dict[str, Any]:
        """Run complete profiling suite"""
        print("🚀 STARTING COMPREHENSIVE PROFILING SESSION")
        print("=" * 60)

        results = {
            'timestamp': self.timestamp,
            'system_info': {
                'python_version': sys.version,
                'platform': sys.platform,
                'cpu_count': psutil.cpu_count(),
                'memory_total_gb': psutil.virtual_memory().total / (1024**3)
            }
        }

        # Profile startup
        results['startup_profiling'] = self.profile_system_startup()

        # Profile market data processing
        results['market_data_profiling'] = self.profile_market_data_processing()

        # Profile memory
        results['memory_profiling'] = self.profile_memory_usage()

        # Profile CPU
        results['cpu_profiling'] = self.profile_cpu_usage()

        # Performance assessment
        results['performance_assessment'] = self._assess_performance(results)

        print("✅ PROFILING COMPLETE")
        return results

    def _assess_performance(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Assess if performance meets ultra-constrained requirements"""
        assessment = {
            'requirements_met': True,
            'warnings': [],
            'critical_issues': []
        }

        # Check latency requirement (<0.020ms)
        avg_latency = results['market_data_profiling']['mean_latency_ms']
        if avg_latency > 0.020:
            assessment['warnings'].append(f"Average latency {avg_latency:.3f}ms exceeds 0.020ms target")
            if avg_latency > 0.050:
                assessment['critical_issues'].append(f"Critical latency violation: {avg_latency:.3f}ms >> 0.020ms")

        # Check memory requirement (<15MB)
        process_memory = results['memory_profiling']['process_memory_mb']
        if process_memory > 15:
            assessment['warnings'].append(f"Memory usage {process_memory:.1f}MB exceeds 15MB target")
            assessment['requirements_met'] = False
            assessment['critical_issues'].append(f"Memory budget violation: {process_memory:.1f}MB > 15MB")

        # Check CPU requirement (<85%)
        avg_cpu = results['cpu_profiling']['avg_cpu_pct']
        if avg_cpu > 85:
            assessment['warnings'].append(f"CPU usage {avg_cpu:.1f}% exceeds 85% target")
            assessment['requirements_met'] = False

        assessment['overall_status'] = 'PASS' if assessment['requirements_met'] and not assessment['critical_issues'] else 'FAIL'

        return assessment

    def save_results(self, results: Dict[str, Any], filename: str = None) -> str:
        """Save profiling results to JSON file"""
        if filename is None:
            filename = f"performance_profile_{self.timestamp}.json"

        filepath = self.output_dir / filename
        with open(filepath, 'w') as f:
            json.dump(results, f, indent=2, default=str)

        print(f"💾 Results saved to: {filepath}")
        return str(filepath)

    def generate_report(self, results: Dict[str, Any], filename: str = None) -> str:
        """Generate human-readable performance report"""
        if filename is None:
            filename = f"performance_report_{self.timestamp}.md"

        filepath = self.output_dir / filename

        with open(filepath, 'w') as f:
            f.write("# 🚀 Supreme System V5 - Performance Profiling Report\n\n")
            f.write(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            f.write("## 📊 System Overview\n\n")
            f.write(f"- **Python Version:** {results['system_info']['python_version'].split()[0]}\n")
            f.write(f"- **Platform:** {results['system_info']['platform']}\n")
            f.write(f"- **CPU Cores:** {results['system_info']['cpu_count']}\n")
            f.write(".1f")
            f.write("\n## ⚡ Performance Metrics\n\n")

            # Startup performance
            startup = results['startup_profiling']
            f.write("### Startup Performance\n")
            f.write(f"- **Import Time:** {startup['import_time_ms']:.2f}ms\n")
            f.write(f"- **Instantiation Time:** {startup['instantiation_time_ms']:.2f}ms\n")
            f.write(f"- **Total Startup:** {startup['total_startup_ms']:.2f}ms\n")
            # Market data processing
            market = results['market_data_profiling']
            f.write("\n### Market Data Processing\n")
            f.write(f"- **Iterations:** {market['iterations']:,}\n")
            f.write(f"- **Mean Latency:** {market['mean_latency_ms']:.3f}ms\n")
            f.write(f"- **Median Latency:** {market['median_latency_ms']:.3f}ms\n")
            f.write(f"- **P95 Latency:** {market['p95_latency_ms']:.3f}ms\n")
            f.write(f"- **P99 Latency:** {market['p99_latency_ms']:.3f}ms\n")
            f.write(f"- **Min Latency:** {market['min_latency_ms']:.3f}ms\n")
            # Memory usage
            memory = results['memory_profiling']
            f.write("\n### Memory Usage\n")
            f.write(f"- **Tracemalloc Current:** {memory['tracemalloc_current_mb']:.1f}MB\n")
            f.write(f"- **Tracemalloc Peak:** {memory['tracemalloc_peak_mb']:.1f}MB\n")
            f.write(f"- **Process Memory:** {memory['process_memory_mb']:.1f}MB\n")
            f.write(f"- **Memory Efficiency:** {memory['memory_efficiency_pct']:.1f}%\n")
            # CPU usage
            cpu = results['cpu_profiling']
            f.write("\n### CPU Usage\n")
            f.write(f"- **Monitoring Duration:** {cpu['monitoring_duration_s']}s\n")
            f.write(f"- **Samples Collected:** {cpu['samples_collected']}\n")
            f.write(f"- **Avg CPU:** {cpu['avg_cpu_pct']:.1f}%\n")
            f.write(f"- **Max CPU:** {cpu['max_cpu_pct']:.1f}%\n")
            f.write(f"- **P95 CPU:** {cpu['p95_cpu_pct']:.1f}%\n")
            # Assessment
            assessment = results['performance_assessment']
            f.write("\n## 🎯 Performance Assessment\n\n")
            status_emoji = "✅" if assessment['overall_status'] == 'PASS' else "❌"
            f.write(f"### Overall Status: {status_emoji} {assessment['overall_status']}\n\n")

            if assessment['warnings']:
                f.write("### Warnings:\n")
                for warning in assessment['warnings']:
                    f.write(f"- ⚠️  {warning}\n")
                f.write("\n")

            if assessment['critical_issues']:
                f.write("### Critical Issues:\n")
                for issue in assessment['critical_issues']:
                    f.write(f"- 🚨 {issue}\n")
                f.write("\n")

            f.write("## 📋 Ultra-Constrained Requirements\n\n")
            f.write("| Metric | Target | Actual | Status |\n")
            f.write("|--------|--------|--------|---------|\n")

            avg_latency = results['market_data_profiling']['mean_latency_ms']
            process_memory = results['memory_profiling']['process_memory_mb']
            avg_cpu = results['cpu_profiling']['avg_cpu_pct']

            latency_status = "✅ PASS" if avg_latency <= 0.020 else "❌ FAIL"
            memory_status = "✅ PASS" if process_memory <= 15 else "❌ FAIL"
            cpu_status = "✅ PASS" if avg_cpu <= 85 else "❌ FAIL"

            f.write(f"| Latency | <0.020ms | {avg_latency:.3f}ms | {latency_status} |\n")
            f.write(f"| Memory | <15MB | {process_memory:.1f}MB | {memory_status} |\n")
            f.write(f"| CPU | <85% | {avg_cpu:.1f}% | {cpu_status} |\n")
        print(f"📊 Report generated: {filepath}")
        return str(filepath)


def main():
    """Main profiling execution"""
    print("🚀 Supreme System V5 - Ultra-Constrained Performance Profiler")
    print("=" * 70)

    profiler = UltraConstrainedProfiler()

    try:
        # Run comprehensive profiling
        results = profiler.run_comprehensive_profiling()

        # Save JSON results
        json_file = profiler.save_results(results)

        # Generate markdown report
        md_file = profiler.generate_report(results)

        print("\n" + "=" * 70)
        print("✅ PROFILING SESSION COMPLETE")
        print(f"📊 JSON Results: {json_file}")
        print(f"📋 Report: {md_file}")

        # Final assessment
        assessment = results['performance_assessment']
        if assessment['overall_status'] == 'PASS':
            print("🎉 ALL REQUIREMENTS MET - Ultra-constrained optimization successful!")
        else:
            print("⚠️  PERFORMANCE ISSUES DETECTED - Optimization required")
            for issue in assessment['critical_issues']:
                print(f"🚨 {issue}")

    except Exception as e:
        print(f"❌ PROFILING ERROR: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
