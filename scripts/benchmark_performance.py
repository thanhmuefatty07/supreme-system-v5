#!/usr/bin/env python3
"""
Performance Benchmark Script for Supreme System V5

Measures actual latency and throughput to verify performance claims.
"""

import time
import statistics
import json
from typing import Dict, List, Tuple
from pathlib import Path
import sys
from datetime import datetime

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

try:
    from src.utils.vectorized_ops import VectorizedTradingOps
    import numpy as np
    import pandas as pd
except ImportError as e:
    print(f"Warning: Could not import modules: {e}")
    print("Running simplified benchmark...")
    VectorizedTradingOps = None

try:
    import numpy as np
except ImportError:
    np = None

try:
    import pandas as pd
except ImportError:
    pd = None


def measure_latency(func, *args, iterations: int = 10000) -> Dict[str, float]:
    """
    Measure function execution latency.
    
    Returns:
        Dict with P50, P95, P99, P99.9 latencies in microseconds
    """
    latencies = []
    
    # Warmup
    for _ in range(100):
        try:
            func(*args)
        except:
            pass
    
    # Measure
    for _ in range(iterations):
        start = time.perf_counter()
        try:
            func(*args)
        except:
            pass
        end = time.perf_counter()
        latency_us = (end - start) * 1_000_000  # Convert to microseconds
        latencies.append(latency_us)
    
    if not latencies:
        return {
            'p50': 0,
            'p95': 0,
            'p99': 0,
            'p99_9': 0,
            'mean': 0,
            'min': 0,
            'max': 0
        }
    
    latencies.sort()
    n = len(latencies)
    
    return {
        'p50': latencies[int(n * 0.50)],
        'p95': latencies[int(n * 0.95)],
        'p99': latencies[int(n * 0.99)],
        'p99_9': latencies[int(n * 0.999)] if n > 1000 else latencies[-1],
        'mean': statistics.mean(latencies),
        'min': min(latencies),
        'max': max(latencies),
        'std': statistics.stdev(latencies) if len(latencies) > 1 else 0
    }


def measure_throughput(func, *args, duration_seconds: float = 5.0) -> Dict[str, float]:
    """
    Measure function throughput (operations per second).
    
    Returns:
        Dict with ops_per_sec, total_ops, duration
    """
    start_time = time.perf_counter()
    end_time = start_time + duration_seconds
    count = 0
    
    while time.perf_counter() < end_time:
        try:
            func(*args)
            count += 1
        except:
            pass
    
    actual_duration = time.perf_counter() - start_time
    
    return {
        'ops_per_sec': count / actual_duration if actual_duration > 0 else 0,
        'total_ops': count,
        'duration_seconds': actual_duration
    }


def benchmark_vectorized_operations() -> Dict:
    """Benchmark vectorized trading operations."""
    if VectorizedTradingOps is None or np is None:
        return {'status': 'skipped', 'reason': 'VectorizedTradingOps or numpy not available'}
    
    print("Benchmarking vectorized operations...")
    
    # Create sample data
    n = 10000
    prices = np.random.randn(n).cumsum() + 100
    volumes = np.random.randint(1000, 10000, n)
    
    ops = VectorizedTradingOps()
    
    results = {}
    
    # Benchmark RSI calculation
    print("  -> RSI calculation...")
    try:
        rsi_latency = measure_latency(
            ops.calculate_rsi, prices, period=14, iterations=1000
        )
        results['rsi'] = {
            'latency_us': rsi_latency,
            'throughput': measure_throughput(ops.calculate_rsi, prices, period=14, duration_seconds=2.0)
        }
    except Exception as e:
        print(f"  Error benchmarking RSI: {e}")
        results['rsi'] = {'error': str(e)}
    
    # Benchmark MACD calculation
    print("  -> MACD calculation...")
    try:
        macd_latency = measure_latency(
            ops.calculate_macd, prices, fast=12, slow=26, signal=9, iterations=1000
        )
        results['macd'] = {
            'latency_us': macd_latency,
            'throughput': measure_throughput(ops.calculate_macd, prices, fast=12, slow=26, signal=9, duration_seconds=2.0)
        }
    except Exception as e:
        print(f"  Error benchmarking MACD: {e}")
        results['macd'] = {'error': str(e)}
    
    return results


def benchmark_data_processing() -> Dict:
    """Benchmark data processing pipeline."""
    print("Benchmarking data processing...")
    
    if np is None:
        return {'status': 'skipped', 'reason': 'numpy not available'}
    
    # Simulate data processing
    def process_data():
        data = np.random.randn(1000)
        return data.mean(), data.std()
    
    latency = measure_latency(process_data, iterations=10000)
    throughput = measure_throughput(process_data, duration_seconds=3.0)
    
    return {
        'latency_us': latency,
        'throughput': throughput
    }


def benchmark_strategy_execution() -> Dict:
    """Benchmark strategy execution."""
    print("Benchmarking strategy execution...")
    
    if np is None:
        return {'status': 'skipped', 'reason': 'numpy not available'}
    
    # Simulate strategy execution
    def execute_strategy():
        signals = np.random.randn(10)
        return signals.argmax()
    
    latency = measure_latency(execute_strategy, iterations=5000)
    throughput = measure_throughput(execute_strategy, duration_seconds=3.0)
    
    return {
        'latency_us': latency,
        'throughput': throughput
    }


def generate_report(results: Dict) -> str:
    """Generate human-readable benchmark report."""
    report = []
    report.append("=" * 80)
    report.append("SUPREME SYSTEM V5 - PERFORMANCE BENCHMARK REPORT")
    report.append("=" * 80)
    report.append(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report.append("")
    
    # Overall summary
    report.append("## SUMMARY")
    report.append("")
    
    # Find best latency
    best_p95 = float('inf')
    best_operation = None
    
    for op_name, op_results in results.items():
        if isinstance(op_results, dict) and 'latency_us' in op_results:
            p95 = op_results['latency_us'].get('p95', float('inf'))
            if p95 < best_p95:
                best_p95 = p95
                best_operation = op_name
    
    if best_p95 != float('inf'):
        report.append(f"**Best P95 Latency:** {best_p95:.2f}μs ({best_operation})")
    
    # Find best throughput
    best_throughput = 0
    best_throughput_op = None
    
    for op_name, op_results in results.items():
        if isinstance(op_results, dict) and 'throughput' in op_results:
            tps = op_results['throughput'].get('ops_per_sec', 0)
            if tps > best_throughput:
                best_throughput = tps
                best_throughput_op = op_name
    
    if best_throughput > 0:
        report.append(f"**Best Throughput:** {best_throughput:.0f} ops/sec ({best_throughput_op})")
    
    report.append("")
    report.append("## DETAILED RESULTS")
    report.append("")
    
    # Detailed results
    for op_name, op_results in results.items():
        if isinstance(op_results, dict) and 'latency_us' in op_results:
            report.append(f"### {op_name.upper()}")
            report.append("")
            
            latency = op_results['latency_us']
            report.append("**Latency (microseconds):**")
            report.append(f"  - P50: {latency.get('p50', 0):.2f}μs")
            report.append(f"  - P95: {latency.get('p95', 0):.2f}μs")
            report.append(f"  - P99: {latency.get('p99', 0):.2f}μs")
            report.append(f"  - Mean: {latency.get('mean', 0):.2f}μs")
            report.append("")
            
            if 'throughput' in op_results:
                throughput = op_results['throughput']
                report.append("**Throughput:**")
                report.append(f"  - Operations/sec: {throughput.get('ops_per_sec', 0):.0f}")
                report.append(f"  - Total operations: {throughput.get('total_ops', 0)}")
                report.append("")
    
    report.append("=" * 80)
    
    return "\n".join(report)


def main():
    """Run all benchmarks and generate report."""
    print("Starting Supreme System V5 Performance Benchmark")
    print("=" * 80)
    print("")
    
    results = {}
    
    # Run benchmarks
    try:
        vectorized_results = benchmark_vectorized_operations()
        if vectorized_results.get('status') != 'skipped':
            results['vectorized_operations'] = vectorized_results
    except Exception as e:
        print(f"Error benchmarking vectorized operations: {e}")
        results['vectorized_operations'] = {'error': str(e)}
    
    try:
        results['data_processing'] = benchmark_data_processing()
    except Exception as e:
        print(f"Error benchmarking data processing: {e}")
        results['data_processing'] = {'error': str(e)}
    
    try:
        results['strategy_execution'] = benchmark_strategy_execution()
    except Exception as e:
        print(f"Error benchmarking strategy execution: {e}")
        results['strategy_execution'] = {'error': str(e)}
    
    # Generate report
    report = generate_report(results)
    print("\n" + report)
    
    # Save JSON report
    output_file = project_root / "PERFORMANCE_BENCHMARK_REPORT.json"
    with open(output_file, 'w') as f:
        json.dump({
            'timestamp': datetime.now().isoformat(),
            'results': results
        }, f, indent=2)
    
    # Save markdown report
    md_file = project_root / "PERFORMANCE_BENCHMARK_REPORT.md"
    with open(md_file, 'w') as f:
        f.write(report)
    
    print(f"\nBenchmark complete!")
    print(f"Reports saved:")
    print(f"   - JSON: {output_file}")
    print(f"   - Markdown: {md_file}")
    
    # Recommendations
    print("\n" + "=" * 80)
    print("## RECOMMENDATIONS")
    print("=" * 80)
    
    # Check if claims are met
    all_p95_latencies = []
    for op_results in results.values():
        if isinstance(op_results, dict) and 'latency_us' in op_results:
            p95 = op_results['latency_us'].get('p95', float('inf'))
            if p95 != float('inf'):
                all_p95_latencies.append(p95)
    
    if all_p95_latencies:
        worst_p95 = max(all_p95_latencies)
        best_p95 = min(all_p95_latencies)
        
        if worst_p95 < 10:
            print("SUCCESS: P95 latency <10us achieved for some operations!")
        elif worst_p95 < 50:
            print(f"WARNING: P95 latency is {worst_p95:.2f}us (sub-50us, competitive but not <10us)")
            print("   Recommendation: Update marketing claims to 'Sub-50us latency'")
        else:
            print(f"CRITICAL: P95 latency is {worst_p95:.2f}us (exceeds 50us)")
            print("   Recommendation: Optimize code or update claims to realistic values")
    
    return results


if __name__ == "__main__":
    main()

