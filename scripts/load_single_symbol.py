#!/usr/bin/env python3
"""
Load Test Suite for Supreme System V5.
Tests single-symbol optimized performance under realistic conditions.
"""

import asyncio
import time
import sys
import os
import argparse
from typing import Dict, List, Any
from prometheus_client import Histogram, Gauge, start_http_server
import numpy as np

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'python'))

# Prometheus Metrics for Load Test
STRATEGY_LATENCY = Histogram(
    'strategy_latency_seconds',
    'Strategy calculation latency distribution',
    ['strategy', 'percentile'],
    buckets=[0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0, 5.0]
)

CPU_PERCENT_GAUGE = Gauge(
    'cpu_percent_gauge',
    'CPU usage percentage during load test',
    ['phase']
)

MEMORY_IN_USE_BYTES = Gauge(
    'memory_in_use_bytes',
    'Memory usage in bytes during load test',
    ['phase']
)

EVENT_SKIP_RATIO = Gauge(
    'event_skip_ratio',
    'Ratio of events filtered during load test',
    ['test_type']
)

from supreme_system_v5.optimized import OptimizedTechnicalAnalyzer
from supreme_system_v5.strategies import ScalpingStrategy
from supreme_system_v5.monitoring import AdvancedResourceMonitor

def generate_price_feed(symbol: str, duration_minutes: int, tick_rate: int) -> List[Dict[str, Any]]:
    """
    Generate realistic price feed for testing.

    Args:
        symbol: Trading symbol
        duration_minutes: Test duration
        tick_rate: Ticks per second

    Returns:
        List of price updates
    """
    import math
    import random

    total_ticks = duration_minutes * 60 * tick_rate
    price_data = []

    base_price = 50000.0
    current_time = time.time()

    for i in range(total_ticks):
        # Generate trending price with noise
        trend = 200 * math.sin(i / (tick_rate * 60))  # Slow trend
        noise = 20 * (random.random() - 0.5)  # Random noise
        micro_noise = 0.1 * (random.random() - 0.5)  # Micro movements

        price = base_price + trend + noise + micro_noise
        volume = 100 + random.randint(0, 200)  # Realistic volume

        price_data.append({
            'symbol': symbol,
            'price': round(price, 2),
            'volume': volume,
            'timestamp': current_time + (i / tick_rate)
        })

    return price_data

async def run_load_test(symbol: str, tick_rate: int, duration_minutes: int, enable_monitoring: bool = True):
    """
    Run comprehensive load test.

    Args:
        symbol: Trading symbol
        tick_rate: Ticks per second
        duration_minutes: Test duration
        enable_monitoring: Enable resource monitoring
    """
    print("🧪 SUPREME SYSTEM V5 - LOAD TEST SUITE")
    print("=" * 70)
    print(f"Symbol: {symbol}")
    print(f"Tick Rate: {tick_rate} ticks/sec")
    print(f"Duration: {duration_minutes} minutes")
    print(f"Total Ticks: {duration_minutes * 60 * tick_rate:,}")
    print()

    # Initialize components
    print("🚀 Initializing optimized components...")

    # Strategy configuration
    strategy_config = {
        'symbol': symbol,
        'position_size_pct': 0.02,
        'stop_loss_pct': 0.01,
        'take_profit_pct': 0.02,
        'ema_period': 14,
        'rsi_period': 14,
        'macd_fast': 12,
        'macd_slow': 26,
        'macd_signal': 9,
        'price_history_size': 100,
        'event_config': {
            'min_price_change_pct': 0.0001,  # 0.01% - very permissive for testing
            'min_volume_multiplier': 1.1,     # 1.1x - very permissive for testing
            'max_time_gap_seconds': 10        # Process every 10s
        }
    }

    strategy = ScalpingStrategy(strategy_config)

    # Resource monitor
    monitor_config = {
        'cpu_high_threshold': 88.0,
        'memory_high_threshold': 3.86,
        'latency_high_threshold': 200,
        'monitoring_interval': 5.0,
        'optimization_check_interval': 60.0
    }

    monitor = AdvancedResourceMonitor(monitor_config)
    if enable_monitoring:
        monitor.start_monitoring()

    # Generate test data
    print("📊 Generating realistic price feed...")
    price_feed = generate_price_feed(symbol, duration_minutes, tick_rate)

    # Run test
    print("🏃 Running load test...")
    start_time = time.time()

    signals_generated = 0
    trades_executed = 0
    events_processed = 0
    events_skipped = 0

    strategy_latencies = []

    for i, tick in enumerate(price_feed):
        # Process tick with latency measurement
        tick_start_time = time.perf_counter()
        signal = strategy.add_price_data(
            price=tick['price'],
            volume=tick['volume'],
            timestamp=tick['timestamp']
        )
        latency = time.perf_counter() - tick_start_time

        # Record strategy latency
        STRATEGY_LATENCY.labels(strategy='optimized_scalping', percentile='raw').observe(latency)
        strategy_latencies.append(latency)

        events_processed += 1

        if signal is None:
            events_skipped += 1
        else:
            signals_generated += 1
            if signal['action'] in ['BUY', 'SELL', 'CLOSE']:
                trades_executed += 1

        # Progress reporting
        if (i + 1) % (tick_rate * 30) == 0:  # Every 30 seconds
            elapsed = time.time() - start_time
            progress = (i + 1) / len(price_feed) * 100
            print(f"   Progress: {progress:.1f}% complete ({elapsed:.1f}s elapsed)")
    # Final results
    test_duration = time.time() - start_time
    ticks_per_second = len(price_feed) / max(test_duration, 0.001)  # Prevent division by zero

    print("\n📈 LOAD TEST RESULTS")
    print("=" * 70)
    print(f"Test Duration: {test_duration:.2f}s")
    print(f"Total Ticks Processed: {len(price_feed):,}")
    print(f"Processing Rate: {ticks_per_second:.1f} ticks/second")
    print(f"Signals Generated: {signals_generated}")
    print(f"Trades Executed: {trades_executed}")
    print(f"Event Skip Ratio: {events_skipped/events_processed:.3f}")

    # Performance metrics
    if enable_monitoring:
        health_report = monitor.get_system_health_report()
        performance_metrics = monitor.get_performance_metrics()

        print("\n🔧 RESOURCE UTILIZATION")
        print(f"Average CPU Usage: {performance_metrics.get('avg_cpu_percent', 0):.1f}%")
        print(f"Average Memory Usage: {performance_metrics.get('avg_memory_gb', 0):.1f}GB")
        print(f"Average Latency: {performance_metrics.get('avg_indicator_latency_ms', 0):.2f}ms")
        print(f"Event Skip Ratio: {performance_metrics.get('avg_event_skip_ratio', 0):.3f}")
        print(f"Performance Profile: {health_report.get('performance_profile', 'unknown')}")

        print("\n📊 DETAILED METRICS")
        print(f"Average Latency: {performance_metrics.get('avg_indicator_latency_ms', 0):.2f}ms")
        print(f"Event Skip Ratio: {performance_metrics.get('avg_event_skip_ratio', 0):.3f}")
        print(f"Indicator Measurements: {performance_metrics.get('indicator_measurements', 0)}")

    # Record p50 and p95 latency metrics
    if strategy_latencies:
        p50_latency = np.percentile(strategy_latencies, 50)
        p95_latency = np.percentile(strategy_latencies, 95)

        STRATEGY_LATENCY.labels(strategy='optimized_scalping', percentile='p50').observe(p50_latency)
        STRATEGY_LATENCY.labels(strategy='optimized_scalping', percentile='p95').observe(p95_latency)

        print("\n📈 STRATEGY LATENCY METRICS")
        print(f"P50 Latency: {p50_latency:.4f}s")
        print(f"P95 Latency: {p95_latency:.4f}s")
    # Record event skip ratio
    skip_ratio = events_skipped / events_processed if events_processed > 0 else 0
    EVENT_SKIP_RATIO.labels(test_type='load_test').set(skip_ratio)

    # Acceptance criteria validation
    print("\n✅ ACCEPTANCE CRITERIA VALIDATION")
    print("=" * 70)

    criteria_passed = 0
    criteria_total = 0

    def check_criteria(name: str, condition: bool, target: str):
        nonlocal criteria_passed, criteria_total
        criteria_total += 1
        status = "✅ PASS" if condition else "❌ FAIL"
        print(f"{status} {name}: {target}")
        if condition:
            criteria_passed += 1

    # CPU validation
    if enable_monitoring:
        avg_cpu = performance_metrics.get('avg_cpu_percent', 0)
        check_criteria("CPU Usage", avg_cpu < 88.0, f"{avg_cpu:.1f}% < 88%")

        # Memory validation
        avg_ram = performance_metrics.get('avg_memory_gb', 0)
        check_criteria("RAM Usage", avg_ram < 3.86, f"{avg_ram:.2f}GB < 3.86GB")

    # Event skip ratio
    skip_ratio = events_skipped / events_processed
    check_criteria("Event Skip Ratio", 0.2 <= skip_ratio <= 0.8, f"{skip_ratio:.3f} in [0.2, 0.8]")

    # Processing rate
    check_criteria("Tick Processing Rate", ticks_per_second >= tick_rate * 0.95, f"{ticks_per_second:.1f} >= {tick_rate * 0.95:.1f}")

    print(f"\n🎯 OVERALL RESULT: {criteria_passed}/{criteria_total} acceptance criteria met")

    # Stop monitoring
    if enable_monitoring:
        monitor.stop_monitoring()

    return {
        'duration': test_duration,
        'ticks_processed': len(price_feed),
        'signals_generated': signals_generated,
        'trades_executed': trades_executed,
        'event_skip_ratio': events_skipped / events_processed if events_processed > 0 else 0,
        'ticks_per_second': ticks_per_second,
        'criteria_passed': criteria_passed,
        'criteria_total': criteria_total
    }

def main():
    """Main entry point with system monitoring."""
    import psutil

    parser = argparse.ArgumentParser(description='Supreme System V5 Load Test Suite')
    parser.add_argument('--symbol', default='BTC-USDT', help='Trading symbol')
    parser.add_argument('--rate', type=int, default=20, help='Ticks per second')
    parser.add_argument('--duration-min', type=int, default=5, help='Test duration in minutes')
    parser.add_argument('--no-monitoring', action='store_true', help='Disable resource monitoring')
    parser.add_argument('--prometheus-port', type=int, default=9092, help='Prometheus metrics port')
    parser.add_argument('--output-json', type=str, help='Output results to JSON file')

    args = parser.parse_args()

    # Start Prometheus metrics server
    print(f"📊 Starting Prometheus metrics server on port {args.prometheus_port}...")
    start_http_server(args.prometheus_port)

    # Monitor initial system resources
    process = psutil.Process()
    initial_cpu = psutil.cpu_percent(interval=1)
    initial_memory = process.memory_info().rss

    CPU_PERCENT_GAUGE.labels(phase='initial').set(initial_cpu)
    MEMORY_IN_USE_BYTES.labels(phase='initial').set(initial_memory)

    print(f"   Initial CPU: {initial_cpu:.1f}%")
    print(f"   Initial Memory: {initial_memory / (1024**2):.1f}MB")

    # Run async test
    async def run_test():
        return await run_load_test(
            symbol=args.symbol,
            tick_rate=args.rate,
            duration_minutes=args.duration_min,
            enable_monitoring=not args.no_monitoring
        )

    results = asyncio.run(run_test())

    # Monitor final system resources
    final_cpu = psutil.cpu_percent(interval=1)
    final_memory = process.memory_info().rss

    CPU_PERCENT_GAUGE.labels(phase='final').set(final_cpu)
    MEMORY_IN_USE_BYTES.labels(phase='final').set(final_memory)

    print("\n📈 Final System Resources:")
    print(f"   Final CPU: {final_cpu:.1f}%")
    print(f"   Final Memory: {final_memory / (1024**2):.1f}MB")

    # Save results to JSON if requested
    if args.output_json:
        import json
        output_data = {
            'timestamp': time.time(),
            'symbol': args.symbol,
            'tick_rate': args.rate,
            'duration_minutes': args.duration_min,
            'results': results,
            'performance_metrics': performance_metrics if 'performance_metrics' in locals() else {},
            'system_resources': {
                'initial_cpu': initial_cpu,
                'final_cpu': final_cpu,
                'initial_memory_mb': initial_memory / (1024**2),
                'final_memory_mb': final_memory / (1024**2)
            }
        }

        with open(args.output_json, 'w') as f:
            json.dump(output_data, f, indent=2)
        print(f"📊 Results saved to {args.output_json}")

    # Exit with success/failure code
    success_rate = results['criteria_passed'] / results['criteria_total']
    sys.exit(0 if success_rate >= 0.8 else 1)  # 80% pass rate required

if __name__ == "__main__":
    main()
