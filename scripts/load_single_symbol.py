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

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'python'))

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
    print("🧪 SUPREME SYSTEM V5 - LOAD TEST SUITE"    print("=" * 70)
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
            'min_price_change_pct': 0.0005,  # 0.05%
            'min_volume_multiplier': 2.0,
            'max_time_gap_seconds': 30
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

    for i, tick in enumerate(price_feed):
        # Process tick
        signal = strategy.add_price_data(
            price=tick['price'],
            volume=tick['volume'],
            timestamp=tick['timestamp']
        )

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
            print(".1f"
    # Final results
    test_duration = time.time() - start_time
    ticks_per_second = len(price_feed) / test_duration

    print("
📈 LOAD TEST RESULTS"    print("=" * 70)
    print(".2f")
    print(f"Total Ticks Processed: {len(price_feed):,}")
    print(".1f")
    print(f"Signals Generated: {signals_generated}")
    print(f"Trades Executed: {trades_executed}")
    print(f"Event Skip Ratio: {events_skipped/events_processed:.3f}")

    # Performance metrics
    if enable_monitoring:
        health_report = monitor.get_system_health_report()
        performance_metrics = monitor.get_performance_stats()

        print("
🔧 RESOURCE UTILIZATION"        print(".1f")
        print(".1f")
        print(".2f")
        print(".3f")
        print(f"Performance Profile: {health_report['performance_profile']}")

        print("
📊 DETAILED METRICS"        print(".1f")
        print(".2f")
        print(".1f")
        print(f"Event Skip Ratio: {performance_metrics['avg_event_skip_ratio']:.3f}")
        print(f"Indicator Measurements: {performance_metrics['indicator_measurements']}")

    # Acceptance criteria validation
    print("
✅ ACCEPTANCE CRITERIA VALIDATION"    print("=" * 70)

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
        avg_cpu = performance_metrics['avg_cpu_percent']
        check_criteria("CPU Usage", avg_cpu < 88.0, f"{avg_cpu:.1f}% < 88%")

        # Memory validation
        avg_ram = performance_metrics['avg_memory_gb']
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
    """Main entry point."""
    parser = argparse.ArgumentParser(description='Supreme System V5 Load Test Suite')
    parser.add_argument('--symbol', default='BTC-USDT', help='Trading symbol')
    parser.add_argument('--rate', type=int, default=20, help='Ticks per second')
    parser.add_argument('--duration-min', type=int, default=5, help='Test duration in minutes')
    parser.add_argument('--no-monitoring', action='store_true', help='Disable resource monitoring')

    args = parser.parse_args()

    # Run async test
    async def run_test():
        return await run_load_test(
            symbol=args.symbol,
            tick_rate=args.rate,
            duration_minutes=args.duration_min,
            enable_monitoring=not args.no_monitoring
        )

    results = asyncio.run(run_test())

    # Exit with success/failure code
    success_rate = results['criteria_passed'] / results['criteria_total']
    sys.exit(0 if success_rate >= 0.8 else 1)  # 80% pass rate required

if __name__ == "__main__":
    main()
