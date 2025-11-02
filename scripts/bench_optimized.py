#!/usr/bin/env python3
"""
Performance Benchmark Suite for Supreme System V5 Optimized Components.
Micro-benchmarks and parity validation against reference implementations.
"""

import time
import sys
import os
import numpy as np
import statistics
from typing import List, Dict, Any, Tuple

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'python'))

from supreme_system_v5.optimized import (
    UltraOptimizedEMA, UltraOptimizedRSI, UltraOptimizedMACD,
    CircularBuffer, SmartEventProcessor, OptimizedTechnicalAnalyzer
)

class ReferenceEMA:
    """Reference EMA implementation for parity validation."""
    def __init__(self, period: int):
        self.period = period
        self.alpha = 2.0 / (period + 1.0)
        self.value = None
        self.prices = []

    def update(self, price: float) -> float:
        if self.value is None:
            self.value = price
        else:
            self.value = self.alpha * price + (1 - self.alpha) * self.value
        self.prices.append(price)
        return self.value

class ReferenceRSI:
    """Reference RSI implementation for parity validation."""
    def __init__(self, period: int = 14):
        self.period = period
        self.gains = []
        self.losses = []
        self.prev_price = None

    def update(self, price: float) -> float:
        if self.prev_price is None:
            self.prev_price = price
            return None

        change = price - self.prev_price
        self.prev_price = price

        if change > 0:
            self.gains.append(change)
            self.losses.append(0.0)
        else:
            self.gains.append(0.0)
            self.losses.append(abs(change))

        if len(self.gains) < self.period:
            return None

        # Keep only last 'period' values
        self.gains = self.gains[-self.period:]
        self.losses = self.losses[-self.period:]

        avg_gain = sum(self.gains) / self.period
        avg_loss = sum(self.losses) / self.period

        if avg_loss == 0:
            return 100.0

        rs = avg_gain / avg_loss
        return 100.0 - (100.0 / (1.0 + rs))

class ReferenceMACD:
    """Reference MACD implementation for parity validation."""
    def __init__(self, fast_period: int = 12, slow_period: int = 26, signal_period: int = 9):
        self.fast_ema = ReferenceEMA(fast_period)
        self.slow_ema = ReferenceEMA(slow_period)
        self.signal_ema = ReferenceEMA(signal_period)
        self.macd_line = None
        self.signal_line = None

    def update(self, price: float) -> Tuple[float, float, float]:
        fast = self.fast_ema.update(price)
        slow = self.slow_ema.update(price)

        if fast is None or slow is None:
            return None

        self.macd_line = fast - slow
        self.signal_line = self.signal_ema.update(self.macd_line)

        if self.signal_line is None:
            return None

        histogram = self.macd_line - self.signal_line
        return (self.macd_line, self.signal_line, histogram)

def generate_test_data(num_samples: int = 10000, volatility: float = 0.02) -> List[float]:
    """Generate realistic price data for testing."""
    prices = [100.0]  # Starting price

    for _ in range(num_samples - 1):
        # Random walk with trend
        change = np.random.normal(0, volatility)
        new_price = prices[-1] * (1 + change)
        prices.append(max(new_price, 0.01))  # Prevent negative prices

    return prices

def benchmark_indicator(name: str, optimized_func, reference_func, test_data: List[float], num_runs: int = 100) -> Dict[str, Any]:
    """
    Benchmark optimized vs reference implementation.

    Args:
        name: Indicator name
        optimized_func: Optimized implementation callable
        reference_func: Reference implementation callable
        test_data: Test price data
        num_runs: Number of benchmark runs

    Returns:
        Benchmark results dictionary
    """
    print(f"\n🔬 Benchmarking {name}...")

    # Warmup runs
    for _ in range(5):
        for price in test_data[:100]:
            optimized_func(price)
            reference_func(price)

    # Reset for actual benchmarking
    optimized_func.reset() if hasattr(optimized_func, 'reset') else None
    reference_func.reset() if hasattr(reference_func, 'reset') else None

    # Benchmark optimized
    optimized_times = []
    optimized_results = []

    for run in range(num_runs):
        start_time = time.perf_counter()

        results = []
        for price in test_data:
            result = optimized_func(price)
            if result is not None:
                results.append(result)

        elapsed = time.perf_counter() - start_time
        optimized_times.append(elapsed)
        optimized_results = results  # Store last run results

    # Benchmark reference
    reference_times = []
    reference_results = []

    for run in range(num_runs):
        start_time = time.perf_counter()

        results = []
        for price in test_data:
            result = reference_func(price)
            if result is not None:
                results.append(result)

        elapsed = time.perf_counter() - start_time
        reference_times.append(elapsed)
        reference_results = results  # Store last run results

    # Calculate statistics
    opt_median = statistics.median(optimized_times) * 1000  # Convert to ms
    ref_median = statistics.median(reference_times) * 1000
    speedup = ref_median / opt_median if opt_median > 0 else float('inf')

    # Validate parity
    parity_ok = True
    if len(optimized_results) == len(reference_results):
        for opt, ref in zip(optimized_results, reference_results):
            if abs(opt - ref) > 1e-6:  # 1e-6 tolerance as per roadmap
                parity_ok = False
                break
    else:
        parity_ok = False

    return {
        'indicator': name,
        'optimized_median_ms': opt_median,
        'reference_median_ms': ref_median,
        'speedup_factor': speedup,
        'parity_valid': parity_ok,
        'optimized_results': optimized_results,
        'reference_results': reference_results,
        'samples_processed': len(optimized_results)
    }

def benchmark_circular_buffer(size: int = 1000, operations: int = 100000) -> Dict[str, Any]:
    """Benchmark CircularBuffer performance."""
    print("\n🔬 Benchmarking CircularBuffer...")
    from supreme_system_v5.optimized import CircularBuffer

    buffer = CircularBuffer(size)

    # Benchmark append operations
    start_time = time.perf_counter()
    for i in range(operations):
        buffer.append(float(i % 100))
    append_time = time.perf_counter() - start_time

    # Benchmark access operations
    start_time = time.perf_counter()
    for _ in range(operations // 10):  # Fewer access operations
        _ = buffer.get_latest(10)
    access_time = time.perf_counter() - start_time

    append_ms = (append_time / operations) * 1000
    access_ms = (access_time / (operations // 10)) * 1000

    return {
        'component': 'CircularBuffer',
        'append_latency_ms': append_ms,
        'access_latency_ms': access_ms,
        'memory_efficient': True,  # Fixed size prevents growth
        'operations_tested': operations
    }

def benchmark_event_processor(test_data: List[Tuple[float, float]], num_runs: int = 10) -> Dict[str, Any]:
    """Benchmark SmartEventProcessor."""
    print("\n🔬 Benchmarking SmartEventProcessor...")
    from supreme_system_v5.optimized import SmartEventProcessor

    config = {
        'min_price_change_pct': 0.0005,
        'min_volume_multiplier': 2.0,
        'max_time_gap_seconds': 30
    }

    processor = SmartEventProcessor(config)

    # Generate timestamps
    timestamps = [time.time() + i * 0.1 for i in range(len(test_data))]

    # Benchmark processing
    events_processed = 0
    events_skipped = 0

    start_time = time.perf_counter()
    for run in range(num_runs):
        processor.reset()
        for (price, volume), timestamp in zip(test_data, timestamps):
            if processor.should_process(price, volume, timestamp):
                events_processed += 1
            else:
                events_skipped += 1
    elapsed = time.perf_counter() - start_time

    total_events = events_processed + events_skipped
    skip_ratio = events_skipped / total_events if total_events > 0 else 0

    return {
        'component': 'SmartEventProcessor',
        'total_events': total_events,
        'events_processed': events_processed,
        'events_skipped': events_skipped,
        'skip_ratio': skip_ratio,
        'processing_time_ms': (elapsed / num_runs) * 1000,
        'target_skip_ratio': 0.7
    }

def run_full_benchmark_suite():
    """Run complete benchmark suite."""
    print("🚀 SUPREME SYSTEM V5 - PERFORMANCE BENCHMARK SUITE")
    print("=" * 70)

    # Generate test data
    print("📊 Generating test data...")
    test_prices = generate_test_data(5000, 0.01)  # 5000 samples, moderate volatility
    test_volume_data = [(price, 100 + np.random.randint(0, 200)) for price in test_prices]

    results = []

    # Benchmark individual indicators
    print("\n🔬 INDICATOR BENCHMARKS")
    print("-" * 50)

    # EMA Benchmark
    ema_optimized = UltraOptimizedEMA(14)
    ema_reference = ReferenceEMA(14)
    ema_result = benchmark_indicator("EMA(14)", ema_optimized.update, ema_reference.update, test_prices)
    results.append(ema_result)

    # RSI Benchmark
    rsi_optimized = UltraOptimizedRSI(14)
    rsi_reference = ReferenceRSI(14)
    rsi_result = benchmark_indicator("RSI(14)", rsi_optimized.update, rsi_reference.update, test_prices)
    results.append(rsi_result)

    # MACD Benchmark
    macd_optimized = UltraOptimizedMACD(12, 26, 9)
    macd_reference = ReferenceMACD(12, 26, 9)
    macd_result = benchmark_indicator("MACD(12,26,9)", macd_optimized.update, macd_reference.update, test_prices)
    results.append(macd_result)

    # Component benchmarks
    cb_result = benchmark_circular_buffer()
    results.append(cb_result)

    ep_result = benchmark_event_processor(test_volume_data)
    results.append(ep_result)

    # Summary report
    print("\n📊 BENCHMARK RESULTS SUMMARY")
    print("=" * 70)

    print("\n🔬 INDICATOR PERFORMANCE")
    print("-" * 50)
    for result in results[:3]:  # EMA, RSI, MACD
        status = "✅" if result['parity_valid'] else "❌"
        print(f"{status} {result['indicator']}:")
        print(".2f"        print(".1f"        print(f"   Parity Valid: {result['parity_valid']}")

    print("\n🔧 COMPONENT PERFORMANCE")
    print("-" * 50)

    # CircularBuffer results
    cb = next(r for r in results if r.get('component') == 'CircularBuffer')
    print("CircularBuffer:")
    print(".3f"    print(".3f"
    # SmartEventProcessor results
    ep = next(r for r in results if r.get('component') == 'SmartEventProcessor')
    print("SmartEventProcessor:")
    print(f"   Events Processed: {ep['events_processed']}")
    print(f"   Events Skipped: {ep['events_skipped']}")
    print(".3f"    print(".3f"
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

    # Indicator latency requirements
    for result in results[:3]:
        latency_ok = result['optimized_median_ms'] < 200  # < 200ms median
        check_criteria(f"{result['indicator']} Latency", latency_ok,
                      ".2f"
    # Parity validation
    for result in results[:3]:
        check_criteria(f"{result['indicator']} Parity", result['parity_valid'],
                      "±1e-6 tolerance")

    # Event processing efficiency
    ep = next(r for r in results if r.get('component') == 'SmartEventProcessor')
    skip_ratio_ok = 0.2 <= ep['skip_ratio'] <= 0.8
    check_criteria("Event Skip Ratio", skip_ratio_ok,
                   ".3f"
    print(f"\n🎯 OVERALL RESULT: {criteria_passed}/{criteria_total} acceptance criteria met")

    # Performance recommendations
    print("
💡 PERFORMANCE RECOMMENDATIONS"    print("-" * 50)

    avg_speedup = statistics.mean([r['speedup_factor'] for r in results if 'speedup_factor' in r])
    print(".1f"
    if avg_speedup > 5:
        print("   ✅ Excellent optimization achieved!")
    elif avg_speedup > 2:
        print("   ⚠️  Moderate optimization - room for improvement")

    ep = next(r for r in results if r.get('component') == 'SmartEventProcessor')
    if ep['skip_ratio'] > 0.7:
        print("   ✅ High event filtering efficiency")
    elif ep['skip_ratio'] > 0.5:
        print("   ⚠️  Moderate filtering - may need tuning")

    return results

def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(description='Supreme System V5 Performance Benchmark Suite')
    parser.add_argument('--samples', type=int, default=5000, help='Number of test samples')
    parser.add_argument('--runs', type=int, default=10, help='Number of benchmark runs')

    args = parser.parse_args()

    # Run benchmark suite
    results = run_full_benchmark_suite()

    # Determine exit code based on acceptance criteria
    criteria_passed = sum(1 for r in results[:3] if r['parity_valid'] and r['optimized_median_ms'] < 200)
    criteria_total = 6  # 3 indicators * 2 criteria each

    success_rate = criteria_passed / criteria_total
    sys.exit(0 if success_rate >= 0.8 else 1)  # 80% pass rate

if __name__ == "__main__":
    main()