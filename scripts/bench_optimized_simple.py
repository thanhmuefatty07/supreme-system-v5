#!/usr/bin/env python3
"""
Simple Benchmark for Supreme System V5 Optimized Components.
"""

import time
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'python'))

from supreme_system_v5.optimized import UltraOptimizedEMA, UltraOptimizedRSI, UltraOptimizedMACD

def generate_test_data(num_samples: int = 1000) -> list:
    """Generate test price data."""
    import random
    prices = [100.0]
    for _ in range(num_samples - 1):
        change = random.uniform(-2, 2)
        new_price = prices[-1] * (1 + change/100)
        prices.append(max(new_price, 0.01))
    return prices

def benchmark_indicator(name: str, indicator_class, period: int, test_data: list, num_runs: int = 10) -> dict:
    """Benchmark a single indicator."""
    print(f"Benchmarking {name}...")

    # Benchmark optimized version
    indicator = indicator_class(period)
    start_time = time.time()

    for run in range(num_runs):
        for price in test_data:
            indicator.update(price)

    elapsed = time.time() - start_time
    avg_time = elapsed / num_runs

    return {
        'indicator': name,
        'avg_time_per_run': avg_time,
        'total_samples': len(test_data),
        'runs': num_runs
    }

def main():
    """Run simple benchmark suite."""
    print("Supreme System V5 - Simple Benchmark")
    print("=" * 40)

    # Generate test data
    test_data = generate_test_data(1000)
    print(f"Generated {len(test_data)} test samples")

    results = []

    # Benchmark EMA
    ema_result = benchmark_indicator("EMA(14)", UltraOptimizedEMA, 14, test_data)
    results.append(ema_result)

    # Benchmark RSI
    rsi_result = benchmark_indicator("RSI(14)", UltraOptimizedRSI, 14, test_data)
    results.append(rsi_result)

    # Benchmark MACD
    macd_result = benchmark_indicator("MACD(12,26,9)", UltraOptimizedMACD, 12, test_data)
    results.append(macd_result)

    print("\nBenchmark Results:")
    print("-" * 40)
    for result in results:
        print(f"{result['indicator']}: {result['avg_time_per_run']:.4f}s")
    print("\nBenchmark completed successfully!")
    return True

if __name__ == "__main__":
    main()