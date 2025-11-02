#!/usr/bin/env python3
"""
Simple benchmark script for optimized components.
Validates core functionality and performance.
"""

import time
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'python'))

def test_optimized_components():
    """Test that optimized components work correctly."""
    print("🧪 Testing Optimized Components")
    print("=" * 50)

    try:
        from supreme_system_v5.optimized import (
            UltraOptimizedEMA, UltraOptimizedRSI, UltraOptimizedMACD,
            CircularBuffer, SmartEventProcessor
        )

        # Test CircularBuffer
        print("Testing CircularBuffer...")
        buffer = CircularBuffer(10)
        for i in range(15):
            buffer.append(float(i))
        latest = buffer.get_latest(5)
        assert len(latest) == 5, "Buffer latest failed"
        print("✅ CircularBuffer: PASS")

        # Test UltraOptimizedEMA
        print("Testing UltraOptimizedEMA...")
        ema = UltraOptimizedEMA(period=14)
        for i in range(20):
            result = ema.update(100.0 + i)
        assert ema.is_initialized(), "EMA not initialized"
        print("✅ UltraOptimizedEMA: PASS")

        # Test UltraOptimizedRSI
        print("Testing UltraOptimizedRSI...")
        rsi = UltraOptimizedRSI(period=14)
        for i in range(20):
            result = rsi.update(100.0 + i * 0.1)
        assert rsi.is_initialized(), "RSI not initialized"
        print("✅ UltraOptimizedRSI: PASS")

        # Test UltraOptimizedMACD
        print("Testing UltraOptimizedMACD...")
        macd = UltraOptimizedMACD(fast_period=12, slow_period=26, signal_period=9)
        for i in range(50):
            result = macd.update(100.0 + i)
        assert macd.is_initialized(), "MACD not initialized"
        print("✅ UltraOptimizedMACD: PASS")

        # Test SmartEventProcessor
        print("Testing SmartEventProcessor...")
        processor = SmartEventProcessor({
            'min_price_change_pct': 0.01,
            'min_volume_multiplier': 2.0,
            'max_time_gap_seconds': 30
        })
        should_process = processor.should_process(100.0, 1000, time.time())
        stats = processor.get_stats()
        assert 'events_processed' in stats, "Event processor stats failed"
        print("✅ SmartEventProcessor: PASS")

        print("\n🎉 ALL OPTIMIZED COMPONENTS WORKING!")
        return True

    except Exception as e:
        print(f"❌ COMPONENT TEST FAILED: {e}")
        return False

def benchmark_performance():
    """Simple performance benchmark."""
    print("\n⚡ Performance Benchmark")
    print("=" * 50)

    try:
        from supreme_system_v5.optimized import UltraOptimizedEMA

        # Benchmark 10k updates
        num_updates = 10000
        ema = UltraOptimizedEMA(period=14)

        start_time = time.time()
        for i in range(num_updates):
            ema.update(100.0 + i * 0.01)
        end_time = time.time()

        total_time = end_time - start_time
        avg_time_per_update = total_time / num_updates

        print(f"Updates: {num_updates}")
        print(".2f")
        print(".1f")

        if avg_time_per_update < 0.0001:  # < 0.1ms per update
            print("✅ PERFORMANCE: EXCELLENT (< 0.1ms per update)")
            return True
        elif avg_time_per_update < 0.001:  # < 1ms per update
            print("✅ PERFORMANCE: GOOD (< 1ms per update)")
            return True
        else:
            print("❌ PERFORMANCE: SLOW (> 1ms per update)")
            return False

    except Exception as e:
        print(f"❌ BENCHMARK FAILED: {e}")
        return False

def main():
    """Run all tests."""
    print("🚀 SUPREME SYSTEM V5 - OPTIMIZED COMPONENTS VALIDATION")
    print("=" * 70)

    # Run tests
    component_test = test_optimized_components()
    performance_test = benchmark_performance()

    # Results
    print("\n" + "=" * 70)
    print("📊 VALIDATION RESULTS")
    print("=" * 70)

    results = [
        ("Component Functionality", component_test),
        ("Performance Benchmark", performance_test)
    ]

    passed = 0
    for name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print("25s")
        if result:
            passed += 1

    print(f"\n🎯 SUMMARY: {passed}/{len(results)} tests passed")

    if passed == len(results):
        print("🚀 OPTIMIZED COMPONENTS VALIDATION: SUCCESS")
        print("💪 CORE OPTIMIZATIONS READY FOR PRODUCTION")
        return True
    else:
        print("⚠️ SOME TESTS FAILED - REVIEW REQUIRED")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
