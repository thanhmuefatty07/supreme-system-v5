"""
Performance benchmarking tests for Supreme System V5.

Tests cover:
- Vectorized operations performance
- Memory optimization benchmarks
- Async I/O performance
- End-to-end pipeline performance
- Regression detection
- Hardware-specific optimizations
"""

import time
import pytest
import numpy as np
import pandas as pd
from typing import Dict, List, Any
import psutil
import gc

from src.utils.vectorized_ops import (
    VectorizedTradingOps, benchmark_all_implementations, SYSTEM_INFO
)
from src.utils.memory_optimizer import (
    MemoryOptimizer, benchmark_memory_optimization, optimize_trading_data_pipeline
)
from src.data.binance_client import AsyncBinanceClient
from src.data.data_pipeline import DataPipeline


class TestPerformanceRegression:
    """Test performance regression detection."""

    def test_vectorized_performance_regression(self):
        """Test for performance regression in vectorized operations."""
        results = benchmark_all_implementations()

        # Define performance thresholds (these should be based on baseline measurements)
        min_speedup_threshold = 2.0  # At least 2x speedup over pandas
        max_time_threshold = 5.0     # Max 5 seconds for benchmark suite

        # Check overall speedup
        overall_speedup = results['overall_speedup']
        assert overall_speedup >= min_speedup_threshold, \
            f"Performance regression detected: {overall_speedup:.2f}x speedup, " \
            f"minimum required: {min_speedup_threshold}x"

        # Check total time
        total_time = results['total_numba_time']
        assert total_time <= max_time_threshold, \
            f"Performance too slow: {total_time:.2f}s, maximum allowed: {max_time_threshold}s"

        # Check individual operations
        assert results['sma_numba_speedup'] >= 1.5
        assert results['ema_numba_speedup'] >= 1.5
        assert results['rsi_numba_speedup'] >= 1.5

    def test_memory_optimization_performance(self):
        """Test memory optimization performance."""
        results = benchmark_memory_optimization()

        # Define memory optimization thresholds
        min_savings_percent = 10.0  # At least 10% memory savings
        max_optimization_time = 2.0  # Max 2 seconds for optimization

        # Check memory savings
        savings_percent = results['savings_percent']
        assert savings_percent >= min_savings_percent, \
            f"Memory optimization insufficient: {savings_percent:.1f}% savings, " \
            f"minimum required: {min_savings_percent}%"

        # Check optimization time
        opt_time = results['optimization_time_seconds']
        assert opt_time <= max_optimization_time, \
            f"Memory optimization too slow: {opt_time:.2f}s, " \
            f"maximum allowed: {max_optimization_time}s"


class TestScalabilityBenchmarks:
    """Test scalability with different data sizes."""

    @pytest.mark.parametrize("data_size", [1000, 10000, 50000, 100000])
    def test_vectorized_scalability(self, data_size):
        """Test vectorized operations scalability."""
        # Generate test data
        prices = np.random.uniform(100, 200, data_size)

        # Test SMA calculation
        start_time = time.time()
        sma = VectorizedTradingOps.calculate_sma_numba(prices, window=20)
        sma_time = time.time() - start_time

        # Test EMA calculation
        start_time = time.time()
        ema = VectorizedTradingOps.calculate_ema_numba(prices, span=12)
        ema_time = time.time() - start_time

        # Test RSI calculation
        start_time = time.time()
        rsi = VectorizedTradingOps.calculate_rsi_numba(prices, period=14)
        rsi_time = time.time() - start_time

        # Calculate throughput (operations per second)
        sma_throughput = data_size / sma_time
        ema_throughput = data_size / ema_time
        rsi_throughput = data_size / rsi_time

        # Verify reasonable performance (at least 1000 ops/sec for small data, scales up)
        min_throughput = max(1000, data_size / 10)  # Scales with data size

        assert sma_throughput >= min_throughput, \
            f"SMA throughput too low: {sma_throughput:.0f} ops/sec"
        assert ema_throughput >= min_throughput, \
            f"EMA throughput too low: {ema_throughput:.0f} ops/sec"
        assert rsi_throughput >= min_throughput / 2, \
            f"RSI throughput too low: {rsi_throughput:.0f} ops/sec (RSI is more complex)"

    @pytest.mark.parametrize("data_size", [1000, 10000, 50000])
    def test_memory_scalability(self, data_size):
        """Test memory optimization scalability."""
        # Create test DataFrame
        df = pd.DataFrame({
            'timestamp': pd.date_range('2024-01-01', periods=data_size, freq='1min'),
            'open': np.random.uniform(100, 110, data_size),
            'high': np.random.uniform(105, 115, data_size),
            'low': np.random.uniform(95, 105, data_size),
            'close': np.random.uniform(100, 110, data_size),
            'volume': np.random.randint(1000, 10000, data_size),
            'extra_col': ['redundant'] * data_size
        })

        # Force suboptimal memory usage
        df['volume'] = df['volume'].astype('int64')
        df['extra_col'] = df['extra_col'].astype('object')

        original_memory = df.memory_usage(deep=True).sum()

        # Time optimization
        start_time = time.time()
        optimized_df = optimize_trading_data_pipeline(df)
        optimization_time = time.time() - start_time

        optimized_memory = optimized_df.memory_usage(deep=True).sum()
        savings_percent = (original_memory - optimized_memory) / original_memory * 100

        # Verify optimization effectiveness scales
        assert savings_percent >= 5.0, \
            f"Memory optimization ineffective: {savings_percent:.1f}% savings"

        # Optimization time should scale reasonably (not exponentially)
        max_time_per_thousand = 0.1  # Max 100ms per 1000 rows
        max_expected_time = (data_size / 1000) * max_time_per_thousand

        assert optimization_time <= max_expected_time, \
            f"Optimization too slow: {optimization_time:.3f}s for {data_size} rows"


class TestConcurrentPerformance:
    """Test concurrent operation performance."""

    @pytest.mark.asyncio
    async def test_async_concurrency_performance(self):
        """Test async operation concurrency performance."""
        # Create mock client
        client = AsyncBinanceClient(api_key="test", api_secret="test", testnet=True)

        # Mock responses
        mock_klines = [
            [1640995200000, "100.0", "105.0", "95.0", "102.0", "10000.0",
             1640998800000, "0", "100", "0", "0", "0"]
        ]

        from aioresponses import aioresponses
        with aioresponses() as m:
            # Mock multiple concurrent requests
            for i in range(10):
                m.get(f"{client.base_url}/api/v3/klines", payload=mock_klines)

            async with client:
                symbols = [f"TEST{i}USDT" for i in range(10)]

                start_time = time.time()
                results = await client.get_multiple_symbols_data(
                    symbols, "1h", "2022-01-01", limit=1
                )
                total_time = time.time() - start_time

                # All requests should succeed
                successful_requests = sum(1 for r in results.values() if r is not None)
                assert successful_requests == len(symbols)

                # Concurrent requests should be reasonably fast
                # (Sequential would take ~10 * rate_limit_delay, concurrent should be much faster)
                min_expected_time = client.rate_limit_delay * 2  # Allow some overhead
                assert total_time >= min_expected_time, \
                    f"Concurrent requests too fast (possible rate limit bypass): {total_time:.3f}s"

                max_expected_time = client.rate_limit_delay * len(symbols) * 2  # Allow 2x sequential
                assert total_time <= max_expected_time, \
                    f"Concurrent requests too slow: {total_time:.3f}s"


class TestMemoryUsageBenchmarks:
    """Test memory usage under different scenarios."""

    def test_peak_memory_usage(self):
        """Test peak memory usage during operations."""
        optimizer = MemoryOptimizer()

        initial_memory = optimizer.get_current_memory_mb()

        # Perform memory-intensive operations
        with optimizer.monitor_memory_usage("memory_intensive_test"):
            # Create large datasets
            large_data = []
            for i in range(10):
                df = pd.DataFrame({
                    'timestamp': pd.date_range('2024-01-01', periods=10000, freq='1min'),
                    'prices': np.random.uniform(100, 200, 10000),
                    'volumes': np.random.randint(1000, 10000, 10000)
                })
                large_data.append(df)

            # Process all data
            processed_data = []
            for df in large_data:
                optimized = optimize_trading_data_pipeline(df)
                processed_data.append(optimized)

            # Force cleanup
            del large_data
            gc.collect()

        final_memory = optimizer.get_current_memory_mb()
        memory_increase = final_memory - initial_memory

        # Memory increase should be reasonable (< 500MB)
        assert memory_increase < 500, \
            f"Excessive memory usage: +{memory_increase:.1f} MB"

    def test_memory_leak_detection(self):
        """Test for memory leaks in repeated operations."""
        optimizer = MemoryOptimizer()

        memory_usage = []

        # Perform same operation multiple times
        for i in range(5):
            # Create and process data
            df = pd.DataFrame({
                'timestamp': pd.date_range('2024-01-01', periods=5000, freq='1min'),
                'prices': np.random.uniform(100, 200, 5000),
                'volumes': np.random.randint(1000, 10000, 5000)
            })

            optimized = optimize_trading_data_pipeline(df)
            memory_usage.append(optimizer.get_current_memory_mb())

            # Force garbage collection
            del df, optimized
            gc.collect()

        # Memory usage should not consistently increase
        memory_differences = np.diff(memory_usage)
        average_increase = np.mean(memory_differences)

        # Allow small increases but detect significant leaks
        assert average_increase < 10, \
            f"Potential memory leak detected: average increase {average_increase:.1f} MB per iteration"


class TestHardwareSpecificPerformance:
    """Test performance on different hardware configurations."""

    def test_cpu_core_utilization(self):
        """Test CPU core utilization in parallel operations."""
        import multiprocessing

        available_cores = multiprocessing.cpu_count()
        physical_cores = psutil.cpu_count(logical=False)

        # Test batch processing with different core counts
        data_batch = np.random.normal(100, 5, (5, 10000))  # 5 symbols, 10k periods each
        indicators = np.array([0, 1, 2])  # SMA, EMA, RSI

        start_time = time.time()
        results = VectorizedTradingOps.batch_indicator_calculation_numba(data_batch, indicators)
        batch_time = time.time() - start_time

        # Verify results
        assert results.shape == (5, 10000, 3)

        # Performance should scale with available cores
        min_expected_throughput = 10000 * 5 / 10  # At least 50k operations per second
        actual_throughput = (5 * 10000 * 3) / batch_time  # symbols * periods * indicators

        assert actual_throughput >= min_expected_throughput, \
            f"Batch processing throughput too low: {actual_throughput:.0f} ops/sec"

    def test_avx512_performance_impact(self):
        """Test AVX-512 performance impact when available."""
        data_size = 50000
        prices = np.random.uniform(100, 200, data_size)

        # Test RSI calculation (good candidate for SIMD)
        start_time = time.time()
        rsi = VectorizedTradingOps.calculate_rsi_numba(prices, period=14)
        rsi_time = time.time() - start_time

        # Calculate throughput
        throughput = data_size / rsi_time

        # AVX-512 should provide better performance
        if SYSTEM_INFO.get('avx512_supported', False):
            # With AVX-512, expect higher throughput
            min_avx_throughput = 50000  # 50k ops/sec with AVX-512
            assert throughput >= min_avx_throughput, \
                f"AVX-512 performance below expected: {throughput:.0f} ops/sec"
        else:
            # Without AVX-512, still expect reasonable performance
            min_base_throughput = 20000  # 20k ops/sec baseline
            assert throughput >= min_base_throughput, \
                f"Base performance below expected: {throughput:.0f} ops/sec"


class TestEndToEndPerformance:
    """Test end-to-end performance of complete workflows."""

    def test_complete_trading_pipeline_performance(self):
        """Test performance of complete trading data pipeline."""
        # Create comprehensive test dataset
        symbols = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA']
        data_points_per_symbol = 5000

        test_data = {}
        for symbol in symbols:
            df = pd.DataFrame({
                'timestamp': pd.date_range('2024-01-01', periods=data_points_per_symbol, freq='1min'),
                'open': np.random.uniform(100, 200, data_points_per_symbol),
                'high': np.random.uniform(105, 205, data_points_per_symbol),
                'low': np.random.uniform(95, 195, data_points_per_symbol),
                'close': np.random.uniform(100, 200, data_points_per_symbol),
                'volume': np.random.randint(1000, 100000, data_points_per_symbol)
            })
            test_data[symbol] = df

        # Test sequential processing
        sequential_start = time.time()
        sequential_results = {}
        for symbol, df in test_data.items():
            # Optimize memory
            optimized_df = optimize_trading_data_pipeline(df)

            # Calculate indicators
            indicators = VectorizedTradingOps.calculate_indicators_optimal(
                optimized_df['close'], ['sma_20', 'ema_12', 'rsi_14']
            )

            sequential_results[symbol] = {
                'data': optimized_df,
                'indicators': indicators
            }

        sequential_time = time.time() - sequential_start

        # Calculate total throughput
        total_data_points = len(symbols) * data_points_per_symbol
        sequential_throughput = total_data_points / sequential_time

        # Verify reasonable performance
        min_throughput = 10000  # At least 10k data points per second
        assert sequential_throughput >= min_throughput, \
            f"Pipeline throughput too low: {sequential_throughput:.0f} data points/sec"

        # Verify all symbols processed
        assert len(sequential_results) == len(symbols)
        for symbol in symbols:
            assert symbol in sequential_results
            assert len(sequential_results[symbol]['data']) == data_points_per_symbol

    def test_memory_efficiency_under_load(self):
        """Test memory efficiency during high-load operations."""
        optimizer = MemoryOptimizer()
        initial_memory = optimizer.get_current_memory_mb()

        peak_memory = initial_memory

        # Simulate high-load scenario
        with optimizer.monitor_memory_usage("high_load_test"):
            # Create multiple large datasets
            datasets = []
            for i in range(5):
                df = pd.DataFrame({
                    'timestamp': pd.date_range('2024-01-01', periods=10000, freq='1min'),
                    'prices': np.random.uniform(100, 200, 10000),
                    'volumes': np.random.randint(1000, 10000, 10000),
                    'indicators': np.random.uniform(0, 1, 10000)
                })

                # Optimize and store
                optimized = optimize_trading_data_pipeline(df)
                datasets.append(optimized)

                # Track peak memory
                current_memory = optimizer.get_current_memory_mb()
                peak_memory = max(peak_memory, current_memory)

        final_memory = optimizer.get_current_memory_mb()
        memory_increase = peak_memory - initial_memory

        # Memory increase should be reasonable for the data size
        # 5 datasets * 10k rows * ~1KB per row ≈ 50MB expected
        max_reasonable_increase = 200  # 200MB max reasonable increase

        assert memory_increase <= max_reasonable_increase, \
            f"Excessive memory usage under load: +{memory_increase:.1f} MB"

        # Memory should be mostly cleaned up
        cleanup_efficiency = (peak_memory - final_memory) / memory_increase
        assert cleanup_efficiency >= 0.5, \
            f"Poor memory cleanup: {cleanup_efficiency:.1f}% of peak memory retained"


class TestPerformanceBaselines:
    """Establish and test against performance baselines."""

    def test_performance_baselines(self):
        """Test against established performance baselines."""
        # These baselines should be established based on reference hardware
        # and updated when significant performance improvements are made

        baselines = {
            'sma_calculation': {'min_throughput': 50000, 'max_time': 0.1},
            'rsi_calculation': {'min_throughput': 30000, 'max_time': 0.2},
            'memory_optimization': {'min_savings_percent': 15, 'max_time': 1.0},
            'dataframe_processing': {'min_throughput': 20000, 'max_time': 2.0}
        }

        # Test SMA baseline
        data_size = 100000
        prices = np.random.uniform(100, 200, data_size)

        start_time = time.time()
        sma = VectorizedTradingOps.calculate_sma_numba(prices, window=20)
        sma_time = time.time() - start_time

        sma_throughput = data_size / sma_time
        assert sma_throughput >= baselines['sma_calculation']['min_throughput']
        assert sma_time <= baselines['sma_calculation']['max_time']

        # Test RSI baseline
        start_time = time.time()
        rsi = VectorizedTradingOps.calculate_rsi_numba(prices, period=14)
        rsi_time = time.time() - start_time

        rsi_throughput = data_size / rsi_time
        assert rsi_throughput >= baselines['rsi_calculation']['min_throughput']
        assert rsi_time <= baselines['rsi_calculation']['max_time']

        # Test memory optimization baseline
        df = pd.DataFrame({
            'timestamp': pd.date_range('2024-01-01', periods=50000, freq='1min'),
            'price': prices[:50000],
            'volume': np.random.randint(1000, 10000, 50000)
        })

        original_memory = df.memory_usage(deep=True).sum()

        start_time = time.time()
        optimized_df = optimize_trading_data_pipeline(df)
        opt_time = time.time() - start_time

        optimized_memory = optimized_df.memory_usage(deep=True).sum()
        savings_percent = (original_memory - optimized_memory) / original_memory * 100

        assert savings_percent >= baselines['memory_optimization']['min_savings_percent']
        assert opt_time <= baselines['memory_optimization']['max_time']


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
