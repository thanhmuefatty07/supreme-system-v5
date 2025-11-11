#!/usr/bin/env python3
"""
Memory stress tests for Supreme System V5.

Tests system performance and memory usage under high load conditions,
large datasets, concurrent operations, and memory-intensive scenarios.
"""

import pytest
import psutil
import os
import gc
import time
import threading
from unittest.mock import Mock, patch
import pandas as pd
import numpy as np
from memory_profiler import profile as memory_profile

from src.backtesting.production_backtester import ProductionBacktester
from src.strategies.breakout import ImprovedBreakoutStrategy
from src.data.data_pipeline import DataPipeline


class TestMemoryStress:
    """Test memory usage and performance under stress conditions."""

    @pytest.fixture
    def large_market_data(self):
        """Generate large dataset for memory testing."""
        np.random.seed(42)
        n_points = 50000  # Large dataset

        # Generate realistic price data
        base_price = 50000.0
        timestamps = pd.date_range('2024-01-01', periods=n_points, freq='1min')

        # Create trending, volatile price movements
        trend = np.linspace(0, 0.5, n_points)  # Upward trend
        volatility = np.random.normal(0, 0.003, n_points)  # 0.3% volatility
        gaps = np.random.choice([0, 0.01, -0.01], n_points, p=[0.96, 0.02, 0.02])

        price_changes = trend + volatility + gaps
        prices = base_price * np.cumprod(1 + price_changes)

        data = pd.DataFrame({
            'timestamp': timestamps,
            'open': prices * np.random.uniform(0.999, 1.001, n_points),
            'high': prices * np.random.uniform(1.0005, 1.003, n_points),
            'low': prices * np.random.uniform(0.997, 0.9995, n_points),
            'close': prices,
            'volume': np.random.uniform(1000, 10000, n_points)
        })

        return data

    @pytest.fixture
    def memory_monitor(self):
        """Memory usage monitoring utility."""
        class MemoryMonitor:
            def __init__(self):
                self.process = psutil.Process(os.getpid())
                self.baseline = self.process.memory_info().rss / 1024 / 1024  # MB

            def get_current_usage(self):
                """Get current memory usage in MB."""
                return self.process.memory_info().rss / 1024 / 1024

            def get_memory_delta(self):
                """Get memory usage delta from baseline."""
                return self.get_current_usage() - self.baseline

        return MemoryMonitor()

    def test_large_dataset_backtesting_memory_usage(self, large_market_data, memory_monitor):
        """Test memory usage during backtesting with large datasets."""
        # Force garbage collection before test
        gc.collect()

        initial_memory = memory_monitor.get_current_usage()

        strategy = ImprovedBreakoutStrategy()
        backtester = ProductionBacktester()

        # Run backtest on large dataset
        results = backtester.run_backtest(
            strategy=strategy,
            data=large_market_data,
            initial_capital=100000
        )

        final_memory = memory_monitor.get_current_usage()
        memory_delta = final_memory - initial_memory

        # Memory usage should be reasonable (< 500MB delta)
        assert memory_delta < 500, f"Memory usage too high: {memory_delta}MB"

        # Results should be valid
        assert results is not None
        assert 'total_return' in results

    def test_concurrent_strategy_execution_memory(self, large_market_data, memory_monitor):
        """Test memory usage with concurrent strategy execution."""
        gc.collect()

        initial_memory = memory_monitor.get_current_usage()

        strategies = [
            ImprovedBreakoutStrategy(lookback_period=20),
            ImprovedBreakoutStrategy(lookback_period=30),
            ImprovedBreakoutStrategy(lookback_period=40),
        ]

        backtester = ProductionBacktester()

        # Run multiple backtests concurrently
        threads = []
        results = []

        def run_backtest(strategy, data_chunk):
            result = backtester.run_backtest(
                strategy=strategy,
                data=data_chunk,
                initial_capital=10000
            )
            results.append(result)

        # Split data for concurrent processing
        chunk_size = len(large_market_data) // len(strategies)
        for i, strategy in enumerate(strategies):
            start_idx = i * chunk_size
            end_idx = (i + 1) * chunk_size if i < len(strategies) - 1 else len(large_market_data)
            data_chunk = large_market_data.iloc[start_idx:end_idx]

            thread = threading.Thread(target=run_backtest, args=(strategy, data_chunk))
            threads.append(thread)
            thread.start()

        # Wait for all threads
        for thread in threads:
            thread.join()

        final_memory = memory_monitor.get_current_usage()
        memory_delta = final_memory - initial_memory

        # Memory usage should be controlled
        assert memory_delta < 300, f"Concurrent memory usage too high: {memory_delta}MB"
        assert len(results) == len(strategies)

    def test_data_pipeline_memory_optimization(self, large_market_data, memory_monitor):
        """Test data pipeline memory optimization."""
        gc.collect()

        initial_memory = memory_monitor.get_current_usage()

        pipeline = DataPipeline()

        # Process large dataset through pipeline
        processed_data = pipeline.process_data(large_market_data, "BTCUSDT")

        final_memory = memory_monitor.get_current_usage()
        memory_delta = final_memory - initial_memory

        # Memory usage should be efficient
        assert memory_delta < 200, f"Pipeline memory usage too high: {memory_delta}MB"

        # Data should be processed
        if processed_data is not None:
            assert len(processed_data) > 0

    def test_memory_leak_prevention(self, memory_monitor):
        """Test prevention of memory leaks during repeated operations."""
        gc.collect()

        initial_memory = memory_monitor.get_current_usage()

        strategy = ImprovedBreakoutStrategy()
        backtester = ProductionBacktester()

        # Run multiple backtests in sequence
        for i in range(10):
            # Create fresh data each time to test cleanup
            test_data = large_market_data.iloc[:1000]  # Smaller chunk

            results = backtester.run_backtest(
                strategy=strategy,
                data=test_data,
                initial_capital=10000
            )

            # Force cleanup between runs
            del results
            gc.collect()

        final_memory = memory_monitor.get_current_usage()
        memory_delta = final_memory - initial_memory

        # Memory should not accumulate significantly (< 50MB)
        assert memory_delta < 50, f"Memory leak detected: {memory_delta}MB accumulation"

    def test_large_parameter_optimization_memory(self, large_market_data, memory_monitor):
        """Test memory usage during parameter optimization."""
        gc.collect()

        initial_memory = memory_monitor.get_current_usage()

        from src.backtesting.walk_forward import AdvancedWalkForwardOptimizer

        optimizer = AdvancedWalkForwardOptimizer()

        # Test parameter optimization (limited scope for memory test)
        param_ranges = {
            'lookback_period': (10, 30),
            'breakout_threshold': (0.01, 0.05)
        }

        # Use smaller dataset for optimization test
        test_data = large_market_data.iloc[:5000]

        try:
            results = optimizer.optimize_parameters(
                strategy_class=ImprovedBreakoutStrategy,
                data=test_data,
                param_ranges=param_ranges,
                max_evals=20  # Limited evaluations for memory test
            )

            final_memory = memory_monitor.get_current_usage()
            memory_delta = final_memory - initial_memory

            # Optimization should be memory efficient
            assert memory_delta < 150, f"Optimization memory usage too high: {memory_delta}MB"

        except Exception:
            # Optimization might fail due to complexity, but should not cause memory issues
            final_memory = memory_monitor.get_current_usage()
            memory_delta = final_memory - initial_memory
            assert memory_delta < 100, f"Even failed optimization should not use excessive memory: {memory_delta}MB"

    def test_dataframe_memory_optimization(self, memory_monitor):
        """Test DataFrame memory optimization utilities."""
        gc.collect()

        initial_memory = memory_monitor.get_current_usage()

        # Create memory-inefficient DataFrame
        large_df = pd.DataFrame({
            'float64_col': np.random.random(100000),
            'int64_col': np.random.randint(0, 100, 100000),
            'object_col': ['string'] * 100000,
            'category_col': pd.Categorical(np.random.choice(['A', 'B', 'C'], 100000))
        })

        # Convert to less memory-efficient types
        large_df['float64_col'] = large_df['float64_col'].astype(np.float32)
        large_df['int64_col'] = large_df['int64_col'].astype(np.int32)

        memory_before = large_df.memory_usage(deep=True).sum() / 1024 / 1024  # MB

        # Apply memory optimization
        from src.utils.data_utils import optimize_dataframe_memory
        optimized_df = optimize_dataframe_memory(large_df)

        memory_after = optimized_df.memory_usage(deep=True).sum() / 1024 / 1024  # MB

        final_memory = memory_monitor.get_current_usage()
        memory_delta = final_memory - initial_memory

        # Should reduce memory usage
        assert memory_after <= memory_before
        # Process memory should not grow excessively
        assert memory_delta < 100, f"Memory optimization process used too much memory: {memory_delta}MB"

    def test_garbage_collection_under_load(self, large_market_data, memory_monitor):
        """Test garbage collection behavior under memory load."""
        gc.collect()

        initial_memory = memory_monitor.get_current_usage()

        # Create many temporary objects
        temp_objects = []
        for i in range(100):
            # Create temporary DataFrames
            temp_df = large_market_data.copy()
            temp_objects.append(temp_df)

            # Process each one
            strategy = ImprovedBreakoutStrategy()
            signal = strategy.generate_signal(temp_df.iloc[:50])
            temp_objects.append(signal)

        # Clear references
        del temp_objects
        gc.collect()  # Force garbage collection

        final_memory = memory_monitor.get_current_usage()
        memory_delta = final_memory - initial_memory

        # Memory should be reclaimed after GC
        assert memory_delta < 100, f"Memory not properly reclaimed: {memory_delta}MB remaining"

    def test_memory_fragmentation_handling(self, memory_monitor):
        """Test handling of memory fragmentation during operations."""
        import sys

        gc.collect()

        initial_memory = memory_monitor.get_current_usage()

        # Create fragmented memory pattern
        fragments = []
        for i in range(50):
            # Allocate varying sizes to create fragmentation
            size = np.random.randint(1000, 10000)
            fragment = np.zeros(size, dtype=np.float64)
            fragments.append(fragment)

            # Simulate processing
            if i % 10 == 0:
                # Periodically process and free some memory
                processed = np.sum(fragment)
                fragments = fragments[-10:]  # Keep only recent fragments

        # Clean up
        del fragments
        gc.collect()

        final_memory = memory_monitor.get_current_usage()
        memory_delta = final_memory - initial_memory

        # Should handle fragmentation without excessive memory usage
        assert memory_delta < 200, f"Memory fragmentation handling failed: {memory_delta}MB"

    def test_peak_memory_monitoring(self, large_market_data, memory_monitor):
        """Test peak memory usage monitoring during intensive operations."""
        gc.collect()

        peak_memory = memory_monitor.get_current_usage()

        # Perform memory-intensive operations
        for i in range(20):
            # Create multiple large DataFrames
            temp_data = large_market_data.copy()
            temp_data['extra_col'] = np.random.random(len(temp_data))

            # Perform computations
            strategy = ImprovedBreakoutStrategy()
            signals = []
            for j in range(0, len(temp_data), 100):  # Process in chunks
                chunk = temp_data.iloc[j:j+100]
                signal = strategy.generate_signal(chunk)
                signals.append(signal)

            # Track peak memory
            current_memory = memory_monitor.get_current_usage()
            peak_memory = max(peak_memory, current_memory)

            # Clean up
            del temp_data, signals
            gc.collect()

        peak_delta = peak_memory - memory_monitor.baseline

        # Peak memory should be reasonable (< 800MB above baseline)
        assert peak_delta < 800, f"Peak memory usage too high: {peak_delta}MB above baseline"

    def test_memory_efficient_data_structures(self, memory_monitor):
        """Test memory-efficient data structures and algorithms."""
        gc.collect()

        initial_memory = memory_monitor.get_current_usage()

        # Test efficient data structures for trading data
        from collections import deque

        # Use deque for efficient append/pop operations
        price_queue = deque(maxlen=1000)
        signal_queue = deque(maxlen=1000)

        # Simulate real-time data processing
        for i in range(2000):
            price = 50000 + np.random.normal(0, 100)
            price_queue.append(price)

            # Generate signals from recent data
            if len(price_queue) >= 20:
                recent_prices = list(price_queue)[-20:]
                # Simple signal generation
                signal = 1 if recent_prices[-1] > np.mean(recent_prices) else -1
                signal_queue.append(signal)

        final_memory = memory_monitor.get_current_usage()
        memory_delta = final_memory - initial_memory

        # Efficient structures should use minimal memory
        assert memory_delta < 50, f"Efficient structures used too much memory: {memory_delta}MB"
        assert len(price_queue) == 1000  # Should maintain maxlen
        assert len(signal_queue) == 1000
