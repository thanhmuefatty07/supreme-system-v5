#!/usr/bin/env python3
"""
Performance Profiler for Supreme System V5

Comprehensive performance analysis and optimization tools for trading system.
Provides detailed profiling, benchmarking, and optimization recommendations.
"""

import cProfile
import pstats
import time
import tracemalloc
import psutil
import os
from pathlib import Path
from typing import Dict, Any, List, Optional, Callable
from functools import wraps
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PerformanceProfiler:
    """Comprehensive performance profiling and optimization toolkit."""

    def __init__(self, output_dir: str = "performance_reports"):
        """
        Initialize performance profiler.

        Args:
            output_dir: Directory to save performance reports
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.process = psutil.Process(os.getpid())

    def profile_function(self, func: Callable, *args, **kwargs) -> Dict[str, Any]:
        """
        Profile a function's performance using cProfile.

        Args:
            func: Function to profile
            *args: Positional arguments for function
            **kwargs: Keyword arguments for function

        Returns:
            Dict containing profiling results
        """
        profiler = cProfile.Profile()
        start_time = time.time()

        # Start memory tracing
        tracemalloc.start()

        try:
            # Profile the function
            profiler.enable()
            result = func(*args, **kwargs)
            profiler.disable()

            # Collect memory stats
            current, peak = tracemalloc.get_traced_memory()

            # Collect CPU and time stats
            end_time = time.time()
            execution_time = end_time - start_time

            # Generate profiling report
            stats = pstats.Stats(profiler)
            stats.sort_stats('cumulative')

            # Save detailed report
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            report_file = self.output_dir / f"profile_{func.__name__}_{timestamp}.txt"
            stats.print_stats(file=open(report_file, 'w'))

            # Extract key metrics
            profile_stats = {
                'function_name': func.__name__,
                'execution_time': execution_time,
                'cpu_percent': self.process.cpu_percent(),
                'memory_current_mb': current / 1024 / 1024,
                'memory_peak_mb': peak / 1024 / 1024,
                'memory_usage_mb': self.process.memory_info().rss / 1024 / 1024,
                'report_file': str(report_file)
            }

            logger.info(f"Profiled {func.__name__}: {execution_time:.3f}s, "
                       f"Peak memory: {peak/1024/1024:.1f}MB")

            return {
                'result': result,
                'profile_stats': profile_stats,
                'success': True
            }

        except Exception as e:
            logger.error(f"Profiling failed for {func.__name__}: {e}")
            return {
                'result': None,
                'profile_stats': {},
                'success': False,
                'error': str(e)
            }
        finally:
            tracemalloc.stop()

    def benchmark_strategy(self, strategy_class, data_size: int = 10000,
                          iterations: int = 5) -> Dict[str, Any]:
        """
        Benchmark trading strategy performance.

        Args:
            strategy_class: Strategy class to benchmark
            data_size: Size of test data
            iterations: Number of benchmark iterations

        Returns:
            Dict containing benchmark results
        """
        # Generate test data
        dates = pd.date_range('2024-01-01', periods=data_size, freq='1min')
        data = pd.DataFrame({
            'timestamp': dates,
            'open': 100 + np.random.normal(0, 2, data_size),
            'high': 102 + np.random.normal(0, 1, data_size),
            'low': 98 + np.random.normal(0, 1, data_size),
            'close': 100 + np.random.normal(0, 2, data_size),
            'volume': np.random.randint(1000, 10000, data_size)
        })

        # Initialize strategy
        strategy = strategy_class("benchmark_strategy", {})

        results = []

        for i in range(iterations):
            logger.info(f"Benchmark iteration {i+1}/{iterations}")

            # Profile signal generation
            profile_result = self.profile_function(
                strategy.generate_trade_signal,
                data,
                100000.0  # portfolio value
            )

            if profile_result['success']:
                results.append(profile_result['profile_stats'])

        # Calculate averages
        if results:
            avg_time = np.mean([r['execution_time'] for r in results])
            avg_memory = np.mean([r['memory_peak_mb'] for r in results])
            throughput = data_size / avg_time  # signals per second

            benchmark_summary = {
                'strategy_name': strategy_class.__name__,
                'data_size': data_size,
                'iterations': iterations,
                'avg_execution_time': avg_time,
                'avg_memory_peak_mb': avg_memory,
                'throughput_signals_per_sec': throughput,
                'timestamp': datetime.now().isoformat()
            }

            # Save benchmark report
            report_file = self.output_dir / f"benchmark_{strategy_class.__name__}_{datetime.now():%Y%m%d_%H%M%S}.json"

            import json
            with open(report_file, 'w') as f:
                json.dump(benchmark_summary, f, indent=2, default=str)

            logger.info(f"Benchmark complete: {throughput:.1f} signals/sec, "
                       f"Avg time: {avg_time:.3f}s")

            return benchmark_summary

        return {}

    def memory_analysis(self, func: Callable, *args, **kwargs) -> Dict[str, Any]:
        """
        Detailed memory analysis of a function.

        Args:
            func: Function to analyze
            *args, **kwargs: Arguments for function

        Returns:
            Dict containing memory analysis results
        """
        tracemalloc.start()
        snapshots = []

        # Take initial snapshot
        snapshots.append(tracemalloc.take_snapshot())

        try:
            # Execute function
            start_time = time.time()
            result = func(*args, **kwargs)
            end_time = time.time()

            # Take final snapshot
            snapshots.append(tracemalloc.take_snapshot())

            # Analyze memory differences
            stats = snapshots[1].compare_to(snapshots[0], 'lineno')

            # Top memory consumers
            top_stats = stats[:10]  # Top 10 lines

            memory_report = {
                'execution_time': end_time - start_time,
                'memory_increase': sum(stat.size_diff for stat in stats),
                'top_memory_consumers': [
                    {
                        'file': stat.traceback[0].filename,
                        'line': stat.traceback[0].lineno,
                        'size_mb': stat.size_diff / 1024 / 1024,
                        'count': stat.count_diff
                    }
                    for stat in top_stats if stat.size_diff > 0
                ]
            }

            return {
                'result': result,
                'memory_report': memory_report,
                'success': True
            }

        except Exception as e:
            logger.error(f"Memory analysis failed: {e}")
            return {
                'result': None,
                'memory_report': {},
                'success': False,
                'error': str(e)
            }
        finally:
            tracemalloc.stop()

    def load_test(self, func: Callable, concurrent_users: int = 10,
                 duration_seconds: int = 60, *args, **kwargs) -> Dict[str, Any]:
        """
        Perform load testing with concurrent execution.

        Args:
            func: Function to load test
            concurrent_users: Number of concurrent executions
            duration_seconds: Test duration
            *args, **kwargs: Arguments for function

        Returns:
            Dict containing load test results
        """
        results = []
        errors = []

        def worker():
            """Worker function for load testing."""
            local_results = []
            start_time = time.time()

            while time.time() - start_time < duration_seconds:
                try:
                    result = func(*args, **kwargs)
                    local_results.append({
                        'success': True,
                        'execution_time': time.time() - time.time(),  # This will be overridden
                        'timestamp': time.time()
                    })
                except Exception as e:
                    errors.append({
                        'error': str(e),
                        'timestamp': time.time()
                    })

            return local_results

        logger.info(f"Starting load test: {concurrent_users} concurrent users for {duration_seconds}s")

        start_time = time.time()

        # Execute with thread pool
        with ThreadPoolExecutor(max_workers=concurrent_users) as executor:
            futures = [executor.submit(worker) for _ in range(concurrent_users)]

            for future in as_completed(futures):
                try:
                    results.extend(future.result())
                except Exception as e:
                    errors.append({'error': str(e)})

        end_time = time.time()
        total_duration = end_time - start_time

        # Calculate metrics
        successful_calls = len(results)
        failed_calls = len(errors)
        total_calls = successful_calls + failed_calls

        if results:
            execution_times = [r['execution_time'] for r in results if 'execution_time' in r]
            avg_response_time = np.mean(execution_times) if execution_times else 0
            throughput = successful_calls / total_duration  # calls per second
        else:
            avg_response_time = 0
            throughput = 0

        load_test_report = {
            'concurrent_users': concurrent_users,
            'test_duration_seconds': duration_seconds,
            'actual_duration_seconds': total_duration,
            'total_calls': total_calls,
            'successful_calls': successful_calls,
            'failed_calls': failed_calls,
            'success_rate': successful_calls / total_calls if total_calls > 0 else 0,
            'throughput_calls_per_sec': throughput,
            'avg_response_time': avg_response_time,
            'errors': errors[:10],  # First 10 errors
            'timestamp': datetime.now().isoformat()
        }

        # Save report
        report_file = self.output_dir / f"load_test_{func.__name__}_{datetime.now():%Y%m%d_%H%M%S}.json"
        import json
        with open(report_file, 'w') as f:
            json.dump(load_test_report, f, indent=2, default=str)

        logger.info(f"Load test complete: {throughput:.1f} calls/sec, "
                   f"Success rate: {load_test_report['success_rate']:.1%}")

        return load_test_report


def profile_decorator(output_dir: str = "performance_reports"):
    """
    Decorator for automatic performance profiling.

    Args:
        output_dir: Directory to save profiling reports

    Returns:
        Decorated function with profiling
    """
    def decorator(func):
        profiler = PerformanceProfiler(output_dir)

        @wraps(func)
        def wrapper(*args, **kwargs):
            logger.info(f"Starting performance profiling for {func.__name__}")

            result = profiler.profile_function(func, *args, **kwargs)

            if result['success']:
                stats = result['profile_stats']
                logger.info(f"Performance profile complete for {func.__name__}: "
                           f"{stats['execution_time']:.3f}s, "
                           f"{stats['memory_peak_mb']:.1f}MB peak memory")

            return result['result']

        return wrapper
    return decorator


def benchmark_strategies():
    """Benchmark all available trading strategies."""
    from src.strategies.momentum import MomentumStrategy
    from src.strategies.mean_reversion import MeanReversionStrategy
    from src.strategies.breakout import ImprovedBreakoutStrategy

    profiler = PerformanceProfiler()

    strategies = [
        (MomentumStrategy, "Momentum Strategy"),
        (MeanReversionStrategy, "Mean Reversion Strategy"),
        (ImprovedBreakoutStrategy, "Breakout Strategy")
    ]

    results = []

    for strategy_class, name in strategies:
        logger.info(f"Benchmarking {name}")
        try:
            result = profiler.benchmark_strategy(strategy_class, data_size=5000, iterations=3)
            results.append({
                'strategy': name,
                'performance': result
            })
        except Exception as e:
            logger.error(f"Failed to benchmark {name}: {e}")
            results.append({
                'strategy': name,
                'error': str(e)
            })

    # Save comprehensive benchmark report
    benchmark_file = profiler.output_dir / f"strategy_benchmark_{datetime.now():%Y%m%d_%H%M%S}.json"
    import json
    with open(benchmark_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)

    logger.info(f"Strategy benchmarking complete. Results saved to {benchmark_file}")
    return results


def memory_optimization_analysis():
    """Analyze memory usage patterns and provide optimization recommendations."""
    profiler = PerformanceProfiler()

    # Test large dataframe processing
    def process_large_data():
        # Simulate processing large trading data
        n_rows = 100000
        data = pd.DataFrame({
            'timestamp': pd.date_range('2024-01-01', periods=n_rows, freq='1min'),
            'open': 100 + np.random.normal(0, 2, n_rows),
            'high': 102 + np.random.normal(0, 1, n_rows),
            'low': 98 + np.random.normal(0, 1, n_rows),
            'close': 100 + np.random.normal(0, 2, n_rows),
            'volume': np.random.randint(1000, 10000, n_rows)
        })

        # Simulate strategy calculations
        data['returns'] = data['close'].pct_change()
        data['sma_20'] = data['close'].rolling(20).mean()
        data['sma_50'] = data['close'].rolling(50).mean()
        data['rsi'] = calculate_rsi_vectorized(data['close'], 14)

        return data

    logger.info("Analyzing memory usage patterns...")
    memory_result = profiler.memory_analysis(process_large_data)

    if memory_result['success']:
        report = memory_result['memory_report']
        logger.info(f"Memory analysis complete: {report['memory_increase']/1024/1024:.1f}MB increase")

        # Provide optimization recommendations
        recommendations = []

        if report['memory_increase'] > 100 * 1024 * 1024:  # > 100MB
            recommendations.append("Consider processing data in chunks")
            recommendations.append("Use dtypes optimization for DataFrames")
            recommendations.append("Implement lazy evaluation for large datasets")

        if report['top_memory_consumers']:
            for consumer in report['top_memory_consumers'][:3]:
                recommendations.append(f"Optimize memory usage in {consumer['file']}:{consumer['line']}")

        return {
            'memory_report': report,
            'recommendations': recommendations
        }

    return {}


def calculate_rsi_vectorized(prices: pd.Series, period: int = 14) -> pd.Series:
    """Calculate RSI using vectorized operations for better performance."""
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi


if __name__ == "__main__":
    # Run comprehensive performance analysis
    logger.info("Starting comprehensive performance analysis...")

    # Benchmark strategies
    strategy_results = benchmark_strategies()

    # Memory analysis
    memory_results = memory_optimization_analysis()

    # Load testing example
    def sample_trading_function():
        # Simulate a simple trading calculation
        data = pd.DataFrame({
            'close': np.random.uniform(100, 110, 100)
        })
        return data['close'].rolling(20).mean().iloc[-1]

    profiler = PerformanceProfiler()
    load_results = profiler.load_test(
        sample_trading_function,
        concurrent_users=5,
        duration_seconds=10
    )

    logger.info("Performance analysis complete!")
    logger.info(f"Strategy benchmarks: {len(strategy_results)} strategies tested")
    logger.info(f"Load test results: {load_results['throughput_calls_per_sec']:.1f} calls/sec")
    logger.info(f"Reports saved to: {profiler.output_dir}")
