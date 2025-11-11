"""
Load Testing and Stress Testing Utilities for Supreme System V5

Comprehensive load testing framework for performance validation under high concurrency.
Tests system stability, throughput, and resource usage under stress conditions.
"""

import asyncio
import time
import threading
import multiprocessing
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
from typing import Dict, List, Tuple, Any, Callable, Optional
import logging
import psutil
import statistics
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class LoadTestResult:
    """Results from a load test execution."""
    test_name: str
    duration_seconds: float
    total_requests: int
    successful_requests: int
    failed_requests: int
    success_rate: float
    avg_response_time: float
    min_response_time: float
    max_response_time: float
    p95_response_time: float
    p99_response_time: float
    throughput_requests_per_sec: float
    peak_memory_mb: float
    avg_cpu_percent: float
    errors: List[str] = field(default_factory=list)
    timestamps: List[float] = field(default_factory=list)
    response_times: List[float] = field(default_factory=list)


class LoadTester:
    """Advanced load testing framework."""

    def __init__(self, max_workers: int = None):
        """
        Initialize load tester.

        Args:
            max_workers: Maximum number of worker threads/processes
        """
        self.max_workers = max_workers or multiprocessing.cpu_count() * 2
        self.process = psutil.Process()

    def run_concurrent_load_test(self,
                               test_function: Callable,
                               concurrent_users: int,
                               duration_seconds: int,
                               ramp_up_seconds: int = 10,
                               *args, **kwargs) -> LoadTestResult:
        """
        Run a concurrent load test.

        Args:
            test_function: Function to test
            concurrent_users: Number of concurrent users
            duration_seconds: Test duration
            ramp_up_seconds: Time to ramp up to full concurrency
            *args, **kwargs: Arguments for test function

        Returns:
            LoadTestResult with comprehensive metrics
        """
        logger.info(f"Starting concurrent load test: {concurrent_users} users, "
                   f"{duration_seconds}s duration, {ramp_up_seconds}s ramp-up")

        results = []
        errors = []
        response_times = []
        timestamps = []

        start_time = time.time()
        test_end_time = start_time + duration_seconds

        # Ramp-up phase
        active_users = 0
        ramp_up_increment = concurrent_users / ramp_up_seconds if ramp_up_seconds > 0 else concurrent_users

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = []

            while time.time() < test_end_time:
                current_time = time.time()

                # Calculate target concurrent users based on ramp-up
                elapsed_ramp_time = current_time - start_time
                target_users = min(concurrent_users,
                                 int(ramp_up_increment * min(elapsed_ramp_time, ramp_up_seconds)))

                # Submit new tasks to reach target concurrency
                while len(futures) < target_users:
                    future = executor.submit(self._execute_test_with_timing,
                                           test_function, args, kwargs)
                    futures.append(future)

                # Collect completed results
                completed_futures = []
                for future in futures:
                    if future.done():
                        try:
                            result = future.result(timeout=0.1)
                            results.append(result)
                            if result['success']:
                                response_times.append(result['response_time'])
                                timestamps.append(result['timestamp'])
                            else:
                                errors.append(result.get('error', 'Unknown error'))
                        except Exception as e:
                            errors.append(f"Future collection error: {e}")
                        completed_futures.append(future)

                # Remove completed futures
                for completed in completed_futures:
                    futures.remove(completed)

                # Small delay to prevent tight looping
                time.sleep(0.01)

        # Calculate final metrics
        total_duration = time.time() - start_time
        successful_requests = len([r for r in results if r['success']])
        failed_requests = len(results) - successful_requests

        metrics = self._calculate_load_metrics(
            results, response_times, total_duration, errors
        )

        result = LoadTestResult(
            test_name=f"concurrent_load_{concurrent_users}users_{duration_seconds}s",
            duration_seconds=total_duration,
            total_requests=len(results),
            successful_requests=successful_requests,
            failed_requests=failed_requests,
            success_rate=successful_requests / len(results) if results else 0,
            avg_response_time=metrics['avg_response_time'],
            min_response_time=metrics['min_response_time'],
            max_response_time=metrics['max_response_time'],
            p95_response_time=metrics['p95_response_time'],
            p99_response_time=metrics['p99_response_time'],
            throughput_requests_per_sec=metrics['throughput'],
            peak_memory_mb=metrics['peak_memory_mb'],
            avg_cpu_percent=metrics['avg_cpu_percent'],
            errors=errors[:100],  # Limit error storage
            timestamps=timestamps,
            response_times=response_times
        )

        logger.info(f"Load test completed: {result.throughput_requests_per_sec:.1f} req/sec, "
                   f"{result.success_rate:.1%} success rate")

        return result

    def run_stress_test(self,
                       test_function: Callable,
                       max_concurrent_users: int,
                       duration_per_level: int = 30,
                       user_increment: int = 10,
                       *args, **kwargs) -> List[LoadTestResult]:
        """
        Run a stress test with increasing concurrency levels.

        Args:
            test_function: Function to test
            max_concurrent_users: Maximum concurrent users to test
            duration_per_level: Duration to test each concurrency level
            user_increment: Increment between concurrency levels
            *args, **kwargs: Arguments for test function

        Returns:
            List of LoadTestResult for each concurrency level
        """
        logger.info(f"Starting stress test: max {max_concurrent_users} users, "
                   f"{user_increment} user increments, {duration_per_level}s per level")

        results = []

        for concurrent_users in range(user_increment, max_concurrent_users + 1, user_increment):
            logger.info(f"Testing with {concurrent_users} concurrent users")

            result = self.run_concurrent_load_test(
                test_function,
                concurrent_users,
                duration_per_level,
                ramp_up_seconds=5,  # Quick ramp-up for stress test
                *args, **kwargs
            )

            results.append(result)

            # Check for breaking point (high failure rate or extreme response times)
            if result.success_rate < 0.8 or result.avg_response_time > 5.0:
                logger.warning(f"Breaking point detected at {concurrent_users} users: "
                             f"{result.success_rate:.1%} success, {result.avg_response_time:.2f}s avg response")
                break

            # Small delay between test levels
            time.sleep(2)

        logger.info(f"Stress test completed: tested up to {len(results) * user_increment} users")
        return results

    def run_spike_test(self,
                      test_function: Callable,
                      base_users: int,
                      spike_users: int,
                      spike_duration: int = 60,
                      total_duration: int = 300,
                      *args, **kwargs) -> LoadTestResult:
        """
        Run a spike test with sudden load increases.

        Args:
            test_function: Function to test
            base_users: Baseline concurrent users
            spike_users: Number of users during spike
            spike_duration: Duration of spike in seconds
            total_duration: Total test duration
            *args, **kwargs: Arguments for test function

        Returns:
            LoadTestResult for the entire spike test
        """
        logger.info(f"Starting spike test: base {base_users} users, "
                   f"spike {spike_users} users for {spike_duration}s")

        results = []
        errors = []
        response_times = []
        timestamps = []

        start_time = time.time()
        test_end_time = start_time + total_duration
        spike_start_time = start_time + (total_duration - spike_duration) / 2
        spike_end_time = spike_start_time + spike_duration

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = []

            while time.time() < test_end_time:
                current_time = time.time()

                # Determine current target concurrency
                if spike_start_time <= current_time <= spike_end_time:
                    target_users = spike_users
                else:
                    target_users = base_users

                # Adjust active users
                while len(futures) < target_users:
                    future = executor.submit(self._execute_test_with_timing,
                                           test_function, args, kwargs)
                    futures.append(future)

                # Clean up completed futures
                completed_futures = []
                for future in futures:
                    if future.done():
                        try:
                            result = future.result(timeout=0.1)
                            results.append(result)
                            if result['success']:
                                response_times.append(result['response_time'])
                                timestamps.append(result['timestamp'])
                            else:
                                errors.append(result.get('error', 'Unknown error'))
                        except Exception as e:
                            errors.append(f"Future collection error: {e}")
                        completed_futures.append(future)

                for completed in completed_futures:
                    futures.remove(completed)

                time.sleep(0.01)

        # Calculate metrics
        total_duration = time.time() - start_time
        successful_requests = len([r for r in results if r['success']])
        failed_requests = len(results) - successful_requests

        metrics = self._calculate_load_metrics(
            results, response_times, total_duration, errors
        )

        result = LoadTestResult(
            test_name=f"spike_test_base{base_users}_spike{spike_users}_{total_duration}s",
            duration_seconds=total_duration,
            total_requests=len(results),
            successful_requests=successful_requests,
            failed_requests=failed_requests,
            success_rate=successful_requests / len(results) if results else 0,
            **metrics
        )

        logger.info(f"Spike test completed: {result.throughput_requests_per_sec:.1f} req/sec")
        return result

    def _execute_test_with_timing(self, test_function: Callable,
                                args: tuple, kwargs: dict) -> Dict[str, Any]:
        """Execute test function with timing measurement."""
        start_time = time.time()

        try:
            result = test_function(*args, **kwargs)
            response_time = time.time() - start_time

            return {
                'success': True,
                'result': result,
                'response_time': response_time,
                'timestamp': time.time()
            }

        except Exception as e:
            response_time = time.time() - start_time

            return {
                'success': False,
                'error': str(e),
                'response_time': response_time,
                'timestamp': time.time()
            }

    def _calculate_load_metrics(self, results: List[Dict],
                              response_times: List[float],
                              duration: float,
                              errors: List[str]) -> Dict[str, float]:
        """Calculate comprehensive load test metrics."""
        if not response_times:
            return {
                'avg_response_time': 0.0,
                'min_response_time': 0.0,
                'max_response_time': 0.0,
                'p95_response_time': 0.0,
                'p99_response_time': 0.0,
                'throughput': 0.0,
                'peak_memory_mb': 0.0,
                'avg_cpu_percent': 0.0
            }

        # Response time metrics
        avg_response_time = statistics.mean(response_times)
        min_response_time = min(response_times)
        max_response_time = max(response_times)

        # Percentile calculations
        sorted_times = sorted(response_times)
        p95_index = int(len(sorted_times) * 0.95)
        p99_index = int(len(sorted_times) * 0.99)

        p95_response_time = sorted_times[min(p95_index, len(sorted_times) - 1)]
        p99_response_time = sorted_times[min(p99_index, len(sorted_times) - 1)]

        # Throughput
        throughput = len(results) / duration

        # Resource usage (simplified - would need more sophisticated monitoring)
        peak_memory_mb = self.process.memory_info().rss / 1024 / 1024
        avg_cpu_percent = self.process.cpu_percent()

        return {
            'avg_response_time': avg_response_time,
            'min_response_time': min_response_time,
            'max_response_time': max_response_time,
            'p95_response_time': p95_response_time,
            'p99_response_time': p99_response_time,
            'throughput': throughput,
            'peak_memory_mb': peak_memory_mb,
            'avg_cpu_percent': avg_cpu_percent
        }


def create_trading_load_test_scenario(strategy_class=None) -> Callable:
    """
    Create a realistic trading load test scenario.

    Args:
        strategy_class: Trading strategy class to test

    Returns:
        Test function for load testing
    """
    def trading_scenario():
        """Simulate a complete trading scenario."""
        try:
            # Generate random market data
            n_periods = np.random.randint(100, 1000)
            data = pd.DataFrame({
                'timestamp': pd.date_range('2024-01-01', periods=n_periods, freq='1min'),
                'open': 100 + np.random.normal(0, 2, n_periods),
                'high': 102 + np.random.normal(0, 1, n_periods),
                'low': 98 + np.random.normal(0, 1, n_periods),
                'close': 100 + np.random.normal(0, 2, n_periods),
                'volume': np.random.randint(1000, 10000, n_periods)
            })

            if strategy_class:
                # Initialize strategy
                strategy = strategy_class("load_test_strategy", {})

                # Generate trading signal
                signal = strategy.generate_trade_signal(data, 100000.0)

                # Simulate order execution
                if signal['action'] != 'HOLD':
                    # Mock order execution logic
                    time.sleep(np.random.uniform(0.001, 0.01))  # Simulate network delay

                return {'signal': signal, 'data_processed': n_periods}
            else:
                # Simple data processing test
                returns = data['close'].pct_change().mean()
                volatility = data['close'].pct_change().std()
                return {'returns': returns, 'volatility': volatility, 'data_processed': n_periods}

        except Exception as e:
            raise RuntimeError(f"Trading scenario failed: {e}")

    return trading_scenario


def run_comprehensive_load_test_suite(max_users: int = 50) -> Dict[str, Any]:
    """
    Run a comprehensive load test suite.

    Args:
        max_users: Maximum concurrent users for stress testing

    Returns:
        Comprehensive test results
    """
    logger.info("Starting comprehensive load test suite")

    tester = LoadTester()
    test_scenario = create_trading_load_test_scenario()

    results = {}

    # 1. Baseline performance test
    logger.info("Running baseline performance test...")
    baseline_result = tester.run_concurrent_load_test(
        test_scenario, concurrent_users=5, duration_seconds=30
    )
    results['baseline'] = baseline_result

    # 2. Concurrent load test
    logger.info("Running concurrent load test...")
    load_result = tester.run_concurrent_load_test(
        test_scenario, concurrent_users=20, duration_seconds=60
    )
    results['concurrent_load'] = load_result

    # 3. Stress test
    logger.info("Running stress test...")
    stress_results = tester.run_stress_test(
        test_scenario, max_concurrent_users=max_users,
        duration_per_level=20, user_increment=10
    )
    results['stress_test'] = stress_results

    # 4. Spike test
    logger.info("Running spike test...")
    spike_result = tester.run_spike_test(
        test_scenario, base_users=10, spike_users=max_users // 2,
        spike_duration=30, total_duration=120
    )
    results['spike_test'] = spike_result

    # Generate summary report
    summary = {
        'total_tests_run': len(results),
        'baseline_throughput': baseline_result.throughput_requests_per_sec,
        'max_stress_throughput': max([r.throughput_requests_per_sec for r in stress_results]) if stress_results else 0,
        'spike_test_success_rate': spike_result.success_rate,
        'recommended_max_users': min(max_users,
                                   max([r.successful_requests for r in stress_results]) if stress_results else max_users),
        'test_timestamp': datetime.now().isoformat()
    }

    results['summary'] = summary

    logger.info("Load test suite completed!")
    logger.info(f"Baseline throughput: {summary['baseline_throughput']:.1f} req/sec")
    logger.info(f"Max stress throughput: {summary['max_stress_throughput']:.1f} req/sec")

    return results


if __name__ == "__main__":
    # Run comprehensive load test suite
    results = run_comprehensive_load_test_suite(max_users=30)
    print("Load testing suite completed!")
