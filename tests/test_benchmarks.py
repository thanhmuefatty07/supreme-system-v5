# tests/test_benchmarks.py
"""
Benchmark Tests - ULTRA SFL Implementation
Performance comparison between Rust and Python indicators
"""

import time
import numpy as np
import pytest
from typing import List, Tuple, Callable
from dataclasses import dataclass

# Import TA library for Python fallbacks
try:
    import ta
    TA_AVAILABLE = True
except ImportError:
    TA_AVAILABLE = False
    ta = None

# Import Rust indicators
try:
    from supreme_engine_rs import (
        fast_ema, fast_rsi, fast_sma, fast_macd,
        fast_bollinger_bands, fast_vwap, fast_stochastic
    )
    RUST_AVAILABLE = True
except ImportError:
    RUST_AVAILABLE = False

# Import Python implementations
from supreme_system_v5.strategies import TechnicalIndicators


@dataclass
class BenchmarkResult:
    """Result of a benchmark run"""
    name: str
    data_size: int
    implementation: str
    execution_time: float
    throughput: float  # operations per second
    memory_usage: float = 0.0


@dataclass
class BenchmarkConfig:
    """Configuration for benchmark tests"""
    data_sizes: List[int] = None
    iterations: int = 3
    warmup_iterations: int = 1

    def __post_init__(self):
        if self.data_sizes is None:
            self.data_sizes = [1000, 10000, 50000, 100000]


class IndicatorBenchmark:
    """Benchmark suite for technical indicators"""

    def __init__(self, config: BenchmarkConfig = None):
        self.config = config or BenchmarkConfig()
        self.results: List[BenchmarkResult] = []

        # Generate test data
        self.test_data = self._generate_test_data()

    def _generate_test_data(self) -> Dict[int, Tuple[List[float], List[float], List[float], List[float]]]:
        """Generate test datasets of various sizes"""
        np.random.seed(42)  # For reproducible results

        data = {}
        for size in self.config.data_sizes:
            # Generate realistic price data with trends and volatility
            base_price = 50000.0
            volatility = 0.02  # 2% daily volatility

            # Random walk with drift
            prices = [base_price]
            for i in range(size - 1):
                change = np.random.normal(0.0001, volatility)  # Small upward drift
                new_price = prices[-1] * (1 + change)
                prices.append(new_price)

            # Generate OHLCV data
            highs = [p * (1 + abs(np.random.normal(0, volatility/2))) for p in prices]
            lows = [p * (1 - abs(np.random.normal(0, volatility/2))) for p in prices]
            closes = prices
            volumes = [np.random.uniform(1000000, 10000000) for _ in range(size)]

            data[size] = (closes, highs, lows, volumes)

        return data

    def benchmark_indicator(self, name: str, func: Callable, data_size: int,
                          implementation: str, *args, **kwargs) -> BenchmarkResult:
        """Benchmark a single indicator function"""

        # Get test data
        closes, highs, lows, volumes = self.test_data[data_size]

        # Prepare arguments based on indicator type
        if name in ['ema', 'sma', 'rsi']:
            indicator_args = (closes, 14)  # period = 14
        elif name == 'macd':
            indicator_args = (closes, 12, 26, 9)  # fast, slow, signal
        elif name == 'bollinger_bands':
            indicator_args = (closes, 20, 2.0)  # period, std_dev
        elif name == 'vwap':
            indicator_args = (highs, lows, closes, volumes)
        elif name == 'stochastic':
            indicator_args = (highs, lows, closes, 14, 3)  # k_period, d_period
        else:
            indicator_args = (closes,)

        # Warmup
        for _ in range(self.config.warmup_iterations):
            try:
                func(*indicator_args, **kwargs)
            except:
                pass  # Some functions might fail during warmup

        # Benchmark
        start_time = time.perf_counter()

        for _ in range(self.config.iterations):
            result = func(*indicator_args, **kwargs)

        end_time = time.perf_counter()

        # Calculate metrics
        total_time = end_time - start_time
        avg_time = total_time / self.config.iterations
        throughput = data_size / avg_time  # data points per second

        return BenchmarkResult(
            name=name,
            data_size=data_size,
            implementation=implementation,
            execution_time=avg_time,
            throughput=throughput
        )

    def run_all_benchmarks(self) -> List[BenchmarkResult]:
        """Run benchmarks for all indicators and implementations"""
        results = []

        for data_size in self.config.data_sizes:
            print(f"📊 Benchmarking with {data_size} data points...")

            # EMA
            if RUST_AVAILABLE:
                try:
                    result = self.benchmark_indicator('ema', fast_ema, data_size, 'rust')
                    results.append(result)
                    print(".4f"                except Exception as e:
                    print(f"⚠️ Rust EMA failed: {e}")

            try:
                result = self.benchmark_indicator('ema', TechnicalIndicators.calculate_ema, data_size, 'python')
                results.append(result)
                print(".4f"            except Exception as e:
                print(f"⚠️ Python EMA failed: {e}")

            # RSI
            if RUST_AVAILABLE:
                try:
                    result = self.benchmark_indicator('rsi', fast_rsi, data_size, 'rust')
                    results.append(result)
                    print(".4f"                except Exception as e:
                    print(f"⚠️ Rust RSI failed: {e}")

            try:
                result = self.benchmark_indicator('rsi', TechnicalIndicators.calculate_rsi, data_size, 'python')
                results.append(result)
                print(".4f"            except Exception as e:
                print(f"⚠️ Python RSI failed: {e}")

            # SMA
            if RUST_AVAILABLE:
                try:
                    result = self.benchmark_indicator('sma', fast_sma, data_size, 'rust')
                    results.append(result)
                    print(".4f"                except Exception as e:
                    print(f"⚠️ Rust SMA failed: {e}")

            try:
                result = self.benchmark_indicator('sma', TechnicalIndicators.calculate_sma, data_size, 'python')
                results.append(result)
                print(".4f"            except Exception as e:
                print(f"⚠️ Python SMA failed: {e}")

            # MACD (only test on smaller datasets due to complexity)
            if data_size <= 50000:
                if RUST_AVAILABLE:
                    try:
                        result = self.benchmark_indicator('macd', fast_macd, data_size, 'rust')
                        results.append(result)
                        print(".4f"                    except Exception as e:
                        print(f"⚠️ Rust MACD failed: {e}")

                try:
                    result = self.benchmark_indicator('macd', TechnicalIndicators.calculate_macd, data_size, 'python')
                    results.append(result)
                    print(".4f"                except Exception as e:
                    print(f"⚠️ Python MACD failed: {e}")

        return results

    def print_summary(self, results: List[BenchmarkResult]):
        """Print comprehensive benchmark summary"""
        print("\n" + "=" * 100)
        print("🚀 SUPREME SYSTEM V5 - INDICATOR BENCHMARK RESULTS")
        print("=" * 100)

        # Group by indicator and data size
        by_indicator_size = {}
        for result in results:
            key = (result.name, result.data_size)
            if key not in by_indicator_size:
                by_indicator_size[key] = {}
            by_indicator_size[key][result.implementation] = result

        # Print results table
        print("<8")
        print("-" * 80)

        for (indicator, data_size), implementations in by_indicator_size.items():
            print("<8")

            rust_result = implementations.get('rust')
            python_result = implementations.get('python')

            if rust_result and python_result:
                speedup = python_result.execution_time / rust_result.execution_time
                print(".1f"                print(".1f"                print(".2f"            elif rust_result:
                print(".4f"            elif python_result:
                print(".4f"
        # Performance analysis
        print("
📈 PERFORMANCE ANALYSIS:"        print("=" * 50)

        # Calculate average speedup
        speedups = []
        for (indicator, data_size), implementations in by_indicator_size.items():
            rust_result = implementations.get('rust')
            python_result = implementations.get('python')
            if rust_result and python_result:
                speedup = python_result.execution_time / rust_result.execution_time
                speedups.append(speedup)

        if speedups:
            avg_speedup = sum(speedups) / len(speedups)
            max_speedup = max(speedups)
            min_speedup = min(speedups)

            print(".1f"            print(".1f"            print(".1f"
        # Scalability analysis
        print("
📊 SCALABILITY ANALYSIS:"        print("=" * 50)

        for indicator in ['ema', 'rsi', 'sma']:
            indicator_results = [(r.data_size, r.execution_time, r.implementation)
                               for r in results
                               if r.name == indicator]

            if len(indicator_results) >= 4:  # Need multiple data points
                print(f"\n🔍 {indicator.upper()} Scalability:")
                for size, time_taken, impl in sorted(indicator_results):
                    efficiency = size / time_taken if time_taken > 0 else 0
                    print("10d")


@pytest.fixture
def benchmark_config():
    """Benchmark configuration fixture"""
    return BenchmarkConfig(
        data_sizes=[1000, 10000, 25000],  # Smaller for CI
        iterations=3,
        warmup_iterations=1
    )


@pytest.mark.benchmark
def test_indicator_benchmarks(benchmark_config):
    """Run comprehensive indicator benchmarks"""
    if not RUST_AVAILABLE and not TA_AVAILABLE:
        pytest.skip("Neither Rust indicators nor TA library available")

    benchmark = IndicatorBenchmark(benchmark_config)
    results = benchmark.run_all_benchmarks()

    # Print results
    benchmark.print_summary(results)

    # Basic assertions
    assert len(results) > 0, "No benchmark results generated"

    # Check that we have some Rust results if available
    if RUST_AVAILABLE:
        rust_results = [r for r in results if r.implementation == 'rust']
        assert len(rust_results) > 0, "No Rust benchmark results"

    # Check that we have Python results
    python_results = [r for r in results if r.implementation == 'python']
    assert len(python_results) > 0, "No Python benchmark results"

    # Verify performance is reasonable (> 1000 data points/sec for small datasets)
    for result in results:
        if result.data_size <= 10000:
            assert result.throughput > 1000, f"Poor performance for {result.name}: {result.throughput:.0f} pts/sec"


@pytest.mark.parametrize("data_size", [1000, 10000])
@pytest.mark.benchmark
def test_ema_performance(data_size, benchmark):
    """Benchmark EMA performance specifically"""
    if not RUST_AVAILABLE:
        pytest.skip("Rust indicators not available")

    # Generate test data
    np.random.seed(42)
    prices = np.random.normal(50000, 5000, data_size).tolist()

    # Benchmark Rust implementation
    result = benchmark(lambda: fast_ema(np.array(prices, dtype=np.float64), 14))

    # Should process at least 10000 data points per second
    throughput = data_size / result
    assert throughput > 10000, f"EMA too slow: {throughput:.0f} pts/sec"


@pytest.mark.parametrize("data_size", [1000, 10000])
@pytest.mark.benchmark
def test_rsi_performance(data_size, benchmark):
    """Benchmark RSI performance specifically"""
    if not RUST_AVAILABLE:
        pytest.skip("Rust indicators not available")

    # Generate test data
    np.random.seed(42)
    prices = np.random.normal(50000, 5000, data_size).tolist()

    # Benchmark Rust implementation
    result = benchmark(lambda: fast_rsi(np.array(prices, dtype=np.float64), 14))

    # Should process at least 10000 data points per second
    throughput = data_size / result
    assert throughput > 10000, f"RSI too slow: {throughput:.0f} pts/sec"


if __name__ == "__main__":
    # Run standalone benchmarks
    config = BenchmarkConfig()
    benchmark = IndicatorBenchmark(config)
    results = benchmark.run_all_benchmarks()
    benchmark.print_summary(results)
