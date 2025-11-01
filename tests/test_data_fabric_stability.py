# tests/test_data_fabric_stability.py
"""
Data Fabric 10-Minute Stability Test - ULTRA SFL Implementation
Tests Data Fabric QoS for 10 minutes with BTC/ETH symbols
Validates: quality_score >= 0.8, error_rate <= 10%
"""

import asyncio
import time
import pytest
from collections import defaultdict
from typing import Dict, List

from supreme_system_v5.data_fabric import DataAggregator
from supreme_system_v5.data_fabric.cache import CacheManager, DataCache


class DataFabricStabilityTest:
    """
    10-Minute Data Fabric Stability Test
    Validates QoS requirements: quality_score >= 0.8, error_rate <= 10%
    """

    def __init__(self, test_duration_minutes: int = 10):
        self.test_duration_seconds = test_duration_minutes * 60
        self.symbols = ["BTC-USDT", "ETH-USDT"]
        self.start_time = None
        self.end_time = None

        # Metrics collection
        self.metrics = {
            "total_requests": 0,
            "successful_requests": 0,
            "failed_requests": 0,
            "quality_scores": [],
            "response_times": [],
            "data_points_per_symbol": defaultdict(int),
            "errors_per_symbol": defaultdict(int),
            "source_failures": defaultdict(int),
        }

        # Components
        self.cache_manager = None
        self.data_aggregator = None

    async def setup(self):
        """Setup test components"""
        print("🔧 Setting up Data Fabric stability test...")

        # Initialize cache
        cache = DataCache()
        await cache.connect()
        self.cache_manager = CacheManager(cache)

        # Initialize data aggregator
        self.data_aggregator = DataAggregator(cache_manager=self.cache_manager)

        # Setup data sources
        await self._setup_data_sources()

        # Start cache manager
        await self.cache_manager.start(symbols=self.symbols)

        print("✅ Data Fabric components initialized")

    async def _setup_data_sources(self):
        """Setup data sources for testing"""
        sources_config = [
            ("binance", "BinancePublicConnector"),
            ("coingecko", "CoinGeckoConnector"),
            ("okx", "OKXPublicConnector"),
        ]

        for source_name, connector_class in sources_config:
            try:
                # Import connector dynamically
                if source_name == "binance":
                    from supreme_system_v5.data_fabric.connectors import BinancePublicConnector
                    connector = BinancePublicConnector()
                elif source_name == "coingecko":
                    from supreme_system_v5.data_fabric.connectors import CoinGeckoConnector
                    connector = CoinGeckoConnector()
                elif source_name == "okx":
                    from supreme_system_v5.data_fabric.connectors import OKXPublicConnector
                    connector = OKXPublicConnector()
                else:
                    continue

                # Create data source
                from supreme_system_v5.data_fabric.aggregator import DataSource
                source = DataSource(
                    name=f"{source_name}_test_source",
                    connector=connector,
                    priority=1,
                    weight=1.0
                )
                self.data_aggregator.add_source(source)
                print(f"✅ Added test data source: {source_name}")

            except Exception as e:
                print(f"⚠️ Failed to setup {source_name}: {e}")

    async def run_stability_test(self) -> Dict:
        """Run the 10-minute stability test"""
        print(f"🚀 Starting Data Fabric stability test ({self.test_duration_seconds}s)...")
        print("🎯 Testing symbols:", ", ".join(self.symbols))
        print("📊 QoS Requirements: quality_score >= 0.8, error_rate <= 10%")
        print("=" * 80)

        self.start_time = time.time()
        self.end_time = self.start_time + self.test_duration_seconds

        # Main test loop
        last_progress_time = self.start_time
        iteration_count = 0

        try:
            while time.time() < self.end_time:
                iteration_start = time.time()

                # Test all symbols
                for symbol in self.symbols:
                    await self._test_symbol_request(symbol)

                # Update iteration count
                iteration_count += 1

                # Progress reporting every 30 seconds
                current_time = time.time()
                if current_time - last_progress_time >= 30:
                    self._report_progress(current_time - self.start_time)
                    last_progress_time = current_time

                # Wait 1 second between iterations (10 iterations per minute)
                elapsed = time.time() - iteration_start
                if elapsed < 1.0:
                    await asyncio.sleep(1.0 - elapsed)

        except Exception as e:
            print(f"❌ Test failed with error: {e}")
            raise

        # Generate final results
        return await self._generate_results()

    async def _test_symbol_request(self, symbol: str):
        """Test a single symbol request"""
        request_start = time.time()

        try:
            self.metrics["total_requests"] += 1

            # Make request
            market_data = await self.data_aggregator.get_market_data(symbol)

            if market_data:
                self.metrics["successful_requests"] += 1
                self.metrics["data_points_per_symbol"][symbol] += 1

                # Calculate response time
                response_time = time.time() - request_start
                self.metrics["response_times"].append(response_time)

                # Get quality metrics if available
                quality_score = getattr(market_data, 'quality_score', None)
                if quality_score is not None:
                    self.metrics["quality_scores"].append(quality_score)

            else:
                self.metrics["failed_requests"] += 1
                self.metrics["errors_per_symbol"][symbol] += 1

        except Exception as e:
            self.metrics["failed_requests"] += 1
            self.metrics["errors_per_symbol"][symbol] += 1
            # Could log error details here if needed

    def _report_progress(self, elapsed_seconds: float):
        """Report test progress"""
        progress_pct = (elapsed_seconds / self.test_duration_seconds) * 100

        total_requests = self.metrics["total_requests"]
        successful = self.metrics["successful_requests"]
        failed = self.metrics["failed_requests"]

        success_rate = (successful / max(total_requests, 1)) * 100
        error_rate = (failed / max(total_requests, 1)) * 100

        avg_response_time = sum(self.metrics["response_times"][-100:]) / max(len(self.metrics["response_times"][-100:]), 1) * 1000  # Last 100 requests

        print(f"📊 Progress: {progress_pct:.1f}% | "
              f"Requests: {total_requests} | "
              f"Success: {success_rate:.1f}% | "
              f"Errors: {error_rate:.1f}% | "
              f"Avg Response: {avg_response_time:.0f}ms")

        # Symbol breakdown
        for symbol in self.symbols:
            symbol_data = self.metrics["data_points_per_symbol"][symbol]
            symbol_errors = self.metrics["errors_per_symbol"][symbol]
            symbol_total = symbol_data + symbol_errors
            if symbol_total > 0:
                symbol_success_rate = (symbol_data / symbol_total) * 100
                print(f"   {symbol}: {symbol_data} data points, {symbol_success_rate:.1f}% success")

    async def _generate_results(self) -> Dict:
        """Generate comprehensive test results"""
        total_time = self.end_time - self.start_time

        # Calculate final metrics
        total_requests = self.metrics["total_requests"]
        successful_requests = self.metrics["successful_requests"]
        failed_requests = self.metrics["failed_requests"]

        success_rate = (successful_requests / max(total_requests, 1)) * 100
        error_rate = (failed_requests / max(total_requests, 1)) * 100

        # Quality score analysis
        quality_scores = self.metrics["quality_scores"]
        avg_quality_score = sum(quality_scores) / max(len(quality_scores), 1)

        # Response time analysis
        response_times = self.metrics["response_times"]
        if response_times:
            avg_response_time = sum(response_times) / len(response_times)
            min_response_time = min(response_times)
            max_response_time = max(response_times)
            p95_response_time = sorted(response_times)[int(len(response_times) * 0.95)]
        else:
            avg_response_time = min_response_time = max_response_time = p95_response_time = 0

        # QoS validation
        qos_passed = avg_quality_score >= 0.8 and error_rate <= 10.0

        results = {
            "test_info": {
                "duration_seconds": total_time,
                "symbols_tested": self.symbols,
                "start_time": self.start_time,
                "end_time": self.end_time,
            },
            "overall_metrics": {
                "total_requests": total_requests,
                "successful_requests": successful_requests,
                "failed_requests": failed_requests,
                "success_rate_percent": success_rate,
                "error_rate_percent": error_rate,
                "requests_per_second": total_requests / max(total_time, 1),
            },
            "quality_metrics": {
                "average_quality_score": avg_quality_score,
                "quality_score_samples": len(quality_scores),
                "quality_score_range": f"{min(quality_scores) if quality_scores else 0:.2f} - {max(quality_scores) if quality_scores else 0:.2f}",
            },
            "performance_metrics": {
                "avg_response_time_ms": avg_response_time * 1000,
                "min_response_time_ms": min_response_time * 1000,
                "max_response_time_ms": max_response_time * 1000,
                "p95_response_time_ms": p95_response_time * 1000,
            },
            "per_symbol_metrics": {
                symbol: {
                    "data_points": self.metrics["data_points_per_symbol"][symbol],
                    "errors": self.metrics["errors_per_symbol"][symbol],
                    "success_rate": (self.metrics["data_points_per_symbol"][symbol] /
                                   max(self.metrics["data_points_per_symbol"][symbol] +
                                       self.metrics["errors_per_symbol"][symbol], 1)) * 100
                }
                for symbol in self.symbols
            },
            "qos_validation": {
                "quality_score_requirement": ">= 0.8",
                "quality_score_actual": f"{avg_quality_score:.3f}",
                "quality_score_passed": avg_quality_score >= 0.8,
                "error_rate_requirement": "<= 10.0%",
                "error_rate_actual": f"{error_rate:.1f}%",
                "error_rate_passed": error_rate <= 10.0,
                "overall_qos_passed": qos_passed,
            },
            "test_status": "PASSED" if qos_passed else "FAILED"
        }

        return results

    async def cleanup(self):
        """Cleanup test resources"""
        print("🧹 Cleaning up test resources...")
        try:
            if self.cache_manager:
                await self.cache_manager.stop()
            print("✅ Cleanup completed")
        except Exception as e:
            print(f"⚠️ Cleanup error: {e}")


@pytest.mark.asyncio
async def test_data_fabric_10min_stability():
    """
    10-Minute Data Fabric Stability Test
    Validates QoS requirements: quality_score >= 0.8, error_rate <= 10%
    """
    test = DataFabricStabilityTest(test_duration_minutes=10)

    try:
        # Setup
        await test.setup()

        # Run test
        results = await test.run_stability_test()

        # Print comprehensive results
        print("\n" + "=" * 100)
        print("🎯 DATA FABRIC 10-MINUTE STABILITY TEST RESULTS")
        print("=" * 100)

        test_info = results["test_info"]
        overall = results["overall_metrics"]
        quality = results["quality_metrics"]
        perf = results["performance_metrics"]
        qos = results["qos_validation"]

        print(f"⏱️  Test Duration: {test_info['duration_seconds']:.0f} seconds")
        print(f"🎯 Symbols Tested: {', '.join(test_info['symbols_tested'])}")
        print(f"📊 Total Requests: {overall['total_requests']}")
        print(f"✅ Successful: {overall['successful_requests']} ({overall['success_rate_percent']:.1f}%)")
        print(f"❌ Failed: {overall['failed_requests']} ({overall['error_rate_percent']:.1f}%)")
        print(f"⚡ Requests/sec: {overall['requests_per_second']:.2f}")

        print(f"\n📈 Quality Metrics:")
        print(f"  Average Quality Score: {quality['average_quality_score']:.3f} ({quality['quality_score_samples']} samples)")
        print(f"  Quality Score Range: {quality['quality_score_range']}")

        print(f"\n⚡ Performance Metrics:")
        print(f"  Avg Response Time: {perf['avg_response_time_ms']:.0f}ms")
        print(f"  P95 Response Time: {perf['p95_response_time_ms']:.0f}ms")
        print(f"  Response Time Range: {perf['min_response_time_ms']:.0f}ms - {perf['max_response_time_ms']:.0f}ms")

        print(f"\n🎯 QoS Validation:")
        print(f"  Quality Score: {qos['quality_score_actual']} {qos['quality_score_requirement']} - {'✅ PASS' if qos['quality_score_passed'] else '❌ FAIL'}")
        print(f"  Error Rate: {qos['error_rate_actual']} {qos['error_rate_requirement']} - {'✅ PASS' if qos['error_rate_passed'] else '❌ FAIL'}")

        print(f"\n🏆 Overall QoS Result: {'✅ PASSED' if qos['overall_qos_passed'] else '❌ FAILED'}")

        # Per-symbol breakdown
        print(f"\n📋 Per-Symbol Results:")
        for symbol, metrics in results["per_symbol_metrics"].items():
            print(f"  {symbol}: {metrics['data_points']} data points, {metrics['errors']} errors, {metrics['success_rate']:.1f}% success")

        print(f"\n🕐 Test Completed: {time.strftime('%Y-%m-%d %H:%M:%S')}")
        print("=" * 100)

        # Assert QoS requirements
        assert qos["quality_score_passed"], f"Quality score {qos['quality_score_actual']} < 0.8 requirement"
        assert qos["error_rate_passed"], f"Error rate {qos['error_rate_actual']} > 10% requirement"
        assert qos["overall_qos_passed"], "Data Fabric QoS requirements not met"

        print("🎉 All QoS requirements met! Data Fabric is stable.")

    finally:
        await test.cleanup()


if __name__ == "__main__":
    # Allow running as standalone script
    asyncio.run(test_data_fabric_10min_stability())
