#!/usr/bin/env python3
"""
Supreme System V5 - LIVE MARKET DATA INTEGRATION TEST
Critical validation of real market data processing and latency requirements

Tests integration with Coinbase, Binance, Kraken APIs with sub-150ms latency
and memory leak detection for continuous operation
"""

import asyncio
import aiohttp
import json
import logging
import psutil
import signal
import sys
import time
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
import argparse
import numpy as np

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('live_market_integration_test.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class MarketDataSource:
    """Base class for market data sources"""

    def __init__(self, name: str, base_url: str):
        self.name = name
        self.base_url = base_url
        self.session: Optional[aiohttp.ClientSession] = None
        self.request_count = 0
        self.error_count = 0
        self.latency_samples: List[float] = []

    async def initialize(self):
        """Initialize HTTP session"""
        self.session = aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=5.0))

    async def close(self):
        """Close HTTP session"""
        if self.session:
            await self.session.close()

    async def get_ticker(self, symbol: str) -> Dict[str, Any]:
        """Get ticker data - override in subclasses"""
        raise NotImplementedError

    def record_latency(self, latency: float):
        """Record latency sample"""
        self.latency_samples.append(latency)
        if len(self.latency_samples) > 1000:  # Keep last 1000 samples
            self.latency_samples = self.latency_samples[-500:]

class CoinbaseSource(MarketDataSource):
    """Coinbase Pro API integration"""

    def __init__(self):
        super().__init__("Coinbase", "https://api.pro.coinbase.com")

    async def get_ticker(self, symbol: str) -> Dict[str, Any]:
        """Get Coinbase ticker data"""
        start_time = time.time()

        try:
            # Coinbase uses BTC-USD format
            coinbase_symbol = symbol.replace('-', '-')
            url = f"{self.base_url}/products/{coinbase_symbol}/ticker"

            async with self.session.get(url) as response:
                if response.status == 200:
                    data = await response.json()
                    latency = time.time() - start_time
                    self.record_latency(latency)

                    return {
                        'source': self.name,
                        'symbol': symbol,
                        'price': float(data.get('price', 0)),
                        'bid': float(data.get('best_bid', 0)),
                        'ask': float(data.get('best_ask', 0)),
                        'volume': float(data.get('volume_24h', 0)),
                        'timestamp': datetime.fromisoformat(data.get('time', datetime.now().isoformat())),
                        'latency_ms': latency * 1000
                    }
                else:
                    self.error_count += 1
                    raise Exception(f"HTTP {response.status}")

        except Exception as e:
            self.error_count += 1
            raise Exception(f"Coinbase API error: {e}")

class BinanceSource(MarketDataSource):
    """Binance API integration"""

    def __init__(self):
        super().__init__("Binance", "https://api.binance.com")

    async def get_ticker(self, symbol: str) -> Dict[str, Any]:
        """Get Binance ticker data"""
        start_time = time.time()

        try:
            # Binance uses BTCUSDT format
            binance_symbol = symbol.replace('-', '')
            url = f"{self.base_url}/api/v3/ticker/24hr?symbol={binance_symbol}"

            async with self.session.get(url) as response:
                if response.status == 200:
                    data = await response.json()
                    latency = time.time() - start_time
                    self.record_latency(latency)

                    return {
                        'source': self.name,
                        'symbol': symbol,
                        'price': float(data.get('lastPrice', 0)),
                        'bid': float(data.get('bidPrice', 0)),
                        'ask': float(data.get('askPrice', 0)),
                        'volume': float(data.get('volume', 0)),
                        'timestamp': datetime.now(),  # Binance doesn't provide timestamp in this endpoint
                        'latency_ms': latency * 1000
                    }
                else:
                    self.error_count += 1
                    raise Exception(f"HTTP {response.status}")

        except Exception as e:
            self.error_count += 1
            raise Exception(f"Binance API error: {e}")

class KrakenSource(MarketDataSource):
    """Kraken API integration"""

    def __init__(self):
        super().__init__("Kraken", "https://api.kraken.com")

    async def get_ticker(self, symbol: str) -> Dict[str, Any]:
        """Get Kraken ticker data"""
        start_time = time.time()

        try:
            # Kraken uses XXBTZUSD format
            kraken_symbol = symbol.replace('BTC-', 'XXBTZ').replace('ETH-', 'XETHZ').replace('USD', 'USD')
            url = f"{self.base_url}/0/public/Ticker?pair={kraken_symbol}"

            async with self.session.get(url) as response:
                if response.status == 200:
                    data = await response.json()
                    latency = time.time() - start_time
                    self.record_latency(latency)

                    if 'result' in data and kraken_symbol in data['result']:
                        ticker_data = data['result'][kraken_symbol]

                        return {
                            'source': self.name,
                            'symbol': symbol,
                            'price': float(ticker_data['c'][0]),  # Last trade closed price
                            'bid': float(ticker_data['b'][0]),    # Best bid
                            'ask': float(ticker_data['a'][0]),    # Best ask
                            'volume': float(ticker_data['v'][1]), # Volume
                            'timestamp': datetime.now(),
                            'latency_ms': latency * 1000
                        }
                    else:
                        raise Exception("Invalid response format")

                else:
                    self.error_count += 1
                    raise Exception(f"HTTP {response.status}")

        except Exception as e:
            self.error_count += 1
            raise Exception(f"Kraken API error: {e}")

class MarketDataAggregator:
    """Aggregates data from multiple market sources"""

    def __init__(self, sources: List[str]):
        self.sources: List[MarketDataSource] = []
        self.price_history: Dict[str, List[float]] = {}
        self.latency_stats: Dict[str, List[float]] = {}

        # Initialize sources
        for source_name in sources:
            if source_name.lower() == 'coinbase':
                self.sources.append(CoinbaseSource())
            elif source_name.lower() == 'binance':
                self.sources.append(BinanceSource())
            elif source_name.lower() == 'kraken':
                self.sources.append(KrakenSource())

    async def initialize(self):
        """Initialize all data sources"""
        for source in self.sources:
            await source.initialize()
            self.price_history[source.name] = []
            self.latency_stats[source.name] = []

        logger.info(f"Initialized {len(self.sources)} market data sources")

    async def close(self):
        """Close all data sources"""
        for source in self.sources:
            await source.close()

    async def get_aggregated_data(self, symbol: str) -> Dict[str, Any]:
        """Get aggregated market data from all sources"""
        tasks = [source.get_ticker(symbol) for source in self.sources]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        aggregated_data = {
            'symbol': symbol,
            'timestamp': datetime.now().isoformat(),
            'sources': [],
            'aggregated': {}
        }

        successful_results = []

        for result in results:
            if isinstance(result, Exception):
                aggregated_data['sources'].append({
                    'source': 'unknown',
                    'error': str(result),
                    'success': False
                })
            else:
                aggregated_data['sources'].append(result)
                successful_results.append(result)

                # Update price history
                self.price_history[result['source']].append(result['price'])
                self.latency_stats[result['source']].append(result['latency_ms'])

                # Keep only last 1000 samples
                if len(self.price_history[result['source']]) > 1000:
                    self.price_history[result['source']] = self.price_history[result['source']][-500:]
                if len(self.latency_stats[result['source']]) > 1000:
                    self.latency_stats[result['source']] = self.latency_stats[result['source']][-500:]

        # Calculate aggregated metrics
        if successful_results:
            prices = [r['price'] for r in successful_results if r['price'] > 0]
            latencies = [r['latency_ms'] for r in successful_results]

            aggregated_data['aggregated'] = {
                'price_count': len(prices),
                'avg_price': np.mean(prices) if prices else 0,
                'price_spread': max(prices) - min(prices) if len(prices) > 1 else 0,
                'avg_latency_ms': np.mean(latencies) if latencies else 0,
                'max_latency_ms': max(latencies) if latencies else 0,
                'min_latency_ms': min(latencies) if latencies else 0
            }

        return aggregated_data

    def get_source_stats(self) -> Dict[str, Any]:
        """Get statistics for each data source"""
        stats = {}

        for source in self.sources:
            source_name = source.name
            prices = self.price_history.get(source_name, [])
            latencies = self.latency_stats.get(source_name, [])

            stats[source_name] = {
                'request_count': source.request_count,
                'error_count': source.error_count,
                'success_rate': (source.request_count - source.error_count) / max(source.request_count, 1),
                'price_samples': len(prices),
                'avg_latency_ms': np.mean(latencies) if latencies else 0,
                'max_latency_ms': max(latencies) if latencies else 0,
                'price_volatility': np.std(prices) if len(prices) > 1 else 0
            }

        return stats

class LiveMarketIntegrationTester:
    """Main live market data integration testing engine"""

    def __init__(self, sources: List[str], duration_hours: float = 2.0,
                 latency_alert_ms: int = 150, memory_limit_gb: float = 2.2):
        self.sources = sources
        self.duration_hours = duration_hours
        self.latency_alert_ms = latency_alert_ms
        self.memory_limit_gb = memory_limit_gb

        # Core components
        self.aggregator = MarketDataAggregator(sources)
        self.process = psutil.Process()

        # Test state
        self.is_running = False
        self.test_start_time: Optional[datetime] = None

        # Results tracking
        self.market_data_samples: List[Dict[str, Any]] = []
        self.latency_violations: List[Dict[str, Any]] = []
        self.memory_stats: List[Dict[str, Any]] = []
        self.error_events: List[Dict[str, Any]] = []

        # Results
        self.results = {
            'configuration': {
                'sources': sources,
                'duration_hours': duration_hours,
                'latency_alert_ms': latency_alert_ms,
                'memory_limit_gb': memory_limit_gb
            },
            'market_data_samples': [],
            'latency_violations': [],
            'memory_stats': [],
            'error_events': [],
            'source_stats': {},
            'success': False
        }

        logger.info(f"Live market integration tester initialized - Sources: {sources}, "
                   f"Duration: {duration_hours}h, Latency alert: {latency_alert_ms}ms")

    def signal_handler(self, signum, frame):
        """Handle graceful shutdown"""
        logger.info("Shutdown signal received - stopping market integration test")
        self.is_running = False

    def monitor_memory_usage(self) -> Dict[str, Any]:
        """Monitor memory usage during test"""
        memory_info = self.process.memory_info()
        system_memory = psutil.virtual_memory()

        return {
            'timestamp': datetime.now().isoformat(),
            'process_rss_mb': memory_info.rss / (1024 * 1024),
            'process_vms_mb': memory_info.vms / (1024 * 1024),
            'system_used_mb': system_memory.used / (1024 * 1024),
            'system_available_mb': system_memory.available / (1024 * 1024),
            'memory_limit_exceeded': (memory_info.rss / (1024 ** 3)) > self.memory_limit_gb
        }

    async def validate_latency_requirements(self, data: Dict[str, Any]):
        """Validate latency requirements for market data"""
        violations = []

        for source_data in data.get('sources', []):
            if isinstance(source_data, dict) and 'latency_ms' in source_data:
                latency = source_data['latency_ms']
                if latency > self.latency_alert_ms:
                    violation = {
                        'timestamp': datetime.now().isoformat(),
                        'source': source_data.get('source', 'unknown'),
                        'latency_ms': latency,
                        'threshold_ms': self.latency_alert_ms,
                        'violation_amount': latency - self.latency_alert_ms
                    }
                    violations.append(violation)
                    self.latency_violations.append(violation)

        if violations:
            logger.warning(f"🚨 Latency violations detected: {len(violations)} sources exceeded {self.latency_alert_ms}ms")

        return violations

    async def run_integration_test(self) -> Dict[str, Any]:
        """Execute the live market data integration test"""
        self.test_start_time = datetime.now()
        end_time = self.test_start_time + timedelta(hours=self.duration_hours)
        self.is_running = True

        logger.info("🚀 STARTING LIVE MARKET DATA INTEGRATION TEST")
        logger.info(f"Sources: {self.sources}")
        logger.info(f"Duration: {self.duration_hours} hours")
        logger.info(f"Latency Alert Threshold: {self.latency_alert_ms}ms")

        # Setup signal handlers
        signal.signal(signal.SIGINT, self.signal_handler)
        signal.signal(signal.SIGTERM, self.signal_handler)

        # Initialize data sources
        await self.aggregator.initialize()

        try:
            symbols = ['BTC-USD', 'ETH-USD']
            sample_count = 0
            last_memory_check = 0
            last_status_update = 0

            while self.is_running and datetime.now() < end_time:
                current_time = time.time()

                # Collect market data from all sources for all symbols
                for symbol in symbols:
                    try:
                        market_data = await self.aggregator.get_aggregated_data(symbol)

                        # Validate latency requirements
                        await self.validate_latency_requirements(market_data)

                        # Record sample
                        self.market_data_samples.append(market_data)
                        self.results['market_data_samples'].append(market_data)
                        sample_count += 1

                    except Exception as e:
                        error_event = {
                            'timestamp': datetime.now().isoformat(),
                            'symbol': symbol,
                            'error': str(e),
                            'error_type': type(e).__name__
                        }
                        self.error_events.append(error_event)
                        self.results['error_events'].append(error_event)
                        logger.error(f"Error fetching {symbol} data: {e}")

                # Monitor memory every 30 seconds
                if current_time - last_memory_check >= 30:
                    memory_stats = self.monitor_memory_usage()
                    self.memory_stats.append(memory_stats)
                    self.results['memory_stats'].append(memory_stats)

                    # Check memory limit
                    if memory_stats['memory_limit_exceeded']:
                        error_msg = ".1f"                        logger.error(error_msg)
                        self.results['error_events'].append({
                            'timestamp': memory_stats['timestamp'],
                            'error': error_msg,
                            'error_type': 'MemoryLimitExceeded'
                        })
                        self.is_running = False
                        break

                    last_memory_check = current_time

                # Status update every 5 minutes
                if current_time - last_status_update >= 300:
                    elapsed_hours = (datetime.now() - self.test_start_time).total_seconds() / 3600
                    progress = (elapsed_hours / self.duration_hours) * 100

                    source_stats = self.aggregator.get_source_stats()
                    total_requests = sum(stats['request_count'] for stats in source_stats.values())
                    total_errors = sum(stats['error_count'] for stats in source_stats.values())

                    logger.info(f"Progress: {progress:.1f}% | "
                              f"Samples: {sample_count} | "
                              f"Requests: {total_requests} | "
                              f"Errors: {total_errors} | "
                              f"Latency Violations: {len(self.latency_violations)}")

                    last_status_update = current_time

                # Rate limiting: 2 requests per second max (to avoid API limits)
                await asyncio.sleep(0.5)

            # Test completed successfully
            if datetime.now() >= end_time:
                self.results['success'] = True
                self.results['source_stats'] = self.aggregator.get_source_stats()
                logger.info("✅ LIVE MARKET DATA INTEGRATION TEST COMPLETED")

        except Exception as e:
            error_msg = f"Integration test failed: {str(e)}"
            logger.error(error_msg, exc_info=True)
            self.results['error_events'].append({
                'timestamp': datetime.now().isoformat(),
                'error': error_msg,
                'error_type': type(e).__name__
            })

        finally:
            self.is_running = False
            await self.aggregator.close()

        return self.results

    def analyze_results(self) -> Dict[str, Any]:
        """Analyze test results"""
        analysis = {
            'latency_analysis': {},
            'reliability_analysis': {},
            'memory_analysis': {},
            'performance_analysis': {},
            'success_criteria': {}
        }

        market_samples = self.results['market_data_samples']
        latency_violations = self.results['latency_violations']
        memory_stats = self.results['memory_stats']
        source_stats = self.results.get('source_stats', {})

        # Latency analysis
        if market_samples:
            all_latencies = []
            for sample in market_samples:
                for source_data in sample.get('sources', []):
                    if isinstance(source_data, dict) and 'latency_ms' in source_data:
                        all_latencies.append(source_data['latency_ms'])

            if all_latencies:
                analysis['latency_analysis'] = {
                    'total_measurements': len(all_latencies),
                    'avg_latency_ms': np.mean(all_latencies),
                    'max_latency_ms': max(all_latencies),
                    'min_latency_ms': min(all_latencies),
                    'latency_violations': len(latency_violations),
                    'violation_rate': len(latency_violations) / len(all_latencies),
                    'p95_latency_ms': np.percentile(all_latencies, 95),
                    'p99_latency_ms': np.percentile(all_latencies, 99)
                }

        # Reliability analysis
        total_requests = sum(stats.get('request_count', 0) for stats in source_stats.values())
        total_errors = sum(stats.get('error_count', 0) for stats in source_stats.values())

        analysis['reliability_analysis'] = {
            'total_requests': total_requests,
            'total_errors': total_errors,
            'overall_success_rate': (total_requests - total_errors) / max(total_requests, 1),
            'source_reliability': source_stats
        }

        # Memory analysis
        if memory_stats:
            memory_usage = [s['process_rss_mb'] for s in memory_stats]
            analysis['memory_analysis'] = {
                'max_memory_mb': max(memory_usage),
                'avg_memory_mb': np.mean(memory_usage),
                'memory_limit_gb': self.memory_limit_gb,
                'memory_limit_exceeded': any(s['memory_limit_exceeded'] for s in memory_stats)
            }

        # Performance analysis
        analysis['performance_analysis'] = {
            'total_samples': len(market_samples),
            'samples_per_hour': len(market_samples) / max(self.duration_hours, 0.1),
            'latency_violations': len(latency_violations),
            'error_events': len(self.results['error_events'])
        }

        # Success criteria
        analysis['success_criteria'] = {
            'latency_compliance': len(latency_violations) == 0,
            'memory_compliance': not any(s['memory_limit_exceeded'] for s in memory_stats),
            'reliability_threshold': analysis['reliability_analysis']['overall_success_rate'] >= 0.95,
            'data_collection_success': len(market_samples) >= 100  # At least 100 samples
        }

        return analysis

    def save_results(self, output_file: Optional[str] = None) -> str:
        """Save test results to file"""
        if not output_file:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file = f"live_market_integration_test_results_{timestamp}.json"

        # Add analysis to results
        self.results['analysis'] = self.analyze_results()

        with open(output_file, 'w') as f:
            json.dump(self.results, f, indent=2, default=str)

        logger.info(f"Results saved to {output_file}")
        return output_file

    def print_summary(self):
        """Print test summary"""
        print("\n" + "=" * 70)
        print("📊 LIVE MARKET DATA INTEGRATION TEST RESULTS")
        print("=" * 70)

        analysis = self.analyze_results()

        if self.results['success']:
            print("✅ TEST PASSED - Live market data integration successful")

            latency = analysis['latency_analysis']
            reliability = analysis['reliability_analysis']
            memory = analysis['memory_analysis']
            performance = analysis['performance_analysis']

            print("⚡ Latency Analysis:"            print(".1f"            print(".1f"            print(f"   Violations: {latency['latency_violations']}")
            print(".1f"
            print("🔄 Reliability Analysis:"            print(f"   Total Requests: {reliability['total_requests']}")
            print(f"   Success Rate: {reliability['overall_success_rate']:.1f}%")
            print(f"   Error Rate: {(1 - reliability['overall_success_rate'])*100:.1f}%")

            print("💾 Memory Analysis:"            print(".1f"            print(f"   Memory Limit Exceeded: {'❌ YES' if memory['memory_limit_exceeded'] else '✅ NO'}")

            print("📈 Performance Analysis:"            print(f"   Total Samples: {performance['total_samples']}")
            print(".1f"            print(f"   Error Events: {performance['error_events']}")

        else:
            print("❌ TEST FAILED")
            for error in self.results['error_events'][-3:]:
                print(f"🔴 {error['error']}")

        criteria = analysis['success_criteria']
        print("🎯 Success Criteria:"        print(f"   Latency Compliance: {'✅' if criteria['latency_compliance'] else '❌'}")
        print(f"   Memory Compliance: {'✅' if criteria['memory_compliance'] else '❌'}")
        print(f"   Reliability Threshold: {'✅' if criteria['reliability_threshold'] else '❌'}")
        print(f"   Data Collection: {'✅' if criteria['data_collection_success'] else '❌'}")

        print("=" * 70)

def main():
    parser = argparse.ArgumentParser(description='Live Market Data Integration Test for Supreme System V5')
    parser.add_argument('--sources', nargs='+', default=['coinbase', 'binance'],
                       help='Market data sources to test (coinbase, binance, kraken)')
    parser.add_argument('--duration', type=float, default=2.0,
                       help='Test duration in hours (default: 2.0)')
    parser.add_argument('--latency-alert', type=int, default=150,
                       help='Latency alert threshold in milliseconds (default: 150)')
    parser.add_argument('--memory-limit', type=float, default=2.2,
                       help='Memory limit in GB (default: 2.2)')
    parser.add_argument('--output', type=str,
                       help='Output file for results (default: auto-generated)')

    args = parser.parse_args()

    print("📊 SUPREME SYSTEM V5 - LIVE MARKET DATA INTEGRATION TEST")
    print("=" * 65)
    print(f"Sources: {args.sources}")
    print(f"Duration: {args.duration} hours")
    print(f"Latency Alert: {args.latency_alert}ms")
    print(f"Memory Limit: {args.memory_limit}GB")

    # Run the test
    tester = LiveMarketIntegrationTester(
        sources=args.sources,
        duration_hours=args.duration,
        latency_alert_ms=args.latency_alert,
        memory_limit_gb=args.memory_limit
    )

    def signal_handler(signum, frame):
        logger.info("Shutdown signal received")
        tester.is_running = False

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    try:
        results = asyncio.run(tester.run_integration_test())

        # Save results
        output_file = tester.save_results(args.output)

        # Print summary
        tester.print_summary()

        # Exit with appropriate code
        analysis = tester.analyze_results()
        criteria = analysis['success_criteria']
        all_criteria_met = all(criteria.values())

        sys.exit(0 if results['success'] and all_criteria_met else 1)

    except KeyboardInterrupt:
        logger.info("Test interrupted by user")
        tester.save_results(args.output)
        sys.exit(1)
    except Exception as e:
        logger.error(f"Critical test failure: {e}", exc_info=True)
        tester.save_results(args.output)
        sys.exit(1)

if __name__ == "__main__":
    main()
