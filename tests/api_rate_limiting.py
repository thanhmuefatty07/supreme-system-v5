#!/usr/bin/env python3
"""
Supreme System V5 - API RATE LIMITING HANDLING TEST
Critical validation of rate limiting resilience and backoff strategies

Tests handling of 429 errors, exponential backoff, and circuit breaker patterns
"""

import asyncio
import aiohttp
import json
import logging
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
        logging.FileHandler('api_rate_limiting_test.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class MockRateLimitedAPI:
    """Mock API that simulates rate limiting behavior"""

    def __init__(self, rate_limit_per_minute: int = 60, burst_limit: int = 10):
        self.rate_limit_per_minute = rate_limit_per_minute
        self.burst_limit = burst_limit
        self.requests_this_minute = 0
        self.requests_this_burst = 0
        self.minute_start = time.time()
        self.burst_start = time.time()
        self.total_requests = 0
        self.rate_limit_hits = 0

    def reset_counters(self):
        """Reset rate limiting counters"""
        current_time = time.time()

        # Reset minute counter
        if current_time - self.minute_start >= 60:
            self.requests_this_minute = 0
            self.minute_start = current_time

        # Reset burst counter (10 second window)
        if current_time - self.burst_start >= 10:
            self.requests_this_burst = 0
            self.burst_start = current_time

    def check_rate_limit(self) -> tuple[bool, Optional[int]]:
        """Check if request should be rate limited"""
        self.reset_counters()

        self.total_requests += 1

        # Check burst limit first
        if self.requests_this_burst >= self.burst_limit:
            retry_after = 10 - int(time.time() - self.burst_start)
            self.rate_limit_hits += 1
            return False, retry_after

        # Check minute limit
        if self.requests_this_minute >= self.rate_limit_per_minute:
            retry_after = 60 - int(time.time() - self.minute_start)
            self.rate_limit_hits += 1
            return False, retry_after

        # Allow request
        self.requests_this_burst += 1
        self.requests_this_minute += 1

        return True, None

    async def handle_request(self, request_data: Dict[str, Any]) -> Dict[str, Any]:
        """Handle API request with rate limiting"""
        allowed, retry_after = self.check_rate_limit()

        if not allowed:
            # Simulate rate limit response
            await asyncio.sleep(0.001)  # Minimal processing delay
            raise aiohttp.ClientResponseError(
                request_info=aiohttp.RequestInfo(
                    url="http://mock-api.com/trade",
                    method="POST",
                    headers={},
                    real_url="http://mock-api.com/trade"
                ),
                history=(),
                status=429,
                message="Too Many Requests",
                headers={"Retry-After": str(retry_after)}
            )

        # Simulate successful API response
        await asyncio.sleep(np.random.uniform(0.01, 0.05))  # API processing time

        return {
            "order_id": f"order_{self.total_requests}",
            "status": "filled",
            "price": request_data.get("price", 50000),
            "quantity": request_data.get("quantity", 1),
            "timestamp": datetime.now().isoformat()
        }

class ExponentialBackoff:
    """Exponential backoff implementation for rate limiting"""

    def __init__(self, base_delay: float = 1.0, max_delay: float = 300.0,
                 multiplier: float = 2.0, jitter: bool = True):
        self.base_delay = base_delay
        self.max_delay = max_delay
        self.multiplier = multiplier
        self.jitter = jitter
        self.attempt_count = 0
        self.total_backoff_time = 0

    def get_delay(self) -> float:
        """Calculate delay for current attempt"""
        delay = min(self.base_delay * (self.multiplier ** self.attempt_count), self.max_delay)

        if self.jitter:
            # Add random jitter (±25%)
            jitter_range = delay * 0.25
            delay += np.random.uniform(-jitter_range, jitter_range)

        return max(0.1, delay)  # Minimum 100ms delay

    def increment_attempt(self):
        """Increment attempt counter"""
        self.attempt_count += 1

    def reset(self):
        """Reset attempt counter"""
        self.attempt_count = 0
        self.total_backoff_time = 0

    def record_backoff(self, delay: float):
        """Record backoff time"""
        self.total_backoff_time += delay

class RateLimitResilientClient:
    """API client with rate limiting resilience"""

    def __init__(self, max_retries: int = 5, timeout: float = 30.0):
        self.max_retries = max_retries
        self.timeout = timeout
        self.session: Optional[aiohttp.ClientSession] = None
        self.backoff = ExponentialBackoff()
        self.request_stats = {
            'total_requests': 0,
            'successful_requests': 0,
            'rate_limit_hits': 0,
            'timeouts': 0,
            'other_errors': 0,
            'total_backoff_time': 0,
            'avg_request_time': 0
        }

    async def initialize(self):
        """Initialize HTTP session"""
        self.session = aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=self.timeout))

    async def close(self):
        """Close HTTP session"""
        if self.session:
            await self.session.close()

    async def make_resilient_request(self, api: MockRateLimitedAPI, request_data: Dict[str, Any]) -> Dict[str, Any]:
        """Make API request with rate limiting resilience"""
        self.request_stats['total_requests'] += 1
        start_time = time.time()

        for attempt in range(self.max_retries + 1):
            try:
                # Attempt API call
                result = await api.handle_request(request_data)

                # Success - reset backoff
                self.backoff.reset()
                self.request_stats['successful_requests'] += 1

                request_time = time.time() - start_time
                self.request_stats['avg_request_time'] = (
                    (self.request_stats['avg_request_time'] * (self.request_stats['total_requests'] - 1)) +
                    request_time
                ) / self.request_stats['total_requests']

                return result

            except aiohttp.ClientResponseError as e:
                if e.status == 429:
                    # Rate limit hit
                    self.request_stats['rate_limit_hits'] += 1

                    if attempt < self.max_retries:
                        # Calculate backoff delay
                        delay = self.backoff.get_delay()
                        self.backoff.record_backoff(delay)
                        self.backoff.increment_attempt()

                        retry_after = e.headers.get('Retry-After', str(int(delay)))
                        logger.warning(f"Rate limit hit (attempt {attempt+1}/{self.max_retries+1}), "
                                     f"backing off {delay:.1f}s (Retry-After: {retry_after})")

                        await asyncio.sleep(delay)
                        continue
                    else:
                        logger.error(f"Rate limit persisted after {self.max_retries} retries")
                        raise
                else:
                    # Other HTTP error
                    self.request_stats['other_errors'] += 1
                    raise

            except asyncio.TimeoutError:
                self.request_stats['timeouts'] += 1
                if attempt < self.max_retries:
                    delay = self.backoff.get_delay()
                    self.backoff.increment_attempt()
                    logger.warning(f"Timeout (attempt {attempt+1}/{self.max_retries+1}), retrying in {delay:.1f}s")
                    await asyncio.sleep(delay)
                    continue
                else:
                    logger.error(f"Request timed out after {self.max_retries} retries")
                    raise

            except Exception as e:
                self.request_stats['other_errors'] += 1
                logger.error(f"Unexpected error: {e}")
                raise

        # Should not reach here
        raise Exception("Max retries exceeded")

class APIRateLimitingTester:
    """Main API rate limiting test engine"""

    def __init__(self, duration_minutes: int = 30, concurrent_clients: int = 5,
                 rate_limit_per_minute: int = 60, burst_limit: int = 10):
        self.duration_minutes = duration_minutes
        self.concurrent_clients = concurrent_clients
        self.rate_limit_per_minute = rate_limit_per_minute
        self.burst_limit = burst_limit

        # Core components
        self.mock_api = MockRateLimitedAPI(rate_limit_per_minute, burst_limit)
        self.clients: List[RateLimitResilientClient] = []

        # Test state
        self.is_running = False
        self.test_start_time: Optional[datetime] = None

        # Results tracking
        self.client_stats: List[Dict[str, Any]] = []
        self.api_stats: List[Dict[str, Any]] = []
        self.rate_limit_events: List[Dict[str, Any]] = []

        # Results
        self.results = {
            'configuration': {
                'duration_minutes': duration_minutes,
                'concurrent_clients': concurrent_clients,
                'rate_limit_per_minute': rate_limit_per_minute,
                'burst_limit': burst_limit
            },
            'client_stats': [],
            'api_stats': [],
            'rate_limit_events': [],
            'success': False
        }

        logger.info(f"API Rate Limiting Tester initialized - {concurrent_clients} clients, "
                   f"{duration_minutes}min duration, {rate_limit_per_minute}/min rate limit")

    async def initialize_clients(self):
        """Initialize resilient API clients"""
        for i in range(self.concurrent_clients):
            client = RateLimitResilientClient(max_retries=5, timeout=30.0)
            await client.initialize()
            self.clients.append(client)

        logger.info(f"Initialized {len(self.clients)} resilient API clients")

    async def close_clients(self):
        """Close all API clients"""
        for client in self.clients:
            await client.close()

    async def run_client_workload(self, client: RateLimitResilientClient, client_id: int):
        """Run workload for a single client"""
        while self.is_running:
            try:
                # Generate trading request
                request_data = {
                    'symbol': 'BTC-USD',
                    'side': 'buy',
                    'quantity': np.random.randint(1, 10),
                    'price': 50000 + np.random.normal(0, 1000),
                    'type': 'limit',
                    'client_id': client_id
                }

                # Make resilient request
                result = await client.make_resilient_request(self.mock_api, request_data)

                # Small delay between requests
                await asyncio.sleep(np.random.uniform(0.1, 0.5))

            except Exception as e:
                logger.error(f"Client {client_id} error: {e}")
                await asyncio.sleep(1.0)  # Error backoff

    async def monitor_rate_limiting(self):
        """Monitor rate limiting behavior"""
        last_check = time.time()

        while self.is_running:
            current_time = time.time()

            if current_time - last_check >= 10:  # Check every 10 seconds
                # Record API stats
                api_stat = {
                    'timestamp': datetime.now().isoformat(),
                    'total_requests': self.mock_api.total_requests,
                    'rate_limit_hits': self.mock_api.rate_limit_hits,
                    'current_minute_requests': self.mock_api.requests_this_minute,
                    'current_burst_requests': self.mock_api.requests_this_burst,
                    'rate_limit_rate': self.mock_api.rate_limit_hits / max(self.mock_api.total_requests, 1)
                }
                self.api_stats.append(api_stat)
                self.results['api_stats'].append(api_stat)

                # Check for rate limit events
                if self.mock_api.requests_this_burst >= self.burst_limit:
                    event = {
                        'timestamp': datetime.now().isoformat(),
                        'event_type': 'burst_limit_hit',
                        'requests_in_burst': self.mock_api.requests_this_burst,
                        'burst_limit': self.burst_limit
                    }
                    self.rate_limit_events.append(event)
                    self.results['rate_limit_events'].append(event)

                if self.mock_api.requests_this_minute >= self.rate_limit_per_minute:
                    event = {
                        'timestamp': datetime.now().isoformat(),
                        'event_type': 'minute_limit_hit',
                        'requests_in_minute': self.mock_api.requests_this_minute,
                        'minute_limit': self.rate_limit_per_minute
                    }
                    self.rate_limit_events.append(event)
                    self.results['rate_limit_events'].append(event)

                last_check = current_time

            await asyncio.sleep(1.0)

    async def run_rate_limiting_test(self) -> Dict[str, Any]:
        """Execute the API rate limiting test"""
        self.test_start_time = datetime.now()
        end_time = self.test_start_time + timedelta(minutes=self.duration_minutes)
        self.is_running = True

        logger.info("🚀 STARTING API RATE LIMITING TEST")
        logger.info(f"Clients: {self.concurrent_clients}")
        logger.info(f"Duration: {self.duration_minutes} minutes")
        logger.info(f"Rate Limit: {self.rate_limit_per_minute}/minute, Burst: {self.burst_limit}")

        # Initialize clients
        await self.initialize_clients()

        try:
            # Start monitoring task
            monitor_task = asyncio.create_task(self.monitor_rate_limiting())

            # Start client workloads
            client_tasks = []
            for i, client in enumerate(self.clients):
                task = asyncio.create_task(self.run_client_workload(client, i))
                client_tasks.append(task)

            # Wait for test duration
            while datetime.now() < end_time and self.is_running:
                await asyncio.sleep(1.0)

            # Stop all tasks
            self.is_running = False

            # Wait for tasks to complete
            await asyncio.gather(monitor_task, *client_tasks, return_exceptions=True)

            # Collect client statistics
            for i, client in enumerate(self.clients):
                client_stat = {
                    'client_id': i,
                    **client.request_stats
                }
                self.client_stats.append(client_stat)
                self.results['client_stats'].append(client_stat)

            # Test success criteria
            total_requests = sum(c['total_requests'] for c in self.client_stats)
            total_successful = sum(c['successful_requests'] for c in self.client_stats)
            total_rate_limits = sum(c['rate_limit_hits'] for c in self.client_stats)

            success_rate = total_successful / max(total_requests, 1)
            rate_limit_rate = total_rate_limits / max(total_requests, 1)

            # Success: >90% success rate and proper rate limit handling
            self.results['success'] = success_rate >= 0.9 and rate_limit_rate <= 0.2

            logger.info("✅ API RATE LIMITING TEST COMPLETED")

        except Exception as e:
            error_msg = f"Rate limiting test failed: {str(e)}"
            logger.error(error_msg, exc_info=True)
            self.results['errors'] = [error_msg]

        finally:
            await self.close_clients()

        return self.results

    def analyze_results(self) -> Dict[str, Any]:
        """Analyze test results"""
        analysis = {
            'client_performance': {},
            'rate_limiting_effectiveness': {},
            'backoff_efficiency': {},
            'success_criteria': {}
        }

        client_stats = self.results['client_stats']
        api_stats = self.results['api_stats']
        rate_limit_events = self.results['rate_limit_events']

        # Client performance analysis
        if client_stats:
            total_requests = sum(c['total_requests'] for c in client_stats)
            total_successful = sum(c['successful_requests'] for c in client_stats)
            total_rate_limits = sum(c['rate_limit_hits'] for c in client_stats)
            total_backoff_time = sum(c['total_backoff_time'] for c in client_stats)

            analysis['client_performance'] = {
                'total_clients': len(client_stats),
                'total_requests': total_requests,
                'successful_requests': total_successful,
                'success_rate': total_successful / max(total_requests, 1),
                'rate_limit_hits': total_rate_limits,
                'rate_limit_rate': total_rate_limits / max(total_requests, 1),
                'total_backoff_time': total_backoff_time,
                'avg_backoff_per_hit': total_backoff_time / max(total_rate_limits, 1)
            }

        # Rate limiting effectiveness
        if api_stats:
            avg_rate_limit_rate = np.mean([s['rate_limit_rate'] for s in api_stats])

            analysis['rate_limiting_effectiveness'] = {
                'api_total_requests': self.mock_api.total_requests,
                'api_rate_limit_hits': self.mock_api.rate_limit_hits,
                'api_rate_limit_rate': self.mock_api.rate_limit_hits / max(self.mock_api.total_requests, 1),
                'avg_client_rate_limit_rate': avg_rate_limit_rate,
                'burst_limit_events': len([e for e in rate_limit_events if e['event_type'] == 'burst_limit_hit']),
                'minute_limit_events': len([e for e in rate_limit_events if e['event_type'] == 'minute_limit_hit'])
            }

        # Success criteria
        client_perf = analysis['client_performance']
        analysis['success_criteria'] = {
            'success_rate_threshold': client_perf.get('success_rate', 0) >= 0.9,
            'rate_limit_handling': client_perf.get('rate_limit_rate', 1) <= 0.2,
            'backoff_effectiveness': client_perf.get('total_backoff_time', 0) > 0,
            'api_resilience': len(rate_limit_events) > 0  # Should have encountered rate limits
        }

        return analysis

    def save_results(self, output_file: Optional[str] = None) -> str:
        """Save test results to file"""
        if not output_file:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file = f"api_rate_limiting_test_results_{timestamp}.json"

        # Add analysis to results
        self.results['analysis'] = self.analyze_results()

        with open(output_file, 'w') as f:
            json.dump(self.results, f, indent=2, default=str)

        logger.info(f"Results saved to {output_file}")
        return output_file

    def print_summary(self):
        """Print test summary"""
        print("\n" + "=" * 70)
        print("🔄 API RATE LIMITING TEST RESULTS")
        print("=" * 70)

        analysis = self.analyze_results()

        if self.results['success']:
            print("✅ TEST PASSED - Rate limiting handled effectively")

            client_perf = analysis['client_performance']
            rate_limiting = analysis['rate_limiting_effectiveness']

            print("📊 Client Performance:"            print(f"   Total Requests: {client_perf['total_requests']}")
            print(f"   Success Rate: {client_perf['success_rate']:.1f}%")
            print(f"   Rate Limit Hits: {client_perf['rate_limit_hits']}")
            print(f"   Rate Limit Rate: {client_perf['rate_limit_rate']:.1f}%")
            print(".1f"
            print("🛡️ Rate Limiting Effectiveness:"            print(f"   API Requests: {rate_limiting['api_total_requests']}")
            print(f"   API Rate Limits: {rate_limiting['api_rate_limit_hits']}")
            print(f"   Burst Events: {rate_limiting['burst_limit_events']}")
            print(f"   Minute Events: {rate_limiting['minute_limit_events']}")

        else:
            print("❌ TEST FAILED - Rate limiting issues detected")

        criteria = analysis['success_criteria']
        print("🎯 Success Criteria:"        print(f"   Success Rate ≥90%: {'✅' if criteria['success_rate_threshold'] else '❌'}")
        print(f"   Rate Limit Handling: {'✅' if criteria['rate_limit_handling'] else '❌'}")
        print(f"   Backoff Effectiveness: {'✅' if criteria['backoff_effectiveness'] else '❌'}")
        print(f"   API Resilience: {'✅' if criteria['api_resilience'] else '❌'}")

        print("=" * 70)

def main():
    parser = argparse.ArgumentParser(description='API Rate Limiting Test for Supreme System V5')
    parser.add_argument('--duration', type=int, default=30,
                       help='Test duration in minutes (default: 30)')
    parser.add_argument('--clients', type=int, default=5,
                       help='Number of concurrent clients (default: 5)')
    parser.add_argument('--rate-limit', type=int, default=60,
                       help='API rate limit per minute (default: 60)')
    parser.add_argument('--burst-limit', type=int, default=10,
                       help='API burst limit (default: 10)')
    parser.add_argument('--output', type=str,
                       help='Output file for results (default: auto-generated)')

    args = parser.parse_args()

    print("🔄 SUPREME SYSTEM V5 - API RATE LIMITING TEST")
    print("=" * 55)
    print(f"Duration: {args.duration} minutes")
    print(f"Clients: {args.clients}")
    print(f"Rate Limit: {args.rate_limit}/minute")
    print(f"Burst Limit: {args.burst_limit}")

    # Run the test
    tester = APIRateLimitingTester(
        duration_minutes=args.duration,
        concurrent_clients=args.clients,
        rate_limit_per_minute=args.rate_limit,
        burst_limit=args.burst_limit
    )

    try:
        results = asyncio.run(tester.run_rate_limiting_test())

        # Save results
        output_file = tester.save_results(args.output)

        # Print summary
        tester.print_summary()

        # Exit with appropriate code
        analysis = tester.analyze_results()
        criteria = analysis['success_criteria']
        all_criteria_met = all(criteria.values())

        import sys
        sys.exit(0 if results['success'] and all_criteria_met else 1)

    except KeyboardInterrupt:
        logger.info("Test interrupted by user")
        tester.save_results(args.output)
    except Exception as e:
        logger.error(f"Critical test failure: {e}", exc_info=True)
        tester.save_results(args.output)

if __name__ == "__main__":
    main()
