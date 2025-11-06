#!/usr/bin/env python3
"""
Supreme System V5 - NETWORK FAILURE SIMULATION
Critical production readiness validation for trading system resilience

Network Failure Simulation với exponential backoff và circuit breaker
Optimized cho trading system resilience
"""

import asyncio
import aiohttp
import json
import logging
import random
import sys
import time
from datetime import datetime, timedelta
from typing import Optional, List, Dict, Any
import argparse

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('network_failure_simulation.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class CircuitBreaker:
    """Circuit breaker implementation for network resilience"""

    def __init__(self, failure_threshold: int = 5, reset_timeout: int = 60):
        self.failure_count = 0
        self.failure_threshold = failure_threshold
        self.reset_timeout = reset_timeout
        self.last_failure_time: Optional[datetime] = None
        self.state = "CLOSED"  # CLOSED, OPEN, HALF_OPEN
        self.success_count = 0
        self.half_open_success_threshold = 2

        logger.info(f"Circuit breaker initialized - threshold: {failure_threshold}, timeout: {reset_timeout}s")

    def can_execute(self) -> bool:
        """Check if request can be executed"""
        now = datetime.now()

        if self.state == "OPEN":
            if self.last_failure_time and (now - self.last_failure_time).seconds > self.reset_timeout:
                self.state = "HALF_OPEN"
                self.success_count = 0
                logger.info("Circuit breaker transitioning to HALF_OPEN")
                return True
            return False

        return True

    def on_success(self):
        """Handle successful request"""
        if self.state == "HALF_OPEN":
            self.success_count += 1
            if self.success_count >= self.half_open_success_threshold:
                self.state = "CLOSED"
                self.failure_count = 0
                logger.info("Circuit breaker transitioning to CLOSED")
        else:
            # Reset failure count on success in CLOSED state
            self.failure_count = max(0, self.failure_count - 1)

    def on_failure(self):
        """Handle failed request"""
        self.failure_count += 1
        self.last_failure_time = datetime.now()

        if self.failure_count >= self.failure_threshold:
            self.state = "OPEN"
            logger.warning(f"Circuit breaker tripped OPEN after {self.failure_count} failures")

    def get_status(self) -> Dict[str, Any]:
        """Get circuit breaker status"""
        return {
            'state': self.state,
            'failure_count': self.failure_count,
            'last_failure_time': self.last_failure_time.isoformat() if self.last_failure_time else None,
            'time_since_last_failure': (datetime.now() - self.last_failure_time).seconds if self.last_failure_time else None
        }

class ExponentialBackoff:
    """Exponential backoff implementation"""

    def __init__(self, base_delay: float = 1.0, max_delay: float = 60.0, multiplier: float = 2.0, jitter: bool = True):
        self.base_delay = base_delay
        self.max_delay = max_delay
        self.multiplier = multiplier
        self.jitter = jitter
        self.attempt_count = 0

    def get_delay(self) -> float:
        """Calculate delay for current attempt"""
        delay = min(self.base_delay * (self.multiplier ** self.attempt_count), self.max_delay)

        if self.jitter:
            # Add random jitter (±25%)
            jitter_range = delay * 0.25
            delay += random.uniform(-jitter_range, jitter_range)

        return max(0.1, delay)  # Minimum 100ms delay

    def increment_attempt(self):
        """Increment attempt counter"""
        self.attempt_count += 1

    def reset(self):
        """Reset attempt counter"""
        self.attempt_count = 0

class OptimizedNetworkFailureTest:
    """Network failure simulation optimized for trading system resilience"""

    def __init__(self, packet_loss: float = 0.1, latency_spike_ms: int = 500,
                 api_timeout_s: int = 30, simulate_rate_limiting: bool = True):
        self.packet_loss = packet_loss
        self.latency_spike_ms = latency_spike_ms / 1000.0  # Convert to seconds
        self.api_timeout_s = api_timeout_s
        self.simulate_rate_limiting = simulate_rate_limiting

        # Circuit breaker and backoff
        self.circuit_breaker = CircuitBreaker(failure_threshold=3, reset_timeout=30)
        self.backoff = ExponentialBackoff(base_delay=1.0, max_delay=30.0)

        # Statistics
        self.request_stats = {
            'total_requests': 0,
            'successful_requests': 0,
            'failed_requests': 0,
            'packet_loss_events': 0,
            'latency_spikes': 0,
            'rate_limit_hits': 0,
            'timeouts': 0,
            'circuit_breaker_trips': 0
        }

        # Test results
        self.results = {
            'configuration': {
                'packet_loss': packet_loss,
                'latency_spike_ms': latency_spike_ms,
                'api_timeout_s': api_timeout_s,
                'simulate_rate_limiting': simulate_rate_limiting
            },
            'statistics': self.request_stats,
            'circuit_breaker_states': [],
            'errors': [],
            'success': False,
            'start_time': None,
            'end_time': None
        }

        logger.info("Optimized Network Failure Test initialized")
        logger.info(f"Packet Loss: {packet_loss*100:.1f}%, Latency Spike: {latency_spike_ms}ms, Timeout: {api_timeout_s}s")

    async def simulate_api_call(self, session: aiohttp.ClientSession, endpoint: str,
                               request_data: Dict[str, Any], attempt: int = 0) -> Dict[str, Any]:
        """Simulate API call với network failure injection"""
        self.request_stats['total_requests'] += 1

        # Check circuit breaker
        if not self.circuit_breaker.can_execute():
            raise Exception("Circuit breaker is OPEN")

        start_time = time.time()

        try:
            # Inject packet loss
            if random.random() < self.packet_loss:
                self.request_stats['packet_loss_events'] += 1
                await asyncio.sleep(random.uniform(0.1, 1.0))  # Simulate connection delay
                raise aiohttp.ClientError("Simulated packet loss")

            # Inject latency spike (5% chance)
            if random.random() < 0.05:
                self.request_stats['latency_spikes'] += 1
                await asyncio.sleep(self.latency_spike_ms)

            # Simulate rate limiting (2% chance)
            if self.simulate_rate_limiting and random.random() < 0.02:
                self.request_stats['rate_limit_hits'] += 1
                await asyncio.sleep(random.uniform(1.0, 5.0))  # Simulate rate limit delay

                # Return rate limit response
                response_time = time.time() - start_time
                return {
                    'status': 'rate_limited',
                    'retry_after': random.randint(10, 60),
                    'response_time': response_time,
                    'attempt': attempt
                }

            # Simulate successful API call
            await asyncio.sleep(random.uniform(0.05, 0.2))  # Simulate network latency

            # Generate realistic trading response
            response = self.generate_trading_response(request_data)

            response_time = time.time() - start_time

            response.update({
                'status': 'success',
                'response_time': response_time,
                'attempt': attempt
            })

            self.circuit_breaker.on_success()
            self.backoff.reset()
            self.request_stats['successful_requests'] += 1

            return response

        except asyncio.TimeoutError:
            self.request_stats['timeouts'] += 1
            raise
        except Exception as e:
            self.request_stats['failed_requests'] += 1
            self.circuit_breaker.on_failure()

            # Check if we should retry with backoff
            if attempt < 3:  # Max 3 attempts
                delay = self.backoff.get_delay()
                logger.warning(f"Request failed (attempt {attempt+1}), retrying in {delay:.1f}s: {e}")
                self.backoff.increment_attempt()
                await asyncio.sleep(delay)
                return await self.simulate_api_call(session, endpoint, request_data, attempt + 1)

            raise

    def generate_trading_response(self, request_data: Dict[str, Any]) -> Dict[str, Any]:
        """Generate realistic trading API response"""
        symbol = request_data.get('symbol', 'BTC-USD')
        side = request_data.get('side', 'buy')
        quantity = request_data.get('quantity', random.randint(1, 10))

        # Generate realistic order response
        order_id = f"order_{random.randint(100000, 999999)}"
        price = request_data.get('price', random.uniform(50000, 60000))

        return {
            'order_id': order_id,
            'symbol': symbol,
            'side': side,
            'quantity': quantity,
            'price': price,
            'status': 'filled',
            'timestamp': datetime.now().isoformat(),
            'fees': price * quantity * 0.001  # 0.1% fee
        }

    async def simulate_trading_request(self, session: aiohttp.ClientSession, request_id: int) -> Dict[str, Any]:
        """Simulate a complete trading request with retries"""
        # Generate trading request data
        trading_data = {
            'symbol': random.choice(['BTC-USD', 'ETH-USD', 'SOL-USD', 'ADA-USD']),
            'side': random.choice(['buy', 'sell']),
            'quantity': random.randint(1, 10),
            'price': random.uniform(50000, 60000),
            'type': 'limit',
            'request_id': request_id
        }

        try:
            result = await self.simulate_api_call(session, '/api/v1/orders', trading_data)
            return {
                'request_id': request_id,
                'success': True,
                'result': result,
                'total_attempts': result.get('attempt', 0) + 1
            }

        except Exception as e:
            return {
                'request_id': request_id,
                'success': False,
                'error': str(e),
                'circuit_breaker_state': self.circuit_breaker.state
            }

    async def run_failure_simulation(self, duration_minutes: int = 30,
                                   concurrent_requests: int = 10) -> Dict[str, Any]:
        """Run comprehensive network failure simulation"""
        self.results['start_time'] = datetime.now().isoformat()

        logger.info("🚀 STARTING NETWORK FAILURE SIMULATION")
        logger.info(f"Duration: {duration_minutes} minutes")
        logger.info(f"Concurrent Requests: {concurrent_requests}")
        logger.info(f"Packet Loss Rate: {self.packet_loss*100:.1f}%")
        logger.info(f"Latency Spike: {self.latency_spike_ms*1000:.0f}ms")

        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=self.api_timeout_s)) as session:
            tasks = []
            request_counter = 0
            start_time = time.time()
            end_time = start_time + (duration_minutes * 60)
            last_status_update = 0

            try:
                while time.time() < end_time:
                    current_time = time.time()

                    # Status update every 30 seconds
                    if current_time - last_status_update >= 30:
                        elapsed_minutes = (current_time - start_time) / 60
                        progress = (elapsed_minutes / duration_minutes) * 100

                        circuit_status = self.circuit_breaker.get_status()
                        self.results['circuit_breaker_states'].append({
                            'timestamp': datetime.now().isoformat(),
                            'elapsed_minutes': elapsed_minutes,
                            **circuit_status
                        })

                        logger.info(f"Progress: {progress:.1f}% | Executions: {successful}/{total} | Avg Time: {execution_time:.3f}s | Latency Violations: {len(self.latency_violations)}")
                        last_status_update = current_time

                    # Maintain concurrent requests
                    active_tasks = [t for t in tasks if not t.done()]
                    if len(active_tasks) < concurrent_requests:
                        # Start new requests
                        for _ in range(min(concurrent_requests - len(active_tasks), 5)):
                            task = asyncio.create_task(
                                self.simulate_trading_request(session, request_counter)
                            )
                            tasks.append(task)
                            request_counter += 1

                    # Clean up completed tasks periodically
                    if len(tasks) > 100:
                        tasks = [t for t in tasks if not t.done()]

                    # Small delay to prevent excessive CPU usage
                    await asyncio.sleep(0.1)

                # Wait for remaining tasks to complete
                logger.info("Waiting for remaining requests to complete...")
                results = await asyncio.gather(*tasks, return_exceptions=True)

                # Process results
                success_count = sum(1 for r in results if isinstance(r, dict) and r.get('success', False))
                failure_count = len(results) - success_count

                self.results['statistics'].update({
                    'final_success_count': success_count,
                    'final_failure_count': failure_count,
                    'success_rate': success_count / len(results) if results else 0
                })

                # Test success criteria
                success_rate = success_count / len(results) if results else 0
                circuit_breaker_trips = sum(1 for state in self.results['circuit_breaker_states']
                                           if state.get('state') == 'OPEN')

                self.results['success'] = (
                    success_rate >= 0.85 and  # 85% success rate
                    circuit_breaker_trips <= 5  # Max 5 circuit breaker trips
                )

                logger.info("✅ NETWORK FAILURE SIMULATION COMPLETED")

            except Exception as e:
                error_msg = f"Simulation failed: {str(e)}"
                logger.error(error_msg, exc_info=True)
                self.results['errors'].append({
                    'timestamp': datetime.now().isoformat(),
                    'error': error_msg
                })

        self.results['end_time'] = datetime.now().isoformat()
        return self.results

    def print_summary(self):
        """Print test summary"""
        print("\n" + "=" * 70)
        print("🌐 NETWORK FAILURE SIMULATION RESULTS")
        print("=" * 70)

        stats = self.results['statistics']

        if self.results['success']:
            print("✅ TEST PASSED - System is resilient to network failures")

            print("📊 Request Statistics:"            print(f"   Total Requests: {stats.get('total_requests', 0)}")
            print(f"   Successful: {stats.get('successful_requests', 0)}")
            print(f"   Failed: {stats.get('failed_requests', 0)}")
            print(".1f"
            print("🌐 Network Failure Events:"            print(f"   Packet Loss: {stats.get('packet_loss_events', 0)}")
            print(f"   Latency Spikes: {stats.get('latency_spikes', 0)}")
            print(f"   Rate Limit Hits: {stats.get('rate_limit_hits', 0)}")
            print(f"   Timeouts: {stats.get('timeouts', 0)}")

            print("🔄 Circuit Breaker:"            states = self.results.get('circuit_breaker_states', [])
            if states:
                trips = sum(1 for s in states if s.get('state') == 'OPEN')
                print(f"   Circuit Breaker Trips: {trips}")

        else:
            print("❌ TEST FAILED - Network resilience issues detected")

            success_rate = stats.get('final_success_count', 0) / max(stats.get('total_requests', 1), 1)
            if success_rate < 0.85:
                print(".1f"
            circuit_states = self.results.get('circuit_breaker_states', [])
            trips = sum(1 for s in circuit_states if s.get('state') == 'OPEN')
            if trips > 5:
                print(f"🔴 Too many circuit breaker trips: {trips} (max allowed: 5)")

            # Show errors
            for error in self.results.get('errors', [])[-3:]:
                print(f"🔴 {error['error']}")

        print("=" * 70)

    def save_results(self, output_file: Optional[str] = None) -> str:
        """Save test results to file"""
        if not output_file:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file = f"network_failure_simulation_results_{timestamp}.json"

        with open(output_file, 'w') as f:
            json.dump(self.results, f, indent=2, default=str)

        logger.info(f"Results saved to {output_file}")
        return output_file

def main():
    parser = argparse.ArgumentParser(description='Network Failure Simulation for Supreme System V5')
    parser.add_argument('--duration', type=int, default=30,
                       help='Test duration in minutes (default: 30)')
    parser.add_argument('--packet-loss', type=float, default=0.1,
                       help='Packet loss rate (0.0-1.0, default: 0.1)')
    parser.add_argument('--latency-spike', type=int, default=500,
                       help='Latency spike in milliseconds (default: 500)')
    parser.add_argument('--api-timeout', type=int, default=30,
                       help='API timeout in seconds (default: 30)')
    parser.add_argument('--concurrent-requests', type=int, default=10,
                       help='Number of concurrent requests (default: 10)')
    parser.add_argument('--no-rate-limiting', action='store_true',
                       help='Disable rate limiting simulation')
    parser.add_argument('--output', type=str,
                       help='Output file for results (default: auto-generated)')

    args = parser.parse_args()

    # Validate parameters
    if not 0.0 <= args.packet_loss <= 1.0:
        logger.error("Packet loss must be between 0.0 and 1.0")
        sys.exit(1)

    print("🌐 SUPREME SYSTEM V5 - NETWORK FAILURE SIMULATION")
    print("=" * 55)
    print(f"Duration: {args.duration} minutes")
    print(f"Packet Loss: {args.packet_loss*100:.1f}%")
    print(f"Latency Spike: {args.latency_spike}ms")
    print(f"API Timeout: {args.api_timeout}s")
    print(f"Concurrent Requests: {args.concurrent_requests}")

    # Run the simulation
    test = OptimizedNetworkFailureTest(
        packet_loss=args.packet_loss,
        latency_spike_ms=args.latency_spike,
        api_timeout_s=args.api_timeout,
        simulate_rate_limiting=not args.no_rate_limiting
    )

    try:
        results = asyncio.run(test.run_failure_simulation(
            duration_minutes=args.duration,
            concurrent_requests=args.concurrent_requests
        ))

        # Save results
        output_file = test.save_results(args.output)

        # Print summary
        test.print_summary()

        # Exit with appropriate code
        sys.exit(0 if results['success'] else 1)

    except KeyboardInterrupt:
        logger.info("Simulation interrupted by user")
        test.save_results(args.output)
        sys.exit(1)
    except Exception as e:
        logger.error(f"Critical simulation failure: {e}", exc_info=True)
        test.save_results(args.output)
        sys.exit(1)

if __name__ == "__main__":
    main()
