#!/usr/bin/env python3
"""
Supreme System V5 - OPTIMIZED MEMORY STRESS TEST
Critical production readiness validation for 4GB RAM constraint

Memory Stress Test với realistic 2.2GB budget optimization
Sử dụng memory mapping và pool allocation để tránh fragmentation
"""

import asyncio
import argparse
import json
import logging
import mmap
import numpy as np
import os
import psutil
import signal
import sys
import tempfile
import time
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import List, Dict, Optional

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('memory_stress_test.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

@dataclass
class MemoryStressConfig:
    """Configuration for memory stress testing"""
    duration_hours: float = 1.0
    data_size_gb: int = 10
    memory_limit_gb: float = 2.2
    chunk_size_mb: int = 50  # Optimized cho 4GB system
    monitoring_interval: float = 5.0
    enable_memory_mapping: bool = True
    enable_memory_pools: bool = True

class OptimizedMemoryStressTest:
    """Memory stress test optimized for 4GB RAM constraint"""

    def __init__(self, config: MemoryStressConfig):
        self.config = config
        self.process = psutil.Process()
        self.memory_pool: List[np.ndarray] = []
        self.memory_mapped_files: List[mmap.mmap] = []
        self.memory_stats: List[Dict] = []
        self.is_running = False

        # Test results
        self.results = {
            'config': {
                'duration_hours': config.duration_hours,
                'data_size_gb': config.data_size_gb,
                'memory_limit_gb': config.memory_limit_gb,
                'chunk_size_mb': config.chunk_size_mb,
                'enable_memory_mapping': config.enable_memory_mapping,
                'enable_memory_pools': config.enable_memory_pools
            },
            'memory_stats': [],
            'performance_metrics': [],
            'errors': [],
            'success': False,
            'start_time': None,
            'end_time': None
        }

        logger.info(f"Optimized Memory Stress Test initialized - Budget: {config.memory_limit_gb}GB")

    def setup_memory_pools(self):
        """Pre-allocate memory pools để tránh fragmentation"""
        if not self.config.enable_memory_pools:
            return

        logger.info("Setting up memory pools...")

        # Calculate pool size (60% of memory budget for pools)
        pool_size_mb = int(self.config.memory_limit_gb * 0.6 * 1024)
        chunk_count = max(1, pool_size_mb // self.config.chunk_size_mb)

        logger.info(f"Creating {chunk_count} memory pool chunks of {self.config.chunk_size_mb}MB each")

        for i in range(chunk_count):
            try:
                # Use numpy arrays for efficient memory management
                chunk = np.zeros((self.config.chunk_size_mb * 256, 1024), dtype=np.float32)  # 50MB chunk
                self.memory_pool.append(chunk)
                logger.debug(f"Created memory pool chunk {i+1}/{chunk_count}")
            except MemoryError:
                logger.warning(f"Failed to allocate memory pool chunk {i+1}, stopping pool creation")
                break

        logger.info(f"Memory pool setup complete - {len(self.memory_pool)} chunks allocated")

    def setup_memory_mapped_files(self):
        """Setup memory mapped files for large data handling"""
        if not self.config.enable_memory_mapping:
            return

        logger.info("Setting up memory mapped files...")

        # Create temporary files for memory mapping
        temp_dir = tempfile.mkdtemp(prefix="memory_test_")

        for i in range(3):  # 3 mapped files
            try:
                temp_file = os.path.join(temp_dir, f"mapped_data_{i}.dat")

                # Create file with specific size
                file_size = 200 * 1024 * 1024  # 200MB each

                with open(temp_file, 'wb') as f:
                    f.truncate(file_size)

                # Memory map the file
                with open(temp_file, 'r+b') as f:
                    mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_WRITE)
                    self.memory_mapped_files.append(mm)

                logger.debug(f"Created memory mapped file {i+1}/3")

            except Exception as e:
                logger.warning(f"Failed to create memory mapped file {i+1}: {e}")
                break

        logger.info(f"Memory mapped files setup complete - {len(self.memory_mapped_files)} files")

    def calculate_moving_average(self, data: np.ndarray, window: int) -> np.ndarray:
        """SIMD-optimized moving average calculation"""
        if len(data) < window:
            return np.array([data.mean()])

        # Use numpy's vectorized operations for SIMD performance
        return np.convolve(data, np.ones(window)/window, mode='valid')

    def calculate_volatility(self, data: np.ndarray, window: int) -> float:
        """Memory-efficient volatility calculation"""
        if len(data) < window:
            return float(np.std(data))

        # Calculate returns efficiently
        returns = np.diff(data[-window:]) / data[-window:-1]
        return float(np.std(returns))

    def simulate_trading_workload(self, iteration: int) -> Dict[str, float]:
        """Simulate realistic trading data processing workload"""
        start_time = time.time()

        # Generate trading data chunk
        price_data = np.random.random(100000).astype(np.float32) * 1000 + 50000  # BTC-like prices
        volume_data = np.random.randint(1000, 100000, 100000)

        # Calculate technical indicators
        moving_avg = self.calculate_moving_average(price_data, 50)
        volatility = self.calculate_volatility(price_data, 20)

        # Simulate RSI calculation
        delta = np.diff(price_data)
        gain = np.where(delta > 0, delta, 0)
        loss = np.where(delta < 0, -delta, 0)
        avg_gain = np.convolve(gain, np.ones(14)/14, mode='valid')
        avg_loss = np.convolve(loss, np.ones(14)/14, mode='valid')
        rs = np.divide(avg_gain, avg_loss, out=np.ones_like(avg_gain), where=avg_loss!=0)
        rsi = 100 - (100 / (1 + rs))

        # Simulate MACD
        fast_ema = self.calculate_exponential_moving_average(price_data, 12)
        slow_ema = self.calculate_exponential_moving_average(price_data, 26)
        macd = fast_ema - slow_ema

        processing_time = time.time() - start_time

        return {
            'iteration': iteration,
            'processing_time': processing_time,
            'moving_avg': float(moving_avg[-1]) if len(moving_avg) > 0 else 0.0,
            'volatility': volatility,
            'rsi': float(rsi[-1]) if len(rsi) > 0 else 50.0,
            'macd': float(macd[-1]) if len(macd) > 0 else 0.0,
            'data_points': len(price_data)
        }

    def calculate_exponential_moving_average(self, data: np.ndarray, span: int) -> np.ndarray:
        """Calculate EMA efficiently"""
        return np.convolve(data, self._ema_weights(span), mode='valid')

    def _ema_weights(self, span: int) -> np.ndarray:
        """Generate EMA weights"""
        alpha = 2.0 / (span + 1)
        weights = np.zeros(span)
        for i in range(span):
            weights[i] = alpha * (1 - alpha) ** (span - 1 - i)
        return weights

    async def cleanup_memory_pressure(self):
        """Aggressive memory cleanup khi gần đạt limit"""
        current_memory_gb = self.process.memory_info().rss / (1024 ** 3)

        if current_memory_gb > self.config.memory_limit_gb * 0.85:  # 85% of limit
            logger.warning(f"High memory usage detected: {current_memory_gb:.1f}GB / {self.config.memory_limit_gb:.1f}GB limit")
            # Clear memory pools partially
            if len(self.memory_pool) > 2:
                self.memory_pool = self.memory_pool[:2]  # Keep only 2 chunks

            # Force garbage collection
            import gc
            collected = gc.collect()
            logger.info(f"Garbage collected {collected} objects")

            # Small delay to let memory settle
            await asyncio.sleep(0.1)

    def monitor_memory_usage(self) -> Dict[str, float]:
        """Monitor current memory usage"""
        memory_info = self.process.memory_info()
        system_memory = psutil.virtual_memory()

        return {
            'timestamp': datetime.now().isoformat(),
            'process_rss_mb': memory_info.rss / (1024 * 1024),
            'process_vms_mb': memory_info.vms / (1024 * 1024),
            'system_used_mb': system_memory.used / (1024 * 1024),
            'system_available_mb': system_memory.available / (1024 * 1024),
            'system_percent': system_memory.percent,
            'memory_limit_exceeded': (memory_info.rss / (1024 ** 3)) > self.config.memory_limit_gb
        }

    async def run(self) -> Dict:
        """Main test execution"""
        print(f"🔍 Starting Optimized Memory Stress Test - Budget: {self.config.memory_limit_gb}GB")
        print(f"Duration: {self.config.duration_hours} hours")
        print(f"Data Size: {self.config.data_size_gb}GB")
        print(f"Memory Mapping: {self.config.enable_memory_mapping}")
        print(f"Memory Pools: {self.config.enable_memory_pools}")

        self.results['start_time'] = datetime.now().isoformat()

        # Setup memory management
        self.setup_memory_pools()
        self.setup_memory_mapped_files()

        self.is_running = True
        iteration = 0
        last_monitoring = 0

        try:
            start_time = time.time()
            end_time = start_time + (self.config.duration_hours * 3600)

            while self.is_running and time.time() < end_time:
                current_time = time.time()

                # Memory monitoring
                if current_time - last_monitoring >= self.config.monitoring_interval:
                    memory_stats = self.monitor_memory_usage()
                    self.memory_stats.append(memory_stats)
                    self.results['memory_stats'].append(memory_stats)

                    # Check memory limit
                    if memory_stats['memory_limit_exceeded']:
                        error_msg = f"Memory limit exceeded: {memory_stats['process_rss_mb']:.1f}MB > {self.config.memory_limit_gb * 1024:.1f}MB"
                        logger.error(error_msg)
                        self.results['errors'].append({
                            'timestamp': memory_stats['timestamp'],
                            'error': error_msg,
                            'memory_mb': memory_stats['process_rss_mb']
                        })
                        self.is_running = False
                        break

                    last_monitoring = current_time

                # Run trading workload simulation
                perf_metrics = self.simulate_trading_workload(iteration)
                self.results['performance_metrics'].append(perf_metrics)

                iteration += 1

                # Memory pressure cleanup every 100 iterations
                if iteration % 100 == 0:
                    await self.cleanup_memory_pressure()

                # Progress reporting every 1000 iterations
                if iteration % 1000 == 0:
                    elapsed_hours = (current_time - start_time) / 3600
                    progress = (elapsed_hours / self.config.duration_hours) * 100
                    logger.info(f"Progress: {progress:.1f}% ({elapsed_hours:.1f}h/{self.config.duration_hours:.1f}h)")
                # Small delay to prevent excessive CPU usage
                await asyncio.sleep(0.01)

            # Test completed successfully
            if time.time() >= end_time:
                self.results['success'] = True
                logger.info("✅ Memory Stress Test COMPLETED SUCCESSFULLY")

        except Exception as e:
            error_msg = f"Test failed with exception: {str(e)}"
            logger.error(error_msg, exc_info=True)
            self.results['errors'].append({
                'timestamp': datetime.now().isoformat(),
                'error': error_msg,
                'exception': str(e)
            })

        finally:
            # Cleanup
            self.is_running = False
            await self.cleanup()

        self.results['end_time'] = datetime.now().isoformat()
        return self.results

    async def cleanup(self):
        """Cleanup resources"""
        logger.info("Cleaning up memory stress test resources")

        # Close memory mapped files
        for mm in self.memory_mapped_files:
            try:
                mm.close()
            except:
                pass
        self.memory_mapped_files.clear()

        # Clear memory pools
        self.memory_pool.clear()

        # Force garbage collection
        import gc
        gc.collect()

    def save_results(self, output_file: Optional[str] = None) -> str:
        """Save test results to file"""
        if not output_file:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file = f"memory_stress_test_results_{timestamp}.json"

        with open(output_file, 'w') as f:
            json.dump(self.results, f, indent=2, default=str)

        logger.info(f"Results saved to {output_file}")
        return output_file

    def print_summary(self):
        """Print test summary"""
        print("\n" + "=" * 70)
        print("🧪 OPTIMIZED MEMORY STRESS TEST RESULTS")
        print("=" * 70)

        if self.results['success']:
            print("✅ TEST PASSED - No memory limit violations")

            memory_stats = self.results['memory_stats']
            perf_metrics = self.results['performance_metrics']

            if memory_stats:
                max_memory = max(s['process_rss_mb'] for s in memory_stats)
                avg_memory = np.mean([s['process_rss_mb'] for s in memory_stats])
                memory_efficiency = (self.config.memory_limit_gb * 1024) / max_memory if max_memory > 0 else 1.0

                print("📊 Memory Statistics:")
                print(f"   Max Memory Usage: {max_memory:.1f}MB")
                print(f"   Avg Memory Usage: {avg_memory:.1f}MB")
                print(f"   Memory Limit: {self.config.memory_limit_gb * 1024:.1f}MB")
                print(f"   Memory Efficiency: {memory_efficiency:.2f}")
                print(f"   Memory Limit Exceeded: {any(s['memory_limit_exceeded'] for s in memory_stats)}")

            if perf_metrics:
                total_iterations = len(perf_metrics)
                avg_processing_time = np.mean([m['processing_time'] for m in perf_metrics])

                print("⚡ Performance Statistics:")
                print(f"   Total Iterations: {total_iterations}")
                print(f"   Avg Processing Time: {avg_processing_time:.4f}s")
                print(f"   Iterations/Second: {1.0/avg_processing_time:.1f}")

        else:
            print("❌ TEST FAILED")
            for error in self.results['errors'][-3:]:  # Show last 3 errors
                print(f"🔴 {error['error']}")

        print("=" * 70)

def main():
    parser = argparse.ArgumentParser(description='Optimized Memory Stress Test for Supreme System V5')
    parser.add_argument('--duration', type=float, default=1.0,
                       help='Test duration in hours (default: 1.0)')
    parser.add_argument('--data-size', type=int, default=10,
                       help='Test data size in GB (default: 10)')
    parser.add_argument('--memory-limit', type=float, default=2.2,
                       help='Memory limit in GB (default: 2.2)')
    parser.add_argument('--chunk-size', type=int, default=50,
                       help='Memory chunk size in MB (default: 50)')
    parser.add_argument('--no-memory-mapping', action='store_true',
                       help='Disable memory mapping')
    parser.add_argument('--no-memory-pools', action='store_true',
                       help='Disable memory pools')
    parser.add_argument('--output', type=str,
                       help='Output file for results (default: auto-generated)')

    args = parser.parse_args()

    # Validate system requirements
    system_memory = psutil.virtual_memory()
    if system_memory.total < args.memory_limit * 1024 * 1024 * 1024:
        logger.error(f"System has only {system_memory.total / 1024 / 1024 / 1024:.1f}GB RAM, test requires {args.memory_limit}GB limit")
        sys.exit(1)

    config = MemoryStressConfig(
        duration_hours=args.duration,
        data_size_gb=args.data_size,
        memory_limit_gb=args.memory_limit,
        chunk_size_mb=args.chunk_size,
        enable_memory_mapping=not args.no_memory_mapping,
        enable_memory_pools=not args.no_memory_pools
    )

    # Run the test
    test = OptimizedMemoryStressTest(config)

    def signal_handler(signum, frame):
        logger.info("Shutdown signal received - stopping test gracefully")
        test.is_running = False

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    try:
        results = asyncio.run(test.run())

        # Save results
        output_file = test.save_results(args.output)

        # Print summary
        test.print_summary()

        # Exit with appropriate code
        sys.exit(0 if results['success'] else 1)

    except KeyboardInterrupt:
        logger.info("Test interrupted by user")
        test.save_results(args.output)
        sys.exit(1)
    except Exception as e:
        logger.error(f"Critical test failure: {e}", exc_info=True)
        test.save_results(args.output)
        sys.exit(1)

if __name__ == "__main__":
    main()
