#!/usr/bin/env python3
"""
Supreme System V5 - Clean Production Entry Point
Hybrid Python + Rust Trading System
"""

import asyncio
import logging
import sys
from pathlib import Path
from typing import Optional

# Add python source to path
sys.path.insert(0, str(Path(__file__).parent / "python"))

# Configure clean logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(), logging.FileHandler("logs/system.log")],
)

logger = logging.getLogger(__name__)

# Create logs directory
Path("logs").mkdir(exist_ok=True)

try:
    from supreme_system_v5 import (
        TradingSystem,
        get_system_info,
        benchmark_system,
        RUST_ENGINE_AVAILABLE,
    )

    SYSTEM_AVAILABLE = True
except ImportError as e:
    logger.error(f"System import failed: {e}")
    SYSTEM_AVAILABLE = False


class CleanSupremeSystem:
    """Clean Supreme System V5 without error-prone features"""

    def __init__(self):
        self.version = "5.0.0-clean"
        self.logger = logger
        self.system: Optional[TradingSystem] = None

    async def initialize(self):
        """Initialize system components"""
        self.logger.info("Initializing Supreme System V5...")

        if not SYSTEM_AVAILABLE:
            self.logger.error("Core system not available")
            return False

        try:
            # Get system info
            info = get_system_info()
            self.logger.info(f"Version: {info['version']}")
            self.logger.info(f"Rust engine: {info['rust_engine_available']}")
            self.logger.info(f"Hardware: {info['hardware_profile']}")

            # Initialize trading system
            from supreme_system_v5 import get_system

            self.system = get_system()
            await self.system.initialize()

            self.logger.info("System initialized successfully")
            return True

        except Exception as e:
            self.logger.error(f"Initialization failed: {e}")
            return False

    async def run_demonstration(self):
        """Run clean system demonstration"""
        self.logger.info("Running system demonstration...")

        try:
            if RUST_ENGINE_AVAILABLE:
                # Test Rust engine
                import supreme_engine_rs as rust_engine
                import numpy as np

                # Test data
                test_data = np.random.randn(1000) + 100.0

                # Test indicators
                start_time = asyncio.get_event_loop().time()
                sma_result = rust_engine.fast_sma(test_data, 20)
                rust_time = (asyncio.get_event_loop().time() - start_time) * 1000

                self.logger.info(f"Rust SMA calculation: {rust_time:.2f}ms")
                self.logger.info(f"SMA result length: {len(sma_result)}")

            # System benchmark
            if SYSTEM_AVAILABLE:
                benchmark_result = benchmark_system(5)
                self.logger.info(
                    f"Benchmark completed: {benchmark_result['benchmark_duration_s']:.2f}s"
                )

            self.logger.info("Demonstration completed successfully")
            return True

        except Exception as e:
            self.logger.error(f"Demonstration failed: {e}")
            return False

    async def start(self):
        """Start clean system"""
        self.logger.info("Starting Supreme System V5...")

        success = await self.initialize()
        if not success:
            return False

        success = await self.run_demonstration()
        if not success:
            return False

        if self.system:
            await self.system.start()
            self.logger.info("Trading system started")

        return True

    async def stop(self):
        """Stop system gracefully"""
        self.logger.info("Stopping system...")

        if self.system:
            await self.system.stop()

        self.logger.info("System stopped")


async def main():
    """Clean main entry point"""
    print("Supreme System V5 - Clean Production System")
    print("=" * 50)

    system = CleanSupremeSystem()

    try:
        success = await system.start()
        if success:
            print("System running successfully!")
            # Keep running until interrupted
            try:
                while True:
                    await asyncio.sleep(1)
            except KeyboardInterrupt:
                print("\nShutdown requested...")
        else:
            print("System startup failed")

    except Exception as e:
        logger.error(f"System error: {e}")
    finally:
        await system.stop()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nSystem interrupted")
    except Exception as e:
        print(f"\nSystem error: {e}")
        sys.exit(1)
