#!/usr/bin/env python3
"""
Supreme System V5 - Real-Time Backtesting Demo
Demonstrates manual stop capability and detailed monitoring
"""

import asyncio
import sys
from pathlib import Path

# Add python directory to path
sys.path.insert(0, str(Path(__file__).parent / "python"))

from supreme_system_v5.backtest import BacktestConfig, run_realtime_backtest


async def demo_backtesting():
    """Run a demo backtesting session"""
    print("🚀 Supreme System V5 - Real-Time Backtesting Demo")
    print("=" * 60)
    print("This demo will:")
    print("1. Initialize real-time data feeds")
    print("2. Run scalping strategy with risk management")
    print("3. Provide detailed progress monitoring")
    print("4. Generate comprehensive results when stopped")
    print()
    print("🎛️ Press Ctrl+C at any time to stop and view results")
    print("=" * 60)

    # Create custom configuration for demo
    config = BacktestConfig(
        symbols=["BTC-USDT", "ETH-USDT"],
        data_sources=["binance", "coingecko"],
        initial_balance=50000.0,
        max_position_size=0.05,  # 5% per position
        realtime_interval=2.0,   # 2 second updates
        enable_risk_management=True,
        historical_days=7,       # Quick demo
        enable_detailed_logging=True
    )

    try:
        await run_realtime_backtest(config)
    except KeyboardInterrupt:
        print("\n🛑 Demo interrupted by user - results displayed above")
    except Exception as e:
        print(f"\n❌ Demo failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    print("🎯 Starting Supreme System V5 Real-Time Backtesting Demo...")
    print("💡 This will run until you manually stop it with Ctrl+C")
    print()

    try:
        asyncio.run(demo_backtesting())
    except KeyboardInterrupt:
        print("\n👋 Demo completed! Check the results above.")
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        sys.exit(1)
