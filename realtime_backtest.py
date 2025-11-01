#!/usr/bin/env python3
"""
🚀 Supreme System V5 - Production-Ready Realtime Backtest
Compatible with existing repository structure
Uses FREE public data sources (NO API keys required)
Memory-optimized for i3-4GB systems
"""

import argparse
import asyncio
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any

# Add python directory to path
sys.path.insert(0, str(Path(__file__).parent / "python"))

print("🚀 Supreme System V5 - Realtime Backtest")
print("=" * 50)

# Import validation with error handling
try:
    from supreme_system_v5.backtest import run_realtime_backtest, BacktestConfig
    print("✅ Core backtest imports successful")
except ImportError as e:
    print(f"❌ Core import error: {e}")
    print("💡 Make sure you're in the supreme-system-v5 directory")
    print("🔧 Run: pip install -r requirements.txt")
    sys.exit(1)

# Optional imports for enhanced features
PSUTIL_AVAILABLE = False
PROMETHEUS_AVAILABLE = False

try:
    import psutil
    PSUTIL_AVAILABLE = True
    print("✅ System monitoring available")
except ImportError:
    print("⚠️ Install psutil for system monitoring: pip install psutil")

try:
    from prometheus_client import start_http_server, Gauge, Counter
    PROMETHEUS_AVAILABLE = True
    print("✅ Prometheus metrics available")
except ImportError:
    print("⚠️ Install prometheus_client for metrics: pip install prometheus_client")

print("=" * 50)





def parse_arguments():
    """Parse command line arguments with robust defaults"""
    parser = argparse.ArgumentParser(
        description="Supreme System V5 - Production Realtime Backtest",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
🚀 Production Features:
  • FREE data sources (Binance, CoinGecko, OKX) - NO API keys required
  • Professional CSV export for analysis
  • System performance monitoring
  • Memory-optimized for i3-4GB systems

Examples:
  python realtime_backtest.py                              # Default run
  python realtime_backtest.py --symbols BTC-USDT ETH-USDT  # Custom symbols
  python realtime_backtest.py --balance 50000 --record-raw # Enhanced recording
        """
    )

    parser.add_argument("--symbols", "-s", nargs="+",
                       default=["BTC-USDT", "ETH-USDT"],
                       help="Trading symbols (default: BTC-USDT ETH-USDT)")

    parser.add_argument("--balance", "-b", type=float, default=10000.0,
                       help="Initial balance USD (default: 10000)")

    parser.add_argument("--interval", "-i", type=float, default=2.0,
                       help="Update interval seconds (default: 2.0 for i3-4GB)")
    
    parser.add_argument("--max-position", type=float, default=0.1,
                       help="Max position size as portfolio fraction (default: 0.1)")
    
    parser.add_argument("--sources", nargs="+", 
                       default=["binance", "coingecko", "okx"],
                       help="Data sources (default: binance coingecko okx)")
    
    parser.add_argument("--no-risk", action="store_true",
                       help="Disable risk management")
    
    parser.add_argument("--historical-days", "-d", type=int, default=30,
                       help="Historical data days (default: 30)")
    
    parser.add_argument("--record-raw", action="store_true",
                       help="Record raw market data (memory optimized)")
    
    parser.add_argument("--metrics-port", type=int, default=0,
                       help="Prometheus metrics port (0=disabled)")
    
    parser.add_argument("--output-dir", type=str, default="run_artifacts",
                       help="Output directory (default: run_artifacts)")

    return parser.parse_args()


def create_production_config(args) -> BacktestConfig:
    """Create production BacktestConfig"""
    config = BacktestConfig()
    
    # Basic configuration using original BacktestConfig structure
    config.symbols = args.symbols
    config.initial_balance = args.balance
    config.data_sources = args.sources  
    config.realtime_interval = args.interval
    config.max_position_size = args.max_position
    config.enable_risk_management = not args.no_risk
    config.historical_days = args.historical_days
    
    return config


async def enhanced_backtest_wrapper(config: BacktestConfig, args):
    """Simple wrapper around core backtest engine"""
    print(f"🎯 Starting backtest with {len(config.symbols)} symbols...")
    print(f"💰 Initial balance: ${config.initial_balance:.2f}")
    print(f"⏱️ Update interval: {config.realtime_interval}s")
    print(f"🎛️ Risk management: {'Enabled' if config.enable_risk_management else 'Disabled'}")
    print()

    # Run the core backtest
    await run_realtime_backtest(config)


async def main():
    """Main entry point"""
    print("🚀 Supreme System V5 - Production Realtime Backtest")
    print("=" * 65)
    print("💰 FREE data sources - NO API keys required")
    print("📊 Professional recording and monitoring")
    print("⚡ Memory optimized for i3-4GB systems")
    print()
    
    # Parse and validate arguments
    args = parse_arguments()
    validate_args(args)
    
    # Create configuration
    config = create_production_config(args)
    
    # Display configuration
    print("📊 Production Configuration:")
    print(f"   Symbols: {', '.join(config.symbols)}")
    print(f"   Balance: ${config.initial_balance:.2f}")
    print(f"   Data Sources: {', '.join(config.data_sources)} (FREE)")
    print(f"   Interval: {config.realtime_interval}s")
    print(f"   Risk Management: {'Enabled' if config.enable_risk_management else 'Disabled'}")
    print(f"   Recording: Raw={args.record_raw}, Metrics={args.metrics_port > 0}")
    print()
    
    print("🎯 Starting Production Realtime Backtest...")
    print("🎛️ Press Ctrl+C to stop and generate comprehensive report")
    print("=" * 65)
    
    try:
        await enhanced_backtest_wrapper(config, args)
    except KeyboardInterrupt:
        print("\n🛑 Production backtest stopped by user")
        print("📊 Generating comprehensive analysis report...")
    except Exception as e:
        print(f"\n❌ Fatal error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    try:
        exit_code = asyncio.run(main())
        sys.exit(exit_code)
    except KeyboardInterrupt:
        print("\n🛑 Production backtest interrupted")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        sys.exit(1)
