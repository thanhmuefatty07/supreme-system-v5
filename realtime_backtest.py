#!/usr/bin/env python3
"""
Real-Time Backtesting CLI - Supreme System V5 ULTRA SFL
Enterprise-grade real-time backtesting with manual stop capability
"""

import argparse
import asyncio
import sys
from pathlib import Path

# Add python directory to path
sys.path.insert(0, str(Path(__file__).parent / "python"))

from supreme_system_v5.backtest import run_realtime_backtest, BacktestConfig


def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="Supreme System V5 - Real-Time Backtesting Engine",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python realtime_backtest.py                              # Default configuration
  python realtime_backtest.py --symbols BTC-USDT ETH-USDT  # Custom symbols
  python realtime_backtest.py --balance 50000             # Custom balance
  python realtime_backtest.py --no-risk                    # Disable risk management
        """
    )

    parser.add_argument(
        "--symbols", "-s",
        nargs="+",
        default=["BTC-USDT", "ETH-USDT"],
        help="Trading symbols to backtest (default: BTC-USDT ETH-USDT)"
    )

    parser.add_argument(
        "--balance", "-b",
        type=float,
        default=10000.0,
        help="Initial portfolio balance in USD (default: 10000.0)"
    )

    parser.add_argument(
        "--sources",
        nargs="+",
        default=["binance", "coingecko"],
        help="Data sources to use (default: binance coingecko)"
    )

    parser.add_argument(
        "--interval", "-i",
        type=float,
        default=1.0,
        help="Update interval in seconds (default: 1.0)"
    )

    parser.add_argument(
        "--max-position",
        type=float,
        default=0.1,
        help="Maximum position size as fraction of portfolio (default: 0.1)"
    )

    parser.add_argument(
        "--no-risk",
        action="store_true",
        help="Disable risk management"
    )

    parser.add_argument(
        "--historical-days", "-d",
        type=int,
        default=30,
        help="Days of historical data to load (default: 30)"
    )

    parser.add_argument(
        "--strategy-config",
        type=str,
        help="Path to strategy configuration JSON file"
    )

    return parser.parse_args()


def create_config_from_args(args):
    """Create BacktestConfig from command line arguments"""
    config = BacktestConfig()

    # Basic settings
    config.symbols = args.symbols
    config.initial_balance = args.balance
    config.data_sources = args.sources
    config.realtime_interval = args.interval
    config.max_position_size = args.max_position
    config.enable_risk_management = not args.no_risk
    config.historical_days = args.historical_days

    # Strategy configuration (if provided)
    if args.strategy_config:
        try:
            import json
            with open(args.strategy_config, 'r') as f:
                config.strategy_config = json.load(f)
        except Exception as e:
            print(f"⚠️ Failed to load strategy config: {e}")
            print("Using default strategy configuration...")

    return config


async def main():
    """Main entry point"""
    print("🚀 Supreme System V5 - Real-Time Backtesting Engine")
    print("=" * 60)

    # Parse arguments
    args = parse_arguments()

    print("📊 Configuration:")
    print(f"  Symbols: {', '.join(args.symbols)}")
    print(f"  Initial Balance: ${args.balance:.2f}")
    print(f"  Data Sources: {', '.join(args.sources)}")
    print(f"  Risk Management: {'Disabled' if args.no_risk else 'Enabled'}")
    print(f"  Historical Days: {args.historical_days}")
    print(f"  Max Position Size: {args.max_position:.1%}")
    print()

    # Create configuration
    config = create_config_from_args(args)

    print("🎯 Starting Real-Time Backtesting...")
    print("🎛️ Press Ctrl+C at any time to stop backtesting and view results")
    print("=" * 60)

    try:
        # Run backtesting
        await run_realtime_backtest(config)

    except KeyboardInterrupt:
        print("\n🛑 Backtesting stopped by user")
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
        print("\n🛑 Backtesting interrupted")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        sys.exit(1)
