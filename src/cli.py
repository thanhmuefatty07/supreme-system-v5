#!/usr/bin/env python3
"""
Supreme System V5 - Command Line Interface

Real implementation of the trading system CLI.
"""

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd

from .data.binance_client import BinanceClient
from .risk.risk_manager import RiskManager
from .strategies.moving_average import MovingAverageStrategy


def setup_logging(verbose: bool = False):
    """Setup logging configuration"""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )


def cmd_data(args):
    """Handle data-related commands"""
    print("Supreme System V5 - Data Management")

    if args.action == "download":
        print(f"Downloading {args.symbol} data from {args.start_date} to {args.end_date}")

        # Initialize client
        client = BinanceClient(testnet=True)

        try:
            # Download data
            data = client.get_historical_klines(
                symbol=args.symbol,
                interval=args.interval,
                start_date=args.start_date,
                end_date=args.end_date
            )

            if data is not None and not data.empty:
                print(f"✅ Downloaded {len(data)} records")
                print(f"   Date range: {data['timestamp'].min()} to {data['timestamp'].max()}")
                print(f"   Price range: ${data['close'].min():.2f} - ${data['close'].max():.2f}")

                # Save to CSV
                filename = f"{args.symbol}_{args.interval}_{args.start_date.replace('-', '')}_{args.end_date.replace('-', '')}.csv"
                data.to_csv(filename, index=False)
                print(f"   Saved to: {filename}")
            else:
                print("❌ No data downloaded")

        except Exception as e:
            print(f"❌ Error: {e}")
            return 1

    elif args.action == "test":
        print("Testing Binance API connection...")
        client = BinanceClient(testnet=True)

        if client.test_connection():
            print("✅ API connection successful")
        else:
            print("❌ API connection failed")
            return 1

    return 0


def cmd_backtest(args):
    """Handle backtesting commands"""
    print("Supreme System V5 - Backtesting")

    try:
        # Load data
        if not Path(args.data_file).exists():
            print(f"❌ Data file not found: {args.data_file}")
            return 1

        print(f"Loading data from {args.data_file}...")
        data = pd.read_csv(args.data_file)
        print(f"✅ Loaded {len(data)} records")

        # Initialize strategy
        strategy = MovingAverageStrategy(
            short_window=args.short_window,
            long_window=args.long_window
        )

        # Initialize risk manager
        risk_manager = RiskManager(
            initial_capital=args.capital,
            max_position_size=args.max_position,
            stop_loss=args.stop_loss
        )

        # Run backtest
        print("Running backtest...")
        results = risk_manager.run_backtest(data, strategy)

        # Display results
        print("\n📊 BACKTEST RESULTS")
        print("=" * 50)
        print(".2f")
        print(".2f")
        print(".2%")
        print(".2f")
        print(".2%")
        print(f"Total Trades: {results['total_trades']}")
        print(".2f")
        print(".4f")

        return 0

    except Exception as e:
        print(f"❌ Backtest error: {e}")
        return 1


def main():
    """Main CLI entry point"""
    parser = argparse.ArgumentParser(
        description="Supreme System V5 - Real Algorithmic Trading System"
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable verbose logging"
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Data command
    data_parser = subparsers.add_parser("data", help="Data management commands")
    data_parser.add_argument(
        "action",
        choices=["download", "test"],
        help="Action to perform"
    )
    data_parser.add_argument(
        "--symbol",
        default="ETHUSDT",
        help="Trading symbol (default: ETHUSDT)"
    )
    data_parser.add_argument(
        "--interval",
        default="1h",
        choices=["1m", "3m", "5m", "15m", "30m", "1h", "2h", "4h", "6h", "8h", "12h", "1d"],
        help="Time interval (default: 1h)"
    )
    data_parser.add_argument(
        "--start-date",
        help="Start date (YYYY-MM-DD)"
    )
    data_parser.add_argument(
        "--end-date",
        help="End date (YYYY-MM-DD)"
    )

    # Backtest command
    backtest_parser = subparsers.add_parser("backtest", help="Run backtesting")
    backtest_parser.add_argument(
        "data_file",
        help="Path to CSV data file"
    )
    backtest_parser.add_argument(
        "--capital",
        type=float,
        default=10000,
        help="Initial capital (default: 10000)"
    )
    backtest_parser.add_argument(
        "--short-window",
        type=int,
        default=10,
        help="Short MA window (default: 10)"
    )
    backtest_parser.add_argument(
        "--long-window",
        type=int,
        default=30,
        help="Long MA window (default: 30)"
    )
    backtest_parser.add_argument(
        "--max-position",
        type=float,
        default=0.1,
        help="Max position size (default: 0.1)"
    )
    backtest_parser.add_argument(
        "--stop-loss",
        type=float,
        default=0.02,
        help="Stop loss percentage (default: 0.02)"
    )

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return 1

    # Setup logging
    setup_logging(args.verbose)

    # Execute command
    if args.command == "data":
        return cmd_data(args)
    elif args.command == "backtest":
        import pandas as pd
        return cmd_backtest(args)

    return 0


if __name__ == "__main__":
    sys.exit(main())
