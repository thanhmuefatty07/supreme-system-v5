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

    parser.add_argument("--max-position", type=float, default=0.02,
                       help="Max position size fraction (default: 0.02)")

    parser.add_argument("--record-raw", action="store_true",
                       help="Record raw market data")

    parser.add_argument("--no-risk", action="store_true",
                       help="Disable risk management")

    parser.add_argument("--historical-days", "-d", type=int, default=7,
                       help="Historical data days (default: 7)")

    return parser.parse_args()


def create_config_from_args(args):
    """Create BacktestConfig from command line arguments"""
    config = BacktestConfig()

    # Basic settings
    config.symbols = args.symbols
    config.initial_balance = args.balance
    config.realtime_interval = args.interval
    config.max_position_size = args.max_position
    config.enable_risk_management = not args.no_risk
    config.historical_days = args.historical_days

    # Data sources (free ones only)
    config.data_sources = ["binance", "coingecko", "okx"]

    return config


async def main():
    """Main entry point"""
    print("🚀 Supreme System V5 - Production Realtime Backtest")
    print("=" * 65)
    print("💰 FREE data sources - NO API keys required")
    print("📊 Professional recording and monitoring")
    print("=" * 65)

    # Parse arguments
    args = parse_arguments()

    # Create configuration
    config = create_config_from_args(args)

    # Display configuration
    print("📊 Production Configuration:")
    print(f"   Symbols: {', '.join(config.symbols)}")
    print(f"   Balance: ${config.initial_balance:.2f}")
    print(f"   Data Sources: {', '.join(config.data_sources)} (FREE)")
    print(f"   Interval: {config.realtime_interval}s")
    print(f"   Risk Management: {'Enabled' if config.enable_risk_management else 'Disabled'}")
    print(f"   Recording: Raw={args.record_raw}")
    print()

    print("🎯 Starting Production Realtime Backtest...")
    print("🎛️ Press Ctrl+C to stop and generate comprehensive report")
    print("=" * 65)

    try:
        # Run the backtest using the existing system
        await run_realtime_backtest(config)

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
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        
        # Create timestamped run directory
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.run_dir = self.out_dir / f"run_{timestamp}"
        self.run_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize files
        self.trades_file = self.run_dir / "trades.csv"
        self.performance_file = self.run_dir / "performance.csv"
        self.config_file = self.run_dir / "config.json"
        self.log_file = self.run_dir / "backtest.log"
        
        # Initialize CSV headers
        self._init_files()
        
        # Raw data recording
        self.record_raw = record_raw
        self.raw_data = deque(maxlen=50000) if record_raw else None  # i3-4GB optimized
        
        print(f"📁 Recording to: {self.run_dir}")
    
    def _init_files(self):
        """Initialize CSV files with headers"""
        # Trades CSV
        with open(self.trades_file, 'w') as f:
            f.write("timestamp,symbol,side,quantity,entry_price,exit_price,pnl,duration_seconds,metadata\n")
        
        # Performance CSV  
        with open(self.performance_file, 'w') as f:
            f.write("timestamp,balance,pnl,pnl_percent,drawdown_percent,total_trades,exposure,cpu_percent,memory_mb\n")
    
    def record_trade(self, trade_data: Dict[str, Any]):
        """Record completed trade"""
        try:
            with open(self.trades_file, 'a') as f:
                timestamp = int(trade_data.get('timestamp', time.time()))
                symbol = trade_data.get('symbol', '')
                side = trade_data.get('side', trade_data.get('type', ''))
                quantity = trade_data.get('quantity', 0)
                entry_price = trade_data.get('entry_price', 0)
                exit_price = trade_data.get('exit_price', entry_price)
                pnl = trade_data.get('pnl', 0)
                duration = trade_data.get('duration', 0)
                metadata = str(trade_data.get('metadata', '')).replace(',', ';')
                
                f.write(f"{timestamp},{symbol},{side},{quantity:.8f},{entry_price:.8f},"
                       f"{exit_price:.8f},{pnl:.8f},{duration},{metadata}\n")
        except Exception as e:
            print(f"⚠️ Trade recording error: {e}")
    
    def record_performance(self, metrics: Dict[str, Any]):
        """Record performance metrics"""
        try:
            # Get system metrics if available
            cpu_percent = 0
            memory_mb = 0
            if PSUTIL_AVAILABLE:
                process = psutil.Process()
                cpu_percent = process.cpu_percent()
                memory_mb = process.memory_info().rss / 1024 / 1024
            
            with open(self.performance_file, 'a') as f:
                timestamp = int(time.time())
                balance = metrics.get('balance', 0)
                pnl = metrics.get('pnl', 0) 
                pnl_percent = metrics.get('pnl_percent', 0)
                drawdown_percent = metrics.get('max_drawdown_pct', 0)
                total_trades = metrics.get('total_trades', 0)
                exposure = metrics.get('exposure', 0)
                
                f.write(f"{timestamp},{balance:.2f},{pnl:.2f},{pnl_percent:.2f},"
                       f"{drawdown_percent:.2f},{total_trades},{exposure:.2f},"
                       f"{cpu_percent:.1f},{memory_mb:.1f}\n")
        except Exception as e:
            print(f"⚠️ Performance recording error: {e}")
    
    def save_config(self, config: Dict[str, Any]):
        """Save configuration"""
        try:
            with open(self.config_file, 'w') as f:
                json.dump(config, f, indent=2, default=str)
        except Exception as e:
            print(f"⚠️ Config save error: {e}")
    
    def log_message(self, message: str):
        """Log message to file"""
        try:
            with open(self.log_file, 'a') as f:
                timestamp = datetime.now().strftime("%H:%M:%S")
                f.write(f"[{timestamp}] {message}\n")
        except Exception as e:
            print(f"⚠️ Logging error: {e}")


class ProductionMetricsServer:
    """Production-ready metrics server"""
    def __init__(self, port: int = 0):
        self.port = port
        self.enabled = port > 0 and PROMETHEUS_AVAILABLE
        
        if self.enabled:
            try:
                self.balance_gauge = Gauge('backtest_balance_usd', 'Portfolio balance')
                self.pnl_gauge = Gauge('backtest_pnl_percent', 'PnL percentage')
                self.trades_counter = Counter('backtest_trades_total', 'Total trades')
                self.cpu_gauge = Gauge('backtest_cpu_percent', 'CPU usage')
                
                start_http_server(port)
                print(f"📊 Metrics server started on http://localhost:{port}")
            except Exception as e:
                print(f"⚠️ Metrics server failed: {e}")
                self.enabled = False
    
    def update_metrics(self, metrics: Dict[str, Any]):
        """Update Prometheus metrics"""
        if not self.enabled:
            return
        
        try:
            self.balance_gauge.set(metrics.get('balance', 0))
            self.pnl_gauge.set(metrics.get('pnl_percent', 0))
            self.trades_counter._value._value = metrics.get('total_trades', 0)
            
            if PSUTIL_AVAILABLE:
                self.cpu_gauge.set(psutil.Process().cpu_percent())
        except Exception as e:
            print(f"⚠️ Metrics update error: {e}")


def validate_args(args):
    """Validate command line arguments"""
    errors = []
    
    if not args.symbols:
        errors.append("At least one symbol is required")
    
    if args.balance <= 0:
        errors.append("Initial balance must be > 0")
        
    if args.interval <= 0:
        errors.append("Update interval must be > 0")
        
    if not (0 <= args.max_position <= 1):
        errors.append("Max position size must be between 0 and 1")
    
    if errors:
        print("❌ Configuration errors:")
        for error in errors:
            print(f"   • {error}")
        sys.exit(2)


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
