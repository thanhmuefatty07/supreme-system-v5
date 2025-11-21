#!/usr/bin/env python3
"""
Supreme System V5 - Live Trading Demo Runner

Run a safe demonstration of live algorithmic trading with real Binance data.
This demo uses MOCK trading (no real orders placed) but processes REAL market data.

Features:
- Live Binance WebSocket connection
- Real-time market data processing
- Strategy signal generation
- Risk management calculations
- Performance monitoring
- Graceful shutdown handling

Usage:
    python scripts/run_live_demo.py [duration_seconds]

Arguments:
    duration_seconds: Demo duration in seconds (default: 60)

Controls:
    Ctrl+C: Stop demo gracefully
    Logs: Monitor real-time activity
"""

import asyncio
import logging
import os
import signal
import sys
import time
from typing import Dict, Any, Optional
from datetime import datetime

# Determine paths
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)

# Add src to path
sys.path.insert(0, os.path.join(project_root, 'src'))

try:
    # Import modules
    from trading.live_trading_engine_v2 import LiveTradingEngineV2
    from strategies.mean_reversion import MeanReversionStrategy
    from strategies.momentum import MomentumStrategy
    from strategies.sma_crossover import SMACrossover
    from data.live_data_manager import LiveDataManager

except ImportError as e:
    print(f"Import error: {e}")
    print(f"Project root: {project_root}")
    print(f"Script dir: {script_dir}")
    print("Trying alternative import method...")

    try:
        # Fallback: run as module
        import subprocess
        import sys
        result = subprocess.run([
            sys.executable, "-m", "src.supreme_system_app",
            "--demo", "30", "momentum", "BTCUSDT"
        ], cwd=project_root, capture_output=True, text=True)

        print("Demo output:")
        print(result.stdout)
        if result.stderr:
            print("Errors:")
            print(result.stderr)
        sys.exit(result.returncode)

    except Exception as fallback_e:
        print(f"Fallback also failed: {fallback_e}")
        print("Please run from project root: python scripts/run_live_demo.py")
        sys.exit(1)

# Configure logging for demo
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('demo_live_trading.log')
    ]
)
logger = logging.getLogger(__name__)


class LiveDemoRunner:
    """
    Manages the live trading demo with real market data and mock execution.
    """

    def __init__(self, symbol: str = "BTCUSDT", strategy_name: str = "momentum", duration: int = 60):
        """
        Initialize the demo runner.

        Args:
            symbol: Trading symbol (e.g., 'BTCUSDT')
            strategy_name: Strategy to use ('sma', 'mean_reversion', 'momentum')
            duration: Demo duration in seconds
        """
        self.symbol = symbol
        self.strategy_name = strategy_name
        self.duration = duration
        self.engine: Optional[LiveTradingEngineV2] = None
        self.start_time = None
        self.running = False

        # Demo statistics
        self.stats = {
            'data_points_received': 0,
            'signals_generated': 0,
            'start_price': None,
            'current_price': None,
            'price_changes': 0
        }

        logger.info(f"🎯 Live Demo initialized: {symbol} with {strategy_name} strategy for {duration}s")

    async def setup_demo(self):
        """Set up the trading engine and data manager."""

        # Mock exchange for safe demo (no real trading)
        class MockExchange:
            """Mock exchange that simulates order execution without real trading."""

            async def fetch_order_book(self, symbol):
                # Return mock order book with reasonable spreads
                base_price = 50000.0 if symbol == "BTCUSDT" else 3000.0
                spread = base_price * 0.001  # 0.1% spread
                return {
                    'asks': [[base_price + spread, 10.0]],
                    'bids': [[base_price - spread, 10.0]]
                }

            async def create_order(self, symbol, side, quantity, price):
                # Mock successful order execution
                order_id = f"demo_{symbol}_{side}_{int(time.time())}"
                logger.info(f"📋 MOCK ORDER: {side.upper()} {quantity} {symbol} @ ${price:.2f} (ID: {order_id})")
                return {'id': order_id, 'status': 'filled'}

        mock_exchange = MockExchange()

        # Create strategy based on selection
        strategy_config = self._get_strategy_config()
        strategy_class = self._get_strategy_class()
        strategy = strategy_class(strategy_config)

        # Engine configuration for demo
        engine_config = {
            'initial_capital': 10000.0,
            'symbols': [self.symbol],
            'data_interval': '1m',  # 1-minute klines for responsive demo
            'trade_log_path': f'demo_trades_{self.symbol.lower()}.jsonl',
            'risk_config': {
                'max_risk_per_trade': 0.02,  # Conservative 2% risk per trade
                'kelly_mode': 'half',
                'daily_loss_limit': 0.05,
                'max_position_pct': 0.1,   # Max 10% position
                'max_portfolio_pct': 0.5,  # Max 50% exposure
                'max_consecutive_losses': 3
            },
            'data_config': {
                'validate_data': True,
                'buffer_size': 100,
                'reconnect_delay': 2.0,
                'max_reconnect_attempts': 5,
                'timeout': 5.0,
                'ping_interval': 10.0
            }
        }

        # Create the trading engine
        self.engine = LiveTradingEngineV2(mock_exchange, strategy, engine_config)

        # Add custom data callback for demo statistics
        self.engine.data_manager.add_data_callback(self._on_demo_data_received)

        logger.info("✅ Demo setup complete - ready for live data!")

    def _get_strategy_config(self) -> Dict[str, Any]:
        """Get strategy configuration based on selection."""
        base_config = {
            'buffer_size': 50,  # Smaller buffer for demo responsiveness
            'min_signal_strength': 0.1
        }

        if self.strategy_name == "sma":
            base_config.update({
                'fast_window': 5,
                'slow_window': 15
            })
        elif self.strategy_name == "mean_reversion":
            base_config.update({
                'lookback_period': 10,
                'entry_threshold': 1.5,
                'use_rsi': True,
                'rsi_period': 14,
                'rsi_overbought': 70,
                'rsi_oversold': 30
            })
        elif self.strategy_name == "momentum":
            base_config.update({
                'short_period': 8,
                'long_period': 21,
                'signal_period': 5,
                'roc_period': 7,
                'trend_threshold': 0.01,
                'volume_confirmation': False  # Simplify for demo
            })

        return base_config

    def _get_strategy_class(self):
        """Get strategy class based on selection."""
        strategies = {
            'sma': SMACrossover,
            'mean_reversion': MeanReversionStrategy,
            'momentum': MomentumStrategy
        }
        return strategies.get(self.strategy_name, MomentumStrategy)

    async def _on_demo_data_received(self, data: Dict[str, Any]):
        """Callback for tracking demo statistics."""
        self.stats['data_points_received'] += 1

        # Track price changes
        current_price = data.get('close', 0)
        if self.stats['start_price'] is None:
            self.stats['start_price'] = current_price

        if self.stats['current_price'] is not None and self.stats['current_price'] != current_price:
            self.stats['price_changes'] += 1

        self.stats['current_price'] = current_price

    async def run_demo(self):
        """Run the live trading demo."""
        logger.info("🚀 Starting Live Trading Demo...")
        logger.info("📡 Connecting to Binance WebSocket for real market data...")
        logger.info("⚠️  This is a SAFE DEMO - No real trading occurs!")
        logger.info("=" * 60)

        # Set up signal handlers
        def signal_handler(signum, frame):
            logger.info("🛑 Demo interrupted by user")
            self.running = False

        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)

        self.running = True
        self.start_time = time.time()

        try:
            # Start the live trading
            demo_task = asyncio.create_task(self.engine.start_live_trading())

            # Monitor progress
            last_status_time = 0
            while self.running and (time.time() - self.start_time) < self.duration:
                current_time = time.time()
                elapsed = int(current_time - self.start_time)

                # Print status every 10 seconds
                if current_time - last_status_time >= 10:
                    await self._print_status(elapsed)
                    last_status_time = current_time

                await asyncio.sleep(1)

            logger.info("🏁 Demo duration completed")

        except Exception as e:
            logger.error(f"Demo error: {e}")
            import traceback
            traceback.print_exc()
        finally:
            logger.info("🧹 Shutting down demo...")
            self.running = False

            if self.engine:
                await self.engine._shutdown()

    async def _print_status(self, elapsed: int):
        """Print current demo status."""
        if not self.engine:
            return

        # Get status from engine and data manager
        data_status = self.engine.get_data_status()

        # Calculate progress
        progress = min(100, int((elapsed / self.duration) * 100))

        print(f"\n⏱️  [{elapsed:3d}s/{self.duration}s] ({progress:3d}%)")
        print(f"   📡 Data Connected: {data_status['data_connected']}")
        print(f"   💬 Messages: {data_status['messages_received']}")
        print(f"   🔄 Reconnects: {data_status['reconnect_count']}")
        print(f"   ⚠️  Errors: {data_status['data_errors']}")

        if self.stats['current_price']:
            print(f"   💰 Current Price: ${self.stats['current_price']:,.2f}")

        if self.stats['start_price'] and self.stats['current_price']:
            change_pct = ((self.stats['current_price'] - self.stats['start_price']) / self.stats['start_price']) * 100
            print(f"   📈 Change: {change_pct:+.2f}%")

        print(f"   📊 Data Points: {self.stats['data_points_received']}")
        print(f"   🎯 Price Changes: {self.stats['price_changes']}")

        # Engine performance
        try:
            engine_status = self.engine.get_status()
            print(f"   💵 P&L: ${engine_status['total_pnl']:+,.2f}")
            print(f"   🎯 Signals: {engine_status['total_signals']}")
        except:
            pass

        print("-" * 40, end="", flush=True)

    async def show_summary(self):
        """Show demo summary."""
        if not self.engine:
            return

        duration = time.time() - self.start_time
        data_status = self.engine.get_data_status()

        print("\n" + "=" * 60)
        print("🎯 LIVE TRADING DEMO SUMMARY")
        print("=" * 60)
        print(f"⏱️  Duration: {duration:.1f} seconds")
        print(f"📊 Symbol: {self.symbol}")
        print(f"🎲 Strategy: {self.strategy_name}")
        print()
        print("📡 DATA STREAM PERFORMANCE:")
        print(f"🔗 Connected: {data_status['data_connected']}")
        print(f"💬 Messages Received: {data_status['messages_received']}")
        print(f"⚠️  Errors: {data_status['data_errors']}")
        print(f"🔄 Reconnects: {data_status['reconnect_count']}")
        print(f"📶 Data Rate: {self.stats['data_points_received']/duration:.1f} msg/sec")
        print()
        print("💰 MARKET DATA:")
        if self.stats['start_price'] and self.stats['current_price']:
            change = self.stats['current_price'] - self.stats['start_price']
            change_pct = (change / self.stats['start_price']) * 100
            print(f"📈 Start Price: ${self.stats['start_price']:,.2f}")
            print(f"🏁 End Price: ${self.stats['current_price']:,.2f}")
            print(f"📊 Change: ${change:+,.2f} ({change_pct:+.2f}%)")
        print(f"🔄 Price Updates: {self.stats['price_changes']}")
        print()
        print("🤖 TRADING ENGINE:")
        try:
            engine_status = self.engine.get_status()
            print(f"💵 Total P&L: ${engine_status['total_pnl']:+,.2f}")
            print(f"🎯 Signals Generated: {engine_status['total_signals']}")
            print(f"✅ Signals Executed: {engine_status['executed_signals']}")
            print(f"🏆 Win Rate: {engine_status.get('win_rate', 0):.1%}")
            print(f"🛡️  Circuit Breaker: {'ACTIVE' if engine_status.get('circuit_breaker_active', False) else 'OK'}")
        except Exception as e:
            print(f"⚠️  Could not retrieve engine status: {e}")
        print()
        print("✅ Demo completed successfully!")
        print("🔍 Check logs for detailed activity")
        print("=" * 60)


async def main():
    """Main demo function."""
    # Parse command line arguments
    duration = 60  # default 1 minute
    strategy = "momentum"  # default strategy
    symbol = "BTCUSDT"  # default symbol

    if len(sys.argv) > 1:
        try:
            duration = int(sys.argv[1])
        except ValueError:
            print("Invalid duration, using default 60 seconds")

    if len(sys.argv) > 2:
        strategy = sys.argv[2]

    if len(sys.argv) > 3:
        symbol = sys.argv[3]

    print("🚀 Supreme System V5 - Live Trading Demo")
    print(f"📊 Symbol: {symbol}")
    print(f"🎲 Strategy: {strategy}")
    print(f"⏱️  Duration: {duration} seconds")
    print("📡 Real market data from Binance WebSocket")
    print("🛡️  Safe demo - no real trading")
    print("=" * 60)

    demo = LiveDemoRunner(symbol=symbol, strategy_name=strategy, duration=duration)

    try:
        await demo.setup_demo()
        await demo.run_demo()
        await demo.show_summary()
    except KeyboardInterrupt:
        print("\n🛑 Demo interrupted by user")
        await demo.show_summary()
    except Exception as e:
        print(f"\n❌ Demo failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    # Run the demo
    asyncio.run(main())
