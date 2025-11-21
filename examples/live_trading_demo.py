#!/usr/bin/env python3
"""
Supreme System V5 - Live Trading Demo

Demonstrates the complete live data integration pipeline:
1. LiveDataManager connects to Binance WebSocket
2. Real-time market data flows to LiveTradingEngineV2
3. Strategies process live data and generate signals
4. Risk management and execution engine handle trades

This is a SAFE DEMO - no real trading occurs, only simulation with live data.
"""

import asyncio
import logging
import signal
import sys
from typing import Dict, Any

from src.trading.live_trading_engine_v2 import LiveTradingEngineV2
from src.strategies.mean_reversion import MeanReversionStrategy
from src.strategies.momentum import MomentumStrategy
from src.data.live_data_manager import LiveDataManager

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class LiveTradingDemo:
    """
    Demonstration of live trading with real market data.

    This demo shows how the system processes live Binance data
    through the complete trading pipeline.
    """

    def __init__(self):
        self.engine = None
        self.data_manager = None
        self.running = False

    async def setup_demo(self):
        """Set up the demo with mock exchange and live data."""

        # Mock exchange client (for demo purposes - no real trading)
        class MockExchange:
            async def fetch_order_book(self, symbol):
                # Return mock order book
                return {
                    'asks': [[50000.0, 10.0]],
                    'bids': [[49900.0, 10.0]]
                }

            async def create_order(self, symbol, side, quantity, price):
                # Mock order execution
                logger.info(f"MOCK ORDER: {side.upper()} {quantity} {symbol} @ {price}")
                return {'id': f'mock_order_{symbol}_{side}', 'status': 'filled'}

        mock_exchange = MockExchange()

        # Configure strategies
        mean_reversion_config = {
            'lookback_period': 10,  # Shorter for demo
            'entry_threshold': 1.5,
            'use_rsi': True,
            'min_signal_strength': 0.1,
            'buffer_size': 50
        }

        momentum_config = {
            'short_period': 8,
            'long_period': 21,
            'signal_period': 5,
            'roc_period': 7,
            'trend_threshold': 0.01,
            'min_signal_strength': 0.1,
            'buffer_size': 50
        }

        # Create strategies
        strategies = {
            'mean_reversion': MeanReversionStrategy(mean_reversion_config),
            'momentum': MomentumStrategy(momentum_config)
        }

        # Choose strategy for demo (you can change this)
        selected_strategy = strategies['momentum']  # Change to 'mean_reversion' if preferred

        # Engine configuration
        engine_config = {
            'initial_capital': 10000.0,
            'symbols': ['BTCUSDT'],  # Single symbol for demo
            'data_interval': '1m',   # 1-minute klines
            'trade_log_path': 'demo_live_trades.jsonl',
            'risk_config': {
                'max_risk_per_trade': 0.01,  # Conservative 1% risk
                'kelly_mode': 'half',
                'daily_loss_limit': 0.05,
                'max_position_pct': 0.05,   # Max 5% position
                'max_portfolio_pct': 0.20    # Max 20% exposure
            },
            'data_config': {
                'validate_data': True,
                'buffer_size': 100,
                'reconnect_delay': 2.0,      # Faster reconnect for demo
                'max_reconnect_attempts': 5,
                'timeout': 5.0,
                'ping_interval': 10.0
            }
        }

        # Create engine with live data integration
        self.engine = LiveTradingEngineV2(mock_exchange, selected_strategy, engine_config)

        logger.info("🎯 Live Trading Demo initialized!")
        logger.info(f"📊 Strategy: {selected_strategy.__class__.__name__}")
        logger.info(f"💰 Initial Capital: ${engine_config['initial_capital']:,.2f}")
        logger.info(f"📈 Symbols: {engine_config['symbols']}")
        logger.info("⚠️  This is a SAFE DEMO - No real trading occurs!")

    async def run_demo(self, duration_seconds: int = 60):
        """
        Run the live trading demo for specified duration.

        Args:
            duration_seconds: How long to run the demo
        """
        logger.info(f"🚀 Starting Live Trading Demo for {duration_seconds} seconds...")
        logger.info("📡 Connecting to Binance WebSocket for real market data...")

        # Set up signal handlers for graceful shutdown
        def signal_handler(signum, frame):
            logger.info("🛑 Shutdown signal received")
            self.running = False

        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)

        self.running = True
        start_time = asyncio.get_event_loop().time()

        try:
            # Create tasks for concurrent execution
            streaming_task = asyncio.create_task(self.engine.start_live_trading())

            # Monitor progress
            while self.running and (asyncio.get_event_loop().time() - start_time) < duration_seconds:
                await asyncio.sleep(5)  # Check every 5 seconds

                # Report status
                data_status = self.engine.get_data_status()
                engine_status = self.engine.get_status()

                elapsed = int(asyncio.get_event_loop().time() - start_time)
                logger.info(f"⏱️  Elapsed: {elapsed}s | "
                          f"📊 Data Connected: {data_status['data_connected']} | "
                          f"💬 Messages: {data_status['messages_received']} | "
                          f"📈 PnL: ${engine_status['total_pnl']:.2f} | "
                          f"🎯 Signals: {engine_status['total_signals']}")

                # Safety check - stop if too many errors
                if data_status.get('data_errors', 0) > 10:
                    logger.warning("Too many data errors, stopping demo")
                    break

            logger.info("🏁 Demo duration completed")

        except Exception as e:
            logger.error(f"Demo error: {e}")
        finally:
            logger.info("🧹 Cleaning up...")
            self.running = False

            # Graceful shutdown
            await self.engine._shutdown()

    async def show_summary(self):
        """Show demo summary and results."""
        if not self.engine:
            return

        status = self.engine.get_status()
        data_status = self.engine.get_data_status()

        print("\n" + "="*60)
        print("🎯 LIVE TRADING DEMO SUMMARY")
        print("="*60)
        print(f"💰 Initial Capital:    ${10000.0:,.2f}")
        print(f"💵 Final Capital:      ${status['portfolio_value']:,.2f}")
        print(f"📈 Total P&L:          ${status['total_pnl']:,.2f}")
        print(f"🎯 Total Signals:      {status['total_signals']}")
        print(f"✅ Signals Executed:   {status['executed_signals']}")
        print(f"🏆 Winning Trades:     {status['successful_signals']}")
        print(f"📊 Win Rate:           {status['win_rate']:.1%}")
        print()
        print("📡 DATA STREAM STATUS:")
        print(f"🔗 Connected:          {data_status['data_connected']}")
        print(f"💬 Messages Received:  {data_status['messages_received']}")
        print(f"⚠️  Errors:            {data_status['data_errors']}")
        print(f"🔄 Reconnects:         {data_status['reconnect_count']}")
        print(f"⏱️  Uptime:            {data_status.get('uptime_seconds', 0):.1f}s")
        print()
        print("🎲 RISK MANAGEMENT:")
        print(f"🛡️  Circuit Breaker:   {'ACTIVE' if status['circuit_breaker_active'] else 'OK'}")
        print(f"📉 Current Drawdown:   {status['risk_metrics']['current_daily_drawdown']:.2%}")
        print("="*60)
        print("✅ Demo completed successfully!")
        print("💡 Real market data was processed through the complete trading pipeline")
        print("🔒 No real trading occurred - this was a safe simulation")


async def main():
    """Main demo function."""
    print("🚀 Supreme System V5 - Live Trading Demo")
    print("📡 Connecting to REAL Binance market data (safe simulation)")
    print("="*60)

    demo = LiveTradingDemo()

    try:
        # Setup
        await demo.setup_demo()

        # Run demo for 30 seconds (adjust as needed)
        await demo.run_demo(duration_seconds=30)

        # Show results
        await demo.show_summary()

    except KeyboardInterrupt:
        print("\n🛑 Demo interrupted by user")
    except Exception as e:
        print(f"\n❌ Demo failed: {e}")
        import traceback
        traceback.print_exc()
    finally:
        print("\n👋 Demo finished!")


if __name__ == "__main__":
    # Run the demo
    asyncio.run(main())
