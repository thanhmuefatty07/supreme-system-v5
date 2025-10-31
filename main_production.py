#!/usr/bin/env python3
"""
Supreme System V5 - Production Entry Point
ULTRA SFL implementation - complete trading system
"""

import asyncio
import signal
import sys
import os
from pathlib import Path
from datetime import datetime
from typing import Optional

# Add python package to path
sys.path.insert(0, str(Path(__file__).parent / "python"))

# Production imports
from loguru import logger
from rich.console import Console
from rich.panel import Panel
from rich.live import Live
from rich.table import Table

from supreme_system_v5.core import SupremeSystem, SystemConfig
from supreme_system_v5.exchanges import OKXConnector, BinanceConnector, ExchangeConfig

console = Console()

class ProductionTradingSystem:
    """
    Production-ready trading system orchestrator
    Manages all components for live trading
    """
    
    def __init__(self):
        """Initialize production system"""
        self.config = self._load_production_config()
        self.supreme_system = SupremeSystem()
        self.exchanges = {}
        self.running = False
        self.start_time = datetime.now()
        
        logger.info("🎆 Production Trading System initialized")
    
    def _load_production_config(self) -> SystemConfig:
        """
        Load production configuration from environment
        """
        config = SystemConfig()
        
        # Override with environment variables if available
        config.max_position_size = float(os.getenv('MAX_POSITION_SIZE', '0.01'))
        config.stop_loss_percent = float(os.getenv('STOP_LOSS_PERCENT', '0.005'))
        config.max_memory_mb = int(os.getenv('MAX_MEMORY_MB', '3500'))
        
        # Trading symbols from environment
        symbols_env = os.getenv('TRADING_SYMBOLS', 'BTC-USDT,ETH-USDT')
        config.trading_symbols = [s.strip() for s in symbols_env.split(',')]
        
        logger.info(f"📜 Production config loaded: {len(config.trading_symbols)} symbols")
        return config
    
    def _setup_signal_handlers(self):
        """
        Setup graceful shutdown signal handlers
        """
        def signal_handler(signum, frame):
            logger.info(f"Received signal {signum}, initiating graceful shutdown")
            asyncio.create_task(self.shutdown())
        
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
        logger.info("⚙️ Signal handlers configured")
    
    async def initialize_exchanges(self):
        """
        Initialize exchange connections
        """
        logger.info("🔗 Initializing exchange connections...")
        
        # OKX configuration
        okx_config = ExchangeConfig(
            api_key=os.getenv('OKX_API_KEY', ''),
            secret_key=os.getenv('OKX_SECRET_KEY', ''),
            passphrase=os.getenv('OKX_PASSPHRASE', ''),
            sandbox=os.getenv('TRADING_MODE', 'sandbox') == 'sandbox'
        )
        
        # Binance configuration
        binance_config = ExchangeConfig(
            api_key=os.getenv('BINANCE_API_KEY', ''),
            secret_key=os.getenv('BINANCE_SECRET_KEY', ''),
            sandbox=os.getenv('TRADING_MODE', 'sandbox') == 'sandbox'
        )
        
        # Initialize exchanges
        try:
            self.exchanges['okx'] = OKXConnector(okx_config)
            okx_connected = await self.exchanges['okx'].connect()
            
            if okx_connected:
                logger.info("✅ OKX exchange connected")
                # Setup market data callback
                self.exchanges['okx'].set_market_data_callback(self._handle_market_data)
                # Subscribe to symbols
                await self.exchanges['okx'].subscribe_market_data(self.config.trading_symbols)
            else:
                logger.error("❌ OKX connection failed")
                
        except Exception as e:
            logger.error(f"❌ OKX initialization failed: {e}")
        
        try:
            self.exchanges['binance'] = BinanceConnector(binance_config)
            binance_connected = await self.exchanges['binance'].connect()
            
            if binance_connected:
                logger.info("✅ Binance exchange connected (backup)")
            else:
                logger.warning("⚠️ Binance connection failed (backup only)")
                
        except Exception as e:
            logger.warning(f"⚠️ Binance backup failed: {e}")
    
    async def _handle_market_data(self, symbol: str, price: float, volume: float, bid: float, ask: float):
        """
        Handle incoming market data from exchanges
        """
        # Update core system with market data
        self.supreme_system.core.update_market_data(symbol, price, volume, bid, ask)
        
        # Generate and potentially execute trading signals
        signal = self.supreme_system.core.generate_trading_signal(symbol)
        
        if signal.action != "HOLD" and signal.confidence > 0.7:
            # High confidence signal - execute trade
            executed = self.supreme_system.core.execute_trade(signal)
            if executed:
                logger.info(f"✅ Trade executed: {signal.action} {symbol} @ {price:.4f}")
    
    def _create_status_table(self) -> Table:
        """
        Create real-time status table for display
        """
        table = Table(title="Supreme System V5 - Live Status")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="white")
        
        try:
            status = self.supreme_system.get_status()
            uptime = datetime.now() - self.start_time
            
            table.add_row("Status", "✅ RUNNING" if status['status'] == 'running' else "❌ STOPPED")
            table.add_row("Uptime", str(uptime).split('.')[0])  # Remove microseconds
            table.add_row("Memory Usage", f"{status['core_info']['memory_usage_mb']:.1f} MB")
            table.add_row("CPU Usage", f"{status['core_info']['cpu_percent']:.1f}%")
            table.add_row("Active Symbols", str(len(status['core_info']['active_symbols'])))
            table.add_row("Total Trades", str(status['performance']['total_trades']))
            table.add_row("P&L", f"${status['performance']['total_pnl']:.2f}")
            
        except Exception as e:
            table.add_row("Error", str(e))
        
        return table
    
    async def run_with_live_display(self):
        """
        Run system with live status display
        """
        console.print("🚀 [bold green]Starting Production Trading System[/bold green]")
        console.print(f"📅 Start time: {self.start_time.strftime('%Y-%m-%d %H:%M:%S')}")
        console.print(f"💼 Trading symbols: {', '.join(self.config.trading_symbols)}")
        console.print(f"🔧 Mode: {'LIVE' if os.getenv('TRADING_MODE') == 'live' else 'SANDBOX'}")
        
        # Setup signal handlers
        self._setup_signal_handlers()
        
        # Initialize exchanges
        await self.initialize_exchanges()
        
        # Start supreme system
        self.running = True
        
        # Create tasks
        trading_task = asyncio.create_task(self.supreme_system.run())
        
        # Live display (optional - comment out for headless)
        # with Live(self._create_status_table(), refresh_per_second=2) as live:
        #     while self.running:
        #         live.update(self._create_status_table())
        #         await asyncio.sleep(0.5)
        
        # Wait for trading task
        await trading_task
    
    async def shutdown(self):
        """
        Graceful shutdown of production system
        """
        logger.info("🛑 Initiating graceful shutdown...")
        
        self.running = False
        
        # Stop supreme system
        await self.supreme_system.core.stop()
        
        # Disconnect exchanges
        for exchange_name, exchange in self.exchanges.items():
            try:
                await exchange.disconnect()
                logger.info(f"✅ {exchange_name} disconnected")
            except Exception as e:
                logger.error(f"❌ {exchange_name} disconnect error: {e}")
        
        logger.info("✅ Production system shutdown completed")

async def main():
    """
    Main production entry point
    """
    # Configure logging for production
    logger.remove()  # Remove default handler
    
    # File logging for production
    logger.add(
        "logs/supreme_system_{time:YYYY-MM-DD}.log",
        rotation="1 day",
        retention="30 days",
        level="INFO",
        format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {level: <8} | {name}:{function}:{line} - {message}"
    )
    
    # Console logging
    logger.add(
        sys.stdout,
        level="INFO",
        format="<green>{time:HH:mm:ss.SSS}</green> | <level>{level: <8}</level> | <cyan>{message}</cyan>"
    )
    
    try:
        # Create and run production system
        production_system = ProductionTradingSystem()
        await production_system.run_with_live_display()
        
    except Exception as e:
        logger.error(f"💥 Production system fatal error: {e}")
        console.print(f"\n❌ [bold red]FATAL ERROR: {e}[/bold red]")
        sys.exit(1)

if __name__ == "__main__":
    # Production banner
    console.print(Panel(
        "🏆 [bold yellow]SUPREME SYSTEM V5 - PRODUCTION MODE[/bold yellow]\n"
        "Hybrid Python+Rust High-Performance Trading System\n"
        "Optimized for i3-4GB systems | Futures Scalping Specialist\n\n"
        "⚡ Ultra-Low Latency | 🔒 Enterprise Security | 📊 Real-time Monitoring",
        title="Production Trading System",
        border_style="gold1",
        width=80
    ))
    
    # Ensure logs directory exists
    Path("logs").mkdir(exist_ok=True)
    
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        console.print("\n🛑 [yellow]Production system interrupted by user[/yellow]")
    except Exception as e:
        console.print(f"\n💥 [bold red]FATAL: {e}[/bold red]")
        sys.exit(1)
