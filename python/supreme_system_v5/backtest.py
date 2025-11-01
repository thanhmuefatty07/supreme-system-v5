# python/supreme_system_v5/backtest.py
"""
Real-Time Backtesting Engine - ULTRA SFL Implementation
Enterprise-grade backtesting with real-time data integration
"""

import asyncio
import signal
import sys
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from threading import Event, Thread
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
from loguru import logger

from .data_fabric import DataAggregator
from .data_fabric.cache import CacheManager
from .event_bus import get_event_bus, create_market_data_event, EventPriority
from .risk import RiskManager, RiskLimits, PortfolioState
from .strategies import ScalpingStrategy, SignalType, TradingSignal
from .utils import get_logger


@dataclass
class BacktestConfig:
    """Configuration for real-time backtesting"""

    # Data sources
    symbols: List[str] = field(default_factory=lambda: ["BTC-USDT", "ETH-USDT"])
    data_sources: List[str] = field(default_factory=lambda: ["binance", "coingecko"])

    # Time configuration
    historical_days: int = 30  # Days of historical data to load
    realtime_interval: float = 1.0  # Seconds between real-time updates

    # Trading configuration
    initial_balance: float = 10000.0
    max_position_size: float = 0.1  # 10% of portfolio per position
    enable_risk_management: bool = True

    # Strategy configuration
    strategy_config: Dict[str, Any] = field(default_factory=dict)

    # Performance tracking
    enable_detailed_logging: bool = True
    log_interval: int = 100  # Log every N updates


@dataclass
class BacktestMetrics:
    """Real-time backtesting performance metrics"""

    # Session info
    start_time: float = field(default_factory=time.time)
    total_updates: int = 0
    active_symbols: int = 0

    # Portfolio
    initial_balance: float = 10000.0
    current_balance: float = 10000.0
    peak_balance: float = 10000.0
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0

    # Performance
    total_pnl: float = 0.0
    total_pnl_percent: float = 0.0
    max_drawdown: float = 0.0
    max_drawdown_percent: float = 0.0
    sharpe_ratio: float = 0.0
    win_rate: float = 0.0

    # Risk metrics
    current_exposure: float = 0.0
    max_exposure: float = 0.0
    risk_violations: int = 0

    # Timing
    avg_update_time: float = 0.0
    max_update_time: float = 0.0
    total_processing_time: float = 0.0

    # Data quality
    data_points_received: int = 0
    data_quality_issues: int = 0
    source_failures: int = 0

    def update_pnl(self, new_balance: float):
        """Update P&L and drawdown calculations"""
        self.current_balance = new_balance
        self.total_pnl = new_balance - self.initial_balance
        self.total_pnl_percent = (self.total_pnl / self.initial_balance) * 100

        # Update peak balance
        if new_balance > self.peak_balance:
            self.peak_balance = new_balance

        # Calculate drawdown
        current_drawdown = self.peak_balance - new_balance
        current_drawdown_percent = (current_drawdown / self.peak_balance) * 100

        if current_drawdown > self.max_drawdown:
            self.max_drawdown = current_drawdown
            self.max_drawdown_percent = current_drawdown_percent

    def calculate_win_rate(self):
        """Calculate win rate from trade statistics"""
        if self.total_trades > 0:
            self.win_rate = (self.winning_trades / self.total_trades) * 100
        else:
            self.win_rate = 0.0

    def get_summary(self) -> Dict[str, Any]:
        """Get comprehensive backtest summary"""
        runtime = time.time() - self.start_time
        updates_per_second = self.total_updates / max(runtime, 1)

        return {
            "session_duration": f"{runtime:.2f}s",
            "total_updates": self.total_updates,
            "updates_per_second": f"{updates_per_second:.2f}",
            "active_symbols": self.active_symbols,

            "portfolio": {
                "initial_balance": f"${self.initial_balance:,.2f}",
                "current_balance": f"${self.current_balance:,.2f}",
                "total_pnl": f"${self.total_pnl:,.2f}",
                "total_pnl_percent": f"{self.total_pnl_percent:+.2f}%",
                "peak_balance": f"${self.peak_balance:,.2f}",
            },

            "performance": {
                "max_drawdown": f"${self.max_drawdown:,.2f}",
                "max_drawdown_percent": f"{self.max_drawdown_percent:.2f}%",
                "sharpe_ratio": f"{self.sharpe_ratio:.2f}",
                "win_rate": f"{self.win_rate:.1f}%",
            },

            "trading": {
                "total_trades": self.total_trades,
                "winning_trades": self.winning_trades,
                "losing_trades": self.losing_trades,
                "current_exposure": f"${self.current_exposure:,.2f}",
                "max_exposure": f"${self.max_exposure:,.2f}",
                "risk_violations": self.risk_violations,
            },

            "performance": {
                "avg_update_time": f"{self.avg_update_time*1000:.2f}ms",
                "max_update_time": f"{self.max_update_time*1000:.2f}ms",
                "total_processing_time": f"{self.total_processing_time:.2f}s",
            },

            "data_quality": {
                "data_points_received": self.data_points_received,
                "data_quality_issues": self.data_quality_issues,
                "source_failures": self.source_failures,
                "data_quality_score": f"{((self.data_points_received - self.data_quality_issues) / max(self.data_points_received, 1)) * 100:.1f}%",
            },

            "timestamp": datetime.now().isoformat(),
        }


class RealTimeBacktestEngine:
    """
    Real-Time Backtesting Engine - ULTRA SFL Implementation
    Performs live backtesting with real-time market data
    """

    def __init__(self, config: Optional[BacktestConfig] = None):
        self.config = config or BacktestConfig()
        self.metrics = BacktestMetrics()
        self.metrics.initial_balance = self.config.initial_balance
        self.metrics.current_balance = self.config.initial_balance
        self.metrics.peak_balance = self.config.initial_balance

        # Core components
        self.data_aggregator: Optional[DataAggregator] = None
        self.cache_manager: Optional[CacheManager] = None
        self.risk_manager: Optional[RiskManager] = None
        self.strategy: Optional[ScalpingStrategy] = None

        # Backtesting state
        self.running = False
        self.stop_event = Event()
        self.update_thread: Optional[Thread] = None

        # Trading state
        self.portfolio: Dict[str, Dict[str, Any]] = {}
        self.pending_orders: List[Dict[str, Any]] = []
        self.completed_trades: List[Dict[str, Any]] = []

        # Data tracking
        self.price_history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=1000))
        self.last_update_time = time.time()

        # Signal handling for manual stop
        self.manual_stop_received = False

        logger.info("🎯 Real-Time Backtest Engine initialized")
        logger.info(f"📊 Configuration: {len(self.config.symbols)} symbols, {self.config.historical_days}d history")

    def setup_signal_handler(self):
        """Setup signal handler for manual stop (Ctrl+C)"""
        def signal_handler(signum, frame):
            logger.info("🛑 Manual stop signal received - preparing to stop backtesting...")
            self.manual_stop_received = True
            self.stop_event.set()

        signal.signal(signal.SIGINT, signal_handler)
        logger.info("🎛️ Signal handler configured - Press Ctrl+C to stop backtesting")

    async def initialize_components(self):
        """Initialize all backtesting components"""
        try:
            logger.info("🔧 Initializing backtesting components...")

            # Initialize cache manager
            from supreme_system_v5.data_fabric.cache import DataCache
            cache = DataCache()
            await cache.connect()
            self.cache_manager = CacheManager(cache)

            # Initialize data aggregator
            self.data_aggregator = DataAggregator(cache_manager=self.cache_manager)

            # Setup data sources
            for source_name in self.config.data_sources:
                try:
                    if source_name == "binance":
                        from supreme_system_v5.data_fabric.connectors import BinancePublicConnector
                        connector = BinancePublicConnector()
                    elif source_name == "coingecko":
                        from supreme_system_v5.data_fabric.connectors import CoinGeckoConnector
                        connector = CoinGeckoConnector()
                    elif source_name == "okx":
                        from supreme_system_v5.data_fabric.connectors import OKXPublicConnector
                        connector = OKXPublicConnector()
                    else:
                        logger.warning(f"⚠️ Unknown data source: {source_name}")
                        continue

                    # Create data source
                    from supreme_system_v5.data_fabric.aggregator import DataSource
                    source = DataSource(
                        name=f"{source_name}_source",
                        connector=connector,
                        priority=1,
                        weight=1.0
                    )
                    self.data_aggregator.add_source(source)
                    logger.info(f"✅ Added data source: {source_name}")

                except Exception as e:
                    logger.error(f"❌ Failed to initialize {source_name}: {e}")

            # Initialize risk manager
            if self.config.enable_risk_management:
                risk_limits = RiskLimits(
                    max_drawdown_percent=15.0,  # More lenient for backtesting
                    max_daily_loss_usd=self.config.initial_balance * 0.05,
                    max_position_size_usd=self.config.initial_balance * self.config.max_position_size,
                    max_leverage=2.0
                )
                portfolio_state = PortfolioState(total_value=self.config.initial_balance)
                self.risk_manager = RiskManager(limits=risk_limits, portfolio_state=portfolio_state)

            # Initialize strategy
            self.strategy = ScalpingStrategy(
                risk_manager=self.risk_manager,
                config=self.config.strategy_config
            )

            # Start cache manager
            await self.cache_manager.start(symbols=self.config.symbols)

            logger.info("✅ All backtesting components initialized successfully")

        except Exception as e:
            logger.error(f"❌ Component initialization failed: {e}")
            raise

    async def load_historical_data(self):
        """Load historical data for backtesting warmup"""
        logger.info(f"📚 Loading {self.config.historical_days} days of historical data...")

        for symbol in self.config.symbols:
            try:
                # In a real implementation, this would fetch historical data
                # For now, we'll simulate with some basic data points
                logger.info(f"📊 Loading historical data for {symbol}")

                # Simulate historical data loading
                await asyncio.sleep(0.1)  # Simulate API calls

                # Add some mock historical data for strategy warmup
                base_price = 35000.0 if symbol.startswith("BTC") else 2000.0
                for i in range(100):  # Add 100 historical points
                    timestamp = time.time() - (100 - i) * 60  # 100 minutes ago to now
                    price = base_price + (i - 50) * (base_price * 0.001)  # Small price movements
                    volume = 1000000.0 + i * 10000

                    # Add to strategy price history
                    self.strategy.add_price_data(symbol, price, volume, timestamp)
                    self.price_history[symbol].append((timestamp, price, volume))

                logger.info(f"✅ Loaded historical data for {symbol}")

            except Exception as e:
                logger.error(f"❌ Failed to load historical data for {symbol}: {e}")

        logger.info("🎯 Historical data loading complete")

    def start_backtesting(self):
        """Start the real-time backtesting process"""
        if self.running:
            logger.warning("⚠️ Backtesting already running")
            return

        logger.info("🚀 Starting Real-Time Backtesting Engine...")
        logger.info("🎛️ Press Ctrl+C to stop backtesting manually")
        logger.info("=" * 80)

        self.running = True
        self.setup_signal_handler()

        # Start backtesting in a separate thread to allow async operations
        self.update_thread = Thread(target=self._run_backtesting_sync)
        self.update_thread.daemon = True
        self.update_thread.start()

        # Wait for stop signal
        try:
            while not self.stop_event.is_set():
                time.sleep(0.1)
        except KeyboardInterrupt:
            self.stop_event.set()

        # Wait for thread to finish
        if self.update_thread and self.update_thread.is_alive():
            self.update_thread.join(timeout=5.0)

        # Print final results
        self.print_final_results()

    def _run_backtesting_sync(self):
        """Run backtesting loop in synchronous context"""
        try:
            # Create new event loop for this thread
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

            # Run the async backtesting loop
            loop.run_until_complete(self._run_backtesting_async())

        except Exception as e:
            logger.error(f"❌ Backtesting thread error: {e}")
        finally:
            try:
                loop.close()
            except:
                pass

    async def _run_backtesting_async(self):
        """Main backtesting loop"""
        logger.info("🔄 Entering real-time backtesting loop...")

        update_count = 0
        last_log_time = time.time()

        try:
            while not self.stop_event.is_set():
                update_start_time = time.time()

                try:
                    # Fetch real-time data for all symbols
                    await self._fetch_realtime_data()

                    # Process trading logic
                    await self._process_trading_logic()

                    # Update metrics
                    self._update_metrics(update_start_time)

                    update_count += 1

                    # Log progress periodically
                    current_time = time.time()
                    if current_time - last_log_time >= 5.0:  # Log every 5 seconds
                        self._log_progress()
                        last_log_time = current_time

                    # Wait for next update
                    await asyncio.sleep(self.config.realtime_interval)

                except Exception as e:
                    logger.error(f"❌ Update cycle error: {e}")
                    self.metrics.data_quality_issues += 1
                    await asyncio.sleep(1.0)  # Brief pause on error

        except Exception as e:
            logger.error(f"❌ Fatal backtesting error: {e}")
        finally:
            logger.info("🛑 Backtesting loop terminated")

    async def _fetch_realtime_data(self):
        """Fetch real-time market data"""
        for symbol in self.config.symbols:
            try:
                # Get market data from aggregator
                market_data = await self.data_aggregator.get_market_data(symbol)

                if market_data:
                    # Add to strategy
                    self.strategy.add_price_data(
                        symbol,
                        market_data.price,
                        market_data.volume_24h,
                        market_data.timestamp
                    )

                    # Track in history
                    self.price_history[symbol].append((
                        market_data.timestamp,
                        market_data.price,
                        market_data.volume_24h
                    ))

                    self.metrics.data_points_received += 1

                    # Publish to event bus
                    event = create_market_data_event(
                        symbol,
                        market_data.price,
                        market_data.volume_24h,
                        "backtest",
                        market_data.bid,
                        market_data.ask,
                        market_data.timestamp
                    )
                    event_bus = get_event_bus()
                    await event_bus.publish(event)

                else:
                    self.metrics.source_failures += 1

            except Exception as e:
                logger.warning(f"⚠️ Failed to fetch data for {symbol}: {e}")
                self.metrics.data_quality_issues += 1

    async def _process_trading_logic(self):
        """Process trading signals and execute trades"""
        for symbol in self.config.symbols:
            try:
                # Generate trading signal
                market_data = await self._get_latest_market_data(symbol)
                if not market_data:
                    continue

                signal = self.strategy.generate_signal(symbol, market_data)

                if signal:
                    # Check risk management
                    if self.risk_manager:
                        risk_assessment = self.risk_manager.evaluate_trade(
                            symbol,
                            signal.entry_price * abs(signal.quantity),
                            1.0
                        )

                        if not risk_assessment.approved:
                            logger.warning(f"🚫 Risk check failed for {symbol}: {risk_assessment.reasoning}")
                            self.metrics.risk_violations += 1
                            continue

                    # Execute trade
                    await self._execute_trade(signal)

            except Exception as e:
                logger.error(f"❌ Trading logic error for {symbol}: {e}")

    async def _get_latest_market_data(self, symbol: str):
        """Get latest market data for a symbol"""
        try:
            # Get from aggregator
            market_data = await self.data_aggregator.get_market_data(symbol)
            if market_data:
                # Convert to MarketData format expected by strategy
                from supreme_system_v5.core import MarketData
                return MarketData(
                    symbol=symbol,
                    price=market_data.price,
                    volume=market_data.volume_24h,
                    timestamp=market_data.timestamp,
                    bid=market_data.bid,
                    ask=market_data.ask
                )
        except Exception as e:
            logger.warning(f"⚠️ Failed to get market data for {symbol}: {e}")
        return None

    async def _execute_trade(self, signal: TradingSignal):
        """Execute a trading signal"""
        try:
            # Calculate actual position size
            max_position_value = abs(signal.quantity) * signal.entry_price
            if self.risk_manager:
                risk_assessment = self.risk_manager.evaluate_trade(
                    signal.symbol, max_position_value, 1.0
                )
                if risk_assessment.adjusted_position_size:
                    max_position_value = risk_assessment.adjusted_position_size

            # Calculate quantity based on position size
            quantity = max_position_value / signal.entry_price

            # Create trade record
            trade = {
                "id": f"trade_{int(time.time() * 1000000)}",
                "symbol": signal.symbol,
                "type": signal.signal_type.value,
                "entry_price": signal.entry_price,
                "quantity": quantity,
                "position_value": max_position_value,
                "stop_loss": signal.stop_loss,
                "take_profit": signal.take_profit,
                "timestamp": signal.timestamp,
                "reasoning": signal.reasoning,
                "status": "executed"
            }

            # Update portfolio
            if signal.signal_type in [SignalType.BUY, SignalType.SELL]:
                # New position
                self.portfolio[signal.symbol] = {
                    "type": signal.signal_type.value,
                    "entry_price": signal.entry_price,
                    "quantity": quantity,
                    "stop_loss": signal.stop_loss,
                    "take_profit": signal.take_profit,
                    "timestamp": signal.timestamp,
                    "unrealized_pnl": 0.0
                }

                # Update balance (simplified - in real trading this would be more complex)
                self.metrics.current_balance -= max_position_value * 0.001  # Simulate fees
                self.metrics.total_trades += 1

                logger.info(f"✅ Executed {signal.signal_type.value} {quantity:.4f} {signal.symbol} @ ${signal.entry_price:.2f}")

            elif signal.signal_type in [SignalType.CLOSE_LONG, SignalType.CLOSE_SHORT]:
                # Close position
                if signal.symbol in self.portfolio:
                    position = self.portfolio[signal.symbol]

                    # Calculate P&L
                    if position["type"] == "BUY":
                        pnl = (signal.entry_price - position["entry_price"]) * position["quantity"]
                    else:
                        pnl = (position["entry_price"] - signal.entry_price) * position["quantity"]

                    # Update metrics
                    self.metrics.current_balance += pnl
                    self.metrics.total_pnl += pnl

                    if pnl > 0:
                        self.metrics.winning_trades += 1
                    else:
                        self.metrics.losing_trades += 1

                    # Remove position
                    del self.portfolio[signal.symbol]

                    logger.info(f"✅ Closed {signal.symbol} position with P&L: ${pnl:.2f}")

            # Update metrics
            self.metrics.update_pnl(self.metrics.current_balance)
            self.metrics.calculate_win_rate()

            # Update exposure
            total_exposure = sum(
                abs(pos["quantity"] * pos["entry_price"])
                for pos in self.portfolio.values()
            )
            self.metrics.current_exposure = total_exposure
            self.metrics.max_exposure = max(self.metrics.max_exposure, total_exposure)

        except Exception as e:
            logger.error(f"❌ Trade execution error: {e}")

    def _update_metrics(self, update_start_time: float):
        """Update performance metrics"""
        update_time = time.time() - update_start_time
        self.metrics.total_updates += 1
        self.metrics.total_processing_time += update_time

        # Update timing metrics
        if self.metrics.total_updates == 1:
            self.metrics.avg_update_time = update_time
        else:
            self.metrics.avg_update_time = (
                (self.metrics.avg_update_time * (self.metrics.total_updates - 1)) + update_time
            ) / self.metrics.total_updates

        self.metrics.max_update_time = max(self.metrics.max_update_time, update_time)
        self.metrics.active_symbols = len(self.config.symbols)

    def _log_progress(self):
        """Log current backtesting progress"""
        runtime = time.time() - self.metrics.start_time
        balance_change = self.metrics.current_balance - self.metrics.initial_balance
        balance_change_pct = (balance_change / self.metrics.initial_balance) * 100

        logger.info(
            f"📊 Progress: {self.metrics.total_updates} updates | "
            f"Runtime: {runtime:.0f}s | "
            f"Balance: ${self.metrics.current_balance:.2f} ({balance_change_pct:+.2f}%) | "
            f"Trades: {self.metrics.total_trades} | "
            f"Exposure: ${self.metrics.current_exposure:.2f} | "
            f"Data Points: {self.metrics.data_points_received}"
        )

        if self.portfolio:
            positions = ", ".join([
                f"{symbol}: {pos['quantity']:.4f}@{pos['entry_price']:.2f}"
                for symbol, pos in self.portfolio.items()
            ])
            logger.info(f"📈 Active Positions: {positions}")

    def print_final_results(self):
        """Print comprehensive backtest results"""
        print("\n" + "=" * 100)
        print("🎯 SUPREME SYSTEM V5 - REAL-TIME BACKTESTING RESULTS")
        print("=" * 100)

        summary = self.metrics.get_summary()

        print(f"⏱️  Session Duration: {summary['session_duration']}")
        print(f"📊 Total Updates: {summary['total_updates']}")
        print(f"⚡ Updates/Second: {summary['updates_per_second']}")
        print(f"🎯 Active Symbols: {summary['active_symbols']}")

        print("\n💰 PORTFOLIO PERFORMANCE:")
        print(f"  Initial Balance: {summary['portfolio']['initial_balance']}")
        print(f"  Final Balance: {summary['portfolio']['current_balance']}")
        print(f"  Total P&L: {summary['portfolio']['total_pnl']} ({summary['portfolio']['total_pnl_percent']})")
        print(f"  Peak Balance: {summary['portfolio']['peak_balance']}")

        print("\n📈 PERFORMANCE METRICS:")
        print(f"  Max Drawdown: {summary['performance']['max_drawdown']} ({summary['performance']['max_drawdown_percent']})")
        print(f"  Sharpe Ratio: {summary['performance']['sharpe_ratio']}")
        print(f"  Win Rate: {summary['performance']['win_rate']}")

        print("\n🛡️  TRADING STATISTICS:")
        print(f"  Total Trades: {summary['trading']['total_trades']}")
        print(f"  Winning Trades: {summary['trading']['winning_trades']}")
        print(f"  Losing Trades: {summary['trading']['losing_trades']}")
        print(f"  Current Exposure: {summary['trading']['current_exposure']}")
        print(f"  Max Exposure: {summary['trading']['max_exposure']}")
        print(f"  Risk Violations: {summary['trading']['risk_violations']}")

        print("\n⚡ SYSTEM PERFORMANCE:")
        print(f"  Avg Update Time: {summary['performance']['avg_update_time']}")
        print(f"  Max Update Time: {summary['performance']['max_update_time']}")
        print(f"  Total Processing Time: {summary['performance']['total_processing_time']}")

        print("\n📊 DATA QUALITY:")
        print(f"  Data Points Received: {summary['data_quality']['data_points_received']}")
        print(f"  Data Quality Issues: {summary['data_quality']['data_quality_issues']}")
        print(f"  Source Failures: {summary['data_quality']['source_failures']}")
        print(f"  Data Quality Score: {summary['data_quality']['data_quality_score']}")

        print(f"\n🕐 Report Generated: {summary['timestamp']}")
        print("=" * 100)

        # Detailed trade log
        if self.completed_trades:
            print("\n📋 DETAILED TRADE LOG:")
            print("-" * 100)
            for trade in self.completed_trades[-10:]:  # Show last 10 trades
                print(f"  {trade['timestamp']:.0f} | {trade['symbol']} | {trade['type']} | "
                      f"Qty: {trade['quantity']:.4f} | Price: ${trade['entry_price']:.2f} | "
                      f"P&L: ${trade.get('pnl', 0):.2f}")

        print("\n🎯 BACKTESTING SESSION COMPLETE")
        print("Press Enter to exit...")
        input()

    async def cleanup(self):
        """Clean up resources"""
        logger.info("🧹 Cleaning up backtesting resources...")

        try:
            if self.cache_manager:
                await self.cache_manager.stop()

            if hasattr(self, 'data_aggregator') and self.data_aggregator:
                # Close any connections if needed
                pass

            logger.info("✅ Cleanup completed")

        except Exception as e:
            logger.error(f"❌ Cleanup error: {e}")


async def run_realtime_backtest(config: Optional[BacktestConfig] = None):
    """
    Run real-time backtesting with manual stop capability

    Args:
        config: Backtest configuration (optional)
    """
    engine = RealTimeBacktestEngine(config)

    try:
        # Initialize components
        await engine.initialize_components()

        # Load historical data for warmup
        await engine.load_historical_data()

        # Start backtesting (this will block until manual stop)
        engine.start_backtesting()

    except KeyboardInterrupt:
        logger.info("🛑 Backtesting interrupted by user")
    except Exception as e:
        logger.error(f"❌ Backtesting failed: {e}")
        raise
    finally:
        await engine.cleanup()


# Legacy compatibility
class BacktestEngine:
    """Legacy backtest engine stub for compatibility"""

    def __init__(self):
        logger.info("Backtest Engine stub initialized (Python side).")

    def run_backtest(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Runs a mock backtest, simulating Rust call."""
        logger.info(f"Running mock backtest with config: {config}")
        # In a real scenario, this would call the Rust backtesting engine
        return {"result": "mock_success", "trades": 10, "pnl": 1000.0}
