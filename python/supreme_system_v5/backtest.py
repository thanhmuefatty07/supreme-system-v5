# python/supreme_system_v5/backtest.py
"""
Real-Time Backtesting Engine - ULTRA SFL Implementation (Hardened)
- Continuous run until manual stop (Ctrl+C)
- Graceful shutdown with exhaustive final report
- Resilient data loop with backoff and self-healing
"""

import asyncio
import signal
import sys
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime
from threading import Event, Thread
from typing import Any, Dict, List, Optional

from loguru import logger

from .data_fabric import DataAggregator
from .data_fabric.cache import CacheManager
from .event_bus import get_event_bus, create_market_data_event
from .risk import DynamicRiskManager, PortfolioState
from .strategies import ScalpingStrategy, SignalType, TradingSignal


@dataclass
class BacktestConfig:
    symbols: List[str] = field(default_factory=lambda: ["BTC-USDT", "ETH-USDT"]) 
    data_sources: List[str] = field(default_factory=lambda: ["binance", "coingecko", "okx"]) 
    realtime_interval: float = 1.0
    initial_balance: float = 10_000.0
    max_position_size: float = 0.02
    enable_risk_management: bool = True
    log_every_seconds: float = 5.0


@dataclass
class BacktestMetrics:
    start_time: float = field(default_factory=time.time)
    total_updates: int = 0
    initial_balance: float = 10_000.0
    current_balance: float = 10_000.0
    peak_balance: float = 10_000.0
    total_trades: int = 0
    wins: int = 0
    losses: int = 0
    max_drawdown: float = 0.0
    max_drawdown_pct: float = 0.0
    exposure: float = 0.0
    max_exposure: float = 0.0
    data_points: int = 0
    data_failures: int = 0
    update_time_avg: float = 0.0
    update_time_max: float = 0.0

    def update_pnl(self, bal: float):
        self.current_balance = bal
        if bal > self.peak_balance:
            self.peak_balance = bal
        dd = self.peak_balance - bal
        dd_pct = (dd / self.peak_balance) * 100 if self.peak_balance else 0.0
        if dd > self.max_drawdown:
            self.max_drawdown = dd
            self.max_drawdown_pct = dd_pct

    def summary(self) -> Dict[str, Any]:
        runtime = time.time() - self.start_time
        return {
            "runtime_s": round(runtime, 2),
            "updates": self.total_updates,
            "balance": round(self.current_balance, 2),
            "pnl": round(self.current_balance - self.initial_balance, 2),
            "pnl_pct": round((self.current_balance - self.initial_balance) / max(self.initial_balance, 1) * 100, 2),
            "peak_balance": round(self.peak_balance, 2),
            "max_dd": round(self.max_drawdown, 2),
            "max_dd_pct": round(self.max_drawdown_pct, 2),
            "trades": self.total_trades,
            "wins": self.wins,
            "losses": self.losses,
            "exposure": round(self.exposure, 2),
            "max_exposure": round(self.max_exposure, 2),
            "data_points": self.data_points,
            "data_failures": self.data_failures,
            "avg_update_ms": round(self.update_time_avg * 1000, 2),
            "max_update_ms": round(self.update_time_max * 1000, 2),
            "timestamp": datetime.now().isoformat()
        }


class RealTimeBacktestEngine:
    def __init__(self, config: Optional[BacktestConfig] = None):
        self.config = config or BacktestConfig()
        self.metrics = BacktestMetrics()
        self.metrics.initial_balance = self.config.initial_balance
        self.metrics.current_balance = self.config.initial_balance
        self.metrics.peak_balance = self.config.initial_balance

        self.data_aggregator: Optional[DataAggregator] = None
        self.cache_manager: Optional[CacheManager] = None
        self.risk_manager: Optional[RiskManager] = None
        self.strategy: Optional[ScalpingStrategy] = None

        self.running = False
        self.stop_event = Event()
        self.update_thread: Optional[Thread] = None

        self.portfolio: Dict[str, Dict[str, Any]] = {}
        self.completed_trades: List[Dict[str, Any]] = []
        self.price_history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=2000))

        logger.info("🎯 Real-Time Backtest Engine (Hardened) initialized")

    def setup_signal_handler(self):
        def handler(signum, frame):
            logger.info("🛑 Manual stop signal received. Shutting down gracefully...")
            self.stop_event.set()
        signal.signal(signal.SIGINT, handler)
        signal.signal(signal.SIGTERM, handler)

    async def initialize_components(self):
        from supreme_system_v5.data_fabric.cache import DataCache
        cache = DataCache()
        await cache.connect()
        self.cache_manager = CacheManager(cache)

        self.data_aggregator = DataAggregator(cache_manager=self.cache_manager)
        for src in self.config.data_sources:
            try:
                from supreme_system_v5.data_fabric.aggregator import DataSource
                if src == "binance":
                    from supreme_system_v5.data_fabric.connectors import BinancePublicConnector as C
                elif src == "coingecko":
                    from supreme_system_v5.data_fabric.connectors import CoinGeckoConnector as C
                elif src == "okx":
                    from supreme_system_v5.data_fabric.connectors import OKXPublicConnector as C
                else:
                    logger.warning(f"Unknown data source: {src}")
                    continue
                source = DataSource(name=f"{src}_source", connector=C(), priority=1, weight=1.0)
                self.data_aggregator.add_source(source)
                logger.info(f"✅ Added data source: {src}")
            except Exception as e:
                logger.error(f"❌ Init source {src} failed: {e}")

        if self.config.enable_risk_management:
            limits = RiskLimits(
                max_drawdown_percent=15.0,
                max_daily_loss_usd=self.config.initial_balance * 0.05,
                max_position_size_usd=self.config.initial_balance * self.config.max_position_size,
                max_leverage=2.0,
            )
            self.risk_manager = RiskManager(limits=limits, portfolio_state=PortfolioState(total_value=self.config.initial_balance))

        self.strategy = ScalpingStrategy(risk_manager=self.risk_manager, config={})
        await self.cache_manager.start(symbols=self.config.symbols)
        logger.info("✅ Components initialized")

    async def _fetch_realtime_data(self):
        for symbol in self.config.symbols:
            try:
                md = await self.data_aggregator.get_market_data(symbol)
                if not md:
                    self.metrics.data_failures += 1
                    continue
                self.strategy.add_price_data(symbol, md.price, md.volume_24h, md.timestamp)
                self.price_history[symbol].append((md.timestamp, md.price, md.volume_24h))
                self.metrics.data_points += 1

                event_bus = get_event_bus()
                await event_bus.publish(create_market_data_event(symbol, md.price, md.volume_24h, "backtest", bid=md.bid, ask=md.ask, timestamp=md.timestamp))
            except Exception as e:
                logger.warning(f"⚠️ Data fetch error {symbol}: {e}")
                self.metrics.data_failures += 1

    async def _process_trading(self):
        for symbol in self.config.symbols:
            try:
                md = await self.data_aggregator.get_market_data(symbol)
                if not md:
                    continue
                from supreme_system_v5.core import MarketData
                market_data = MarketData(symbol=symbol, timestamp=md.timestamp, price=md.price, volume=md.volume_24h, bid=md.bid, ask=md.ask)

                signal: Optional[TradingSignal] = self.strategy.generate_signal(symbol, market_data)
                if not signal:
                    continue

                position_value = abs(signal.quantity * signal.entry_price)
                if self.risk_manager:
                    assessment = self.risk_manager.evaluate_trade(symbol, position_value, 1.0)
                    if not assessment.approved:
                        logger.warning(f"🚫 Risk reject {symbol}: {assessment.reasoning}")
                        continue
                    if assessment.adjusted_position_size:
                        position_value = assessment.adjusted_position_size

                qty = max(position_value / max(signal.entry_price, 1e-9), 0.0)

                # open / close
                if signal.signal_type in [SignalType.BUY, SignalType.SELL]:
                    self.portfolio[symbol] = {
                        "type": signal.signal_type.value,
                        "entry_price": signal.entry_price,
                        "quantity": qty,
                        "stop_loss": signal.stop_loss,
                        "take_profit": signal.take_profit,
                        "timestamp": signal.timestamp,
                    }
                    self.metrics.current_balance -= position_value * 0.001  # fee model
                    self.metrics.total_trades += 1
                    logger.info(f"✅ {signal.signal_type.value} {qty:.4f} {symbol} @ {signal.entry_price:.2f}")
                elif signal.signal_type in [SignalType.CLOSE_LONG, SignalType.CLOSE_SHORT]:
                    if symbol in self.portfolio:
                        pos = self.portfolio[symbol]
                        pnl = (signal.entry_price - pos["entry_price"]) * pos["quantity"] if pos["type"] == "BUY" else (pos["entry_price"] - signal.entry_price) * pos["quantity"]
                        self.metrics.current_balance += pnl
                        (self.metrics.wins if pnl > 0 else self.metrics.losses).__iadd__(1)  # increment
                        self.completed_trades.append({"symbol": symbol, "type": pos["type"], "quantity": pos["quantity"], "entry_price": pos["entry_price"], "exit_price": signal.entry_price, "pnl": pnl, "timestamp": time.time()})
                        del self.portfolio[symbol]
                        logger.info(f"✅ Close {symbol} P&L: {pnl:.2f}")

                # update exposure
                exposure = sum(abs(p["quantity"] * p["entry_price"]) for p in self.portfolio.values())
                self.metrics.exposure = exposure
                self.metrics.max_exposure = max(self.metrics.max_exposure, exposure)
                self.metrics.update_pnl(self.metrics.current_balance)
            except Exception as e:
                logger.error(f"❌ Trading error {symbol}: {e}")

    def _log_progress(self):
        s = self.metrics.summary()
        logger.info(
            f"📊 updates={s['updates']} | bal=${s['balance']} ({s['pnl_pct']}%) | trades={s['trades']} | exp=${s['exposure']} | data={s['data_points']} pts"
        )
        if self.portfolio:
            positions = ", ".join(f"{sym}:{p['quantity']:.4f}@{p['entry_price']:.2f}" for sym,p in self.portfolio.items())
            logger.info(f"📈 positions: {positions}")

    async def _loop(self):
        logger.info("🔄 Entering realtime backtest loop (Ctrl+C to stop)")
        last_log = time.time()
        backoff = 1.0
        try:
            while not self.stop_event.is_set():
                start = time.time()
                try:
                    await self._fetch_realtime_data()
                    await self._process_trading()
                    # successful cycle → reset backoff
                    backoff = 1.0
                except Exception as e:
                    logger.error(f"💥 Update cycle error: {e}")
                    backoff = min(backoff * 2, 30.0)
                    await asyncio.sleep(backoff)
                    continue

                # metrics timing
                dt = time.time() - start
                self.metrics.total_updates += 1
                self.metrics.update_time_max = max(self.metrics.update_time_max, dt)
                self.metrics.update_time_avg = ((self.metrics.update_time_avg * (self.metrics.total_updates - 1)) + dt) / self.metrics.total_updates

                # periodic log
                if time.time() - last_log >= self.config.log_every_seconds:
                    self._log_progress()
                    last_log = time.time()

                await asyncio.sleep(self.config.realtime_interval)
        finally:
            logger.info("🛑 Realtime backtest loop terminated")

    def start(self):
        if self.running:
            logger.warning("Already running")
            return
        self.running = True
        self.setup_signal_handler()

        def runner():
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            loop.run_until_complete(self._loop())
            loop.close()

        self.update_thread = Thread(target=runner, daemon=True)
        self.update_thread.start()

        try:
            while not self.stop_event.is_set():
                time.sleep(0.1)
        except KeyboardInterrupt:
            self.stop_event.set()

        if self.update_thread and self.update_thread.is_alive():
            self.update_thread.join(timeout=5)

        self.print_report()

    def print_report(self):
        s = self.metrics.summary()
        print("\n" + "="*96)
        print("🎯 SUPREME SYSTEM V5 - REALTIME BACKTEST REPORT")
        print("="*96)
        print(f"⏱ Runtime: {s['runtime_s']}s  |  Updates: {s['updates']}  |  Avg/Max: {s['avg_update_ms']}ms/{s['max_update_ms']}ms")
        print(f"💰 Balance: ${s['balance']}  |  PnL: ${s['pnl']} ({s['pnl_pct']}%)  |  Peak: ${s['peak_balance']}")
        print(f"📉 Max DD: ${s['max_dd']} ({s['max_dd_pct']}%)  |  Exposure: ${s['exposure']} (max ${s['max_exposure']})")
        print(f"🧪 Data: {s['data_points']} points  |  Failures: {s['data_failures']}")
        if self.completed_trades:
            print("\n📋 LAST 10 TRADES:")
            for t in self.completed_trades[-10:]:
                print(f"  {int(t['timestamp'])} | {t['symbol']} | {t['type']} | qty={t['quantity']:.4f} | in={t['entry_price']:.2f} | out={t.get('exit_price','-')} | pnl={t.get('pnl',0):.2f}")
        print("="*96 + "\n")


async def run_realtime_backtest(config: Optional[BacktestConfig] = None):
    engine = RealTimeBacktestEngine(config)
    try:
        await engine.initialize_components()
        engine.start()  # blocks until manual stop
    finally:
        if engine.cache_manager:
            await engine.cache_manager.stop()
