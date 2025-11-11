#!/usr/bin/env python3
"""
Supreme System V5 - Unified Paper Trading Engine

Consolidated paper trading system supporting both real-time WebSocket data
and simulated market data for comprehensive testing and development.
"""

import time
import logging
import json
import threading
import signal
import sys
import os
import random
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any, Union
import pandas as pd
import numpy as np

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from data.realtime_client import BinanceWebSocketClient
from data.data_pipeline import DataPipeline
from strategies.moving_average import MovingAverageStrategy
from strategies.mean_reversion import MeanReversionStrategy
from strategies.momentum import MomentumStrategy
from strategies.breakout import BreakoutStrategy
from risk.advanced_risk_manager import AdvancedRiskManager


class PaperTradingPosition:
    """Paper trading position simulation."""

    def __init__(self, symbol: str, side: str, quantity: float, entry_price: float):
        self.symbol = symbol
        self.side = side  # 'LONG' or 'SHORT'
        self.quantity = quantity
        self.entry_price = entry_price
        self.current_price = entry_price
        self.unrealized_pnl = 0.0
        self.entry_time = datetime.now()
        self.last_update = datetime.now()

    def update_price(self, price: float):
        """Update position with new price."""
        self.current_price = price
        self.last_update = datetime.now()

        # Calculate unrealized P&L
        if self.side == 'LONG':
            self.unrealized_pnl = (self.current_price - self.entry_price) * self.quantity
        else:  # SHORT
            self.unrealized_pnl = (self.entry_price - self.current_price) * self.quantity

    def get_duration(self) -> timedelta:
        """Get position duration."""
        return datetime.now() - self.entry_time

    def to_dict(self) -> Dict[str, Any]:
        """Convert position to dictionary."""
        return {
            'symbol': self.symbol,
            'side': self.side,
            'quantity': self.quantity,
            'entry_price': self.entry_price,
            'current_price': self.current_price,
            'unrealized_pnl': self.unrealized_pnl,
            'entry_time': self.entry_time.isoformat(),
            'duration': str(self.get_duration())
        }


class SimulatedMarketData:
    """Simulate real market data với realistic price movements."""

    def __init__(self, symbols: Optional[List[str]] = None):
        self.symbols = symbols or ['BTCUSDT', 'ETHUSDT', 'ADAUSDT', 'DOTUSDT', 'LINKUSDT']
        self.current_prices: Dict[str, float] = {}
        self.price_history: Dict[str, List[Dict[str, Any]]] = {}
        self.start_time = datetime.now()

        # Initialize prices
        base_prices = {
            'BTCUSDT': 95000,
            'ETHUSDT': 3200,
            'ADAUSDT': 0.85,
            'DOTUSDT': 8.50,
            'LINKUSDT': 18.50
        }

        for symbol in self.symbols:
            self.current_prices[symbol] = base_prices.get(symbol, 100.0)
            self.price_history[symbol] = []

    def generate_price_update(self, symbol: str) -> Dict[str, Any]:
        """Generate realistic price update for a symbol."""
        current_price = self.current_prices[symbol]

        # Simulate realistic price movements (volatility between 0.1% - 2% per update)
        volatility = random.uniform(0.001, 0.02)
        trend = random.choice([-1, 1]) * random.uniform(0.0005, 0.005)

        # Add some random walk component
        change = current_price * (trend + np.random.normal(0, volatility))

        # Ensure price doesn't go negative
        new_price = max(current_price + change, current_price * 0.1)

        self.current_prices[symbol] = new_price

        # Generate OHLCV data
        timestamp = datetime.now()
        ohlcv = {
            'timestamp': timestamp,
            'symbol': symbol,
            'open': current_price,
            'high': max(current_price, new_price) * random.uniform(1.001, 1.01),
            'low': min(current_price, new_price) * random.uniform(0.99, 0.999),
            'close': new_price,
            'volume': random.uniform(1000, 10000)
        }

        # Store in history
        self.price_history[symbol].append(ohlcv)

        # Keep only last 1000 entries per symbol
        if len(self.price_history[symbol]) > 1000:
            self.price_history[symbol] = self.price_history[symbol][-1000:]

        return ohlcv

    def get_current_price(self, symbol: str) -> float:
        """Get current price for a symbol."""
        return self.current_prices.get(symbol, 100.0)

    def get_price_history(self, symbol: str, limit: int = 100) -> List[Dict[str, Any]]:
        """Get price history for a symbol."""
        return self.price_history.get(symbol, [])[-limit:]


class PaperTradingEngine:
    """
    Unified Paper Trading Engine supporting both real-time and simulated modes.

    Features:
    - Real-time WebSocket trading with Binance data
    - Simulated trading with realistic market data
    - Multiple strategy support
    - Advanced risk management
    - Comprehensive performance tracking
    - Automated reporting
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the paper trading engine.

        Args:
            config: Configuration dictionary
        """
        self.config = config or self._get_default_config()
        self.logger = logging.getLogger(__name__)

        # Trading mode
        self.mode = self.config.get('mode', 'simulated')  # 'realtime' or 'simulated'

        # Initialize components based on mode
        self._initialize_components()

        # Trading state
        self.is_running = False
        self.positions: Dict[str, PaperTradingPosition] = {}
        self.portfolio_value = self.config['initial_capital']
        self.portfolio_history = []

        # Performance tracking
        self.total_trades = 0
        self.winning_trades = 0
        self.total_pnl = 0.0
        self.max_drawdown = 0.0
        self.start_time = None

        # Signal handling for graceful shutdown
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)

    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration."""
        return {
            'mode': 'simulated',  # 'realtime' or 'simulated'
            'initial_capital': 10000.0,
            'symbols': ['ETHUSDT', 'BTCUSDT'],
            'strategies': ['momentum', 'moving_average', 'mean_reversion', 'breakout'],
            'update_interval': 5,  # seconds for simulated mode
            'max_positions': 5,
            'position_size_pct': 0.1,  # 10% of capital per position
            'enable_risk_management': True,
            'enable_reporting': True,
            'report_interval': 300,  # 5 minutes
            'log_level': 'INFO'
        }

    def _initialize_components(self):
        """Initialize trading components based on mode."""
        # Initialize strategies
        self.strategies = self._initialize_strategies()

        # Initialize risk manager
        self.risk_manager = AdvancedRiskManager(
            initial_capital=self.config['initial_capital']
        )

        if self.mode == 'realtime':
            # Real-time components
            self.websocket_client = BinanceWebSocketClient()
            self.data_pipeline = DataPipeline()
            self.market_data = None
        else:
            # Simulated components
            self.market_data = SimulatedMarketData(self.config['symbols'])
            self.websocket_client = None
            self.data_pipeline = None

        # Initialize data structures
        self.symbol_data: Dict[str, List[Dict[str, Any]]] = {
            symbol: [] for symbol in self.config['symbols']
        }

    def _initialize_strategies(self) -> Dict[str, Any]:
        """Initialize trading strategies."""
        strategies = {}

        strategy_classes = {
            'momentum': MomentumStrategy,
            'moving_average': MovingAverageStrategy,
            'mean_reversion': MeanReversionStrategy,
            'breakout': BreakoutStrategy
        }

        for strategy_name in self.config['strategies']:
            if strategy_name in strategy_classes:
                strategies[strategy_name] = strategy_classes[strategy_name]()

        return strategies

    def start(self):
        """Start the paper trading engine."""
        self.logger.info(f"Starting Paper Trading Engine in {self.mode} mode")
        self.is_running = True
        self.start_time = datetime.now()

        if self.mode == 'realtime':
            self._start_realtime_trading()
        else:
            self._start_simulated_trading()

    def stop(self):
        """Stop the paper trading engine."""
        self.logger.info("Stopping Paper Trading Engine")
        self.is_running = False

        # Close all positions
        self._close_all_positions()

        # Generate final report
        if self.config['enable_reporting']:
            self._generate_report()

    def _start_realtime_trading(self):
        """Start real-time trading with WebSocket data."""
        # Implementation for real-time mode
        # This would use the BinanceWebSocketClient and DataPipeline
        # For now, we'll implement the structure

        def on_price_update(data):
            """Handle real-time price updates."""
            self._process_market_data(data)

        # Connect to WebSocket and start processing
        # This is a placeholder for the real implementation
        self.logger.info("Real-time trading mode not fully implemented yet")

    def _start_simulated_trading(self):
        """Start simulated trading with generated data."""
        self.logger.info("Starting simulated trading")

        def trading_loop():
            """Main trading loop for simulated mode."""
            while self.is_running:
                try:
                    # Generate market data updates
                    for symbol in self.config['symbols']:
                        price_data = self.market_data.generate_price_update(symbol)
                        self._process_market_data(price_data)

                    # Update positions
                    self._update_positions()

                    # Run strategies
                    self._run_strategies()

                    # Periodic reporting
                    self._periodic_reporting()

                    # Wait for next update
                    time.sleep(self.config['update_interval'])

                except Exception as e:
                    self.logger.error(f"Error in trading loop: {e}")
                    time.sleep(1)

        # Start trading loop in background thread
        trading_thread = threading.Thread(target=trading_loop, daemon=True)
        trading_thread.start()

        # Keep main thread alive
        try:
            while self.is_running:
                time.sleep(1)
        except KeyboardInterrupt:
            self.stop()

    def _process_market_data(self, data: Dict[str, Any]):
        """Process incoming market data."""
        symbol = data['symbol']

        # Store data
        self.symbol_data[symbol].append(data)

        # Keep only recent data (last 1000 points)
        if len(self.symbol_data[symbol]) > 1000:
            self.symbol_data[symbol] = self.symbol_data[symbol][-1000:]

    def _run_strategies(self):
        """Run all enabled strategies."""
        for symbol in self.config['symbols']:
            if symbol in self.symbol_data and len(self.symbol_data[symbol]) >= 50:
                # Convert to DataFrame for strategy processing
                df = pd.DataFrame(self.symbol_data[symbol][-100:])
                df['timestamp'] = pd.to_datetime(df['timestamp'])

                for strategy_name, strategy in self.strategies.items():
                    try:
                        signal = strategy.generate_signal(df)

                        if signal != 0:  # Non-neutral signal
                            self._process_signal(symbol, signal, strategy_name, df)

                    except Exception as e:
                        self.logger.error(f"Error running {strategy_name} on {symbol}: {e}")

    def _process_signal(self, symbol: str, signal: int, strategy_name: str, data: pd.DataFrame):
        """Process trading signal."""
        current_price = data['close'].iloc[-1]

        # Check if we already have a position for this symbol
        if symbol in self.positions:
            # Check if signal contradicts current position
            current_position = self.positions[symbol]
            if (signal == 1 and current_position.side == 'SHORT') or \
               (signal == -1 and current_position.side == 'LONG'):
                # Close existing position
                self._close_position(symbol)
            else:
                # Same direction signal - skip or scale
                return

        # Check position limits
        if len(self.positions) >= self.config['max_positions']:
            return

        # Calculate position size
        position_size = self._calculate_position_size(current_price)

        # Apply risk management
        if self.config['enable_risk_management']:
            risk_assessment = self.risk_manager.assess_trade_risk(
                symbol=symbol,
                signal=signal,
                price=current_price,
                confidence=0.8,
                market_data=data
            )

            if not risk_assessment['approved']:
                self.logger.info(f"Trade rejected by risk management: {symbol}")
                return

            position_size = min(position_size, risk_assessment['recommended_size'])

        # Open new position
        side = 'LONG' if signal == 1 else 'SHORT'
        position = PaperTradingPosition(symbol, side, position_size, current_price)
        self.positions[symbol] = position

        self.total_trades += 1
        self.logger.info(f"Opened {side} position for {symbol} at {current_price:.2f}")

    def _calculate_position_size(self, price: float) -> float:
        """Calculate position size based on configuration."""
        capital_per_position = self.portfolio_value * self.config['position_size_pct']
        return capital_per_position / price

    def _update_positions(self):
        """Update all open positions with latest prices."""
        for symbol, position in self.positions.items():
            if self.mode == 'simulated':
                current_price = self.market_data.get_current_price(symbol)
            else:
                # In realtime mode, get from latest data
                if symbol in self.symbol_data and self.symbol_data[symbol]:
                    current_price = self.symbol_data[symbol][-1]['close']
                else:
                    continue

            position.update_price(current_price)

        # Update portfolio value
        self._update_portfolio_value()

    def _update_portfolio_value(self):
        """Update total portfolio value."""
        cash_value = self.portfolio_value

        # Add unrealized P&L from positions
        total_unrealized_pnl = sum(pos.unrealized_pnl for pos in self.positions.values())

        current_portfolio_value = cash_value + total_unrealized_pnl

        # Record in history
        self.portfolio_history.append({
            'timestamp': datetime.now(),
            'value': current_portfolio_value,
            'positions': len(self.positions)
        })

    def _close_position(self, symbol: str):
        """Close a position and realize P&L."""
        if symbol not in self.positions:
            return

        position = self.positions[symbol]
        pnl = position.unrealized_pnl

        # Update portfolio
        self.portfolio_value += pnl
        self.total_pnl += pnl

        if pnl > 0:
            self.winning_trades += 1

        # Remove position
        del self.positions[symbol]

        self.logger.info(".2f"
                        ".2f")

    def _close_all_positions(self):
        """Close all open positions."""
        symbols_to_close = list(self.positions.keys())
        for symbol in symbols_to_close:
            self._close_position(symbol)

    def _periodic_reporting(self):
        """Generate periodic performance reports."""
        # Report every 5 minutes (configurable)
        if not hasattr(self, '_last_report_time'):
            self._last_report_time = datetime.now()

        if (datetime.now() - self._last_report_time).seconds >= self.config['report_interval']:
            self._generate_report()
            self._last_report_time = datetime.now()

    def _generate_report(self):
        """Generate comprehensive performance report."""
        if not self.start_time:
            return

        runtime = datetime.now() - self.start_time
        win_rate = (self.winning_trades / max(self.total_trades, 1)) * 100

        report = {
            'runtime': str(runtime),
            'total_trades': self.total_trades,
            'winning_trades': self.winning_trades,
            'win_rate': ".1f",
            'total_pnl': ".2f",
            'portfolio_value': ".2f",
            'max_drawdown': ".2f",
            'open_positions': len(self.positions),
            'mode': self.mode,
            'timestamp': datetime.now().isoformat()
        }

        # Save report
        report_dir = Path('reports')
        report_dir.mkdir(exist_ok=True)

        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        report_file = report_dir / f'paper_trading_report_{timestamp}.json'

        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2, default=str)

        self.logger.info(f"Performance report saved to {report_file}")

    def _signal_handler(self, signum, frame):
        """Handle shutdown signals."""
        self.logger.info("Received shutdown signal")
        self.stop()

    def get_status(self) -> Dict[str, Any]:
        """Get current engine status."""
        return {
            'is_running': self.is_running,
            'mode': self.mode,
            'portfolio_value': ".2f",
            'open_positions': len(self.positions),
            'total_trades': self.total_trades,
            'win_rate': ".1f",
            'runtime': str(datetime.now() - (self.start_time or datetime.now()))
        }


def main():
    """Main entry point for paper trading."""
    import argparse

    parser = argparse.ArgumentParser(description='Supreme System V5 Paper Trading')
    parser.add_argument('--mode', choices=['realtime', 'simulated'],
                       default='simulated', help='Trading mode')
    parser.add_argument('--capital', type=float, default=10000.0,
                       help='Initial capital')
    parser.add_argument('--symbols', nargs='+', default=['ETHUSDT'],
                       help='Symbols to trade')
    parser.add_argument('--duration', type=int, default=3600,
                       help='Duration in seconds (0 for unlimited)')

    args = parser.parse_args()

    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    # Create configuration
    config = {
        'mode': args.mode,
        'initial_capital': args.capital,
        'symbols': args.symbols,
        'duration': args.duration
    }

    # Start trading engine
    engine = PaperTradingEngine(config)

    try:
        engine.start()
    except KeyboardInterrupt:
        pass
    finally:
        engine.stop()


if __name__ == '__main__':
    main()

