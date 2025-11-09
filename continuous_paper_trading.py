#!/usr/bin/env python3
"""
Supreme System V5 - Continuous Paper Trading System

24/7+ Paper Trading with Real Market Data
Real-time strategy execution, performance monitoring, and automated reporting
"""

import time
import logging
import json
import threading
from datetime import datetime, timedelta
from pathlib import Path
import signal
import sys
import os
import pandas as pd
import numpy as np

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

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

        if self.side == 'LONG':
            self.unrealized_pnl = (price - self.entry_price) * self.quantity
        else:  # SHORT
            self.unrealized_pnl = (self.entry_price - price) * self.quantity

    def close_position(self, exit_price: float) -> dict:
        """Close position and return trade details."""
        realized_pnl = self.unrealized_pnl
        fees = abs(realized_pnl) * 0.001  # 0.1% fee
        net_pnl = realized_pnl - fees

        trade = {
            'symbol': self.symbol,
            'side': self.side,
            'quantity': self.quantity,
            'entry_price': self.entry_price,
            'exit_price': exit_price,
            'entry_time': self.entry_time.isoformat(),
            'exit_time': datetime.now().isoformat(),
            'realized_pnl': realized_pnl,
            'fees': fees,
            'net_pnl': net_pnl
        }

        return trade


class ContinuousPaperTrading:
    """
    Continuous Paper Trading System

    Features:
    - 24/7 real market data streaming
    - Multi-strategy execution
    - Live performance monitoring
    - Automated reporting
    - Risk management
    - Web dashboard integration
    """

    def __init__(self, duration_hours: int = 24):
        self.duration_hours = duration_hours
        self.start_time = datetime.now()
        self.end_time = self.start_time + timedelta(hours=duration_hours)

        # Initialize components
        self.ws_client = None
        self.data_pipeline = DataPipeline()
        self.risk_manager = AdvancedRiskManager(initial_capital=10000)

        # Trading state
        self.is_running = False
        self.capital = 10000.0
        self.positions = {}  # symbol -> PaperTradingPosition
        self.completed_trades = []
        self.strategy_signals = {}

        # Performance tracking
        self.portfolio_values = []
        self.daily_returns = []
        self.total_trades = 0
        self.winning_trades = 0

        # Strategies
        self.strategies = {
            'Moving Average': MovingAverageStrategy(),
            'Mean Reversion': MeanReversionStrategy(),
            'Momentum': MomentumStrategy(),
            'Breakout': BreakoutStrategy()
        }

        # Market data cache
        self.market_data = {}  # symbol -> recent OHLCV data
        self.current_prices = {}  # symbol -> current price

        # Logging setup
        self._setup_logging()

        # Signal handling for graceful shutdown
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)

    def _setup_logging(self):
        """Setup comprehensive logging."""
        log_dir = Path("logs")
        log_dir.mkdir(exist_ok=True)

        self.logger = logging.getLogger("PaperTrading")
        self.logger.setLevel(logging.INFO)

        # File handler
        fh = logging.FileHandler(log_dir / f"paper_trading_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
        fh.setLevel(logging.INFO)

        # Console handler
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)

        # Formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        fh.setFormatter(formatter)
        ch.setFormatter(formatter)

        self.logger.addHandler(fh)
        self.logger.addHandler(ch)

    def _signal_handler(self, signum, frame):
        """Handle shutdown signals gracefully."""
        self.logger.info(f"Received signal {signum}, initiating graceful shutdown...")
        self.stop()

    def start(self):
        """Start continuous paper trading."""
        self.logger.info("="*60)
        self.logger.info("🚀 SUPREME SYSTEM V5 - CONTINUOUS PAPER TRADING")
        self.logger.info("="*60)
        self.logger.info(f"Duration: {self.duration_hours} hours")
        self.logger.info(f"Start Time: {self.start_time}")
        self.logger.info(f"End Time: {self.end_time}")
        self.logger.info(f"Initial Capital: ${self.capital:,.2f}")
        self.logger.info("="*60)

        self.is_running = True

        try:
            # Initialize market data streaming
            self._initialize_market_data()

            # Start trading loop
            self._trading_loop()

        except Exception as e:
            self.logger.error(f"Critical error in paper trading: {e}")
            import traceback
            traceback.print_exc()
        finally:
            self.stop()

    def stop(self):
        """Stop paper trading and generate final report."""
        if not self.is_running:
            return

        self.logger.info("⏹️ Stopping paper trading...")
        self.is_running = False

        # Close all positions
        self._close_all_positions()

        # Stop WebSocket client
        if self.ws_client:
            self.ws_client.stop()

        # Generate final report
        self._generate_final_report()

        self.logger.info("✅ Paper trading stopped successfully")

    def _initialize_market_data(self):
        """Initialize real-time market data streaming."""
        self.logger.info("📡 Initializing market data streams...")

        self.ws_client = BinanceWebSocketClient()

        # Subscribe to major cryptocurrencies
        symbols = ['BTCUSDT', 'ETHUSDT', 'ADAUSDT', 'DOTUSDT', 'LINKUSDT']

        for symbol in symbols:
            # Price ticker for real-time prices
            self.ws_client.subscribe_price_stream(symbol, self._price_callback)

            # K-line data for strategy signals
            self.ws_client.subscribe_kline_stream(symbol, '1m', self._kline_callback)

            # Initialize market data cache
            self.market_data[symbol] = []
            self.current_prices[symbol] = 50000.0  # Default price

        # Start WebSocket client
        self.ws_client.start()

        # Wait for initial data
        self.logger.info("⏳ Waiting for market data initialization...")
        time.sleep(10)

    def _price_callback(self, data: dict):
        """Handle real-time price updates."""
        if 's' in data and 'c' in data:
            symbol = data['s']
            price = float(data['c'])

            self.current_prices[symbol] = price

            # Update positions
            if symbol in self.positions:
                self.positions[symbol].update_price(price)

    def _kline_callback(self, data: dict):
        """Handle k-line (candlestick) data."""
        if 'k' in data:
            kline = data['k']
            symbol = data['s']

            # Convert k-line to OHLCV format
            ohlcv = {
                'timestamp': pd.Timestamp.fromtimestamp(kline['t'] / 1000),
                'open': float(kline['o']),
                'high': float(kline['h']),
                'low': float(kline['l']),
                'close': float(kline['c']),
                'volume': float(kline['v'])
            }

            # Maintain rolling window of data (last 100 candles)
            if symbol not in self.market_data:
                self.market_data[symbol] = []

            self.market_data[symbol].append(ohlcv)
            if len(self.market_data[symbol]) > 100:
                self.market_data[symbol] = self.market_data[symbol][-100:]

    def _trading_loop(self):
        """Main trading loop."""
        self.logger.info("🎯 Starting trading loop...")

        while self.is_running and datetime.now() < self.end_time:
            try:
                # Update portfolio value
                self._update_portfolio_value()

                # Generate and execute signals
                self._generate_signals()

                # Check risk limits
                self._check_risk_limits()

                # Log status every minute
                if int(time.time()) % 60 == 0:
                    self._log_status()

                # Small delay to prevent excessive CPU usage
                time.sleep(5)

            except Exception as e:
                self.logger.error(f"Error in trading loop: {e}")
                time.sleep(10)

        self.logger.info("🏁 Trading duration completed")

    def _generate_signals(self):
        """Generate trading signals from all strategies."""
        for symbol in self.market_data.keys():
            if symbol not in self.market_data or not self.market_data[symbol]:
                continue

            # Convert to DataFrame for strategy processing
            df = pd.DataFrame(self.market_data[symbol][-50:])  # Last 50 candles

            if len(df) < 20:  # Need minimum data
                continue

            for strategy_name, strategy in self.strategies.items():
                try:
                    signal = strategy.generate_signal(df)

                    if signal != 0:  # Non-zero signal
                        self._execute_signal(strategy_name, symbol, signal)

                except Exception as e:
                    self.logger.warning(f"Error generating signal for {strategy_name} on {symbol}: {e}")

    def _execute_signal(self, strategy_name: str, symbol: str, signal: int):
        """Execute trading signal."""
        current_price = self.current_prices.get(symbol, 0)
        if current_price <= 0:
            return

        # Risk assessment
        risk_assessment = self.risk_manager.assess_trade_risk(
            symbol=symbol,
            signal=signal,
            price=current_price,
            confidence=0.8,  # Medium confidence for paper trading
            market_data=pd.DataFrame(self.market_data[symbol][-20:])
        )

        if not risk_assessment['approved']:
            self.logger.debug(f"Signal rejected by risk management: {strategy_name} {symbol}")
            return

        # Check if we already have a position in this symbol
        if symbol in self.positions:
            existing_position = self.positions[symbol]

            # If signal is opposite to current position, close it first
            if (signal == 1 and existing_position.side == 'SHORT') or \
               (signal == -1 and existing_position.side == 'LONG'):
                self._close_position(symbol, "SIGNAL_REVERSE")
            else:
                # Same direction, skip (avoid over-trading)
                return

        # Calculate position size
        position_size = risk_assessment['recommended_size']

        # Create position
        side = 'LONG' if signal == 1 else 'SHORT'
        position = PaperTradingPosition(symbol, side, position_size, current_price)

        self.positions[symbol] = position
        self.total_trades += 1

        self.logger.info(f"📈 {strategy_name}: {side} {position_size:.4f} {symbol} at ${current_price:.2f}")

    def _close_position(self, symbol: str, reason: str):
        """Close position."""
        if symbol not in self.positions:
            return

        position = self.positions[symbol]
        exit_price = self.current_prices.get(symbol, position.current_price)

        # Execute close
        trade = position.close_position(exit_price)
        trade['reason'] = reason

        # Update capital
        self.capital += trade['net_pnl']

        # Track performance
        self.completed_trades.append(trade)

        if trade['net_pnl'] > 0:
            self.winning_trades += 1

        # Remove position
        del self.positions[symbol]

        self.logger.info(f"🔒 Closed {symbol} position: ${trade['net_pnl']:+.2f} P&L ({reason})")

    def _close_all_positions(self):
        """Close all open positions."""
        symbols_to_close = list(self.positions.keys())
        for symbol in symbols_to_close:
            self._close_position(symbol, "END_OF_PERIOD")

    def _update_portfolio_value(self):
        """Update portfolio value for tracking."""
        positions_value = sum(pos.unrealized_pnl for pos in self.positions.values())
        total_value = self.capital + positions_value

        self.portfolio_values.append({
            'timestamp': datetime.now(),
            'total_value': total_value,
            'cash': self.capital,
            'positions_value': positions_value,
            'num_positions': len(self.positions)
        })

        # Keep only last 1000 values to prevent memory issues
        if len(self.portfolio_values) > 1000:
            self.portfolio_values = self.portfolio_values[-1000:]

    def _check_risk_limits(self):
        """Check and enforce risk limits."""
        # Maximum drawdown check
        if self.portfolio_values:
            current_value = self.portfolio_values[-1]['total_value']
            peak_value = max(pv['total_value'] for pv in self.portfolio_values)

            if peak_value > 0:
                drawdown = (current_value - peak_value) / peak_value

                if drawdown < -0.10:  # 10% drawdown
                    self.logger.warning(f"⚠️ Portfolio drawdown: {drawdown:.1%}")
                    if drawdown < -0.15:  # 15% drawdown - close positions
                        self.logger.warning("🚨 High drawdown - closing all positions")
                        self._close_all_positions()

        # Maximum positions check
        if len(self.positions) > 3:  # Max 3 concurrent positions
            self.logger.warning("⚠️ Too many positions - reducing exposure")
            # Close oldest position
            oldest_symbol = min(self.positions.keys(),
                              key=lambda s: self.positions[s].entry_time)
            self._close_position(oldest_symbol, "POSITION_LIMIT")

    def _log_status(self):
        """Log current trading status."""
        if self.portfolio_values:
            current_value = self.portfolio_values[-1]['total_value']
            pnl = current_value - 10000  # Initial capital
            pnl_pct = pnl / 10000

            win_rate = (self.winning_trades / self.total_trades * 100) if self.total_trades > 0 else 0

            self.logger.info(
                f"📊 Status: ${current_value:,.2f} | "
                f"P&L: ${pnl:+,.2f} ({pnl_pct:+.1%}) | "
                f"Positions: {len(self.positions)} | "
                f"Trades: {self.total_trades} | "
                f"Win Rate: {win_rate:.1f}%"
            )

    def _generate_final_report(self):
        """Generate comprehensive final report."""
        self.logger.info("="*60)
        self.logger.info("📊 FINAL PAPER TRADING REPORT")
        self.logger.info("="*60)

        # Basic statistics
        end_time = datetime.now()
        duration = end_time - self.start_time

        final_value = self.portfolio_values[-1]['total_value'] if self.portfolio_values else self.capital
        total_pnl = final_value - 10000
        total_return = total_pnl / 10000

        win_rate = (self.winning_trades / self.total_trades * 100) if self.total_trades > 0 else 0

        # Performance metrics
        if len(self.portfolio_values) > 1:
            daily_values = [pv['total_value'] for pv in self.portfolio_values]
            daily_returns = [(daily_values[i] - daily_values[i-1]) / daily_values[i-1]
                           for i in range(1, len(daily_values))]

            if daily_returns:
                volatility = np.std(daily_returns) * np.sqrt(252)  # Annualized
                sharpe_ratio = np.mean(daily_returns) / np.std(daily_returns) * np.sqrt(252) if np.std(daily_returns) > 0 else 0

                # Maximum drawdown
                peak = np.maximum.accumulate(daily_values)
                drawdowns = (np.array(daily_values) - peak) / peak
                max_drawdown = np.min(drawdowns)
            else:
                volatility = 0
                sharpe_ratio = 0
                max_drawdown = 0
        else:
            volatility = 0
            sharpe_ratio = 0
            max_drawdown = 0

        # Report
        self.logger.info(f"Duration: {duration}")
        self.logger.info(f"Start: {self.start_time}")
        self.logger.info(f"End: {end_time}")
        self.logger.info("")
        self.logger.info("PERFORMANCE:")
        self.logger.info(f"Initial Capital: $10,000.00")
        self.logger.info(f"Final Value: ${final_value:,.2f}")
        self.logger.info(f"Total P&L: ${total_pnl:+,.2f}")
        self.logger.info(f"Total Return: {total_return:+.1%}")
        self.logger.info("")
        self.logger.info("TRADING STATISTICS:")
        self.logger.info(f"Total Trades: {self.total_trades}")
        self.logger.info(f"Winning Trades: {self.winning_trades}")
        self.logger.info(f"Win Rate: {win_rate:.1f}%")
        self.logger.info("")
        self.logger.info("RISK METRICS:")
        self.logger.info(f"Volatility: {volatility:.1%}")
        self.logger.info(f"Sharpe Ratio: {sharpe_ratio:.2f}")
        self.logger.info(f"Max Drawdown: {max_drawdown:.1%}")
        self.logger.info("")
        self.logger.info("STRATEGY PERFORMANCE:")

        # Strategy-specific performance
        strategy_performance = {}
        for trade in self.completed_trades:
            strategy = trade.get('strategy', 'Unknown')
            if strategy not in strategy_performance:
                strategy_performance[strategy] = []
            strategy_performance[strategy].append(trade['net_pnl'])

        for strategy, pnl_list in strategy_performance.items():
            if pnl_list:
                total_pnl = sum(pnl_list)
                num_trades = len(pnl_list)
                avg_pnl = total_pnl / num_trades
                win_rate = len([p for p in pnl_list if p > 0]) / num_trades * 100

                self.logger.info(f"  {strategy}:")
                self.logger.info(f"    Trades: {num_trades}")
                self.logger.info(f"    Total P&L: ${total_pnl:+,.2f}")
                self.logger.info(f"    Avg P&L: ${avg_pnl:+,.2f}")
                self.logger.info(f"    Win Rate: {win_rate:.1f}%")
        # Save detailed report
        report = {
            'metadata': {
                'start_time': self.start_time.isoformat(),
                'end_time': end_time.isoformat(),
                'duration_hours': duration.total_seconds() / 3600,
                'symbols_traded': list(set(t['symbol'] for t in self.completed_trades))
            },
            'performance': {
                'initial_capital': 10000,
                'final_value': final_value,
                'total_pnl': total_pnl,
                'total_return': total_return,
                'volatility': volatility,
                'sharpe_ratio': sharpe_ratio,
                'max_drawdown': max_drawdown
            },
            'trading_stats': {
                'total_trades': self.total_trades,
                'winning_trades': self.winning_trades,
                'win_rate': win_rate,
                'avg_trade_pnl': total_pnl / self.total_trades if self.total_trades > 0 else 0
            },
            'trades': self.completed_trades,
            'portfolio_history': self.portfolio_values[-100:]  # Last 100 data points
        }

        # Save to file
        reports_dir = Path("reports")
        reports_dir.mkdir(exist_ok=True)

        report_file = reports_dir / f"paper_trading_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2, default=str)

        self.logger.info("")
        self.logger.info(f"💾 Detailed report saved: {report_file}")
        self.logger.info("="*60)

        # Final recommendation
        if total_return > 0:
            self.logger.info("✅ PAPER TRADING SUCCESSFUL - Ready for live trading!")
        else:
            self.logger.info("⚠️ Paper trading showed losses - Review strategy parameters")

        self.logger.info("="*60)


def main():
    """Main function to run continuous paper trading."""
    import argparse

    parser = argparse.ArgumentParser(description='Supreme System V5 - Continuous Paper Trading')
    parser.add_argument('--duration', type=int, default=24,
                       help='Trading duration in hours (default: 24)')
    parser.add_argument('--capital', type=float, default=10000,
                       help='Initial capital (default: 10000)')

    args = parser.parse_args()

    print(f"🚀 Starting Continuous Paper Trading for {args.duration} hours")
    print(f"💰 Initial Capital: ${args.capital:,.2f}")
    print("Press Ctrl+C to stop gracefully")
    print("="*60)

    # Initialize and start trading
    trader = ContinuousPaperTrading(duration_hours=args.duration)
    trader.capital = args.capital

    try:
        trader.start()
    except KeyboardInterrupt:
        print("\n⏹️ Received interrupt signal...")
        trader.stop()
    except Exception as e:
        print(f"\n❌ Critical error: {e}")
        trader.stop()
        raise


if __name__ == "__main__":
    main()
