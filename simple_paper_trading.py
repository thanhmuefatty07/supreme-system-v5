#!/usr/bin/env python3
"""
Supreme System V5 - Simple Paper Trading

Paper trading với simulated market data để tránh WebSocket connection issues.
Chạy liên tục 24h+ với real market simulation.
"""

import time
import logging
import json
import signal
import sys
from datetime import datetime, timedelta
from pathlib import Path
import random
import pandas as pd
import numpy as np

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from strategies.moving_average import MovingAverageStrategy
from strategies.mean_reversion import MeanReversionStrategy
from strategies.momentum import MomentumStrategy
from strategies.breakout import BreakoutStrategy
from risk.advanced_risk_manager import AdvancedRiskManager


class SimulatedMarketData:
    """Simulate real market data với realistic price movements."""

    def __init__(self, symbols=None):
        self.symbols = symbols or ['BTCUSDT', 'ETHUSDT', 'ADAUSDT', 'DOTUSDT', 'LINKUSDT']
        self.current_prices = {}
        self.price_history = {}

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

    def generate_price_update(self):
        """Generate realistic price updates."""
        updates = {}

        for symbol in self.symbols:
            # Simulate price movement with some volatility
            volatility = {
                'BTCUSDT': 0.005,  # 0.5% volatility
                'ETHUSDT': 0.008,  # 0.8% volatility
                'ADAUSDT': 0.012,  # 1.2% volatility
                'DOTUSDT': 0.010,  # 1.0% volatility
                'LINKUSDT': 0.015  # 1.5% volatility
            }.get(symbol, 0.01)

            # Random walk with mean reversion
            change_pct = random.gauss(0, volatility)
            new_price = self.current_prices[symbol] * (1 + change_pct)

            # Ensure price doesn't go negative
            new_price = max(new_price, self.current_prices[symbol] * 0.5)

            # Generate OHLCV data
            high = new_price * (1 + abs(random.gauss(0, volatility/2)))
            low = new_price * (1 - abs(random.gauss(0, volatility/2)))
            open_price = self.current_prices[symbol]
            volume = random.uniform(100, 10000)

            ohlcv = {
                'timestamp': datetime.now(),
                'open': open_price,
                'high': high,
                'low': low,
                'close': new_price,
                'volume': volume
            }

            self.current_prices[symbol] = new_price

            # Keep price history (last 100 points)
            self.price_history[symbol].append(ohlcv)
            if len(self.price_history[symbol]) > 100:
                self.price_history[symbol] = self.price_history[symbol][-100:]

            updates[symbol] = ohlcv

        return updates

    def get_symbol_data(self, symbol, limit=50):
        """Get recent data for a symbol."""
        if symbol not in self.price_history:
            return []
        return self.price_history[symbol][-limit:]

    def get_current_price(self, symbol):
        """Get current price for a symbol."""
        return self.current_prices.get(symbol, 100.0)


class SimplePaperTrading:
    """
    Simple Paper Trading System

    Uses simulated market data to avoid WebSocket connection issues.
    Runs continuously with real strategy execution and risk management.
    """

    def __init__(self, duration_hours: int = 24, capital: float = 10000):
        self.duration_hours = duration_hours
        self.capital = capital
        self.start_time = datetime.now()
        self.end_time = self.start_time + timedelta(hours=duration_hours)

        # Initialize components
        self.market_data = SimulatedMarketData()
        self.risk_manager = AdvancedRiskManager(capital)

        # Trading state
        self.is_running = False
        self.positions = {}  # symbol -> position dict
        self.completed_trades = []
        self.portfolio_values = []
        self.total_trades = 0
        self.winning_trades = 0

        # Strategies
        self.strategies = {
            'Moving Average': MovingAverageStrategy(),
            'Mean Reversion': MeanReversionStrategy(),
            'Momentum': MomentumStrategy(),
            'Breakout': BreakoutStrategy()
        }

        # Setup logging
        self._setup_logging()

        # Signal handling
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)

    def _setup_logging(self):
        """Setup logging without emojis to avoid encoding issues."""
        log_dir = Path("logs")
        log_dir.mkdir(exist_ok=True)

        self.logger = logging.getLogger("PaperTrading")
        self.logger.setLevel(logging.INFO)

        # File handler
        fh = logging.FileHandler(log_dir / f"simple_paper_trading_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
        fh.setLevel(logging.INFO)

        # Console handler
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)

        # Formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(levelname)s - %(message)s'
        )
        fh.setFormatter(formatter)
        ch.setFormatter(formatter)

        self.logger.addHandler(fh)
        self.logger.addHandler(ch)

    def _signal_handler(self, signum, frame):
        """Handle shutdown signals."""
        self.logger.info(f"Received signal {signum}, initiating graceful shutdown...")
        self.stop()

    def start(self):
        """Start paper trading."""
        self.logger.info("="*60)
        self.logger.info("SUPREME SYSTEM V5 - SIMPLE PAPER TRADING")
        self.logger.info("="*60)
        self.logger.info(f"Duration: {self.duration_hours} hours")
        self.logger.info(f"Start Time: {self.start_time}")
        self.logger.info(f"End Time: {self.end_time}")
        self.logger.info(f"Initial Capital: ${self.capital:,.2f}")
        self.logger.info("Strategies: Moving Average, Mean Reversion, Momentum, Breakout")
        self.logger.info("="*60)

        self.is_running = True

        try:
            self._trading_loop()
        except Exception as e:
            self.logger.error(f"Critical error in paper trading: {e}")
            import traceback
            traceback.print_exc()
        finally:
            self.stop()

    def stop(self):
        """Stop paper trading and generate report."""
        if not self.is_running:
            return

        self.logger.info("Stopping paper trading...")
        self.is_running = False

        # Close all positions
        self._close_all_positions()

        # Generate final report
        self._generate_final_report()

        self.logger.info("Paper trading stopped successfully")

    def _trading_loop(self):
        """Main trading loop."""
        self.logger.info("Starting trading loop...")

        while self.is_running and datetime.now() < self.end_time:
            try:
                # Generate market updates
                price_updates = self.market_data.generate_price_update()

                # Update positions with new prices
                self._update_positions(price_updates)

                # Generate and execute signals
                self._generate_signals()

                # Update portfolio value
                self._update_portfolio_value()

                # Check risk limits
                self._check_risk_limits()

                # Log status every minute
                current_time = datetime.now()
                if current_time.second < 5:  # Log roughly every minute
                    self._log_status()

                # Small delay for realistic timing
                time.sleep(2)

            except Exception as e:
                self.logger.error(f"Error in trading loop: {e}")
                time.sleep(5)

        self.logger.info("Trading duration completed")

    def _generate_signals(self):
        """Generate trading signals from all strategies."""
        for symbol in self.market_data.symbols:
            # Get recent data for strategy
            data = self.market_data.get_symbol_data(symbol, 50)

            if len(data) < 20:  # Need minimum data
                continue

            # Convert to DataFrame
            df = pd.DataFrame(data)

            for strategy_name, strategy in self.strategies.items():
                try:
                    signal = strategy.generate_signal(df)

                    if signal != 0:  # Non-zero signal
                        self._execute_signal(strategy_name, symbol, signal)

                except Exception as e:
                    self.logger.warning(f"Error generating signal for {strategy_name} on {symbol}: {e}")

    def _execute_signal(self, strategy_name: str, symbol: str, signal: int):
        """Execute trading signal."""
        current_price = self.market_data.get_current_price(symbol)

        # Risk assessment
        risk_assessment = self.risk_manager.assess_trade_risk(
            symbol=symbol,
            signal=signal,
            price=current_price,
            confidence=0.8,
            market_data=pd.DataFrame(self.market_data.get_symbol_data(symbol, 30))
        )

        if not risk_assessment['approved']:
            return

        # Check existing position
        if symbol in self.positions:
            existing_position = self.positions[symbol]

            # If signal is opposite to current position, close it first
            if (signal == 1 and existing_position['side'] == 'SHORT') or \
               (signal == -1 and existing_position['side'] == 'LONG'):
                self._close_position(symbol, "SIGNAL_REVERSE")

        # Calculate position size
        position_size = risk_assessment['recommended_size']

        # Create position
        side = 'LONG' if signal == 1 else 'SHORT'
        position = {
            'symbol': symbol,
            'side': side,
            'quantity': position_size,
            'entry_price': current_price,
            'entry_time': datetime.now(),
            'current_price': current_price,
            'unrealized_pnl': 0.0,
            'strategy': strategy_name
        }

        self.positions[symbol] = position
        self.capital -= position_size * current_price  # Deduct capital
        self.total_trades += 1

        self.logger.info(f"TRADE: {strategy_name} {side} {position_size:.4f} {symbol} at ${current_price:.2f}")

    def _update_positions(self, price_updates):
        """Update positions with new prices."""
        for symbol, ohlcv in price_updates.items():
            if symbol in self.positions:
                position = self.positions[symbol]
                position['current_price'] = ohlcv['close']

                # Calculate unrealized P&L
                if position['side'] == 'LONG':
                    position['unrealized_pnl'] = (ohlcv['close'] - position['entry_price']) * position['quantity']
                else:  # SHORT
                    position['unrealized_pnl'] = (position['entry_price'] - ohlcv['close']) * position['quantity']

    def _close_position(self, symbol: str, reason: str):
        """Close position."""
        if symbol not in self.positions:
            return

        position = self.positions[symbol]
        exit_price = position['current_price']

        # Calculate realized P&L
        if position['side'] == 'LONG':
            realized_pnl = (exit_price - position['entry_price']) * position['quantity']
        else:
            realized_pnl = (position['entry_price'] - exit_price) * position['quantity']

        fees = abs(realized_pnl) * 0.001  # 0.1% fee
        net_pnl = realized_pnl - fees

        # Update capital
        self.capital += position['quantity'] * exit_price + net_pnl

        # Track performance
        if net_pnl > 0:
            self.winning_trades += 1

        # Create trade record
        trade = {
            'symbol': symbol,
            'side': position['side'],
            'quantity': position['quantity'],
            'entry_price': position['entry_price'],
            'exit_price': exit_price,
            'entry_time': position['entry_time'].isoformat(),
            'exit_time': datetime.now().isoformat(),
            'realized_pnl': realized_pnl,
            'fees': fees,
            'net_pnl': net_pnl,
            'strategy': position['strategy'],
            'reason': reason
        }

        self.completed_trades.append(trade)

        # Remove position
        del self.positions[symbol]

        self.logger.info(f"CLOSED: {symbol} {position['side']} P&L ${net_pnl:+.2f} ({reason})")

    def _close_all_positions(self):
        """Close all open positions."""
        symbols_to_close = list(self.positions.keys())
        for symbol in symbols_to_close:
            self._close_position(symbol, "END_OF_PERIOD")

    def _update_portfolio_value(self):
        """Update portfolio value."""
        positions_value = sum(pos['unrealized_pnl'] for pos in self.positions.values())
        total_value = self.capital + positions_value

        self.portfolio_values.append({
            'timestamp': datetime.now(),
            'total_value': total_value,
            'cash': self.capital,
            'positions_value': positions_value,
            'num_positions': len(self.positions)
        })

        # Keep only last 1000 values
        if len(self.portfolio_values) > 1000:
            self.portfolio_values = self.portfolio_values[-1000:]

    def _check_risk_limits(self):
        """Check risk limits."""
        if self.portfolio_values:
            current_value = self.portfolio_values[-1]['total_value']
            initial_capital = 10000
            drawdown = (current_value - initial_capital) / initial_capital

            # Close positions if drawdown exceeds 15%
            if drawdown < -0.15 and self.positions:
                self.logger.warning(f"Portfolio drawdown: {drawdown:.1%}")
                self._close_all_positions()

    def _log_status(self):
        """Log current status."""
        if self.portfolio_values:
            current_value = self.portfolio_values[-1]['total_value']
            pnl = current_value - 10000
            pnl_pct = pnl / 10000

            win_rate = (self.winning_trades / self.total_trades * 100) if self.total_trades > 0 else 0

            self.logger.info(
                f"STATUS: ${current_value:,.2f} | "
                f"P&L: ${pnl:+,.2f} ({pnl_pct:+.1%}) | "
                f"Positions: {len(self.positions)} | "
                f"Trades: {self.total_trades} | "
                ".1f"
            )

    def _generate_final_report(self):
        """Generate comprehensive final report."""
        self.logger.info("="*60)
        self.logger.info("FINAL PAPER TRADING REPORT")
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
                if np.std(daily_returns) > 0:
                    sharpe_ratio = np.mean(daily_returns) / np.std(daily_returns) * np.sqrt(252)
                else:
                    sharpe_ratio = 0

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
        self.logger.info(f"Initial Capital: ${10000:,.2f}")
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
        reports_dir = Path("reports")
        reports_dir.mkdir(exist_ok=True)

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
            'trades': self.completed_trades[-50:],  # Last 50 trades
            'portfolio_history': self.portfolio_values[-100:]  # Last 100 data points
        }

        report_file = reports_dir / f"simple_paper_trading_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2, default=str)

        self.logger.info("")
        self.logger.info(f"Report saved: {report_file}")
        self.logger.info("="*60)

        # Final recommendation
        if total_return > 0:
            self.logger.info("SUCCESSFUL: Paper trading showed positive returns!")
        else:
            self.logger.info("NEUTRAL: Paper trading completed - review strategy parameters")


def main():
    """Main function."""
    import argparse

    parser = argparse.ArgumentParser(description='Supreme System V5 - Simple Paper Trading')
    parser.add_argument('--duration', type=int, default=24,
                       help='Trading duration in hours (default: 24)')
    parser.add_argument('--capital', type=float, default=10000,
                       help='Initial capital (default: 10000)')

    args = parser.parse_args()

    print(f"Starting Simple Paper Trading for {args.duration} hours with ${args.capital:,.2f}")
    print("This uses simulated market data to avoid connection issues.")
    print("Press Ctrl+C to stop gracefully")
    print("="*60)

    # Start trading
    trader = SimplePaperTrading(duration_hours=args.duration, capital=args.capital)

    try:
        trader.start()
    except KeyboardInterrupt:
        print("\nInterrupted by user...")
        trader.stop()
    except Exception as e:
        print(f"\nCritical error: {e}")
        trader.stop()
        raise


if __name__ == "__main__":
    main()
