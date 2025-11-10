#!/usr/bin/env python3
"""
Debug Continuous Paper Trading

Simulate continuous running like real paper trading to reproduce DataFrame boolean errors.
"""

import sys
import pandas as pd
import numpy as np
import time
from datetime import datetime
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from strategies.moving_average import MovingAverageStrategy
from strategies.mean_reversion import MeanReversionStrategy
from strategies.momentum import MomentumStrategy
from strategies.breakout import BreakoutStrategy
from risk.advanced_risk_manager import AdvancedRiskManager


# Exact copy of SimulatedMarketData from simple_paper_trading.py
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
            change_pct = np.random.normal(0, volatility)
            new_price = self.current_prices[symbol] * (1 + change_pct)

            # Ensure price doesn't go negative
            new_price = max(new_price, self.current_prices[symbol] * 0.5)

            # Generate OHLCV data
            high = new_price * (1 + abs(np.random.normal(0, volatility/2)))
            low = new_price * (1 - abs(np.random.normal(0, volatility/2)))
            open_price = self.current_prices[symbol]
            volume = np.random.uniform(100, 10000)

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

        data = self.price_history[symbol][-limit:]
        return data

    def get_current_price(self, symbol):
        """Get current price for a symbol."""
        return self.current_prices.get(symbol, 100.0)


def continuous_debug_simulation(iterations=50):
    """Simulate continuous paper trading to reproduce DataFrame boolean errors."""
    print("🔄 CONTINUOUS PAPER TRADING DEBUG SIMULATION")
    print("="*70)
    print(f"Running {iterations} iterations to simulate continuous trading...")

    # Initialize components exactly like paper trading
    market_data = SimulatedMarketData()

    # Initialize strategies exactly like paper trading
    strategies = {
        'Moving Average': MovingAverageStrategy(short_window=5, long_window=20),
        'Mean Reversion': MeanReversionStrategy(lookback_period=20, entry_threshold=2.0),
        'Momentum': MomentumStrategy(short_period=12, long_period=26, signal_period=9),
        'Breakout': BreakoutStrategy(lookback_period=20, breakout_threshold=0.02)
    }

    error_count = 0
    total_signals_generated = 0

    # Run continuous simulation
    for iteration in range(iterations):
        if iteration % 10 == 0:
            print(f"\n--- Iteration {iteration}/{iterations} ---")

        try:
            # Generate market updates (like paper trading)
            price_updates = market_data.generate_price_update()

            # Process each symbol (like paper trading loop)
            for symbol in market_data.symbols:
                # Get data exactly like paper trading (line 259)
                data = market_data.get_symbol_data(symbol, 50)

                if len(data) < 20:  # Same check as paper trading
                    continue

                # Convert to DataFrame exactly like paper trading (line 265)
                df = pd.DataFrame(data)

                # Test each strategy exactly like paper trading (lines 267-276)
                for strategy_name, strategy in strategies.items():
                    try:
                        # Call generate_signal exactly like paper trading (line 269)
                        signal = strategy.generate_signal(df)

                        if signal != 0:  # Same check as paper trading (line 271)
                            total_signals_generated += 1

                    except Exception as e:
                        error_count += 1
                        print(f"❌ ITERATION {iteration}: {strategy_name} on {symbol} - {e}")

                        # If we find the DataFrame boolean error, capture details
                        if "truth value of a DataFrame is ambiguous" in str(e):
                            print(f"🎯 FOUND THE BUG! Strategy: {strategy_name}, Symbol: {symbol}")
                            print(f"DataFrame shape: {df.shape}")
                            print(f"DataFrame columns: {list(df.columns)}")

                            # Check for NaN values
                            nan_counts = df.isnull().sum()
                            if nan_counts.sum() > 0:
                                print(f"NaN values found: {nan_counts}")

                            # Try to identify which line caused the error
                            print("Attempting to identify problematic code...")

                            # Return detailed error info
                            return {
                                'error_found': True,
                                'iteration': iteration,
                                'strategy': strategy_name,
                                'symbol': symbol,
                                'error': str(e),
                                'dataframe_info': {
                                    'shape': df.shape,
                                    'columns': list(df.columns),
                                    'dtypes': str(df.dtypes),
                                    'nan_counts': nan_counts.to_dict()
                                }
                            }

        except Exception as e:
            print(f"💥 CRITICAL ERROR in iteration {iteration}: {e}")
            continue

        # Small delay to simulate real timing
        time.sleep(0.01)

    print("\n✅ SIMULATION COMPLETED")
    print(f"Total iterations: {iterations}")
    print(f"Errors encountered: {error_count}")
    print(f"Signals generated: {total_signals_generated}")

    if error_count == 0:
        print("🎉 NO ERRORS FOUND - Issue might be environment-specific")
        return {'error_found': False, 'total_iterations': iterations, 'errors': error_count}
    else:
        print("⚠️  ERRORS FOUND but not the DataFrame boolean error")
        return {'error_found': False, 'total_iterations': iterations, 'errors': error_count}


def main():
    """Main debugging function."""
    print("🔍 SUPREME SYSTEM V5 - CONTINUOUS PAPER TRADING DEBUG")
    print("="*80)

    try:
        result = continuous_debug_simulation(iterations=100)  # Run 100 iterations

        if result.get('error_found'):
            print("\n🎯 SUCCESS: DataFrame boolean error reproduced!")
            print(f"Details: {result}")
            return 1
        else:
            print("\n🤔 Could not reproduce the error with current simulation")
            print("The issue might be:")
            print("1. Threading/race conditions")
            print("2. Memory issues with longer running")
            print("3. Different random seed/data patterns")
            print("4. Environment-specific issues")
            return 0

    except Exception as e:
        print(f"\n💥 CRITICAL ERROR: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())
