#!/usr/bin/env python3
"""
Debug Paper Trading - Exact Replication

Replicate the exact conditions from paper trading to identify DataFrame boolean errors.
"""

import sys
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
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


def test_paper_trading_exact():
    """Test the exact paper trading flow that causes errors."""
    print("🔧 TESTING EXACT PAPER TRADING FLOW")
    print("="*60)

    # Initialize components exactly like paper trading
    market_data = SimulatedMarketData()

    # Generate price history (simulate running for a while)
    print("Generating price history (like paper trading startup)...")
    for i in range(60):  # Generate 60 data points like paper trading
        market_data.generate_price_update()

    print("✓ Price history generated")

    # Initialize strategies exactly like paper trading
    strategies = {
        'Moving Average': MovingAverageStrategy(short_window=5, long_window=20),
        'Mean Reversion': MeanReversionStrategy(lookback_period=20, entry_threshold=2.0),
        'Momentum': MomentumStrategy(short_period=12, long_period=26, signal_period=9),
        'Breakout': BreakoutStrategy(lookback_period=20, breakout_threshold=0.02)
    }

    print("✓ Strategies initialized")

    # Test each symbol exactly like paper trading does
    for symbol in market_data.symbols:
        print(f"\n{'='*40}")
        print(f"Testing symbol: {symbol}")
        print(f"{'='*40}")

        # Get data exactly like paper trading does (line 259)
        data = market_data.get_symbol_data(symbol, 50)

        if len(data) < 20:  # Same check as paper trading (line 261-262)
            print(f"⚠️  Skipping {symbol}: insufficient data ({len(data)} points)")
            continue

        # Convert to DataFrame exactly like paper trading (line 265)
        df = pd.DataFrame(data)

        print(f"✓ Data retrieved: {len(data)} points")
        print(f"✓ DataFrame shape: {df.shape}")
        print(f"✓ Columns: {list(df.columns)}")

        # Check for NaN values (potential cause of DataFrame boolean errors)
        nan_counts = df.isnull().sum()
        if nan_counts.sum() > 0:
            print(f"⚠️  WARNING: NaN values found:\n{nan_counts}")
        else:
            print("✓ No NaN values found")

        # Test each strategy exactly like paper trading (lines 267-276)
        for strategy_name, strategy in strategies.items():
            try:
                print(f"\n--- Testing {strategy_name} ---")

                # Call generate_signal exactly like paper trading (line 269)
                signal = strategy.generate_signal(df)

                if signal != 0:  # Same check as paper trading (line 271)
                    print(f"✅ {strategy_name}: Generated signal {signal}")
                else:
                    print(f"ℹ️  {strategy_name}: No signal (signal = {signal})")

            except Exception as e:
                print(f"❌ {strategy_name}: ERROR - {e}")
                import traceback
                traceback.print_exc()
                return False

    print(f"\n{'='*60}")
    print("✅ PAPER TRADING EXACT REPLICATION - ALL TESTS PASSED")
    return True


def main():
    """Main debugging function."""
    print("🔍 SUPREME SYSTEM V5 - PAPER TRADING EXACT DEBUG")
    print("="*80)

    try:
        success = test_paper_trading_exact()

        if success:
            print("\n🎉 SUCCESS: No DataFrame boolean errors found!")
            print("This suggests the issue might be:")
            print("1. Race conditions during continuous running")
            print("2. Memory issues after long running")
            print("3. Different data patterns over time")
            return 0
        else:
            print("\n❌ FAILURE: DataFrame boolean errors reproduced!")
            return 1

    except Exception as e:
        print(f"\n💥 CRITICAL ERROR: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())
