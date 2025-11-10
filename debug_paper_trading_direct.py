#!/usr/bin/env python3
"""
Debug Paper Trading Direct

Import and run the actual paper trading components to reproduce DataFrame boolean errors.
"""

import sys
import pandas as pd
import numpy as np
import time
from datetime import datetime
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

# Import directly from simple_paper_trading.py
from simple_paper_trading import SimulatedMarketData, SimplePaperTrading


def test_direct_paper_trading():
    """Test by directly using the paper trading classes."""
    print("🔧 TESTING DIRECT PAPER TRADING COMPONENTS")
    print("="*60)

    try:
        # Create market data simulator (from paper trading)
        market_data = SimulatedMarketData()
        print("✓ Market data simulator created")

        # Generate some price history
        print("Generating price history...")
        for i in range(60):
            market_data.generate_price_update()

        print("✓ Price history generated")

        # Create a minimal paper trading instance to test signal generation
        # We'll manually call the _generate_signals method

        # Initialize strategies (exact same as paper trading)
        from strategies.moving_average import MovingAverageStrategy
        from strategies.mean_reversion import MeanReversionStrategy
        from strategies.momentum import MomentumStrategy
        from strategies.breakout import BreakoutStrategy

        strategies = {
            'Moving Average': MovingAverageStrategy(short_window=5, long_window=20),
            'Mean Reversion': MeanReversionStrategy(lookback_period=20, entry_threshold=2.0),
            'Momentum': MomentumStrategy(short_period=12, long_period=26, signal_period=9),
            'Breakout': BreakoutStrategy(lookback_period=20, breakout_threshold=0.02)
        }

        print("✓ Strategies initialized")

        # Test signal generation like paper trading does
        error_count = 0
        total_tests = 0

        # Run multiple iterations
        for iteration in range(20):
            print(f"\n--- Testing Iteration {iteration + 1}/20 ---")

            # Generate new price data
            market_data.generate_price_update()

            # Test each symbol
            for symbol in market_data.symbols:
                data = market_data.get_symbol_data(symbol, 50)

                if len(data) < 20:
                    continue

                df = pd.DataFrame(data)

                # Test each strategy
                for strategy_name, strategy in strategies.items():
                    total_tests += 1
                    try:
                        signal = strategy.generate_signal(df)
                        print(f"✓ {strategy_name} on {symbol}: signal = {signal}")

                    except Exception as e:
                        error_count += 1
                        print(f"❌ {strategy_name} on {symbol}: ERROR - {e}")

                        # Check if this is the DataFrame boolean error
                        if "truth value of a DataFrame is ambiguous" in str(e):
                            print(f"🎯 FOUND THE BUG! Strategy: {strategy_name}, Symbol: {symbol}")
                            print(f"DataFrame info: shape={df.shape}, columns={list(df.columns)}")

                            # Check for NaN values
                            nan_counts = df.isnull().sum()
                            print(f"NaN counts: {nan_counts}")

                            # Return error details
                            return {
                                'error_found': True,
                                'strategy': strategy_name,
                                'symbol': symbol,
                                'error': str(e),
                                'dataframe_shape': df.shape,
                                'dataframe_columns': list(df.columns),
                                'nan_counts': nan_counts.to_dict()
                            }

        print("\n✅ DIRECT TESTING COMPLETED")
        print(f"Total tests: {total_tests}")
        print(f"Errors: {error_count}")

        if error_count == 0:
            print("🎉 NO ERRORS FOUND")
            return {'error_found': False, 'total_tests': total_tests, 'errors': error_count}
        else:
            print("⚠️  ERRORS FOUND but not the DataFrame boolean error")
            return {'error_found': False, 'total_tests': total_tests, 'errors': error_count}

    except Exception as e:
        print(f"💥 CRITICAL ERROR: {e}")
        import traceback
        traceback.print_exc()
        return {'error_found': False, 'critical_error': str(e)}


def main():
    """Main debugging function."""
    print("🔍 SUPREME SYSTEM V5 - DIRECT PAPER TRADING DEBUG")
    print("="*80)

    try:
        result = test_direct_paper_trading()

        if result.get('error_found'):
            print("\n🎯 SUCCESS: DataFrame boolean error reproduced!")
            print(f"Error details: {result}")
            return 1
        else:
            print("\n🤔 Could not reproduce the error")
            print("Possible reasons:")
            print("1. Error only occurs under specific conditions")
            print("2. Error requires longer running time")
            print("3. Error is environment-specific")
            print("4. Error might be fixed or not reproducible in isolation")
            return 0

    except Exception as e:
        print(f"\n💥 CRITICAL ERROR: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())
