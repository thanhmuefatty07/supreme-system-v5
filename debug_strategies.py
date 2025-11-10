#!/usr/bin/env python3
"""
Debug script to identify DataFrame boolean comparison issues in strategies.
"""

import sys
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Add src to path
sys.path.insert(0, 'src')

from strategies.moving_average import MovingAverageStrategy
from strategies.mean_reversion import MeanReversionStrategy
from strategies.momentum import MomentumStrategy
from strategies.breakout import BreakoutStrategy

def create_test_data():
    """Create test OHLCV data."""
    # Create realistic test data
    dates = pd.date_range(start='2024-01-01', periods=100, freq='5min')

    # Simulate price movement
    np.random.seed(42)
    base_price = 50000
    prices = [base_price]
    for i in range(99):
        change = np.random.normal(0, 0.005)  # 0.5% volatility
        new_price = prices[-1] * (1 + change)
        prices.append(new_price)

    # Create OHLCV data
    data = []
    for i, (date, close) in enumerate(zip(dates, prices)):
        # Generate OHLC from close price with some spread
        spread = abs(np.random.normal(0, close * 0.002))
        high = close + spread
        low = close - spread
        open_price = close - spread/2 + np.random.normal(0, spread/4)
        volume = np.random.uniform(1000, 10000)

        data.append({
            'timestamp': date,
            'open': open_price,
            'high': high,
            'low': low,
            'close': close,
            'volume': volume
        })

    return data

def test_strategy_isolation(strategy_class, strategy_name, test_data):
    """Test individual strategy in isolation."""
    print(f"\n{'='*60}")
    print(f"TESTING {strategy_name.upper()}")
    print('='*60)

    try:
        # Create strategy instance
        if strategy_name == "Moving Average":
            strategy = strategy_class(short_window=5, long_window=20)
        elif strategy_name == "Mean Reversion":
            strategy = strategy_class(lookback_period=20, entry_threshold=2.0)
        elif strategy_name == "Momentum":
            strategy = strategy_class(short_period=12, long_period=26, signal_period=9)
        elif strategy_name == "Breakout":
            strategy = strategy_class(lookback_period=20, breakout_threshold=0.02)
        else:
            strategy = strategy_class()

        print(f"✓ Strategy instance created: {strategy}")

        # Test data validation
        df = pd.DataFrame(test_data)
        print(f"✓ DataFrame created with shape: {df.shape}")
        print(f"✓ Columns: {list(df.columns)}")

        is_valid = strategy.validate_data(df)
        print(f"✓ Data validation: {'PASSED' if is_valid else 'FAILED'}")

        if not is_valid:
            print("❌ Data validation failed - stopping test")
            return False

        # Test signal generation with different data sizes
        for data_size in [10, 30, 50, 100]:
            test_df = df.tail(data_size).copy()
            print(f"\n--- Testing with {data_size} data points ---")

            try:
                signal = strategy.generate_signal(test_df)
                print(f"✓ Signal generated: {signal} (type: {type(signal)})")

                if not isinstance(signal, int) or signal not in [-1, 0, 1]:
                    print(f"⚠️  WARNING: Invalid signal value: {signal}")
                    return False

            except Exception as e:
                print(f"❌ ERROR generating signal: {e}")
                import traceback
                traceback.print_exc()
                return False

        print(f"✅ {strategy_name} - ALL TESTS PASSED")
        return True

    except Exception as e:
        print(f"❌ CRITICAL ERROR in {strategy_name}: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_paper_trading_integration():
    """Test the paper trading integration specifically."""
    print(f"\n{'='*60}")
    print("TESTING PAPER TRADING INTEGRATION")
    print('='*60)

    # Import the paper trading classes
    from simple_paper_trading import SimulatedMarketData

    try:
        # Create market data simulator
        market_data = SimulatedMarketData(['BTCUSDT'])
        print("✓ Market data simulator created")

        # Generate some price updates
        for i in range(5):
            updates = market_data.generate_price_update()
            print(f"✓ Price update {i+1}: {updates}")

        # Test data retrieval
        btc_data = market_data.get_symbol_data('BTCUSDT', 10)
        print(f"✓ Retrieved {len(btc_data)} data points for BTCUSDT")

        # Convert to DataFrame as done in paper trading
        df = pd.DataFrame(btc_data)
        print(f"✓ DataFrame conversion: {df.shape}")
        print(f"✓ DataFrame columns: {list(df.columns)}")

        # Test with strategies
        strategies = [
            (MovingAverageStrategy(short_window=5, long_window=20), "Moving Average"),
            (MeanReversionStrategy(lookback_period=20), "Mean Reversion"),
            (MomentumStrategy(), "Momentum"),
            (BreakoutStrategy(), "Breakout")
        ]

        for strategy, name in strategies:
            try:
                if df.shape[0] >= 20:  # Minimum data requirement
                    signal = strategy.generate_signal(df)
                    print(f"✓ {name}: signal = {signal}")
                else:
                    print(f"⚠️  {name}: insufficient data ({df.shape[0]} points)")
            except Exception as e:
                print(f"❌ {name}: ERROR - {e}")
                return False

        print("✅ Paper trading integration - PASSED")
        return True

    except Exception as e:
        print(f"❌ Paper trading integration ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run comprehensive debugging."""
    print("🔍 SUPREME SYSTEM V5 - STRATEGY DEBUGGING")
    print("="*60)

    # Create test data
    test_data = create_test_data()
    print(f"✓ Test data created: {len(test_data)} data points")

    # Test strategies individually
    strategies_to_test = [
        (MovingAverageStrategy, "Moving Average"),
        (MeanReversionStrategy, "Mean Reversion"),
        (MomentumStrategy, "Momentum"),
        (BreakoutStrategy, "Breakout")
    ]

    results = []
    for strategy_class, name in strategies_to_test:
        result = test_strategy_isolation(strategy_class, name, test_data)
        results.append((name, result))

    # Test paper trading integration
    paper_trading_result = test_paper_trading_integration()
    results.append(("Paper Trading Integration", paper_trading_result))

    # Summary
    print(f"\n{'='*60}")
    print("DEBUGGING SUMMARY")
    print('='*60)

    passed = 0
    total = len(results)

    for name, result in results:
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"{name:25} {status}")
        if result:
            passed += 1

    print(f"\nOverall: {passed}/{total} tests passed")

    if passed == total:
        print("🎉 ALL TESTS PASSED - Strategies are working correctly!")
        return 0
    else:
        print("⚠️  SOME TESTS FAILED - Need to fix issues before production")
        return 1

if __name__ == "__main__":
    exit(main())
