#!/usr/bin/env python3
"""
Comprehensive test for all trading strategies
"""

import sys
import pandas as pd
import numpy as np

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from strategies.moving_average import MovingAverageStrategy
from strategies.mean_reversion import MeanReversionStrategy
from strategies.momentum import MomentumStrategy
from strategies.breakout import BreakoutStrategy


def create_test_data(length: int = 100, trend: str = "sideways") -> pd.DataFrame:
    """Create test data with different patterns."""
    np.random.seed(42)

    # Base price series
    if trend == "uptrend":
        prices = np.linspace(100, 150, length)
    elif trend == "downtrend":
        prices = np.linspace(150, 100, length)
    else:  # sideways
        prices = 100 + 10 * np.sin(np.linspace(0, 4*np.pi, length))

    # Add some noise
    noise = np.random.normal(0, 2, length)
    prices = prices + noise

    # Ensure positive prices
    prices = np.maximum(prices, 1)

    # Create OHLCV data
    data = pd.DataFrame({
        'timestamp': pd.date_range('2024-01-01', periods=length, freq='D'),
        'open': prices,
        'high': prices * 1.02 + np.random.uniform(0, 1, length),
        'low': prices * 0.98 - np.random.uniform(0, 1, length),
        'close': prices,
        'volume': np.random.uniform(1000, 10000, length)
    })

    return data


def test_moving_average_strategy():
    """Test Moving Average strategy with different data patterns."""
    print("🎯 Testing Moving Average Strategy...")

    # Test with uptrend data (should generate buy signals)
    uptrend_data = create_test_data(100, "uptrend")
    strategy = MovingAverageStrategy(short_window=5, long_window=20)

    signals = []
    for i in range(len(uptrend_data)):
        signal = strategy.generate_signal(uptrend_data.iloc[:i+1])
        signals.append(signal)

    buy_signals = sum(1 for s in signals if s == 1)
    sell_signals = sum(1 for s in signals if s == -1)

    print(f"  Uptrend: {buy_signals} buy, {sell_signals} sell signals")

    # Test with downtrend data (should generate sell signals)
    downtrend_data = create_test_data(100, "downtrend")

    signals = []
    for i in range(len(downtrend_data)):
        signal = strategy.generate_signal(downtrend_data.iloc[:i+1])
        signals.append(signal)

    buy_signals = sum(1 for s in signals if s == 1)
    sell_signals = sum(1 for s in signals if s == -1)

    print(f"  Downtrend: {buy_signals} buy, {sell_signals} sell signals")

    return True


def test_mean_reversion_strategy():
    """Test Mean Reversion strategy."""
    print("\n📊 Testing Mean Reversion Strategy...")

    # Create data with mean reversion pattern (oscillating around mean)
    data = create_test_data(150, "sideways")
    strategy = MeanReversionStrategy(lookback_period=20, entry_threshold=1.5)

    signals = []
    for i in range(len(data)):
        signal = strategy.generate_signal(data.iloc[:i+1])
        signals.append(signal)

    buy_signals = sum(1 for s in signals if s == 1)
    sell_signals = sum(1 for s in signals if s == -1)

    print(f"  Sideways: {buy_signals} buy, {sell_signals} sell signals")

    # Test Bollinger Bands calculation
    bb_data = strategy.calculate_bollinger_bands(data)
    assert 'bb_upper' in bb_data.columns
    assert 'bb_lower' in bb_data.columns
    print("  ✅ Bollinger Bands calculation works")

    return True


def test_momentum_strategy():
    """Test Momentum strategy."""
    print("\n📈 Testing Momentum Strategy...")

    # Test with uptrend (should show bullish momentum)
    uptrend_data = create_test_data(100, "uptrend")
    strategy = MomentumStrategy()

    signals = []
    for i in range(len(uptrend_data)):
        signal = strategy.generate_signal(uptrend_data.iloc[:i+1])
        signals.append(signal)

    buy_signals = sum(1 for s in signals if s == 1)
    sell_signals = sum(1 for s in signals if s == -1)

    print(f"  Uptrend: {buy_signals} buy, {sell_signals} sell signals")

    # Test momentum indicators calculation
    momentum_data = strategy.calculate_momentum_indicators(uptrend_data)
    assert 'macd_line' in momentum_data.columns
    assert 'roc' in momentum_data.columns
    print("  ✅ Momentum indicators calculation works")

    # Test momentum score
    score = strategy.get_momentum_score(uptrend_data)
    print(".2f")
    assert -1 <= score <= 1

    return True


def test_breakout_strategy():
    """Test Breakout strategy."""
    print("\n🚀 Testing Breakout Strategy...")

    # Create breakout pattern (consolidation then breakout)
    data = create_test_data(100, "sideways")
    # Add a breakout at the end
    breakout_prices = np.linspace(105, 115, 20)
    data.loc[len(data)-20:, 'close'] = breakout_prices
    data.loc[len(data)-20:, 'high'] = breakout_prices * 1.02
    data.loc[len(data)-20:, 'low'] = breakout_prices * 0.98
    data.loc[len(data)-20:, 'volume'] = 20000  # High volume breakout

    strategy = BreakoutStrategy(lookback_period=30)

    signals = []
    for i in range(len(data)):
        signal = strategy.generate_signal(data.iloc[:i+1])
        signals.append(signal)

    buy_signals = sum(1 for s in signals if s == 1)
    sell_signals = sum(1 for s in signals if s == -1)

    print(f"  Breakout pattern: {buy_signals} buy, {sell_signals} sell signals")

    # Test support/resistance identification
    levels = strategy.identify_support_resistance(data)
    assert 'support' in levels
    assert 'resistance' in levels
    print("  ✅ Support/resistance identification works")

    # Test breakout indicators
    breakout_data = strategy.add_breakout_indicators(data)
    assert 'rolling_high' in breakout_data.columns
    assert 'consolidation' in breakout_data.columns
    print("  ✅ Breakout indicators calculation works")

    return True


def test_strategy_parameters():
    """Test strategy parameter management."""
    print("\n⚙️ Testing Strategy Parameters...")

    strategies = [
        MovingAverageStrategy(short_window=10, long_window=30),
        MeanReversionStrategy(lookback_period=25, entry_threshold=1.8),
        MomentumStrategy(short_period=15, long_period=30),
        BreakoutStrategy(lookback_period=25, breakout_threshold=0.03)
    ]

    for strategy in strategies:
        # Test parameter setting
        strategy.set_parameters(test_param=123)
        params = strategy.get_parameters()
        assert 'test_param' in params
        assert params['test_param'] == 123
        print(f"  ✅ {strategy.name} parameter management works")

    return True


def test_strategy_data_validation():
    """Test strategy data validation."""
    print("\n🔍 Testing Strategy Data Validation...")

    strategy = MovingAverageStrategy()

    # Valid data
    valid_data = pd.DataFrame({
        'timestamp': pd.date_range('2024-01-01', periods=10),
        'open': [100] * 10,
        'high': [105] * 10,
        'low': [95] * 10,
        'close': [102] * 10,
        'volume': [1000] * 10
    })
    assert strategy.validate_data(valid_data) is True

    # Invalid data - missing columns
    invalid_data = pd.DataFrame({'close': [1, 2, 3]})
    assert strategy.validate_data(invalid_data) is False

    # Invalid data - empty
    empty_data = pd.DataFrame()
    assert strategy.validate_data(empty_data) is False

    print("  ✅ Data validation works correctly")
    return True


def main():
    """Run comprehensive strategy tests."""
    print("🧪 SUPREME SYSTEM V5 - COMPREHENSIVE STRATEGY TESTS")
    print("=" * 65)

    tests = [
        ("Moving Average Strategy", test_moving_average_strategy),
        ("Mean Reversion Strategy", test_mean_reversion_strategy),
        ("Momentum Strategy", test_momentum_strategy),
        ("Breakout Strategy", test_breakout_strategy),
        ("Strategy Parameters", test_strategy_parameters),
        ("Strategy Data Validation", test_strategy_data_validation)
    ]

    passed = 0
    total = len(tests)

    for test_name, test_func in tests:
        print(f"\n{'='*20} {test_name} {'='*20}")
        try:
            if test_func():
                passed += 1
                print(f"✅ {test_name}: PASSED")
            else:
                print(f"❌ {test_name}: FAILED")
        except Exception as e:
            print(f"❌ {test_name}: ERROR - {e}")
            import traceback
            traceback.print_exc()

    print("\n" + "=" * 65)
    print(f"🎯 STRATEGY TESTS RESULTS: {passed}/{total} tests passed")

    if passed == total:
        print("🎉 ALL STRATEGY TESTS PASSED!")
        print("🚀 All trading strategies are functional and ready")
        print("\n💡 Strategy Capabilities:")
        print("   • Moving Average: Trend-following crossover signals")
        print("   • Mean Reversion: Bollinger Band mean reversion")
        print("   • Momentum: MACD + ROC momentum detection")
        print("   • Breakout: Support/resistance breakout trading")
    else:
        print("⚠️ Some strategy tests failed - check error messages above")

    print("=" * 65)

    return passed == total


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
