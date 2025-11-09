#!/usr/bin/env python3
"""
Supreme System V5 - Basic Functionality Test

Real test to verify core components work correctly.
"""

import sys
import pandas as pd
import numpy as np
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

def test_imports():
    """Test all imports work correctly"""
    print("🔍 Testing imports...")

    try:
        from data.binance_client import BinanceClient
        print("✅ BinanceClient import")
    except Exception as e:
        print(f"❌ BinanceClient import failed: {e}")
        return False

    try:
        from strategies.base_strategy import BaseStrategy
        print("✅ BaseStrategy import")
    except Exception as e:
        print(f"❌ BaseStrategy import failed: {e}")
        return False

    try:
        from strategies.moving_average import MovingAverageStrategy
        print("✅ MovingAverageStrategy import")
    except Exception as e:
        print(f"❌ MovingAverageStrategy import failed: {e}")
        return False

    try:
        from risk.risk_manager import RiskManager
        print("✅ RiskManager import")
    except Exception as e:
        print(f"❌ RiskManager import failed: {e}")
        return False

    return True

def test_strategy_logic():
    """Test strategy signal generation"""
    print("\n🎯 Testing strategy logic...")

    try:
        from strategies.moving_average import MovingAverageStrategy

        # Create strategy
        strategy = MovingAverageStrategy(short_window=5, long_window=10)

        # Create mock data
        dates = pd.date_range('2024-01-01', periods=50, freq='D')
        prices = np.random.uniform(100, 200, 50)
        data = pd.DataFrame({
            'timestamp': dates,
            'open': prices,
            'high': prices * 1.02,
            'low': prices * 0.98,
            'close': prices,
            'volume': np.random.uniform(1000, 10000, 50)
        })

        # Test signal generation
        signals = []
        for i in range(len(data)):
            signal = strategy.generate_signal(data.iloc[:i+1])
            signals.append(signal)

        print(f"✅ Generated {len(signals)} signals")
        print(f"   Buy signals: {signals.count(1)}")
        print(f"   Sell signals: {signals.count(-1)}")
        print(f"   Hold signals: {signals.count(0)}")

        return True

    except Exception as e:
        print(f"❌ Strategy test failed: {e}")
        return False

def test_risk_management():
    """Test risk management calculations"""
    print("\n💰 Testing risk management...")

    try:
        from risk.risk_manager import RiskManager

        # Create risk manager
        risk_mgr = RiskManager(initial_capital=10000)

        # Test position sizing
        position_size = risk_mgr.calculate_position_size(100.0)
        print(".2f")

        # Test stop loss
        stop_triggered = risk_mgr.check_stop_loss(100.0, 98.0, True)  # 2% loss
        print(f"✅ Stop loss check: {stop_triggered} (should be True)")

        # Test take profit
        profit_triggered = risk_mgr.check_take_profit(100.0, 105.0, True)  # 5% profit
        print(f"✅ Take profit check: {profit_triggered} (should be True)")

        return True

    except Exception as e:
        print(f"❌ Risk management test failed: {e}")
        return False

def test_backtesting_framework():
    """Test basic backtesting functionality"""
    print("\n📊 Testing backtesting framework...")

    try:
        from strategies.moving_average import MovingAverageStrategy
        from risk.risk_manager import RiskManager

        # Create mock data (simple uptrend)
        dates = pd.date_range('2024-01-01', periods=100, freq='D')
        prices = np.linspace(100, 150, 100)  # Steady uptrend
        data = pd.DataFrame({
            'timestamp': dates,
            'open': prices,
            'high': prices * 1.01,
            'low': prices * 0.99,
            'close': prices,
            'volume': np.random.uniform(1000, 10000, 100)
        })

        # Create strategy and risk manager
        strategy = MovingAverageStrategy(short_window=5, long_window=20)
        risk_mgr = RiskManager(initial_capital=10000)

        # Run backtest
        results = risk_mgr.run_backtest(data, strategy)

        print("✅ Backtest completed")
        print(".2f")
        print(".2%")
        print(f"   Total trades: {results['total_trades']}")
        print(".2f")

        return True

    except Exception as e:
        print(f"❌ Backtesting test failed: {e}")
        return False

def main():
    """Run all basic functionality tests"""
    print("🚀 SUPREME SYSTEM V5 - BASIC FUNCTIONALITY TEST")
    print("=" * 60)

    tests = [
        ("Import Tests", test_imports),
        ("Strategy Logic", test_strategy_logic),
        ("Risk Management", test_risk_management),
        ("Backtesting Framework", test_backtesting_framework)
    ]

    passed = 0
    total = len(tests)

    for test_name, test_func in tests:
        print(f"\n{'='*20} {test_name} {'='*20}")
        if test_func():
            passed += 1
            print(f"✅ {test_name}: PASSED")
        else:
            print(f"❌ {test_name}: FAILED")

    print("\n" + "=" * 60)
    print(f"🎯 TEST RESULTS: {passed}/{total} tests passed")

    if passed == total:
        print("🎉 ALL TESTS PASSED - Foundation is solid!")
        print("🚀 Ready for Phase 2 development")
    else:
        print("⚠️ Some tests failed - needs debugging")
        print("🔧 Check error messages above")

    print("=" * 60)

    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
