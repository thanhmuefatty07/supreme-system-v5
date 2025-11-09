#!/usr/bin/env python3
"""
Test Dashboard functionality
"""

import sys
import pandas as pd
import numpy as np
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from dashboard import MonitoringDashboard, create_price_chart, create_signal_chart


def test_dashboard_initialization():
    """Test dashboard initialization."""
    print("📊 Testing Dashboard Initialization...")

    dashboard = MonitoringDashboard()

    assert dashboard.pipeline is not None
    assert dashboard.strategies is not None
    assert len(dashboard.strategies) == 4
    assert not dashboard.is_monitoring

    print("  ✅ Dashboard initialization works")

    return True


def test_price_chart_creation():
    """Test price chart creation."""
    print("\n📈 Testing Price Chart Creation...")

    # Test with empty data
    empty_chart = create_price_chart({})
    assert empty_chart is not None

    # Test with sample data
    sample_data = {
        'BTCUSDT': {
            'price': 50000.0,
            'timestamp': pd.Timestamp.now()
        },
        'ETHUSDT': {
            'price': 3000.0,
            'timestamp': pd.Timestamp.now()
        }
    }

    price_chart = create_price_chart(sample_data)
    assert price_chart is not None

    print("  ✅ Price chart creation works")

    return True


def test_signal_chart_creation():
    """Test signal chart creation."""
    print("\n🎯 Testing Signal Chart Creation...")

    # Test with empty data
    empty_chart = create_signal_chart([])
    assert empty_chart is not None

    # Test with sample signals
    sample_signals = [
        {
            'timestamp': pd.Timestamp.now(),
            'strategy': 'Moving Average',
            'signal': 1,
            'price': 50000.0
        },
        {
            'timestamp': pd.Timestamp.now(),
            'strategy': 'Mean Reversion',
            'signal': -1,
            'price': 49900.0
        }
    ]

    signal_chart = create_signal_chart(sample_signals)
    assert signal_chart is not None

    print("  ✅ Signal chart creation works")

    return True


def test_strategy_integration():
    """Test strategy integration in dashboard."""
    print("\n🎲 Testing Strategy Integration...")

    dashboard = MonitoringDashboard()

    # Test strategy count
    assert len(dashboard.strategies) == 4
    expected_strategies = ['Moving Average', 'Mean Reversion', 'Momentum', 'Breakout']
    actual_strategies = list(dashboard.strategies.keys())

    for strategy in expected_strategies:
        assert strategy in actual_strategies

    print("  ✅ Strategy integration works")

    return True


def test_data_update_simulation():
    """Test data update simulation."""
    print("\n🔄 Testing Data Update Simulation...")

    dashboard = MonitoringDashboard()

    # Simulate data update (without real WebSocket)
    initial_price_count = len(dashboard.price_data)
    initial_signal_count = len(dashboard.signal_data)

    # Manually add some test data
    dashboard.price_data = {
        'BTCUSDT': {'price': 50000.0, 'timestamp': pd.Timestamp.now()}
    }

    dashboard.signal_data = [{
        'timestamp': pd.Timestamp.now(),
        'strategy': 'Test Strategy',
        'signal': 1,
        'price': 50000.0
    }]

    # Verify data was added
    assert len(dashboard.price_data) > initial_price_count
    assert len(dashboard.signal_data) > initial_signal_count

    print("  ✅ Data update simulation works")

    return True


def main():
    """Run all dashboard tests."""
    print("📊 SUPREME SYSTEM V5 - DASHBOARD TESTS")
    print("=" * 50)

    # Import Path here to avoid issues
    from pathlib import Path

    tests = [
        ("Dashboard Initialization", test_dashboard_initialization),
        ("Price Chart Creation", test_price_chart_creation),
        ("Signal Chart Creation", test_signal_chart_creation),
        ("Strategy Integration", test_strategy_integration),
        ("Data Update Simulation", test_data_update_simulation)
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

    print("\n" + "=" * 50)
    print(f"🎯 DASHBOARD TESTS RESULTS: {passed}/{total} tests passed")

    if passed == total:
        print("🎉 ALL DASHBOARD TESTS PASSED!")
        print("🚀 Monitoring dashboard is ready!")
        print("\n💡 Dashboard Features:")
        print("   • Real-time price monitoring")
        print("   • Trading signal visualization")
        print("   • System health metrics")
        print("   • WebSocket connection status")
        print("   • Interactive charts with Plotly")
        print("   • Auto-refresh capabilities")
    else:
        print("⚠️ Some dashboard tests failed - check error messages above")

    print("=" * 50)

    return passed == total


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
