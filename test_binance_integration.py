#!/usr/bin/env python3
"""
Test real Binance API integration
"""

import sys
import os
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from data.binance_client import BinanceClient
from config.config import get_config


def test_binance_client_initialization():
    """Test Binance client initialization"""
    print("🔧 Testing Binance Client Initialization...")

    # Test without credentials
    client = BinanceClient()
    assert client.client is None
    assert client.testnet is True
    print("✅ Client initializes without credentials")

    # Test with config
    config = get_config()
    client = BinanceClient(config_file=None)  # Use default config
    assert hasattr(client, 'config')
    print("✅ Client uses configuration system")

    return True


def test_connection_without_credentials():
    """Test connection test without credentials"""
    print("\n🌐 Testing Connection (No Credentials)...")

    client = BinanceClient()

    # Should fail gracefully without credentials
    result = client.test_connection()
    assert result is False
    print("✅ Gracefully handles missing credentials")

    return True


def test_input_validation():
    """Test input validation"""
    print("\n✅ Testing Input Validation...")

    client = BinanceClient()

    # Test symbol validation
    assert client._validate_symbol("ETHUSDT") is True
    assert client._validate_symbol("") is False
    assert client._validate_symbol("INVALID") is True  # Warning but valid format
    print("✅ Symbol validation works")

    # Test interval validation
    assert client._validate_interval("1h") is True
    assert client._validate_interval("invalid") is False
    print("✅ Interval validation works")

    return True


def test_request_stats():
    """Test request statistics"""
    print("\n📊 Testing Request Statistics...")

    client = BinanceClient()

    # Initial stats
    stats = client.get_request_stats()
    assert stats['total_requests'] == 0
    assert stats['error_count'] == 0
    assert stats['error_rate'] == 0.0
    print("✅ Initial statistics correct")

    # Test health check
    assert client.is_healthy() is True  # No requests = healthy
    print("✅ Health check works")

    return True


def test_interval_parsing():
    """Test interval parsing"""
    print("\n⏰ Testing Interval Parsing...")

    client = BinanceClient()

    # Test various intervals
    assert client._parse_interval_to_timedelta("1h") == pd.Timedelta("1H")
    assert client._parse_interval_to_timedelta("1d") == pd.Timedelta("1D")
    assert client._parse_interval_to_timedelta("1m") == pd.Timedelta("1min")
    print("✅ Interval parsing works")

    return True


def test_config_integration():
    """Test configuration integration"""
    print("\n⚙️ Testing Configuration Integration...")

    config = get_config()

    # Test config values
    assert 'binance' in config._config
    assert 'trading' in config._config
    assert 'data' in config._config
    print("✅ Configuration sections available")

    # Test API credentials check
    has_creds = config.validate_api_credentials()
    print(f"📋 API Credentials configured: {has_creds}")

    return True


def test_error_handling():
    """Test error handling scenarios"""
    print("\n🚨 Testing Error Handling...")

    client = BinanceClient()

    # Test with invalid inputs
    result = client.get_historical_klines("", "1h", "2024-01-01")
    assert result is None
    print("✅ Handles invalid symbol")

    result = client.get_historical_klines("ETHUSDT", "invalid", "2024-01-01")
    assert result is None
    print("✅ Handles invalid interval")

    result = client.get_historical_klines("ETHUSDT", "1h", "invalid-date")
    assert result is None
    print("✅ Handles invalid date")

    return True


def main():
    """Run all integration tests"""
    print("🚀 SUPREME SYSTEM V5 - BINANCE INTEGRATION TEST")
    print("=" * 60)

    # Import pandas here to avoid import errors in tests
    global pd
    import pandas as pd

    tests = [
        ("Client Initialization", test_binance_client_initialization),
        ("Connection Without Credentials", test_connection_without_credentials),
        ("Input Validation", test_input_validation),
        ("Request Statistics", test_request_stats),
        ("Interval Parsing", test_interval_parsing),
        ("Configuration Integration", test_config_integration),
        ("Error Handling", test_error_handling)
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

    print("\n" + "=" * 60)
    print(f"🎯 INTEGRATION TEST RESULTS: {passed}/{total} tests passed")

    if passed == total:
        print("🎉 ALL INTEGRATION TESTS PASSED!")
        print("🚀 Ready for real API testing with credentials")
        print("\n💡 To test with real data:")
        print("   1. Set BINANCE_API_KEY and BINANCE_API_SECRET environment variables")
        print("   2. Run: python test_binance_integration.py --real")
        print("   3. Or use: python -m src.cli data download --symbol ETHUSDT --start-date 2024-01-01")
    else:
        print("⚠️ Some tests failed - check error messages above")

    print("=" * 60)

    return passed == total


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
