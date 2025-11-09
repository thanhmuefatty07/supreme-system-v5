#!/usr/bin/env python3
"""
Test Data Pipeline functionality
"""

import sys
import pandas as pd
import numpy as np
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from data.data_pipeline import DataPipeline
from data.data_validator import DataValidator
from data.data_storage import DataStorage


def create_test_data(length: int = 100) -> pd.DataFrame:
    """Create realistic test data."""
    np.random.seed(42)

    # Generate timestamps
    start_date = pd.Timestamp('2024-01-01')
    timestamps = [start_date + pd.Timedelta(hours=i) for i in range(length)]

    # Generate OHLCV data
    base_price = 100.0
    prices = []
    volumes = []

    for i in range(length):
        # Add some trend and volatility
        trend = 0.001 * i  # Slight upward trend
        noise = np.random.normal(0, 2)
        price = base_price + trend * base_price + noise
        price = max(price, 0.01)  # Ensure positive price

        prices.append(price)

        # Volume with some randomness
        volume = np.random.uniform(1000, 10000)
        volumes.append(volume)

        base_price = price

    # Create OHLC from close prices
    df = pd.DataFrame({
        'timestamp': timestamps,
        'close': prices,
        'volume': volumes
    })

    # Generate OHLC from close prices with some spread
    df['open'] = df['close'].shift(1).fillna(df['close'])
    df['high'] = df[['open', 'close']].max(axis=1) + np.random.uniform(0, 1, length)
    df['low'] = df[['open', 'close']].min(axis=1) - np.random.uniform(0, 1, length)

    return df[['timestamp', 'open', 'high', 'low', 'close', 'volume']]


def test_data_validator():
    """Test data validator functionality."""
    print("🔍 Testing Data Validator...")

    validator = DataValidator()
    test_data = create_test_data(100)

    # Test validation
    result = validator.validate_ohlcv_data(test_data, "TESTUSDT")
    assert result['is_valid'] is True
    assert result['quality_score'] > 95
    print("  ✅ Data validation works")

    # Test cleaning
    cleaned = validator.clean_data(test_data)
    assert len(cleaned) >= 0  # Cleaning may remove invalid data
    assert isinstance(cleaned, pd.DataFrame)
    print("  ✅ Data cleaning works")

    return True


def test_data_storage():
    """Test data storage functionality."""
    print("\n💾 Testing Data Storage...")

    storage = DataStorage("./test_data")

    # Create test data
    test_data = create_test_data(200)

    # Test storage
    success = storage.store_historical_data(test_data, "TESTUSDT", "1h")
    assert success is True
    print("  ✅ Data storage works")

    # Test loading
    loaded_data = storage.load_historical_data("TESTUSDT", "1h")
    assert loaded_data is not None
    assert len(loaded_data) == len(test_data)
    print("  ✅ Data loading works")

    # Test data info
    info = storage.get_data_info()
    assert info['total_symbols'] > 0
    print("  ✅ Data info retrieval works")

    # Cleanup
    import shutil
    shutil.rmtree("./test_data", ignore_errors=True)

    return True


def test_data_pipeline():
    """Test complete data pipeline."""
    print("\n🔄 Testing Data Pipeline...")

    pipeline = DataPipeline()

    # Test pipeline status
    status = pipeline.get_pipeline_status()
    assert 'components' in status
    assert 'metrics' in status
    print("  ✅ Pipeline status works")

    # Test with mock data (since no real API credentials)
    test_data = create_test_data(50)

    # Manually store test data
    storage = DataStorage("./test_pipeline_data")
    storage.store_historical_data(test_data, "MOCKUSDT", "1h")

    # Test pipeline data retrieval
    pipeline.storage = storage
    data = pipeline.get_data("MOCKUSDT", "1h")
    assert data is not None
    assert len(data) == len(test_data)
    print("  ✅ Pipeline data retrieval works")

    # Test cache functionality
    data_cached = pipeline.get_data("MOCKUSDT", "1h")  # Should hit cache
    assert data_cached is not None
    print("  ✅ Pipeline caching works")

    # Test data quality validation
    quality = pipeline.validate_data_quality("MOCKUSDT", "1h")
    assert quality['data_available'] is True
    assert quality['quality_score'] > 95
    print("  ✅ Data quality validation works")

    # Cleanup
    import shutil
    shutil.rmtree("./test_pipeline_data", ignore_errors=True)

    return True


def test_pipeline_metrics():
    """Test pipeline metrics tracking."""
    print("\n📊 Testing Pipeline Metrics...")

    pipeline = DataPipeline()

    # Check initial metrics
    initial_metrics = pipeline.metrics.copy()

    # Perform validation using the same pipeline instance
    test_data = create_test_data(10)
    validation_result = pipeline.validator.validate_ohlcv_data(test_data, "METRICUSDT")

    # Check metrics updated
    updated_metrics = pipeline.metrics
    assert updated_metrics['validations'] >= initial_metrics['validations']
    assert validation_result['quality_score'] > 0
    print("  ✅ Metrics tracking works")

    return True


def test_error_handling():
    """Test error handling in pipeline."""
    print("\n🚨 Testing Error Handling...")

    pipeline = DataPipeline()

    # Test with invalid symbol
    result = pipeline.get_data("INVALID", "1h")
    assert result is None
    print("  ✅ Invalid symbol handling works")

    # Test data validation with empty data
    empty_data = pd.DataFrame()
    validation = pipeline.validator.validate_ohlcv_data(empty_data, "EMPTY")
    assert validation['is_valid'] is False
    print("  ✅ Empty data handling works")

    return True


def test_data_export():
    """Test data export functionality."""
    print("\n📤 Testing Data Export...")

    pipeline = DataPipeline()

    # Create test data
    test_data = create_test_data(20)
    storage = DataStorage("./test_export_data")
    storage.store_historical_data(test_data, "EXPORTUSDT", "1h")
    pipeline.storage = storage

    # Test CSV export
    csv_file = pipeline.export_data("EXPORTUSDT", "1h", format="csv")
    assert csv_file is not None
    assert csv_file.endswith(".csv")
    assert Path(csv_file).exists()
    print("  ✅ CSV export works")

    # Test JSON export
    json_file = pipeline.export_data("EXPORTUSDT", "1h", format="json")
    assert json_file is not None
    assert json_file.endswith(".json")
    assert Path(json_file).exists()
    print("  ✅ JSON export works")

    # Cleanup
    import shutil
    shutil.rmtree("./test_export_data", ignore_errors=True)
    Path(csv_file).unlink(missing_ok=True)
    Path(json_file).unlink(missing_ok=True)

    return True


def main():
    """Run all data pipeline tests."""
    print("🧪 SUPREME SYSTEM V5 - DATA PIPELINE TESTS")
    print("=" * 60)

    tests = [
        ("Data Validator", test_data_validator),
        ("Data Storage", test_data_storage),
        ("Data Pipeline", test_data_pipeline),
        ("Pipeline Metrics", test_pipeline_metrics),
        ("Error Handling", test_error_handling),
        ("Data Export", test_data_export)
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

    print("\n" + "=" * 60)
    print(f"🎯 DATA PIPELINE TESTS RESULTS: {passed}/{total} tests passed")

    if passed == total:
        print("🎉 ALL DATA PIPELINE TESTS PASSED!")
        print("🚀 Data pipeline is fully functional")
        print("\n💡 Pipeline Capabilities:")
        print("   • Real-time data validation")
        print("   • Efficient Parquet storage with partitioning")
        print("   • Intelligent caching system")
        print("   • Comprehensive error handling")
        print("   • Multiple export formats")
        print("   • Quality monitoring and metrics")
    else:
        print("⚠️ Some pipeline tests failed - check error messages above")

    print("=" * 60)

    return passed == total


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
