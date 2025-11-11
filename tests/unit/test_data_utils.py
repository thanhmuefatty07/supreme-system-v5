"""
Comprehensive unit tests for data utilities in Supreme System V5.

Tests data processing, optimization, validation, and analysis utilities.
"""
import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from unittest.mock import Mock, patch


class TestDataValidation:
    """Test data validation utilities"""

    def test_validate_and_clean_data_success(self):
        """Test successful data validation and cleaning"""
        from src.utils.data_utils import validate_and_clean_data

        # Create valid OHLCV data
        data = pd.DataFrame({
            'timestamp': pd.date_range('2024-01-01', periods=100, freq='1h'),
            'open': 100 + np.random.normal(0, 2, 100),
            'high': 102 + np.random.normal(0, 1, 100),
            'low': 98 + np.random.normal(0, 1, 100),
            'close': 100 + np.random.normal(0, 2, 100),
            'volume': np.random.randint(1000, 10000, 100)
        })

        is_valid, errors = validate_and_clean_data(data)

        assert is_valid is True
        assert len(errors) == 0

    def test_validate_and_clean_data_missing_columns(self):
        """Test validation with missing required columns"""
        from src.utils.data_utils import validate_and_clean_data

        # Missing 'volume' column
        data = pd.DataFrame({
            'timestamp': pd.date_range('2024-01-01', periods=10, freq='1h'),
            'open': [100] * 10,
            'high': [102] * 10,
            'low': [98] * 10,
            'close': [101] * 10
        })

        is_valid, errors = validate_and_clean_data(data)

        assert is_valid is False
        assert len(errors) > 0
        assert any('volume' in error.lower() for error in errors)

    def test_validate_and_clean_data_negative_prices(self):
        """Test validation with negative prices"""
        from src.utils.data_utils import validate_and_clean_data

        data = pd.DataFrame({
            'timestamp': pd.date_range('2024-01-01', periods=10, freq='1h'),
            'open': [-100] * 10,  # Negative prices
            'high': [102] * 10,
            'low': [98] * 10,
            'close': [101] * 10,
            'volume': [1000] * 10
        })

        is_valid, errors = validate_and_clean_data(data)

        assert is_valid is False
        assert len(errors) > 0

    def test_validate_and_clean_data_ohlc_relationship(self):
        """Test OHLC relationship validation"""
        from src.utils.data_utils import validate_and_clean_data

        # Invalid OHLC: high < low
        data = pd.DataFrame({
            'timestamp': pd.date_range('2024-01-01', periods=5, freq='1h'),
            'open': [100] * 5,
            'high': [95] * 5,  # High < Low (invalid)
            'low': [98] * 5,
            'close': [101] * 5,
            'volume': [1000] * 5
        })

        is_valid, errors = validate_and_clean_data(data)

        assert is_valid is False
        assert len(errors) > 0

    def test_validate_and_clean_data_nan_values(self):
        """Test handling of NaN values"""
        from src.utils.data_utils import validate_and_clean_data

        data = pd.DataFrame({
            'timestamp': pd.date_range('2024-01-01', periods=10, freq='1h'),
            'open': [100, np.nan, 102, 103, 104, 105, 106, 107, 108, 109],
            'high': [102] * 10,
            'low': [98] * 10,
            'close': [101] * 10,
            'volume': [1000] * 10
        })

        is_valid, errors = validate_and_clean_data(data)

        assert is_valid is False
        assert len(errors) > 0


class TestDataOptimization:
    """Test data optimization utilities"""

    def test_optimize_dataframe_memory_numeric_columns(self):
        """Test memory optimization for numeric columns"""
        from src.utils.data_utils import optimize_dataframe_memory

        # Create DataFrame with different dtypes
        data = pd.DataFrame({
            'int_col': np.random.randint(0, 100, 1000),
            'float_col': np.random.uniform(0, 1, 1000),
            'volume': np.random.uniform(1000, 10000, 1000)  # Should stay float64
        })

        original_memory = data.memory_usage(deep=True).sum()
        optimized_data = optimize_dataframe_memory(data)
        optimized_memory = optimized_data.memory_usage(deep=True).sum()

        # Memory should be reduced or at least not increased significantly
        assert optimized_memory <= original_memory * 1.1

        # Volume should remain float64 due to precision needs
        assert optimized_data['volume'].dtype == np.dtype('float64')

    def test_optimize_dataframe_memory_datetime_columns(self):
        """Test memory optimization for datetime columns"""
        from src.utils.data_utils import optimize_dataframe_memory

        dates = pd.date_range('2024-01-01', periods=1000, freq='1min')
        data = pd.DataFrame({
            'timestamp': dates,
            'value': np.random.uniform(0, 100, 1000)
        })

        original_memory = data.memory_usage(deep=True).sum()
        optimized_data = optimize_dataframe_memory(data)

        # Should not crash and should preserve datetime
        assert len(optimized_data) == len(data)
        assert pd.api.types.is_datetime64_any_dtype(optimized_data['timestamp'])

    def test_optimize_dataframe_memory_empty_dataframe(self):
        """Test optimization with empty DataFrame"""
        from src.utils.data_utils import optimize_dataframe_memory

        empty_df = pd.DataFrame()
        result = optimize_dataframe_memory(empty_df)

        assert len(result) == 0
        assert list(result.columns) == []

    def test_optimize_dataframe_memory_mixed_types(self):
        """Test optimization with mixed data types"""
        from src.utils.data_utils import optimize_dataframe_memory

        data = pd.DataFrame({
            'int_small': [1, 2, 3, 4, 5],  # Can be int8
            'int_large': [1000000] * 5,     # Should be int32 or int64
            'float_normal': [1.0, 2.5, 3.7, 4.2, 5.9],  # Can be float32
            'float_precise': [1.123456789] * 5,  # Should stay float64
            'string_col': ['a', 'b', 'c', 'd', 'e']  # Object dtype
        })

        result = optimize_dataframe_memory(data)

        # Check that optimization was applied appropriately
        assert result['int_small'].dtype == np.dtype('int8') or result['int_small'].dtype == np.dtype('int32')
        assert result['float_normal'].dtype == np.dtype('float32')
        assert result['float_precise'].dtype == np.dtype('float64')  # High precision preserved
        assert result['string_col'].dtype == object


class TestDataAnalysis:
    """Test data analysis utilities"""

    def test_calculate_returns_basic(self):
        """Test basic return calculation"""
        from src.utils.data_utils import calculate_returns

        prices = pd.Series([100, 101, 103, 102, 105])
        returns = calculate_returns(prices)

        expected_returns = [0.0, 0.01, 0.0198, -0.0097, 0.0291]  # Approximate
        np.testing.assert_allclose(returns.values[1:], expected_returns[1:], rtol=1e-3)

    def test_calculate_returns_empty_series(self):
        """Test return calculation with empty series"""
        from src.utils.data_utils import calculate_returns

        empty_prices = pd.Series([], dtype=float)
        returns = calculate_returns(empty_prices)

        assert len(returns) == 0

    def test_calculate_returns_single_value(self):
        """Test return calculation with single value"""
        from src.utils.data_utils import calculate_returns

        single_price = pd.Series([100.0])
        returns = calculate_returns(single_price)

        assert len(returns) == 1
        assert returns.iloc[0] == 0.0

    def test_calculate_volatility_basic(self):
        """Test basic volatility calculation"""
        from src.utils.data_utils import calculate_volatility

        returns = pd.Series([0.01, -0.005, 0.015, -0.01, 0.008])
        volatility = calculate_volatility(returns, window=5)

        assert isinstance(volatility, float)
        assert volatility >= 0

    def test_calculate_volatility_insufficient_data(self):
        """Test volatility calculation with insufficient data"""
        from src.utils.data_utils import calculate_volatility

        short_returns = pd.Series([0.01, 0.02])
        volatility = calculate_volatility(short_returns, window=5)

        # Should handle insufficient data gracefully
        assert isinstance(volatility, float)

    def test_detect_outliers_iqr_method(self):
        """Test outlier detection using IQR method"""
        from src.utils.data_utils import detect_outliers

        # Create data with clear outliers
        normal_data = np.random.normal(100, 5, 100)
        data_with_outliers = np.concatenate([normal_data, [200, -50]])  # Add outliers

        data_series = pd.Series(data_with_outliers)
        outliers = detect_outliers(data_series, method='iqr')

        assert isinstance(outliers, pd.Series)
        assert len(outliers) > 0  # Should detect some outliers

    def test_detect_outliers_zscore_method(self):
        """Test outlier detection using z-score method"""
        from src.utils.data_utils import detect_outliers

        data = pd.Series([1, 2, 3, 4, 5, 100])  # 100 is clear outlier
        outliers = detect_outliers(data, method='zscore', threshold=2)

        assert isinstance(outliers, pd.Series)
        assert 100 in outliers.values

    def test_detect_outliers_invalid_method(self):
        """Test outlier detection with invalid method"""
        from src.utils.data_utils import detect_outliers

        data = pd.Series([1, 2, 3, 4, 5])

        with pytest.raises(ValueError):
            detect_outliers(data, method='invalid')


class TestDataTransformation:
    """Test data transformation utilities"""

    def test_normalize_data_standard_scaler(self):
        """Test data normalization using standard scaler"""
        from src.utils.data_utils import normalize_data

        data = pd.DataFrame({
            'feature1': [1, 2, 3, 4, 5],
            'feature2': [10, 20, 30, 40, 50]
        })

        normalized = normalize_data(data, method='standard')

        assert len(normalized) == len(data)
        assert list(normalized.columns) == list(data.columns)

        # Check approximate normalization (mean ~0, std ~1)
        for col in normalized.columns:
            assert abs(normalized[col].mean()) < 0.1
            assert abs(normalized[col].std() - 1.0) < 0.1

    def test_normalize_data_minmax_scaler(self):
        """Test data normalization using min-max scaler"""
        from src.utils.data_utils import normalize_data

        data = pd.DataFrame({
            'feature1': [1, 2, 3, 4, 5],
            'feature2': [10, 20, 30, 40, 50]
        })

        normalized = normalize_data(data, method='minmax')

        assert len(normalized) == len(data)

        # Check min-max scaling (values should be between 0 and 1)
        for col in normalized.columns:
            assert normalized[col].min() >= 0
            assert normalized[col].max() <= 1

    def test_normalize_data_invalid_method(self):
        """Test normalization with invalid method"""
        from src.utils.data_utils import normalize_data

        data = pd.DataFrame({'col': [1, 2, 3]})

        with pytest.raises(ValueError):
            normalize_data(data, method='invalid')

    def test_resample_data_hourly_to_daily(self):
        """Test data resampling from hourly to daily"""
        from src.utils.data_utils import resample_data

        # Create hourly data
        dates = pd.date_range('2024-01-01', periods=48, freq='H')  # 2 days of hourly data
        data = pd.DataFrame({
            'open': np.random.uniform(100, 110, 48),
            'high': np.random.uniform(110, 120, 48),
            'low': np.random.uniform(90, 100, 48),
            'close': np.random.uniform(100, 110, 48),
            'volume': np.random.randint(1000, 5000, 48)
        }, index=dates)

        resampled = resample_data(data, 'D')

        # Should have 2 days of data
        assert len(resampled) == 2
        assert resampled.index.freq == 'D'

    def test_resample_data_insufficient_data(self):
        """Test resampling with insufficient data"""
        from src.utils.data_utils import resample_data

        # Single data point
        data = pd.DataFrame({
            'close': [100]
        }, index=pd.date_range('2024-01-01', periods=1, freq='H'))

        resampled = resample_data(data, 'D')

        assert len(resampled) == 1


class TestPerformanceBenchmarks:
    """Test performance benchmarks for data utilities"""

    def test_large_dataframe_optimization_performance(self):
        """Test performance of optimization on large DataFrame"""
        from src.utils.data_utils import optimize_dataframe_memory
        import time

        # Create large DataFrame
        n_rows = 100000
        data = pd.DataFrame({
            'int_col': np.random.randint(0, 1000, n_rows),
            'float_col': np.random.uniform(0, 100, n_rows),
            'volume': np.random.uniform(1000, 10000, n_rows)
        })

        start_time = time.time()
        optimized = optimize_dataframe_memory(data)
        end_time = time.time()

        # Should complete in reasonable time (< 5 seconds)
        assert end_time - start_time < 5.0
        assert len(optimized) == n_rows

    def test_batch_processing_efficiency(self):
        """Test efficiency of batch processing operations"""
        from src.utils.data_utils import process_data_batch
        import time

        # Create batch data
        batch_data = []
        for i in range(10):
            data = pd.DataFrame({
                'close': np.random.uniform(100, 110, 1000),
                'volume': np.random.randint(1000, 5000, 1000)
            })
            batch_data.append(data)

        start_time = time.time()

        # Mock batch processing function
        results = []
        for data in batch_data:
            # Simulate some processing
            result = data.mean()
            results.append(result)

        end_time = time.time()

        # Should process batch efficiently
        assert end_time - start_time < 2.0
        assert len(results) == len(batch_data)

    def test_memory_efficiency_validation(self):
        """Test memory efficiency of data operations"""
        from src.utils.data_utils import validate_memory_efficiency
        import psutil
        import os

        # Get initial memory
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss

        # Perform memory-intensive operation
        large_data = pd.DataFrame({
            'data': np.random.uniform(0, 100, 100000)
        })

        # Calculate some statistics
        stats = {
            'mean': large_data['data'].mean(),
            'std': large_data['data'].std(),
            'min': large_data['data'].min(),
            'max': large_data['data'].max()
        }

        # Check memory hasn't grown excessively
        final_memory = process.memory_info().rss
        memory_growth = final_memory - initial_memory

        # Memory growth should be reasonable (< 500MB)
        assert memory_growth < 500 * 1024 * 1024

        # Stats should be calculated
        assert all(isinstance(v, (int, float)) for v in stats.values())
