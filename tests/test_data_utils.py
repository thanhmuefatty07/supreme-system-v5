#!/usr/bin/env python3
"""
Tests for Supreme System V5 data utilities.

Tests data processing, validation, and optimization utilities.
"""

import pytest
import pandas as pd
import numpy as np
from unittest.mock import Mock, patch
from datetime import datetime, timedelta

# Import data utilities
try:
    from src.utils.data_utils import (
        optimize_dataframe_memory,
        chunk_dataframe,
        validate_ohlcv_data,
        calculate_returns,
        detect_outliers,
        resample_ohlcv,
        align_dataframes,
        calculate_rolling_stats,
        find_data_gaps,
        fill_missing_data
    )
except ImportError:
    # Fallback for testing
    optimize_dataframe_memory = None
    chunk_dataframe = None
    validate_ohlcv_data = None
    calculate_returns = None
    detect_outliers = None
    resample_ohlcv = None
    align_dataframes = None
    calculate_rolling_stats = None
    find_data_gaps = None
    fill_missing_data = None


@pytest.fixture
def sample_ohlcv_data():
    """Generate sample OHLCV data for testing."""
    np.random.seed(42)

    # Generate 1000 periods of data
    n_periods = 1000
    dates = pd.date_range('2024-01-01', periods=n_periods, freq='1H')

    # Generate realistic price data
    base_price = 50000
    returns = np.random.normal(0.0001, 0.01, n_periods)  # Small mean return, 1% volatility
    prices = base_price * np.cumprod(1 + returns)

    # Create OHLCV data
    data = pd.DataFrame({
        'timestamp': dates,
        'open': prices,
        'high': prices * np.random.uniform(1.001, 1.01, n_periods),
        'low': prices * np.random.uniform(0.99, 0.999, n_periods),
        'close': prices * (1 + np.random.normal(0, 0.005, n_periods)),
        'volume': np.random.uniform(1000, 10000, n_periods)
    })

    # Ensure OHLC relationships are correct
    data['high'] = np.maximum(data[['open', 'close']].max(axis=1), data['high'])
    data['low'] = np.minimum(data[['open', 'close']].min(axis=1), data['low'])

    return data


@pytest.fixture
def large_dataframe():
    """Generate a large dataframe for performance testing."""
    np.random.seed(123)
    n_rows = 100000

    data = pd.DataFrame({
        'timestamp': pd.date_range('2024-01-01', periods=n_rows, freq='1min'),
        'price': np.random.uniform(40000, 60000, n_rows),
        'volume': np.random.uniform(100, 10000, n_rows),
        'category': np.random.choice(['A', 'B', 'C'], n_rows)
    })

    return data


@pytest.mark.skipif(optimize_dataframe_memory is None, reason="Data utilities not available")
class TestDataOptimization:
    """Test data optimization utilities."""

    def test_optimize_dataframe_memory_numeric(self, large_dataframe):
        """Test memory optimization for numeric columns."""
        original_memory = large_dataframe.memory_usage(deep=True).sum()

        # Optimize dataframe
        optimized_df = optimize_dataframe_memory(large_dataframe)

        optimized_memory = optimized_df.memory_usage(deep=True).sum()

        # Should reduce memory usage
        assert optimized_memory <= original_memory

        # Data should remain the same
        pd.testing.assert_frame_equal(large_dataframe.reset_index(drop=True),
                                    optimized_df.reset_index(drop=True))

    def test_optimize_dataframe_memory_categorical(self, large_dataframe):
        """Test memory optimization for categorical columns."""
        # Add categorical column
        df_with_cat = large_dataframe.copy()
        df_with_cat['category'] = df_with_cat['category'].astype('category')

        original_memory = df_with_cat.memory_usage(deep=True).sum()

        optimized_df = optimize_dataframe_memory(df_with_cat)

        optimized_memory = optimized_df.memory_usage(deep=True).sum()

        # Should reduce or maintain memory usage
        assert optimized_memory <= original_memory * 1.1  # Allow small increase

    def test_chunk_dataframe_basic(self, large_dataframe):
        """Test basic dataframe chunking."""
        chunk_size = 10000
        chunks = list(chunk_dataframe(large_dataframe, chunk_size))

        # Should create correct number of chunks
        expected_chunks = len(large_dataframe) // chunk_size + 1
        assert len(chunks) == expected_chunks

        # Each chunk should be <= chunk_size
        for chunk in chunks:
            assert len(chunk) <= chunk_size

        # All data should be preserved
        reconstructed = pd.concat(chunks, ignore_index=True)
        pd.testing.assert_frame_equal(large_dataframe, reconstructed)

    def test_chunk_dataframe_edge_cases(self):
        """Test chunking edge cases."""
        # Empty dataframe
        empty_df = pd.DataFrame()
        chunks = list(chunk_dataframe(empty_df, 100))
        assert len(chunks) == 0

        # Small dataframe
        small_df = pd.DataFrame({'a': [1, 2, 3]})
        chunks = list(chunk_dataframe(small_df, 10))
        assert len(chunks) == 1
        pd.testing.assert_frame_equal(chunks[0], small_df)


@pytest.mark.skipif(validate_ohlcv_data is None, reason="Data validation not available")
class TestDataValidation:
    """Test data validation utilities."""

    def test_validate_ohlcv_data_valid(self, sample_ohlcv_data):
        """Test validation of valid OHLCV data."""
        is_valid, errors = validate_ohlcv_data(sample_ohlcv_data)

        assert is_valid
        assert len(errors) == 0

    def test_validate_ohlcv_data_missing_columns(self):
        """Test validation with missing required columns."""
        invalid_data = pd.DataFrame({
            'timestamp': pd.date_range('2024-01-01', periods=10),
            'price': np.random.uniform(50000, 51000, 10)
        })

        is_valid, errors = validate_ohlcv_data(invalid_data)

        assert not is_valid
        assert len(errors) > 0
        assert any('missing' in error.lower() for error in errors)

    def test_validate_ohlcv_data_ohlc_relationships(self, sample_ohlcv_data):
        """Test OHLC relationship validation."""
        # Create invalid OHLC data where high < close
        invalid_data = sample_ohlcv_data.copy()
        invalid_data.loc[0, 'high'] = invalid_data.loc[0, 'close'] * 0.9  # High < close

        is_valid, errors = validate_ohlcv_data(invalid_data)

        assert not is_valid
        assert len(errors) > 0

    def test_validate_ohlcv_data_negative_values(self, sample_ohlcv_data):
        """Test validation of negative price/volume values."""
        invalid_data = sample_ohlcv_data.copy()
        invalid_data.loc[0, 'close'] = -100  # Negative price

        is_valid, errors = validate_ohlcv_data(invalid_data)

        assert not is_valid
        assert len(errors) > 0

    def test_validate_ohlcv_data_nan_values(self, sample_ohlcv_data):
        """Test validation of NaN values."""
        invalid_data = sample_ohlcv_data.copy()
        invalid_data.loc[0, 'close'] = np.nan

        is_valid, errors = validate_ohlcv_data(invalid_data)

        assert not is_valid
        assert len(errors) > 0


@pytest.mark.skipif(calculate_returns is None, reason="Return calculation not available")
class TestReturnCalculations:
    """Test return calculation utilities."""

    def test_calculate_returns_simple(self, sample_ohlcv_data):
        """Test simple return calculation."""
        returns = calculate_returns(sample_ohlcv_data['close'])

        assert len(returns) == len(sample_ohlcv_data) - 1  # One less than input
        assert not returns.isna().any()

        # Check first return calculation
        expected_first_return = (sample_ohlcv_data['close'].iloc[1] /
                                sample_ohlcv_data['close'].iloc[0] - 1)
        assert abs(returns.iloc[0] - expected_first_return) < 1e-10

    def test_calculate_returns_log(self, sample_ohlcv_data):
        """Test logarithmic return calculation."""
        log_returns = calculate_returns(sample_ohlcv_data['close'], method='log')

        assert len(log_returns) == len(sample_ohlcv_data) - 1
        assert not log_returns.isna().any()

        # Log returns should be approximately equal to simple returns for small changes
        simple_returns = calculate_returns(sample_ohlcv_data['close'], method='simple')
        diff = abs(log_returns - simple_returns)
        assert (diff < 0.01).all()  # Should be very close for small returns

    def test_calculate_returns_percentage(self, sample_ohlcv_data):
        """Test percentage return calculation."""
        pct_returns = calculate_returns(sample_ohlcv_data['close'], method='percentage')

        assert len(pct_returns) == len(sample_ohlcv_data) - 1
        assert not pct_returns.isna().any()

        # Percentage returns should be simple returns * 100
        simple_returns = calculate_returns(sample_ohlcv_data['close'], method='simple')
        expected_pct = simple_returns * 100
        pd.testing.assert_series_equal(pct_returns, expected_pct)


@pytest.mark.skipif(detect_outliers is None, reason="Outlier detection not available")
class TestOutlierDetection:
    """Test outlier detection utilities."""

    def test_detect_outliers_iqr(self, sample_ohlcv_data):
        """Test IQR-based outlier detection."""
        data_with_outliers = sample_ohlcv_data['close'].copy()

        # Add some extreme outliers
        data_with_outliers.iloc[10] = data_with_outliers.mean() * 10  # Extreme high
        data_with_outliers.iloc[20] = data_with_outliers.mean() * 0.1  # Extreme low

        outliers = detect_outliers(data_with_outliers, method='iqr')

        assert isinstance(outliers, pd.Series)
        assert outliers.sum() >= 2  # Should detect at least the added outliers

    def test_detect_outliers_zscore(self, sample_ohlcv_data):
        """Test Z-score-based outlier detection."""
        data_with_outliers = sample_ohlcv_data['close'].copy()

        # Add outliers beyond 3 standard deviations
        mean_val = data_with_outliers.mean()
        std_val = data_with_outliers.std()
        data_with_outliers.iloc[10] = mean_val + 5 * std_val  # 5 SD outlier

        outliers = detect_outliers(data_with_outliers, method='zscore', threshold=3)

        assert isinstance(outliers, pd.Series)
        assert outliers.sum() >= 1  # Should detect the extreme outlier

    def test_detect_outliers_mad(self, sample_ohlcv_data):
        """Test MAD-based outlier detection."""
        data_with_outliers = sample_ohlcv_data['close'].copy()

        # Add outliers using MAD method
        data_with_outliers.iloc[10] = data_with_outliers.median() + 10 * data_with_outliers.mad()

        outliers = detect_outliers(data_with_outliers, method='mad')

        assert isinstance(outliers, pd.Series)
        assert outliers.sum() >= 1


@pytest.mark.skipif(resample_ohlcv is None, reason="Resampling not available")
class TestDataResampling:
    """Test data resampling utilities."""

    def test_resample_ohlcv_hourly_to_daily(self, sample_ohlcv_data):
        """Test resampling from hourly to daily data."""
        # Take first week of data
        weekly_data = sample_ohlcv_data.head(24 * 7).copy()

        resampled = resample_ohlcv(weekly_data, '1D')

        # Should have 7 days
        assert len(resampled) == 7

        # Check OHLC calculations
        for day_data in weekly_data.groupby(weekly_data['timestamp'].dt.date):
            day_date = day_data[0]
            day_prices = day_data[1]

            resampled_row = resampled[resampled['timestamp'].dt.date == day_date]

            if len(resampled_row) > 0:
                assert resampled_row['open'].iloc[0] == day_prices['open'].iloc[0]
                assert resampled_row['close'].iloc[0] == day_prices['close'].iloc[-1]
                assert resampled_row['high'].iloc[0] == day_prices['high'].max()
                assert resampled_row['low'].iloc[0] == day_prices['low'].min()

    def test_resample_ohlcv_volume_aggregation(self, sample_ohlcv_data):
        """Test volume aggregation during resampling."""
        hourly_data = sample_ohlcv_data.head(24).copy()  # One day

        resampled = resample_ohlcv(hourly_data, '1D')

        # Volume should be summed
        expected_volume = hourly_data['volume'].sum()
        assert resampled['volume'].iloc[0] == expected_volume


@pytest.mark.skipif(align_dataframes is None, reason="Data alignment not available")
class TestDataAlignment:
    """Test dataframe alignment utilities."""

    def test_align_dataframes_same_index(self):
        """Test alignment of dataframes with same index."""
        df1 = pd.DataFrame({'A': [1, 2, 3]}, index=[0, 1, 2])
        df2 = pd.DataFrame({'B': [4, 5, 6]}, index=[0, 1, 2])

        aligned1, aligned2 = align_dataframes(df1, df2)

        pd.testing.assert_frame_equal(aligned1, df1)
        pd.testing.assert_frame_equal(aligned2, df2)

    def test_align_dataframes_different_index(self):
        """Test alignment of dataframes with different indices."""
        df1 = pd.DataFrame({'A': [1, 2, 3]}, index=[0, 1, 2])
        df2 = pd.DataFrame({'B': [4, 5, 6]}, index=[1, 2, 3])

        aligned1, aligned2 = align_dataframes(df1, df2)

        # Should align on common index [1, 2]
        assert len(aligned1) == 2
        assert len(aligned2) == 2
        assert aligned1.index.equals(aligned2.index)

    def test_align_dataframes_fill_values(self):
        """Test alignment with custom fill values."""
        df1 = pd.DataFrame({'A': [1, 2]}, index=[0, 1])
        df2 = pd.DataFrame({'B': [4, 5]}, index=[1, 2])

        aligned1, aligned2 = align_dataframes(df1, df2, fill_value=0)

        assert aligned1.loc[2, 'A'] == 0
        assert aligned2.loc[0, 'B'] == 0


@pytest.mark.skipif(calculate_rolling_stats is None, reason="Rolling stats not available")
class TestRollingStatistics:
    """Test rolling statistics calculations."""

    def test_calculate_rolling_stats_basic(self, sample_ohlcv_data):
        """Test basic rolling statistics calculation."""
        prices = sample_ohlcv_data['close']

        rolling_stats = calculate_rolling_stats(prices, window=20)

        assert isinstance(rolling_stats, pd.DataFrame)
        assert 'mean' in rolling_stats.columns
        assert 'std' in rolling_stats.columns
        assert len(rolling_stats) == len(prices)

        # First 19 values should be NaN for 20-period rolling window
        assert rolling_stats['mean'].isna().sum() >= 19

    def test_calculate_rolling_stats_custom_metrics(self, sample_ohlcv_data):
        """Test rolling statistics with custom metrics."""
        prices = sample_ohlcv_data['close']

        rolling_stats = calculate_rolling_stats(
            prices,
            window=10,
            metrics=['mean', 'median', 'skew', 'kurtosis']
        )

        expected_columns = ['mean', 'median', 'skew', 'kurtosis']
        for col in expected_columns:
            assert col in rolling_stats.columns

    def test_calculate_rolling_stats_min_periods(self, sample_ohlcv_data):
        """Test rolling statistics with minimum periods."""
        prices = sample_ohlcv_data['close'].head(50)

        rolling_stats = calculate_rolling_stats(
            prices,
            window=20,
            min_periods=5
        )

        # Should have fewer NaN values with min_periods=5
        nan_count = rolling_stats['mean'].isna().sum()
        assert nan_count < 20  # Less than window size


@pytest.mark.skipif(find_data_gaps is None, reason="Gap detection not available")
class TestGapDetection:
    """Test data gap detection utilities."""

    def test_find_data_gaps_regular_data(self, sample_ohlcv_data):
        """Test gap detection on regular data."""
        gaps = find_data_gaps(sample_ohlcv_data, freq='1H')

        # Regular hourly data should have minimal gaps
        assert isinstance(gaps, pd.DataFrame)
        assert len(gaps) >= 0

    def test_find_data_gaps_with_missing_data(self):
        """Test gap detection with missing timestamps."""
        # Create data with gaps
        dates = pd.date_range('2024-01-01', periods=10, freq='1H')
        # Remove some timestamps to create gaps
        incomplete_dates = dates[[0, 1, 2, 4, 5, 8, 9]]  # Missing 3, 6, 7

        data = pd.DataFrame({
            'timestamp': incomplete_dates,
            'close': np.random.uniform(50000, 51000, len(incomplete_dates))
        })

        gaps = find_data_gaps(data, freq='1H')

        assert len(gaps) > 0
        # Should detect missing periods

    def test_find_data_gaps_large_gaps(self):
        """Test detection of large time gaps."""
        # Create data with a large gap
        early_dates = pd.date_range('2024-01-01', periods=5, freq='1H')
        late_dates = pd.date_range('2024-01-02 12:00:00', periods=5, freq='1H')  # 12 hour gap

        dates = early_dates.append(late_dates)
        data = pd.DataFrame({
            'timestamp': dates,
            'close': np.random.uniform(50000, 51000, len(dates))
        })

        gaps = find_data_gaps(data, freq='1H', max_gap_hours=6)

        assert len(gaps) > 0


@pytest.mark.skipif(fill_missing_data is None, reason="Data filling not available")
class TestMissingDataFilling:
    """Test missing data filling utilities."""

    def test_fill_missing_data_forward_fill(self):
        """Test forward fill data filling."""
        # Create data with missing values
        data = pd.DataFrame({
            'timestamp': pd.date_range('2024-01-01', periods=10, freq='1H'),
            'close': [100, np.nan, np.nan, 103, np.nan, 105, np.nan, np.nan, np.nan, 109]
        })

        filled_data = fill_missing_data(data, method='ffill')

        # Should fill forward
        assert filled_data.loc[1, 'close'] == 100  # Filled from previous
        assert filled_data.loc[2, 'close'] == 100  # Filled from previous
        assert filled_data.loc[4, 'close'] == 103  # Filled from previous

    def test_fill_missing_data_interpolation(self):
        """Test interpolation data filling."""
        data = pd.DataFrame({
            'timestamp': pd.date_range('2024-01-01', periods=5, freq='1H'),
            'close': [100, np.nan, np.nan, np.nan, 110]
        })

        filled_data = fill_missing_data(data, method='interpolate')

        # Should interpolate linearly
        assert not filled_data['close'].isna().any()
        assert filled_data.loc[0, 'close'] == 100
        assert filled_data.loc[4, 'close'] == 110

        # Check interpolated values are monotonic
        assert filled_data['close'].is_monotonic_increasing

    def test_fill_missing_data_limit(self):
        """Test filling with gap limits."""
        data = pd.DataFrame({
            'timestamp': pd.date_range('2024-01-01', periods=10, freq='1H'),
            'close': [100, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, 110]
        })

        # Limit to 3 consecutive fills
        filled_data = fill_missing_data(data, method='interpolate', limit=3)

        # Should only fill first 3 NaN values
        assert filled_data.loc[1, 'close'] != 100  # First NaN filled
        assert filled_data.loc[2, 'close'] != 100  # Second NaN filled
        assert filled_data.loc[3, 'close'] != 100  # Third NaN filled
        assert np.isnan(filled_data.loc[4, 'close'])  # Fourth NaN not filled (limit exceeded)


class TestPerformanceOptimization:
    """Test performance optimization of data utilities."""

    def test_memory_optimization_performance(self, large_dataframe):
        """Test that memory optimization doesn't take excessive time."""
        import time

        start_time = time.time()
        optimized_df = optimize_dataframe_memory(large_dataframe)
        end_time = time.time()

        optimization_time = end_time - start_time

        # Should complete in reasonable time (< 1 second for 100k rows)
        assert optimization_time < 1.0, f"Optimization took {optimization_time}s"

    def test_chunking_performance(self, large_dataframe):
        """Test chunking performance on large dataframes."""
        import time

        chunk_size = 10000
        start_time = time.time()

        chunks = list(chunk_dataframe(large_dataframe, chunk_size))

        end_time = time.time()
        chunking_time = end_time - start_time

        # Should chunk quickly
        assert chunking_time < 0.5, f"Chunking took {chunking_time}s"
        assert len(chunks) > 0

    def test_validation_performance(self, large_dataframe):
        """Test validation performance on large datasets."""
        import time

        # Add required columns for OHLCV validation
        ohlcv_data = large_dataframe.copy()
        ohlcv_data['timestamp'] = pd.date_range('2024-01-01', periods=len(ohlcv_data), freq='1min')
        ohlcv_data['open'] = ohlcv_data['price']
        ohlcv_data['high'] = ohlcv_data['price'] * 1.01
        ohlcv_data['low'] = ohlcv_data['price'] * 0.99
        ohlcv_data['close'] = ohlcv_data['price']
        ohlcv_data['volume'] = ohlcv_data['volume']

        start_time = time.time()
        is_valid, errors = validate_ohlcv_data(ohlcv_data)
        end_time = time.time()

        validation_time = end_time - start_time

        # Should validate quickly even for large datasets
        assert validation_time < 2.0, f"Validation took {validation_time}s"
