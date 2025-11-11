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
        validate_and_clean_data,
        calculate_returns,
        detect_outliers,
        resample_ohlcv,
        calculate_technical_indicators,
        handle_missing_data,
        calculate_correlation_matrix,
        find_highly_correlated_pairs,
        calculate_drawdowns,
        calculate_roll_max_drawdown,
        split_data_by_date,
        calculate_beta,
        calculate_alpha
    )
except ImportError:
    # Fallback for testing
    optimize_dataframe_memory = None
    chunk_dataframe = None
    validate_and_clean_data = None
    calculate_returns = None
    detect_outliers = None
    resample_ohlcv = None
    calculate_technical_indicators = None
    handle_missing_data = None
    calculate_correlation_matrix = None
    find_highly_correlated_pairs = None
    calculate_drawdowns = None
    calculate_roll_max_drawdown = None
    split_data_by_date = None
    calculate_beta = None
    calculate_alpha = None


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

        # Data values should remain approximately the same (allowing dtype changes)
        for col in large_dataframe.columns:
            if col == 'volume':
                # Volume conversion to int may truncate floats, check approximate equality
                original_vals = large_dataframe[col].reset_index(drop=True)
                optimized_vals = optimized_df[col].reset_index(drop=True).astype(float)
                np.testing.assert_allclose(original_vals, optimized_vals, rtol=1e-10, atol=1e-10)
            elif pd.api.types.is_numeric_dtype(large_dataframe[col]):
                pd.testing.assert_series_equal(
                    large_dataframe[col].reset_index(drop=True),
                    optimized_df[col].reset_index(drop=True),
                    check_dtype=False,  # Allow dtype changes
                    check_names=False
                )
            else:
                pd.testing.assert_series_equal(
                    large_dataframe[col].reset_index(drop=True),
                    optimized_df[col].reset_index(drop=True),
                    check_dtype=False
                )

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


@pytest.mark.skipif(validate_and_clean_data is None, reason="Data validation not available")
class TestDataValidation:
    """Test data validation utilities."""

    def test_validate_and_clean_data_valid(self, sample_ohlcv_data):
        """Test validation of valid OHLCV data."""
        is_valid, errors = validate_and_clean_data(sample_ohlcv_data)

        assert is_valid
        assert len(errors) == 0

    def test_validate_and_clean_data_missing_columns(self):
        """Test validation with missing required columns."""
        invalid_data = pd.DataFrame({
            'timestamp': pd.date_range('2024-01-01', periods=10),
            'price': np.random.uniform(50000, 51000, 10)
        })

        is_valid, errors = validate_and_clean_data(invalid_data)

        assert not is_valid
        assert len(errors) > 0
        assert any('missing' in error.lower() for error in errors)

    def test_validate_and_clean_data_ohlc_relationships(self, sample_ohlcv_data):
        """Test OHLC relationship validation."""
        # Create invalid OHLC data where high < close
        invalid_data = sample_ohlcv_data.copy()
        invalid_data.loc[0, 'high'] = invalid_data.loc[0, 'close'] * 0.9  # High < close

        is_valid, errors = validate_and_clean_data(invalid_data)

        assert not is_valid
        assert len(errors) > 0

    def test_validate_and_clean_data_negative_values(self, sample_ohlcv_data):
        """Test validation of negative price/volume values."""
        invalid_data = sample_ohlcv_data.copy()
        invalid_data.loc[0, 'close'] = -100  # Negative price

        is_valid, errors = validate_and_clean_data(invalid_data)

        assert not is_valid
        assert len(errors) > 0

    def test_validate_and_clean_data_nan_values(self, sample_ohlcv_data):
        """Test validation of NaN values."""
        invalid_data = sample_ohlcv_data.copy()
        invalid_data.loc[0, 'close'] = np.nan

        is_valid, errors = validate_and_clean_data(invalid_data)

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


@pytest.mark.skipif(calculate_correlation_matrix is None, reason="Correlation analysis not available")
class TestCorrelationAnalysis:
    """Test correlation analysis utilities."""

    def test_calculate_correlation_matrix_basic(self, sample_ohlcv_data):
        """Test basic correlation matrix calculation."""
        # Add some correlated columns
        data = sample_ohlcv_data.copy()
        data['close_shifted'] = data['close'].shift(1)
        data['close_squared'] = data['close'] ** 2

        corr_matrix = calculate_correlation_matrix(data, columns=['close', 'volume', 'close_shifted', 'close_squared'])

        assert isinstance(corr_matrix, pd.DataFrame)
        assert corr_matrix.shape[0] == 4
        assert corr_matrix.shape[1] == 4

        # Self-correlation should be 1
        assert abs(corr_matrix.loc['close', 'close'] - 1.0) < 1e-10

    def test_find_highly_correlated_pairs(self, sample_ohlcv_data):
        """Test finding highly correlated pairs."""
        # Add highly correlated columns
        data = sample_ohlcv_data.copy()
        data['close_correlated'] = data['close'] * 1.01  # 99% correlation
        data['close_uncorrelated'] = np.random.randn(len(data))  # Random

        pairs = find_highly_correlated_pairs(data, threshold=0.95)

        assert isinstance(pairs, list)

        # Should find the highly correlated pair
        close_pairs = [pair for pair in pairs if 'close' in str(pair) and 'close_correlated' in str(pair)]
        assert len(close_pairs) > 0

    def test_calculate_beta(self, sample_ohlcv_data):
        """Test beta calculation."""
        asset_returns = sample_ohlcv_data['close'].pct_change().dropna()
        market_returns = asset_returns * 0.8 + np.random.normal(0, 0.01, len(asset_returns))  # Similar but not identical

        beta = calculate_beta(asset_returns, market_returns)

        assert isinstance(beta, float)
        # Beta should be close to 1 since returns are similar
        assert 0.5 < beta < 1.5

    def test_calculate_alpha(self, sample_ohlcv_data):
        """Test alpha calculation."""
        asset_returns = sample_ohlcv_data['close'].pct_change().dropna()
        market_returns = asset_returns * 0.8 + np.random.normal(0, 0.01, len(asset_returns))

        alpha = calculate_alpha(asset_returns, market_returns, risk_free_rate=0.02)

        assert isinstance(alpha, float)
        # Alpha can be positive or negative
        assert -1.0 < alpha < 1.0


@pytest.mark.skipif(calculate_drawdowns is None, reason="Drawdown analysis not available")
class TestDrawdownAnalysis:
    """Test drawdown analysis utilities."""

    def test_calculate_drawdowns(self, sample_ohlcv_data):
        """Test drawdown calculation."""
        price_series = sample_ohlcv_data['close']

        drawdowns = calculate_drawdowns(price_series)

        assert isinstance(drawdowns, pd.Series)
        assert len(drawdowns) == len(price_series)

        # Drawdowns should be <= 0 (negative or zero)
        assert (drawdowns <= 0).all()

        # Maximum drawdown should be the most negative value
        max_dd = drawdowns.min()
        assert max_dd <= 0

    def test_calculate_roll_max_drawdown(self, sample_ohlcv_data):
        """Test rolling maximum drawdown calculation."""
        price_series = sample_ohlcv_data['close']

        roll_max_dd = calculate_roll_max_drawdown(price_series, window=50)

        assert isinstance(roll_max_dd, pd.Series)
        assert len(roll_max_dd) == len(price_series)

        # Rolling max drawdown should be <= 0
        assert (roll_max_dd <= 0).all()

    def test_split_data_by_date(self, sample_ohlcv_data):
        """Test data splitting by date."""
        split_date = sample_ohlcv_data['timestamp'].iloc[len(sample_ohlcv_data) // 2]

        train_data, test_data = split_data_by_date(sample_ohlcv_data, split_date)

        assert isinstance(train_data, pd.DataFrame)
        assert isinstance(test_data, pd.DataFrame)

        # Training data should end before split date
        assert train_data['timestamp'].max() < split_date

        # Test data should start at or after split date
        assert test_data['timestamp'].min() >= split_date


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
        is_valid, errors = validate_and_clean_data(ohlcv_data)
        end_time = time.time()

        validation_time = end_time - start_time

        # Should validate quickly even for large datasets
        assert validation_time < 2.0, f"Validation took {validation_time}s"
