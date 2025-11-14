"""
Comprehensive unit tests for vectorized trading operations.

Tests cover:
- Numba JIT compilation
- AVX-512 optimization detection
- Parallel processing
- Hardware-specific optimizations
- Performance benchmarks
- Error handling
"""

import numpy as np
import pandas as pd
import pytest
from unittest.mock import patch, MagicMock

from src.utils.vectorized_ops import (
    VectorizedTradingOps,
    HardwareDetector,
    SYSTEM_INFO,
    OPTIMAL_THREADS,
    AVX512_SUPPORTED,
    CUDA_SUPPORTED,
    benchmark_all_implementations,
    calculate_sma_cuda,
    calculate_ema_cuda
)


class TestHardwareDetector:
    """Test hardware detection capabilities."""

    def test_detect_avx512_support(self):
        """Test AVX-512 detection."""
        result = HardwareDetector.detect_avx512_support()
        assert isinstance(result, bool)

    def test_get_optimal_num_threads(self):
        """Test optimal thread detection."""
        threads = HardwareDetector.get_optimal_num_threads()
        assert threads > 0
        assert isinstance(threads, int)

    def test_detect_cuda_support(self):
        """Test CUDA support detection."""
        result = HardwareDetector.detect_cuda_support()
        assert isinstance(result, bool)

    def test_get_system_info(self):
        """Test system info retrieval."""
        info = HardwareDetector.get_system_info()
        required_keys = ['platform', 'cpu_count', 'cpu_count_physical', 'memory_gb', 'avx512_supported', 'cuda_supported']
        for key in required_keys:
            assert key in info


class TestVectorizedSMA:
    """Test Simple Moving Average calculations."""

    def test_calculate_sma_vectorized_basic(self):
        """Test basic SMA calculation."""
        prices = pd.Series([100, 101, 102, 103, 104, 105])
        sma = VectorizedTradingOps.calculate_sma_vectorized(prices, window=3)

        # Check that SMA values are calculated correctly
        expected = pd.Series([np.nan, np.nan, 101.0, 102.0, 103.0, 104.0])
        pd.testing.assert_series_equal(sma, expected, check_names=False)

    def test_calculate_sma_vectorized_edge_cases(self):
        """Test SMA with edge cases."""
        # Empty series
        empty_prices = pd.Series([], dtype=float)
        empty_sma = VectorizedTradingOps.calculate_sma_vectorized(empty_prices, window=3)
        assert len(empty_sma) == 0

        # Window larger than data
        short_prices = pd.Series([100, 101])
        short_sma = VectorizedTradingOps.calculate_sma_vectorized(short_prices, window=5)
        assert short_sma.isna().all()

    @pytest.mark.skipif(not AVX512_SUPPORTED, reason="AVX-512 not supported")
    def test_calculate_sma_numba_avx512(self):
        """Test Numba SMA with AVX-512."""
        prices = np.array([100.0, 101.0, 102.0, 103.0, 104.0, 105.0])
        sma = VectorizedTradingOps.calculate_sma_numba(prices, window=3)

        expected = np.array([np.nan, np.nan, 101.0, 102.0, 103.0, 104.0])
        np.testing.assert_array_equal(sma[2:], expected[2:])  # Skip NaN values

    def test_calculate_sma_numba_parallel(self):
        """Test Numba SMA parallel processing."""
        prices = np.random.rand(1000).astype(np.float64)
        sma = VectorizedTradingOps.calculate_sma_numba(prices, window=20)

        # Verify output shape and basic properties
        assert len(sma) == len(prices)
        assert np.isnan(sma[0]) and np.isnan(sma[19])  # First 19 values should be NaN
        assert not np.isnan(sma[20])  # 20th value should be valid

    @pytest.mark.skipif(not CUDA_SUPPORTED, reason="CUDA not supported")
    def test_calculate_sma_cuda(self):
        """Test CUDA SMA calculation."""
        prices = np.array([100.0, 101.0, 102.0, 103.0, 104.0, 105.0])
        sma = calculate_sma_cuda(prices, window=3)

        # CUDA implementation should produce similar results
        assert len(sma) == len(prices)


class TestVectorizedEMA:
    """Test Exponential Moving Average calculations."""

    def test_calculate_ema_vectorized_basic(self):
        """Test basic EMA calculation."""
        prices = pd.Series([100, 101, 102, 103, 104, 105])
        ema = VectorizedTradingOps.calculate_ema_vectorized(prices, span=3)

        # Verify EMA is calculated and decreasing volatility
        assert len(ema) == len(prices)
        assert not ema.isna().any()

    def test_calculate_ema_vectorized_convergence(self):
        """Test EMA convergence behavior."""
        # Constant prices should converge to the constant value
        constant_prices = pd.Series([100] * 50)
        ema = VectorizedTradingOps.calculate_ema_vectorized(constant_prices, span=10)

        # EMA should converge to the constant value
        final_values = ema.tail(10)
        assert abs(final_values.mean() - 100) < 0.1

    @pytest.mark.skipif(not AVX512_SUPPORTED, reason="AVX-512 not supported")
    def test_calculate_ema_numba_avx512(self):
        """Test Numba EMA with AVX-512."""
        prices = np.array([100.0, 101.0, 102.0, 103.0, 104.0, 105.0])
        ema = VectorizedTradingOps.calculate_ema_numba(prices, span=3)

        assert len(ema) == len(prices)
        assert not np.isnan(ema).any()
        # EMA should be smoothing the data
        assert ema[0] == 100.0  # First value unchanged
        assert abs(ema[-1] - prices[-1]) < abs(ema[0] - prices[0])  # More smoothing at the end


class TestVectorizedRSI:
    """Test Relative Strength Index calculations."""

    def test_calculate_rsi_vectorized_basic(self):
        """Test basic RSI calculation."""
        # Create trending data
        prices = pd.Series(list(range(100, 120)) + list(range(120, 100, -1)))
        rsi = VectorizedTradingOps.calculate_rsi_vectorized(prices, period=14)

        assert len(rsi) == len(prices)
        # RSI should be between 0 and 100
        valid_rsi = rsi.dropna()
        assert (valid_rsi >= 0).all() and (valid_rsi <= 100).all()

    def test_calculate_rsi_vectorized_extremes(self):
        """Test RSI with extreme price movements."""
        # Strong uptrend followed by downtrend
        uptrend = list(range(100, 150))
        downtrend = list(range(150, 50, -1))
        prices = pd.Series(uptrend + downtrend)

        rsi = VectorizedTradingOps.calculate_rsi_vectorized(prices, period=14)
        valid_rsi = rsi.dropna()

        # Should see high RSI in uptrend and low RSI in downtrend
        uptrend_rsi = valid_rsi[:len(uptrend)]
        downtrend_rsi = valid_rsi[len(uptrend):]

        assert uptrend_rsi.mean() > 50  # Generally higher in uptrend
        assert downtrend_rsi.mean() < 50  # Generally lower in downtrend

    @pytest.mark.skipif(not AVX512_SUPPORTED, reason="AVX-512 not supported")
    def test_calculate_rsi_numba_performance(self):
        """Test Numba RSI performance."""
        prices = np.random.rand(10000) * 100 + 100  # 10k data points
        rsi = VectorizedTradingOps.calculate_rsi_numba(prices, period=14)

        assert len(rsi) == len(prices)
        valid_rsi = rsi[~np.isnan(rsi)]
        assert (valid_rsi >= 0).all() and (valid_rsi <= 100).all()


class TestVectorizedMACD:
    """Test MACD calculations."""

    def test_calculate_macd_vectorized_basic(self):
        """Test basic MACD calculation."""
        prices = pd.Series(np.sin(np.linspace(0, 4*np.pi, 200)) * 10 + 100)
        macd, signal, histogram = VectorizedTradingOps.calculate_macd_vectorized(prices)

        assert len(macd) == len(prices)
        assert len(signal) == len(prices)
        assert len(histogram) == len(prices)

        # MACD and signal should be related
        assert not macd.isna().any()
        assert not signal.isna().any()
        assert not histogram.isna().any()

    def test_calculate_macd_vectorized_crossovers(self):
        """Test MACD crossover detection."""
        # Create data that should produce crossovers
        prices = pd.Series(list(range(100, 150)) + list(range(150, 100, -1)))
        macd, signal, histogram = VectorizedTradingOps.calculate_macd_vectorized(prices)

        # Should have some crossovers (histogram changes sign)
        sign_changes = (histogram > 0) != (histogram.shift(1) > 0)
        sign_changes = sign_changes & histogram.notna() & histogram.shift(1).notna()

        # Should detect at least some crossovers in this data
        assert sign_changes.sum() > 0

    @pytest.mark.skipif(not AVX512_SUPPORTED, reason="AVX-512 not supported")
    def test_calculate_macd_numba_avx512(self):
        """Test Numba MACD with AVX-512."""
        prices = np.sin(np.linspace(0, 4*np.pi, 1000)) * 10 + 100
        macd, signal, histogram = VectorizedTradingOps.calculate_macd_numba(prices)

        assert len(macd) == len(prices)
        assert len(signal) == len(prices)
        assert len(histogram) == len(prices)

        # Histogram should be MACD - signal
        np.testing.assert_array_almost_equal(histogram, macd - signal, decimal=10)


class TestVectorizedBollingerBands:
    """Test Bollinger Bands calculations."""

    def test_calculate_bollinger_bands_vectorized(self):
        """Test Bollinger Bands calculation."""
        # Create data with known volatility
        base_prices = np.linspace(100, 110, 100)
        noise = np.random.normal(0, 2, 100)
        prices = pd.Series(base_prices + noise)

        upper, middle, lower = VectorizedTradingOps.calculate_bollinger_bands_vectorized(
            prices, window=20, num_std=2
        )

        assert len(upper) == len(prices)
        assert len(middle) == len(prices)
        assert len(lower) == len(prices)

        # Upper should be above middle, middle above lower
        assert (upper >= middle).all()
        assert (middle >= lower).all()

        # For stable data, bands should be reasonable
        band_width = (upper - lower) / middle
        assert band_width.mean() > 0  # Some volatility


class TestBatchProcessing:
    """Test batch processing capabilities."""

    def test_batch_indicator_calculation_numba(self):
        """Test batch indicator calculation."""
        # Create batch data (3 symbols, 100 periods)
        np.random.seed(42)
        batch_data = np.random.normal(100, 5, (3, 100))
        indicators = np.array([0, 1, 2])  # SMA, EMA, RSI

        results = VectorizedTradingOps.batch_indicator_calculation_numba(batch_data, indicators)

        assert results.shape == (3, 100, 3)  # (symbols, periods, indicators)

        # Each indicator result should be valid
        for symbol_idx in range(3):
            for ind_idx in range(3):
                indicator_data = results[symbol_idx, :, ind_idx]
                assert len(indicator_data) == 100
                # Should have some valid (non-NaN) values
                assert not np.isnan(indicator_data).all()

    def test_calculate_indicators_optimal(self):
        """Test optimal indicator calculation selection."""
        prices = pd.Series(np.random.rand(1000) * 100 + 100)

        indicators = ['sma_20', 'ema_12', 'rsi_14']
        results = VectorizedTradingOps.calculate_indicators_optimal(prices, indicators)

        assert len(results) == len(indicators)
        for indicator in indicators:
            assert indicator in results
            result = results[indicator]
            assert len(result) == len(prices)

    def test_calculate_indicators_optimal_fallback(self):
        """Test fallback when AVX-512 not available."""
        prices = pd.Series(np.random.rand(100) * 100 + 100)

        # Test with small dataset (should not use Numba)
        indicators = ['sma_20']
        results = VectorizedTradingOps.calculate_indicators_optimal(prices, indicators)

        assert 'sma_20' in results
        assert len(results['sma_20']) == len(prices)


class TestMemoryOptimization:
    """Test memory optimization features."""

    def test_optimize_dataframe_memory(self):
        """Test DataFrame memory optimization."""
        # Create DataFrame with suboptimal dtypes
        df = pd.DataFrame({
            'float_col': [1.0, 2.0, 3.0],
            'int_col': [1, 2, 3],
            'category_col': ['A', 'B', 'A'] * 10
        })

        # Force suboptimal dtypes
        df['float_col'] = df['float_col'].astype('float64')
        df['int_col'] = df['int_col'].astype('int64')

        original_memory = df.memory_usage(deep=True).sum()
        optimized_df = VectorizedTradingOps.optimize_dataframe_memory(df)
        optimized_memory = optimized_df.memory_usage(deep=True).sum()

        # Should reduce memory usage
        assert optimized_memory <= original_memory

        # Data should be preserved
        pd.testing.assert_frame_equal(df, optimized_df, check_dtype=False)


class TestPerformanceBenchmarks:
    """Test performance benchmarking."""

    def test_benchmark_all_implementations(self):
        """Test comprehensive performance benchmarking."""
            results = benchmark_all_implementations()

        # Should return comprehensive results
        required_keys = [
                'sma_numba_time', 'sma_pandas_time', 'sma_numba_speedup',
                'ema_numba_time', 'ema_pandas_time', 'ema_numba_speedup',
                'rsi_numba_time', 'rsi_pandas_time', 'rsi_numba_speedup',
                'macd_numba_time', 'macd_pandas_time', 'macd_numba_speedup',
            'batch_processing_time', 'overall_speedup', 'data_points'
            ]

        for key in required_keys:
                assert key in results

        # Speedups should be positive
        assert results['sma_numba_speedup'] > 0
        assert results['ema_numba_speedup'] > 0
        assert results['rsi_numba_speedup'] > 0
        assert results['macd_numba_speedup'] > 0

    def test_benchmark_with_mock_hardware(self):
        """Test benchmarking with mocked hardware detection."""
        with patch('src.utils.vectorized_ops.AVX512_SUPPORTED', True):
            results = benchmark_all_implementations()
            assert 'sma_numba_time' in results


class TestErrorHandling:
    """Test error handling and edge cases."""

    def test_empty_data_handling(self):
        """Test handling of empty data."""
        empty_prices = np.array([])
        sma = VectorizedTradingOps.calculate_sma_numba(empty_prices, window=3)
        assert len(sma) == 0

    def test_invalid_window_sizes(self):
        """Test invalid window sizes."""
        prices = np.array([100.0, 101.0, 102.0])

        # Window larger than data
        sma = VectorizedTradingOps.calculate_sma_numba(prices, window=10)
        assert len(sma) == len(prices)
        assert np.isnan(sma).all()

    def test_nan_handling(self):
        """Test NaN value handling."""
        prices = np.array([100.0, np.nan, 102.0, 103.0])
        sma = VectorizedTradingOps.calculate_sma_numba(prices, window=2)

        # Should handle NaN gracefully
        assert len(sma) == len(prices)
        # First value should be NaN, others should be calculated where possible


class TestSystemIntegration:
    """Test system-level integration."""

    def test_system_info_integration(self):
        """Test system info is properly integrated."""
        assert isinstance(SYSTEM_INFO, dict)
        assert 'platform' in SYSTEM_INFO
        assert 'cpu_count' in SYSTEM_INFO

    def test_optimal_threads_calculation(self):
        """Test optimal thread calculation."""
        assert OPTIMAL_THREADS > 0
        assert isinstance(OPTIMAL_THREADS, int)

    def test_hardware_flags_consistency(self):
        """Test hardware detection flags are consistent."""
        assert isinstance(AVX512_SUPPORTED, bool)
        assert isinstance(CUDA_SUPPORTED, bool)


if __name__ == "__main__":
    pytest.main([__file__])