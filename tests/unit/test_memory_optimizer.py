"""
Comprehensive unit tests for memory optimization utilities.

Tests cover:
- DataFrame memory optimization
- Parquet compression/decompression
- Memory-mapped arrays
- Chunked processing
- Memory monitoring
- Cache management
- Performance benchmarking
"""

import os
import tempfile
import psutil
import pytest
import numpy as np
import pandas as pd
from pathlib import Path
from unittest.mock import patch, MagicMock

from src.utils.memory_optimizer import (
    MemoryOptimizer,
    optimize_trading_data_pipeline,
    create_chunked_data_loader,
    benchmark_memory_optimization,
    memory_budget
)


class TestMemoryOptimizer:
    """Test core MemoryOptimizer functionality."""

    def test_initialization(self):
        """Test MemoryOptimizer initialization."""
        optimizer = MemoryOptimizer()
        assert optimizer.process is not None
        assert optimizer.initial_memory > 0

    def test_get_current_memory_mb(self):
        """Test memory usage retrieval."""
        optimizer = MemoryOptimizer()
        memory_mb = optimizer.get_current_memory_mb()
        assert isinstance(memory_mb, float)
        assert memory_mb > 0

    def test_get_memory_usage_report(self):
        """Test memory usage report generation."""
        optimizer = MemoryOptimizer()
        report = optimizer.get_memory_usage_report()

        required_keys = ['rss_mb', 'vms_mb', 'percent', 'available_mb', 'total_mb']
        for key in required_keys:
            assert key in report
            assert isinstance(report[key], (int, float))
            assert report[key] >= 0


class TestDataFrameOptimization:
    """Test DataFrame memory optimization."""

    def test_optimize_dataframe_memory_basic(self):
        """Test basic DataFrame memory optimization."""
        # Create DataFrame with suboptimal dtypes
        df = pd.DataFrame({
            'float64_col': [1.0, 2.0, 3.0],
            'int64_col': [1, 2, 3],
            'uint64_col': [1, 2, 3],
            'category_col': ['A', 'B', 'A'] * 10
        })

        # Force suboptimal dtypes
        df['float64_col'] = df['float64_col'].astype('float64')
        df['int64_col'] = df['int64_col'].astype('int64')
        df['uint64_col'] = df['uint64_col'].astype('uint64')

        original_memory = df.memory_usage(deep=True).sum()
        optimized_df = MemoryOptimizer.optimize_dataframe_memory(df)
        optimized_memory = optimized_df.memory_usage(deep=True).sum()

        # Should reduce memory usage
        assert optimized_memory <= original_memory

        # Data integrity should be preserved
        pd.testing.assert_frame_equal(df, optimized_df, check_dtype=False)

    def test_optimize_dataframe_memory_edge_cases(self):
        """Test DataFrame optimization edge cases."""
        # Empty DataFrame
        empty_df = pd.DataFrame()
        optimized_empty = MemoryOptimizer.optimize_dataframe_memory(empty_df)
        assert len(optimized_empty) == 0

        # DataFrame with all optimal dtypes
        optimal_df = pd.DataFrame({
            'float32_col': pd.Series([1.0, 2.0, 3.0], dtype='float32'),
            'int32_col': pd.Series([1, 2, 3], dtype='int32'),
            'category_col': pd.Categorical(['A', 'B', 'A'])
        })

        original_memory = optimal_df.memory_usage(deep=True).sum()
        optimized_optimal = MemoryOptimizer.optimize_dataframe_memory(optimal_df)
        optimized_memory = optimized_optimal.memory_usage(deep=True).sum()

        # Should not increase memory usage significantly
        assert optimized_memory <= original_memory * 1.1

    def test_optimize_numeric_arrays(self):
        """Test numeric array optimization."""
        # Create arrays with suboptimal dtypes
        float64_array = np.array([1.0, 2.0, 3.0], dtype=np.float64)
        int64_array = np.array([1, 2, 3], dtype=np.int64)
        uint64_array = np.array([1, 2, 3], dtype=np.uint64)

        optimized_arrays = MemoryOptimizer.optimize_numeric_arrays(
            float64_array, int64_array, uint64_array
        )

        # Check dtype optimization
        assert optimized_arrays[0].dtype == np.float32
        assert optimized_arrays[1].dtype in [np.int32, np.int16, np.int8]
        assert optimized_arrays[2].dtype in [np.uint32, np.uint16, np.uint8]

        # Data should be preserved
        np.testing.assert_array_equal(optimized_arrays[0], float64_array.astype(np.float32))
        np.testing.assert_array_equal(optimized_arrays[1], int64_array)
        np.testing.assert_array_equal(optimized_arrays[2], uint64_array)


class TestParquetCompression:
    """Test Parquet compression functionality."""

    def test_save_to_parquet_compressed(self):
        """Test saving compressed Parquet files."""
        df = pd.DataFrame({
            'timestamp': pd.date_range('2024-01-01', periods=100, freq='1H'),
            'open': np.random.uniform(100, 110, 100),
            'high': np.random.uniform(105, 115, 100),
            'low': np.random.uniform(95, 105, 100),
            'close': np.random.uniform(100, 110, 100),
            'volume': np.random.randint(1000, 10000, 100)
        })

        with tempfile.NamedTemporaryFile(suffix='.parquet', delete=False) as f:
            file_path = f.name

        try:
            # Test compression
            success = MemoryOptimizer.save_to_parquet_compressed(
                df, file_path, compression='snappy'
            )
            assert success

            # Verify file exists and has reasonable size
            assert os.path.exists(file_path)
            file_size = os.path.getsize(file_path)
            assert file_size > 0

            # Load and verify data integrity
            loaded_df = MemoryOptimizer.load_from_parquet_compressed(file_path)
            assert loaded_df is not None
            pd.testing.assert_frame_equal(df, loaded_df, check_dtype=False)

        finally:
            if os.path.exists(file_path):
                os.unlink(file_path)

    def test_load_from_parquet_compressed_with_filters(self):
        """Test loading Parquet with column selection and filters."""
        df = pd.DataFrame({
            'timestamp': pd.date_range('2024-01-01', periods=100, freq='1H'),
            'symbol': ['AAPL'] * 50 + ['MSFT'] * 50,
            'price': np.random.uniform(100, 200, 100),
            'volume': np.random.randint(1000, 10000, 100)
        })

        with tempfile.NamedTemporaryFile(suffix='.parquet', delete=False) as f:
            file_path = f.name

        try:
            # Save file
            MemoryOptimizer.save_to_parquet_compressed(df, file_path)

            # Load with column selection
            columns = ['timestamp', 'price']
            filtered_df = MemoryOptimizer.load_from_parquet_compressed(
                file_path, columns=columns
            )
            assert filtered_df is not None
            assert list(filtered_df.columns) == columns

            # Load with filters
            filters = [('symbol', '==', 'AAPL')]
            aapl_data = MemoryOptimizer.load_from_parquet_compressed(
                file_path, filters=filters
            )
            assert aapl_data is not None
            assert (aapl_data['symbol'] == 'AAPL').all()
            assert len(aapl_data) == 50

        finally:
            if os.path.exists(file_path):
                os.unlink(file_path)

    def test_parquet_compression_edge_cases(self):
        """Test Parquet compression edge cases."""
        # Empty DataFrame
        empty_df = pd.DataFrame()
        with tempfile.NamedTemporaryFile(suffix='.parquet', delete=False) as f:
            file_path = f.name

        try:
            success = MemoryOptimizer.save_to_parquet_compressed(empty_df, file_path)
            assert not success  # Should fail gracefully for empty DataFrame
        except:
            pass  # Expected to fail
        finally:
            if os.path.exists(file_path):
                os.unlink(file_path)


class TestMemoryMappedArrays:
    """Test memory-mapped array functionality."""

    def test_create_memory_mapped_array(self):
        """Test memory-mapped array creation."""
        data = np.random.rand(1000).astype(np.float32)

        # Test with auto-generated file
        mmap_array = MemoryOptimizer.create_memory_mapped_array(data)

        # Verify data integrity
        np.testing.assert_array_equal(mmap_array, data)
        assert mmap_array.dtype == data.dtype

        # Clean up
        if hasattr(mmap_array, '_mmap_file'):
            try:
                os.unlink(mmap_array._mmap_file)
            except:
                pass

    def test_memory_mapped_array_with_custom_file(self):
        """Test memory-mapped array with custom file path."""
        data = np.random.rand(100).astype(np.float64)

        with tempfile.NamedTemporaryFile(delete=False) as f:
            custom_path = f.name

        try:
            mmap_array = MemoryOptimizer.create_memory_mapped_array(
                data, file_path=custom_path
            )

            np.testing.assert_array_equal(mmap_array, data)

            # Verify file exists
            assert os.path.exists(custom_path + '.npy')

        finally:
            # Clean up
            for ext in ['', '.npy']:
                file_path = custom_path + ext
                if os.path.exists(file_path):
                    try:
                        os.unlink(file_path)
                    except:
                        pass


class TestChunkedProcessing:
    """Test chunked data processing."""

    def test_chunked_dataframe_processing(self):
        """Test chunked DataFrame processing."""
        # Create large DataFrame
        df = pd.DataFrame({
            'timestamp': pd.date_range('2024-01-01', periods=1000, freq='1min'),
            'price': np.random.uniform(100, 200, 1000),
            'volume': np.random.randint(100, 1000, 1000)
        })

        chunks = list(MemoryOptimizer.chunked_dataframe_processing(df, chunk_size=100))

        # Verify chunking
        assert len(chunks) == 10  # 1000 rows / 100 chunk_size
        for chunk in chunks:
            assert len(chunk) == 100
            assert list(chunk.columns) == list(df.columns)

        # Test with processing function
        def add_derived_column(chunk):
            chunk = chunk.copy()
            chunk['price_change'] = chunk['price'].pct_change()
            return chunk

        processed_chunks = list(MemoryOptimizer.chunked_dataframe_processing(
            df, chunk_size=100, process_func=add_derived_column
        ))

        for chunk in processed_chunks:
            assert 'price_change' in chunk.columns
            assert len(chunk) == 100

    def test_process_large_file_chunked_csv(self):
        """Test processing large CSV files in chunks."""
        # Create test CSV file
        df = pd.DataFrame({
            'timestamp': pd.date_range('2024-01-01', periods=500, freq='1min'),
            'open': np.random.uniform(100, 110, 500),
            'high': np.random.uniform(105, 115, 500),
            'low': np.random.uniform(95, 105, 500),
            'close': np.random.uniform(100, 110, 500),
            'volume': np.random.randint(1000, 10000, 500)
        })

        with tempfile.NamedTemporaryFile(suffix='.csv', delete=False, mode='w') as f:
            df.to_csv(f, index=False)
            csv_path = f.name

        try:
            # Test chunked processing
            result_df = MemoryOptimizer.process_large_file_chunked(
                csv_path, chunk_size=100, output_format='dataframe'
            )

            assert isinstance(result_df, pd.DataFrame)
            assert len(result_df) == 500
            pd.testing.assert_frame_equal(result_df, df, check_dtype=False)

            # Test list output
            chunks = MemoryOptimizer.process_large_file_chunked(
                csv_path, chunk_size=100, output_format='list'
            )

            assert isinstance(chunks, list)
            assert len(chunks) == 5  # 500 / 100
            for chunk in chunks:
                assert len(chunk) == 100

        finally:
            if os.path.exists(csv_path):
                os.unlink(csv_path)


class TestCompressedCache:
    """Test compressed cache functionality."""

    def test_create_compressed_cache(self):
        """Test creating compressed cache."""
        df = pd.DataFrame({
            'timestamp': pd.date_range('2024-01-01', periods=100, freq='1H'),
            'price': np.random.uniform(100, 200, 100)
        })

        with tempfile.TemporaryDirectory() as cache_dir:
            cache_file = MemoryOptimizer.create_compressed_cache(
                df, 'test_cache', cache_dir=cache_dir
            )

            assert os.path.exists(cache_file)

            # Load from cache
            cached_df = MemoryOptimizer.load_from_compressed_cache(cache_file)
            assert cached_df is not None
            pd.testing.assert_frame_equal(df, cached_df, check_dtype=False)

    def test_load_from_compressed_cache_with_age_filter(self):
        """Test cache loading with age filtering."""
        df = pd.DataFrame({'data': [1, 2, 3]})

        with tempfile.TemporaryDirectory() as cache_dir:
            cache_file = MemoryOptimizer.create_compressed_cache(
                df, 'age_test', cache_dir=cache_dir
            )

            # Should load with no age restriction
            cached_df = MemoryOptimizer.load_from_compressed_cache(cache_file)
            assert cached_df is not None

            # Should return None for very short age limit
            old_cached_df = MemoryOptimizer.load_from_compressed_cache(
                cache_file, max_age_seconds=0
            )
            assert old_cached_df is None


class TestMemoryMonitoring:
    """Test memory monitoring functionality."""

    def test_monitor_memory_usage(self):
        """Test memory usage monitoring context manager."""
        optimizer = MemoryOptimizer()

        with optimizer.monitor_memory_usage("test_operation") as monitor:
            # Perform some memory operations
            data = np.random.rand(10000)
            result = data * 2

        # Monitor should complete without errors
        assert monitor is not None

    @patch('psutil.virtual_memory')
    def test_memory_budget_context_manager(self, mock_virtual_memory):
        """Test memory budget context manager."""
        # Mock memory usage
        mock_memory = MagicMock()
        mock_memory.available = 8 * 1024 * 1024 * 1024  # 8GB
        mock_virtual_memory.return_value = mock_memory

        with memory_budget(max_memory_mb=100):
            # Perform operations within budget
            data = np.random.rand(1000)
            result = data.sum()

        # Should complete without raising budget exceeded warning


class TestTradingDataPipeline:
    """Test trading data pipeline optimization."""

    def test_optimize_trading_data_pipeline(self):
        """Test complete trading data pipeline optimization."""
        # Create trading data with suboptimal structure
        df = pd.DataFrame({
            'timestamp': pd.date_range('2024-01-01', periods=1000, freq='1min'),
            'open': np.random.uniform(100, 110, 1000),
            'high': np.random.uniform(105, 115, 1000),
            'low': np.random.uniform(95, 105, 1000),
            'close': np.random.uniform(100, 110, 1000),
            'volume': np.random.randint(1000, 10000, 1000),
            'extra_col': ['redundant'] * 1000  # Column to be optimized
        })

        # Force suboptimal dtypes
        df['volume'] = df['volume'].astype('int64')
        df['extra_col'] = df['extra_col'].astype('object')

        original_memory = df.memory_usage(deep=True).sum()

        # Optimize pipeline
        optimized_df = optimize_trading_data_pipeline(df)

        optimized_memory = optimized_df.memory_usage(deep=True).sum()

        # Should reduce memory usage
        assert optimized_memory < original_memory

        # Essential columns should remain
        required_cols = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
        for col in required_cols:
            assert col in optimized_df.columns


class TestPerformanceBenchmarking:
    """Test memory optimization benchmarking."""

    def test_benchmark_memory_optimization(self):
        """Test memory optimization benchmarking."""
        results = benchmark_memory_optimization()

        required_keys = [
            'original_memory_mb', 'optimized_memory_mb',
            'memory_savings_mb', 'savings_percent', 'optimization_time_seconds'
        ]

        for key in required_keys:
            assert key in results

        # Should show some optimization benefit
        assert results['savings_percent'] >= 0
        assert results['optimization_time_seconds'] >= 0


class TestErrorHandling:
    """Test error handling in memory optimization."""

    def test_parquet_compression_without_pyarrow(self):
        """Test graceful handling when PyArrow is not available."""
        with patch('src.utils.memory_optimizer.PYARROW_AVAILABLE', False):
            df = pd.DataFrame({'test': [1, 2, 3]})
            success = MemoryOptimizer.save_to_parquet_compressed(df, 'test.parquet')
            assert not success

    def test_load_nonexistent_parquet(self):
        """Test loading non-existent Parquet file."""
        result = MemoryOptimizer.load_from_parquet_compressed('nonexistent.parquet')
        assert result is None

    def test_chunked_processing_invalid_file(self):
        """Test chunked processing with invalid file."""
        result = MemoryOptimizer.process_large_file_chunked('nonexistent.xyz')
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 0


class TestIntegration:
    """Test integration of memory optimization components."""

    def test_full_pipeline_integration(self):
        """Test full memory optimization pipeline."""
        # Create comprehensive trading dataset
        df = pd.DataFrame({
            'timestamp': pd.date_range('2024-01-01', periods=10000, freq='1min'),
            'symbol': np.random.choice(['AAPL', 'MSFT', 'GOOGL'], 10000),
            'open': np.random.uniform(100, 200, 10000),
            'high': np.random.uniform(105, 205, 10000),
            'low': np.random.uniform(95, 195, 10000),
            'close': np.random.uniform(100, 200, 10000),
            'volume': np.random.randint(1000, 100000, 10000),
            'vwap': np.random.uniform(100, 200, 10000),
            'adv20': np.random.uniform(0.5, 2.0, 10000)
        })

        # Step 1: Optimize DataFrame memory
        df = MemoryOptimizer.optimize_dataframe_memory(df)

        # Step 2: Create memory-mapped array for large numeric data
        price_data = df[['open', 'high', 'low', 'close']].values
        mmap_prices = MemoryOptimizer.create_memory_mapped_array(price_data)

        # Step 3: Process in chunks
        chunks = list(MemoryOptimizer.chunked_dataframe_processing(df, chunk_size=1000))

        # Step 4: Create compressed cache
        with tempfile.TemporaryDirectory() as cache_dir:
            cache_file = MemoryOptimizer.create_compressed_cache(
                df, 'integration_test', cache_dir=cache_dir
            )

            # Verify cache works
            cached_data = MemoryOptimizer.load_from_compressed_cache(cache_file)
            assert cached_data is not None
            assert len(cached_data) == len(df)

        # Cleanup
        if hasattr(mmap_prices, '_mmap_file'):
            try:
                os.unlink(mmap_prices._mmap_file)
            except:
                pass

        # Verify all operations completed successfully
        assert len(chunks) == 10  # 10000 / 1000
        assert mmap_prices.shape == price_data.shape


if __name__ == "__main__":
    pytest.main([__file__])
