"""
Memory Optimization Utilities for Supreme System V5

Advanced memory management and optimization techniques for trading systems.
Reduces memory footprint while maintaining performance.
"""

import gc
import psutil
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Iterator
from contextlib import contextmanager
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


class MemoryOptimizer:
    """Memory optimization and management utilities."""

    def __init__(self):
        self.process = psutil.Process()
        self.initial_memory = self.get_current_memory_mb()

    def get_current_memory_mb(self) -> float:
        """Get current memory usage in MB."""
        return self.process.memory_info().rss / 1024 / 1024

    def get_memory_usage_report(self) -> Dict[str, Any]:
        """Get comprehensive memory usage report."""
        memory_info = self.process.memory_info()

        return {
            'rss_mb': memory_info.rss / 1024 / 1024,
            'vms_mb': memory_info.vms / 1024 / 1024,
            'percent': self.process.memory_percent(),
            'available_mb': psutil.virtual_memory().available / 1024 / 1024,
            'total_mb': psutil.virtual_memory().total / 1024 / 1024
        }

    @staticmethod
    def optimize_dataframe_dtypes(df: pd.DataFrame) -> pd.DataFrame:
        """
        Optimize DataFrame memory usage by selecting optimal dtypes.

        Args:
            df: Input DataFrame

        Returns:
            DataFrame with optimized dtypes
        """
        optimized_df = df.copy()

        for col in optimized_df.columns:
            col_data = optimized_df[col]

            if col_data.dtype == 'object':
                # Try to convert to category if few unique values
                if col_data.nunique() / len(col_data) < 0.5:
                    optimized_df[col] = col_data.astype('category')
                continue

            if col_data.dtype == 'float64':
                # Check if we can downcast
                if (col_data % 1 == 0).all():  # All integers
                    min_val, max_val = col_data.min(), col_data.max()
                    if min_val >= 0:
                        if max_val < 2**8:
                            optimized_df[col] = col_data.astype('uint8')
                        elif max_val < 2**16:
                            optimized_df[col] = col_data.astype('uint16')
                        elif max_val < 2**32:
                            optimized_df[col] = col_data.astype('uint32')
                        else:
                            optimized_df[col] = col_data.astype('uint64')
                    else:
                        if min_val >= -2**7 and max_val < 2**7:
                            optimized_df[col] = col_data.astype('int8')
                        elif min_val >= -2**15 and max_val < 2**15:
                            optimized_df[col] = col_data.astype('int16')
                        elif min_val >= -2**31 and max_val < 2**31:
                            optimized_df[col] = col_data.astype('int32')
                        # Keep int64 if needed
                else:
                    # Float values - try float32
                    try:
                        optimized_df[col] = col_data.astype('float32')
                    except (ValueError, OverflowError):
                        pass  # Keep float64 if conversion fails

            elif col_data.dtype == 'int64':
                # Try to downcast integers
                min_val, max_val = col_data.min(), col_data.max()
                if min_val >= 0:
                    if max_val < 2**8:
                        optimized_df[col] = col_data.astype('uint8')
                    elif max_val < 2**16:
                        optimized_df[col] = col_data.astype('uint16')
                    elif max_val < 2**32:
                        optimized_df[col] = col_data.astype('uint32')
                    else:
                        optimized_df[col] = col_data.astype('uint64')
                else:
                    if min_val >= -2**7 and max_val < 2**7:
                        optimized_df[col] = col_data.astype('int8')
                    elif min_val >= -2**15 and max_val < 2**15:
                        optimized_df[col] = col_data.astype('int16')
                    elif min_val >= -2**31 and max_val < 2**31:
                        optimized_df[col] = col_data.astype('int32')
                    # Keep int64 if needed

        return optimized_df

    @staticmethod
    def optimize_numeric_arrays(*arrays: np.ndarray) -> Tuple[np.ndarray, ...]:
        """
        Optimize memory usage of numeric arrays.

        Args:
            *arrays: NumPy arrays to optimize

        Returns:
            Tuple of optimized arrays
        """
        optimized_arrays = []

        for arr in arrays:
            if arr.dtype == np.float64:
                # Try float32
                if arr.min() > -3.4e38 and arr.max() < 3.4e38:
                    try:
                        optimized_arrays.append(arr.astype(np.float32))
                        continue
                    except (ValueError, OverflowError):
                        pass

            elif arr.dtype == np.int64:
                # Try smaller integer types
                min_val, max_val = arr.min(), arr.max()
                if min_val >= 0:
                    if max_val < 2**8:
                        optimized_arrays.append(arr.astype(np.uint8))
                    elif max_val < 2**16:
                        optimized_arrays.append(arr.astype(np.uint16))
                    elif max_val < 2**32:
                        optimized_arrays.append(arr.astype(np.uint32))
                    else:
                        optimized_arrays.append(arr.astype(np.uint64))
                else:
                    if min_val >= -2**7 and max_val < 2**7:
                        optimized_arrays.append(arr.astype(np.int8))
                    elif min_val >= -2**15 and max_val < 2**15:
                        optimized_arrays.append(arr.astype(np.int16))
                    elif min_val >= -2**31 and max_val < 2**31:
                        optimized_arrays.append(arr.astype(np.int32))
                    else:
                        optimized_arrays.append(arr)
            else:
                optimized_arrays.append(arr)

        return tuple(optimized_arrays)

    @staticmethod
    def chunked_dataframe_processing(df: pd.DataFrame,
                                   chunk_size: int = 10000,
                                   process_func: callable = None) -> Iterator[pd.DataFrame]:
        """
        Process DataFrame in chunks to reduce memory usage.

        Args:
            df: DataFrame to process
            chunk_size: Size of each chunk
            process_func: Optional function to apply to each chunk

        Yields:
            Processed chunks
        """
        for start_idx in range(0, len(df), chunk_size):
            end_idx = min(start_idx + chunk_size, len(df))
            chunk = df.iloc[start_idx:end_idx].copy()

            if process_func:
                chunk = process_func(chunk)

            yield chunk

    @staticmethod
    def create_memory_efficient_series(data: List[float],
                                     dtype: Optional[str] = None) -> pd.Series:
        """
        Create memory-efficient pandas Series.

        Args:
            data: Data to create series from
            dtype: Optional dtype specification

        Returns:
            Memory-optimized Series
        """
        if dtype:
            return pd.Series(data, dtype=dtype)

        # Auto-detect optimal dtype
        if all(isinstance(x, int) for x in data[:100]):  # Sample first 100
            min_val, max_val = min(data), max(data)
            if min_val >= 0:
                if max_val < 2**8:
                    return pd.Series(data, dtype='uint8')
                elif max_val < 2**16:
                    return pd.Series(data, dtype='uint16')
                elif max_val < 2**32:
                    return pd.Series(data, dtype='uint32')
            else:
                if min_val >= -2**7 and max_val < 2**7:
                    return pd.Series(data, dtype='int8')
                elif min_val >= -2**15 and max_val < 2**15:
                    return pd.Series(data, dtype='int16')
                elif min_val >= -2**31 and max_val < 2**31:
                    return pd.Series(data, dtype='int32')

        # Default to float32 for numeric data
        return pd.Series(data, dtype='float32')

    @staticmethod
    def garbage_collect_forced():
        """Force garbage collection and return memory freed."""
        before = psutil.virtual_memory().available
        gc.collect()
        after = psutil.virtual_memory().available
        freed_mb = (after - before) / 1024 / 1024

        logger.info(f"Garbage collection freed {freed_mb:.2f} MB")
        return freed_mb

    def monitor_memory_usage(self, operation_name: str) -> '_MemoryMonitor':
        """
        Context manager to monitor memory usage during an operation.

        Args:
            operation_name: Name of the operation being monitored

        Returns:
            Memory monitor context manager
        """
        return _MemoryMonitor(self, operation_name)


class _MemoryMonitor:
    """Context manager for monitoring memory usage."""

    def __init__(self, optimizer: MemoryOptimizer, operation_name: str):
        self.optimizer = optimizer
        self.operation_name = operation_name
        self.start_memory = None

    def __enter__(self):
        self.start_memory = self.optimizer.get_current_memory_mb()
        logger.debug(f"Starting memory monitoring for {self.operation_name}: {self.start_memory:.2f} MB")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        end_memory = self.optimizer.get_current_memory_mb()
        memory_delta = end_memory - self.start_memory

        if abs(memory_delta) > 1.0:  # Log if change > 1MB
            logger.info(f"Memory change for {self.operation_name}: {memory_delta:+.2f} MB "
                       f"(start: {self.start_memory:.1f} MB, end: {end_memory:.1f} MB)")

        return False


@contextmanager
def memory_budget(max_memory_mb: float):
    """
    Context manager that monitors memory usage and warns if budget exceeded.

    Args:
        max_memory_mb: Maximum allowed memory usage in MB
    """
    optimizer = MemoryOptimizer()
    start_memory = optimizer.get_current_memory_mb()

    try:
        yield
    finally:
        end_memory = optimizer.get_current_memory_mb()
        memory_used = end_memory - start_memory

        if memory_used > max_memory_mb:
            logger.warning(f"Memory budget exceeded: used {memory_used:.2f} MB, "
                         f"budget {max_memory_mb:.2f} MB")
        else:
            logger.debug(f"Memory usage within budget: {memory_used:.2f} MB used, "
                        f"{max_memory_mb:.2f} MB budget")


def optimize_trading_data_pipeline(data: pd.DataFrame) -> pd.DataFrame:
    """
    Comprehensive optimization for trading data pipeline.

    Args:
        data: Raw trading data DataFrame

    Returns:
        Fully optimized DataFrame
    """
    optimizer = MemoryOptimizer()

    logger.info(f"Starting data pipeline optimization. Initial memory: {optimizer.get_current_memory_mb():.2f} MB")

    # Step 1: Optimize dtypes
    with optimizer.monitor_memory_usage("dtype_optimization"):
        data = optimizer.optimize_dataframe_dtypes(data)

    # Step 2: Remove unnecessary columns if they exist
    columns_to_drop = []
    if 'quote_asset_volume' in data.columns:
        columns_to_drop.append('quote_asset_volume')
    if 'number_of_trades' in data.columns and data['number_of_trades'].isna().all():
        columns_to_drop.append('number_of_trades')

    if columns_to_drop:
        with optimizer.monitor_memory_usage("column_removal"):
            data = data.drop(columns=columns_to_drop)

    # Step 3: Optimize numeric precision where safe
    numeric_columns = data.select_dtypes(include=[np.number]).columns
    for col in numeric_columns:
        if col in ['volume', 'amount']:
            # Volume data can often be stored with lower precision
            if data[col].dtype == 'float64':
                try:
                    data[col] = data[col].astype('float32')
                except (ValueError, OverflowError):
                    pass

    # Step 4: Force garbage collection
    freed_memory = optimizer.garbage_collect_forced()

    final_memory = optimizer.get_current_memory_mb()
    logger.info(f"Data pipeline optimization complete. Final memory: {final_memory:.2f} MB")

    return data


def create_chunked_data_loader(file_path: str,
                              chunk_size: int = 50000,
                              optimize_memory: bool = True) -> Iterator[pd.DataFrame]:
    """
    Create a memory-efficient data loader that processes files in chunks.

    Args:
        file_path: Path to data file
        chunk_size: Size of each chunk
        optimize_memory: Whether to optimize memory usage

    Yields:
        Optimized data chunks
    """
    optimizer = MemoryOptimizer()

    file_extension = Path(file_path).suffix.lower()

    if file_extension == '.csv':
        reader = pd.read_csv(file_path, chunksize=chunk_size)
    elif file_extension in ['.parquet', '.pq']:
        # For parquet, we'll need to implement chunked reading differently
        # For now, read the whole file (can be optimized later)
        reader = [pd.read_parquet(file_path)]
    else:
        raise ValueError(f"Unsupported file format: {file_extension}")

    for chunk in reader:
        if optimize_memory:
            chunk = optimizer.optimize_dataframe_dtypes(chunk)

        yield chunk


def benchmark_memory_optimization():
    """Benchmark memory optimization techniques."""
    optimizer = MemoryOptimizer()

    # Create test data
    n_rows = 100000
    test_data = pd.DataFrame({
        'timestamp': pd.date_range('2024-01-01', periods=n_rows, freq='1min'),
        'open': np.random.uniform(100, 110, n_rows),
        'high': np.random.uniform(110, 120, n_rows),
        'low': np.random.uniform(90, 100, n_rows),
        'close': np.random.uniform(100, 110, n_rows),
        'volume': np.random.randint(1000, 10000, n_rows).astype('int64'),
        'symbol': ['AAPL'] * (n_rows // 2) + ['MSFT'] * (n_rows - n_rows // 2)
    })

    # Measure original memory usage
    original_memory = test_data.memory_usage(deep=True).sum() / 1024 / 1024
    logger.info(f"Original DataFrame memory usage: {original_memory:.2f} MB")

    # Apply optimization
    start_time = pd.Timestamp.now()
    optimized_data = optimizer.optimize_dataframe_dtypes(test_data)
    optimization_time = (pd.Timestamp.now() - start_time).total_seconds()

    # Measure optimized memory usage
    optimized_memory = optimized_data.memory_usage(deep=True).sum() / 1024 / 1024
    memory_savings = original_memory - optimized_memory
    savings_percent = (memory_savings / original_memory) * 100

    logger.info(f"Optimized DataFrame memory usage: {optimized_memory:.2f} MB")
    logger.info(f"Memory savings: {memory_savings:.2f} MB ({savings_percent:.1f}%)")
    logger.info(f"Optimization time: {optimization_time:.3f} seconds")

    # Test data integrity
    for col in test_data.columns:
        if col != 'symbol':  # Skip categorical column comparison
            try:
                pd.testing.assert_series_equal(
                    test_data[col], optimized_data[col],
                    check_dtype=False  # Allow dtype changes
                )
            except AssertionError as e:
                logger.error(f"Data integrity check failed for column {col}: {e}")

    return {
        'original_memory_mb': original_memory,
        'optimized_memory_mb': optimized_memory,
        'memory_savings_mb': memory_savings,
        'savings_percent': savings_percent,
        'optimization_time_seconds': optimization_time
    }


if __name__ == "__main__":
    # Run memory optimization benchmark
    results = benchmark_memory_optimization()
    logger.info("Memory optimization benchmark complete!")

    # Test garbage collection
    MemoryOptimizer.garbage_collect_forced()
