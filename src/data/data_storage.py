#!/usr/bin/env python3
"""
Supreme System V5 - Data Storage Module

Efficient data storage and retrieval using Parquet format.
Optimized for time-series financial data with compression and partitioning.
"""

import glob
import logging
import os
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from ..utils.memory_optimizer import MemoryOptimizer


class DataStorage:
    """
    Efficient data storage system using Parquet format.

    Features:
    - Partitioned storage by symbol and date
    - Compression optimization
    - Metadata tracking
    - Incremental updates
    - Query optimization
    """

    def __init__(self, base_dir: str = "./data"):
        """
        Initialize data storage.

        Args:
            base_dir: Base directory for data storage
        """
        self.base_dir = Path(base_dir)
        self.historical_dir = self.base_dir / "historical"
        self.cache_dir = self.base_dir / "cache"
        self.metadata_dir = self.base_dir / "metadata"

        # Create directories
        for dir_path in [self.historical_dir, self.cache_dir, self.metadata_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)

        self.logger = logging.getLogger(__name__)

        # Storage configuration
        self.compression = 'snappy'  # Fast compression for time-series data
        self.chunk_size = 10000     # Rows per chunk for memory efficiency

    def store_historical_data(
        self,
        data: pd.DataFrame,
        symbol: str,
        interval: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        Store historical market data with partitioning.

        Args:
            data: OHLCV DataFrame
            symbol: Trading symbol
            interval: Time interval (e.g., '1h', '1d')
            metadata: Additional metadata

        Returns:
            Success status
        """
        try:
            if data.empty:
                self.logger.warning(f"No data to store for {symbol}")
                return False

            # Validate data structure
            required_columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
            missing_cols = [col for col in required_columns if col not in data.columns]
            if missing_cols:
                self.logger.error(f"Missing required columns: {missing_cols}")
                return False

            # Ensure timestamp is datetime
            data = data.copy()
            data['timestamp'] = pd.to_datetime(data['timestamp'])

            # Add metadata columns
            data['symbol'] = symbol
            data['interval'] = interval
            data['stored_at'] = datetime.now()

            # Partition by year/month for efficient querying
            data['year'] = data['timestamp'].dt.year
            data['month'] = data['timestamp'].dt.month

            # Group by partition and save
            partitions = data.groupby(['symbol', 'interval', 'year', 'month'])

            success_count = 0
            for (sym, interv, year, month), partition_data in partitions:
                # Create partition directory
                partition_dir = self.historical_dir / sym / interv / f"{year:04d}" / f"{month:02d}"
                partition_dir.mkdir(parents=True, exist_ok=True)

                # File path
                filename = f"{sym}_{interv}_{year:04d}_{month:02d}.parquet"
                file_path = partition_dir / filename

                # Convert to PyArrow table for efficient storage
                table = pa.Table.from_pandas(partition_data)

                # Save with compression
                pq.write_table(
                    table,
                    file_path,
                    compression=self.compression,
                    use_dictionary=True,
                    row_group_size=self.chunk_size
                )

                success_count += 1

            # Save metadata
            if metadata:
                metadata.update({
                    'symbol': symbol,
                    'interval': interval,
                    'total_rows': len(data),
                    'partitions': success_count,
                    'stored_at': datetime.now().isoformat(),
                    'compression': self.compression
                })
                self._save_metadata(symbol, interval, metadata)

            self.logger.info(f"Stored {len(data)} rows for {symbol} {interval} in {success_count} partitions")
            return True

        except Exception as e:
            self.logger.error(f"Failed to store data for {symbol}: {e}")
            return False

    def load_historical_data(
        self,
        symbol: str,
        interval: str,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        columns: Optional[List[str]] = None
    ) -> Optional[pd.DataFrame]:
        """
        Load historical data with efficient partitioning.

        Args:
            symbol: Trading symbol
            interval: Time interval
            start_date: Start date filter (YYYY-MM-DD)
            end_date: End date filter (YYYY-MM-DD)
            columns: Specific columns to load

        Returns:
            Combined DataFrame or None if not found
        """
        try:
            # Find all relevant partitions
            pattern = str(self.historical_dir / symbol / interval / "**/*.parquet")
            partition_files = glob.glob(pattern, recursive=True)

            if not partition_files:
                self.logger.warning(f"No data found for {symbol} {interval}")
                return None

            # Filter by date range if specified
            filtered_files = self._filter_files_by_date(partition_files, start_date, end_date)

            if not filtered_files:
                self.logger.warning(f"No data in date range for {symbol} {interval}")
                return None

            # Load and combine data
            dfs = []
            for file_path in filtered_files:
                try:
                    # Load with PyArrow for efficiency
                    table = pq.read_table(file_path, columns=columns)
                    df = table.to_pandas()

                    # Convert timestamp back to datetime
                    df['timestamp'] = pd.to_datetime(df['timestamp'])

                    dfs.append(df)

                except Exception as e:
                    self.logger.warning(f"Failed to load {file_path}: {e}")
                    continue

            if not dfs:
                return None

            # Combine all partitions
            combined_df = pd.concat(dfs, ignore_index=True)

            # Sort by timestamp and remove duplicates
            combined_df = combined_df.sort_values('timestamp').drop_duplicates(subset=['timestamp'])

            # Filter by exact date range if needed
            if start_date:
                start_ts = pd.Timestamp(start_date)
                combined_df = combined_df[combined_df['timestamp'] >= start_ts]

            if end_date:
                end_ts = pd.Timestamp(end_date)
                combined_df = combined_df[combined_df['timestamp'] <= end_ts]

            self.logger.info(f"Loaded {len(combined_df)} rows for {symbol} {interval}")
            return combined_df

        except Exception as e:
            self.logger.error(f"Failed to load data for {symbol}: {e}")
            return None

    def update_data(
        self,
        data: pd.DataFrame,
        symbol: str,
        interval: str,
        update_mode: str = 'append'
    ) -> bool:
        """
        Update existing data with new information.

        Args:
            data: New data to add
            symbol: Trading symbol
            interval: Time interval
            update_mode: 'append', 'replace', or 'merge'

        Returns:
            Success status
        """
        try:
            if update_mode == 'replace':
                # Remove existing data and store new
                self._remove_symbol_data(symbol, interval)
                return self.store_historical_data(data, symbol, interval)

            elif update_mode == 'append':
                # Load existing data and append
                existing_data = self.load_historical_data(symbol, interval)
                if existing_data is not None:
                    # Remove overlapping timestamps
                    new_timestamps = set(data['timestamp'])
                    existing_data = existing_data[~existing_data['timestamp'].isin(new_timestamps)]

                    # Combine and store
                    combined_data = pd.concat([existing_data, data], ignore_index=True)
                    combined_data = combined_data.sort_values('timestamp')
                    return self.store_historical_data(combined_data, symbol, interval)
                else:
                    # No existing data, just store
                    return self.store_historical_data(data, symbol, interval)

            elif update_mode == 'merge':
                # More complex merge logic for overlapping data
                existing_data = self.load_historical_data(symbol, interval)
                if existing_data is not None:
                    # Merge on timestamp, prioritizing newer data
                    merged = pd.concat([existing_data, data])
                    merged = merged.drop_duplicates(subset=['timestamp'], keep='last')
                    merged = merged.sort_values('timestamp')
                    return self.store_historical_data(merged, symbol, interval)
                else:
                    return self.store_historical_data(data, symbol, interval)

            else:
                self.logger.error(f"Unknown update mode: {update_mode}")
                return False

        except Exception as e:
            self.logger.error(f"Failed to update data for {symbol}: {e}")
            return False

    def get_data_info(self, symbol: Optional[str] = None, interval: Optional[str] = None) -> Dict[str, Any]:
        """
        Get information about stored data.

        Args:
            symbol: Filter by symbol
            interval: Filter by interval

        Returns:
            Data information dictionary
        """
        try:
            info = {
                'total_symbols': 0,
                'total_intervals': 0,
                'total_files': 0,
                'total_size_mb': 0.0,
                'symbols': {},
                'last_updated': None
            }

            # Find all parquet files
            pattern = str(self.historical_dir / "**/*.parquet")
            all_files = glob.glob(pattern, recursive=True)

            if not all_files:
                return info

            # Group by symbol and interval
            symbol_info = {}

            for file_path in all_files:
                try:
                    # Extract symbol and interval from path
                    parts = Path(file_path).parts
                    if len(parts) >= 3:
                        file_symbol = parts[-4]  # symbol/interval/year/month/file.parquet
                        file_interval = parts[-3]

                        # Apply filters
                        if symbol and file_symbol != symbol:
                            continue
                        if interval and file_interval != interval:
                            continue

                        # Get file info
                        file_size = os.path.getsize(file_path) / (1024 * 1024)  # MB
                        file_mtime = datetime.fromtimestamp(os.path.getmtime(file_path))

                        if file_symbol not in symbol_info:
                            symbol_info[file_symbol] = {}

                        if file_interval not in symbol_info[file_symbol]:
                            symbol_info[file_symbol][file_interval] = {
                                'files': 0,
                                'size_mb': 0.0,
                                'last_updated': file_mtime,
                                'partitions': []
                            }

                        symbol_info[file_symbol][file_interval]['files'] += 1
                        symbol_info[file_symbol][file_interval]['size_mb'] += file_size
                        symbol_info[file_symbol][file_interval]['last_updated'] = max(
                            symbol_info[file_symbol][file_interval]['last_updated'],
                            file_mtime
                        )

                        info['total_files'] += 1
                        info['total_size_mb'] += file_size

                except Exception as e:
                    self.logger.warning(f"Error processing file {file_path}: {e}")
                    continue

            info['symbols'] = symbol_info
            info['total_symbols'] = len(symbol_info)
            info['total_intervals'] = sum(len(intervals) for intervals in symbol_info.values())

            if all_files:
                info['last_updated'] = datetime.fromtimestamp(
                    max(os.path.getmtime(f) for f in all_files)
                )

            return info

        except Exception as e:
            self.logger.error(f"Failed to get data info: {e}")
            return {}

    def _filter_files_by_date(
        self,
        files: List[str],
        start_date: Optional[str],
        end_date: Optional[str]
    ) -> List[str]:
        """Filter partition files by date range."""
        if not start_date and not end_date:
            return files

        filtered = []

        for file_path in files:
            try:
                # Extract year/month from filename
                filename = Path(file_path).name
                parts = filename.split('_')
                if len(parts) >= 4:
                    year = int(parts[2])
                    month = int(parts[3].split('.')[0])

                    # Check if partition overlaps with date range
                    partition_start = datetime(year, month, 1)
                    if month == 12:
                        partition_end = datetime(year + 1, 1, 1) - timedelta(days=1)
                    else:
                        partition_end = datetime(year, month + 1, 1) - timedelta(days=1)

                    # Date range filtering
                    include_partition = True

                    if start_date:
                        start_filter = pd.Timestamp(start_date)
                        if partition_end < start_filter:
                            include_partition = False

                    if end_date:
                        end_filter = pd.Timestamp(end_date)
                        if partition_start > end_filter:
                            include_partition = False

                    if include_partition:
                        filtered.append(file_path)

            except Exception:
                # If parsing fails, include the file
                filtered.append(file_path)

        return filtered

    def _remove_symbol_data(self, symbol: str, interval: Optional[str] = None):
        """Remove all data for a symbol/interval."""
        try:
            if interval:
                target_dir = self.historical_dir / symbol / interval
            else:
                target_dir = self.historical_dir / symbol

            if target_dir.exists():
                import shutil
                shutil.rmtree(target_dir)
                self.logger.info(f"Removed data directory: {target_dir}")

        except Exception as e:
            self.logger.error(f"Failed to remove data for {symbol}: {e}")

    def _save_metadata(self, symbol: str, interval: str, metadata: Dict[str, Any]):
        """Save metadata for symbol/interval."""
        try:
            metadata_file = self.metadata_dir / f"{symbol}_{interval}_metadata.json"
            metadata_file.parent.mkdir(parents=True, exist_ok=True)

            import json
            with open(metadata_file, 'w') as f:
                json.dump(metadata, f, indent=2, default=str)

        except Exception as e:
            self.logger.error(f"Failed to save metadata: {e}")

    def store_data(self, data: pd.DataFrame, symbol: str, partition_by: str = 'date') -> Dict[str, Any]:
        """
        Store data with advanced compression and memory optimization.

        Enhanced version with compressed Parquet storage, memory optimization,
        and detailed performance metrics.

        Args:
            data: DataFrame to store
            symbol: Trading symbol
            partition_by: Partitioning strategy ('date' or 'date_hour')

        Returns:
            Dict with success status and performance metrics
        """
        result = {
            'success': False,
            'rows_stored': 0,
            'original_size_mb': 0.0,
            'compressed_size_mb': 0.0,
            'compression_ratio': 0.0,
            'duration_seconds': 0.0,
            'error': None
        }

        start_time = datetime.now()

        try:
            if data.empty:
                result['error'] = "No data to store"
                self.logger.warning(f"No data to store for {symbol}")
                return result

            # Validate required columns
            required_columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
            missing_cols = [col for col in required_columns if col not in data.columns]
            if missing_cols:
                result['error'] = f"Missing required columns: {missing_cols}"
                self.logger.error(f"Missing required columns: {missing_cols}")
                return result

            # Optimize memory usage before storage
            original_memory = data.memory_usage(deep=True).sum() / 1024 / 1024
            data = MemoryOptimizer.optimize_dataframe_dtypes(data)
            optimized_memory = data.memory_usage(deep=True).sum() / 1024 / 1024

            # Ensure timestamp is datetime
            data = data.copy()
            data['timestamp'] = pd.to_datetime(data['timestamp'])

            # Add partitioning columns
            data['date'] = data['timestamp'].dt.date
            if partition_by == 'date_hour':
                data['hour'] = data['timestamp'].dt.hour

            # Create symbol directory
            symbol_dir = self.historical_dir / symbol
            symbol_dir.mkdir(parents=True, exist_ok=True)

            # Use advanced compressed storage
            file_path = symbol_dir / f"{symbol}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.parquet"

            # Store with compression
            compression_success = MemoryOptimizer.save_to_parquet_compressed(
                data,
                str(file_path),
                compression='zstd',  # Best compression ratio
                row_group_size=min(50000, len(data))  # Adaptive row group size
            )

            if not compression_success:
                result['error'] = "Failed to save compressed data"
                return result

            # Get compressed file size
            compressed_size = os.path.getsize(file_path) / 1024 / 1024

            # Calculate metrics
            duration = (datetime.now() - start_time).total_seconds()

            result.update({
                'success': True,
                'rows_stored': len(data),
                'original_size_mb': original_memory,
                'compressed_size_mb': compressed_size,
                'compression_ratio': original_memory / compressed_size if compressed_size > 0 else 0,
                'duration_seconds': duration,
                'memory_optimization_ratio': optimized_memory / original_memory if original_memory > 0 else 1.0
            })

            self.logger.info(f"Stored {len(data)} rows for {symbol} with {result['compression_ratio']:.1f}x compression")
            self.logger.info(f"Memory optimization: {result['memory_optimization_ratio']:.2f}x reduction")
            return result

        except Exception as e:
            result['error'] = str(e)
            result['duration_seconds'] = (datetime.now() - start_time).total_seconds()
            self.logger.error(f"Failed to store data for {symbol}: {e}")
            return result

    def query_data(self, symbol: str, start_date: str, end_date: str) -> Optional[pd.DataFrame]:
        """
        Query data by date range for testing framework.

        This is a simplified version that matches testing expectations.

        Args:
            symbol: Trading symbol
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)

        Returns:
            Filtered DataFrame or None
        """
        try:
            symbol_dir = self.historical_dir / symbol

            if not symbol_dir.exists():
                self.logger.warning(f"No data found for {symbol}")
                return None

            # Read parquet dataset with filtering
            # Convert dates to string format for filtering
            start_str = pd.Timestamp(start_date).strftime('%Y-%m-%d')
            end_str = pd.Timestamp(end_date).strftime('%Y-%m-%d')

            dataset = pq.ParquetDataset(
                symbol_dir,
                filters=[
                    ('date', '>=', start_str),
                    ('date', '<=', end_str)
                ]
            )

            # Read only essential columns
            table = dataset.read(columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df = table.to_pandas()

            # Sort by timestamp
            df = df.sort_values('timestamp').reset_index(drop=True)

            self.logger.info(f"Queried {len(df)} rows for {symbol} from {start_date} to {end_date}")
            return df

        except Exception as e:
            self.logger.error(f"Failed to query data for {symbol}: {e}")
            return None

    def cleanup_old_data(self, days_to_keep: int = 365):
        """Clean up data older than specified days."""
        try:
            cutoff_date = datetime.now() - timedelta(days=days_to_keep)
            removed_files = 0

            pattern = str(self.historical_dir / "**/*.parquet")
            all_files = glob.glob(pattern, recursive=True)

            for file_path in all_files:
                try:
                    file_mtime = datetime.fromtimestamp(os.path.getsize(file_path))
                    if file_mtime < cutoff_date:
                        os.remove(file_path)
                        removed_files += 1
                except Exception:
                    continue

            if removed_files > 0:
                self.logger.info(f"Cleaned up {removed_files} old data files")

        except Exception as e:
            self.logger.error(f"Failed to cleanup old data: {e}")
