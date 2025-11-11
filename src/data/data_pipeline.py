#!/usr/bin/env python3
"""
Supreme System V5 - Data Pipeline

Complete data pipeline orchestration:
- Data fetching from multiple sources
- Validation and cleaning
- Efficient storage and retrieval
- Caching and performance optimization
"""

import asyncio
import logging
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta
from typing import Any, AsyncGenerator, Dict, List, Optional, Union

import pandas as pd

from .binance_client import BinanceClient
from .data_storage import DataStorage
from .data_validator import DataValidator
from ..utils.memory_optimizer import MemoryOptimizer, optimize_trading_data_pipeline
from ..utils.vectorized_ops import VectorizedTradingOps


class DataPipeline:
    """
    Complete data pipeline for financial market data.

    Orchestrates:
    - Multi-source data fetching
    - Real-time validation and cleaning
    - Efficient partitioned storage
    - Intelligent caching
    - Performance monitoring
    """

    def __init__(self, config_file: Optional[str] = None):
        """
        Initialize data pipeline.

        Args:
            config_file: Optional configuration file
        """
        self.logger = logging.getLogger(__name__)

        # Initialize components
        self.client = BinanceClient(config_file=config_file)
        self.validator = DataValidator()
        self.storage = DataStorage()

        # Initialize performance optimization components
        self.memory_optimizer = MemoryOptimizer()
        self.vectorized_ops = VectorizedTradingOps()

        # Pipeline metrics
        self.metrics = {
            'fetches': 0,
            'validations': 0,
            'storages': 0,
            'cache_hits': 0,
            'cache_misses': 0,
            'errors': 0,
            'total_processed_rows': 0,
            'avg_processing_time': 0.0
        }

        # Cache for recent data
        self.cache = {}
        self.cache_ttl = 300  # 5 minutes cache TTL

        self.logger.info("Data pipeline initialized")

    def fetch_and_store_data(
        self,
        symbol: str,
        interval: str,
        start_date: str,
        end_date: Optional[str] = None,
        validate: bool = True,
        force_refresh: bool = False
    ) -> Dict[str, Any]:
        """
        Complete pipeline: fetch, validate, and store data.

        Args:
            symbol: Trading symbol
            interval: Time interval
            start_date: Start date
            end_date: End date (optional)
            validate: Whether to validate data
            force_refresh: Force fresh download

        Returns:
            Pipeline execution results
        """
        start_time = time.time()

        result = {
            'success': False,
            'symbol': symbol,
            'interval': interval,
            'rows_processed': 0,
            'validation_passed': False,
            'storage_success': False,
            'duration': 0.0,
            'errors': []
        }

        try:
            self.logger.info(f"Starting pipeline for {symbol} {interval}")

            # Check cache first (unless force refresh)
            if not force_refresh:
                cached_data = self._get_cached_data(symbol, interval, start_date, end_date)
                if cached_data is not None:
                    self.metrics['cache_hits'] += 1
                    result['success'] = True
                    result['rows_processed'] = len(cached_data)
                    result['cached'] = True
                    result['duration'] = time.time() - start_time
                    self.logger.info(f"Cache hit for {symbol} {interval}")
                    return result

            self.metrics['cache_misses'] += 1

            # 1. Fetch data
            self.logger.info(f"Fetching data for {symbol} {interval}")
            raw_data = self.client.get_historical_klines(symbol, interval, start_date, end_date)

            if raw_data is None or raw_data.empty:
                result['errors'].append("No data fetched")
                self.metrics['errors'] += 1
                return result

            self.metrics['fetches'] += 1
            result['rows_processed'] = len(raw_data)

            # 2. Validate and clean data
            if validate:
                self.logger.info(f"Validating data for {symbol} {interval}")
                validation_result = self.validator.validate_ohlcv_data(raw_data, symbol)

                if not validation_result['is_valid']:
                    self.logger.warning(f"Data validation failed for {symbol}: {validation_result['issues']}")

                # Clean data even if validation failed
                cleaned_data = self.validator.clean_data(raw_data)
                actual_cleaned = len(cleaned_data)

                self.metrics['validations'] += 1

                if actual_cleaned != len(raw_data):
                    self.logger.info(f"Data cleaned: {len(raw_data)} -> {actual_cleaned} rows")

                result['validation_passed'] = validation_result['is_valid']
                result['quality_score'] = validation_result['quality_score']
                result['issues'] = validation_result['issues']

            else:
                cleaned_data = raw_data
                result['validation_passed'] = True

            # 3. Store data
            if not cleaned_data.empty:
                self.logger.info(f"Storing data for {symbol} {interval}")

                metadata = {
                    'source': 'binance_api',
                    'fetched_at': datetime.now().isoformat(),
                    'pipeline_version': 'v5.0',
                    'validation_performed': validate,
                    'quality_score': result.get('quality_score', 0)
                }

                storage_success = self.storage.store_historical_data(
                    cleaned_data, symbol, interval, metadata
                )

                if storage_success:
                    self.metrics['storages'] += 1
                    result['storage_success'] = True

                    # Cache the data
                    self._cache_data(symbol, interval, start_date, end_date, cleaned_data)

                else:
                    result['errors'].append("Storage failed")

            result['success'] = result['storage_success']
            result['duration'] = time.time() - start_time
            self.metrics['total_processed_rows'] += result['rows_processed']

            self.logger.info(f"Pipeline completed for {symbol} {interval} in {result['duration']:.2f}s")

        except Exception as e:
            result['errors'].append(str(e))
            result['duration'] = time.time() - start_time
            self.metrics['errors'] += 1
            self.logger.error(f"Pipeline failed for {symbol} {interval}: {e}")

        return result

    async def fetch_and_store_data_async(
        self,
        symbol: str,
        interval: str,
        start_date: str,
        end_date: Optional[str] = None,
        validate: bool = True,
        force_refresh: bool = False
    ) -> Dict[str, Any]:
        """
        Async version of fetch_and_store_data for high-performance pipelines.

        Args:
            symbol: Trading symbol
            interval: Time interval
            start_date: Start date
            end_date: End date (optional)
            validate: Whether to validate data
            force_refresh: Force fresh download

        Returns:
            Pipeline execution results
        """
        start_time = time.time()

        result = {
            'success': False,
            'symbol': symbol,
            'interval': interval,
            'rows_processed': 0,
            'validation_passed': False,
            'storage_success': False,
            'duration': 0.0,
            'errors': []
        }

        try:
            self.logger.info(f"Starting async pipeline for {symbol} {interval}")

            # Check cache first (unless force refresh)
            if not force_refresh:
                cached_data = self._get_cached_data(symbol, interval, start_date, end_date)
                if cached_data is not None:
                    self.metrics['cache_hits'] += 1
                    result['success'] = True
                    result['rows_processed'] = len(cached_data)
                    result['duration'] = time.time() - start_time
                    return result

            # Fetch data asynchronously
            loop = asyncio.get_event_loop()
            with ThreadPoolExecutor() as executor:
                data_future = loop.run_in_executor(
                    executor,
                    self.client.get_historical_klines,
                    symbol, interval, start_date, end_date
                )
                data = await data_future

            if data is None or data.empty:
                result['errors'].append("No data retrieved")
                result['duration'] = time.time() - start_time
                return result

            result['rows_processed'] = len(data)

            # Validate data asynchronously
            if validate:
                validation_result = await loop.run_in_executor(
                    executor,
                    self.validator.validate_ohlcv,
                    data
                )

                if not validation_result['valid']:
                    result['errors'].extend(validation_result['errors'])
                    result['duration'] = time.time() - start_time
                    return result

                result['validation_passed'] = True

            # Store data asynchronously
            storage_success = await loop.run_in_executor(
                executor,
                self.storage.store_data,
                data, symbol
            )

            if storage_success:
                result['storage_success'] = True
                result['success'] = True
                self._cache_data(symbol, interval, data)

                self.metrics['data_processed'] += len(data)
                self.metrics['successful_operations'] += 1
            else:
                result['errors'].append("Storage failed")

        except Exception as e:
            result['errors'].append(str(e))
            result['duration'] = time.time() - start_time
            self.metrics['errors'] += 1
            self.logger.error(f"Async pipeline failed for {symbol} {interval}: {e}")

        result['duration'] = time.time() - start_time
        return result

    async def batch_process_symbols_async(
        self,
        symbols: List[str],
        interval: str,
        start_date: str,
        end_date: Optional[str] = None,
        max_concurrent: int = 5
    ) -> Dict[str, Dict[str, Any]]:
        """
        Process multiple symbols concurrently with async operations.

        Args:
            symbols: List of trading symbols
            interval: Time interval
            start_date: Start date
            end_date: End date (optional)
            max_concurrent: Maximum concurrent operations

        Returns:
            Results for each symbol
        """
        semaphore = asyncio.Semaphore(max_concurrent)
        results: Dict[str, Dict[str, Any]] = {}

        async def process_symbol(symbol: str) -> None:
            async with semaphore:
                result = await self.fetch_and_store_data_async(
                    symbol, interval, start_date, end_date
                )
                results[symbol] = result

        # Create tasks for all symbols
        tasks = [process_symbol(symbol) for symbol in symbols]

        # Execute concurrently
        await asyncio.gather(*tasks, return_exceptions=True)

        return results

    def process_data(self, data: pd.DataFrame, symbol: str) -> Optional[pd.DataFrame]:
        """
        Process data through validation and storage pipeline.

        Args:
            data: DataFrame to process
            symbol: Trading symbol

        Returns:
            Processed DataFrame or None if processing failed
        """
        try:
            if data is None or data.empty:
                self.logger.warning(f"No data to process for {symbol}")
                return None

            # Validate data
            validation_result = self.validator.validate_ohlcv(data)
            if not validation_result['valid']:
                self.logger.error(f"Data validation failed for {symbol}: {validation_result['errors']}")
                return None

            # Store data
            storage_success = self.storage.store_data(data, symbol)
            if not storage_success:
                self.logger.error(f"Data storage failed for {symbol}")
                return None

            # Cache processed data
            self._cache_data(symbol, 'processed', None, None, data)

            self.logger.info(f"Successfully processed {len(data)} records for {symbol}")
            return data

        except Exception as e:
            self.logger.error(f"Data processing failed for {symbol}: {e}")
            return None

    def get_data(
        self,
        symbol: str,
        interval: str,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        use_cache: bool = True
    ) -> Optional[pd.DataFrame]:
        """
        Retrieve data from storage with caching.

        Args:
            symbol: Trading symbol
            interval: Time interval
            start_date: Start date filter
            end_date: End date filter
            use_cache: Whether to use cache

        Returns:
            DataFrame or None
        """
        # Check cache first
        if use_cache:
            cached_data = self._get_cached_data(symbol, interval, start_date, end_date)
            if cached_data is not None:
                self.metrics['cache_hits'] += 1
                return cached_data

        # Load from storage
        data = self.storage.load_historical_data(symbol, interval, start_date, end_date)

        # Cache the loaded data
        if data is not None and use_cache:
            self._cache_data(symbol, interval, start_date, end_date, data)

        return data

    def update_symbol_data(
        self,
        symbol: str,
        interval: str,
        days_back: int = 7
    ) -> Dict[str, Any]:
        """
        Update data for a symbol with recent data.

        Args:
            symbol: Trading symbol
            interval: Time interval
            days_back: Days of historical data to update

        Returns:
            Update results
        """
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days_back)

        return self.fetch_and_store_data(
            symbol=symbol,
            interval=interval,
            start_date=start_date.strftime('%Y-%m-%d'),
            end_date=end_date.strftime('%Y-%m-%d'),
            validate=True,
            force_refresh=True
        )

    def batch_update_symbols(
        self,
        symbols: List[str],
        intervals: List[str],
        days_back: int = 7
    ) -> Dict[str, Any]:
        """
        Batch update multiple symbols and intervals.

        Args:
            symbols: List of trading symbols
            intervals: List of time intervals
            days_back: Days of data to update

        Returns:
            Batch update results
        """
        results = {
            'total_operations': 0,
            'successful_operations': 0,
            'failed_operations': 0,
            'total_rows_processed': 0,
            'total_duration': 0.0,
            'symbol_results': {}
        }

        start_time = time.time()

        for symbol in symbols:
            results['symbol_results'][symbol] = {}

            for interval in intervals:
                results['total_operations'] += 1

                self.logger.info(f"Batch updating {symbol} {interval}")

                result = self.update_symbol_data(symbol, interval, days_back)

                results['symbol_results'][symbol][interval] = result
                results['total_rows_processed'] += result.get('rows_processed', 0)
                results['total_duration'] += result.get('duration', 0)

                if result.get('success', False):
                    results['successful_operations'] += 1
                else:
                    results['failed_operations'] += 1

        results['total_duration'] = time.time() - start_time

        self.logger.info(f"Batch update completed: {results['successful_operations']}/{results['total_operations']} successful")

        return results

    def get_pipeline_status(self) -> Dict[str, Any]:
        """Get comprehensive pipeline status."""
        status = {
            'components': {
                'binance_client': {
                    'healthy': self.client.is_healthy(),
                    'request_stats': self.client.get_request_stats()
                },
                'data_validator': {
                    'initialized': True
                },
                'data_storage': {
                    'data_info': self.storage.get_data_info()
                }
            },
            'metrics': self.metrics.copy(),
            'cache': {
                'entries': len(self.cache),
                'ttl_seconds': self.cache_ttl
            },
            'timestamp': datetime.now().isoformat()
        }

        return status

    def validate_data_quality(
        self,
        symbol: str,
        interval: str,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Perform comprehensive data quality validation.

        Args:
            symbol: Trading symbol
            interval: Time interval
            start_date: Start date
            end_date: End date

        Returns:
            Quality validation results
        """
        # Get data
        data = self.get_data(symbol, interval, start_date, end_date)

        if data is None or data.empty:
            return {
                'symbol': symbol,
                'interval': interval,
                'data_available': False,
                'quality_score': 0.0,
                'issues': ['No data available']
            }

        # Validate quality
        validation = self.validator.validate_ohlcv_data(data, symbol)

        # Generate report
        report = self.validator.generate_quality_report(validation)

        result = {
            'symbol': symbol,
            'interval': interval,
            'data_available': True,
            'rows': len(data),
            'date_range': {
                'start': data['timestamp'].min().isoformat() if not data.empty else None,
                'end': data['timestamp'].max().isoformat() if not data.empty else None
            },
            'quality_score': validation['quality_score'],
            'is_valid': validation['is_valid'],
            'issues': validation['issues'],
            'report': report
        }

        return result

    def _get_cached_data(
        self,
        symbol: str,
        interval: str,
        start_date: Optional[str],
        end_date: Optional[str]
    ) -> Optional[pd.DataFrame]:
        """Get data from cache if available and fresh."""
        cache_key = f"{symbol}_{interval}_{start_date}_{end_date}"

        if cache_key in self.cache:
            cached_item = self.cache[cache_key]
            cache_time = cached_item['timestamp']

            # Check if cache is still fresh
            if (datetime.now() - cache_time).seconds < self.cache_ttl:
                return cached_item['data'].copy()

            # Remove expired cache
            del self.cache[cache_key]

        return None

    def _cache_data(
        self,
        symbol: str,
        interval: str,
        start_date: Optional[str],
        end_date: Optional[str],
        data: pd.DataFrame
    ):
        """Cache data with timestamp."""
        cache_key = f"{symbol}_{interval}_{start_date}_{end_date}"

        self.cache[cache_key] = {
            'data': data.copy(),
            'timestamp': datetime.now()
        }

        # Limit cache size (keep only 50 most recent entries)
        if len(self.cache) > 50:
            # Remove oldest entries
            sorted_cache = sorted(self.cache.items(), key=lambda x: x[1]['timestamp'])
            for old_key, _ in sorted_cache[:-50]:
                del self.cache[old_key]

    def clear_cache(self):
        """Clear all cached data."""
        self.cache.clear()
        self.logger.info("Cache cleared")

    def optimize_storage(self):
        """Optimize data storage (cleanup, compression, etc.)."""
        try:
            # Cleanup old cache entries
            now = datetime.now()
            expired_keys = []

            for key, item in self.cache.items():
                if (now - item['timestamp']).seconds > self.cache_ttl:
                    expired_keys.append(key)

            for key in expired_keys:
                del self.cache[key]

            # Storage cleanup
            self.storage.cleanup_old_data(days_to_keep=365)

            self.logger.info(f"Storage optimization completed. Removed {len(expired_keys)} expired cache entries")

        except Exception as e:
            self.logger.error(f"Storage optimization failed: {e}")

    def export_data(
        self,
        symbol: str,
        interval: str,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        format: str = 'csv'
    ) -> Optional[str]:
        """
        Export data to various formats.

        Args:
            symbol: Trading symbol
            interval: Time interval
            start_date: Start date
            end_date: End date
            format: Export format ('csv', 'json', 'parquet')

        Returns:
            Path to exported file or None
        """
        try:
            # Get data
            data = self.get_data(symbol, interval, start_date, end_date)
            if data is None or data.empty:
                return None

            # Generate filename
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"{symbol}_{interval}_{timestamp}.{format}"

            # Export based on format
            if format == 'csv':
                data.to_csv(filename, index=False)
            elif format == 'json':
                data.to_json(filename, orient='records', date_format='iso')
            elif format == 'parquet':
                data.to_parquet(filename, index=False)
            else:
                self.logger.error(f"Unsupported export format: {format}")
                return None

            self.logger.info(f"Data exported to {filename}")
            return filename

        except Exception as e:
            self.logger.error(f"Export failed: {e}")
            return None
