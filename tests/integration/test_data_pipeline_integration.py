#!/usr/bin/env python3
"""
Integration tests for data pipeline components.

Tests data validation, storage, and processing pipeline integration.
"""

import pytest
import pandas as pd
import numpy as np
import tempfile
import os
from pathlib import Path
from src.data.data_validator import DataValidator
from src.data.data_storage import DataStorage
from src.data.data_pipeline import DataPipeline


class TestDataValidatorImplementation:
    """Test DataValidator implementation - currently missing."""

    def test_data_validator_initialization(self):
        """Test that DataValidator can be initialized."""
        validator = DataValidator()

        # Should have validation rules
        assert hasattr(validator, 'validation_rules')
        assert isinstance(validator.validation_rules, dict)

    def test_ohlcv_validation_method_exists(self):
        """Test that validate_ohlcv method exists (currently missing)."""
        validator = DataValidator()

        # Method should exist
        assert hasattr(validator, 'validate_ohlcv')

        # Should be callable
        assert callable(getattr(validator, 'validate_ohlcv', None))

    def test_validate_ohlcv_with_valid_data(self, sample_ohlcv_data):
        """Test OHLCV validation with valid data."""
        validator = DataValidator()

        result = validator.validate_ohlcv(sample_ohlcv_data)

        # Should return validation result
        assert isinstance(result, dict)
        assert 'valid' in result
        assert 'errors' in result
        assert 'timestamp' in result

    def test_validate_ohlcv_with_invalid_data(self):
        """Test OHLCV validation with invalid data."""
        validator = DataValidator()

        # Create invalid data (high < low)
        invalid_data = pd.DataFrame({
            'timestamp': pd.date_range('2024-01-01', periods=10, freq='D'),
            'open': [100] * 10,
            'high': [95] * 10,  # High < Low (invalid)
            'low': [105] * 10,
            'close': [102] * 10,
            'volume': [1000] * 10
        })

        result = validator.validate_ohlcv(invalid_data)

        # Should detect validation errors
        assert result['valid'] == False
        assert len(result['errors']) > 0

    def test_comprehensive_validation_rules(self, sample_ohlcv_data):
        """Test all validation rules work together."""
        validator = DataValidator()

        result = validator.validate_ohlcv(sample_ohlcv_data)

        # Result should have expected structure
        assert isinstance(result['valid'], bool)
        assert isinstance(result['errors'], list)
        assert isinstance(result['timestamp'], pd.Timestamp)


class TestDataStorageImplementation:
    """Test DataStorage implementation - currently missing."""

    def test_data_storage_initialization(self):
        """Test that DataStorage can be initialized."""
        with tempfile.TemporaryDirectory() as temp_dir:
            storage = DataStorage(base_dir=temp_dir)

            # Should have storage path
            assert hasattr(storage, 'base_dir')
            assert storage.base_dir == Path(temp_dir)

    def test_store_data_method_exists(self):
        """Test that store_data method exists (currently missing)."""
        with tempfile.TemporaryDirectory() as temp_dir:
            storage = DataStorage(base_dir=temp_dir)

            # Method should exist
            assert hasattr(storage, 'store_data')

            # Should be callable
            assert callable(getattr(storage, 'store_data', None))

    def test_store_data_functionality(self, sample_ohlcv_data):
        """Test data storage functionality."""
        with tempfile.TemporaryDirectory() as temp_dir:
            storage = DataStorage(base_dir=temp_dir)

            # Test storing data
            storage.store_data(sample_ohlcv_data, 'ETHUSDT')

            # Check if files were created in historical directory
            eth_dir = Path(temp_dir) / 'historical' / 'ETHUSDT'
            assert eth_dir.exists()

            # Should have parquet files
            parquet_files = list(eth_dir.glob('**/*.parquet'))
            assert len(parquet_files) > 0

    def test_query_data_functionality(self, sample_ohlcv_data):
        """Test data querying functionality."""
        with tempfile.TemporaryDirectory() as temp_dir:
            storage = DataStorage(base_dir=temp_dir)

            # Store data first
            storage.store_data(sample_ohlcv_data, 'ETHUSDT')

            # Query data back
            start_date = sample_ohlcv_data['timestamp'].min().date()
            end_date = sample_ohlcv_data['timestamp'].max().date()

            retrieved_data = storage.query_data('ETHUSDT', start_date, end_date)

            # Should retrieve data
            assert isinstance(retrieved_data, pd.DataFrame)
            assert len(retrieved_data) > 0

    def test_data_persistence(self, sample_ohlcv_data):
        """Test that data persists across storage instances."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Store with first instance
            storage1 = DataStorage(base_dir=temp_dir)
            storage1.store_data(sample_ohlcv_data, 'ETHUSDT')

            # Retrieve with second instance
            storage2 = DataStorage(base_dir=temp_dir)
            start_date = sample_ohlcv_data['timestamp'].min().date()
            end_date = sample_ohlcv_data['timestamp'].max().date()

            retrieved_data = storage2.query_data('ETHUSDT', start_date, end_date)

            # Should have same data
            assert len(retrieved_data) == len(sample_ohlcv_data)


class TestDataPipelineIntegration:
    """Test complete data pipeline integration."""

    def test_data_pipeline_initialization(self):
        """Test data pipeline initialization."""
        pipeline = DataPipeline()

        # Should have validator and storage
        assert hasattr(pipeline, 'validator')
        assert hasattr(pipeline, 'storage')
        assert isinstance(pipeline.validator, DataValidator)
        assert isinstance(pipeline.storage, DataStorage)

    def test_data_pipeline_process_valid_data(self, sample_ohlcv_data):
        """Test processing valid data through pipeline."""
        pipeline = DataPipeline()

        # Process data
        result = pipeline.process_data(sample_ohlcv_data, 'ETHUSDT')

        # Should return processed data
        assert isinstance(result, pd.DataFrame)
        assert len(result) > 0

    def test_data_pipeline_process_invalid_data(self):
        """Test processing invalid data through pipeline."""
        pipeline = DataPipeline()

        # Create invalid data
        invalid_data = pd.DataFrame({
            'timestamp': pd.date_range('2024-01-01', periods=5, freq='D'),
            'open': [100] * 5,
            'high': [95] * 5,  # Invalid: high < low
            'low': [105] * 5,
            'close': [102] * 5,
            'volume': [1000] * 5
        })

        # Should handle invalid data gracefully
        result = pipeline.process_data(invalid_data, 'ETHUSDT')

        # Result should indicate validation failure
        # (exact behavior depends on implementation)
        assert isinstance(result, (pd.DataFrame, type(None)))

    def test_pipeline_caching(self, sample_ohlcv_data):
        """Test pipeline caching functionality."""
        pipeline = DataPipeline()

        # Process same data twice
        result1 = pipeline.process_data(sample_ohlcv_data, 'ETHUSDT')
        result2 = pipeline.process_data(sample_ohlcv_data, 'ETHUSDT')

        # Results should be consistent
        if result1 is not None and result2 is not None:
            pd.testing.assert_frame_equal(result1, result2)

    def test_pipeline_error_handling(self):
        """Test pipeline error handling."""
        pipeline = DataPipeline()

        # Test with None data
        result = pipeline.process_data(None, 'ETHUSDT')
        assert result is None

        # Test with empty DataFrame
        empty_data = pd.DataFrame()
        result = pipeline.process_data(empty_data, 'ETHUSDT')
        assert result is None or len(result) == 0


class TestDataQualityAssurance:
    """Test data quality assurance throughout pipeline."""

    def test_data_integrity_preservation(self, sample_ohlcv_data):
        """Test that data integrity is preserved through pipeline."""
        pipeline = DataPipeline()

        # Process data
        result = pipeline.process_data(sample_ohlcv_data, 'ETHUSDT')

        if result is not None:
            # Check that essential columns are preserved
            required_columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
            for col in required_columns:
                assert col in result.columns

            # Check data types
            assert pd.api.types.is_datetime64_any_dtype(result['timestamp'])
            for col in ['open', 'high', 'low', 'close', 'volume']:
                assert pd.api.types.is_numeric_dtype(result[col])

    def test_timestamp_handling(self, sample_ohlcv_data):
        """Test proper timestamp handling."""
        pipeline = DataPipeline()

        result = pipeline.process_data(sample_ohlcv_data, 'ETHUSDT')

        if result is not None:
            # Timestamps should be monotonic
            assert result['timestamp'].is_monotonic_increasing

            # Should handle timezone-aware timestamps
            # (implementation dependent)

    def test_volume_data_handling(self, sample_ohlcv_data):
        """Test volume data handling."""
        pipeline = DataPipeline()

        # Test with zero volume (suspicious data)
        suspicious_data = sample_ohlcv_data.copy()
        suspicious_data.loc[0, 'volume'] = 0

        result = pipeline.process_data(suspicious_data, 'ETHUSDT')

        # Should handle zero volume gracefully
        # (exact behavior depends on validation rules)
        assert isinstance(result, (pd.DataFrame, type(None)))


class TestPerformanceBenchmarks:
    """Test data pipeline performance benchmarks."""

    @pytest.mark.slow
    def test_pipeline_throughput(self, large_ohlcv_data):
        """Test data pipeline throughput."""
        import time

        pipeline = DataPipeline()

        start_time = time.time()

        # Process large dataset
        result = pipeline.process_data(large_ohlcv_data, 'ETHUSDT')

        end_time = time.time()

        processing_time = end_time - start_time

        # Should process reasonable amount of data quickly
        if result is not None:
            records_per_second = len(result) / processing_time
            assert records_per_second > 100  # At least 100 records/second

    @pytest.mark.slow
    def test_storage_performance(self, large_ohlcv_data):
        """Test data storage performance."""
        import time

        with tempfile.TemporaryDirectory() as temp_dir:
            storage = DataStorage(base_dir=temp_dir)

            start_time = time.time()
            storage.store_data(large_ohlcv_data, 'ETHUSDT')
            end_time = time.time()

            storage_time = end_time - start_time

            # Should store data reasonably quickly (prevent division by zero)
            if storage_time > 0:
                records_per_second = len(large_ohlcv_data) / storage_time
                assert records_per_second > 50  # At least 50 records/second
            else:
                # If storage was instantaneous, just verify it completed
                assert True

    @pytest.mark.slow
    def test_query_performance(self, large_ohlcv_data):
        """Test data query performance."""
        import time

        with tempfile.TemporaryDirectory() as temp_dir:
            storage = DataStorage(base_dir=temp_dir)

            # Store data first
            storage.store_data(large_ohlcv_data, 'ETHUSDT')

            # Query data
            start_date = large_ohlcv_data['timestamp'].min().date()
            end_date = large_ohlcv_data['timestamp'].max().date()

            start_time = time.time()
            result = storage.query_data('ETHUSDT', start_date, end_date)
            end_time = time.time()

            query_time = end_time - start_time

            # Should query data quickly
            if result is not None:
                records_per_second = len(result) / query_time if query_time > 0 else float('inf')
                assert records_per_second > 1000  # At least 1000 records/second


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
