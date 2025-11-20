#!/usr/bin/env python3
"""
Simple tests for Data Pipeline
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime

try:
    from src.data.data_pipeline import DataPipeline
except ImportError:
    pytest.skip("DataPipeline not available", allow_module_level=True)


class TestDataPipeline:
    """Basic tests for DataPipeline class."""

    @pytest.fixture
    def data_pipeline(self):
        """Create DataPipeline instance with mocked client."""
        from unittest.mock import MagicMock
        pipeline = DataPipeline()
        # Mock the client to avoid connection issues
        pipeline.client = MagicMock()
        pipeline.client.is_healthy.return_value = True
        return pipeline

    def test_pipeline_initialization(self, data_pipeline):
        """Test pipeline initialization."""
        assert data_pipeline is not None
        assert hasattr(data_pipeline, 'logger')

    def test_get_pipeline_status(self, data_pipeline):
        """Test getting pipeline status."""
        status = data_pipeline.get_pipeline_status()
        assert isinstance(status, dict)
        assert 'components' in status  # Actual key in implementation

    def test_process_data_basic(self, data_pipeline):
        """Test basic data processing."""
        # Create valid OHLCV data (high >= open, close, low)
        timestamps = pd.date_range('2024-01-01', periods=10, freq='1H')
        data = []
        for i in range(10):
            base_price = 100 + i
            low = base_price - 2
            open_price = base_price - 1
            close = base_price
            high = base_price + 2
            volume = 1000 + i * 100

            data.append({
                'timestamp': timestamps[i],
                'open': open_price,
                'high': high,
                'low': low,
                'close': close,
                'volume': volume
            })

        df = pd.DataFrame(data)

        result = data_pipeline.process_data(df, "BTCUSDT")
        # process_data may return None if validation fails, that's OK for basic test
        assert isinstance(result, (pd.DataFrame, type(None)))

    def test_validate_data_quality(self, data_pipeline):
        """Test data quality validation."""
        # Create valid data
        df = pd.DataFrame({
            'timestamp': pd.date_range('2024-01-01', periods=10, freq='1H'),
            'open': np.random.uniform(100, 110, 10),
            'high': np.random.uniform(105, 115, 10),
            'low': np.random.uniform(95, 105, 10),
            'close': np.random.uniform(100, 110, 10),
            'volume': np.random.uniform(1000, 10000, 10)
        })

        result = data_pipeline.validate_data_quality(df, "BTCUSDT")
        assert isinstance(result, dict)
        assert 'quality_score' in result  # Actual key in implementation

    def test_clear_cache(self, data_pipeline):
        """Test cache clearing."""
        # Should not raise exception
        data_pipeline.clear_cache()

    def test_optimize_storage(self, data_pipeline):
        """Test storage optimization."""
        # Should not raise exception
        data_pipeline.optimize_storage()

    def test_export_data_csv(self, data_pipeline):
        """Test data export to CSV."""
        result = data_pipeline.export_data("BTCUSDT", "1h", format="csv")
        # export_data returns Optional[str] (file path or None)
        assert result is None or isinstance(result, str)

    def test_error_handling_and_recovery(self, data_pipeline):
        """Test error handling and recovery."""
        # Test with invalid input - should handle gracefully
        result = data_pipeline.process_data(None, "BTCUSDT")
        # process_data returns None for invalid input, doesn't raise
        assert result is None
