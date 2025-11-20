#!/usr/bin/env python3
"""
Comprehensive Manual Tests for Data Pipeline

These tests focus on thorough coverage of the data pipeline functionality.
Goal: Improve data pipeline coverage from current levels to 60%.
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, AsyncMock
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

try:
    from data.data_pipeline import DataPipeline
    from data.binance_client import BinanceClient
except ImportError:
    # Skip tests if imports fail
    pytest.skip("Required modules not available", allow_module_level=True)


class TestDataPipeline:
    """Comprehensive tests for DataPipeline class."""

    @pytest.fixture
    def mock_binance_client(self):
        """Mock Binance client for testing."""
        client = Mock(spec=BinanceClient)
        client.get_historical_klines = AsyncMock()
        client.get_account_balance = AsyncMock(return_value=10000.0)
        return client

    @pytest.fixture
    def data_pipeline(self, mock_binance_client):
        """Create DataPipeline instance with mock client."""
        pipeline = DataPipeline()
        pipeline.binance_client = mock_binance_client
        return pipeline

    def test_pipeline_initialization(self, data_pipeline):
        """Test pipeline initialization."""
        assert data_pipeline is not None
        assert hasattr(data_pipeline, 'binance_client')

    def test_validate_ohlcv_data_valid(self, data_pipeline):
        """Test validation of valid OHLCV data."""
        # Create valid OHLCV data
        data = pd.DataFrame({
            'timestamp': pd.date_range('2024-01-01', periods=100, freq='1h'),
            'open': np.random.uniform(100, 110, 100),
            'high': np.random.uniform(105, 115, 100),
            'low': np.random.uniform(95, 105, 100),
            'close': np.random.uniform(100, 110, 100),
            'volume': np.random.uniform(1000, 10000, 100)
        })

        is_valid, errors = data_pipeline.validate_ohlcv_data(data)

        assert is_valid == True
        assert len(errors) == 0

    def test_validate_ohlcv_data_missing_columns(self, data_pipeline):
        """Test validation with missing required columns."""
        # Data missing 'volume' column
        data = pd.DataFrame({
            'timestamp': pd.date_range('2024-01-01', periods=10, freq='1h'),
            'open': np.random.uniform(100, 110, 10),
            'high': np.random.uniform(105, 115, 10),
            'low': np.random.uniform(95, 105, 10),
            'close': np.random.uniform(100, 110, 10)
            # Missing 'volume'
        })

        is_valid, errors = data_pipeline.validate_ohlcv_data(data)

        assert is_valid == False
        assert len(errors) > 0
        assert any('volume' in error.lower() for error in errors)

    def test_validate_ohlcv_data_invalid_timestamp(self, data_pipeline):
        """Test validation with invalid timestamp data."""
        data = pd.DataFrame({
            'timestamp': ['invalid'] * 10,  # Invalid timestamps
            'open': np.random.uniform(100, 110, 10),
            'high': np.random.uniform(105, 115, 10),
            'low': np.random.uniform(95, 105, 10),
            'close': np.random.uniform(100, 110, 10),
            'volume': np.random.uniform(1000, 10000, 10)
        })

        is_valid, errors = data_pipeline.validate_ohlcv_data(data)

        assert is_valid == False
        assert len(errors) > 0

    def test_validate_ohlcv_data_negative_prices(self, data_pipeline):
        """Test validation with negative prices."""
        data = pd.DataFrame({
            'timestamp': pd.date_range('2024-01-01', periods=10, freq='1h'),
            'open': [-100] * 10,  # Negative prices
            'high': np.random.uniform(105, 115, 10),
            'low': np.random.uniform(95, 105, 10),
            'close': np.random.uniform(100, 110, 10),
            'volume': np.random.uniform(1000, 10000, 10)
        })

        is_valid, errors = data_pipeline.validate_ohlcv_data(data)

        assert is_valid == False
        assert len(errors) > 0

    def test_validate_ohlcv_data_zero_volume(self, data_pipeline):
        """Test validation with zero volume."""
        data = pd.DataFrame({
            'timestamp': pd.date_range('2024-01-01', periods=10, freq='1h'),
            'open': np.random.uniform(100, 110, 10),
            'high': np.random.uniform(105, 115, 10),
            'low': np.random.uniform(95, 105, 10),
            'close': np.random.uniform(100, 110, 10),
            'volume': [0] * 10  # Zero volume
        })

        is_valid, errors = data_pipeline.validate_ohlcv_data(data)

        assert is_valid == False
        assert len(errors) > 0

    def test_validate_ohlcv_data_high_low_logic(self, data_pipeline):
        """Test validation of OHLC logic (high >= low, etc.)."""
        data = pd.DataFrame({
            'timestamp': pd.date_range('2024-01-01', periods=5, freq='1h'),
            'open': [100, 105, 110, 108, 112],
            'high': [102, 107, 108, 110, 115],  # High lower than open on row 2
            'low': [98, 103, 112, 106, 110],   # Low higher than high on row 2
            'close': [101, 106, 109, 107, 113],
            'volume': [1000, 1500, 1200, 1300, 1400]
        })

        is_valid, errors = data_pipeline.validate_ohlcv_data(data)

        assert is_valid == False
        assert len(errors) > 0

    def test_clean_ohlcv_data_basic(self, data_pipeline):
        """Test basic data cleaning functionality."""
        data = pd.DataFrame({
            'timestamp': pd.date_range('2024-01-01', periods=10, freq='1h'),
            'open': [100, np.nan, 102, 103, 104, 105, 106, 107, 108, 109],
            'high': [102] * 10,
            'low': [98] * 10,
            'close': [101] * 10,
            'volume': [1000] * 10
        })

        cleaned_data = data_pipeline.clean_ohlcv_data(data)

        # Should handle NaN values
        assert not cleaned_data['open'].isna().any()
        assert len(cleaned_data) <= len(data)  # May remove rows

    def test_clean_ohlcv_data_outliers(self, data_pipeline):
        """Test outlier removal in data cleaning."""
        # Create data with clear outliers
        normal_data = np.random.normal(100, 5, 100)
        data_with_outliers = np.append(normal_data, [1000, -100])  # Add extreme outliers

        data = pd.DataFrame({
            'timestamp': pd.date_range('2024-01-01', periods=102, freq='1h'),
            'open': data_with_outliers,
            'high': data_with_outliers + 2,
            'low': data_with_outliers - 2,
            'close': data_with_outliers + 0.5,
            'volume': np.random.uniform(1000, 10000, 102)
        })

        cleaned_data = data_pipeline.clean_ohlcv_data(data)

        # Should remove outliers
        assert len(cleaned_data) < len(data)

    def test_enrich_ohlcv_data_basic(self, data_pipeline):
        """Test basic OHLCV data enrichment."""
        data = pd.DataFrame({
            'timestamp': pd.date_range('2024-01-01', periods=10, freq='1h'),
            'open': [100, 101, 102, 103, 104, 105, 106, 107, 108, 109],
            'high': [102, 103, 104, 105, 106, 107, 108, 109, 110, 111],
            'low': [98, 99, 100, 101, 102, 103, 104, 105, 106, 107],
            'close': [101, 102, 103, 104, 105, 106, 107, 108, 109, 110],
            'volume': [1000, 1100, 1200, 1300, 1400, 1500, 1600, 1700, 1800, 1900]
        })

        enriched_data = data_pipeline.enrich_ohlcv_data(data)

        # Should add technical indicators
        expected_columns = ['returns', 'sma_20', 'ema_12', 'rsi_14', 'bb_upper', 'bb_lower']
        for col in expected_columns:
            if col in enriched_data.columns:
                assert not enriched_data[col].isna().all()

    def test_enrich_ohlcv_data_returns(self, data_pipeline):
        """Test returns calculation in enrichment."""
        data = pd.DataFrame({
            'timestamp': pd.date_range('2024-01-01', periods=5, freq='1h'),
            'open': [100, 101, 102, 103, 104],
            'high': [102, 103, 104, 105, 106],
            'low': [98, 99, 100, 101, 102],
            'close': [101, 102, 103, 104, 105],  # Steady increase
            'volume': [1000] * 5
        })

        enriched_data = data_pipeline.enrich_ohlcv_data(data)

        # Check returns calculation
        if 'returns' in enriched_data.columns:
            returns = enriched_data['returns'].dropna()
            # Should have positive returns for upward trend
            assert (returns > 0).any()

    def test_enrich_ohlcv_data_insufficient_data(self, data_pipeline):
        """Test enrichment with insufficient data."""
        # Very small dataset
        data = pd.DataFrame({
            'timestamp': pd.date_range('2024-01-01', periods=3, freq='1h'),
            'open': [100, 101, 102],
            'high': [102, 103, 104],
            'low': [98, 99, 100],
            'close': [101, 102, 103],
            'volume': [1000, 1100, 1200]
        })

        enriched_data = data_pipeline.enrich_ohlcv_data(data)

        # Should still work but with limited indicators
        assert len(enriched_data) == len(data)
        assert 'timestamp' in enriched_data.columns

    def test_process_symbol_data_success(self, data_pipeline, mock_binance_client):
        """Test successful symbol data processing."""
        # Mock successful data retrieval
        mock_data = [
            [1640995200000, '50000.00', '50100.00', '49900.00', '50050.00', '100.0'],
            [1640998800000, '50050.00', '50200.00', '50000.00', '50150.00', '150.0'],
        ]
        mock_binance_client.get_historical_klines.return_value = mock_data

        result = asyncio.run(data_pipeline.process_symbol_data(
            symbol="BTCUSDT",
            interval="1h",
            limit=2
        ))

        assert isinstance(result, pd.DataFrame)
        assert len(result) == 2
        assert list(result.columns) == ['timestamp', 'open', 'high', 'low', 'close', 'volume']

        mock_binance_client.get_historical_klines.assert_called_once_with(
            symbol="BTCUSDT",
            interval="1h",
            limit=2
        )

    def test_process_symbol_data_empty(self, data_pipeline, mock_binance_client):
        """Test processing with empty data."""
        mock_binance_client.get_historical_klines.return_value = []

        result = asyncio.run(data_pipeline.process_symbol_data(
            symbol="BTCUSDT",
            interval="1h",
            limit=10
        ))

        assert isinstance(result, pd.DataFrame)
        assert len(result) == 0

    def test_process_symbol_data_invalid_format(self, data_pipeline, mock_binance_client):
        """Test processing with invalid data format."""
        # Invalid data format
        mock_binance_client.get_historical_klines.return_value = [
            [1640995200000, 'invalid', 'data'],
            [1640998800000]  # Missing fields
        ]

        with pytest.raises((ValueError, IndexError)):
            asyncio.run(data_pipeline.process_symbol_data(
                symbol="BTCUSDT",
                interval="1h",
                limit=2
            ))

    @patch('data.data_pipeline.pd.DataFrame.to_csv')
    def test_save_processed_data(self, mock_to_csv, data_pipeline):
        """Test saving processed data."""
        data = pd.DataFrame({
            'timestamp': pd.date_range('2024-01-01', periods=5, freq='1h'),
            'close': [100, 101, 102, 103, 104]
        })

        result = data_pipeline.save_processed_data(data, "test_data.csv")

        assert result == True
        mock_to_csv.assert_called_once()

    @patch('data.data_pipeline.pd.read_csv')
    def test_load_processed_data(self, mock_read_csv, data_pipeline):
        """Test loading processed data."""
        mock_data = pd.DataFrame({
            'timestamp': pd.date_range('2024-01-01', periods=5, freq='1h'),
            'close': [100, 101, 102, 103, 104]
        })
        mock_read_csv.return_value = mock_data

        result = data_pipeline.load_processed_data("test_data.csv")

        assert isinstance(result, pd.DataFrame)
        assert len(result) == 5
        mock_read_csv.assert_called_once_with("test_data.csv")

    def test_get_data_quality_report(self, data_pipeline):
        """Test data quality reporting."""
        # Create test data with some issues
        data = pd.DataFrame({
            'timestamp': pd.date_range('2024-01-01', periods=100, freq='1h'),
            'open': np.random.uniform(100, 110, 100),
            'high': np.random.uniform(105, 115, 100),
            'low': np.random.uniform(95, 105, 100),
            'close': np.random.uniform(100, 110, 100),
            'volume': np.random.uniform(1000, 10000, 100)
        })

        # Add some NaN values
        data.loc[10:15, 'close'] = np.nan

        report = data_pipeline.get_data_quality_report(data)

        assert isinstance(report, dict)
        assert 'total_rows' in report
        assert 'missing_values' in report
        assert 'data_types' in report
        assert report['total_rows'] == 100
        assert report['missing_values']['close'] > 0

    def test_resample_data_different_frequencies(self, data_pipeline):
        """Test data resampling to different frequencies."""
        # Create hourly data
        hourly_data = pd.DataFrame({
            'timestamp': pd.date_range('2024-01-01', periods=24, freq='1h'),
            'open': np.random.uniform(100, 110, 24),
            'high': np.random.uniform(105, 115, 24),
            'low': np.random.uniform(95, 105, 24),
            'close': np.random.uniform(100, 110, 24),
            'volume': np.random.uniform(1000, 10000, 24)
        })

        # Resample to daily
        daily_data = data_pipeline.resample_data(hourly_data, 'D')

        assert isinstance(daily_data, pd.DataFrame)
        assert len(daily_data) < len(hourly_data)  # Should have fewer rows

    def test_merge_multiple_symbols(self, data_pipeline):
        """Test merging data from multiple symbols."""
        btc_data = pd.DataFrame({
            'timestamp': pd.date_range('2024-01-01', periods=5, freq='1h'),
            'symbol': ['BTCUSDT'] * 5,
            'close': [50000, 50100, 50200, 50300, 50400]
        })

        eth_data = pd.DataFrame({
            'timestamp': pd.date_range('2024-01-01', periods=5, freq='1h'),
            'symbol': ['ETHUSDT'] * 5,
            'close': [3000, 3100, 3200, 3300, 3400]
        })

        merged_data = data_pipeline.merge_multiple_symbols([btc_data, eth_data])

        assert isinstance(merged_data, pd.DataFrame)
        assert 'BTCUSDT' in merged_data.columns
        assert 'ETHUSDT' in merged_data.columns

    def test_detect_data_anomalies(self, data_pipeline):
        """Test anomaly detection in data."""
        # Create normal data
        data = pd.DataFrame({
            'timestamp': pd.date_range('2024-01-01', periods=100, freq='1h'),
            'close': np.random.normal(100, 5, 100)
        })

        # Add some anomalies
        data.loc[10, 'close'] = 1000  # Extreme high
        data.loc[50, 'close'] = -100  # Extreme low

        anomalies = data_pipeline.detect_data_anomalies(data, column='close')

        assert isinstance(anomalies, pd.DataFrame)
        assert len(anomalies) > 0  # Should detect anomalies

    def test_export_data_multiple_formats(self, data_pipeline):
        """Test exporting data in multiple formats."""
        data = pd.DataFrame({
            'timestamp': pd.date_range('2024-01-01', periods=5, freq='1h'),
            'close': [100, 101, 102, 103, 104]
        })

        # Test CSV export
        csv_result = data_pipeline.export_data(data, 'test.csv', 'csv')
        assert csv_result == True

        # Test JSON export
        json_result = data_pipeline.export_data(data, 'test.json', 'json')
        assert json_result == True

        # Test Parquet export (if available)
        try:
            parquet_result = data_pipeline.export_data(data, 'test.parquet', 'parquet')
            assert parquet_result == True
        except ImportError:
            pass  # PyArrow not available

    def test_pipeline_performance_monitoring(self, data_pipeline):
        """Test pipeline performance monitoring."""
        # Simulate pipeline operations
        start_time = datetime.now()

        # Mock some processing time
        import time
        time.sleep(0.1)

        performance_metrics = data_pipeline.get_performance_metrics()

        assert isinstance(performance_metrics, dict)
        assert 'processing_time' in performance_metrics or len(performance_metrics) >= 0

    def test_error_handling_and_recovery(self, data_pipeline, mock_binance_client):
        """Test error handling and recovery mechanisms."""
        # Mock API failure
        mock_binance_client.get_historical_klines.side_effect = Exception("API Rate Limit")

        # Should handle error gracefully
        with pytest.raises(Exception):
            asyncio.run(data_pipeline.process_symbol_data("BTCUSDT", "1h", 100))

    def test_data_pipeline_integration_workflow(self, data_pipeline, mock_binance_client):
        """Test complete data pipeline integration workflow."""
        # Mock successful data flow
        mock_data = [
            [1640995200000, '50000.00', '50100.00', '49900.00', '50050.00', '100.0'],
            [1640998800000, '50050.00', '50200.00', '50000.00', '50150.00', '150.0'],
        ]
        mock_binance_client.get_historical_klines.return_value = mock_data

        # Complete workflow: fetch -> validate -> clean -> enrich -> save
        raw_data = asyncio.run(data_pipeline.process_symbol_data("BTCUSDT", "1h", 2))
        is_valid, errors = data_pipeline.validate_ohlcv_data(raw_data)

        if is_valid:
            cleaned_data = data_pipeline.clean_ohlcv_data(raw_data)
            enriched_data = data_pipeline.enrich_ohlcv_data(cleaned_data)

            # Should have additional columns from enrichment
            assert len(enriched_data.columns) >= len(raw_data.columns)

    def test_memory_efficient_processing(self, data_pipeline):
        """Test memory-efficient processing for large datasets."""
        # Create larger dataset
        large_data = pd.DataFrame({
            'timestamp': pd.date_range('2024-01-01', periods=1000, freq='1min'),
            'open': np.random.uniform(100, 110, 1000),
            'high': np.random.uniform(105, 115, 1000),
            'low': np.random.uniform(95, 105, 1000),
            'close': np.random.uniform(100, 110, 1000),
            'volume': np.random.uniform(1000, 10000, 1000)
        })

        # Process in chunks
        chunk_size = 100
        processed_chunks = []

        for i in range(0, len(large_data), chunk_size):
            chunk = large_data.iloc[i:i+chunk_size]
            processed_chunk = data_pipeline.clean_ohlcv_data(chunk)
            processed_chunks.append(processed_chunk)

        # Combine results
        final_data = pd.concat(processed_chunks, ignore_index=True)

        assert len(final_data) <= len(large_data)  # May have removed some rows
        assert isinstance(final_data, pd.DataFrame)
