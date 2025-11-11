#!/usr/bin/env python3
"""
Supreme System V5 - Data Validation Module

Comprehensive data validation for financial market data.
Ensures data quality, integrity, and compliance.
"""

import logging
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


class DataValidator:
    """
    Comprehensive data validator for financial market data.

    Validates:
    - Data completeness and structure
    - Price data integrity (OHLCV)
    - Volume data validation
    - Timestamp continuity and gaps
    - Statistical outliers detection
    - Cross-field consistency
    """

    def __init__(self):
        self.logger = logging.getLogger(__name__)

        # Validation thresholds
        self.max_price_change_pct = 0.50  # 50% max price change per interval
        self.min_volume_threshold = 0.01  # Minimum volume threshold
        self.max_gap_minutes = 60         # Maximum allowed gap in minutes
        self.outlier_std_threshold = 5.0  # Outlier detection threshold

        # Validation rules mapping
        self.validation_rules = {
            'ohlcv': self._validate_price_integrity,
            'volume': self._validate_volume_data,
            'timestamps': self._validate_timestamps,
            'statistics': self._validate_statistics,
            'cross_field': self._validate_cross_field_consistency
        }

    def validate_ohlcv_data(self, data: pd.DataFrame, symbol: str) -> Dict[str, Any]:
        """
        Comprehensive validation of OHLCV data.

        Args:
            data: DataFrame with OHLCV columns
            symbol: Trading symbol for context

        Returns:
            Validation results dictionary
        """
        results = {
            'symbol': symbol,
            'total_rows': len(data),
            'valid_rows': 0,
            'issues': [],
            'quality_score': 0.0,
            'is_valid': False
        }

        if data.empty:
            results['issues'].append('Empty dataset')
            return results

        # Basic structure validation
        structure_ok = self._validate_structure(data)
        if not structure_ok:
            results['issues'].extend(['Missing required columns', 'Invalid data types'])
            return results

        # Price integrity validation
        price_issues = self._validate_price_integrity(data)
        results['issues'].extend(price_issues)

        # Volume validation
        volume_issues = self._validate_volume_data(data)
        results['issues'].extend(volume_issues)

        # Timestamp validation
        timestamp_issues = self._validate_timestamps(data)
        results['issues'].extend(timestamp_issues)

        # Statistical validation
        stat_issues = self._validate_statistics(data)
        results['issues'].extend(stat_issues)

        # Cross-field consistency
        consistency_issues = self._validate_cross_field_consistency(data)
        results['issues'].extend(consistency_issues)

        # Calculate quality score
        results['valid_rows'] = len(data) - len(results['issues'])
        results['quality_score'] = max(0, (results['valid_rows'] / len(data)) * 100)
        results['is_valid'] = len(results['issues']) == 0

        self.logger.info(f"Data validation for {symbol}: {results['quality_score']:.1f}% quality, "
                        f"{len(results['issues'])} issues")

        return results

    def _validate_structure(self, data: pd.DataFrame) -> bool:
        """Validate basic data structure and required columns."""
        required_columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume']

        # Check for required columns
        missing_columns = [col for col in required_columns if col not in data.columns]
        if missing_columns:
            self.logger.error(f"Missing required columns: {missing_columns}")
            return False

        # Validate data types
        try:
            # Convert timestamp if needed
            if not pd.api.types.is_datetime64_any_dtype(data['timestamp']):
                data['timestamp'] = pd.to_datetime(data['timestamp'])

            # Convert numeric columns
            numeric_columns = ['open', 'high', 'low', 'close', 'volume']
            for col in numeric_columns:
                data[col] = pd.to_numeric(data[col], errors='coerce')

            # Check for NaN values in critical columns
            critical_columns = ['timestamp', 'close', 'volume']
            for col in critical_columns:
                if data[col].isnull().any():
                    self.logger.error(f"Null values found in {col}")
                    return False

        except Exception as e:
            self.logger.error(f"Data type conversion failed: {e}")
            return False

        return True

    def _validate_price_integrity(self, data: pd.DataFrame) -> List[str]:
        """Validate price data integrity."""
        issues = []

        try:
            # OHLC relationship validation
            invalid_ohlc = (
                (data['high'] < data['low']) |
                (data['open'] > data['high']) |
                (data['open'] < data['low']) |
                (data['close'] > data['high']) |
                (data['close'] < data['low'])
            )

            if invalid_ohlc.any():
                invalid_count = invalid_ohlc.sum()
                issues.append(f"Invalid OHLC relationships in {invalid_count} rows")

            # Price change validation (detect extreme moves)
            price_changes = data['close'].pct_change().abs()
            extreme_changes = price_changes > self.max_price_change_pct

            if extreme_changes.any():
                extreme_count = extreme_changes.sum()
                issues.append(f"Extreme price changes (>50%) in {extreme_count} rows")

            # Negative prices
            negative_prices = (data[['open', 'high', 'low', 'close']] <= 0).any(axis=1)
            if negative_prices.any():
                neg_count = negative_prices.sum()
                issues.append(f"Negative or zero prices in {neg_count} rows")

        except Exception as e:
            issues.append(f"Price integrity validation error: {e}")

        return issues

    def _validate_price_consistency(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Validate price consistency across OHLC."""
        errors = []

        # Check open/close within high/low range
        if all(col in data.columns for col in ['open', 'high', 'low', 'close']):
            invalid_open = ((data['open'] > data['high']) | (data['open'] < data['low'])).any()
            invalid_close = ((data['close'] > data['high']) | (data['close'] < data['low'])).any()

            if invalid_open:
                errors.append("Open prices outside high-low range")
            if invalid_close:
                errors.append("Close prices outside high-low range")

        return {'valid': len(errors) == 0, 'errors': errors}

    def _validate_volume_data(self, data: pd.DataFrame) -> List[str]:
        """Validate volume data."""
        issues = []

        try:
            # Negative volume
            negative_volume = data['volume'] < 0
            if negative_volume.any():
                neg_vol_count = negative_volume.sum()
                issues.append(f"Negative volume in {neg_vol_count} rows")

            # Zero volume (might be ok for some instruments)
            zero_volume = data['volume'] == 0
            if zero_volume.sum() > len(data) * 0.1:  # More than 10% zero volume
                zero_pct = (zero_volume.sum() / len(data)) * 100
                issues.append(f"High zero volume percentage: {zero_pct:.1f}%")

            # Extremely low volume
            low_volume = data['volume'] < self.min_volume_threshold
            if low_volume.any():
                low_vol_count = low_volume.sum()
                issues.append(f"Extremely low volume (<{self.min_volume_threshold}) in {low_vol_count} rows")

        except Exception as e:
            issues.append(f"Volume validation error: {e}")

        return issues

    def _validate_timestamps(self, data: pd.DataFrame) -> List[str]:
        """Validate timestamp continuity and ordering."""
        issues = []

        try:
            timestamps = data['timestamp'].sort_values()

            # Check for duplicates
            duplicates = timestamps.duplicated().sum()
            if duplicates > 0:
                issues.append(f"Duplicate timestamps: {duplicates}")

            # Check for monotonic ordering
            if not timestamps.is_monotonic_increasing:
                issues.append("Timestamps not in chronological order")

            # Check for gaps (if we can determine expected interval)
            if len(timestamps) > 1:
                time_diffs = timestamps.diff().dropna()
                median_diff = time_diffs.median()

                # Find gaps larger than expected
                if pd.notna(median_diff):
                    expected_max_gap = median_diff * 3  # Allow 3x normal gap
                    gaps = time_diffs > expected_max_gap
                    if gaps.any():
                        gap_count = gaps.sum()
                        max_gap_hours = (time_diffs.max() / pd.Timedelta(hours=1))
                        issues.append(f"Large time gaps ({gap_count} gaps, max {max_gap_hours:.1f}h)")

        except Exception as e:
            issues.append(f"Timestamp validation error: {e}")

        return issues

    def _validate_statistics(self, data: pd.DataFrame) -> List[str]:
        """Statistical validation for outlier detection."""
        issues = []

        try:
            # Price outlier detection using z-score
            for col in ['close', 'volume']:
                if col in data.columns:
                    values = data[col].dropna()
                    if len(values) > 10:  # Need minimum data for stats
                        mean_val = values.mean()
                        std_val = values.std()

                        if std_val > 0:
                            z_scores = np.abs((values - mean_val) / std_val)
                            outliers = z_scores > self.outlier_std_threshold

                            if outliers.any():
                                outlier_count = outliers.sum()
                                issues.append(f"Statistical outliers in {col}: {outlier_count} values")

        except Exception as e:
            issues.append(f"Statistical validation error: {e}")

        return issues

    def _validate_cross_field_consistency(self, data: pd.DataFrame) -> List[str]:
        """Validate consistency across related fields."""
        issues = []

        try:
            # Volume-price correlation check (basic sanity)
            if len(data) > 10:
                volume_price_corr = data['volume'].corr(data['close'])
                if pd.notna(volume_price_corr):
                    # Very high correlation might indicate data issues
                    if abs(volume_price_corr) > 0.95:
                        issues.append("Unusually high volume-price correlation")

            # Daily range validation
            daily_range = (data['high'] - data['low']) / data['close']
            extreme_range = daily_range > 0.20  # 20% daily range

            if extreme_range.any():
                extreme_count = extreme_range.sum()
                issues.append(f"Extreme daily ranges (>20%) in {extreme_count} rows")

        except Exception as e:
            issues.append(f"Cross-field validation error: {e}")

        return issues

    def clean_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Clean and sanitize data based on validation rules.

        Args:
            data: Raw data to clean

        Returns:
            Cleaned DataFrame
        """
        df = data.copy()

        try:
            # Remove rows with null timestamps
            df = df.dropna(subset=['timestamp'])

            # Remove rows with invalid OHLC relationships
            valid_ohlc = (
                (df['high'] >= df['low']) &
                (df['open'] <= df['high']) & (df['open'] >= df['low']) &
                (df['close'] <= df['high']) & (df['close'] >= df['low'])
            )
            df = df[valid_ohlc]

            # Remove negative prices and volumes
            df = df[(df['close'] > 0) & (df['volume'] >= 0)]

            # Remove extreme outliers (3-sigma rule)
            for col in ['close', 'volume']:
                if col in df.columns:
                    mean_val = df[col].mean()
                    std_val = df[col].std()
                    if std_val > 0:
                        z_scores = np.abs((df[col] - mean_val) / std_val)
                        df = df[z_scores <= 3]

            # Sort by timestamp
            df = df.sort_values('timestamp').reset_index(drop=True)

            # Remove duplicate timestamps (keep last)
            df = df.drop_duplicates(subset=['timestamp'], keep='last')

            self.logger.info(f"Data cleaning: {len(data)} -> {len(df)} rows")

        except Exception as e:
            self.logger.error(f"Data cleaning error: {e}")

        return df

    def validate_ohlcv(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Simple OHLCV validation method for testing framework.

        This is a simplified version that returns validation results
        compatible with the testing expectations.

        Args:
            data: DataFrame with OHLCV columns

        Returns:
            Validation result dictionary
        """
        errors = []

        try:
            # Basic validation checks
            if data.empty:
                return {
                    'valid': False,
                    'errors': ['Empty dataset'],
                    'timestamp': pd.Timestamp.now()
                }

            # Check required columns
            required_columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
            missing_columns = [col for col in required_columns if col not in data.columns]
            if missing_columns:
                errors.append(f"Missing required columns: {missing_columns}")

            # Check OHLC relationships
            if 'high' in data.columns and 'low' in data.columns:
                invalid_ohlc = (data['high'] < data['low']).any()
                if invalid_ohlc:
                    errors.append("High prices below low prices detected")

            # Check price ranges
            if 'open' in data.columns and 'high' in data.columns and 'low' in data.columns:
                invalid_open = ((data['open'] > data['high']) | (data['open'] < data['low'])).any()
                if invalid_open:
                    errors.append("Open prices outside high-low range")

            if 'close' in data.columns and 'high' in data.columns and 'low' in data.columns:
                invalid_close = ((data['close'] > data['high']) | (data['close'] < data['low'])).any()
                if invalid_close:
                    errors.append("Close prices outside high-low range")

            # Check for negative prices
            price_columns = ['open', 'high', 'low', 'close']
            for col in price_columns:
                if col in data.columns:
                    negative_prices = (data[col] <= 0).any()
                    if negative_prices:
                        errors.append(f"Negative or zero {col} prices detected")

            # Check volume
            if 'volume' in data.columns:
                negative_volume = (data['volume'] < 0).any()
                if negative_volume:
                    errors.append("Negative volume detected")

            # Check timestamp ordering
            if 'timestamp' in data.columns and len(data) > 1:
                if not data['timestamp'].is_monotonic_increasing:
                    errors.append("Timestamps not in ascending order")

        except Exception as e:
            errors.append(f"Validation error: {str(e)}")

        return {
            'valid': len(errors) == 0,
            'errors': errors,
            'timestamp': pd.Timestamp.now()
        }

    def generate_quality_report(self, validation_results: Dict[str, Any]) -> str:
        """
        Generate a human-readable quality report.

        Args:
            validation_results: Results from validate_ohlcv_data

        Returns:
            Formatted quality report
        """
        report = f"""
DATA QUALITY REPORT - {validation_results.get('symbol', 'Unknown')}
{'='*60}

SUMMARY:
  Total Rows:     {validation_results.get('total_rows', 0)}
  Valid Rows:     {validation_results.get('valid_rows', 0)}
  Quality Score:  {validation_results.get('quality_score', 0):.1f}%
  Overall Status: {'✅ VALID' if validation_results.get('is_valid', False) else '❌ ISSUES FOUND'}

ISSUES FOUND:
"""

        issues = validation_results.get('issues', [])
        if not issues:
            report += "  ✅ No issues detected\n"
        else:
            for i, issue in enumerate(issues, 1):
                report += f"  {i}. {issue}\n"

        report += f"\n{'='*60}"

        return report

