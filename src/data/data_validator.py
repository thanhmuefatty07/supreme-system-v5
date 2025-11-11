#!/usr/bin/env python3
"""
Supreme System V5 - Data Validation Module

Comprehensive data validation for financial market data.
Ensures data quality, integrity, and compliance.
"""

import logging
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple, Union
from decimal import Decimal

import numpy as np
import pandas as pd
from pydantic import BaseModel, Field, validator, model_validator
from pydantic import ValidationError


# Pydantic Models for Input Validation

class OHLCVDataPoint(BaseModel):
    """Single OHLCV data point with comprehensive validation."""
    timestamp: datetime = Field(..., description="Data timestamp")
    open: Decimal = Field(..., gt=0, description="Opening price")
    high: Decimal = Field(..., gt=0, description="Highest price")
    low: Decimal = Field(..., gt=0, description="Lowest price")
    close: Decimal = Field(..., gt=0, description="Closing price")
    volume: Decimal = Field(..., ge=0, description="Trading volume")

    @model_validator(mode='after')
    def validate_ohlc_relationships(self):
        """Validate OHLC relationships."""
        open_price = self.open
        high_price = self.high
        low_price = self.low
        close_price = self.close

        # High >= Low
        if high_price < low_price:
            raise ValueError("High price must be >= low price")

        # Open and Close within high-low range
        if not (low_price <= open_price <= high_price):
            raise ValueError("Open price must be within high-low range")

        if not (low_price <= close_price <= high_price):
            raise ValueError("Close price must be within high-low range")

        return self

    @validator('timestamp')
    def validate_timestamp(cls, v):
        """Validate timestamp is not in future and not too old."""
        now = datetime.now()
        if v > now + timedelta(minutes=1):  # Allow 1 minute future tolerance
            raise ValueError("Timestamp cannot be in the future")

        # Not older than 10 years
        ten_years_ago = now - timedelta(days=3650)
        if v < ten_years_ago:
            raise ValueError("Timestamp cannot be older than 10 years")

        return v

    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat(),
            Decimal: lambda v: float(v)
        }


class TradingSymbol(BaseModel):
    """Trading symbol validation."""
    symbol: str = Field(..., min_length=1, max_length=20, pattern=r'^[A-Z0-9]+$')

    @validator('symbol')
    def validate_symbol_format(cls, v):
        """Validate symbol format for common exchanges."""
        # Remove common separators
        clean_symbol = v.replace('/', '').replace('-', '').replace('_', '')

        # Should contain at least one letter and end with quote currency
        if not any(c.isalpha() for c in clean_symbol):
            raise ValueError("Symbol must contain at least one letter")

        # Common quote currencies
        quote_currencies = ['USDT', 'BUSD', 'USDC', 'BTC', 'ETH', 'BNB', 'USD']
        if not any(clean_symbol.upper().endswith(qc) for qc in quote_currencies):
            # Allow if it's a known symbol format
            pass

        return v.upper()


class KlineInterval(BaseModel):
    """Kline/candlestick interval validation."""
    interval: str = Field(..., pattern=r'^\d+[mhdwM]$')

    @validator('interval')
    def validate_interval(cls, v):
        """Validate interval format."""
        valid_intervals = [
            '1m', '3m', '5m', '15m', '30m',
            '1h', '2h', '4h', '6h', '8h', '12h',
            '1d', '3d', '1w', '1M'
        ]

        if v not in valid_intervals:
            raise ValueError(f"Invalid interval. Valid intervals: {valid_intervals}")

        return v


class TradingStrategyConfig(BaseModel):
    """Trading strategy configuration validation."""
    name: str = Field(..., min_length=1, max_length=50)
    symbol: str
    initial_capital: Decimal = Field(..., gt=0, le=Decimal('10000000'))  # Max 10M
    risk_per_trade: Decimal = Field(..., gt=0, le=Decimal('1'))  # Max 100%
    max_positions: int = Field(..., ge=1, le=100)
    stop_loss_pct: Decimal = Field(..., gt=0, le=Decimal('0.5'))  # Max 50%
    take_profit_pct: Decimal = Field(..., gt=0, le=Decimal('1'))  # Max 100%

    @model_validator(mode='after')
    def validate_risk_parameters(self):
        """Validate risk parameters make sense together."""
        capital = self.initial_capital
        risk_pct = self.risk_per_trade
        max_pos = self.max_positions

        # Risk per position should not exceed reasonable limits
        risk_per_position = capital * risk_pct / max_pos
        if risk_per_position < Decimal('1'):  # Less than $1 risk
            raise ValueError("Risk per position too low (< $1)")

        if risk_per_position > capital * Decimal('0.1'):  # More than 10% of capital per position
            raise ValueError("Risk per position too high (> 10% of capital)")

        return self


class APIRequestConfig(BaseModel):
    """API request configuration validation."""
    api_key: str = Field(..., min_length=10, max_length=200)
    api_secret: str = Field(..., min_length=10, max_length=200)
    testnet: bool = True
    rate_limit_delay: Decimal = Field(..., ge=0, le=Decimal('10'))  # Max 10 seconds
    timeout: int = Field(..., ge=1, le=300)  # 1-300 seconds
    max_retries: int = Field(..., ge=0, le=10)

    @validator('api_key', 'api_secret')
    def validate_api_credentials(cls, v):
        """Validate API credentials format."""
        # Should be alphanumeric with possible special chars
        import re
        if not re.match(r'^[A-Za-z0-9+/=]+$', v):
            raise ValueError("API credentials should be base64 encoded")

        return v


class DataQueryParams(BaseModel):
    """Data query parameters validation."""
    symbol: str
    interval: str
    start_date: str
    end_date: Optional[str] = None
    limit: int = Field(..., ge=1, le=1000)

    @model_validator(mode='after')
    def validate_date_range(self):
        """Validate date range."""
        start_str = self.start_date
        end_str = self.end_date

        if end_str:
            try:
                start_dt = datetime.strptime(start_str, '%Y-%m-%d')
                end_dt = datetime.strptime(end_str, '%Y-%m-%d')

                if start_dt >= end_dt:
                    raise ValueError("Start date must be before end date")

                # Not more than 1 year range
                if (end_dt - start_dt).days > 365:
                    raise ValueError("Date range cannot exceed 1 year")

            except ValueError as e:
                if "time data" in str(e):
                    raise ValueError("Invalid date format. Use YYYY-MM-DD")
                raise

        return self

    @validator('start_date', 'end_date')
    def validate_date_format(cls, v):
        """Validate date string format."""
        if v is None:
            return v

        try:
            datetime.strptime(v, '%Y-%m-%d')
            return v
        except ValueError:
            raise ValueError("Date must be in YYYY-MM-DD format")


class ValidationResult(BaseModel):
    """Validation result structure."""
    valid: bool
    errors: List[str] = Field(default_factory=list)
    warnings: List[str] = Field(default_factory=list)
    timestamp: datetime = Field(default_factory=datetime.now)
    metadata: Dict[str, Any] = Field(default_factory=dict)


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

    def validate_with_pydantic(self, data: Union[Dict, List[Dict], pd.DataFrame],
                              model_type: str) -> ValidationResult:
        """
        Validate data using Pydantic models.

        Args:
            data: Data to validate (dict, list of dicts, or DataFrame)
            model_type: Type of model to use ('ohlcv', 'symbol', 'config', etc.)

        Returns:
            ValidationResult with detailed feedback
        """
        result = ValidationResult(valid=True, errors=[], warnings=[])

        try:
            if model_type == 'ohlcv_datapoint':
                if isinstance(data, dict):
                    OHLCVDataPoint(**data)
                elif isinstance(data, list):
                    for item in data:
                        OHLCVDataPoint(**item)
                elif isinstance(data, pd.DataFrame):
                    # Convert DataFrame rows to dicts and validate
                    for _, row in data.iterrows():
                        row_dict = row.to_dict()
                        # Convert numpy types to native Python types
                        for key, value in row_dict.items():
                            if hasattr(value, 'item'):  # numpy scalar
                                row_dict[key] = value.item()
                            elif isinstance(value, np.datetime64):
                                row_dict[key] = value.astype('datetime64[s]').astype(datetime)
                        OHLCVDataPoint(**row_dict)

            elif model_type == 'trading_symbol':
                TradingSymbol(symbol=data if isinstance(data, str) else data.get('symbol'))

            elif model_type == 'kline_interval':
                KlineInterval(interval=data if isinstance(data, str) else data.get('interval'))

            elif model_type == 'strategy_config':
                TradingStrategyConfig(**data)

            elif model_type == 'api_config':
                APIRequestConfig(**data)

            elif model_type == 'query_params':
                DataQueryParams(**data)

            else:
                result.errors.append(f"Unknown model type: {model_type}")
                result.valid = False

        except ValidationError as e:
            result.valid = False
            result.errors = [f"{err['loc'][0]}: {err['msg']}" for err in e.errors()]
        except Exception as e:
            result.valid = False
            result.errors = [f"Validation error: {str(e)}"]

        return result

    def validate_dataframe_with_models(self, df: pd.DataFrame) -> ValidationResult:
        """
        Validate entire DataFrame using Pydantic models.

        Args:
            df: DataFrame to validate

        Returns:
            ValidationResult with comprehensive feedback
        """
        result = ValidationResult(valid=True, errors=[], warnings=[])

        try:
            # First check basic structure
            required_cols = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
            missing_cols = [col for col in required_cols if col not in df.columns]
            if missing_cols:
                result.errors.append(f"Missing required columns: {missing_cols}")
                result.valid = False
                return result

            # Validate each row using Pydantic
            invalid_rows = 0
            for idx, row in df.iterrows():
                try:
                    row_dict = {}
                    for col in required_cols:
                        value = row[col]
                        if hasattr(value, 'item'):  # numpy scalar
                            value = value.item()
                        elif isinstance(value, np.datetime64):
                            value = value.astype('datetime64[s]').astype(datetime)
                        row_dict[col] = value

                    OHLCVDataPoint(**row_dict)

                except ValidationError as e:
                    invalid_rows += 1
                    if invalid_rows <= 5:  # Only log first 5 errors
                        result.errors.append(f"Row {idx}: {e.errors()[0]['msg']}")
                except Exception as e:
                    invalid_rows += 1
                    if invalid_rows <= 5:
                        result.errors.append(f"Row {idx}: {str(e)}")

            if invalid_rows > 0:
                result.errors.append(f"Total invalid rows: {invalid_rows}/{len(df)}")
                result.valid = False

            # Add metadata
            result.metadata = {
                'total_rows': len(df),
                'valid_rows': len(df) - invalid_rows,
                'invalid_rows': invalid_rows,
                'validity_percentage': ((len(df) - invalid_rows) / len(df) * 100) if len(df) > 0 else 0
            }

        except Exception as e:
            result.valid = False
            result.errors = [f"DataFrame validation failed: {str(e)}"]

        return result

    def validate_api_inputs(self, **kwargs) -> ValidationResult:
        """
        Validate API input parameters.

        Args:
            **kwargs: API parameters to validate

        Returns:
            ValidationResult
        """
        result = ValidationResult(valid=True, errors=[], warnings=[])

        # Validate symbol if provided
        if 'symbol' in kwargs:
            symbol_result = self.validate_with_pydantic({'symbol': kwargs['symbol']}, 'trading_symbol')
            if not symbol_result.valid:
                result.errors.extend(symbol_result.errors)

        # Validate interval if provided
        if 'interval' in kwargs:
            interval_result = self.validate_with_pydantic({'interval': kwargs['interval']}, 'kline_interval')
            if not interval_result.valid:
                result.errors.extend(interval_result.errors)

        # Validate dates if provided
        date_params = {}
        if 'start_date' in kwargs:
            date_params['start_date'] = kwargs['start_date']
        if 'end_date' in kwargs:
            date_params['end_date'] = kwargs['end_date']

        if date_params:
            date_result = self.validate_with_pydantic(date_params, 'query_params')
            if not date_result.valid:
                result.errors.extend(date_result.errors)

        # Validate numeric parameters
        if 'limit' in kwargs:
            limit = kwargs['limit']
            if not isinstance(limit, int) or limit < 1 or limit > 1000:
                result.errors.append("Limit must be integer between 1 and 1000")

        result.valid = len(result.errors) == 0
        return result

    def validate_strategy_config(self, config: Dict[str, Any]) -> ValidationResult:
        """
        Validate trading strategy configuration.

        Args:
            config: Strategy configuration dictionary

        Returns:
            ValidationResult
        """
        return self.validate_with_pydantic(config, 'strategy_config')

    def sanitize_input_data(self, data: Union[Dict, List[Dict], pd.DataFrame],
                           data_type: str = 'ohlcv') -> Union[Dict, List[Dict], pd.DataFrame]:
        """
        Sanitize and normalize input data.

        Args:
            data: Input data to sanitize
            data_type: Type of data ('ohlcv', 'config', etc.)

        Returns:
            Sanitized data
        """
        try:
            if data_type == 'ohlcv':
                if isinstance(data, pd.DataFrame):
                    # Convert DataFrame to list of dicts, sanitize, then back to DataFrame
                    records = []
                    for _, row in data.iterrows():
                        record = {}
                        for col in data.columns:
                            value = row[col]
                            if hasattr(value, 'item'):  # numpy scalar
                                value = value.item()
                            elif isinstance(value, np.datetime64):
                                value = value.astype('datetime64[s]').astype(datetime)
                            record[col] = value
                        records.append(record)

                    # Sanitize each record
                    sanitized_records = []
                    for record in records:
                        try:
                            # Try to create Pydantic model (this will sanitize)
                            model = OHLCVDataPoint(**record)
                            sanitized_records.append(model.dict())
                        except ValidationError:
                            # If validation fails, try to fix common issues
                            sanitized_record = self._fix_common_data_issues(record)
                            try:
                                model = OHLCVDataPoint(**sanitized_record)
                                sanitized_records.append(model.dict())
                            except ValidationError:
                                # Skip invalid records
                                continue

                    # Convert back to DataFrame
                    if sanitized_records:
                        return pd.DataFrame(sanitized_records)
                    else:
                        return pd.DataFrame()

                elif isinstance(data, list):
                    sanitized_list = []
                    for item in data:
                        try:
                            model = OHLCVDataPoint(**item)
                            sanitized_list.append(model.dict())
                        except ValidationError:
                            sanitized_item = self._fix_common_data_issues(item)
                            try:
                                model = OHLCVDataPoint(**sanitized_item)
                                sanitized_list.append(model.dict())
                            except ValidationError:
                                continue
                    return sanitized_list

                elif isinstance(data, dict):
                    try:
                        model = OHLCVDataPoint(**data)
                        return model.dict()
                    except ValidationError:
                        sanitized = self._fix_common_data_issues(data)
                        model = OHLCVDataPoint(**sanitized)
                        return model.dict()

        except Exception as e:
            self.logger.error(f"Data sanitization failed: {e}")
            return data  # Return original data if sanitization fails

        return data

    def _fix_common_data_issues(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Fix common data issues before validation.

        Args:
            data: Data dictionary with potential issues

        Returns:
            Fixed data dictionary
        """
        fixed = data.copy()

        # Fix timestamp
        if 'timestamp' in fixed:
            ts = fixed['timestamp']
            if isinstance(ts, str):
                try:
                    fixed['timestamp'] = datetime.fromisoformat(ts.replace('Z', '+00:00'))
                except:
                    # Try other formats
                    for fmt in ['%Y-%m-%d %H:%M:%S', '%Y-%m-%dT%H:%M:%S', '%Y-%m-%d']:
                        try:
                            fixed['timestamp'] = datetime.strptime(ts, fmt)
                            break
                        except:
                            continue

        # Fix numeric types and ensure positive values
        price_fields = ['open', 'high', 'low', 'close']
        for field in price_fields:
            if field in fixed:
                value = fixed[field]
                if isinstance(value, (int, float, str)):
                    try:
                        numeric_value = float(value)
                        fixed[field] = Decimal(str(max(0.00000001, numeric_value)))  # Ensure positive
                    except:
                        fixed[field] = Decimal('1.0')  # Default value

        # Fix volume
        if 'volume' in fixed:
            value = fixed['volume']
            if isinstance(value, (int, float, str)):
                try:
                    numeric_value = float(value)
                    fixed['volume'] = Decimal(str(max(0, numeric_value)))
                except:
                    fixed['volume'] = Decimal('0')

        return fixed

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

