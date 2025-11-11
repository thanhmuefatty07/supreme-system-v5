#!/usr/bin/env python3
"""
Supreme System V5 - Binance API Client

Real implementation for connecting to Binance API and downloading market data.
Enhanced with configuration management and robust error handling.
"""

import os
import time
import logging
from typing import Optional, Dict, List, Any
from datetime import datetime, timedelta

import pandas as pd
import requests
from binance.client import Client
from binance.exceptions import BinanceAPIException, BinanceRequestException

# Robust import handling for config
try:
    from ..config.config import get_config
except ImportError:
    try:
        from config.config import get_config
    except ImportError:
        # Fallback for when config is not available
        def get_config():
            return None


class BinanceClient:
    """
    Real Binance API client for market data and trading operations.

    This is a production-ready implementation with proper error handling,
    rate limiting, and connection management.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        api_secret: Optional[str] = None,
        testnet: Optional[bool] = None,
        rate_limit_delay: Optional[float] = None,
        config_file: Optional[str] = None
    ):
        """
        Initialize Binance client with enhanced configuration.

        Args:
            api_key: Binance API key (optional, uses config if not provided)
            api_secret: Binance API secret (optional, uses config if not provided)
            testnet: Use testnet instead of live trading (optional, uses config)
            rate_limit_delay: Delay between API calls (optional, uses config)
            config_file: Path to configuration file (optional)
        """
        # Load configuration
        self.config = get_config()
        if config_file:
            try:
                from config.config import load_config
                self.config = load_config(config_file)
            except ImportError:
                self.config = None

        # Get credentials and settings from config (with fallbacks)
        if self.config:
            self.api_key = api_key or self.config.get('binance.api_key')
            self.api_secret = api_secret or self.config.get('binance.api_secret')
            self.testnet = testnet if testnet is not None else self.config.get('binance.testnet', True)
            self.rate_limit_delay = rate_limit_delay or self.config.get('binance.rate_limit_delay', 0.1)
            self.timeout = self.config.get('binance.timeout', 30)
            self.max_retries = self.config.get('data.max_retries', 3)
        else:
            # Fallback defaults when config is not available
            self.api_key = api_key
            self.api_secret = api_secret
            self.testnet = testnet if testnet is not None else True
            self.rate_limit_delay = rate_limit_delay or 0.1
            self.timeout = 30
            self.max_retries = 3
        self.retry_delay = (self.config.get('data.retry_delay', 1.0) if self.config else 1.0)

        self.client: Optional[Client] = None
        self.logger = logging.getLogger(__name__)
        self.last_request_time = 0
        self.request_count = 0
        self.error_count = 0

        # Initialize client if credentials are available
        if self.api_key and self.api_secret:
            self._initialize_client()
        else:
            self.logger.warning("No API credentials provided - operating in read-only mode")

    def _initialize_client(self) -> bool:
        """Initialize the Binance client with enhanced error handling."""
        try:
            self.client = Client(
                api_key=self.api_key,
                api_secret=self.api_secret,
                testnet=self.testnet,
                requests_params={'timeout': self.timeout}
            )
            self.logger.info(f"✅ Binance client initialized (testnet: {self.testnet})")
            return True
        except Exception as e:
            self.logger.error(f"❌ Failed to initialize Binance client: {e}")
            return False

    def test_connection(self) -> bool:
        """Test connection to Binance API."""
        if not self.client:
            self.logger.error("❌ Client not initialized")
            return False

        try:
            # Test with ping
            self.client.ping()
            self.logger.info("✅ Binance API connection successful")
            return True
        except Exception as e:
            self.logger.error(f"❌ Binance API connection failed: {e}")
            return False

    def _rate_limit_wait(self):
        """Apply rate limiting."""
        current_time = time.time()
        time_since_last = current_time - self.last_request_time

        if time_since_last < self.rate_limit_delay:
            time.sleep(self.rate_limit_delay - time_since_last)

        self.last_request_time = time.time()

    def get_historical_klines(
        self,
        symbol: str,
        interval: str,
        start_date: str,
        end_date: Optional[str] = None,
        limit: int = 500
    ) -> Optional[pd.DataFrame]:
        """
        Download historical klines (candlestick) data with retry logic.

        Args:
            symbol: Trading symbol (e.g., 'ETHUSDT')
            interval: Kline interval (e.g., '1h', '4h', '1d')
            start_date: Start date in 'YYYY-MM-DD' format
            end_date: End date in 'YYYY-MM-DD' format (optional)
            limit: Maximum number of records per request

        Returns:
            DataFrame with OHLCV data or None if failed
        """
        if not self.client:
            self.logger.error("❌ Client not initialized")
            return None

        # Validate inputs
        if not self._validate_symbol(symbol):
            return None

        if not self._validate_interval(interval):
            return None

        # Convert dates
        try:
            start_ts = int(datetime.strptime(start_date, '%Y-%m-%d').timestamp() * 1000)
            if end_date:
                end_ts = int(datetime.strptime(end_date, '%Y-%m-%d').timestamp() * 1000)
            else:
                # Default to 30 days from start
                end_ts = start_ts + (30 * 24 * 60 * 60 * 1000)
        except ValueError as e:
            self.logger.error(f"❌ Invalid date format: {e}")
            return None

        # Retry logic
        for attempt in range(self.max_retries):
            try:
                self._rate_limit_wait()
                self.request_count += 1

                self.logger.info(f"📊 Downloading {symbol} {interval} data from {start_date} (attempt {attempt + 1})")

                # Get klines
                klines = self.client.get_historical_klines(
                    symbol=symbol,
                    interval=interval,
                    start_str=str(start_ts),
                    end_str=str(end_ts),
                    limit=limit
                )

                if not klines:
                    self.logger.warning(f"⚠️ No data received for {symbol}")
                    return pd.DataFrame()

                # Convert to DataFrame
                df = self._process_klines_data(klines, symbol, interval)

                self.logger.info(f"✅ Downloaded {len(df)} records for {symbol} {interval}")
                return df

            except BinanceAPIException as e:
                self.error_count += 1
                error_code = e.code
                error_msg = e.message

                # Handle specific API errors
                if error_code == -1121:
                    self.logger.error(f"❌ Invalid symbol: {symbol}")
                    return None
                elif error_code == -1021:
                    self.logger.warning(f"⚠️ Timestamp error for {symbol}, retrying...")
                elif error_code == -1003:
                    self.logger.warning(f"⚠️ Rate limit exceeded, waiting longer...")
                    time.sleep(5)  # Additional delay for rate limits
                else:
                    self.logger.error(f"❌ Binance API error: {error_msg}")

                if attempt < self.max_retries - 1:
                    wait_time = self.retry_delay * (2 ** attempt)  # Exponential backoff
                    self.logger.info(f"⏳ Retrying in {wait_time:.1f} seconds...")
                    time.sleep(wait_time)
                else:
                    self.logger.error(f"❌ Failed after {self.max_retries} attempts")
                    return None

            except (BinanceRequestException, requests.exceptions.RequestException) as e:
                self.error_count += 1
                self.logger.warning(f"⚠️ Network error: {e}")
                if attempt < self.max_retries - 1:
                    time.sleep(self.retry_delay)
                else:
                    self.logger.error(f"❌ Network failed after {self.max_retries} attempts")
                    return None

            except Exception as e:
                self.error_count += 1
                self.logger.error(f"❌ Unexpected error: {e}")
                return None

        return None

    def get_symbol_info(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Get symbol information."""
        if not self.client:
            return None

        try:
            self._rate_limit_wait()
            info = self.client.get_symbol_info(symbol)
            return info
        except Exception as e:
            self.logger.error(f"❌ Failed to get symbol info: {e}")
            return None

    def get_exchange_info(self) -> Optional[Dict[str, Any]]:
        """Get exchange information."""
        if not self.client:
            return None

        try:
            self._rate_limit_wait()
            info = self.client.get_exchange_info()
            return info
        except Exception as e:
            self.logger.error(f"❌ Failed to get exchange info: {e}")
            return None

    def get_server_time(self) -> Optional[int]:
        """Get server time."""
        if not self.client:
            return None

        try:
            self._rate_limit_wait()
            time_data = self.client.get_server_time()
            return time_data.get('serverTime')
        except Exception as e:
            self.logger.error(f"❌ Failed to get server time: {e}")
            return None

    def _validate_symbol(self, symbol: str) -> bool:
        """Validate trading symbol format."""
        if not symbol or not isinstance(symbol, str):
            self.logger.error("❌ Symbol must be a non-empty string")
            return False

        # Basic format validation (should be like 'BTCUSDT', 'ETHUSDT')
        if not symbol.endswith(('USDT', 'BUSD', 'USDC', 'BTC')):
            self.logger.warning(f"⚠️ Unusual symbol format: {symbol}")

        return True

    def _validate_interval(self, interval: str) -> bool:
        """Validate Kline interval."""
        valid_intervals = [
            '1m', '3m', '5m', '15m', '30m',
            '1h', '2h', '4h', '6h', '8h', '12h',
            '1d', '3d', '1w', '1M'
        ]

        if interval not in valid_intervals:
            self.logger.error(f"❌ Invalid interval: {interval}. Valid: {valid_intervals}")
            return False

        return True

    def _process_klines_data(self, klines: List[List], symbol: str, interval: str) -> pd.DataFrame:
        """Process raw klines data into DataFrame."""
        try:
            # Convert to DataFrame
            df = pd.DataFrame(klines, columns=[
                'timestamp', 'open', 'high', 'low', 'close', 'volume',
                'close_time', 'quote_volume', 'trades', 'taker_buy_base',
                'taker_buy_quote', 'ignore'
            ])

            # Convert data types
            numeric_columns = ['open', 'high', 'low', 'close', 'volume',
                             'quote_volume', 'trades', 'taker_buy_base', 'taker_buy_quote']

            for col in numeric_columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')

            # Convert timestamp
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')

            # Keep only essential columns
            df = df[['timestamp', 'open', 'high', 'low', 'close', 'volume']]

            # Remove any NaN rows
            df = df.dropna()

            # Validate data quality
            if len(df) == 0:
                self.logger.warning(f"⚠️ No valid data after processing for {symbol}")
                return pd.DataFrame()

            # Check for data gaps (basic validation)
            try:
                time_diff = df['timestamp'].diff().dropna()
                expected_diff = self._parse_interval_to_timedelta(interval)
                if expected_diff:
                    gaps = time_diff > expected_diff * 1.5  # Allow some tolerance
                    if gaps.any():
                        gap_count = gaps.sum()
                        self.logger.warning(f"⚠️ Found {gap_count} data gaps in {symbol} {interval} data")
            except Exception:
                # Skip gap detection if interval parsing fails
                pass

            # Add metadata
            df.attrs['symbol'] = symbol
            df.attrs['interval'] = interval
            df.attrs['source'] = 'binance'

            return df

        except Exception as e:
            self.logger.error(f"❌ Failed to process klines data: {e}")
            return pd.DataFrame()

    def get_request_stats(self) -> Dict[str, Any]:
        """Get API request statistics."""
        return {
            'total_requests': self.request_count,
            'error_count': self.error_count,
            'error_rate': self.error_count / max(self.request_count, 1),
            'success_rate': (self.request_count - self.error_count) / max(self.request_count, 1)
        }

    def is_healthy(self) -> bool:
        """Check if client is healthy (low error rate)."""
        stats = self.get_request_stats()
        return stats['error_rate'] < 0.1  # Less than 10% error rate

    def _parse_interval_to_timedelta(self, interval: str) -> Optional[pd.Timedelta]:
        """Parse Binance interval string to Timedelta."""
        interval_map = {
            '1m': '1min', '3m': '3min', '5m': '5min', '15m': '15min', '30m': '30min',
            '1h': '1H', '2h': '2H', '4h': '4H', '6h': '6H', '8h': '8H', '12h': '12H',
            '1d': '1D', '3d': '3D', '1w': '1W', '1M': '1M'
        }

        try:
            if interval in interval_map:
                return pd.Timedelta(interval_map[interval])
            else:
                # Fallback for unrecognized intervals
                return pd.Timedelta('1H')  # Default to 1 hour
        except Exception:
            return None
