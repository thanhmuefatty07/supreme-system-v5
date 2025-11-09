#!/usr/bin/env python3
"""
Supreme System V5 - Binance API Client

Real implementation for connecting to Binance API and downloading market data.
"""

import os
import time
import logging
from typing import Optional, Dict, List, Any
from datetime import datetime, timedelta

import pandas as pd
import requests
from binance.client import Client


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
        testnet: bool = True,
        rate_limit_delay: float = 0.1
    ):
        """
        Initialize Binance client.

        Args:
            api_key: Binance API key (optional, uses env var if not provided)
            api_secret: Binance API secret (optional, uses env var if not provided)
            testnet: Use testnet instead of live trading
            rate_limit_delay: Delay between API calls to avoid rate limits
        """
        self.api_key = api_key or os.getenv("BINANCE_API_KEY")
        self.api_secret = api_secret or os.getenv("BINANCE_API_SECRET")
        self.testnet = testnet
        self.rate_limit_delay = rate_limit_delay

        self.client: Optional[Client] = None
        self.logger = logging.getLogger(__name__)
        self.last_request_time = 0

        # Initialize client if credentials are available
        if self.api_key and self.api_secret:
            self._initialize_client()

    def _initialize_client(self) -> bool:
        """Initialize the Binance client."""
        try:
            self.client = Client(
                api_key=self.api_key,
                api_secret=self.api_secret,
                testnet=self.testnet
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
        Download historical klines (candlestick) data.

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

        try:
            self._rate_limit_wait()

            # Convert dates to timestamps
            start_ts = int(datetime.strptime(start_date, '%Y-%m-%d').timestamp() * 1000)

            if end_date:
                end_ts = int(datetime.strptime(end_date, '%Y-%m-%d').timestamp() * 1000)
            else:
                # Default to 30 days from start
                end_ts = start_ts + (30 * 24 * 60 * 60 * 1000)

            self.logger.info(f"📊 Downloading {symbol} {interval} data from {start_date}")

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

            self.logger.info(f"✅ Downloaded {len(df)} records for {symbol} {interval}")

            return df

        except Exception as e:
            self.logger.error(f"❌ Failed to download data: {e}")
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
