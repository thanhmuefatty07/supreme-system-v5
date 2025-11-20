#!/usr/bin/env python3
"""
Supreme System V5 - Bybit API Client

Real implementation for connecting to Bybit API and downloading market data.
Enhanced with configuration management and robust error handling.
Uses pybit library with async wrapper for compatibility.
"""

import asyncio
import hashlib
import hmac
import logging
import os
import time
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Union
from urllib.parse import urlencode

import aiohttp
import pandas as pd

try:
    from pybit.unified_trading import HTTP
    from pybit import spot
    PYBIT_AVAILABLE = True
except ImportError:
    PYBIT_AVAILABLE = False
    HTTP = None
    spot = None

try:
    import orjson as json
except ImportError:
    import json

# Import Pydantic models for validation
try:
    from .data_validator import (
        TradingSymbol, KlineInterval, APIRequestConfig,
        DataQueryParams, ValidationResult
    )
except ImportError:
    ValidationResult = None

# Import secrets manager
try:
    from ..utils.secrets_manager import SecretsManager, get_secrets_manager
    SECRETS_AVAILABLE = True
except ImportError:
    SECRETS_AVAILABLE = False
    get_secrets_manager = None

# Import config
try:
    from ..config.config import get_config
except ImportError:
    try:
        from config.config import get_config
    except ImportError:
        def get_config() -> Optional[Dict[str, Any]]:
            return None


def get_bybit_credentials() -> Dict[str, Any]:
    """Get Bybit credentials from secrets manager."""
    if not SECRETS_AVAILABLE or not get_secrets_manager:
        return {}
    
    secrets_manager = get_secrets_manager()
    if not secrets_manager:
        return {}
    
    return {
        'api_key': secrets_manager.get_secret('BYBIT_API_KEY'),
        'api_secret': secrets_manager.get_secret('BYBIT_SECRET_KEY'),
        'testnet': secrets_manager.get_secret('BYBIT_TESTNET', 'true').lower() == 'true'
    }


class AsyncBybitClient:
    """
    Async Bybit API client for high-performance market data operations.
    
    This implementation wraps pybit library with async interface and provides
    compatibility with existing Binance client structure.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        api_secret: Optional[str] = None,
        testnet: Optional[bool] = None,
        rate_limit_delay: Optional[float] = None,
        config_file: Optional[str] = None,
        max_concurrent_requests: int = 10,
        use_secrets_manager: bool = True
    ):
        """
        Initialize async Bybit client with secure secrets management.

        Args:
            api_key: Bybit API key (optional, uses secrets manager if not provided)
            api_secret: Bybit API secret (optional, uses secrets manager if not provided)
            testnet: Use testnet instead of live trading (optional, uses secrets manager)
            rate_limit_delay: Delay between API calls (optional)
            config_file: Path to configuration file (fallback)
            max_concurrent_requests: Maximum concurrent requests for connection pooling
            use_secrets_manager: Whether to use secure secrets management
        """
        if not PYBIT_AVAILABLE:
            raise ImportError(
                "pybit library is required for Bybit integration. "
                "Install with: pip install pybit"
            )

        # Initialize secrets manager
        self.secrets_manager = None
        if use_secrets_manager and SECRETS_AVAILABLE:
            self.secrets_manager = get_secrets_manager()

        # Load credentials - prefer secrets manager, fallback to config
        if self.secrets_manager:
            creds = get_bybit_credentials()
            self.api_key = api_key or creds.get('api_key')
            self.api_secret = api_secret or creds.get('api_secret')
            self.testnet = testnet if testnet is not None else creds.get('testnet', True)
        else:
            # Fallback to environment variables or config
            self.config = get_config()
            if config_file:
                try:
                    from config.config import load_config
                    self.config = load_config(config_file)
                except ImportError:
                    self.config = None

            if self.config:
                self.api_key = api_key or self.config.get('bybit.api_key') or os.getenv('BYBIT_API_KEY')
                self.api_secret = api_secret or self.config.get('bybit.api_secret') or os.getenv('BYBIT_SECRET_KEY')
                testnet_config = self.config.get('bybit.testnet')
                if testnet_config is not None:
                    self.testnet = testnet if testnet is not None else testnet_config
                else:
                    self.testnet = testnet if testnet is not None else os.getenv('BYBIT_TESTNET', 'true').lower() == 'true'
            else:
                self.api_key = api_key or os.getenv('BYBIT_API_KEY')
                self.api_secret = api_secret or os.getenv('BYBIT_SECRET_KEY')
                self.testnet = testnet if testnet is not None else os.getenv('BYBIT_TESTNET', 'true').lower() == 'true'

        # Other settings
        self.rate_limit_delay = rate_limit_delay or 0.05
        self.timeout = 30
        self.max_retries = 3
        self.retry_delay = 1.0
        self.max_concurrent_requests = max_concurrent_requests

        # Initialize pybit HTTP client
        self._http_client = None
        self._spot_client = None

        # Session management for async operations
        self._session: Optional[aiohttp.ClientSession] = None
        self._semaphore = asyncio.Semaphore(max_concurrent_requests)

        self.logger = logging.getLogger(__name__)
        self.last_request_time = 0
        self.request_count = 0
        self.error_count = 0

        # Initialize pybit clients
        self._initialize_pybit_clients()

        self.logger.info(f"✅ Async Bybit client initialized (testnet: {self.testnet}, secure: {bool(self.secrets_manager)})")

    def _initialize_pybit_clients(self):
        """Initialize pybit HTTP and Spot clients."""
        try:
            # Unified Trading API (supports spot, linear, inverse)
            self._http_client = HTTP(
                testnet=self.testnet,
                api_key=self.api_key,
                api_secret=self.api_secret,
            )
            
            # Spot Trading API (legacy, for compatibility)
            if self.api_key and self.api_secret:
                self._spot_client = spot.HTTP(
                    endpoint="https://api-testnet.bybit.com" if self.testnet else "https://api.bybit.com",
                    api_key=self.api_key,
                    api_secret=self.api_secret,
                )
            
            self.logger.info("✅ Pybit clients initialized")
        except Exception as e:
            self.logger.error(f"❌ Failed to initialize pybit clients: {e}")
            raise

    async def __aenter__(self):
        """Async context manager entry."""
        await self.initialize_session()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.close_session()

    async def initialize_session(self):
        """Initialize aiohttp session for async operations."""
        if self._session is None:
            connector = aiohttp.TCPConnector(
                limit=self.max_concurrent_requests,
                limit_per_host=self.max_concurrent_requests,
                ttl_dns_cache=300,
                use_dns_cache=True,
                keepalive_timeout=60,
                enable_cleanup_closed=True
            )

            timeout = aiohttp.ClientTimeout(total=self.timeout)

            self._session = aiohttp.ClientSession(
                connector=connector,
                timeout=timeout,
                headers={
                    'User-Agent': 'SupremeSystemV5/1.0',
                    'Accept': 'application/json',
                    'Content-Type': 'application/json'
                }
            )

    async def close_session(self):
        """Close aiohttp session."""
        if self._session:
            await self._session.close()
            self._session = None

    async def _apply_rate_limit(self):
        """Apply async rate limiting."""
        async with self._semaphore:
            current_time = time.time()
            time_since_last = current_time - self.last_request_time

            if time_since_last < self.rate_limit_delay:
                await asyncio.sleep(self.rate_limit_delay - time_since_last)

            self.last_request_time = time.time()

    async def test_connection(self) -> bool:
        """Test connection to Bybit API asynchronously."""
        try:
            # Test with server time endpoint
            result = await asyncio.to_thread(
                self._http_client.get_server_time
            )
            
            if result and result.get('retCode') == 0:
                self.logger.info("✅ Bybit API connection successful")
                return True
            else:
                self.logger.error(f"❌ Bybit API connection failed: {result}")
                return False
        except Exception as e:
            self.error_count += 1
            self.logger.error(f"❌ Bybit API connection failed: {e}")
            return False

    def _convert_interval(self, interval: str) -> str:
        """Convert standard interval to Bybit format."""
        interval_map = {
            '1m': '1',
            '3m': '3',
            '5m': '5',
            '15m': '15',
            '30m': '30',
            '1h': '60',
            '2h': '120',
            '4h': '240',
            '6h': '360',
            '12h': '720',
            '1d': 'D',
            '1w': 'W',
            '1M': 'M'
        }
        return interval_map.get(interval, interval)

    def _process_klines_data(self, klines: List[List], symbol: str, interval: str) -> pd.DataFrame:
        """Process Bybit klines list into DataFrame."""
        if not klines:
            return pd.DataFrame()

        # Bybit returns klines in reverse chronological order
        # Format: [start_time, open, high, low, close, volume, turnover]
        df = pd.DataFrame(klines, columns=[
            'timestamp', 'open', 'high', 'low', 'close', 'volume', 'turnover'
        ])

        # Convert timestamp (milliseconds) to datetime
        df['timestamp'] = pd.to_datetime(df['timestamp'].astype(int), unit='ms')
        df.set_index('timestamp', inplace=True)

        # Convert to float
        for col in ['open', 'high', 'low', 'close', 'volume', 'turnover']:
            df[col] = df[col].astype(float)

        # Sort by timestamp (oldest first)
        df.sort_index(inplace=True)

        # Add metadata
        df['symbol'] = symbol
        df['interval'] = interval

        return df

    async def get_historical_klines(
        self,
        symbol: str,
        interval: str,
        start_date: str,
        end_date: Optional[str] = None,
        limit: int = 200
    ) -> Optional[pd.DataFrame]:
        """
        Download historical klines (candlestick) data asynchronously.

        Args:
            symbol: Trading symbol (e.g., 'BTCUSDT')
            interval: Kline interval (e.g., '1h', '4h', '1d')
            start_date: Start date in 'YYYY-MM-DD' format
            end_date: End date in 'YYYY-MM-DD' format (optional)
            limit: Maximum number of records per request (Bybit max: 200)

        Returns:
            DataFrame with OHLCV data or None if failed
        """
        if not self._http_client:
            self._initialize_pybit_clients()

        # Validate inputs
        if not self._validate_symbol(symbol):
            return None

        if not self._validate_interval(interval):
            return None

        # Convert dates to timestamps
        try:
            start_dt = datetime.strptime(start_date, '%Y-%m-%d')
            start_ts = int(start_dt.timestamp() * 1000)
            
            if end_date:
                end_dt = datetime.strptime(end_date, '%Y-%m-%d')
                end_ts = int(end_dt.timestamp() * 1000)
            else:
                # Default to 30 days from start
                end_ts = start_ts + (30 * 24 * 60 * 60 * 1000)
        except ValueError as e:
            self.logger.error(f"❌ Invalid date format: {e}")
            return None

        # Convert interval to Bybit format
        bybit_interval = self._convert_interval(interval)
        
        # Limit to Bybit's maximum
        limit = min(limit, 200)

        # Retry logic
        for attempt in range(self.max_retries):
            try:
                await self._apply_rate_limit()
                self.request_count += 1

                self.logger.info(f"📊 Downloading {symbol} {interval} data from {start_date} (attempt {attempt + 1})")

                # Use pybit to get klines
                # Bybit uses 'linear' category for USDT perpetuals, 'spot' for spot trading
                category = 'linear'  # Default to linear, can be 'spot' or 'inverse'
                
                result = await asyncio.to_thread(
                    self._http_client.get_kline,
                    category=category,
                    symbol=symbol.upper(),
                    interval=bybit_interval,
                    start=start_ts,
                    end=end_ts,
                    limit=limit
                )

                if result and result.get('retCode') == 0:
                    klines = result.get('result', {}).get('list', [])
                    
                    if not klines:
                        self.logger.warning(f"⚠️ No data received for {symbol}")
                        return pd.DataFrame()

                    # Process and convert to DataFrame
                    df = self._process_klines_data(klines, symbol, interval)
                    
                    self.logger.info(f"✅ Downloaded {len(df)} records for {symbol} {interval}")
                    return df
                else:
                    error_msg = result.get('retMsg', 'Unknown error') if result else 'No response'
                    self.logger.error(f"❌ Bybit API error: {error_msg}")
                    
                    if attempt < self.max_retries - 1:
                        await asyncio.sleep(self.retry_delay * (2 ** attempt))
                    else:
                        return None

            except Exception as e:
                self.error_count += 1
                self.logger.error(f"❌ Error downloading klines: {e}")
                
                if attempt < self.max_retries - 1:
                    await asyncio.sleep(self.retry_delay * (2 ** attempt))
                else:
                    return None

        return None


    def _validate_symbol(self, symbol: str) -> bool:
        """Validate trading symbol format."""
        if not symbol or len(symbol) < 3:
            return False
        return True

    def _validate_interval(self, interval: str) -> bool:
        """Validate interval format."""
        valid_intervals = ['1m', '3m', '5m', '15m', '30m', '1h', '2h', '4h', '6h', '12h', '1d', '1w', '1M']
        return interval in valid_intervals

    async def get_symbol_info(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Get symbol information asynchronously."""
        try:
            await self._apply_rate_limit()
            self.request_count += 1

            # Get instrument info
            result = await asyncio.to_thread(
                self._http_client.get_instruments_info,
                category='linear',
                symbol=symbol.upper()
            )

            if result and result.get('retCode') == 0:
                instruments = result.get('result', {}).get('list', [])
                if instruments:
                    return instruments[0]
            return None
        except Exception as e:
            self.error_count += 1
            self.logger.error(f"❌ Failed to get symbol info: {e}")
            return None

    async def get_exchange_info(self) -> Optional[Dict[str, Any]]:
        """Get exchange information asynchronously."""
        try:
            await self._apply_rate_limit()
            self.request_count += 1

            # Get all instruments
            result = await asyncio.to_thread(
                self._http_client.get_instruments_info,
                category='linear'
            )

            if result and result.get('retCode') == 0:
                return result.get('result', {})
            return None
        except Exception as e:
            self.error_count += 1
            self.logger.error(f"❌ Failed to get exchange info: {e}")
            return None

    async def get_server_time(self) -> Optional[int]:
        """Get server time asynchronously."""
        try:
            await self._apply_rate_limit()
            result = await asyncio.to_thread(
                self._http_client.get_server_time
            )

            if result and result.get('retCode') == 0:
                server_time = result.get('result', {}).get('timeSecond', 0)
                return int(server_time) * 1000  # Convert to milliseconds
            return None
        except Exception as e:
            self.error_count += 1
            self.logger.error(f"❌ Failed to get server time: {e}")
            return None

    def get_security_stats(self) -> Dict[str, Any]:
        """Get comprehensive security statistics."""
        return {
            'request_count': self.request_count,
            'error_count': self.error_count,
            'error_rate': self.error_count / max(self.request_count, 1),
            'secrets_manager_available': bool(self.secrets_manager),
            'last_request_time': self.last_request_time,
            'testnet': self.testnet
        }


# Backward compatibility - synchronous wrapper
class BybitClient:
    """
    Synchronous wrapper for AsyncBybitClient.
    
    This maintains backward compatibility while providing async performance benefits.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        api_secret: Optional[str] = None,
        testnet: Optional[bool] = None,
        rate_limit_delay: Optional[float] = None,
        config_file: Optional[str] = None,
        use_secrets_manager: bool = True
    ):
        self.async_client = AsyncBybitClient(
            api_key=api_key,
            api_secret=api_secret,
            testnet=testnet,
            rate_limit_delay=rate_limit_delay,
            config_file=config_file,
            use_secrets_manager=use_secrets_manager
        )
        self.logger = logging.getLogger(__name__)

    def __enter__(self):
        """Context manager entry - initialize async client."""
        asyncio.run(self.async_client.initialize_session())
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - close async client."""
        asyncio.run(self.async_client.close_session())

    def test_connection(self) -> bool:
        """Test connection synchronously."""
        try:
            return asyncio.run(self.async_client.test_connection())
        except Exception as e:
            self.logger.error(f"Connection test failed: {e}")
            return False

    def get_historical_klines(
        self,
        symbol: str,
        interval: str,
        start_date: str,
        end_date: Optional[str] = None,
        limit: int = 200
    ) -> Optional[pd.DataFrame]:
        """Get historical klines synchronously."""
        try:
            return asyncio.run(
                self.async_client.get_historical_klines(
                    symbol, interval, start_date, end_date, limit
                )
            )
        except Exception as e:
            self.logger.error(f"Failed to get historical klines: {e}")
            return None

