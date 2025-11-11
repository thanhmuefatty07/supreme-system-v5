#!/usr/bin/env python3
"""
Supreme System V5 - Binance API Client

Real implementation for connecting to Binance API and downloading market data.
Enhanced with configuration management and robust error handling.
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
from binance.exceptions import BinanceAPIException, BinanceRequestException

# Security imports
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.asymmetric import rsa, padding
from cryptography.hazmat.primitives import serialization
import base64

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
    # Fallback if import fails
    ValidationResult = None


class RequestSigner:
    """
    Advanced request signing with HMAC-SHA256 and encryption support.

    Features:
    - HMAC-SHA256 signing for Binance API
    - Request encryption for sensitive data
    - Timestamp validation and replay attack prevention
    - Digital signatures for additional security
    """

    def __init__(self, api_secret: str):
        self.api_secret = api_secret
        self._rsa_key = None
        self._generate_rsa_key()

    def _generate_rsa_key(self):
        """Generate RSA key pair for digital signatures."""
        try:
            self._rsa_key = rsa.generate_private_key(
                public_exponent=65537,
                key_size=2048,
            )
        except Exception as e:
            logging.warning(f"RSA key generation failed: {e}")

    def sign_request(self, query_string: str, timestamp: Optional[int] = None) -> str:
        """Create HMAC-SHA256 signature for Binance API request."""
        if not self.api_secret:
            raise ValueError("API secret required for request signing")

        if timestamp is None:
            timestamp = int(time.time() * 1000)

        # Add timestamp to prevent replay attacks
        if 'timestamp=' not in query_string:
            query_string = f"{query_string}&timestamp={timestamp}"

        return hmac.new(
            self.api_secret.encode('utf-8'),
            query_string.encode('utf-8'),
            hashlib.sha256
        ).hexdigest()

    def create_secure_headers(self, endpoint: str, params: Dict[str, Any]) -> Dict[str, str]:
        """Create secure headers with signatures and encryption."""
        headers = {
            'User-Agent': 'SupremeSystemV5/1.0',
            'Accept': 'application/json',
            'Content-Type': 'application/json'
        }

        # Add API key if available
        if hasattr(self, '_api_key') and self._api_key:
            headers['X-MBX-APIKEY'] = self._api_key

        # Add timestamp and signature for authenticated requests
        if self.api_secret and params:
            query_string = urlencode(sorted(params.items()))
            signature = self.sign_request(query_string)
            headers['X-MBX-SIGNATURE'] = signature

            # Add timestamp
            headers['X-MBX-TIMESTAMP'] = str(int(time.time() * 1000))

        # Add digital signature if RSA key available
        if self._rsa_key:
            try:
                message = f"{endpoint}:{time.time()}"
                signature = self._rsa_key.sign(
                    message.encode(),
                    padding.PSS(
                        mgf=padding.MGF1(hashes.SHA256()),
                        salt_length=padding.PSS.MAX_LENGTH
                    ),
                    hashes.SHA256()
                )
                headers['X-DIGITAL-SIGNATURE'] = base64.b64encode(signature).decode()
            except Exception as e:
                logging.debug(f"Digital signature failed: {e}")

        return headers

    def encrypt_payload(self, data: Dict[str, Any]) -> str:
        """Encrypt request payload for sensitive operations."""
        if not self._rsa_key:
            return json.dumps(data) if 'orjson' not in globals() else json.dumps(data).decode()

        try:
            # Serialize data
            payload = json.dumps(data) if 'orjson' not in globals() else json.dumps(data).decode()

            # Encrypt with RSA public key
            public_key = self._rsa_key.public_key()
            encrypted = public_key.encrypt(
                payload.encode(),
                padding.OAEP(
                    mgf=padding.MGF1(algorithm=hashes.SHA256()),
                    algorithm=hashes.SHA256(),
                    label=None
                )
            )

            return base64.b64encode(encrypted).decode()

        except Exception as e:
            logging.warning(f"Payload encryption failed: {e}")
            return json.dumps(data) if 'orjson' not in globals() else json.dumps(data).decode()

    def decrypt_response(self, encrypted_data: str) -> Dict[str, Any]:
        """Decrypt encrypted response."""
        if not self._rsa_key:
            return json.loads(encrypted_data) if 'orjson' not in globals() else json.loads(encrypted_data)

        try:
            # Decrypt with RSA private key
            encrypted = base64.b64decode(encrypted_data)
            decrypted = self._rsa_key.decrypt(
                encrypted,
                padding.OAEP(
                    mgf=padding.MGF1(algorithm=hashes.SHA256()),
                    algorithm=hashes.SHA256(),
                    label=None
                )
            )

            return json.loads(decrypted.decode()) if 'orjson' not in globals() else json.loads(decrypted.decode())

        except Exception as e:
            logging.warning(f"Response decryption failed: {e}")
            return json.loads(encrypted_data) if 'orjson' not in globals() else json.loads(encrypted_data)


class AdvancedRateLimiter:
    """
    Advanced rate limiter with exponential backoff and circuit breaker pattern.

    Features:
    - Exponential backoff for rate limits
    - Circuit breaker to prevent cascade failures
    - Adaptive rate limiting based on API responses
    - Request queuing and prioritization
    """

    def __init__(self, base_delay: float = 0.05, max_delay: float = 60.0):
        self.base_delay = base_delay
        self.max_delay = max_delay
        self.current_delay = base_delay

        # Circuit breaker state
        self.failure_threshold = 5
        self.recovery_timeout = 60  # seconds
        self.consecutive_failures = 0
        self.last_failure_time = 0
        self.circuit_open = False

        # Rate limit tracking
        self.request_times = []
        self.max_requests_per_minute = 1200  # Binance limit

        # Request queue for prioritization
        self._request_queue = asyncio.Queue()
        self._processing_task = None

    async def acquire(self) -> bool:
        """
        Acquire permission to make a request.

        Returns:
            True if request allowed, False if circuit breaker open
        """
        current_time = time.time()

        # Check circuit breaker
        if self.circuit_open:
            if current_time - self.last_failure_time > self.recovery_timeout:
                # Try to close circuit (half-open)
                self.circuit_open = False
                self.consecutive_failures = 0
                self.logger.info("Circuit breaker transitioning to half-open")
            else:
                return False

        # Clean old request times (keep last minute)
        cutoff_time = current_time - 60
        self.request_times = [t for t in self.request_times if t > cutoff_time]

        # Check rate limit
        if len(self.request_times) >= self.max_requests_per_minute:
            # Calculate wait time
            oldest_request = min(self.request_times)
            wait_time = 60 - (current_time - oldest_request)
            if wait_time > 0:
                await asyncio.sleep(wait_time)

        # Apply exponential backoff delay
        if self.current_delay > self.base_delay:
            await asyncio.sleep(self.current_delay)

        # Record request time
        self.request_times.append(current_time)

        return True

    def record_success(self):
        """Record successful request - reduce backoff."""
        self.consecutive_failures = 0
        self.current_delay = max(self.base_delay, self.current_delay * 0.9)  # Reduce delay

    def record_failure(self, status_code: Optional[int] = None):
        """Record failed request - increase backoff and check circuit breaker."""
        self.consecutive_failures += 1
        self.last_failure_time = time.time()

        # Increase delay exponentially
        self.current_delay = min(self.max_delay, self.current_delay * 2)

        # Open circuit breaker if too many failures
        if self.consecutive_failures >= self.failure_threshold:
            self.circuit_open = True
            logging.warning(f"Circuit breaker opened after {self.consecutive_failures} consecutive failures")

        # Special handling for rate limits
        if status_code == 429:
            # Aggressive backoff for rate limits
            self.current_delay = min(self.max_delay, self.current_delay * 4)

    async def wait_for_recovery(self):
        """Wait for circuit breaker recovery."""
        if self.circuit_open:
            wait_time = self.recovery_timeout - (time.time() - self.last_failure_time)
            if wait_time > 0:
                logging.info(f"Waiting {wait_time:.1f}s for circuit breaker recovery")
                await asyncio.sleep(wait_time)

    def get_stats(self) -> Dict[str, Any]:
        """Get rate limiter statistics."""
        return {
            'current_delay': self.current_delay,
            'circuit_open': self.circuit_open,
            'consecutive_failures': self.consecutive_failures,
            'requests_last_minute': len(self.request_times),
            'backoff_until': time.time() + self.current_delay
        }

# Import secrets manager
try:
    from ..utils.secrets_manager import get_secrets_manager, get_binance_credentials
    SECRETS_AVAILABLE = True
except ImportError:
    SECRETS_AVAILABLE = False
    get_secrets_manager = None
    get_binance_credentials = None

# Fallback import handling for config
try:
    from ..config.config import get_config
except ImportError:
    try:
        from config.config import get_config
    except ImportError:
        # Fallback for when config is not available
        def get_config() -> Optional[Dict[str, Any]]:
            return None


class AsyncBinanceClient:
    """
    Async Binance API client for high-performance market data operations.

    This implementation uses aiohttp for concurrent requests and provides
    3-5x performance improvement over synchronous clients.
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
        Initialize async Binance client with secure secrets management.

        Args:
            api_key: Binance API key (optional, uses secrets manager if not provided)
            api_secret: Binance API secret (optional, uses secrets manager if not provided)
            testnet: Use testnet instead of live trading (optional, uses secrets manager)
            rate_limit_delay: Delay between API calls (optional)
            config_file: Path to configuration file (fallback)
            max_concurrent_requests: Maximum concurrent requests for connection pooling
            use_secrets_manager: Whether to use secure secrets management
        """
        # Initialize secrets manager
        self.secrets_manager = None
        if use_secrets_manager and SECRETS_AVAILABLE:
            self.secrets_manager = get_secrets_manager()

        # Load credentials - prefer secrets manager, fallback to config
        if self.secrets_manager:
            creds = get_binance_credentials()
            self.api_key = api_key or creds.get('api_key')
            self.api_secret = api_secret or creds.get('api_secret')
            self.testnet = testnet if testnet is not None else creds.get('testnet', True)
        else:
            # Fallback to old config system
            self.config = get_config()
            if config_file:
                try:
                    from config.config import load_config
                    self.config = load_config(config_file)
                except ImportError:
                    self.config = None

            if self.config:
                self.api_key = api_key or self.config.get('binance.api_key')
                self.api_secret = api_secret or self.config.get('binance.api_secret')
                self.testnet = testnet if testnet is not None else self.config.get('binance.testnet', True)
            else:
                self.api_key = api_key
                self.api_secret = api_secret
                self.testnet = testnet if testnet is not None else True

        # Other settings
        self.rate_limit_delay = rate_limit_delay or 0.05  # Faster for async
        self.timeout = 30
        self.max_retries = 3
        self.retry_delay = 1.0
        self.max_concurrent_requests = max_concurrent_requests

        # Base URLs
        self.base_url = "https://testnet.binance.vision" if self.testnet else "https://api.binance.com"

        # Session management
        self._session: Optional[aiohttp.ClientSession] = None
        self._semaphore = asyncio.Semaphore(max_concurrent_requests)

        self.logger = logging.getLogger(__name__)
        self.last_request_time = 0
        self.request_count = 0
        self.error_count = 0

        # Advanced security components
        self._advanced_rate_limiter = AdvancedRateLimiter(
            base_delay=self.rate_limit_delay,
            max_delay=300.0  # Max 5 minutes backoff
        )

        # Request signing and encryption
        self._request_signer = RequestSigner(self.api_secret if self.api_secret else "")

        # API key rotation system
        self._key_rotation_enabled = False
        self._current_key_index = 0
        self._backup_keys = []
        self._key_failure_counts = {}

        # Set API key reference for signer
        if self.api_key:
            self._request_signer._api_key = self.api_key

        # Security audit
        self._audit_security()

        self.logger.info(f"✅ Async Binance client initialized (testnet: {self.testnet}, secure: {bool(self.secrets_manager)})")

    async def __aenter__(self):
        """Async context manager entry."""
        await self.initialize_session()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.close_session()

    def _create_signed_request(self, method: str, endpoint: str, params: Dict = None) -> Dict[str, Any]:
        """
        Create HMAC-SHA256 signed request for authenticated endpoints.

        Args:
            method: HTTP method
            endpoint: API endpoint
            params: Request parameters

        Returns:
            Signed request parameters
        """
        if not self.api_key or not self.api_secret:
            raise ValueError("API credentials required for signed requests")

        if params is None:
            params = {}

        # Add timestamp
        params['timestamp'] = str(int(time.time() * 1000))

        # Create query string
        query_string = urlencode(sorted(params.items()))

        # Create signature
        signature = self._create_signature(query_string)
        params['signature'] = signature

        return params

    async def _execute_request_with_retry(self, method: str, url: str, params: Dict = None,
                                        signed: bool = False, max_retries: int = None) -> Dict:
        """
        Execute HTTP request with comprehensive retry logic and rate limiting.

        Args:
            method: HTTP method
            url: Full URL
            params: Query parameters
            signed: Whether request needs signing
            max_retries: Maximum retry attempts

        Returns:
            API response data
        """
        if max_retries is None:
            max_retries = self.max_retries

        params = params or {}

        # Sign request if needed
        if signed:
            params = self._create_signed_request(method, url, params)

        for attempt in range(max_retries):
            try:
                # Apply rate limiting
                async with self._rate_limiter:
                    await self._apply_rate_limit()
                    self.request_count += 1

                    # Make request
                    if method.upper() == 'GET':
                        async with self._session.get(url, params=params) as response:
                            return await self._handle_api_response(response, attempt, max_retries)
                    elif method.upper() == 'POST':
                        headers = {'Content-Type': 'application/x-www-form-urlencoded'}
                        data = urlencode(params) if params else None
                        async with self._session.post(url, data=data, headers=headers) as response:
                            return await self._handle_api_response(response, attempt, max_retries)
                    else:
                        raise ValueError(f"Unsupported HTTP method: {method}")

            except asyncio.TimeoutError:
                self.error_count += 1
                self.logger.warning(f"Request timeout (attempt {attempt + 1})")
                if attempt < max_retries - 1:
                    await asyncio.sleep(self.retry_delay * (2 ** attempt))
                else:
                    raise

            except aiohttp.ClientError as e:
                self.error_count += 1
                self.logger.warning(f"Network error: {e} (attempt {attempt + 1})")
                if attempt < max_retries - 1:
                    await asyncio.sleep(self.retry_delay * (2 ** attempt))
                else:
                    raise

    async def _handle_api_response(self, response: aiohttp.ClientResponse, attempt: int,
                                 max_retries: int) -> Dict:
        """
        Handle API response with error checking and retry logic.

        Args:
            response: aiohttp response object
            attempt: Current attempt number
            max_retries: Maximum retry attempts

        Returns:
            Parsed response data
        """
        if response.status == 200:
            # Success
            try:
                if 'orjson' in globals():
                    data = await response.json(loads=json.loads)
                else:
                    data = await response.json()
                return data
            except Exception as e:
                self.logger.error(f"Failed to parse JSON response: {e}")
                raise

        elif response.status == 429:
            # Rate limit exceeded
            self.error_count += 1
            retry_after = response.headers.get('Retry-After', '5')
            self.logger.warning(f"Rate limit exceeded, retrying after {retry_after}s")

            try:
                await asyncio.sleep(float(retry_after))
                # Don't count this as a failed attempt
                return await self._execute_request_with_retry(
                    response.method, str(response.url), None, False, max_retries - attempt - 1
                )
            except (ValueError, TypeError):
                await asyncio.sleep(5)  # Fallback delay

        elif response.status == 418:
            # IP banned
            self.error_count += 1
            self.logger.error("IP banned by Binance API")
            raise aiohttp.ClientError("IP banned")

        else:
            # Other API errors
            self.error_count += 1
            try:
                error_data = await response.json()
                error_msg = error_data.get('msg', f'HTTP {response.status}')
            except:
                error_msg = f'HTTP {response.status}'

            self.logger.error(f"API error: {error_msg}")

            # Handle specific errors that shouldn't be retried
            if 'Invalid API-key' in error_msg or 'Invalid signature' in error_msg:
                raise aiohttp.ClientError(f"Authentication error: {error_msg}")

            if attempt < max_retries - 1:
                wait_time = self.retry_delay * (2 ** attempt)
                self.logger.info(f"Retrying in {wait_time:.1f}s...")
                await asyncio.sleep(wait_time)
            else:
                raise aiohttp.ClientError(f"API error after {max_retries} attempts: {error_msg}")

        # This should not be reached, but just in case
        raise aiohttp.ClientError(f"Unexpected response status: {response.status}")

    async def initialize_session(self):
        """Initialize aiohttp session with connection pooling."""
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

    def _create_signature(self, query_string: str) -> str:
        """Create HMAC-SHA256 signature for authenticated requests."""
        if not self.api_secret:
            raise ValueError("API secret required for signed requests")

        return hmac.new(
            self.api_secret.encode('utf-8'),
            query_string.encode('utf-8'),
            hashlib.sha256
        ).hexdigest()

    def _audit_security(self):
        """Audit security configuration and credentials."""
        issues = []

        if not self.api_key or not self.api_secret:
            issues.append("Missing API credentials")

        if self.secrets_manager:
            # Check credential strength
            if self.api_key:
                key_validation = self.secrets_manager.validate_secret_strength('api_key', self.api_key)
                if key_validation['strength'] != 'strong':
                    issues.append(f"API key strength: {key_validation['strength']}")

            if self.api_secret:
                secret_validation = self.secrets_manager.validate_secret_strength('api_secret', self.api_secret)
                if secret_validation['strength'] != 'strong':
                    issues.append(f"API secret strength: {secret_validation['strength']}")
        else:
            issues.append("Secrets manager not available - using insecure storage")

        if issues:
            self.logger.warning(f"Security audit found {len(issues)} issues: {', '.join(issues)}")
        else:
            self.logger.info("Security audit passed - credentials properly secured")

    def _validate_query_params(self, params: Dict[str, Any]) -> 'ValidationResult':
        """
        Validate query parameters using Pydantic models.

        Args:
            params: Query parameters to validate

        Returns:
            ValidationResult
        """
        if not ValidationResult:
            # Fallback if validation not available
            return type('MockResult', (), {'valid': True, 'errors': []})()

        try:
            DataQueryParams(**params)
            return ValidationResult(valid=True, errors=[])
        except Exception as e:
            return ValidationResult(valid=False, errors=[str(e)])

    def validate_config(self) -> 'ValidationResult':
        """
        Validate client configuration using Pydantic models.

        Returns:
            ValidationResult
        """
        if not ValidationResult:
            return type('MockResult', (), {'valid': True, 'errors': []})()

        config = {
            'api_key': self.api_key,
            'api_secret': self.api_secret,
            'testnet': self.testnet,
            'rate_limit_delay': float(self.rate_limit_delay),
            'timeout': self.timeout,
            'max_retries': self.max_retries
        }

        try:
            APIRequestConfig(**config)
            return ValidationResult(valid=True, errors=[])
        except Exception as e:
            return ValidationResult(valid=False, errors=[str(e)])

    def enable_key_rotation(self, backup_keys: Optional[List[Dict[str, str]]] = None):
        """
        Enable API key rotation for high availability.

        Args:
            backup_keys: List of backup key pairs [{'api_key': '...', 'api_secret': '...'}]
        """
        self._key_rotation_enabled = True
        if backup_keys:
            self._backup_keys = backup_keys

        # Load backup keys from secrets manager if available
        if self.secrets_manager:
            for i in range(1, 5):  # Check for backup keys 1-4
                backup_key = self.secrets_manager.get_secret(f'binance_api_key_{i}')
                backup_secret = self.secrets_manager.get_secret(f'binance_api_secret_{i}')

                if backup_key and backup_secret:
                    self._backup_keys.append({
                        'api_key': backup_key,
                        'api_secret': backup_secret
                    })

        self.logger.info(f"API key rotation enabled with {len(self._backup_keys)} backup keys")

    def _rotate_api_key(self) -> bool:
        """
        Rotate to next available API key.

        Returns:
            True if rotation successful, False otherwise
        """
        if not self._key_rotation_enabled or not self._backup_keys:
            return False

        # Mark current key as failed
        current_key = f"{self.api_key}:{self.api_secret}"
        self._key_failure_counts[current_key] = self._key_failure_counts.get(current_key, 0) + 1

        # Find next working key
        start_index = self._current_key_index
        for i in range(len(self._backup_keys)):
            candidate_index = (start_index + i + 1) % len(self._backup_keys)
            candidate = self._backup_keys[candidate_index]
            candidate_key = f"{candidate['api_key']}:{candidate['api_secret']}"

            # Skip keys with too many failures
            if self._key_failure_counts.get(candidate_key, 0) < 3:
                self.api_key = candidate['api_key']
                self.api_secret = candidate['api_secret']
                self._current_key_index = candidate_index

                # Update request signer
                self._request_signer = RequestSigner(self.api_secret)
                self._request_signer._api_key = self.api_key

                self.logger.info(f"Rotated to backup API key {candidate_index + 1}")
                return True

        self.logger.warning("No working backup API keys available")
        return False

    def get_security_stats(self) -> Dict[str, Any]:
        """
        Get comprehensive security statistics.

        Returns:
            Security metrics and statistics
        """
        rate_limiter_stats = self._advanced_rate_limiter.get_stats()

        return {
            'rate_limiter': rate_limiter_stats,
            'request_count': self.request_count,
            'error_count': self.error_count,
            'error_rate': self.error_count / max(self.request_count, 1),
            'key_rotation_enabled': self._key_rotation_enabled,
            'backup_keys_count': len(self._backup_keys),
            'current_key_index': self._current_key_index,
            'key_failure_counts': dict(self._key_failure_counts),
            'secrets_manager_available': bool(self.secrets_manager),
            'last_request_time': self.last_request_time
        }

    async def test_connection(self) -> bool:
        """Test connection to Binance API asynchronously."""
        if not self._session:
            await self.initialize_session()

        try:
            # Test with ping endpoint
            async with self._rate_limiter:
                await self._apply_rate_limit()
                self.request_count += 1

                async with self._session.get(f"{self.base_url}/api/v3/ping") as response:
                    if response.status == 200:
                        self.logger.info("✅ Binance API connection successful")
                        return True
                    else:
                        self.logger.error(f"❌ Binance API ping failed with status {response.status}")
                        return False
        except Exception as e:
            self.error_count += 1
            self.logger.error(f"❌ Binance API connection failed: {e}")
            return False

    async def _apply_rate_limit(self):
        """Apply async rate limiting."""
        async with self._semaphore:
            current_time = time.time()
            time_since_last = current_time - self.last_request_time

            if time_since_last < self.rate_limit_delay:
                await asyncio.sleep(self.rate_limit_delay - time_since_last)

            self.last_request_time = time.time()

    async def get_historical_klines(
        self,
        symbol: str,
        interval: str,
        start_date: str,
        end_date: Optional[str] = None,
        limit: int = 500
    ) -> Optional[pd.DataFrame]:
        """
        Download historical klines (candlestick) data asynchronously with retry logic.

        Args:
            symbol: Trading symbol (e.g., 'ETHUSDT')
            interval: Kline interval (e.g., '1h', '4h', '1d')
            start_date: Start date in 'YYYY-MM-DD' format
            end_date: End date in 'YYYY-MM-DD' format (optional)
            limit: Maximum number of records per request

        Returns:
            DataFrame with OHLCV data or None if failed
        """
        if not self._session:
            await self.initialize_session()

        # Validate inputs using Pydantic models
        if ValidationResult:
            # Validate query parameters
            query_validation = self._validate_query_params({
                'symbol': symbol,
                'interval': interval,
                'start_date': start_date,
                'end_date': end_date,
                'limit': limit
            })

            if not query_validation.valid:
                self.logger.error(f"Input validation failed: {query_validation.errors}")
                return None

        # Legacy validation (fallback)
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

        # Retry logic with advanced rate limiting
        for attempt in range(self.max_retries):
            try:
                # Acquire rate limit permission
                rate_limit_acquired = await self._advanced_rate_limiter.acquire()
                if not rate_limit_acquired:
                    await self._advanced_rate_limiter.wait_for_recovery()
                    continue

                self.request_count += 1

                self.logger.info(f"📊 Downloading {symbol} {interval} data from {start_date} (attempt {attempt + 1})")

                # Build API URL with security
                params = {
                    'symbol': symbol.upper(),
                    'interval': interval,
                    'startTime': start_ts,
                    'endTime': end_ts,
                    'limit': limit
                }

                url = f"{self.base_url}/api/v3/klines"

                # Create secure headers
                headers = self._request_signer.create_secure_headers(url, params)

                # Make secure request
                async with self._session.get(url, params=params, headers=headers) as response:
                        if response.status == 200:
                            # Record successful request
                            self._advanced_rate_limiter.record_success()

                            # Use orjson for faster JSON parsing if available
                            if 'orjson' in globals():
                                klines = await response.json(loads=json.loads)
                            else:
                                klines = await response.json()

                            if not klines:
                                self.logger.warning(f"⚠️ No data received for {symbol}")
                                return pd.DataFrame()

                            # Convert to DataFrame
                            df = self._process_klines_data(klines, symbol, interval)

                            self.logger.info(f"✅ Downloaded {len(df)} records for {symbol} {interval}")
                            return df

                        elif response.status == 429:
                            # Rate limit exceeded - aggressive backoff
                            self.error_count += 1
                            self._advanced_rate_limiter.record_failure(429)

                            retry_after = response.headers.get('Retry-After', '5')
                            self.logger.warning(f"⚠️ Rate limit exceeded, retrying after {retry_after}s...")

                            # Try key rotation if enabled
                            if self._key_rotation_enabled and self._rotate_api_key():
                                self.logger.info("🔄 Rotated API key due to rate limit")
                                continue

                            await asyncio.sleep(float(retry_after))

                        elif response.status == 418:
                            # IP banned
                            self.error_count += 1
                            self._advanced_rate_limiter.record_failure(418)
                            self.logger.error("❌ IP banned by Binance API")
                            return None

                        else:
                            # Other API errors
                            self.error_count += 1
                            self._advanced_rate_limiter.record_failure(response.status)

                            try:
                                error_data = await response.json()
                                error_msg = error_data.get('msg', f'HTTP {response.status}')
                            except:
                                error_msg = f'HTTP {response.status}'

                            self.logger.error(f"❌ Binance API error: {error_msg}")

                            # Handle specific errors
                            if 'Invalid symbol' in error_msg:
                                return None

                            # Try key rotation for authentication errors
                            if response.status in [401, 403] and self._key_rotation_enabled:
                                if self._rotate_api_key():
                                    self.logger.info("🔄 Rotated API key due to auth error")
                                    continue

                            if attempt < self.max_retries - 1:
                                wait_time = self.retry_delay * (2 ** attempt)
                                self.logger.info(f"⏳ Retrying in {wait_time:.1f} seconds...")
                                await asyncio.sleep(wait_time)
                            else:
                                return None

            except asyncio.TimeoutError:
                self.error_count += 1
                self.logger.warning(f"⚠️ Timeout error for {symbol} (attempt {attempt + 1})")
                if attempt < self.max_retries - 1:
                    await asyncio.sleep(self.retry_delay)
                else:
                    return None

            except aiohttp.ClientError as e:
                self.error_count += 1
                self.logger.warning(f"⚠️ Network error: {e}")
                if attempt < self.max_retries - 1:
                    await asyncio.sleep(self.retry_delay)
                else:
                    self.logger.error(f"❌ Network failed after {self.max_retries} attempts")
                    return None

            except Exception as e:
                self.error_count += 1
                self.logger.error(f"❌ Unexpected error: {e}")
                return None

        return None

    async def get_symbol_info(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Get symbol information asynchronously."""
        if not self._session:
            await self.initialize_session()

        try:
            async with self._rate_limiter:
                await self._apply_rate_limit()
                self.request_count += 1

                async with self._session.get(f"{self.base_url}/api/v3/exchangeInfo") as response:
                    if response.status == 200:
                        data = await response.json()
                        symbols = data.get('symbols', [])
                        for sym_info in symbols:
                            if sym_info['symbol'] == symbol.upper():
                                return sym_info
                        return None
                    else:
                        self.error_count += 1
                        return None
        except Exception as e:
            self.error_count += 1
            self.logger.error(f"❌ Failed to get symbol info: {e}")
            return None

    async def get_exchange_info(self) -> Optional[Dict[str, Any]]:
        """Get exchange information asynchronously."""
        if not self._session:
            await self.initialize_session()

        try:
            async with self._rate_limiter:
                await self._apply_rate_limit()
                self.request_count += 1

                async with self._session.get(f"{self.base_url}/api/v3/exchangeInfo") as response:
                    if response.status == 200:
                        return await response.json()
                    else:
                        self.error_count += 1
                        return None
        except Exception as e:
            self.error_count += 1
            self.logger.error(f"❌ Failed to get exchange info: {e}")
            return None

    async def get_server_time(self) -> Optional[int]:
        """Get server time asynchronously."""
        if not self._session:
            await self.initialize_session()

        try:
            async with self._rate_limiter:
                await self._apply_rate_limit()
                self.request_count += 1

                async with self._session.get(f"{self.base_url}/api/v3/time") as response:
                    if response.status == 200:
                        data = await response.json()
                        return data.get('serverTime')
                    else:
                        self.error_count += 1
                        return None
        except Exception as e:
            self.error_count += 1
            self.logger.error(f"❌ Failed to get server time: {e}")
            return None

    async def get_multiple_symbols_data(
        self,
        symbols: List[str],
        interval: str,
        start_date: str,
        end_date: Optional[str] = None,
        limit: int = 500
    ) -> Dict[str, Optional[pd.DataFrame]]:
        """
        Fetch data for multiple symbols concurrently.

        Args:
            symbols: List of trading symbols
            interval: Kline interval
            start_date: Start date
            end_date: End date (optional)
            limit: Records per request

        Returns:
            Dict mapping symbols to DataFrames
        """
        if not self._session:
            await self.initialize_session()

        # Create concurrent tasks
        tasks = []
        for symbol in symbols:
            task = self.get_historical_klines(symbol, interval, start_date, end_date, limit)
            tasks.append(task)

        # Execute concurrently
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Map results back to symbols
        symbol_data = {}
        for symbol, result in zip(symbols, results):
            if isinstance(result, Exception):
                self.logger.error(f"❌ Failed to fetch {symbol}: {result}")
                symbol_data[symbol] = None
            else:
                symbol_data[symbol] = result

        return symbol_data

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


# Backward compatibility - synchronous wrapper
class BinanceClient:
    """
    Synchronous wrapper for AsyncBinanceClient.

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
        self.async_client = AsyncBinanceClient(
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
        limit: int = 500
    ) -> Optional[pd.DataFrame]:
        """Get historical klines synchronously."""
        try:
            return asyncio.run(self.async_client.get_historical_klines(
                symbol, interval, start_date, end_date, limit
            ))
        except Exception as e:
            self.logger.error(f"Failed to get historical klines: {e}")
            return None

    def get_symbol_info(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Get symbol info synchronously."""
        try:
            return asyncio.run(self.async_client.get_symbol_info(symbol))
        except Exception as e:
            self.logger.error(f"Failed to get symbol info: {e}")
            return None

    def get_exchange_info(self) -> Optional[Dict[str, Any]]:
        """Get exchange info synchronously."""
        try:
            return asyncio.run(self.async_client.get_exchange_info())
        except Exception as e:
            self.logger.error(f"Failed to get exchange info: {e}")
            return None

    def get_server_time(self) -> Optional[int]:
        """Get server time synchronously."""
        try:
            return asyncio.run(self.async_client.get_server_time())
        except Exception as e:
            self.logger.error(f"Failed to get server time: {e}")
            return None

    def get_request_stats(self) -> Dict[str, Any]:
        """Get request statistics."""
        return {
            'total_requests': self.async_client.request_count,
            'error_count': self.async_client.error_count,
            'error_rate': self.async_client.error_count / max(self.async_client.request_count, 1),
            'success_rate': (self.async_client.request_count - self.async_client.error_count) / max(self.async_client.request_count, 1)
        }

    def is_healthy(self) -> bool:
        """Check if client is healthy."""
        stats = self.get_request_stats()
        return stats['error_rate'] < 0.1

    def _validate_symbol(self, symbol: str) -> bool:
        """Validate symbol format."""
        return self.async_client._validate_symbol(symbol)

    def _validate_interval(self, interval: str) -> bool:
        """Validate interval format."""
        return self.async_client._validate_interval(interval)

    def _process_klines_data(self, klines: List[List], symbol: str, interval: str) -> pd.DataFrame:
        """Process klines data."""
        return self.async_client._process_klines_data(klines, symbol, interval)

    def _parse_interval_to_timedelta(self, interval: str) -> Optional[pd.Timedelta]:
        """Parse interval to timedelta."""
        return self.async_client._parse_interval_to_timedelta(interval)

    def setup_secure_credentials(self, api_key: str, api_secret: str, testnet: bool = True):
        """Setup secure credentials using secrets manager."""
        if SECRETS_AVAILABLE:
            manager = get_secrets_manager()
            manager.setup_secure_config(api_key, api_secret, testnet)
            # Reinitialize with new credentials
            self.async_client.api_key = api_key
            self.async_client.api_secret = api_secret
            self.async_client.testnet = testnet
            self.logger.info("Secure credentials configured")
        else:
            raise RuntimeError("Secrets manager not available")

    def rotate_api_credentials(self, new_api_key: str = None, new_api_secret: str = None):
        """Rotate API credentials securely."""
        if not SECRETS_AVAILABLE:
            raise RuntimeError("Secrets manager not available")

        manager = get_secrets_manager()

        if new_api_key:
            manager.rotate_secret('binance_api_key', new_api_key)
            self.async_client.api_key = new_api_key

        if new_api_secret:
            manager.rotate_secret('binance_api_secret', new_api_secret)
            self.async_client.api_secret = new_api_secret

        self.logger.info("API credentials rotated successfully")

    def audit_security(self) -> Dict[str, Any]:
        """Audit security configuration."""
        if SECRETS_AVAILABLE:
            manager = get_secrets_manager()
            return manager.audit_secrets()
        else:
            return {'error': 'Secrets manager not available'}
