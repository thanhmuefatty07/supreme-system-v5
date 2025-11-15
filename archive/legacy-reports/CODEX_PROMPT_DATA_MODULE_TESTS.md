# 📝 CODEX WEB PROMPT - Data Module Tests

**Date:** 2025-11-14  
**Purpose:** Generate comprehensive tests for `src/data/` module

---

## 🎯 **PROMPT FOR CODEX WEB**

```
Generate comprehensive unit tests for data pipeline modules in src/data/.

Context:
- Module: src/data/
- Key Classes:
  1. DataPipeline (src/data/data_pipeline.py) - Main orchestration class
  2. DataValidator (src/data/data_validator.py) - Data validation with Pydantic models
  3. DataStorage (src/data/data_storage.py) - Parquet storage with partitioning
  4. BinanceClient (src/data/binance_client.py) - Synchronous wrapper
  5. AsyncBinanceClient (src/data/binance_client.py) - Async implementation with rate limiting
  6. BinanceWebSocketClient (src/data/realtime_client.py) - Real-time WebSocket client
- Dependencies: pandas, pyarrow, aiohttp, websockets, binance library, pydantic

Requirements:
1. Coverage: Test all public methods của mỗi class
2. Edge Cases:
   - Empty DataFrames
   - Invalid data formats
   - Missing columns
   - NaN values
   - Network errors (mock)
   - File I/O errors (mock)
   - Invalid timestamps
   - Out-of-range values
   - Concurrent access (for async classes)
3. Test Structure:
   - One test class per main class
   - Use pytest fixtures for sample data
   - Use unittest.mock for external dependencies
   - Test async methods với pytest-asyncio
   - Test error handling và retries
4. Integration:
   - Test data pipeline workflow (fetch → validate → store)
   - Test storage và retrieval
   - Test validation pipeline
5. Performance:
   - Test với large datasets (1000+ rows)
   - Measure storage/retrieval latency
6. Documentation:
   - Docstrings for each test class
   - Clear test method names
   - Comments for complex test logic

Expected Output:
- File: tests/unit/test_data_codex.py
- Structure: Organized by class
- Coverage: >90% for data module
- All tests should pass với current implementation
- Follow same pattern as test_advanced_risk_manager_codex.py và test_strategies_codex.py
```

---

## 📋 **DETAILED REQUIREMENTS BY CLASS**

### **1. DataPipeline Tests:**

**Key Methods to Test:**
- `__init__(config_file, use_async)` - Initialization với config và async mode
- `fetch_and_store_data()` - Complete sync pipeline (fetch → validate → store)
- `fetch_and_store_data_async()` - Complete async pipeline
- `fetch_multiple_symbols_async()` - Concurrent fetching với semaphore
- `process_data()` - Data processing và validation
- `get_data()` - Data retrieval với caching
- `update_symbol_data()` - Update recent data
- `batch_update_symbols()` - Batch updates
- `validate_data_quality()` - Quality validation
- `get_pipeline_status()` - Status reporting
- `export_data()` - Export to CSV/JSON/Parquet
- `clear_cache()` - Cache management
- `optimize_storage()` - Storage optimization
- `_get_cached_data()` - Cache retrieval
- `_cache_data()` - Cache storage

**Edge Cases:**
- Empty data sources
- Network failures (mock AsyncBinanceClient)
- Invalid data formats
- Concurrent requests với max_concurrent limits
- Large datasets (1000+ rows)
- Cache expiration và TTL
- Force refresh vs cached data
- Storage failures

### **2. DataValidator Tests:**

**Key Methods to Test:**
- `__init__()` - Initialization với thresholds
- `validate_with_pydantic()` - Pydantic model validation (ohlcv_datapoint, trading_symbol, kline_interval, strategy_config, api_config, query_params)
- `validate_dataframe_with_models()` - DataFrame validation với Pydantic
- `validate_api_inputs()` - API parameter validation
- `validate_strategy_config()` - Strategy config validation
- `sanitize_input_data()` - Data sanitization
- `validate_ohlcv_data()` - Comprehensive OHLCV validation
- `validate_ohlcv()` - Simple OHLCV validation (for testing framework)
- `clean_data()` - Data cleaning với outlier removal
- `generate_quality_report()` - Quality report generation
- `_validate_structure()` - Structure validation
- `_validate_price_integrity()` - Price integrity checks
- `_validate_volume_data()` - Volume validation
- `_validate_timestamps()` - Timestamp validation
- `_validate_statistics()` - Statistical outlier detection
- `_validate_cross_field_consistency()` - Cross-field consistency
- `_fix_common_data_issues()` - Data fixing logic

**Edge Cases:**
- Invalid OHLC relationships (high < low, open/close outside range)
- Missing required columns
- Out-of-range values (extreme price changes >50%)
- Duplicate timestamps
- Negative prices/volumes
- Zero volume warnings (>10% zero volume)
- Large time gaps
- Statistical outliers (5-sigma threshold)
- NaN values trong DataFrame
- Invalid timestamp formats
- Pydantic ValidationError handling

### **3. DataStorage Tests:**

**Key Methods to Test:**
- `__init__(base_dir)` - Initialization với directory structure
- `store_historical_data()` - Store với partitioning by year/month
- `load_historical_data()` - Load với date filtering
- `update_data()` - Update với modes: append, replace, merge
- `get_data_info()` - Get storage information
- `store_data()` - Enhanced storage với compression metrics
- `query_data()` - Query by date range
- `cleanup_old_data()` - Cleanup old files
- `_filter_files_by_date()` - Date filtering logic
- `_remove_symbol_data()` - Remove data for symbol
- `_save_metadata()` - Metadata persistence

**Edge Cases:**
- File system errors (PermissionError, OSError)
- Corrupted Parquet files
- Missing files/directories
- Large file handling (10000+ rows per partition)
- Concurrent read/write (use threading locks)
- Empty DataFrames
- Invalid date ranges
- Partition directory creation
- Compression ratio calculations
- Memory optimization metrics
- Metadata file I/O errors

### **4. BinanceClient & AsyncBinanceClient Tests:**

**Key Methods to Test (AsyncBinanceClient):**
- `__init__()` - Initialization với secrets manager
- `initialize_session()` - aiohttp session setup
- `close_session()` - Session cleanup
- `test_connection()` - Connection testing
- `get_historical_klines()` - Async klines download với retry logic
- `get_symbol_info()` - Symbol information
- `get_exchange_info()` - Exchange information
- `get_server_time()` - Server time
- `get_multiple_symbols_data()` - Concurrent fetching
- `get_request_stats()` - Request statistics
- `is_healthy()` - Health check
- `validate_config()` - Config validation với Pydantic
- `enable_key_rotation()` - API key rotation
- `get_security_stats()` - Security metrics
- `_create_signature()` - HMAC-SHA256 signing
- `_validate_symbol()` - Symbol validation
- `_validate_interval()` - Interval validation
- `_process_klines_data()` - Data processing
- `_apply_rate_limit()` - Rate limiting
- `_execute_request_with_retry()` - Retry logic
- `_handle_api_response()` - Response handling

**Key Methods to Test (BinanceClient - Sync wrapper):**
- `__init__()` - Wrapper initialization
- `test_connection()` - Sync connection test
- `get_historical_klines()` - Sync wrapper
- `get_symbol_info()` - Sync wrapper
- `get_exchange_info()` - Sync wrapper
- `get_server_time()` - Sync wrapper
- `get_request_stats()` - Stats wrapper
- `is_healthy()` - Health wrapper

**Edge Cases:**
- API rate limits (429 status) với exponential backoff
- Network timeouts (asyncio.TimeoutError)
- Invalid API keys (401/403)
- Invalid symbols
- Invalid time ranges
- Circuit breaker activation (5 consecutive failures)
- Key rotation scenarios
- Rate limiter semaphore limits
- Empty responses
- JSON parsing errors
- Connection pool exhaustion
- Retry logic với max_retries

### **5. RequestSigner & AdvancedRateLimiter Tests:**

**Key Methods to Test (RequestSigner):**
- `__init__(api_secret)` - Initialization
- `sign_request()` - HMAC-SHA256 signing
- `create_secure_headers()` - Header creation
- `encrypt_payload()` - RSA encryption
- `decrypt_response()` - RSA decryption

**Key Methods to Test (AdvancedRateLimiter):**
- `__init__()` - Initialization
- `acquire()` - Rate limit acquisition
- `record_success()` - Success recording
- `record_failure()` - Failure recording với circuit breaker
- `wait_for_recovery()` - Circuit breaker recovery
- `get_stats()` - Statistics

**Edge Cases:**
- Missing API secret
- RSA key generation failures
- Circuit breaker state transitions
- Rate limit queue management
- Exponential backoff calculations

### **6. BinanceWebSocketClient Tests:**

**Key Methods to Test:**
- `__init__(config_file)` - Initialization
- `add_stream()` - Add stream subscription
- `remove_stream()` - Remove stream
- `subscribe_price_stream()` - Price ticker subscription
- `subscribe_trade_stream()` - Trade stream subscription
- `subscribe_kline_stream()` - Kline subscription
- `subscribe_depth_stream()` - Order book depth subscription
- `start()` - Start WebSocket client
- `stop()` - Stop client
- `get_stream_data()` - Get buffered data
- `get_latest_price()` - Get latest price
- `get_recent_trades()` - Get recent trades
- `get_order_book()` - Get order book
- `get_metrics()` - Performance metrics
- `add_connect_callback()` - Connection callbacks
- `add_disconnect_callback()` - Disconnect callbacks
- `add_error_callback()` - Error callbacks
- `add_message_callback()` - Message callbacks
- `_run_client()` - Thread-based client runner
- `_async_client()` - Async client loop
- `_connect_and_listen()` - Connection và message handling
- `_subscribe_to_streams()` - Stream subscription
- `_handle_message()` - Message parsing và buffering
- `_ping_loop()` - Keep-alive pings
- `_handle_reconnection()` - Reconnection với exponential backoff

**Edge Cases:**
- Connection failures (websockets.exceptions.ConnectionClosed)
- WebSocket errors
- Message parsing errors (JSONDecodeError)
- Reconnection scenarios với max_reconnect_attempts
- Buffer overflow (1000+ messages)
- Thread safety (threading.Thread)
- Event loop management (asyncio.new_event_loop)
- Ping failures
- Invalid stream names
- Callback exceptions
- Testnet vs live URL switching

---

## ✅ **EXPECTED TEST STRUCTURE**

```python
# tests/unit/test_data_codex.py

import pytest
import pandas as pd
import numpy as np
from unittest.mock import Mock, patch, MagicMock
import asyncio

from src.data.data_pipeline import DataPipeline
from src.data.data_validator import DataValidator
from src.data.data_storage import DataStorage
from src.data.binance_client import BinanceClient, AsyncBinanceClient
from src.data.realtime_client import RealtimeClient

@pytest.fixture
def sample_ohlcv_data():
    """Sample OHLCV DataFrame for testing."""
    ...

class TestDataPipeline:
    """Test DataPipeline orchestration."""
    ...

class TestDataValidator:
    """Test data validation logic."""
    ...

class TestDataStorage:
    """Test data storage và retrieval."""
    ...

class TestBinanceClient:
    """Test Binance API client."""
    ...

class TestAsyncBinanceClient:
    """Test async Binance client."""
    ...

class TestRealtimeClient:
    """Test real-time data client."""
    ...
```

---

## 🎯 **SUCCESS CRITERIA**

- ✅ All public methods tested
- ✅ Edge cases covered
- ✅ Error handling tested
- ✅ Integration tests included
- ✅ Performance tests included
- ✅ All tests pass
- ✅ Coverage >90% for data module

---

**Ready to use với Codex Web!**



