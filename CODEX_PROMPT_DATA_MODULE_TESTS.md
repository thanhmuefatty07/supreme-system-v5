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
  1. DataPipeline (src/data/data_pipeline.py)
  2. DataValidator (src/data/data_validator.py)
  3. DataStorage (src/data/data_storage.py)
  4. BinanceClient (src/data/binance_client.py)
  5. AsyncBinanceClient (src/data/binance_client.py)
  6. RealtimeClient (src/data/realtime_client.py)
- Dependencies: pandas, pyarrow, aiohttp, binance library

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
- `__init__()` - Initialization với config
- `fetch_data()` - Data fetching từ multiple sources
- `process_data()` - Data processing và validation
- `store_data()` - Data storage
- `get_data()` - Data retrieval
- `validate_data()` - Data validation
- Error handling và retries

**Edge Cases:**
- Empty data sources
- Network failures
- Invalid data formats
- Concurrent requests
- Large datasets

### **2. DataValidator Tests:**

**Key Methods to Test:**
- `validate_ohlcv()` - OHLCV validation
- `validate_dataframe()` - DataFrame validation
- `clean_data()` - Data cleaning
- `detect_anomalies()` - Anomaly detection
- Pydantic model validation

**Edge Cases:**
- Invalid OHLC relationships (high < low)
- Missing required columns
- Out-of-range values
- Duplicate timestamps
- Timezone issues

### **3. DataStorage Tests:**

**Key Methods to Test:**
- `save_data()` - Save data to Parquet
- `load_data()` - Load data from Parquet
- `query_data()` - Query data by date range/symbol
- `delete_data()` - Delete data
- `get_metadata()` - Get storage metadata

**Edge Cases:**
- File system errors
- Corrupted Parquet files
- Missing files
- Large file handling
- Concurrent read/write

### **4. BinanceClient Tests:**

**Key Methods to Test:**
- `get_klines()` - Get historical klines
- `get_ticker()` - Get ticker data
- `get_orderbook()` - Get orderbook
- Request signing và authentication
- Rate limiting handling
- Error handling (API errors, network errors)

**Edge Cases:**
- API rate limits
- Network timeouts
- Invalid API keys
- Invalid symbols
- Invalid time ranges

### **5. AsyncBinanceClient Tests:**

**Key Methods to Test:**
- All async versions of BinanceClient methods
- Concurrent requests
- Async error handling
- Connection pooling

**Edge Cases:**
- Async context manager errors
- Concurrent request limits
- Async timeout handling

### **6. RealtimeClient Tests:**

**Key Methods to Test:**
- WebSocket connection
- Real-time data streaming
- Reconnection logic
- Message parsing

**Edge Cases:**
- Connection failures
- WebSocket errors
- Message parsing errors
- Reconnection scenarios

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

