# 📋 Data Module Tests Generation Guide

**Date:** 2025-01-14  
**Status:** Ready for Codex Web

---

## 🎯 **OBJECTIVE**

Generate comprehensive unit tests for `src/data/` module using Codex Web, following the same pattern as `test_advanced_risk_manager_codex.py` and `test_strategies_codex.py`.

---

## 📝 **STEPS TO GENERATE TESTS**

### **Step 1: Open Codex Web**

1. Navigate to Codex Web interface
2. Select the project: `supreme-system-v5`

### **Step 2: Use the Prompt Template**

Copy the entire content from `CODEX_PROMPT_DATA_MODULE_TESTS.md` and paste it into Codex Web.

**Key Points:**
- The prompt includes detailed requirements for all 6 main classes
- Edge cases are specified for each class
- Expected test structure is provided
- Follow the same pattern as existing Codex-generated tests

### **Step 3: Generate Tests**

1. Codex Web will generate `tests/unit/test_data_codex.py`
2. Expected file size: ~1500-2000 lines
3. Expected test count: 100-150 tests
4. Coverage target: >90% for data module

### **Step 4: Copy Generated Code**

1. Copy the generated test file from Codex Web
2. Save it locally as `tests/unit/test_data_codex.py`

### **Step 5: Run Tests Locally**

```bash
# Run all data tests
pytest tests/unit/test_data_codex.py -v

# Run with coverage
pytest tests/unit/test_data_codex.py --cov=src/data --cov-report=term-missing
```

### **Step 6: Fix Any Issues**

Similar to previous test generation:
- Fix implementation mismatches
- Adjust test assertions to match actual behavior
- Handle edge cases properly
- Use proper mocking for external dependencies

---

## 📊 **EXPECTED TEST STRUCTURE**

```python
# tests/unit/test_data_codex.py

import pytest
import pandas as pd
import numpy as np
from unittest.mock import Mock, patch, MagicMock, AsyncMock
import asyncio
from datetime import datetime, timedelta

from src.data.data_pipeline import DataPipeline
from src.data.data_validator import DataValidator, OHLCVDataPoint, TradingSymbol
from src.data.data_storage import DataStorage
from src.data.binance_client import BinanceClient, AsyncBinanceClient
from src.data.realtime_client import BinanceWebSocketClient

@pytest.fixture
def sample_ohlcv_data():
    """Sample OHLCV DataFrame for testing."""
    dates = pd.date_range('2024-01-01', periods=100, freq='1H')
    return pd.DataFrame({
        'timestamp': dates,
        'open': np.random.uniform(100, 200, 100),
        'high': np.random.uniform(200, 300, 100),
        'low': np.random.uniform(50, 100, 100),
        'close': np.random.uniform(100, 200, 100),
        'volume': np.random.uniform(1000, 10000, 100)
    })

class TestDataPipeline:
    """Test DataPipeline orchestration."""
    # Tests for sync và async methods
    # Cache testing
    # Error handling
    # Integration tests

class TestDataValidator:
    """Test data validation logic."""
    # Pydantic model tests
    # DataFrame validation
    # Edge cases (NaN, Inf, invalid OHLC)
    # Data cleaning tests

class TestDataStorage:
    """Test data storage và retrieval."""
    # Parquet storage tests
    # Partitioning tests
    # Query tests
    # Update modes (append/replace/merge)
    # File I/O error handling

class TestAsyncBinanceClient:
    """Test async Binance client."""
    # Async method tests với pytest-asyncio
    # Rate limiting tests
    # Retry logic tests
    # Error handling (429, 401, timeout)
    # Connection pooling tests

class TestBinanceClient:
    """Test sync Binance client wrapper."""
    # Sync wrapper tests
    # Context manager tests

class TestBinanceWebSocketClient:
    """Test real-time WebSocket client."""
    # Connection tests
    # Stream subscription tests
    # Message handling tests
    # Reconnection tests
    # Callback tests
    # Buffer management tests

class TestRequestSigner:
    """Test request signing và encryption."""
    # HMAC signing tests
    # RSA encryption tests
    # Header creation tests

class TestAdvancedRateLimiter:
    """Test rate limiting và circuit breaker."""
    # Rate limit acquisition tests
    # Circuit breaker tests
    # Exponential backoff tests
```

---

## ✅ **SUCCESS CRITERIA**

- ✅ All public methods tested
- ✅ Edge cases covered (empty data, errors, invalid inputs)
- ✅ Error handling tested (network errors, API errors, file I/O errors)
- ✅ Integration tests included (pipeline workflow)
- ✅ Performance tests included (large datasets, concurrent requests)
- ✅ Async tests use `pytest-asyncio` properly
- ✅ Mocking used for external dependencies (Binance API, file system)
- ✅ All tests pass locally
- ✅ Coverage >90% for data module

---

## 🔧 **COMMON FIXES NEEDED**

Based on previous test generation experience:

1. **Async Method Testing:**
   ```python
   @pytest.mark.asyncio
   async def test_async_method():
       # Use AsyncMock for async dependencies
       with patch('src.data.binance_client.AsyncBinanceClient.get_historical_klines', new_callable=AsyncMock):
           # Test code
   ```

2. **Mocking External APIs:**
   ```python
   @patch('src.data.binance_client.aiohttp.ClientSession')
   async def test_api_call(mock_session):
       # Mock response
       mock_response = AsyncMock()
       mock_response.status = 200
       mock_response.json = AsyncMock(return_value={'data': []})
       mock_session.get.return_value.__aenter__.return_value = mock_response
   ```

3. **File System Mocking:**
   ```python
   @patch('src.data.data_storage.Path.mkdir')
   @patch('src.data.data_storage.pq.write_table')
   def test_storage(mock_write, mock_mkdir):
       # Test storage logic
   ```

4. **Pydantic Validation:**
   ```python
   def test_pydantic_validation():
       with pytest.raises(ValidationError):
           OHLCVDataPoint(timestamp=datetime.now(), open=-1, high=100, low=50, close=75, volume=1000)
   ```

---

## 📈 **NEXT STEPS AFTER GENERATION**

1. Run tests: `pytest tests/unit/test_data_codex.py -v`
2. Fix any failing tests
3. Run coverage: `pytest tests/unit/test_data_codex.py --cov=src/data`
4. Commit tests: `git add tests/unit/test_data_codex.py && git commit -m "Add comprehensive tests for data module"`
5. Update coverage report
6. Move to next module: `src/exchanges/`

---

**Ready to generate tests with Codex Web!** 🚀

