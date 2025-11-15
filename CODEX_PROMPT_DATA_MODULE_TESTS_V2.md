# Codex Prompt: Data Module Test Generation
Supreme System V5 - Comprehensive Test Suite for Data Module

---

## 🎯 OBJECTIVE

Generate comprehensive test suite for Supreme System V5 data module to achieve 70%+ coverage.

---

## 📂 CONTEXT

### Project Structure:
```
src/data/
├── __init__.py
├── data_manager.py          # Main data orchestration
├── data_fetcher.py          # API data fetching
├── data_processor.py        # Data cleaning and transformation
├── data_validator.py        # Data quality checks
└── data_cache.py            # Caching layer

tests/unit/data/
├── __init__.py
├── test_data_manager.py     # TO BE GENERATED
├── test_data_fetcher.py     # TO BE GENERATED
├── test_data_processor.py   # TO BE GENERATED
├── test_data_validator.py   # TO BE GENERATED
└── test_data_cache.py       # TO BE GENERATED
```

### Existing Test Quality Standards (from risk module):
- **Coverage**: 80-95% for critical modules
- **Edge Cases**: NaN, Inf, zero, negative, empty, insufficient data
- **Test Types**: Unit, integration, property-based, performance
- **Documentation**: Comprehensive docstrings for every test
- **Fixtures**: Reusable fixtures for common test data

---

## 📝 DETAILED REQUIREMENTS

### Test File 1: test_data_manager.py

**Module to test:** `src/data/data_manager.py`

**Key classes/methods:**
- `DataManager.__init__()`
- `DataManager.fetch_data(symbols, interval, start_date, end_date)`
- `DataManager.get_cached_data(symbols, interval)`
- `DataManager.clear_cache()`
- Error handling for API failures
- Retry logic

**Test requirements:**
1. **Happy path tests:**
   - Fetch data successfully for single symbol
   - Fetch data for multiple symbols
   - Use cached data when available
   - Clear cache successfully

2. **Edge case tests:**
   - Empty symbols list
   - Invalid symbols (e.g., "INVALID_TICKER")
   - Invalid date ranges (start > end)
   - API returns empty data
   - API returns malformed data
   - Network timeout
   - Rate limit exceeded

3. **Integration tests:**
   - End-to-end: Fetch → Cache → Retrieve
   - Multiple concurrent requests
   - Cache expiration handling

4. **Performance tests:**
   - Benchmark fetch time for 1, 10, 100 symbols
   - Memory usage during large data fetch

**Fixtures needed:**
- `mock_api_response` - Valid API response data
- `mock_api_error` - API error responses
- `data_manager` - Initialized DataManager instance

**Expected coverage:** >80%

---

### Test File 2: test_data_fetcher.py

**Module to test:** `src/data/data_fetcher.py`

**Key classes/methods:**
- `DataFetcher.__init__(source='yahoo' / 'binance' / 'alpha_vantage')`
- `DataFetcher.fetch(symbol, interval, start, end)`
- `DataFetcher.validate_response(response)`
- Source-specific methods (Yahoo, Binance, Alpha Vantage)

**Test requirements:**
1. **Per-source tests:**
   - Yahoo Finance: Fetch OHLCV data
   - Binance: Fetch klines
   - Alpha Vantage: Fetch time series

2. **Error handling:**
   - Invalid API key
   - Rate limiting
   - Network errors
   - Parsing errors
   - Timeout handling

3. **Data validation:**
   - Response format validation
   - Required fields present
   - Data types correct
   - Timestamp ordering

**Mocking strategy:**
- Mock external API calls (don't hit real APIs in tests)
- Use `responses` library or `unittest.mock`
- Provide realistic mock data

**Expected coverage:** >85%

---

### Test File 3: test_data_processor.py

**Module to test:** `src/data/data_processor.py`

**Key methods:**
- `clean_data(df)` - Remove NaN, outliers
- `resample_data(df, interval)` - Change timeframe
- `calculate_returns(df)` - Price returns
- `normalize_data(df)` - Normalization
- `handle_missing_data(df, method='ffill'/'bfill'/'drop')`

**Test requirements:**
1. **Data cleaning:**
   - Remove NaN values
   - Handle outliers (>3 std dev)
   - Forward fill / backward fill
   - Drop incomplete rows

2. **Transformations:**
   - Resample: 1min → 5min, 1hour → 1day
   - Calculate returns: Simple, log returns
   - Normalize: Min-max, z-score

3. **Edge cases:**
   - Empty DataFrame
   - Single row DataFrame
   - All NaN DataFrame
   - Mixed data types
   - Timezone issues

**Property-based tests:**
- Use `hypothesis` library
- Generate random DataFrames with various shapes, NaN patterns
- Verify invariants (e.g., output shape, no NaN after cleaning)

**Expected coverage:** >90%

---

### Test File 4: test_data_validator.py

**Module to test:** `src/data/data_validator.py`

**Key methods:**
- `validate_schema(df, required_columns)` - Check columns present
- `validate_data_quality(df)` - Check for NaN, outliers, gaps
- `validate_timestamps(df)` - Check ordering, duplicates
- `validate_ohlc(df)` - Check OHLC relationships (high >= low, etc.)

**Test requirements:**
1. **Schema validation:**
   - Valid schema passes
   - Missing columns detected
   - Extra columns allowed
   - Wrong data types detected

2. **Quality checks:**
   - Detect NaN values
   - Detect outliers
   - Detect duplicate timestamps
   - Detect gaps in data

3. **OHLC validation:**
   - Valid: High >= Open, Close, Low
   - Invalid: High < Low (should fail)
   - Invalid: Negative prices
   - Invalid: Zero volume

**Expected coverage:** >90%

---

### Test File 5: test_data_cache.py

**Module to test:** `src/data/data_cache.py`

**Key methods:**
- `DataCache.get(key)` - Retrieve from cache
- `DataCache.set(key, value, ttl)` - Store in cache
- `DataCache.delete(key)` - Remove from cache
- `DataCache.clear()` - Clear all cache
- `DataCache.exists(key)` - Check if key exists

**Test requirements:**
1. **Basic operations:**
   - Set and get value
   - Delete value
   - Clear cache
   - Check existence

2. **TTL (Time To Live):**
   - Cache expires after TTL
   - Cache doesn't expire before TTL
   - Update TTL on re-set

3. **Edge cases:**
   - Get non-existent key (return None)
   - Set with zero/negative TTL
   - Very large values
   - Unicode keys/values

4. **Concurrency:**
   - Multiple threads reading/writing
   - Thread-safe operations

**Expected coverage:** >85%

---

## ⚙️ TEST PATTERNS (From Existing Tests)

### Pattern 1: Fixture-based Setup

```python
import pytest
import pandas as pd
from src.data.data_manager import DataManager

@pytest.fixture
def data_manager():
    """Fixture providing initialized DataManager."""
    return DataManager(source='yahoo')

@pytest.fixture
def sample_data():
    """Fixture providing sample OHLCV DataFrame."""
    return pd.DataFrame({
        'timestamp': pd.date_range('2024-01-01', periods=100, freq='1h'),
        'open': [100.0 + i for i in range(100)],
        'high': [101.0 + i for i in range(100)],
        'low': [99.0 + i for i in range(100)],
        'close': [100.5 + i for i in range(100)],
        'volume': [1000 + i*10 for i in range(100)]
    })
```

### Pattern 2: Edge Case Testing

```python
def test_data_processor_handles_nan():
    """Test data processor handles NaN values correctly."""
    # Setup: Create DataFrame with NaN
    df = pd.DataFrame({
        'close': [100.0, np.nan, 102.0, np.nan, 104.0]
    })
    
    # Execute: Clean data
    result = clean_data(df, method='ffill')
    
    # Verify: No NaN in result
    assert not result.isna().any().any(), "NaN values should be filled"
    assert result['close'].tolist() == [100.0, 100.0, 102.0, 102.0, 104.0]
```

### Pattern 3: Property-Based Testing

```python
from hypothesis import given, strategies as st
import hypothesis.extra.pandas as pdst

@given(pdst.data_frames(
    columns=[
        pdst.column('close', dtype=float),
        pdst.column('volume', dtype=int)
    ],
    rows=st.integers(min_value=10, max_value=1000)
))
def test_calculate_returns_properties(df):
    """Test that calculate_returns maintains invariants."""
    # Assume: Valid input data
    df = df[df['close'] > 0]  # Only positive prices
    
    # Execute
    returns = calculate_returns(df)
    
    # Verify invariants
    assert len(returns) == len(df) - 1, "Returns should be n-1 length"
    assert returns.isna().sum() == 0, "No NaN in returns"
    assert returns.abs().max() < 1.0, "No >100% single-period returns"
```

### Pattern 4: Performance Benchmarking

```python
import pytest

def test_data_fetch_performance(benchmark, data_manager):
    """Benchmark data fetching performance."""
    # Benchmark: Run fetch and measure time
    result = benchmark(
        data_manager.fetch_data,
        symbols=['AAPL'],
        interval='1h',
        days=30
    )
    
    # Verify: Result is valid
    assert len(result) > 0
    
    # Performance target: <1000ms for single symbol, 30 days
    assert benchmark.stats['mean'] < 1.0, "Fetch should complete in <1s"
```

---

## ✅ VALIDATION CHECKLIST

**After generating tests, verify:**

- [ ] All tests run and pass locally
- [ ] Coverage >70% for data module (run `pytest --cov=src/data`)
- [ ] No test failures or errors
- [ ] All edge cases covered (NaN, Inf, empty, invalid)
- [ ] Docstrings present for all tests
- [ ] Fixtures are reusable and well-documented
- [ ] No external API calls in unit tests (all mocked)
- [ ] Performance tests have realistic targets
- [ ] Property-based tests have meaningful invariants

---

## 📝 CODEX WEB PROMPT

### Copy this prompt to Codex Web:

```
Generate comprehensive pytest test suite for Supreme System V5 data module.

Context:
- Project: Production trading platform
- Module: src/data/ (data_manager, data_fetcher, data_processor, data_validator, data_cache)
- Existing standards: 80-95% coverage for critical modules, comprehensive edge case testing
- Testing stack: pytest, pytest-cov, pytest-mock, hypothesis, pytest-benchmark

Requirements:
1. Create 5 test files: test_data_manager.py, test_data_fetcher.py, test_data_processor.py, test_data_validator.py, test_data_cache.py
2. Each test file should have:
   - Fixtures for common test data and mocks
   - Happy path tests (basic functionality)
   - Edge case tests (NaN, Inf, zero, negative, empty, invalid inputs)
   - Error handling tests (network errors, API errors, validation errors)
   - Integration tests where appropriate
   - Performance benchmarks for critical paths
   - Property-based tests using Hypothesis for data transformations
3. Mock all external API calls (Yahoo Finance, Binance, Alpha Vantage)
4. Target: 70%+ coverage for src/data/ module
5. Follow patterns from existing tests in repository
6. All tests must have comprehensive docstrings
7. Use type hints throughout
8. Follow PEP 8 and project code style

Test patterns to follow:
- Fixture-based setup (see example above)
- Edge case testing with parametrize
- Property-based testing with Hypothesis
- Performance benchmarking with pytest-benchmark
- Clear test names: test_<method>_<scenario>_<expected_outcome>

Generate all 5 test files with comprehensive coverage.
```

---

## 🛠️ POST-GENERATION WORKFLOW

### Step 1: Review Generated Tests (30-60 min)
- Read through all test files
- Check for logical errors
- Verify mocking is correct
- Ensure edge cases are comprehensive

### Step 2: Fix Syntax/Import Errors (30-60 min)
- Run: `pytest tests/unit/data/ --collect-only`
- Fix any import errors
- Fix any syntax errors
- Ensure all dependencies installed

### Step 3: Run Tests and Fix Failures (1-3 hours)
- Run: `pytest tests/unit/data/ -v`
- Debug failing tests
- Fix test logic or source code as needed
- Re-run until all pass

### Step 4: Verify Coverage (15 min)
- Run: `pytest tests/unit/data/ --cov=src/data --cov-report=term-missing`
- Check: Coverage >70%
- If <70%: Identify uncovered lines and add tests

### Step 5: Code Review (30 min)
- Review test quality
- Check docstrings
- Verify fixtures are reusable
- Ensure tests are maintainable

### Step 6: Commit (5 min)
- Commit all test files
- Message: "Add comprehensive data module test suite (70%+ coverage)"
- Push to branch

---

## 📊 EXPECTED OUTCOMES

**Test Statistics:**
- Total tests: 80-120 tests (across 5 files)
- Pass rate: 100%
- Coverage: 70-85% for src/data/
- Execution time: <30 seconds

**Test Breakdown:**
- Unit tests: 60-80 (75%)
- Integration tests: 10-20 (15%)
- Property-based tests: 5-10 (7%)
- Performance tests: 5-10 (3%)

---

## 🔄 ITERATION PLAN

If first generation doesn't achieve 70% coverage:

**Iteration 1:** Focus on uncovered critical paths
- Identify uncovered lines from coverage report
- Generate additional tests for those lines
- Target: +10-15% coverage

**Iteration 2:** Add property-based tests
- Use Hypothesis for data transformations
- Verify invariants and edge cases
- Target: +5-10% coverage

**Iteration 3:** Add integration tests
- End-to-end workflows
- Multi-component interactions
- Target: +5% coverage

---

## 📞 SUPPORT

Questions during test generation?
Email: thanhmuefatty07@gmail.com
Subject: "Data Module Tests - Question"

---

**Created:** November 16, 2025  
**Version:** 2.0  
**Status:** Ready for Codex Web
