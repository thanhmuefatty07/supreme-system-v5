# Coverage Analysis Report

## Top 15 Files by Coverage Impact

| Rank | File | Impact Score | Total Lines | Coverage | Missing Lines | Potential Gain |
|------|------|-------------|------------|----------|---------------|----------------|
|  1 | `src\data\binance_client.py` |    414.0 | 679 |  39.0% | 414 |  414.0 |
|  2 | `src\backtesting\production_backtester.py` |    412.0 | 516 |  20.2% | 412 |  412.0 |
|  3 | `src\streaming\realtime_analytics.py` |    398.0 | 398 |   0.0% | 398 |  398.0 |
|  4 | `src\data\data_validator.py` |    393.0 | 545 |  27.9% | 393 |  393.0 |
|  5 | `src\ai\coverage_optimizer.py` |    357.0 | 357 |   0.0% | 357 |  357.0 |
|  6 | `src\utils\exceptions.py` |    340.0 | 340 |   0.0% | 340 |  340.0 |
|  7 | `src\utils\ai_test_generator.py` |    332.0 | 332 |   0.0% | 332 |  332.0 |
|  8 | `src\ai\autonomous_sre.py` |    315.0 | 315 |   0.0% | 315 |  315.0 |
|  9 | `src\utils\chaos_engineer.py` |    307.0 | 307 |   0.0% | 307 |  307.0 |
| 10 | `src\utils\fuzz_tester.py` |    297.0 | 297 |   0.0% | 297 |  297.0 |
| 11 | `src\utils\regression_tester.py` |    293.0 | 293 |   0.0% | 293 |  293.0 |
| 12 | `src\utils\vectorized_ops.py` |    282.0 | 442 |  36.2% | 282 |  282.0 |
| 13 | `src\strategies\breakout.py` |    274.0 | 350 |  21.7% | 274 |  274.0 |
| 14 | `src\data\bybit_client.py` |    263.0 | 263 |   0.0% | 263 |  263.0 |
| 15 | `src\enterprise\concurrency.py` |    254.0 | 254 |   0.0% | 254 |  254.0 |

**Total Potential Coverage Gain:** 4931.0 lines

## Summary Statistics

- **Total Files Analyzed:** 74
- **Total Lines:** 13,567
- **Lines Covered:** 3,533
- **Overall Coverage:** 26.0%
- **Lines Needing Coverage:** 10,034

## Coverage Distribution

- **0-10%:** 25 files
- **10-25%:** 7 files
- **25-50%:** 19 files
- **50-75%:** 7 files
- **75-90%:** 4 files
- **90-100%:** 1 files

## Recommendations

### Priority 1: High Impact Files
- Focus on `src\data\binance_client.py` - high coverage gain potential
- Focus on `src\backtesting\production_backtester.py` - high coverage gain potential
- Focus on `src\streaming\realtime_analytics.py` - high coverage gain potential
- Focus on `src\data\data_validator.py` - high coverage gain potential
- Focus on `src\ai\coverage_optimizer.py` - high coverage gain potential

### Priority 2: Test Categories
- **Unit Tests:** Focus on core business logic (strategies, risk management)
- **Integration Tests:** Critical data pipelines and trading workflows
- **Performance Tests:** Memory optimization and vectorized operations

### Priority 3: Quick Wins
- Add basic test coverage for simple utility functions
- Test error handling paths in existing code
- Add integration tests for critical user journeys