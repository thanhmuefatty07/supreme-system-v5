# Kelly Criterion Comparison Report

**Backtest Date:** 2025-11-22 01:29:59
**Initial Capital:** $10,000
**Total Trades:** 1000

## Performance Summary

| Method | Final Capital | Total Return | Max Drawdown | Sharpe Ratio | Win Rate |
|--------|---------------|--------------|--------------|--------------|----------|
| Adaptive Kelly | $10,869 | 8.7% | 0.3% | 0.20 | 55.1% |
| Static Kelly | $15,125 | 51.2% | 1.3% | 3.33 | 55.1% |
| Fixed Size (2%) | $10,869 | 8.7% | 0.3% | 0.20 | 55.1% |

## Detailed Metrics

### Adaptive Kelly
- **Trades Executed:** 1000
- **Total P&L:** $869
- **Final EWMA Win Rate:** 0.665
- **Final EWMA R/R Ratio:** 15.877
- **Circuit Breaker Active:** False

### Static Kelly
- **Trades Executed:** 1000
- **Total P&L:** $5,125

### Fixed Size (2%)
- **Trades Executed:** 1000
- **Total P&L:** $869

