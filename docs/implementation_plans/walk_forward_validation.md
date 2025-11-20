# Walk-Forward Validation Implementation Plan

## Objective

Implement walk-forward (time series) cross-validation to prevent look-ahead bias.

## Problem Statement

Standard K-fold cross-validation randomly splits data, causing future information
to leak into training when applied to time series. This leads to overoptimistic
performance estimates.

## Solution

Walk-forward validation ensures training data always precedes test data 
chronologically, providing realistic performance estimates.

## Technical Specification

### Algorithm

```
for each split i in n_splits:
    train_end = (i+1) * (total_size / n_splits)
    test_start = train_end + gap
    test_end = test_start + test_size
    
    train_indices = [0, ..., train_end - 1]
    test_indices = [test_start, ..., test_end - 1]
    
    yield train_indices, test_indices
```

### Parameters

- `n_splits`: Number of folds (default: 5)
- `test_size`: Size of test set (default: auto)
- `gap`: Gap between train and test (default: 0)
- `expanding_window`: If True, train size grows; if False, slides (default: True)

## Expected Improvements

- Realistic performance estimation
- No look-ahead bias
- Proper time series validation
- Compatible with any model

## Files to Create

- `src/data/validation.py` (NEW)
- `tests/data/test_validation.py` (NEW)
- `examples/walk_forward_example.py` (NEW)

## Acceptance Criteria

- [ ] All 13 tests pass
- [ ] No look-ahead bias
- [ ] Works with expanding and sliding windows
- [ ] Gap parameter works correctly
- [ ] Compatible with any estimator

