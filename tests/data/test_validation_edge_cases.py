"""Edge case tests for walk-forward validation"""

import pytest
import numpy as np
from src.data.validation import WalkForwardValidator


def test_minimum_data_size():
    """Test with minimum viable data size"""
    X = np.arange(20).reshape(-1, 1)
    validator = WalkForwardValidator(n_splits=3, test_size=2)
    
    splits = list(validator.split(X))
    assert len(splits) == 3


def test_large_dataset():
    """Test with large dataset"""
    X = np.arange(10000).reshape(-1, 1)
    validator = WalkForwardValidator(n_splits=10)
    
    splits = list(validator.split(X))
    assert len(splits) == 10
    
    # Verify last test uses end of data
    _, last_test = splits[-1]
    assert max(last_test) == len(X) - 1


def test_single_feature():
    """Test with single feature"""
    X = np.arange(50).reshape(-1, 1)
    validator = WalkForwardValidator(n_splits=5)
    
    splits = list(validator.split(X))
    assert len(splits) == 5


def test_many_features():
    """Test with many features"""
    X = np.random.randn(100, 50)
    validator = WalkForwardValidator(n_splits=5)
    
    splits = list(validator.split(X))
    assert len(splits) == 5


def test_min_train_size_constraint():
    """Test minimum training size constraint"""
    X = np.arange(160).reshape(-1, 1)  # Enough data: 160 samples for 5 splits + min_train_size=30
    validator = WalkForwardValidator(
        n_splits=5,
        min_train_size=30
    )
    
    for train_idx, test_idx in validator.split(X):
        assert len(train_idx) >= 30


def test_very_large_gap():
    """Test with large gap"""
    X = np.arange(100).reshape(-1, 1)
    validator = WalkForwardValidator(n_splits=3, gap=10)
    
    for train_idx, test_idx in validator.split(X):
        gap_actual = min(test_idx) - max(train_idx) - 1
        assert gap_actual == 10

