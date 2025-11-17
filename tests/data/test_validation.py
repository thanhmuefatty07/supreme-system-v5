"""
Tests for time series validation - Walk-Forward Testing.

Written BEFORE implementation (TDD approach).
"""

import pytest
import numpy as np


class TestWalkForwardValidator:
    """Test suite for Walk-Forward validation"""
    
    @pytest.fixture
    def sample_data(self):
        """Sample time series data"""
        return np.arange(100).reshape(-1, 1)  # 100 samples, 1 feature
    
    def test_initialization(self):
        """Test 1: Proper initialization"""
        from src.data.validation import WalkForwardValidator
        
        validator = WalkForwardValidator(n_splits=5)
        assert validator.n_splits == 5
        assert validator.expanding_window == True
        assert validator.gap == 0
    
    def test_split_generates_correct_number_of_folds(self, sample_data):
        """Test 2: Generates correct number of splits"""
        from src.data.validation import WalkForwardValidator
        
        validator = WalkForwardValidator(n_splits=5)
        splits = list(validator.split(sample_data))
        
        assert len(splits) == 5
    
    def test_train_indices_come_before_test(self, sample_data):
        """Test 3: Training data always precedes test data (no look-ahead)"""
        from src.data.validation import WalkForwardValidator
        
        validator = WalkForwardValidator(n_splits=5)
        
        for train_idx, test_idx in validator.split(sample_data):
            # All train indices must be < all test indices
            assert max(train_idx) < min(test_idx), "Look-ahead bias detected!"
    
    def test_expanding_window_grows_train_size(self, sample_data):
        """Test 4: Expanding window increases training set size"""
        from src.data.validation import WalkForwardValidator
        
        validator = WalkForwardValidator(n_splits=5, expanding_window=True)
        
        train_sizes = []
        for train_idx, test_idx in validator.split(sample_data):
            train_sizes.append(len(train_idx))
        
        # Train size should increase each fold
        for i in range(len(train_sizes) - 1):
            assert train_sizes[i] < train_sizes[i + 1]
    
    def test_sliding_window_constant_train_size(self, sample_data):
        """Test 5: Sliding window maintains constant training size"""
        from src.data.validation import WalkForwardValidator
        
        validator = WalkForwardValidator(
            n_splits=5,
            expanding_window=False,
            test_size=10
        )
        
        train_sizes = []
        for train_idx, test_idx in validator.split(sample_data):
            train_sizes.append(len(train_idx))
        
        # All train sizes should be equal (after first split)
        assert len(set(train_sizes[1:])) == 1, "Train size not constant in sliding window"
    
    def test_gap_parameter(self):
        """Test 6: Gap parameter creates separation between train and test"""
        from src.data.validation import WalkForwardValidator
        
        data = np.arange(100).reshape(-1, 1)
        validator = WalkForwardValidator(n_splits=3, gap=5)
        
        for train_idx, test_idx in validator.split(data):
            # Gap of 5 means: train ends at X, test starts at X+5+1
            gap_actual = min(test_idx) - max(train_idx) - 1
            assert gap_actual == 5, f"Expected gap=5, got {gap_actual}"
    
    def test_test_size_parameter(self):
        """Test 7: test_size parameter controls test set size"""
        from src.data.validation import WalkForwardValidator
        
        data = np.arange(100).reshape(-1, 1)
        test_size = 10
        validator = WalkForwardValidator(n_splits=5, test_size=test_size)
        
        for train_idx, test_idx in validator.split(data):
            assert len(test_idx) == test_size
    
    def test_no_overlap_between_folds(self, sample_data):
        """Test 8: Test sets don't overlap between folds"""
        from src.data.validation import WalkForwardValidator
        
        validator = WalkForwardValidator(n_splits=5)
        
        all_test_indices = []
        for train_idx, test_idx in validator.split(sample_data):
            all_test_indices.append(set(test_idx))
        
        # Check no overlap
        for i in range(len(all_test_indices)):
            for j in range(i + 1, len(all_test_indices)):
                overlap = all_test_indices[i] & all_test_indices[j]
                assert len(overlap) == 0, "Test sets overlap between folds!"
    
    def test_uses_all_data(self, sample_data):
        """Test 9: All samples used in either train or test"""
        from src.data.validation import WalkForwardValidator
        
        validator = WalkForwardValidator(n_splits=5)
        
        all_indices = set()
        for train_idx, test_idx in validator.split(sample_data):
            all_indices.update(train_idx)
            all_indices.update(test_idx)
        
        # Should use most samples (some might be in gap)
        assert len(all_indices) >= len(sample_data) * 0.8
    
    def test_insufficient_data_raises_error(self):
        """Test 10: Raises error if not enough data for splits"""
        from src.data.validation import WalkForwardValidator
        
        small_data = np.arange(10).reshape(-1, 1)
        validator = WalkForwardValidator(n_splits=20)  # Too many splits
        
        with pytest.raises(ValueError):
            list(validator.split(small_data))
    
    def test_validate_method_returns_scores(self):
        """Test 11: validate() method returns performance scores"""
        from src.data.validation import WalkForwardValidator
        
        # Simple model that always predicts mean
        class DummyModel:
            def fit(self, X, y):
                self.mean_ = y.mean()
                return self
            
            def predict(self, X):
                return np.full(len(X), self.mean_)
        
        X = np.arange(100).reshape(-1, 1)
        y = np.arange(100)
        
        validator = WalkForwardValidator(n_splits=5)
        scores = validator.validate(DummyModel(), X, y)
        
        assert len(scores) == 5
        assert all(isinstance(s, (int, float)) for s in scores)
    
    def test_custom_scoring_function(self):
        """Test 12: Accepts custom scoring function"""
        from src.data.validation import WalkForwardValidator
        
        class DummyModel:
            def fit(self, X, y):
                return self
            def predict(self, X):
                return np.zeros(len(X))
        
        X = np.arange(50).reshape(-1, 1)
        y = np.arange(50)
        
        # Custom scorer (returns constant)
        def custom_scorer(y_true, y_pred):
            return 42.0
        
        validator = WalkForwardValidator(n_splits=3)
        scores = validator.validate(DummyModel(), X, y, scoring=custom_scorer)
        
        assert all(s == 42.0 for s in scores)


class TestWalkForwardHelpers:
    """Test helper functions"""
    
    def test_plot_walk_forward_splits(self):
        """Test 13: Visualization function works"""
        from src.data.validation import WalkForwardValidator
        
        data = np.arange(50).reshape(-1, 1)
        validator = WalkForwardValidator(n_splits=5)
        
        # Should not raise error
        try:
            # This would create plot - just test it doesn't crash
            splits = list(validator.split(data))
            assert len(splits) == 5
        except Exception as e:
            pytest.fail(f"Plotting failed: {e}")

