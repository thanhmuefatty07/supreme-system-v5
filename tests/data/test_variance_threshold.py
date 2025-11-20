"""
Tests for Variance Threshold feature selection.
Written BEFORE implementation (TDD approach).
"""

import pytest
import numpy as np
import pandas as pd


class TestVarianceThreshold:
    """Test suite for Variance Threshold feature selection"""
    
    @pytest.fixture
    def simple_data(self):
        """Simple dataset with varying variance"""
        return np.array([
            [1, 2, 1],      # Feature 0: low variance
            [1, 3, 2],      # Feature 1: medium variance
            [1, 4, 3],      # Feature 2: higher variance
            [1, 5, 4]
        ], dtype=float)
    
    @pytest.fixture
    def constant_feature_data(self):
        """Data with constant feature"""
        return np.array([
            [1, 10, 100],   # Feature 0: constant (variance=0)
            [1, 20, 200],   # Feature 1: varying
            [1, 30, 300],   # Feature 2: varying
            [1, 40, 400]
        ], dtype=float)
    
    @pytest.fixture
    def dataframe_data(self):
        """DataFrame for testing"""
        return pd.DataFrame({
            'constant': [1.0, 1.0, 1.0, 1.0],
            'low_var': [1.0, 1.1, 1.2, 1.3],
            'high_var': [10.0, 20.0, 30.0, 40.0]
        })
    
    def test_variance_threshold_initialization(self):
        """Test 1: Proper initialization"""
        from src.data.preprocessing import VarianceThreshold
        
        selector = VarianceThreshold()
        assert selector.threshold == 0.0
        assert selector.variances_ is None
        assert selector.n_features_in_ is None
        assert selector._is_fitted is False
    
    def test_variance_threshold_custom_threshold(self):
        """Test 2: Custom threshold initialization"""
        from src.data.preprocessing import VarianceThreshold
        
        selector = VarianceThreshold(threshold=0.5)
        assert selector.threshold == 0.5
    
    def test_fit_calculates_variances(self, simple_data):
        """Test 3: Fit calculates variances correctly"""
        from src.data.preprocessing import VarianceThreshold
        
        selector = VarianceThreshold()
        selector.fit(simple_data)
        
        # Check variances are calculated
        assert selector.variances_ is not None
        assert len(selector.variances_) == 3
        
        # Feature 0 should have variance ≈ 0 (constant)
        assert selector.variances_[0] < 1e-10
        
        # Feature 1 should have variance > 0
        assert selector.variances_[1] > 0
        
        # Feature 2 should have variance > 0
        assert selector.variances_[2] > 0
    
    def test_transform_removes_low_variance_features(self, simple_data):
        """Test 4: Transform removes features below threshold"""
        from src.data.preprocessing import VarianceThreshold
        
        selector = VarianceThreshold(threshold=0.1)
        selector.fit(simple_data)
        transformed = selector.transform(simple_data)
        
        # Feature 0 (constant) should be removed
        assert transformed.shape[1] == 2  # Only 2 features remain
        assert transformed.shape[0] == simple_data.shape[0]  # Same samples
    
    def test_fit_transform_combines_steps(self, simple_data):
        """Test 5: fit_transform works correctly"""
        from src.data.preprocessing import VarianceThreshold
        
        selector = VarianceThreshold(threshold=0.1)
        transformed = selector.fit_transform(simple_data)
        
        # Should remove low variance features
        assert transformed.shape[1] == 2
        assert selector._is_fitted is True
    
    def test_handles_constant_features(self, constant_feature_data):
        """Test 6: Handles constant features (zero variance)"""
        from src.data.preprocessing import VarianceThreshold
        
        selector = VarianceThreshold(threshold=0.0)
        transformed = selector.fit_transform(constant_feature_data)
        
        # Constant feature should be removed
        assert transformed.shape[1] == 2  # Only 2 features remain
    
    def test_handles_all_features_removed(self):
        """Test 7: Handles case where all features are removed"""
        from src.data.preprocessing import VarianceThreshold
        
        # All features constant
        data = np.array([[1, 1], [1, 1], [1, 1]], dtype=float)
        
        selector = VarianceThreshold(threshold=0.0)
        transformed = selector.fit_transform(data)
        
        # Should return empty array with correct shape
        assert transformed.shape[0] == 3  # Same samples
        assert transformed.shape[1] == 0  # No features
    
    def test_handles_single_feature(self):
        """Test 8: Handles single feature input"""
        from src.data.preprocessing import VarianceThreshold
        
        data = np.array([[1], [2], [3], [4]], dtype=float)
        
        selector = VarianceThreshold(threshold=0.0)
        transformed = selector.fit_transform(data)
        
        # Should work with single feature
        assert transformed.shape[1] == 1
    
    def test_works_with_dataframe(self, dataframe_data):
        """Test 9: Works with pandas DataFrame"""
        from src.data.preprocessing import VarianceThreshold
        
        selector = VarianceThreshold(threshold=0.01)
        transformed = selector.fit_transform(dataframe_data)
        
        # Should return DataFrame
        assert isinstance(transformed, pd.DataFrame)
        
        # Constant feature should be removed
        assert 'constant' not in transformed.columns
        assert 'low_var' in transformed.columns or 'high_var' in transformed.columns
    
    def test_prevents_data_leakage(self):
        """Test 10: Uses training statistics for test data"""
        from src.data.preprocessing import VarianceThreshold
        
        train = np.array([
            [1, 10, 100],   # Feature 0: constant
            [1, 20, 200],   # Feature 1: varying
            [1, 30, 300]    # Feature 2: varying
        ], dtype=float)
        
        test = np.array([
            [2, 15, 150],   # Different values
            [2, 25, 250]
        ], dtype=float)
        
        selector = VarianceThreshold(threshold=0.0)
        selector.fit(train)
        
        # Transform test using TRAINING statistics
        test_transformed = selector.transform(test)
        
        # Should use same feature mask as training
        assert test_transformed.shape[1] == train.shape[1] - 1  # Constant removed
    
    def test_get_support_mask(self, simple_data):
        """Test 11: get_support returns correct mask"""
        from src.data.preprocessing import VarianceThreshold
        
        selector = VarianceThreshold(threshold=0.1)
        selector.fit(simple_data)
        
        support = selector.get_support()
        
        # Should be boolean array
        assert support.dtype == bool
        assert len(support) == 3
        
        # Feature 0 (constant) should be False
        assert support[0] == False
    
    def test_get_support_indices(self, simple_data):
        """Test 12: get_support with indices=True returns indices"""
        from src.data.preprocessing import VarianceThreshold
        
        selector = VarianceThreshold(threshold=0.1)
        selector.fit(simple_data)
        
        indices = selector.get_support(indices=True)
        
        # Should be array of indices
        assert isinstance(indices, np.ndarray)
        assert len(indices) == 2  # 2 features selected
        assert 0 not in indices  # Constant feature not selected
    
    def test_inverse_transform_restores_features(self, simple_data):
        """Test 13: inverse_transform restores removed features"""
        from src.data.preprocessing import VarianceThreshold
        
        selector = VarianceThreshold(threshold=0.1)
        transformed = selector.fit_transform(simple_data)
        
        # Inverse transform should restore original shape
        restored = selector.inverse_transform(transformed)
        
        assert restored.shape == simple_data.shape
    
    def test_inverse_transform_with_fill_value(self, simple_data):
        """Test 14: inverse_transform uses fill_value for removed features"""
        from src.data.preprocessing import VarianceThreshold
        
        selector = VarianceThreshold(threshold=0.1)
        transformed = selector.fit_transform(simple_data)
        
        # Inverse transform with custom fill value
        restored = selector.inverse_transform(transformed, fill_value=0.0)
        
        # Removed features should be filled with 0.0
        assert restored.shape == simple_data.shape
        assert np.all(restored[:, 0] == 0.0)  # Constant feature filled
    
    def test_handles_nan_values(self):
        """Test 15: Detects NaN values"""
        from src.data.preprocessing import VarianceThreshold
        
        data = np.array([[1, 2], [np.nan, 4], [3, 6]], dtype=float)
        
        selector = VarianceThreshold()
        
        # Should raise error or handle gracefully
        with pytest.raises((ValueError, RuntimeError)):
            selector.fit(data)

