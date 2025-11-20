"""
Tests for data preprocessing - Z-Score Normalization.

Written BEFORE implementation (TDD approach).
"""

import pytest
import numpy as np
import pandas as pd


class TestZScoreNormalizer:
    """Test suite for Z-Score normalization"""

    @pytest.fixture
    def simple_data(self):
        """Simple dataset for testing"""
        return np.array([[1, 2], [3, 4], [5, 6]], dtype=float)

    @pytest.fixture
    def dataframe_data(self):
        """DataFrame for testing"""
        return pd.DataFrame({
            'feature_1': [1.0, 2.0, 3.0, 4.0, 5.0],
            'feature_2': [10.0, 20.0, 30.0, 40.0, 50.0]
        })

    def test_normalizer_initialization(self):
        """Test 1: Proper initialization"""
        from src.data.preprocessing import ZScoreNormalizer

        normalizer = ZScoreNormalizer()
        assert normalizer.with_mean == True
        assert normalizer.with_std == True
        assert normalizer.mean_ is None
        assert normalizer.std_ is None

    def test_fit_calculates_statistics(self, simple_data):
        """Test 2: Fit calculates mean and std correctly"""
        from src.data.preprocessing import ZScoreNormalizer

        normalizer = ZScoreNormalizer()
        normalizer.fit(simple_data)

        # Check means
        expected_mean = np.array([3.0, 4.0])
        np.testing.assert_array_almost_equal(normalizer.mean_, expected_mean)

        # Check stds
        expected_std = np.array([np.std([1, 3, 5], ddof=0),
                                np.std([2, 4, 6], ddof=0)])
        np.testing.assert_array_almost_equal(normalizer.std_, expected_std)

    def test_transform_normalizes_data(self, simple_data):
        """Test 3: Transform normalizes data correctly"""
        from src.data.preprocessing import ZScoreNormalizer

        normalizer = ZScoreNormalizer()
        normalizer.fit(simple_data)
        normalized = normalizer.transform(simple_data)

        # Normalized data should have mean ≈ 0, std ≈ 1
        assert np.abs(normalized.mean(axis=0)).max() < 1e-6
        assert np.abs(normalized.std(axis=0) - 1.0).max() < 1e-6

    def test_fit_transform_combines_steps(self, simple_data):
        """Test 4: fit_transform works correctly"""
        from src.data.preprocessing import ZScoreNormalizer

        normalizer = ZScoreNormalizer()
        normalized = normalizer.fit_transform(simple_data)

        # Should be equivalent to fit then transform
        assert np.abs(normalized.mean(axis=0)).max() < 1e-6
        assert np.abs(normalized.std(axis=0) - 1.0).max() < 1e-6

    def test_inverse_transform_restores_data(self, simple_data):
        """Test 5: Inverse transform restores original data"""
        from src.data.preprocessing import ZScoreNormalizer

        normalizer = ZScoreNormalizer()
        normalized = normalizer.fit_transform(simple_data)
        restored = normalizer.inverse_transform(normalized)

        # Should match original data
        np.testing.assert_array_almost_equal(restored, simple_data)

    def test_handles_zero_std(self):
        """Test 6: Handles constant features (zero std)"""
        from src.data.preprocessing import ZScoreNormalizer

        # Constant feature (std = 0)
        data = np.array([[1, 5], [1, 10], [1, 15]], dtype=float)

        normalizer = ZScoreNormalizer()
        normalized = normalizer.fit_transform(data)

        # First feature should be zero (constant)
        assert np.all(normalized[:, 0] == 0)
        # Second feature should be normalized
        assert np.abs(normalized[:, 1].mean()) < 1e-6

    def test_handles_nan_values(self):
        """Test 7: Detects NaN values"""
        from src.data.preprocessing import ZScoreNormalizer

        data = np.array([[1, 2], [np.nan, 4], [5, 6]], dtype=float)

        normalizer = ZScoreNormalizer()

        # Should raise error or handle gracefully
        with pytest.raises((ValueError, RuntimeError)):
            normalizer.fit(data)

    def test_works_with_dataframe(self, dataframe_data):
        """Test 8: Works with pandas DataFrame"""
        from src.data.preprocessing import ZScoreNormalizer

        normalizer = ZScoreNormalizer()
        normalized = normalizer.fit_transform(dataframe_data)

        # Should return DataFrame with same columns
        assert isinstance(normalized, pd.DataFrame)
        assert list(normalized.columns) == list(dataframe_data.columns)

        # Check normalization
        assert np.abs(normalized.mean().values).max() < 1e-6

    def test_prevents_data_leakage(self):
        """Test 9: Uses training statistics for test data"""
        from src.data.preprocessing import ZScoreNormalizer

        train = np.array([[1, 2], [3, 4], [5, 6]], dtype=float)
        test = np.array([[7, 8], [9, 10]], dtype=float)

        normalizer = ZScoreNormalizer()
        normalizer.fit(train)

        # Normalize test data using TRAINING statistics
        test_normalized = normalizer.transform(test)

        # Test data should NOT have mean=0, std=1
        # (because we used training statistics)
        test_mean = test_normalized.mean(axis=0)
        assert np.abs(test_mean).max() > 0.1  # Not centered

    def test_with_mean_false(self, simple_data):
        """Test 10: with_mean=False only scales"""
        from src.data.preprocessing import ZScoreNormalizer

        normalizer = ZScoreNormalizer(with_mean=False)
        normalized = normalizer.fit_transform(simple_data)

        # Should NOT center (mean not subtracted)
        assert normalizer.mean_ is None
        # Should only scale
        assert normalizer.std_ is not None

    def test_with_std_false(self, simple_data):
        """Test 11: with_std=False only centers"""
        from src.data.preprocessing import ZScoreNormalizer

        normalizer = ZScoreNormalizer(with_std=False)
        normalized = normalizer.fit_transform(simple_data)

        # Should center
        assert normalizer.mean_ is not None
        # Should NOT scale
        assert normalizer.std_ is None

        # Check mean is zero
        assert np.abs(normalized.mean(axis=0)).max() < 1e-6


class TestZScoreHelpers:
    """Test helper functions"""

    def test_safe_divide(self):
        """Test 12: safe_divide handles edge cases"""
        from src.data.preprocessing import safe_divide

        # Normal division
        assert safe_divide(10, 2) == 5.0

        # Zero denominator
        result = safe_divide(10, 0)
        assert result == 0.0 or np.isnan(result)

        # Array division
        numerator = np.array([1, 2, 3])
        denominator = np.array([1, 0, 3])
        result = safe_divide(numerator, denominator)

        assert result[0] == 1.0
        assert result[1] == 0.0 or np.isnan(result[1])
        assert result[2] == 1.0


# Run: pytest tests/data/test_preprocessing.py -v

