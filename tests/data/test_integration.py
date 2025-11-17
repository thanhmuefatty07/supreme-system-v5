"""Integration tests for preprocessing + validation pipeline"""

import pytest
import numpy as np
from src.data.preprocessing import ZScoreNormalizer
from src.data.validation import WalkForwardValidator


def test_preprocessing_with_validation():
    """Test full pipeline: normalize then validate"""
    # Generate data
    np.random.seed(42)
    X = np.random.randn(100, 5)
    y = X.sum(axis=1) + np.random.randn(100) * 0.1
    
    # Simple model
    class SimpleModel:
        def fit(self, X, y):
            self.normalizer = ZScoreNormalizer()
            X_norm = self.normalizer.fit_transform(X)
            self.coef_ = np.linalg.lstsq(X_norm, y, rcond=None)[0]
            return self
        
        def predict(self, X):
            X_norm = self.normalizer.transform(X)
            return X_norm @ self.coef_
        
        def score(self, X, y):
            y_pred = self.predict(X)
            return 1 - ((y - y_pred) ** 2).sum() / ((y - y.mean()) ** 2).sum()
    
    # Validate
    validator = WalkForwardValidator(n_splits=5)
    scores = validator.validate(SimpleModel(), X, y)
    
    # Should get reasonable scores
    assert len(scores) == 5
    assert all(s > 0.5 for s in scores), "Scores too low"
    assert np.mean(scores) > 0.7, "Mean score too low"


def test_variance_threshold_with_validation():
    """Test variance threshold + validation"""
    from src.data.preprocessing import VarianceThreshold
    
    np.random.seed(42)
    # Create features with varying variance
    X = np.column_stack([
        np.random.randn(100) * 10,  # High variance
        np.ones(100),                # Zero variance (constant)
        np.random.randn(100) * 5,   # Medium variance
        np.random.randn(100) * 0.01, # Low variance
    ])
    y = X[:, 0] + X[:, 2] + np.random.randn(100) * 0.5
    
    # Model with feature selection
    class ModelWithFeatureSelection:
        def fit(self, X, y):
            self.selector = VarianceThreshold(threshold=0.1)
            X_selected = self.selector.fit_transform(X)
            self.coef_ = np.linalg.lstsq(X_selected, y, rcond=None)[0]
            return self
        
        def predict(self, X):
            X_selected = self.selector.transform(X)
            return X_selected @ self.coef_
        
        def score(self, X, y):
            y_pred = self.predict(X)
            return 1 - ((y - y_pred) ** 2).sum() / ((y - y.mean()) ** 2).sum()
    
    # Validate
    validator = WalkForwardValidator(n_splits=5)
    scores = validator.validate(ModelWithFeatureSelection(), X, y)
    
    assert len(scores) == 5
    assert all(s > 0.5 for s in scores)


def test_full_pipeline():
    """Test complete pipeline: normalize + select + validate"""
    from src.data.preprocessing import ZScoreNormalizer, VarianceThreshold
    
    np.random.seed(42)
    X = np.random.randn(150, 10)
    # Add some constant features
    X[:, 3] = 1.0
    X[:, 7] = 2.0
    y = X[:, [0, 1, 2, 4]].sum(axis=1) + np.random.randn(150) * 0.5
    
    class FullPipeline:
        def fit(self, X, y):
            # Step 1: Normalize
            self.normalizer = ZScoreNormalizer()
            X_norm = self.normalizer.fit_transform(X)
            
            # Step 2: Select features
            self.selector = VarianceThreshold(threshold=0.01)
            X_selected = self.selector.fit_transform(X_norm)
            
            # Step 3: Train
            self.coef_ = np.linalg.lstsq(X_selected, y, rcond=None)[0]
            return self
        
        def predict(self, X):
            X_norm = self.normalizer.transform(X)
            X_selected = self.selector.transform(X_norm)
            return X_selected @ self.coef_
        
        def score(self, X, y):
            y_pred = self.predict(X)
            return 1 - ((y - y_pred) ** 2).sum() / ((y - y.mean()) ** 2).sum()
    
    # Validate full pipeline
    validator = WalkForwardValidator(n_splits=5)
    scores = validator.validate(FullPipeline(), X, y)
    
    assert len(scores) == 5
    assert all(s > 0.5 for s in scores), f"Poor scores: {scores}"
    print(f"Pipeline scores: {[f'{s:.3f}' for s in scores]}")
    print(f"Mean: {np.mean(scores):.3f} ± {np.std(scores):.3f}")

