"""
Example: Walk-Forward Validation for Time Series.

Demonstrates:
1. Basic walk-forward validation
2. Expanding vs sliding windows
3. Gap parameter usage
4. Model validation workflow
5. Custom scoring functions
"""

import numpy as np
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.data.validation import WalkForwardValidator


def example_basic_usage():
    """Example 1: Basic walk-forward validation"""
    print("\n" + "="*80)
    print("EXAMPLE 1: Basic Walk-Forward Validation")
    print("="*80)
    
    # Create time series data
    n_samples = 100
    X = np.arange(n_samples).reshape(-1, 1)
    y = np.arange(n_samples) + np.random.randn(n_samples) * 0.1
    
    print(f"\nData: {n_samples} samples")
    print(f"Features shape: {X.shape}")
    print(f"Target shape: {y.shape}")
    
    # Create validator
    validator = WalkForwardValidator(n_splits=5)
    
    print(f"\nWalk-forward splits: {validator.n_splits} folds")
    print("\nSplit details:")
    
    for i, (train_idx, test_idx) in enumerate(validator.split(X)):
        print(f"Fold {i+1}: Train=[0:{max(train_idx)+1}], Test=[{min(test_idx)}:{max(test_idx)+1}]")
        print(f"  Train size: {len(train_idx)}, Test size: {len(test_idx)}")
        print(f"  ✅ No look-ahead: max(train)={max(train_idx)} < min(test)={min(test_idx)}")
    
    print("\n✅ Walk-forward validation ensures no look-ahead bias!")


def example_expanding_vs_sliding():
    """Example 2: Expanding vs Sliding Windows"""
    print("\n" + "="*80)
    print("EXAMPLE 2: Expanding vs Sliding Windows")
    print("="*80)
    
    X = np.arange(100).reshape(-1, 1)
    
    # Expanding window
    print("\n--- Expanding Window (train size grows) ---")
    validator_expanding = WalkForwardValidator(n_splits=5, expanding_window=True)
    
    train_sizes_expanding = []
    for train_idx, test_idx in validator_expanding.split(X):
        train_sizes_expanding.append(len(train_idx))
        print(f"Fold {len(train_sizes_expanding)}: Train size = {len(train_idx)}")
    
    print(f"\nTrain sizes: {train_sizes_expanding}")
    print("✅ Train size increases each fold")
    
    # Sliding window
    print("\n--- Sliding Window (train size constant) ---")
    validator_sliding = WalkForwardValidator(
        n_splits=5,
        expanding_window=False,
        test_size=10
    )
    
    train_sizes_sliding = []
    for train_idx, test_idx in validator_sliding.split(X):
        train_sizes_sliding.append(len(train_idx))
        print(f"Fold {len(train_sizes_sliding)}: Train size = {len(train_idx)}")
    
    print(f"\nTrain sizes: {train_sizes_sliding}")
    print("✅ Train size remains constant (after first fold)")


def example_gap_parameter():
    """Example 3: Gap Parameter"""
    print("\n" + "="*80)
    print("EXAMPLE 3: Gap Parameter (Prevents Data Leakage)")
    print("="*80)
    
    X = np.arange(50).reshape(-1, 1)
    
    print("\n--- Without Gap ---")
    validator_no_gap = WalkForwardValidator(n_splits=3, test_size=5)
    
    for i, (train_idx, test_idx) in enumerate(validator_no_gap.split(X)):
        gap = min(test_idx) - max(train_idx) - 1
        print(f"Fold {i+1}: Gap = {gap}")
    
    print("\n--- With Gap=3 ---")
    validator_with_gap = WalkForwardValidator(n_splits=3, test_size=5, gap=3)
    
    for i, (train_idx, test_idx) in enumerate(validator_with_gap.split(X)):
        gap = min(test_idx) - max(train_idx) - 1
        print(f"Fold {i+1}: Gap = {gap}")
        print(f"  Train ends at: {max(train_idx)}, Test starts at: {min(test_idx)}")
    
    print("\n✅ Gap creates separation between train and test sets")


def example_model_validation():
    """Example 4: Model Validation Workflow"""
    print("\n" + "="*80)
    print("EXAMPLE 4: Model Validation Workflow")
    print("="*80)
    
    # Simple model that predicts mean
    class MeanPredictor:
        def fit(self, X, y):
            self.mean_ = y.mean()
            return self
        
        def predict(self, X):
            return np.full(len(X), self.mean_)
    
    # Create data
    X = np.arange(100).reshape(-1, 1)
    y = np.arange(100) + np.random.randn(100) * 5
    
    # Validate model
    validator = WalkForwardValidator(n_splits=5)
    scores = validator.validate(MeanPredictor(), X, y)
    
    print(f"\nValidation scores: {scores}")
    print(f"Mean score: {np.mean(scores):.4f}")
    print(f"Std score: {np.std(scores):.4f}")
    
    print("\n✅ Model validated using walk-forward cross-validation!")


def example_custom_scoring():
    """Example 5: Custom Scoring Function"""
    print("\n" + "="*80)
    print("EXAMPLE 5: Custom Scoring Function")
    print("="*80)
    
    class DummyModel:
        def fit(self, X, y):
            return self
        def predict(self, X):
            return np.zeros(len(X))
    
    X = np.arange(50).reshape(-1, 1)
    y = np.arange(50)
    
    # Custom scorer: Mean Absolute Error (MAE)
    def mae_scorer(y_true, y_pred):
        return -np.mean(np.abs(y_true - y_pred))  # Negative because lower is better
    
    validator = WalkForwardValidator(n_splits=3)
    scores = validator.validate(DummyModel(), X, y, scoring=mae_scorer)
    
    print(f"\nMAE scores: {scores}")
    print(f"Mean MAE: {np.mean(scores):.4f}")
    
    print("\n✅ Custom scoring function works correctly!")


def main():
    """Run all examples"""
    print("\n" + "="*80)
    print("WALK-FORWARD VALIDATION EXAMPLES")
    print("="*80)
    
    example_basic_usage()
    example_expanding_vs_sliding()
    example_gap_parameter()
    example_model_validation()
    example_custom_scoring()
    
    print("\n" + "="*80)
    print("🎓 KEY TAKEAWAYS:")
    print("="*80)
    print("1. Walk-forward validation prevents look-ahead bias")
    print("2. Training data always precedes test data chronologically")
    print("3. Expanding window: train size grows (uses more historical data)")
    print("4. Sliding window: train size constant (fixed lookback period)")
    print("5. Gap parameter: creates separation between train and test")
    print("6. Works with any model that has fit() and predict() methods")
    print("7. Supports custom scoring functions")
    print("="*80)


if __name__ == "__main__":
    main()

