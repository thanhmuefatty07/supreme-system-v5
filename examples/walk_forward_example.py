"""
Example: Walk-Forward Validation for Time Series.

Demonstrates:
1. Basic walk-forward validation
2. Expanding vs sliding window
3. Gap parameter usage
4. Comparison with K-fold (incorrect for time series)
5. Visualization of splits
"""

import numpy as np
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

try:
    from sklearn.linear_model import LinearRegression
    from sklearn.metrics import r2_score, mean_squared_error
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    print("Warning: scikit-learn not available, some examples will be skipped")

from src.data.validation import (
    WalkForwardValidator,
    plot_walk_forward_splits,
    compare_cv_methods
)


def example_basic_usage():
    """Example 1: Basic walk-forward validation"""
    print("\n" + "="*80)
    print("EXAMPLE 1: Basic Walk-Forward Validation")
    print("="*80)
    
    # Create time series data (trend + noise)
    np.random.seed(42)
    n_samples = 100
    X = np.arange(n_samples).reshape(-1, 1)
    y = 2 * X.ravel() + 10 + np.random.randn(n_samples) * 5
    
    print(f"\nDataset: {n_samples} samples")
    print(f"Features: {X.shape[1]}")
    
    # Walk-forward validation
    validator = WalkForwardValidator(n_splits=5)
    
    print(f"\nWalk-Forward Validation ({validator.n_splits} splits):")
    print("-" * 60)
    
    if SKLEARN_AVAILABLE:
        for fold, (train_idx, test_idx) in enumerate(validator.split(X)):
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]
            
            # Train model
            model = LinearRegression()
            model.fit(X_train, y_train)
            
            # Evaluate
            score = model.score(X_test, y_test)
            
            print(f"Fold {fold+1}: train=[{train_idx[0]:3d}:{train_idx[-1]:3d}], "
                  f"test=[{test_idx[0]:3d}:{test_idx[-1]:3d}], "
                  f"R²={score:.3f}")
    else:
        for fold, (train_idx, test_idx) in enumerate(validator.split(X)):
            print(f"Fold {fold+1}: train=[{train_idx[0]:3d}:{train_idx[-1]:3d}], "
                  f"test=[{test_idx[0]:3d}:{test_idx[-1]:3d}]")
    
    print("\n✅ Training data always precedes test data (no look-ahead bias)")


def example_expanding_vs_sliding():
    """Example 2: Expanding vs Sliding Window"""
    print("\n" + "="*80)
    print("EXAMPLE 2: Expanding vs Sliding Window")
    print("="*80)
    
    np.random.seed(42)
    X = np.arange(100).reshape(-1, 1)
    y = 2 * X.ravel() + np.random.randn(100) * 5
    
    # Expanding window (default)
    print("\n📈 EXPANDING WINDOW (training set grows):")
    print("-" * 60)
    expanding = WalkForwardValidator(n_splits=5, expanding_window=True)
    
    for fold, (train_idx, test_idx) in enumerate(expanding.split(X)):
        print(f"Fold {fold+1}: train_size={len(train_idx):3d}, "
              f"test_size={len(test_idx):2d}")
    
    # Sliding window
    print("\n📊 SLIDING WINDOW (training set size constant):")
    print("-" * 60)
    sliding = WalkForwardValidator(n_splits=5, expanding_window=False, test_size=10)
    
    for fold, (train_idx, test_idx) in enumerate(sliding.split(X)):
        print(f"Fold {fold+1}: train_size={len(train_idx):3d}, "
              f"test_size={len(test_idx):2d}")
    
    print("\n✅ Expanding: more training data each fold")
    print("✅ Sliding: consistent training window size")


def example_gap_parameter():
    """Example 3: Gap parameter (prevents label leakage)"""
    print("\n" + "="*80)
    print("EXAMPLE 3: Gap Parameter")
    print("="*80)
    
    print("\nWhy gaps are important:")
    print("In real trading, labels may be delayed (e.g., next-day close price)")
    print("Gap ensures we don't use information not available at prediction time")
    
    X = np.arange(50).reshape(-1, 1)
    
    print("\n🔴 WITHOUT GAP (may cause label leakage):")
    print("-" * 60)
    no_gap = WalkForwardValidator(n_splits=3, test_size=5, gap=0)
    
    for fold, (train_idx, test_idx) in enumerate(no_gap.split(X)):
        print(f"Fold {fold+1}: train_end={train_idx[-1]:2d}, "
              f"test_start={test_idx[0]:2d}, "
              f"gap={test_idx[0] - train_idx[-1] - 1}")
    
    print("\n🟢 WITH GAP=3 (safe):")
    print("-" * 60)
    with_gap = WalkForwardValidator(n_splits=3, test_size=5, gap=3)
    
    for fold, (train_idx, test_idx) in enumerate(with_gap.split(X)):
        print(f"Fold {fold+1}: train_end={train_idx[-1]:2d}, "
              f"test_start={test_idx[0]:2d}, "
              f"gap={test_idx[0] - train_idx[-1] - 1}")
    
    print("\n✅ Gap of 3 samples prevents label leakage")


def example_validate_method():
    """Example 4: Using validate() method"""
    if not SKLEARN_AVAILABLE:
        print("\n" + "="*80)
        print("EXAMPLE 4: Skipped (scikit-learn not available)")
        print("="*80)
        return
    
    print("\n" + "="*80)
    print("EXAMPLE 4: Automated Validation with validate()")
    print("="*80)
    
    # Generate data
    np.random.seed(42)
    n_samples = 200
    X = np.arange(n_samples).reshape(-1, 1)
    y = 2 * X.ravel() + 10 + np.random.randn(n_samples) * 10
    
    print(f"\nDataset: {n_samples} samples")
    
    # Validate with different models
    models = {
        'Linear Regression': LinearRegression(),
    }
    
    validator = WalkForwardValidator(n_splits=5)
    
    for name, model in models.items():
        # Using default scoring (R²)
        scores_r2 = validator.validate(model, X, y)
        
        # Using custom scoring (MSE)
        scores_mse = validator.validate(
            model, X, y,
            scoring=lambda y_true, y_pred: -mean_squared_error(y_true, y_pred)
        )
        
        print(f"\n{name}:")
        print(f"  R² scores: {[f'{s:.3f}' for s in scores_r2]}")
        print(f"  Mean R²: {np.mean(scores_r2):.3f} ± {np.std(scores_r2):.3f}")
        print(f"  Mean -MSE: {np.mean(scores_mse):.1f} ± {np.std(scores_mse):.1f}")
    
    print("\n✅ Easy automated validation with performance tracking")


def example_comparison_with_kfold():
    """Example 5: Why K-fold is wrong for time series"""
    if not SKLEARN_AVAILABLE:
        print("\n" + "="*80)
        print("EXAMPLE 5: Skipped (scikit-learn not available)")
        print("="*80)
        return
    
    print("\n" + "="*80)
    print("EXAMPLE 5: Walk-Forward vs K-Fold Comparison")
    print("="*80)
    
    print("\n⚠️  WARNING: K-Fold should NOT be used for time series!")
    print("This example shows WHY:\n")
    
    # Create trending data
    np.random.seed(42)
    n_samples = 100
    X = np.arange(n_samples).reshape(-1, 1)
    y = 0.5 * X.ravel() + np.random.randn(n_samples) * 2
    
    results = compare_cv_methods(X, y, LinearRegression())
    
    print("Walk-Forward (CORRECT for time series):")
    print(f"  Scores: {[f'{s:.3f}' for s in results['walk_forward']]}")
    print(f"  Mean: {results['walk_forward_mean']:.3f} ± {results['walk_forward_std']:.3f}")
    
    print("\nK-Fold (INCORRECT - contains look-ahead bias):")
    print(f"  Scores: {[f'{s:.3f}' for s in results['kfold']]}")
    print(f"  Mean: {results['kfold_mean']:.3f} ± {results['kfold_std']:.3f}")
    
    print(f"\nDifference: {results['difference']:.3f}")
    print("(K-Fold often shows better scores due to look-ahead bias)")
    
    print("\n❌ K-Fold randomly mixes past and future → unrealistic performance")
    print("✅ Walk-Forward respects time order → realistic performance")


def main():
    """Run all examples"""
    print("\n" + "="*80)
    print("WALK-FORWARD VALIDATION EXAMPLES")
    print("="*80)
    
    example_basic_usage()
    example_expanding_vs_sliding()
    example_gap_parameter()
    example_validate_method()
    example_comparison_with_kfold()
    
    print("\n" + "="*80)
    print("🎓 KEY TAKEAWAYS:")
    print("="*80)
    print("1. Walk-forward prevents look-ahead bias in time series")
    print("2. Always ensure training precedes test chronologically")
    print("3. Use gaps when labels are delayed")
    print("4. Choose expanding or sliding window based on use case")
    print("5. NEVER use K-fold for time series!")
    print("="*80)


if __name__ == "__main__":
    main()
