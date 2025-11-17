"""
Example: Variance Threshold feature selection.

Demonstrates:
1. Basic variance threshold selection
2. Train/test split handling (preventing data leakage)
3. DataFrame support
4. Inverse transformation
5. Integration with other preprocessing steps
"""

import numpy as np
import pandas as pd
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.data.preprocessing import VarianceThreshold, ZScoreNormalizer


def example_basic_usage():
    """Example 1: Basic variance threshold selection"""
    print("\n" + "="*80)
    print("EXAMPLE 1: Basic Variance Threshold Selection")
    print("="*80)
    
    # Create sample data with constant feature
    X_train = np.array([
        [1.0, 10.0, 100.0],   # Feature 0: constant (variance=0)
        [1.0, 20.0, 200.0],   # Feature 1: varying
        [1.0, 30.0, 300.0],   # Feature 2: varying
        [1.0, 40.0, 400.0]
    ])
    
    print("\nOriginal data:")
    print(X_train)
    print(f"\nShape: {X_train.shape}")
    print(f"Feature 0 variance: {np.var(X_train[:, 0]):.2f}")
    print(f"Feature 1 variance: {np.var(X_train[:, 1]):.2f}")
    print(f"Feature 2 variance: {np.var(X_train[:, 2]):.2f}")
    
    # Apply variance threshold
    selector = VarianceThreshold(threshold=0.0)
    X_selected = selector.fit_transform(X_train)
    
    print("\nSelected data (constant feature removed):")
    print(X_selected)
    print(f"\nShape: {X_selected.shape}")
    print(f"Selected features: {selector.get_support(indices=True)}")
    
    print("\n✅ Constant feature removed!")


def example_train_test_split():
    """Example 2: Proper train/test handling (NO DATA LEAKAGE)"""
    print("\n" + "="*80)
    print("EXAMPLE 2: Train/Test Split (Preventing Data Leakage)")
    print("="*80)
    
    # Training data
    X_train = np.array([
        [1, 10, 100],   # Feature 0: constant
        [1, 20, 200],   # Feature 1: varying
        [1, 30, 300]    # Feature 2: varying
    ], dtype=float)
    
    # Test data (different values)
    X_test = np.array([
        [2, 15, 150],   # Feature 0: still constant (but different value!)
        [2, 25, 250]    # Feature 1-2: varying
    ], dtype=float)
    
    print("\nTraining data:")
    print(X_train)
    print(f"Training variances: {np.var(X_train, axis=0)}")
    
    print("\nTest data:")
    print(X_test)
    print(f"Test variances: {np.var(X_test, axis=0)}")
    
    # Fit selector on TRAINING data only
    selector = VarianceThreshold(threshold=0.0)
    selector.fit(X_train)
    
    print(f"\nTraining feature mask: {selector.get_support()}")
    print(f"Selected feature indices: {selector.get_support(indices=True)}")
    
    # Transform both using TRAINING feature mask
    X_train_selected = selector.transform(X_train)
    X_test_selected = selector.transform(X_test)
    
    print("\nSelected training data:")
    print(X_train_selected)
    print(f"Shape: {X_train_selected.shape}")
    
    print("\nSelected test data (using TRAINING mask):")
    print(X_test_selected)
    print(f"Shape: {X_test_selected.shape}")
    
    print("\n✅ Test data transformed using TRAINING feature mask - NO DATA LEAKAGE!")


def example_dataframe_support():
    """Example 3: Working with pandas DataFrames"""
    print("\n" + "="*80)
    print("EXAMPLE 3: DataFrame Support")
    print("="*80)
    
    # Create DataFrame
    df = pd.DataFrame({
        'constant': [1.0, 1.0, 1.0, 1.0],
        'low_var': [1.0, 1.1, 1.2, 1.3],
        'high_var': [10.0, 20.0, 30.0, 40.0]
    })
    
    print("\nOriginal DataFrame:")
    print(df)
    print("\nVariances:")
    print(df.var())
    
    # Apply variance threshold
    selector = VarianceThreshold(threshold=0.01)
    df_selected = selector.fit_transform(df)
    
    print("\nSelected DataFrame:")
    print(df_selected)
    print(f"\nSelected columns: {list(df_selected.columns)}")
    
    print("\n✅ DataFrame columns preserved, low-variance features removed!")


def example_inverse_transform():
    """Example 4: Inverse transformation (restore removed features)"""
    print("\n" + "="*80)
    print("EXAMPLE 4: Inverse Transformation")
    print("="*80)
    
    # Original data
    X_original = np.array([
        [1, 10, 100],
        [1, 20, 200],
        [1, 30, 300]
    ], dtype=float)
    
    print("\nOriginal data:")
    print(X_original)
    
    # Apply variance threshold
    selector = VarianceThreshold(threshold=0.0)
    X_selected = selector.fit_transform(X_original)
    
    print("\nSelected data (constant feature removed):")
    print(X_selected)
    
    # Inverse transform (restore removed features)
    X_restored = selector.inverse_transform(X_selected, fill_value=0.0)
    
    print("\nRestored data (removed features filled with 0.0):")
    print(X_restored)
    
    print("\n✅ Original shape restored with fill_value for removed features!")


def example_integration_with_normalization():
    """Example 5: Integration with Z-Score Normalization"""
    print("\n" + "="*80)
    print("EXAMPLE 5: Integration with Z-Score Normalization")
    print("="*80)
    
    # Create data with constant and varying features
    X_train = np.array([
        [1, 10, 100],
        [1, 20, 200],
        [1, 30, 300],
        [1, 40, 400]
    ], dtype=float)
    
    X_test = np.array([
        [1, 15, 150],
        [1, 25, 250]
    ], dtype=float)
    
    print("\nStep 1: Remove constant features")
    selector = VarianceThreshold(threshold=0.0)
    X_train_selected = selector.fit_transform(X_train)
    X_test_selected = selector.transform(X_test)
    
    print(f"Training shape after selection: {X_train_selected.shape}")
    print(f"Test shape after selection: {X_test_selected.shape}")
    
    print("\nStep 2: Normalize remaining features")
    normalizer = ZScoreNormalizer()
    X_train_normalized = normalizer.fit_transform(X_train_selected)
    X_test_normalized = normalizer.transform(X_test_selected)
    
    print(f"Training shape after normalization: {X_train_normalized.shape}")
    print(f"Test shape after normalization: {X_test_normalized.shape}")
    
    print("\n✅ Feature selection + normalization pipeline complete!")


def main():
    """Run all examples"""
    print("\n" + "="*80)
    print("VARIANCE THRESHOLD FEATURE SELECTION EXAMPLES")
    print("="*80)
    
    example_basic_usage()
    example_train_test_split()
    example_dataframe_support()
    example_inverse_transform()
    example_integration_with_normalization()
    
    print("\n" + "="*80)
    print("🎓 KEY TAKEAWAYS:")
    print("="*80)
    print("1. Always fit selector on TRAINING data only")
    print("2. Use training feature mask to transform test data")
    print("3. Variance threshold removes constant/near-constant features")
    print("4. Works seamlessly with pandas DataFrames")
    print("5. Can be combined with other preprocessing steps")
    print("6. Inverse transform restores removed features (with fill_value)")
    print("="*80)


if __name__ == "__main__":
    main()

