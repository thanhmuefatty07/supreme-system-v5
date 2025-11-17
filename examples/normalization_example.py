"""
Example: Z-Score Normalization for Feature Preprocessing

Demonstrates how to use ZScoreNormalizer to standardize features
for better model training performance.
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import pandas as pd
from src.data.preprocessing import ZScoreNormalizer


def example_basic_normalization():
    """Example 1: Basic normalization with numpy arrays"""
    print("=" * 60)
    print("Example 1: Basic Normalization")
    print("=" * 60)
    
    # Create sample data with different scales
    data = np.array([
        [0.001, 1000],  # Feature 1: small scale, Feature 2: large scale
        [0.002, 2000],
        [0.003, 3000],
        [0.004, 4000],
        [0.005, 5000]
    ])
    
    print("\nOriginal data:")
    print(data)
    print(f"\nOriginal statistics:")
    print(f"  Feature 1 - Mean: {data[:, 0].mean():.6f}, Std: {data[:, 0].std():.6f}")
    print(f"  Feature 2 - Mean: {data[:, 1].mean():.6f}, Std: {data[:, 1].std():.6f}")
    
    # Normalize
    normalizer = ZScoreNormalizer()
    normalized = normalizer.fit_transform(data)
    
    print("\nNormalized data:")
    print(normalized)
    print(f"\nNormalized statistics:")
    print(f"  Feature 1 - Mean: {normalized[:, 0].mean():.6f}, Std: {normalized[:, 0].std():.6f}")
    print(f"  Feature 2 - Mean: {normalized[:, 1].mean():.6f}, Std: {normalized[:, 1].std():.6f}")
    
    # Verify: mean ≈ 0, std ≈ 1
    assert np.abs(normalized.mean(axis=0)).max() < 1e-6, "Mean should be ~0"
    assert np.abs(normalized.std(axis=0) - 1.0).max() < 1e-6, "Std should be ~1"
    print("\n✅ Normalization verified: Mean ≈ 0, Std ≈ 1")


def example_train_test_split():
    """Example 2: Proper train/test normalization (prevents data leakage)"""
    print("\n" + "=" * 60)
    print("Example 2: Train/Test Split (Prevents Data Leakage)")
    print("=" * 60)
    
    # Simulate train/test split
    train_data = np.array([
        [1, 10],
        [2, 20],
        [3, 30],
        [4, 40],
        [5, 50]
    ])
    
    test_data = np.array([
        [6, 60],
        [7, 70]
    ])
    
    print("\nTraining data:")
    print(train_data)
    print("\nTest data:")
    print(test_data)
    
    # CRITICAL: Fit ONLY on training data
    normalizer = ZScoreNormalizer()
    normalizer.fit(train_data)
    
    print(f"\nTraining statistics:")
    print(f"  Mean: {normalizer.mean_}")
    print(f"  Std: {normalizer.std_}")
    
    # Transform both using TRAINING statistics
    train_normalized = normalizer.transform(train_data)
    test_normalized = normalizer.transform(test_data)
    
    print("\nNormalized training data:")
    print(train_normalized)
    print(f"  Mean: {train_normalized.mean(axis=0)}")
    print(f"  Std: {train_normalized.std(axis=0)}")
    
    print("\nNormalized test data (using training stats):")
    print(test_normalized)
    print(f"  Mean: {test_normalized.mean(axis=0)}")
    print(f"  Std: {test_normalized.std(axis=0)}")
    
    # Note: Test data won't have mean=0, std=1 (correct!)
    print("\n✅ Test data correctly uses training statistics (prevents data leakage)")


def example_dataframe():
    """Example 3: Working with pandas DataFrames"""
    print("\n" + "=" * 60)
    print("Example 3: Pandas DataFrame Support")
    print("=" * 60)
    
    # Create DataFrame
    df = pd.DataFrame({
        'price': [100, 200, 300, 400, 500],
        'volume': [1000, 2000, 3000, 4000, 5000],
        'rsi': [30, 40, 50, 60, 70]
    })
    
    print("\nOriginal DataFrame:")
    print(df)
    print("\nOriginal statistics:")
    print(df.describe())
    
    # Normalize
    normalizer = ZScoreNormalizer()
    df_normalized = normalizer.fit_transform(df)
    
    print("\nNormalized DataFrame:")
    print(df_normalized)
    print("\nNormalized statistics:")
    print(df_normalized.describe())
    
    # Verify DataFrame structure preserved
    assert isinstance(df_normalized, pd.DataFrame), "Should return DataFrame"
    assert list(df_normalized.columns) == list(df.columns), "Columns should match"
    print("\n✅ DataFrame structure preserved")


def example_inverse_transform():
    """Example 4: Denormalizing predictions"""
    print("\n" + "=" * 60)
    print("Example 4: Inverse Transform (Denormalize Predictions)")
    print("=" * 60)
    
    # Original data
    original = np.array([[1, 2], [3, 4], [5, 6]])
    
    print("\nOriginal data:")
    print(original)
    
    # Normalize
    normalizer = ZScoreNormalizer()
    normalized = normalizer.fit_transform(original)
    
    print("\nNormalized data:")
    print(normalized)
    
    # Simulate model predictions (in normalized space)
    predictions_normalized = np.array([[0.5, 0.5], [1.0, 1.0]])
    
    print("\nModel predictions (normalized):")
    print(predictions_normalized)
    
    # Denormalize predictions back to original scale
    predictions_original = normalizer.inverse_transform(predictions_normalized)
    
    print("\nDenormalized predictions (original scale):")
    print(predictions_original)
    
    # Verify round-trip
    restored = normalizer.inverse_transform(normalized)
    assert np.allclose(restored, original), "Round-trip should restore original"
    print("\n✅ Inverse transform verified: Can restore original scale")


def example_edge_cases():
    """Example 5: Handling edge cases"""
    print("\n" + "=" * 60)
    print("Example 5: Edge Cases")
    print("=" * 60)
    
    # Constant feature (zero std)
    data_with_constant = np.array([
        [1, 5],
        [1, 10],
        [1, 15]
    ])
    
    print("\nData with constant feature:")
    print(data_with_constant)
    
    normalizer = ZScoreNormalizer()
    normalized = normalizer.fit_transform(data_with_constant)
    
    print("\nNormalized (constant feature handled):")
    print(normalized)
    print(f"  Constant feature (col 0): {normalized[:, 0]}")
    print(f"  Variable feature (col 1): Mean={normalized[:, 1].mean():.6f}, Std={normalized[:, 1].std():.6f}")
    
    print("\n✅ Constant features handled gracefully (zero std)")


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("Z-Score Normalization Examples")
    print("=" * 60)
    
    try:
        example_basic_normalization()
        example_train_test_split()
        example_dataframe()
        example_inverse_transform()
        example_edge_cases()
        
        print("\n" + "=" * 60)
        print("✅ All examples completed successfully!")
        print("=" * 60)
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()

