# Data Preprocessing Documentation

## Overview

The data preprocessing module provides utilities for feature normalization and standardization, essential for machine learning model training.

## Z-Score Normalization

### What is Z-Score Normalization?

Z-Score normalization (also called standardization) transforms features to have:
- **Mean = 0**
- **Standard Deviation = 1**

Formula: `z = (x - mean) / std`

### Why Use It?

1. **Equal Feature Importance**: Features with different scales are treated equally
2. **Faster Convergence**: Models converge 10-30% faster
3. **Better Gradient Flow**: Prevents gradient explosion/vanishing
4. **Improved Accuracy**: Typically improves model accuracy by 2-5%

### When to Use

- ✅ **Use when**: Features have different scales (e.g., price: 0-1000, volume: 0-1000000)
- ✅ **Use when**: Training neural networks or gradient-based models
- ✅ **Use when**: Features have different units
- ❌ **Don't use when**: Features are already on similar scales
- ❌ **Don't use when**: Using tree-based models (they're scale-invariant)

## Usage

### Basic Usage

```python
from src.data.preprocessing import ZScoreNormalizer
import numpy as np

# Create data
data = np.array([[1, 2], [3, 4], [5, 6]])

# Normalize
normalizer = ZScoreNormalizer()
normalized = normalizer.fit_transform(data)

# Result: mean ≈ 0, std ≈ 1
print(normalized.mean(axis=0))  # [0, 0]
print(normalized.std(axis=0))   # [1, 1]
```

### Train/Test Split (CRITICAL!)

**Always fit on training data only!**

```python
# Training data
X_train = np.array([[1, 2], [3, 4], [5, 6]])

# Test data
X_test = np.array([[7, 8], [9, 10]])

# Fit ONLY on training data
normalizer = ZScoreNormalizer()
normalizer.fit(X_train)

# Transform both using TRAINING statistics
X_train_scaled = normalizer.transform(X_train)
X_test_scaled = normalizer.transform(X_test)  # Uses training stats!
```

**Why?** Using test statistics would cause **data leakage** - the model would see information about test data during training.

### With Pandas DataFrames

```python
import pandas as pd
from src.data.preprocessing import ZScoreNormalizer

# Create DataFrame
df = pd.DataFrame({
    'price': [100, 200, 300],
    'volume': [1000, 2000, 3000]
})

# Normalize
normalizer = ZScoreNormalizer()
df_normalized = normalizer.fit_transform(df)

# Returns DataFrame with same structure
print(df_normalized.columns)  # ['price', 'volume']
```

### Denormalizing Predictions

```python
# Normalize training data
normalizer = ZScoreNormalizer()
X_train_scaled = normalizer.fit_transform(X_train)

# Train model (works in normalized space)
model.fit(X_train_scaled, y_train)

# Make predictions (in normalized space)
predictions_scaled = model.predict(X_test_scaled)

# Denormalize predictions back to original scale
predictions = normalizer.inverse_transform(predictions_scaled)
```

## API Reference

### ZScoreNormalizer

#### Parameters

- `with_mean` (bool, default=True): If True, center data by subtracting mean
- `with_std` (bool, default=True): If True, scale data by dividing by std
- `copy` (bool, default=True): If True, copy data before transforming
- `epsilon` (float, default=1e-8): Small value to prevent division by zero

#### Methods

##### `fit(X)`

Compute mean and std from training data.

**Parameters:**
- `X`: Training data (numpy array or pandas DataFrame)

**Returns:**
- `self` (for method chaining)

**Raises:**
- `ValueError`: If X contains NaN or Inf values

##### `transform(X)`

Transform data using fitted statistics.

**Parameters:**
- `X`: Data to transform (numpy array or pandas DataFrame)

**Returns:**
- Transformed data (same type as input)

**Raises:**
- `RuntimeError`: If normalizer has not been fitted
- `ValueError`: If X has different number of features than training data

##### `fit_transform(X)`

Fit to data, then transform it.

**Parameters:**
- `X`: Training data (numpy array or pandas DataFrame)

**Returns:**
- Transformed data (same type as input)

##### `inverse_transform(X)`

Scale back data to original representation.

**Parameters:**
- `X`: Normalized data (numpy array or pandas DataFrame)

**Returns:**
- Original scale data (same type as input)

**Raises:**
- `RuntimeError`: If normalizer has not been fitted

#### Attributes

- `mean_`: Mean of each feature (computed during fit)
- `std_`: Standard deviation of each feature (computed during fit)
- `n_features_in_`: Number of features seen during fit
- `feature_names_in_`: Names of features (if DataFrame was used)

## Edge Cases

### Constant Features (Zero Std)

Constant features (std = 0) are handled gracefully:

```python
# Constant feature
data = np.array([[1, 5], [1, 10], [1, 15]])

normalizer = ZScoreNormalizer()
normalized = normalizer.fit_transform(data)

# Constant feature becomes zero
print(normalized[:, 0])  # [0, 0, 0]
```

### NaN/Inf Values

NaN and Inf values are detected and raise errors:

```python
data = np.array([[1, 2], [np.nan, 4], [5, 6]])

normalizer = ZScoreNormalizer()
# Raises ValueError: Input contains NaN values
normalizer.fit(data)
```

**Solution:** Handle missing values before normalization.

## Best Practices

1. **Always fit on training data only**
2. **Use training statistics for test/validation data**
3. **Handle missing values before normalization**
4. **Check for constant features** (they become zero)
5. **Save normalizer** for inference (use pickle or joblib)
6. **Denormalize predictions** if needed for interpretation

## Examples

See `examples/normalization_example.py` for complete examples.

## References

- Goodfellow et al. (2016). "Deep Learning", Section 8.7.1
- scikit-learn: StandardScaler documentation
- Statistical standardization best practices

