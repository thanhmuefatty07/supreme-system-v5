# Data Preprocessing Documentation

## Overview

The data preprocessing module provides utilities for feature normalization, standardization, and feature selection, essential for machine learning model training.

## Feature Selection: Variance Threshold

Variance Threshold removes features with variance below a specified threshold. This is useful for removing constant or near-constant features that provide no information to the model.

### What is Variance Threshold?

Variance Threshold is a simple feature selection method that removes features with variance below a threshold:

- **Formula**: `variance = Var(X) = E[(X - μ)²]`
- **Keep feature if**: `variance > threshold`
- **Remove feature if**: `variance ≤ threshold`

### Why Use It?

1. **Reduces Dimensionality**: Removes uninformative features
2. **Improves Model Performance**: Eliminates noise from constant features
3. **Faster Training**: Fewer features = faster computation
4. **Better Generalization**: Reduces overfitting risk

### When to Use

- ✅ **Use when**: Features have constant or near-constant values
- ✅ **Use when**: You want to reduce dimensionality before training
- ✅ **Use when**: Features have very low variance (< 0.01)
- ❌ **Don't use when**: All features have meaningful variance
- ❌ **Don't use when**: Feature variance is important for your model

### Usage

#### Basic Usage

```python
from src.data.preprocessing import VarianceThreshold
import numpy as np

# Create data with constant feature
X_train = np.array([
    [1, 10, 100],   # Feature 0: constant (variance=0)
    [1, 20, 200],   # Feature 1: varying
    [1, 30, 300]    # Feature 2: varying
])

# Fit and transform
selector = VarianceThreshold(threshold=0.0)
X_train_selected = selector.fit_transform(X_train)

# Result: constant feature removed
print(X_train_selected.shape)  # (3, 2) - only 2 features remain
```

#### Train/Test Split (CRITICAL!)

**Always fit on training data only!**

```python
# Training data
X_train = np.array([[1, 10, 100], [1, 20, 200], [1, 30, 300]])

# Test data
X_test = np.array([[1, 15, 150], [1, 25, 250]])

# Fit ONLY on training data
selector = VarianceThreshold(threshold=0.0)
selector.fit(X_train)

# Transform both using TRAINING feature mask
X_train_selected = selector.transform(X_train)
X_test_selected = selector.transform(X_test)  # Uses training mask!
```

**Why?** Using test statistics would cause **data leakage** - the model would see information about test data during training.

#### With Pandas DataFrames

```python
import pandas as pd
from src.data.preprocessing import VarianceThreshold

# Create DataFrame
df = pd.DataFrame({
    'constant': [1.0, 1.0, 1.0, 1.0],
    'low_var': [1.0, 1.1, 1.2, 1.3],
    'high_var': [10.0, 20.0, 30.0, 40.0]
})

# Apply variance threshold
selector = VarianceThreshold(threshold=0.01)
df_selected = selector.fit_transform(df)

# Returns DataFrame with selected columns only
print(df_selected.columns)  # ['low_var', 'high_var']
```

#### Get Selected Features

```python
# Get boolean mask
mask = selector.get_support()
print(mask)  # [False, True, True]

# Get feature indices
indices = selector.get_support(indices=True)
print(indices)  # [1, 2]
```

#### Inverse Transform

```python
# Restore removed features (with fill_value)
X_restored = selector.inverse_transform(X_selected, fill_value=0.0)
print(X_restored.shape)  # (3, 3) - original shape restored
```

### API Reference

#### VarianceThreshold

##### Parameters

- `threshold` (float, default=0.0): Features with variance below this threshold will be removed

##### Methods

- `fit(X)`: Compute variances and determine feature mask
- `transform(X)`: Remove low-variance features
- `fit_transform(X)`: Fit and transform in one step
- `get_support(indices=False)`: Get mask or indices of selected features
- `inverse_transform(X, fill_value=0.0)`: Restore removed features

##### Attributes

- `variances_`: Variance of each feature (computed during fit)
- `n_features_in_`: Number of features seen during fit
- `feature_names_in_`: Names of features (if DataFrame was used)

### Edge Cases

#### All Features Removed

If all features have variance below threshold:

```python
data = np.array([[1, 1], [1, 1], [1, 1]])  # All constant

selector = VarianceThreshold(threshold=0.0)
transformed = selector.fit_transform(data)

print(transformed.shape)  # (3, 0) - empty array
```

#### Single Feature

Works correctly with single feature:

```python
data = np.array([[1], [2], [3], [4]])

selector = VarianceThreshold(threshold=0.0)
transformed = selector.fit_transform(data)

print(transformed.shape)  # (4, 1)
```

### Integration with Other Preprocessing

Variance Threshold works well with other preprocessing steps:

```python
from src.data.preprocessing import VarianceThreshold, ZScoreNormalizer

# Step 1: Remove constant features
selector = VarianceThreshold(threshold=0.0)
X_selected = selector.fit_transform(X_train)

# Step 2: Normalize remaining features
normalizer = ZScoreNormalizer()
X_normalized = normalizer.fit_transform(X_selected)
```

### Examples

See `examples/variance_threshold_example.py` for complete examples.

### References

- scikit-learn: VarianceThreshold documentation
- Feature selection best practices

---

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

