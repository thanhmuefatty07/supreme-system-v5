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

---

## Walk-Forward Validation

### Overview

Walk-forward validation is the CORRECT way to evaluate time series models. Unlike standard K-fold cross-validation, it respects temporal order and prevents look-ahead bias.

### The Problem with K-Fold

```python
# WRONG for time series! ❌
from sklearn.model_selection import KFold

# K-Fold randomly splits data
# Training may use future data → overoptimistic scores
# Example: Train on [future], Test on [past] ← WRONG!
```

### The Solution: Walk-Forward

```python
# CORRECT for time series! ✅
from src.data.validation import WalkForwardValidator

validator = WalkForwardValidator(n_splits=5)

# Always: past → train, immediate future → test
# Example: Train on [past], Test on [future] ← CORRECT!
```

### Usage

#### Basic Splitting

```python
from src.data.validation import WalkForwardValidator

validator = WalkForwardValidator(n_splits=5)

for train_idx, test_idx in validator.split(X):
    # Training data ALWAYS precedes test data
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]
    
    model.fit(X_train, y_train)
    score = model.score(X_test, y_test)
    print(f"Score: {score:.3f}")
```

#### Automated Validation

```python
from src.data.validation import WalkForwardValidator
from sklearn.linear_model import LinearRegression

validator = WalkForwardValidator(n_splits=5)
scores = validator.validate(LinearRegression(), X, y)

print(f"Mean R²: {np.mean(scores):.3f}")
print(f"Std: {np.std(scores):.3f}")
```

#### With Gap Parameter

```python
# Gap prevents label leakage when labels are delayed
# Example: Predicting next-day price, but price finalizes after 2 days

validator = WalkForwardValidator(
    n_splits=5,
    gap=2  # 2-sample gap between train and test
)

# Now there's a safe buffer between training and testing
```

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `n_splits` | int | 5 | Number of validation folds |
| `test_size` | int | None | Test set size (auto if None) |
| `gap` | int | 0 | Gap between train and test |
| `expanding_window` | bool | True | Expanding vs sliding window |
| `min_train_size` | int | None | Minimum training samples |

### Window Types

#### Expanding Window (Default)

Training set grows with each fold:

```
Fold 1: Train=[0:20]    Test=[21:25]
Fold 2: Train=[0:25]    Test=[26:30]  ← Train grew
Fold 3: Train=[0:30]    Test=[31:35]  ← Train grew
```

**Use when:** More recent data is still relevant

#### Sliding Window

Training set size stays constant:

```
Fold 1: Train=[0:20]    Test=[21:25]
Fold 2: Train=[5:25]    Test=[26:30]  ← Train slid
Fold 3: Train=[10:30]   Test=[31:35]  ← Train slid
```

**Use when:** Only recent data is relevant (concept drift)

### Best Practices

**✅ DO:**

- Use walk-forward for ALL time series validation
- Set gap > 0 if labels are delayed
- Choose window type based on data characteristics
- Validate on multiple splits (n_splits ≥ 5)

**❌ DON'T:**

- Use K-fold for time series (look-ahead bias!)
- Shuffle time series data
- Train on future, test on past
- Ignore label delays (causes leakage)

### Comparison with K-Fold

```python
from src.data.validation import compare_cv_methods

results = compare_cv_methods(X, y, model)

print("Walk-Forward (CORRECT):", results['walk_forward_mean'])
print("K-Fold (WRONG):", results['kfold_mean'])
print("Difference:", results['difference'])
# K-Fold usually shows inflated scores due to look-ahead bias
```

### Integration with Preprocessing

```python
from src.data.preprocessing import ZScoreNormalizer
from src.data.validation import WalkForwardValidator

class Pipeline:
    def fit(self, X, y):
        self.normalizer = ZScoreNormalizer()
        X_norm = self.normalizer.fit_transform(X)
        # Train model on normalized data
        return self
    
    def predict(self, X):
        X_norm = self.normalizer.transform(X)
        return self.model.predict(X_norm)

validator = WalkForwardValidator(n_splits=5)
scores = validator.validate(Pipeline(), X, y)
```

### Visualization

```python
from src.data.validation import plot_walk_forward_splits

validator = WalkForwardValidator(n_splits=5, gap=2)
plot_walk_forward_splits(validator, n_samples=100)
# Shows visual representation of train/test splits
```

### Performance Impact

- **Memory:** Negligible
- **Computation:** ~N×n_splits (same as K-fold)
- **Accuracy:** REALISTIC (no optimistic bias)

### Common Pitfalls

**Problem:** K-fold shows better scores than walk-forward

- **Why:** K-fold has look-ahead bias
- **Solution:** Trust walk-forward, ignore K-fold

**Problem:** Scores vary significantly between folds

- **Why:** Model performance changes over time (normal!)
- **Solution:** Report mean ± std, analyze fold-by-fold

**Problem:** Not enough data for validation

- **Why:** Too many splits or too large test_size
- **Solution:** Reduce n_splits or use smaller test_size

### References

- Bergmeir & Benítez (2012). "On the use of cross-validation for time series predictor evaluation"
- Tashman (2000). "Out-of-sample tests of forecasting accuracy: an analysis and review"
- Hyndman & Athanasopoulos (2018). "Forecasting: Principles and Practice"

