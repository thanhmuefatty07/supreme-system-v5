"""
Data preprocessing utilities - Z-Score Normalization.

Provides standardization (Z-score normalization) for feature preprocessing.
"""

import logging
import numpy as np
import pandas as pd
from typing import Union, Optional

logger = logging.getLogger(__name__)


def safe_divide(numerator: Union[float, np.ndarray],
                denominator: Union[float, np.ndarray],
                epsilon: float = 1e-8) -> Union[float, np.ndarray]:
    """
    Safe division that handles zero denominators.
    
    Args:
        numerator: Numerator value(s)
        denominator: Denominator value(s)
        epsilon: Small value to add to denominator if zero
    
    Returns:
        Result of division, or 0/NaN if denominator is zero
    """
    if isinstance(numerator, np.ndarray) or isinstance(denominator, np.ndarray):
        # Handle array case
        denominator_safe = np.where(
            np.abs(denominator) < epsilon,
            epsilon,
            denominator
        )
        result = numerator / denominator_safe
        # Set result to 0 where denominator was zero
        result = np.where(np.abs(denominator) < epsilon, 0.0, result)
        return result
    else:
        # Handle scalar case
        if abs(denominator) < epsilon:
            return 0.0
        return numerator / denominator


class ZScoreNormalizer:
    """
    Z-Score (Standardization) Normalizer.
    
    Standardizes features by removing the mean and scaling to unit variance.
    Formula: z = (x - mean) / std
    
    This normalizer must be fit on training data only, then used to transform
    both training and test data using the training statistics (prevents data leakage).
    
    Args:
        with_mean: If True, center the data before scaling. Default: True
        with_std: If True, scale the data to unit variance. Default: True
        copy: If True, copy data before transforming. Default: True
        epsilon: Small value to add to std to prevent division by zero. Default: 1e-8
    
    Attributes:
        mean_: Mean of each feature (computed during fit)
        std_: Standard deviation of each feature (computed during fit)
        n_features_in_: Number of features seen during fit
        feature_names_in_: Names of features (if DataFrame was used)
    
    Example:
        >>> from src.data.preprocessing import ZScoreNormalizer
        >>> import numpy as np
        >>>
        >>> # Training data
        >>> X_train = np.array([[1, 2], [3, 4], [5, 6]])
        >>> normalizer = ZScoreNormalizer()
        >>> X_train_scaled = normalizer.fit_transform(X_train)
        >>>
        >>> # Test data (use training statistics)
        >>> X_test = np.array([[7, 8], [9, 10]])
        >>> X_test_scaled = normalizer.transform(X_test)
        >>>
        >>> # Denormalize predictions
        >>> predictions_scaled = model.predict(X_test_scaled)
        >>> predictions = normalizer.inverse_transform(predictions_scaled)
    """
    
    def __init__(self,
                 with_mean: bool = True,
                 with_std: bool = True,
                 copy: bool = True,
                 epsilon: float = 1e-8):
        """
        Initialize Z-Score Normalizer.
        
        Args:
            with_mean: Center data by subtracting mean
            with_std: Scale data by dividing by std
            copy: Copy data before transforming
            epsilon: Small value to prevent division by zero
        """
        self.with_mean = with_mean
        self.with_std = with_std
        self.copy = copy
        self.epsilon = epsilon
        
        # Statistics computed during fit
        self.mean_ = None
        self.std_ = None
        self.n_features_in_ = None
        self.feature_names_in_ = None
        self._is_fitted = False
    
    def fit(self, X: Union[np.ndarray, pd.DataFrame]) -> 'ZScoreNormalizer':
        """
        Compute mean and std to be used for later scaling.
        
        Args:
            X: Training data (n_samples, n_features)
        
        Returns:
            self (for method chaining)
        
        Raises:
            ValueError: If X contains NaN or Inf values
        """
        # Convert DataFrame to numpy if needed
        if isinstance(X, pd.DataFrame):
            X_array = X.values
            self.feature_names_in_ = list(X.columns)
        else:
            X_array = np.asarray(X)
            self.feature_names_in_ = None
        
        # Check for NaN or Inf
        if np.isnan(X_array).any():
            raise ValueError("Input contains NaN values. Handle missing values before normalization.")
        
        if np.isinf(X_array).any():
            raise ValueError("Input contains Inf values. Handle infinite values before normalization.")
        
        # Compute statistics
        if self.with_mean:
            self.mean_ = np.mean(X_array, axis=0)
        else:
            self.mean_ = None
        
        if self.with_std:
            self.std_ = np.std(X_array, axis=0, ddof=0)  # Population std (ddof=0)
            # Add epsilon to prevent division by zero
            self.std_ = np.where(self.std_ < self.epsilon, self.epsilon, self.std_)
        else:
            self.std_ = None
        
        self.n_features_in_ = X_array.shape[1] if len(X_array.shape) > 1 else 1
        self._is_fitted = True
        
        logger.info(f"Fitted ZScoreNormalizer on {X_array.shape[0]} samples, {self.n_features_in_} features")
        
        return self
    
    def transform(self, X: Union[np.ndarray, pd.DataFrame]) -> Union[np.ndarray, pd.DataFrame]:
        """
        Perform standardization by centering and scaling.
        
        Args:
            X: Data to transform (n_samples, n_features)
        
        Returns:
            Transformed data (same type as input)
        
        Raises:
            RuntimeError: If normalizer has not been fitted
            ValueError: If X has different number of features than training data
        """
        if not self._is_fitted:
            raise RuntimeError("Normalizer must be fitted before transform. Call fit() first.")
        
        # Convert DataFrame to numpy if needed
        is_dataframe = isinstance(X, pd.DataFrame)
        if is_dataframe:
            X_array = X.values.copy() if self.copy else X.values
            feature_names = list(X.columns)
        else:
            X_array = np.asarray(X, copy=self.copy)
            feature_names = None
        
        # Check feature count
        if len(X_array.shape) == 1:
            X_array = X_array.reshape(-1, 1)
        
        if X_array.shape[1] != self.n_features_in_:
            raise ValueError(
                f"X has {X_array.shape[1]} features, but normalizer was fitted on {self.n_features_in_} features."
            )
        
        # Apply transformation
        if self.with_mean and self.mean_ is not None:
            X_array = X_array - self.mean_
        
        if self.with_std and self.std_ is not None:
            X_array = safe_divide(X_array, self.std_, self.epsilon)
        
        # Convert back to DataFrame if input was DataFrame
        if is_dataframe:
            return pd.DataFrame(X_array, columns=feature_names, index=X.index)
        else:
            return X_array
    
    def fit_transform(self, X: Union[np.ndarray, pd.DataFrame]) -> Union[np.ndarray, pd.DataFrame]:
        """
        Fit to data, then transform it.
        
        Args:
            X: Training data (n_samples, n_features)
        
        Returns:
            Transformed data (same type as input)
        """
        return self.fit(X).transform(X)
    
    def inverse_transform(self, X: Union[np.ndarray, pd.DataFrame]) -> Union[np.ndarray, pd.DataFrame]:
        """
        Scale back the data to the original representation.
        
        Args:
            X: Normalized data (n_samples, n_features)
        
        Returns:
            Original scale data (same type as input)
        
        Raises:
            RuntimeError: If normalizer has not been fitted
        """
        if not self._is_fitted:
            raise RuntimeError("Normalizer must be fitted before inverse_transform. Call fit() first.")
        
        # Convert DataFrame to numpy if needed
        is_dataframe = isinstance(X, pd.DataFrame)
        if is_dataframe:
            X_array = X.values.copy() if self.copy else X.values
            feature_names = list(X.columns)
        else:
            X_array = np.asarray(X, copy=self.copy)
            feature_names = None
        
        # Check feature count
        if len(X_array.shape) == 1:
            X_array = X_array.reshape(-1, 1)
        
        if X_array.shape[1] != self.n_features_in_:
            raise ValueError(
                f"X has {X_array.shape[1]} features, but normalizer was fitted on {self.n_features_in_} features."
            )
        
        # Reverse transformation
        if self.with_std and self.std_ is not None:
            X_array = X_array * self.std_
        
        if self.with_mean and self.mean_ is not None:
            X_array = X_array + self.mean_
        
        # Convert back to DataFrame if input was DataFrame
        if is_dataframe:
            return pd.DataFrame(X_array, columns=feature_names, index=X.index)
        else:
            return X_array
    
    def get_params(self) -> dict:
        """
        Get normalizer parameters and statistics.
        
        Returns:
            Dictionary containing normalizer parameters and fitted statistics
        """
        return {
            'with_mean': self.with_mean,
            'with_std': self.with_std,
            'copy': self.copy,
            'epsilon': self.epsilon,
            'mean_': self.mean_,
            'std_': self.std_,
            'n_features_in_': self.n_features_in_,
            'feature_names_in_': self.feature_names_in_
        }
    
    def __repr__(self) -> str:
        """String representation of the normalizer"""
        fitted_status = "fitted" if self._is_fitted else "not fitted"
        return (
            f"ZScoreNormalizer(with_mean={self.with_mean}, "
            f"with_std={self.with_std}, {fitted_status})"
        )


def normalize_features(
    X_train: Union[np.ndarray, pd.DataFrame],
    X_test: Optional[Union[np.ndarray, pd.DataFrame]] = None,
    method: str = 'zscore'
) -> Union[tuple, Union[np.ndarray, pd.DataFrame]]:
    """
    Normalize features using specified method.
    
    Convenience function for quick normalization.
    
    Args:
        X_train: Training data
        X_test: Test data (optional)
        method: Normalization method ('zscore')
    
    Returns:
        If X_test is None: (X_train_normalized, normalizer)
        If X_test provided: (X_train_normalized, X_test_normalized, normalizer)
    
    Example:
        >>> X_train = np.array([[1, 2], [3, 4]])
        >>> X_test = np.array([[5, 6]])
        >>> X_train_norm, X_test_norm, normalizer = normalize_features(X_train, X_test)
    """
    if method == 'zscore':
        normalizer = ZScoreNormalizer()
    else:
        raise ValueError(f"Unknown normalization method: {method}")
    
    X_train_norm = normalizer.fit_transform(X_train)
    
    if X_test is not None:
        X_test_norm = normalizer.transform(X_test)
        return X_train_norm, X_test_norm, normalizer
    
    return X_train_norm, normalizer


class VarianceThreshold:
    """
    Variance Threshold feature selector.
    
    Removes features with variance below a threshold. This is useful for
    removing constant or near-constant features that provide no information.
    
    Formula:
        variance = Var(X) = E[(X - μ)²]
        
        Keep feature if: variance > threshold
        Remove feature if: variance ≤ threshold
    
    This selector must be fit on training data only, then used to transform
    both training and test data using the training feature mask (prevents data leakage).
    
    Args:
        threshold: Features with variance below this threshold will be removed.
            Default: 0.0 (removes only constant features)
    
    Attributes:
        variances_: Variance of each feature (computed during fit)
        n_features_in_: Number of features seen during fit
        feature_names_in_: Names of features (if DataFrame was used)
        _support_mask: Boolean mask of selected features
    
    Example:
        >>> from src.data.preprocessing import VarianceThreshold
        >>> import numpy as np
        >>>
        >>> # Training data with constant feature
        >>> X_train = np.array([[1, 2, 1], [1, 3, 2], [1, 4, 3]])
        >>> selector = VarianceThreshold(threshold=0.1)
        >>> X_train_selected = selector.fit_transform(X_train)
        >>>
        >>> # Test data (use training feature mask)
        >>> X_test = np.array([[1, 5, 4], [1, 6, 5]])
        >>> X_test_selected = selector.transform(X_test)
        >>>
        >>> # Get selected feature indices
        >>> selected_indices = selector.get_support(indices=True)
        >>> print(f"Selected features: {selected_indices}")
    
    References:
        - scikit-learn: VarianceThreshold
        - Feature selection best practices
    """
    
    def __init__(self, threshold: float = 0.0):
        """
        Initialize Variance Threshold selector.
        
        Args:
            threshold: Features with variance below this threshold will be removed.
                Default: 0.0 (removes only constant features)
        """
        if threshold < 0:
            raise ValueError(f"Threshold must be non-negative, got {threshold}")
        
        self.threshold = threshold
        
        # Statistics computed during fit
        self.variances_ = None
        self.n_features_in_ = None
        self.feature_names_in_ = None
        self._support_mask = None
        self._is_fitted = False
    
    def fit(self, X: Union[np.ndarray, pd.DataFrame]) -> 'VarianceThreshold':
        """
        Compute variances and determine feature mask.
        
        Args:
            X: Training data (n_samples, n_features)
        
        Returns:
            self (for method chaining)
        
        Raises:
            ValueError: If X contains NaN or Inf values
        """
        # Convert DataFrame to numpy if needed
        if isinstance(X, pd.DataFrame):
            X_array = X.values
            self.feature_names_in_ = list(X.columns)
        else:
            X_array = np.asarray(X, dtype=float)
            self.feature_names_in_ = None
        
        # Handle 1D input
        if X_array.ndim == 1:
            X_array = X_array.reshape(-1, 1)
        
        # Check for NaN or Inf
        if not np.isfinite(X_array).all():
            raise ValueError(
                "Input contains NaN or Inf values. "
                "Please handle missing values before feature selection."
            )
        
        # Store number of features
        self.n_features_in_ = X_array.shape[1]
        
        # Calculate variances
        self.variances_ = np.var(X_array, axis=0, ddof=0)  # Population variance
        
        # Create support mask (features to keep)
        self._support_mask = self.variances_ > self.threshold
        
        self._is_fitted = True
        
        logger.info(
            f"VarianceThreshold fitted on {X_array.shape[0]} samples, "
            f"{self.n_features_in_} features. "
            f"Selected {self._support_mask.sum()} features."
        )
        
        return self
    
    def transform(self, X: Union[np.ndarray, pd.DataFrame]) -> Union[np.ndarray, pd.DataFrame]:
        """
        Remove low-variance features using training feature mask.
        
        Args:
            X: Data to transform (n_samples, n_features)
        
        Returns:
            Transformed data with selected features only
        
        Raises:
            RuntimeError: If not fitted yet
            ValueError: If feature count doesn't match
        """
        if not self._is_fitted:
            raise RuntimeError(
                "VarianceThreshold not fitted. Call fit() first."
            )
        
        # Convert DataFrame to numpy if needed
        is_dataframe = isinstance(X, pd.DataFrame)
        if is_dataframe:
            feature_names = list(X.columns)
            X_array = X.values
        else:
            X_array = np.asarray(X, dtype=float)
            feature_names = None
        
        # Handle 1D input
        if X_array.ndim == 1:
            X_array = X_array.reshape(-1, 1)
        
        # Check feature count
        if X_array.shape[1] != self.n_features_in_:
            raise ValueError(
                f"X has {X_array.shape[1]} features, "
                f"but VarianceThreshold was fitted with {self.n_features_in_} features"
            )
        
        # Apply feature mask
        X_selected = X_array[:, self._support_mask]
        
        # Return DataFrame if input was DataFrame
        if is_dataframe:
            # Get selected feature names
            selected_names = [
                name for name, keep in zip(self.feature_names_in_, self._support_mask) if keep
            ]
            return pd.DataFrame(X_selected, columns=selected_names, index=X.index)
        
        return X_selected
    
    def fit_transform(self, X: Union[np.ndarray, pd.DataFrame]) -> Union[np.ndarray, pd.DataFrame]:
        """
        Fit to data, then transform it.
        
        Args:
            X: Training data (n_samples, n_features)
        
        Returns:
            Transformed data with selected features only
        """
        return self.fit(X).transform(X)
    
    def get_support(self, indices: bool = False) -> Union[np.ndarray, np.ndarray]:
        """
        Get a mask, or integer index, of the features selected.
        
        Args:
            indices: If True, return integer indices instead of boolean mask.
                Default: False
        
        Returns:
            Boolean mask or integer indices of selected features
        
        Raises:
            RuntimeError: If not fitted yet
        """
        if not self._is_fitted:
            raise RuntimeError(
                "VarianceThreshold not fitted. Call fit() first."
            )
        
        if indices:
            return np.where(self._support_mask)[0]
        else:
            return self._support_mask.copy()
    
    def inverse_transform(self, X: Union[np.ndarray, pd.DataFrame], fill_value: float = 0.0) -> Union[np.ndarray, pd.DataFrame]:
        """
        Reverse the transformation by restoring removed features.
        
        Args:
            X: Transformed data (n_samples, n_selected_features)
            fill_value: Value to use for removed features. Default: 0.0
        
        Returns:
            Data with original feature count (removed features filled)
        
        Raises:
            RuntimeError: If not fitted yet
            ValueError: If feature count doesn't match selected features
        """
        if not self._is_fitted:
            raise RuntimeError(
                "VarianceThreshold not fitted. Call fit() first."
            )
        
        # Convert DataFrame to numpy if needed
        is_dataframe = isinstance(X, pd.DataFrame)
        if is_dataframe:
            X_array = X.values
            feature_names = list(X.columns)
        else:
            X_array = np.asarray(X, dtype=float)
            feature_names = None
        
        # Handle 1D input
        if X_array.ndim == 1:
            X_array = X_array.reshape(-1, 1)
        
        # Check that we have the right number of selected features
        n_selected = self._support_mask.sum()
        if X_array.shape[1] != n_selected:
            raise ValueError(
                f"X has {X_array.shape[1]} features, "
                f"but {n_selected} features were selected during fit"
            )
        
        # Restore original shape
        X_restored = np.zeros((X_array.shape[0], self.n_features_in_), dtype=X_array.dtype)
        X_restored[:, self._support_mask] = X_array
        
        # Fill removed features
        removed_mask = ~self._support_mask
        if removed_mask.any():
            X_restored[:, removed_mask] = fill_value
        
        # Return DataFrame if input was DataFrame
        if is_dataframe:
            return pd.DataFrame(
                X_restored,
                columns=self.feature_names_in_,
                index=X.index
            )
        
        return X_restored
    
    def get_params(self) -> dict:
        """
        Get selector parameters and statistics.
        
        Returns:
            Dictionary containing selector parameters and fitted statistics
        """
        return {
            'threshold': self.threshold,
            'variances_': self.variances_,
            'n_features_in_': self.n_features_in_,
            'feature_names_in_': self.feature_names_in_,
            'n_features_selected_': self._support_mask.sum() if self._is_fitted else None
        }
    
    def __repr__(self) -> str:
        """String representation of the selector"""
        fitted_status = "fitted" if self._is_fitted else "not fitted"
        n_selected = self._support_mask.sum() if self._is_fitted else None
        return (
            f"VarianceThreshold(threshold={self.threshold}, "
            f"{fitted_status}, "
            f"n_selected={n_selected})"
        )

