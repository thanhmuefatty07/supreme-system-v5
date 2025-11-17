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

