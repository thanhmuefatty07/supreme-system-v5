"""
Time series validation utilities - Walk-Forward Testing.

Provides walk-forward cross-validation to prevent look-ahead bias in time series.
"""

import logging
from typing import Iterator, Tuple, Optional, Callable, Union
import numpy as np

logger = logging.getLogger(__name__)


class WalkForwardValidator:
    """
    Walk-Forward Cross-Validator for time series data.
    
    Ensures training data always precedes test data chronologically,
    preventing look-ahead bias that occurs in standard K-fold CV.
    
    Args:
        n_splits: Number of folds (default: 5)
        test_size: Size of test set per fold. If None, auto-calculated (default: None)
        gap: Gap between train and test sets (default: 0)
        expanding_window: If True, train size grows; if False, slides (default: True)
    
    Attributes:
        n_splits: Number of folds
        test_size: Size of test set per fold
        gap: Gap between train and test
        expanding_window: Whether to use expanding or sliding window
    
    Example:
        >>> from src.data.validation import WalkForwardValidator
        >>> import numpy as np
        >>>
        >>> # Create time series data
        >>> X = np.arange(100).reshape(-1, 1)
        >>> y = np.arange(100)
        >>>
        >>> # Create validator
        >>> validator = WalkForwardValidator(n_splits=5)
        >>>
        >>> # Get splits
        >>> for train_idx, test_idx in validator.split(X):
        ...     print(f"Train: {len(train_idx)}, Test: {len(test_idx)}")
        >>>
        >>> # Validate model
        >>> from sklearn.linear_model import LinearRegression
        >>> model = LinearRegression()
        >>> scores = validator.validate(model, X, y)
        >>> print(f"Mean score: {np.mean(scores)}")
    
    References:
        - Time series cross-validation best practices
        - scikit-learn: TimeSeriesSplit
    """
    
    def __init__(
        self,
        n_splits: int = 5,
        test_size: Optional[int] = None,
        gap: int = 0,
        expanding_window: bool = True
    ):
        """
        Initialize Walk-Forward Validator.
        
        Args:
            n_splits: Number of folds
            test_size: Size of test set per fold (None = auto)
            gap: Gap between train and test sets
            expanding_window: If True, train size grows; if False, slides
        """
        if n_splits < 2:
            raise ValueError(f"n_splits must be >= 2, got {n_splits}")
        if test_size is not None and test_size < 1:
            raise ValueError(f"test_size must be >= 1, got {test_size}")
        if gap < 0:
            raise ValueError(f"gap must be >= 0, got {gap}")
        
        self.n_splits = n_splits
        self.test_size = test_size
        self.gap = gap
        self.expanding_window = expanding_window
    
    def split(
        self,
        X: np.ndarray,
        y: Optional[np.ndarray] = None,
        groups: Optional[np.ndarray] = None
    ) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
        """
        Generate indices to split data into training and test sets.
        
        Args:
            X: Feature array (n_samples, n_features)
            y: Target array (n_samples,) - optional, not used
            groups: Group labels - optional, not used
        
        Yields:
            (train_indices, test_indices) tuples
        
        Raises:
            ValueError: If insufficient data for splits
        """
        n_samples = len(X)
        
        # Calculate test size if not provided
        if self.test_size is None:
            # Use approximately equal-sized test sets
            test_size = max(1, n_samples // (self.n_splits + 1))
        else:
            test_size = self.test_size
        
        # Check if we have enough data
        min_required = test_size * self.n_splits + self.gap * (self.n_splits - 1)
        if self.expanding_window:
            # Expanding window needs less data
            min_required = test_size * self.n_splits + self.gap * (self.n_splits - 1)
        else:
            # Sliding window needs more data
            min_required = test_size * self.n_splits + self.gap * (self.n_splits - 1) + test_size
        
        if n_samples < min_required:
            raise ValueError(
                f"Insufficient data: need at least {min_required} samples, "
                f"got {n_samples}"
            )
        
        # Generate splits
        for i in range(self.n_splits):
            if self.expanding_window:
                # Expanding window: train size grows
                train_end = n_samples - (self.n_splits - i) * test_size - self.gap * (self.n_splits - i - 1)
                if train_end <= 0:
                    train_end = max(1, n_samples // (self.n_splits + 1))
            else:
                # Sliding window: train size constant
                train_end = n_samples - (self.n_splits - i) * test_size - self.gap * (self.n_splits - i - 1)
                if i == 0:
                    # First split: use all available data up to test
                    train_end = n_samples - self.n_splits * test_size - self.gap * (self.n_splits - 1)
                else:
                    # Subsequent splits: maintain constant size
                    first_train_end = n_samples - self.n_splits * test_size - self.gap * (self.n_splits - 1)
                    train_end = first_train_end
            
            test_start = train_end + self.gap
            test_end = test_start + test_size
            
            # Ensure we don't exceed data bounds
            if test_end > n_samples:
                test_end = n_samples
                test_start = max(train_end + self.gap, test_end - test_size)
            
            # Generate indices
            train_indices = np.arange(train_end)
            test_indices = np.arange(test_start, test_end)
            
            # Skip if test set is empty
            if len(test_indices) == 0:
                continue
            
            yield train_indices, test_indices
    
    def validate(
        self,
        estimator: object,
        X: np.ndarray,
        y: np.ndarray,
        scoring: Optional[Callable[[np.ndarray, np.ndarray], float]] = None
    ) -> list:
        """
        Validate estimator using walk-forward cross-validation.
        
        Args:
            estimator: Model with fit() and predict() methods
            X: Feature array (n_samples, n_features)
            y: Target array (n_samples,)
            scoring: Custom scoring function (y_true, y_pred) -> score.
                If None, uses mean squared error (default: None)
        
        Returns:
            List of scores for each fold
        """
        if scoring is None:
            # Default: mean squared error (lower is better, so we negate)
            def scoring(y_true, y_pred):
                return -np.mean((y_true - y_pred) ** 2)
        
        scores = []
        
        for train_idx, test_idx in self.split(X, y):
            # Split data
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]
            
            # Fit model
            estimator.fit(X_train, y_train)
            
            # Predict
            y_pred = estimator.predict(X_test)
            
            # Score
            score = scoring(y_test, y_pred)
            scores.append(score)
            
            logger.debug(
                f"Fold {len(scores)}: train_size={len(train_idx)}, "
                f"test_size={len(test_idx)}, score={score:.4f}"
            )
        
        logger.info(
            f"Walk-forward validation complete: {len(scores)} folds, "
            f"mean_score={np.mean(scores):.4f}"
        )
        
        return scores
    
    def get_n_splits(
        self,
        X: Optional[np.ndarray] = None,
        y: Optional[np.ndarray] = None,
        groups: Optional[np.ndarray] = None
    ) -> int:
        """
        Returns the number of splitting iterations.
        
        Args:
            X: Feature array (optional)
            y: Target array (optional)
            groups: Group labels (optional)
        
        Returns:
            Number of splits
        """
        return self.n_splits
    
    def __repr__(self) -> str:
        """String representation"""
        return (
            f"WalkForwardValidator(n_splits={self.n_splits}, "
            f"test_size={self.test_size}, gap={self.gap}, "
            f"expanding_window={self.expanding_window})"
        )

