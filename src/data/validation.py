"""
Time series validation utilities.

Provides walk-forward validation to prevent look-ahead bias in time series models.
"""

import logging
from typing import Generator, Tuple, Callable, Any, Optional, List
import numpy as np

logger = logging.getLogger(__name__)


class WalkForwardValidator:
    """
    Walk-Forward (Time Series) Cross-Validator.
    
    Implements walk-forward validation that respects temporal order of data.
    Unlike standard K-fold, ensures training data always precedes test data,
    preventing look-ahead bias.
    
    Args:
        n_splits (int): Number of validation folds. Default: 5
        test_size (int, optional): Size of each test set. If None, automatically
            calculated as total_size / (n_splits + 1)
        gap (int): Number of samples to exclude between train and test sets.
            Useful when labels are delayed. Default: 0
        expanding_window (bool): If True, training set grows with each fold.
            If False, uses sliding window with constant training size. Default: True
        min_train_size (int, optional): Minimum training samples. Default: None
    
    Attributes:
        n_splits (int): Number of splits
        test_size (int): Test set size
        gap (int): Gap between train and test
        expanding_window (bool): Window type
        min_train_size (int): Minimum training size
    
    Example:
        >>> # Basic usage
        >>> from src.data.validation import WalkForwardValidator
        >>> validator = WalkForwardValidator(n_splits=5)
        >>> 
        >>> X = np.arange(100).reshape(-1, 1)
        >>> y = np.arange(100)
        >>> 
        >>> for train_idx, test_idx in validator.split(X):
        >>>     X_train, X_test = X[train_idx], X[test_idx]
        >>>     y_train, y_test = y[train_idx], y[test_idx]
        >>>     # Train and evaluate model
        >>> 
        >>> # Automated validation
        >>> from sklearn.linear_model import LinearRegression
        >>> scores = validator.validate(LinearRegression(), X, y)
        >>> print(f"Mean score: {np.mean(scores):.3f}")
    
    References:
        - Bergmeir & Benítez (2012). "On the use of cross-validation for time series"
        - Tashman (2000). "Out-of-sample tests of forecasting accuracy"
    """
    
    def __init__(
        self,
        n_splits: int = 5,
        test_size: Optional[int] = None,
        gap: int = 0,
        expanding_window: bool = True,
        min_train_size: Optional[int] = None
    ):
        if n_splits < 2:
            raise ValueError(f"n_splits must be >= 2, got {n_splits}")
        if gap < 0:
            raise ValueError(f"gap must be >= 0, got {gap}")
        
        self.n_splits = n_splits
        self.test_size = test_size
        self.gap = gap
        self.expanding_window = expanding_window
        self.min_train_size = min_train_size
    
    def split(
        self,
        X: np.ndarray,
        y: Optional[np.ndarray] = None,
        groups: Optional[np.ndarray] = None
    ) -> Generator[Tuple[np.ndarray, np.ndarray], None, None]:
        """
        Generate train/test indices for walk-forward validation.
        
        Args:
            X: Features array (n_samples, n_features)
            y: Target array (optional)
            groups: Group labels (optional, not used)
        
        Yields:
            Tuple of (train_indices, test_indices) for each fold
        
        Raises:
            ValueError: If insufficient data for n_splits
        """
        n_samples = len(X)
        
        # Calculate test_size if not provided
        if self.test_size is None:
            test_size = n_samples // (self.n_splits + 1)
        else:
            test_size = self.test_size
        
        # Validate sufficient data
        min_required = self.n_splits * test_size + self.gap
        if self.min_train_size:
            min_required += self.min_train_size
        
        if n_samples < min_required:
            raise ValueError(
                f"Insufficient data: need at least {min_required} samples for "
                f"{self.n_splits} splits with test_size={test_size}, gap={self.gap}, "
                f"but got {n_samples} samples"
            )
        
        # Calculate split points
        indices = np.arange(n_samples)
        
        # For sliding window, calculate first train size
        first_train_size = None
        if not self.expanding_window:
            # Calculate what the first train size should be
            first_test_start = n_samples - self.n_splits * test_size - self.gap * (self.n_splits - 1)
            first_train_size = first_test_start - self.gap
        
        for i in range(self.n_splits):
            # Calculate test set boundaries (from end backwards)
            test_end = n_samples - (self.n_splits - i - 1) * test_size
            test_start = test_end - test_size
            
            # Calculate train set boundaries
            train_end = test_start - self.gap - 1
            
            if self.expanding_window:
                # Expanding window: train from start
                train_start = 0
            else:
                # Sliding window: maintain constant train size
                if first_train_size is None:
                    first_train_size = train_end + 1
                train_start = max(0, train_end - first_train_size + 1)
            
            # Apply minimum train size constraint
            if self.min_train_size:
                if train_end - train_start + 1 < self.min_train_size:
                    train_start = max(0, train_end - self.min_train_size + 1)
            
            # Create index arrays
            train_indices = indices[train_start:train_end + 1]
            test_indices = indices[test_start:test_end]
            
            # Validate no overlap and correct order
            if len(train_indices) == 0:
                raise ValueError(f"Empty training set in fold {i+1}")
            if len(test_indices) == 0:
                raise ValueError(f"Empty test set in fold {i+1}")
            
            # Ensure temporal order (no look-ahead)
            assert max(train_indices) < min(test_indices), \
                f"Look-ahead bias detected in fold {i+1}!"
            
            yield train_indices, test_indices
    
    def get_n_splits(
        self,
        X: Optional[np.ndarray] = None,
        y: Optional[np.ndarray] = None,
        groups: Optional[np.ndarray] = None
    ) -> int:
        """Return the number of splitting iterations"""
        return self.n_splits
    
    def validate(
        self,
        estimator: Any,
        X: np.ndarray,
        y: np.ndarray,
        scoring: Optional[Callable] = None
    ) -> List[float]:
        """
        Perform walk-forward validation and return scores.
        
        Args:
            estimator: Model with fit() and predict() methods
            X: Features (n_samples, n_features)
            y: Target values (n_samples,)
            scoring: Scoring function (y_true, y_pred) -> score.
                If None, uses default scoring based on estimator
        
        Returns:
            List of scores for each fold
        
        Example:
            >>> from sklearn.linear_model import LinearRegression
            >>> from sklearn.metrics import r2_score
            >>> 
            >>> validator = WalkForwardValidator(n_splits=5)
            >>> scores = validator.validate(
            >>>     LinearRegression(),
            >>>     X, y,
            >>>     scoring=r2_score
            >>> )
            >>> print(f"Mean R²: {np.mean(scores):.3f} ± {np.std(scores):.3f}")
        """
        scores = []
        
        for fold, (train_idx, test_idx) in enumerate(self.split(X, y)):
            # Split data
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]
            
            # Train model
            estimator.fit(X_train, y_train)
            
            # Predict
            y_pred = estimator.predict(X_test)
            
            # Score
            if scoring is not None:
                if callable(scoring):
                    score = scoring(y_test, y_pred)
                else:
                    # Assume scoring is a string (sklearn scorer name)
                    try:
                        from sklearn.metrics import get_scorer
                        scorer = get_scorer(scoring)
                        score = scorer(estimator, X_test, y_test)
                    except ImportError:
                        raise ValueError(f"scikit-learn not available for scorer '{scoring}'")
            else:
                # Use estimator's default score method if available
                if hasattr(estimator, 'score'):
                    score = estimator.score(X_test, y_test)
                else:
                    # Fallback to R² score
                    try:
                        from sklearn.metrics import r2_score
                        score = r2_score(y_test, y_pred)
                    except ImportError:
                        # Last resort: use simple correlation
                        score = np.corrcoef(y_test, y_pred)[0, 1] if len(y_test) > 1 else 0.0
            
            scores.append(score)
            
            logger.info(
                f"Fold {fold+1}/{self.n_splits}: "
                f"train_size={len(train_idx)}, "
                f"test_size={len(test_idx)}, "
                f"score={score:.4f}"
            )
        
        return scores
    
    def __repr__(self) -> str:
        return (
            f"WalkForwardValidator("
            f"n_splits={self.n_splits}, "
            f"test_size={self.test_size}, "
            f"gap={self.gap}, "
            f"expanding_window={self.expanding_window})"
        )


def plot_walk_forward_splits(
    validator: WalkForwardValidator,
    n_samples: int,
    title: str = "Walk-Forward Cross-Validation"
) -> None:
    """
    Visualize walk-forward splits.
    
    Args:
        validator: WalkForwardValidator instance
        n_samples: Number of samples in dataset
        title: Plot title
    
    Example:
        >>> validator = WalkForwardValidator(n_splits=5, gap=2)
        >>> plot_walk_forward_splits(validator, n_samples=100)
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        logger.warning("matplotlib not available, skipping plot")
        return
    
    X_dummy = np.arange(n_samples).reshape(-1, 1)
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    for i, (train_idx, test_idx) in enumerate(validator.split(X_dummy)):
        # Plot train set
        ax.broken_barh(
            [(train_idx[0], len(train_idx))],
            (i, 0.8),
            facecolors='tab:blue',
            label='Train' if i == 0 else ""
        )
        
        # Plot test set
        ax.broken_barh(
            [(test_idx[0], len(test_idx))],
            (i, 0.8),
            facecolors='tab:orange',
            label='Test' if i == 0 else ""
        )
        
        # Plot gap if exists
        if validator.gap > 0:
            gap_start = train_idx[-1] + 1
            ax.broken_barh(
                [(gap_start, validator.gap)],
                (i, 0.8),
                facecolors='tab:gray',
                alpha=0.5,
                label='Gap' if i == 0 else ""
            )
    
    ax.set_ylim(0, validator.n_splits)
    ax.set_xlim(0, n_samples)
    ax.set_xlabel('Sample Index')
    ax.set_ylabel('Fold')
    ax.set_yticks(np.arange(validator.n_splits) + 0.4)
    ax.set_yticklabels([f'Fold {i+1}' for i in range(validator.n_splits)])
    ax.set_title(title)
    ax.legend(loc='upper left')
    
    plt.tight_layout()
    plt.show()


def compare_cv_methods(
    X: np.ndarray,
    y: np.ndarray,
    estimator: Any,
    scoring: Optional[Callable] = None
) -> dict:
    """
    Compare walk-forward vs K-fold cross-validation.
    
    Demonstrates the difference between proper time series validation
    and incorrect random shuffling.
    
    Args:
        X: Features
        y: Target
        estimator: Model to evaluate
        scoring: Scoring function
    
    Returns:
        Dictionary with scores from both methods
    
    Example:
        >>> from sklearn.linear_model import LinearRegression
        >>> results = compare_cv_methods(X, y, LinearRegression())
        >>> print("Walk-Forward:", results['walk_forward'])
        >>> print("K-Fold (WRONG):", results['kfold'])
    """
    try:
        from sklearn.model_selection import cross_val_score, KFold
    except ImportError:
        logger.warning("scikit-learn not available, skipping K-fold comparison")
        return {
            'walk_forward': [],
            'walk_forward_mean': 0.0,
            'walk_forward_std': 0.0,
            'kfold': [],
            'kfold_mean': 0.0,
            'kfold_std': 0.0,
            'difference': 0.0
        }
    
    # Walk-Forward (correct for time series)
    wf_validator = WalkForwardValidator(n_splits=5)
    wf_scores = wf_validator.validate(estimator, X, y, scoring=scoring)
    
    # K-Fold (INCORRECT for time series - for comparison only)
    kf_scores = cross_val_score(
        estimator, X, y,
        cv=KFold(n_splits=5, shuffle=False),
        scoring=scoring
    )
    
    logger.warning(
        "K-Fold scores are shown for comparison but should NOT be used "
        "for time series! They contain look-ahead bias."
    )
    
    return {
        'walk_forward': wf_scores,
        'walk_forward_mean': np.mean(wf_scores),
        'walk_forward_std': np.std(wf_scores),
        'kfold': kf_scores,
        'kfold_mean': np.mean(kf_scores),
        'kfold_std': np.std(kf_scores),
        'difference': np.mean(kf_scores) - np.mean(wf_scores)
    }
