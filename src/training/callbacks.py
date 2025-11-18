"""
Training callbacks for model regularization and optimization.
"""

import copy
import functools
import logging
from typing import Optional, Dict, Any, Callable

logger = logging.getLogger(__name__)

# Lazy import torch to avoid DLL issues on Windows
TORCH_AVAILABLE = False
torch = None
nn = None

# Attempt to import PyTorch
try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
    logger.debug("PyTorch successfully imported")
except (ImportError, OSError) as e:
    TORCH_AVAILABLE = False
    torch = None
    nn = None
    logger.warning(f"PyTorch not available: {e}. Some features will be disabled.")


def requires_torch(func: Callable) -> Callable:
    """
    Decorator to ensure PyTorch is available before executing a function.

    Args:
        func: Function that requires PyTorch

    Returns:
        Wrapped function that checks PyTorch availability

    Raises:
        RuntimeError: If PyTorch is not available
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        if not _check_torch():
            raise RuntimeError(
                f"Function {func.__name__} requires PyTorch, but PyTorch is not available. "
                f"Please install PyTorch or ensure it's properly configured."
            )
        return func(*args, **kwargs)
    return wrapper


def _check_torch():
    """Lazy check for torch availability"""
    global TORCH_AVAILABLE, torch, nn
    if not TORCH_AVAILABLE:
        try:
            import torch as _torch
            import torch.nn as _nn
            torch = _torch
            nn = _nn
            TORCH_AVAILABLE = True
        except (ImportError, OSError):
            TORCH_AVAILABLE = False
    return TORCH_AVAILABLE


class EarlyStopping:
    """
    Early stopping callback to stop training when validation loss stops improving.

    Monitors validation loss and stops training if no improvement is seen for
    `patience` consecutive epochs. Optionally restores the best model weights.

    Args:
        patience (int): Number of epochs with no improvement after which training
            will be stopped. Default: 10
        min_delta (float): Minimum change in monitored value to qualify as an
            improvement. Default: 1e-4
        restore_best_weights (bool): Whether to restore model weights from the
            epoch with the best value of the monitored metric. Default: True
        verbose (bool): If True, prints messages when early stopping triggers.
            Default: True

    Attributes:
        best_loss (float): Best validation loss observed
        wait (int): Number of epochs since last improvement
        stopped_epoch (int): Epoch at which training stopped
        best_weights (dict): State dict of best model

    Example:
        >>> model = MyModel()
        >>> early_stopping = EarlyStopping(patience=10, min_delta=1e-4)
        >>> early_stopping.set_model(model)
        >>>
        >>> for epoch in range(100):
        >>>     train_loss = train_one_epoch(model)
        >>>     val_loss = validate(model)
        >>>
        >>>     if early_stopping.on_epoch_end(epoch, val_loss):
        >>>         print(f"Early stopping triggered at epoch {epoch}")
        >>>         break

    References:
        - Prechelt, L. (1998). "Early Stopping - But When?"
        - Goodfellow et al. (2016). "Deep Learning", Section 7.8
    """

    def __init__(
        self,
        patience: int = 10,
        min_delta: float = 1e-4,
        restore_best_weights: bool = True,
        verbose: bool = True
    ):
        # Input validation
        if patience < 0:
            raise ValueError(f"patience must be >= 0, got {patience}")
        if min_delta < 0:
            raise ValueError(f"min_delta must be >= 0, got {min_delta}")

        self.patience = patience
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weights
        self.verbose = verbose

        # Internal state
        self.wait = 0
        self.best_loss = float('inf')
        self.best_epoch = 0
        self.stopped_epoch = 0
        self.best_weights: Optional[Dict[str, Any]] = None
        self.model: Optional[Any] = None  # Can be any model type

    def set_model(self, model: Any) -> None:
        """Set the model to monitor"""
        self.model = model

    def on_epoch_end(self, epoch: int, val_loss: float) -> bool:
        """
        Called at the end of each epoch.

        Args:
            epoch (int): Current epoch number
            val_loss (float): Validation loss for this epoch

        Returns:
            bool: True if training should stop, False otherwise
        """
        # Check for improvement
        if val_loss < (self.best_loss - self.min_delta):
            # Improvement detected
            self.best_loss = val_loss
            self.best_epoch = epoch
            self.wait = 0

            # Save best weights
            if self.restore_best_weights and self.model is not None:
                if hasattr(self.model, 'state_dict'):
                    # Generic model with state_dict method
                    self.best_weights = copy.deepcopy(self.model.state_dict())
                elif _check_torch() and isinstance(self.model, nn.Module):
                    # PyTorch model
                    self.best_weights = copy.deepcopy(self.model.state_dict())
                elif hasattr(self.model, '__dict__'):
                    # Fallback: save entire model state
                    self.best_weights = copy.deepcopy(self.model.__dict__)

            if self.verbose:
                logger.info(
                    f"Epoch {epoch}: val_loss improved to {val_loss:.6f}, "
                    f"saving model weights"
                )
        else:
            # No improvement
            self.wait += 1

            if self.verbose and self.wait > 0:
                logger.info(
                    f"Epoch {epoch}: val_loss did not improve from {self.best_loss:.6f} "
                    f"(patience: {self.wait}/{self.patience})"
                )

            # Check if should stop
            if self.wait >= self.patience:
                self.stopped_epoch = epoch

                # Restore best weights
                if self.restore_best_weights and self.best_weights is not None:
                    if self.model is not None:
                        if hasattr(self.model, 'load_state_dict'):
                            self.model.load_state_dict(self.best_weights)
                        elif _check_torch() and isinstance(self.model, nn.Module):
                            # PyTorch model
                            self.model.load_state_dict(self.best_weights)
                        elif hasattr(self.model, '__dict__'):
                            self.model.__dict__.update(self.best_weights)

                        if self.verbose:
                            logger.info(
                                f"Restoring model weights from epoch {self.best_epoch} "
                                f"with val_loss={self.best_loss:.6f}"
                            )

                if self.verbose:
                    logger.info(
                        f"Early stopping triggered at epoch {epoch}. "
                        f"Best epoch: {self.best_epoch}, best val_loss: {self.best_loss:.6f}"
                    )

                return True  # Signal to stop training

        return False  # Continue training

    def get_best_metrics(self) -> Dict[str, Any]:
        """Get metrics from best epoch"""
        return {
            'best_loss': self.best_loss,
            'best_epoch': self.best_epoch,
            'stopped_epoch': self.stopped_epoch,
            'patience_used': self.wait
        }

    def __repr__(self) -> str:
        return (
            f"EarlyStopping(patience={self.patience}, "
            f"min_delta={self.min_delta}, "
            f"restore_best_weights={self.restore_best_weights})"
        )


class GradientClipCallback:
    """
    Callback to clip gradients after backward pass.

    Prevents exploding gradients by clipping the global norm of gradients.
    Should be called after loss.backward() but before optimizer.step().

    Args:
        max_norm (float): Maximum gradient norm. Default: 5.0
        norm_type (float): Type of norm to use. Default: 2.0 (L2)
        error_if_nonfinite (bool): Raise error if gradients are NaN/Inf.
            Default: False
        verbose (bool): Log clipping statistics. Default: True

    Attributes:
        clip_count (int): Number of times clipping was triggered
        total_norm_history (list): History of gradient norms

    Example:
        >>> model = MyModel()
        >>> optimizer = torch.optim.Adam(model.parameters())
        >>> grad_clip = GradientClipCallback(max_norm=5.0)
        >>> grad_clip.set_model(model)
        >>>
        >>> for epoch in range(num_epochs):
        >>>     optimizer.zero_grad()
        >>>     loss = compute_loss(model, data)
        >>>     loss.backward()
        >>>
        >>>     # Clip gradients
        >>>     grad_clip.on_after_backward()
        >>>
        >>>     optimizer.step()

    References:
        - Pascanu et al. (2013). "On the difficulty of training RNNs"
        - Goodfellow et al. (2016). "Deep Learning", Section 10.11.1
    """

    def __init__(
        self,
        max_norm: float = 5.0,
        norm_type: float = 2.0,
        error_if_nonfinite: bool = False,
        verbose: bool = True
    ):
        if max_norm <= 0:
            raise ValueError(f"max_norm must be > 0, got {max_norm}")

        self.max_norm = max_norm
        self.norm_type = norm_type
        self.error_if_nonfinite = error_if_nonfinite
        self.verbose = verbose

        self.model = None
        self.clip_count = 0
        self.total_norm_history = []

    @requires_torch
    def set_model(self, model) -> None:
        """Set the model to clip gradients for"""
        self.model = model

    def on_after_backward(self) -> float:
        """
        Called after loss.backward() to clip gradients.

        Returns:
            float: Total gradient norm before clipping
        """
        if self.model is None:
            raise RuntimeError("Model not set. Call set_model() first.")

        # Import here to avoid circular dependency
        from src.utils.training_utils import clip_grad_norm

        # Clip gradients
        total_norm = clip_grad_norm(
            self.model.parameters(),
            max_norm=self.max_norm,
            norm_type=self.norm_type,
            error_if_nonfinite=self.error_if_nonfinite
        )

        # Track statistics
        self.total_norm_history.append(total_norm)

        # Check if clipping occurred
        if total_norm > self.max_norm:
            self.clip_count += 1

            if self.verbose:
                logger.info(
                    f"Gradient norm {total_norm:.4f} exceeded max_norm {self.max_norm:.4f}, "
                    f"clipped (total clips: {self.clip_count})"
                )

        return total_norm

    def get_statistics(self) -> dict:
        """Get clipping statistics"""
        if len(self.total_norm_history) == 0:
            return {
                'clip_count': 0,
                'clip_ratio': 0.0,
                'avg_norm': 0.0,
                'max_norm_seen': 0.0
            }

        return {
            'clip_count': self.clip_count,
            'clip_ratio': self.clip_count / len(self.total_norm_history),
            'avg_norm': sum(self.total_norm_history) / len(self.total_norm_history),
            'max_norm_seen': max(self.total_norm_history),
            'total_steps': len(self.total_norm_history)
        }

    def __repr__(self) -> str:
        return (
            f"GradientClipCallback(max_norm={self.max_norm}, "
            f"norm_type={self.norm_type}, "
            f"error_if_nonfinite={self.error_if_nonfinite})"
        )

