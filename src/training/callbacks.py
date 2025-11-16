"""
Training callbacks for model regularization and optimization.
"""

import copy
import logging
from typing import Optional, Dict, Any

logger = logging.getLogger(__name__)

# Lazy import torch to avoid DLL issues on Windows
TORCH_AVAILABLE = False
torch = None
nn = None

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

