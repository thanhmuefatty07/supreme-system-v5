"""Utility functions for training, data processing, and optimization."""

# Lazy imports to avoid PyTorch DLL loading issues during test collection
def __getattr__(name):
    if name in ['clip_grad_norm', 'get_grad_norm']:
        from src.utils.training_utils import clip_grad_norm, get_grad_norm
        return locals()[name]
    elif name in ['get_optimizer', 'OPTIMIZER_CONFIGS', 'init_weights_he_normal', 'init_weights_xavier_uniform']:
        from src.utils.optimizer_utils import (
            get_optimizer,
            OPTIMIZER_CONFIGS,
            init_weights_he_normal,
            init_weights_xavier_uniform
        )
        return locals()[name]
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")

__all__ = [
    'clip_grad_norm',
    'get_grad_norm',
    'get_optimizer',
    'OPTIMIZER_CONFIGS',
    'init_weights_he_normal',
    'init_weights_xavier_uniform'
]