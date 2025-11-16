"""
Training utilities for gradient clipping and optimization helpers.
"""

import logging
from typing import Iterable, Union, Optional

# Lazy import torch to avoid Windows DLL issues
try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except (ImportError, OSError):
    TORCH_AVAILABLE = False
    torch = None
    nn = None

logger = logging.getLogger(__name__)


def clip_grad_norm(
    parameters: Union["torch.Tensor", Iterable["torch.Tensor"]],
    max_norm: float,
    norm_type: float = 2.0,
    error_if_nonfinite: bool = False
) -> float:
    """
    Clip gradient norm of an iterable of parameters.

    The norm is computed over all gradients together, as if they were
    concatenated into a single vector. Gradients are modified in-place.

    Args:
        parameters: Iterable of parameters or single tensor with gradients
        max_norm: Max norm of the gradients
        norm_type: Type of norm (default: 2.0 for L2 norm)
        error_if_nonfinite: If True, raises error if total norm is NaN or Inf

    Returns:
        Total norm of the parameters (viewed as a single vector)

    Example:
        >>> model = MyModel()
        >>> optimizer = torch.optim.Adam(model.parameters())
        >>>
        >>> for epoch in range(num_epochs):
        >>>     optimizer.zero_grad()
        >>>     loss = compute_loss(model(inputs), targets)
        >>>     loss.backward()
        >>>
        >>>     # Clip gradients before optimizer step
        >>>     total_norm = clip_grad_norm(model.parameters(), max_norm=5.0)
        >>>
        >>>     optimizer.step()

    References:
        - Pascanu et al. (2013). "On the difficulty of training RNNs"
        - PyTorch docs: torch.nn.utils.clip_grad_norm_
    """
    if not TORCH_AVAILABLE:
        raise RuntimeError("PyTorch is required for gradient clipping but is not available")

    # Convert to list if single tensor
    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]

    # Filter parameters that have gradients
    parameters = [p for p in parameters if p.grad is not None]

    # Handle empty case
    if len(parameters) == 0:
        return 0.0

    # Compute total norm
    if norm_type == float('inf'):
        # Infinity norm
        total_norm = max(p.grad.detach().abs().max().item() for p in parameters)
    else:
        # L_p norm
        total_norm = torch.norm(
            torch.stack([
                torch.norm(p.grad.detach(), norm_type)
                for p in parameters
            ]),
            norm_type
        ).item()

    # Check for non-finite values
    if error_if_nonfinite and (torch.isnan(torch.tensor(total_norm)) or torch.isinf(torch.tensor(total_norm))):
        raise RuntimeError(
            f"The total norm of gradients is {total_norm}, which contains "
            f"non-finite values (NaN or Inf). This usually indicates a problem "
            f"with the model or loss function."
        )

    # Clip gradients
    clip_coef = max_norm / (total_norm + 1e-6)
    clip_coef_clamped = min(clip_coef, 1.0)

    for p in parameters:
        p.grad.detach().mul_(clip_coef_clamped)

    return total_norm


def get_grad_norm(
    parameters: Union["torch.Tensor", Iterable["torch.Tensor"]],
    norm_type: float = 2.0
) -> float:
    """
    Get gradient norm without clipping.

    Useful for logging/monitoring gradient magnitudes.

    Args:
        parameters: Iterable of parameters or single tensor
        norm_type: Type of norm (default: 2.0)

    Returns:
        Total gradient norm

    Example:
        >>> grad_norm = get_grad_norm(model.parameters())
        >>> logger.info(f"Gradient norm: {grad_norm:.4f}")
    """
    if not TORCH_AVAILABLE:
        return 0.0  # Return 0 if torch not available

    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]

    parameters = [p for p in parameters if p.grad is not None]

    if len(parameters) == 0:
        return 0.0

    # Check torch availability
    if not _check_torch():
        raise RuntimeError("PyTorch is not available. Gradient norm calculation requires PyTorch.")

    if norm_type == float('inf'):
        total_norm = max(p.grad.detach().abs().max().item() for p in parameters)
    else:
        total_norm = torch.norm(
            torch.stack([
                torch.norm(p.grad.detach(), norm_type)
                for p in parameters
            ]),
            norm_type
        ).item()

    return total_norm
