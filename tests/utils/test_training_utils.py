"""
Tests for training utilities - Gradient Clipping.

Written BEFORE implementation (TDD approach).
"""

import pytest
from unittest.mock import MagicMock, patch

# Lazy import torch to avoid crashes during test collection
try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except (ImportError, OSError):
    TORCH_AVAILABLE = False
    torch = None
    nn = None


@pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not available")
class TestGradientClipping:
    """Test suite for gradient clipping utilities"""

    @pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not available")
    @pytest.fixture
    def simple_model(self):
        """Simple model for testing"""
        model = nn.Linear(10, 1)
        return model

    @pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not available")
    @pytest.fixture
    def model_with_large_grads(self):
        """Model with artificially large gradients"""
        model = nn.Linear(10, 1)
        # Set large gradients manually
        for param in model.parameters():
            param.grad = torch.randn_like(param) * 100  # Large gradients
        return model

    @pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not available")
    def test_clip_grad_norm_reduces_large_gradients(self, model_with_large_grads):
        """Test 1: Clipping reduces gradient norm"""
        from src.utils.training_utils import clip_grad_norm

        # Calculate norm before clipping
        total_norm_before = torch.sqrt(
            sum(param.grad.norm() ** 2 for param in model_with_large_grads.parameters())
        )

        # Clip gradients
        max_norm = 5.0
        total_norm_after = clip_grad_norm(
            model_with_large_grads.parameters(),
            max_norm=max_norm
        )

        # Verify clipping occurred
        assert total_norm_before > max_norm, "Gradients should be large initially"
        assert abs(total_norm_after - max_norm) < 0.01, f"Clipped norm should be ~{max_norm}"

    @pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not available")
    def test_clip_grad_norm_preserves_small_gradients(self, simple_model):
        """Test 2: Small gradients are not clipped"""
        from src.utils.training_utils import clip_grad_norm

        # Set small gradients
        for param in simple_model.parameters():
            param.grad = torch.randn_like(param) * 0.01  # Small gradients

        # Calculate norm before clipping
        total_norm_before = torch.sqrt(
            sum(param.grad.norm() ** 2 for param in simple_model.parameters())
        ).item()

        # Clip with large max_norm
        max_norm = 10.0
        total_norm_after = clip_grad_norm(
            simple_model.parameters(),
            max_norm=max_norm
        )

        # Should not clip (norm < max_norm)
        assert abs(total_norm_after - total_norm_before) < 0.001

    @pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not available")
    def test_clip_grad_norm_handles_nan_gradients(self, simple_model):
        """Test 3: Detect NaN gradients"""
        from src.utils.training_utils import clip_grad_norm

        # Set NaN gradient
        for param in simple_model.parameters():
            param.grad = torch.tensor(float('nan')).expand_as(param)

        # Should raise error with error_if_nonfinite=True
        with pytest.raises(RuntimeError, match="non-finite"):
            clip_grad_norm(
                simple_model.parameters(),
                max_norm=5.0,
                error_if_nonfinite=True
            )

    @pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not available")
    def test_clip_grad_norm_handles_inf_gradients(self, simple_model):
        """Test 4: Detect Inf gradients"""
        from src.utils.training_utils import clip_grad_norm

        # Set Inf gradient
        for param in simple_model.parameters():
            param.grad = torch.tensor(float('inf')).expand_as(param)

        # Should raise error
        with pytest.raises(RuntimeError, match="non-finite"):
            clip_grad_norm(
                simple_model.parameters(),
                max_norm=5.0,
                error_if_nonfinite=True
            )

    def test_clip_grad_norm_with_no_gradients_fails_gracefully(self):
        """Test 5: Handle case where no gradients exist"""
        from src.utils.training_utils import clip_grad_norm

        # Mock parameters with no gradients
        mock_params = [MagicMock() for _ in range(3)]
        for param in mock_params:
            param.grad = None

        # Should return 0.0 and not crash
        total_norm = clip_grad_norm(
            mock_params,
            max_norm=5.0
        )

        assert total_norm == 0.0

    @pytest.mark.skipif(TORCH_AVAILABLE, reason="PyTorch available, skip torch-free test")
    def test_clip_grad_norm_without_torch_raises_error(self):
        """Test 6: Raises error when PyTorch not available"""
        from src.utils.training_utils import clip_grad_norm

        # Mock parameters
        mock_params = [MagicMock()]

        # Should raise error when torch not available
        with pytest.raises(RuntimeError, match="PyTorch is not available"):
            clip_grad_norm(mock_params, max_norm=5.0)


@pytest.mark.skipif(not torch, reason="PyTorch not available")
class TestGradientClippingCallback:
    """Test suite for GradientClip callback"""

    @pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not available")
    @pytest.fixture
    def simple_model(self):
        return nn.Linear(10, 1)

    def test_callback_initialization(self):
        """Test 1: Proper initialization"""
        from src.training.callbacks import GradientClipCallback

        callback = GradientClipCallback(max_norm=5.0)

        assert callback.max_norm == 5.0
        assert callback.norm_type == 2.0
        assert callback.error_if_nonfinite == False

    @pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not available")
    def test_callback_clips_after_backward(self, simple_model):
        """Test 2: Callback clips gradients after backward pass"""
        from src.training.callbacks import GradientClipCallback

        callback = GradientClipCallback(max_norm=1.0)
        callback.set_model(simple_model)

        # Simulate backward pass with large gradients
        for param in simple_model.parameters():
            param.grad = torch.randn_like(param) * 100

        # Call callback
        total_norm = callback.on_after_backward()

        # Verify gradients clipped
        assert total_norm <= 1.0 or abs(total_norm - 1.0) < 0.01

    @pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not available")
    def test_callback_logs_clipping_statistics(self, simple_model, caplog):
        """Test 3: Callback logs when clipping occurs"""
        from src.training.callbacks import GradientClipCallback

        callback = GradientClipCallback(max_norm=1.0, verbose=True)
        callback.set_model(simple_model)

        # Large gradients
        for param in simple_model.parameters():
            param.grad = torch.randn_like(param) * 100

        # Trigger clipping
        with caplog.at_level("INFO"):
            callback.on_after_backward()

        # Should log clipping event
        assert "Gradient norm" in caplog.text or "clipped" in caplog.text.lower()


# Run tests (should FAIL initially - no implementation yet)
# pytest tests/utils/test_training_utils.py -v
