"""Tests for optimizer utilities - AdamW vs Adam comparison"""

import pytest

# Try to import torch, but skip tests if not available
try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except (ImportError, OSError) as e:
    TORCH_AVAILABLE = False
    torch = None
    nn = None
    pytest.skip(f"PyTorch not available: {e}", allow_module_level=True)


@pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not available")
class TestAdamWOptimizer:
    """Basic tests for AdamW optimizer usage"""

    @pytest.fixture
    def simple_model(self):
        return nn.Linear(10, 1)

    def test_adamw_import(self):
        """Test 1: Can import AdamW"""
        from torch.optim import AdamW
        assert AdamW is not None

    def test_adamw_initialization(self, simple_model):
        """Test 2: AdamW initializes correctly"""
        from torch.optim import AdamW

        optimizer = AdamW(
            simple_model.parameters(),
            lr=0.001,
            weight_decay=0.01
        )

        assert optimizer.defaults['lr'] == 0.001
        assert optimizer.defaults['weight_decay'] == 0.01

    def test_adamw_step_works(self, simple_model):
        """Test 3: AdamW can perform optimization step"""
        from torch.optim import AdamW

        optimizer = AdamW(simple_model.parameters(), lr=0.001)

        # Fake backward pass
        x = torch.randn(5, 10)
        y = torch.randn(5, 1)
        output = simple_model(x)
        loss = ((output - y) ** 2).mean()
        loss.backward()

        # Should not raise error
        optimizer.step()
        optimizer.zero_grad()

    def test_adamw_vs_adam_weight_decay(self):
        """Test 4: AdamW handles weight decay differently than Adam"""
        from torch.optim import Adam, AdamW

        model1 = nn.Linear(10, 1)
        model2 = nn.Linear(10, 1)

        # Copy weights
        model2.load_state_dict(model1.state_dict())

        adam = Adam(model1.parameters(), lr=0.01, weight_decay=0.01)
        adamw = AdamW(model2.parameters(), lr=0.01, weight_decay=0.01)

        # One step
        x = torch.randn(5, 10)
        y = torch.randn(5, 1)

        loss1 = ((model1(x) - y) ** 2).mean()
        loss1.backward()
        adam.step()

        loss2 = ((model2(x) - y) ** 2).mean()
        loss2.backward()
        adamw.step()

        # Weights should differ (different weight decay application)
        assert not torch.allclose(model1.weight, model2.weight)


@pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not available")
class TestOptimizerFactory:
    """Test optimizer factory/helper"""

    def test_get_optimizer_adamw(self):
        """Test 5: Factory can create AdamW"""
        from src.utils.optimizer_utils import get_optimizer

        model = nn.Linear(10, 1)
        optimizer = get_optimizer(
            model.parameters(),
            optimizer_name='adamw',
            lr=0.001,
            weight_decay=0.01
        )

        assert optimizer.__class__.__name__ == 'AdamW'

    def test_get_optimizer_adam(self):
        """Test 6: Factory can create Adam"""
        from src.utils.optimizer_utils import get_optimizer

        model = nn.Linear(10, 1)
        optimizer = get_optimizer(
            model.parameters(),
            optimizer_name='adam',
            lr=0.001
        )

        assert optimizer.__class__.__name__ == 'Adam'


@pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not available")
class TestWeightInitialization:
    """Test weight initialization functions"""

    def test_he_normal_linear(self):
        """Test 7: He Normal init for Linear layer"""
        from src.utils.optimizer_utils import init_weights_he_normal

        model = nn.Linear(100, 50)
        init_weights_he_normal(model)

        # Check weights are initialized
        assert model.weight is not None
        assert model.bias is not None

        # Check variance is reasonable (He init property)
        var = model.weight.var().item()
        expected_var = 2.0 / 100  # 2/fan_in for ReLU
        assert 0.001 < var < 0.1  # Reasonable range

    def test_xavier_uniform_linear(self):
        """Test 8: Xavier Uniform init for Linear layer"""
        from src.utils.optimizer_utils import init_weights_xavier_uniform

        model = nn.Linear(100, 50)
        init_weights_xavier_uniform(model)

        # Check weights are initialized
        assert model.weight is not None
        assert model.bias is not None


# Run: pytest tests/test_optimizers.py -v
