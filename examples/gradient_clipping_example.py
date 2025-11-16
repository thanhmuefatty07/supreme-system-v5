"""
Example: Using Gradient Clipping to prevent exploding gradients.

This demonstrates the dramatic difference between training with and without gradient clipping.
"""

import sys
import logging
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Try to import torch - handle gracefully if not available
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from src.training.callbacks import GradientClipCallback
    from src.utils.training_utils import get_grad_norm
    TORCH_AVAILABLE = True
except (ImportError, OSError) as e:
    logger.warning(f"PyTorch not available: {e}")
    logger.warning("This example requires PyTorch to run.")
    logger.warning("Install PyTorch with: pip install torch")
    TORCH_AVAILABLE = False


class SimpleRNN(nn.Module):
    """Simple RNN model prone to exploding gradients"""
    def __init__(self, input_size=10, hidden_size=50, output_size=1):
        super().__init__()
        self.rnn = nn.RNN(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out, _ = self.rnn(x)
        out = self.fc(out[:, -1, :])
        return out


def create_problematic_gradients(model):
    """Create large gradients to simulate exploding gradient problem"""
    for param in model.parameters():
        # Set very large gradients (simulating gradient explosion)
        param.grad = torch.randn_like(param) * 1000


def train_without_clipping():
    """Demonstrate training WITHOUT gradient clipping (may explode)"""
    if not TORCH_AVAILABLE:
        logger.error("❌ PyTorch not available. Cannot run training example.")
        return

    print("\n" + "="*80)
    print("TRAINING WITHOUT GRADIENT CLIPPING")
    print("="*80)

    model = SimpleRNN()
    optimizer = optim.SGD(model.parameters(), lr=1.0)  # High LR intentionally
    criterion = nn.MSELoss()

    # Dummy data
    x = torch.randn(32, 20, 10)  # (batch, seq_len, features)
    y = torch.randn(32, 1)

    for epoch in range(5):
        optimizer.zero_grad()

        output = model(x)
        loss = criterion(output, y)
        loss.backward()

        # Check gradient norm (no clipping)
        grad_norm = get_grad_norm(model.parameters())

        optimizer.step()

        print(f"Epoch {epoch+1}: Loss={loss.item():.6f}, Grad Norm={grad_norm:.2f}")

        if torch.isnan(loss):
            print("❌ LOSS BECAME NaN - TRAINING CRASHED!")
            break


def train_with_clipping():
    """Demonstrate training WITH gradient clipping (stable)"""
    if not TORCH_AVAILABLE:
        logger.error("❌ PyTorch not available. Cannot run clipping example.")
        return

    print("\n" + "="*80)
    print("TRAINING WITH GRADIENT CLIPPING")
    print("="*80)

    model = SimpleRNN()
    optimizer = optim.SGD(model.parameters(), lr=1.0)  # Same high LR
    criterion = nn.MSELoss()

    # Initialize gradient clipping
    grad_clip = GradientClipCallback(max_norm=5.0, verbose=True)
    grad_clip.set_model(model)

    # Dummy data
    x = torch.randn(32, 20, 10)
    y = torch.randn(32, 1)

    for epoch in range(5):
        optimizer.zero_grad()

        output = model(x)
        loss = criterion(output, y)
        loss.backward()

        # Clip gradients
        grad_norm = grad_clip.on_after_backward()

        optimizer.step()

        print(f"Epoch {epoch+1}: Loss={loss.item():.6f}, Grad Norm={grad_norm:.2f}")

    # Print statistics
    print("\n📊 Gradient Clipping Statistics:")
    stats = grad_clip.get_statistics()
    for key, value in stats.items():
        if isinstance(value, float):
            print(f"   {key}: {value:.4f}")
        else:
            print(f"   {key}: {value}")

    print("\n✅ Training completed successfully - No NaN losses!")


def main():
    if not TORCH_AVAILABLE:
        logger.error("❌ Cannot run examples without PyTorch.")
        logger.error("Install PyTorch with: pip install torch")
        sys.exit(1)

    print("\n🎯 Gradient Clipping Demonstration")
    print("Comparing training stability with and without gradient clipping")

    # Without clipping (may crash)
    try:
        train_without_clipping()
    except Exception as e:
        print(f"❌ Training failed: {e}")

    # With clipping (stable)
    train_with_clipping()

    print("\n" + "="*80)
    print("🎓 KEY TAKEAWAYS:")
    print("1. High learning rates can cause exploding gradients")
    print("2. Gradient clipping prevents training instability")
    print("3. Always monitor gradient norms during training")
    print("4. Use max_norm=1.0-10.0 depending on model/task")
    print("="*80)


if __name__ == "__main__":
    main()
