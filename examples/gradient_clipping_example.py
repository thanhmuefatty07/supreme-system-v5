"""
Example: Using Gradient Clipping to prevent exploding gradients

This demonstrates how to use gradient clipping for stable training.
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


def demonstrate_gradient_clipping():
    """Demonstrate gradient clipping concepts"""

    logger.info("🚀 Gradient Clipping Demo")
    logger.info("=" * 60)

    # Try to import torch
    try:
        import torch
        import torch.nn as nn
        import torch.optim as optim
        from src.training.callbacks import GradientClipCallback
        from src.utils.training_utils import clip_grad_norm, get_grad_norm

        logger.info("✅ PyTorch available - running full demo")

        class SimpleRNN(nn.Module):
            """Simple RNN model prone to gradient explosion"""
            def __init__(self, input_size=10, hidden_size=32, output_size=1):
                super().__init__()
                self.rnn = nn.RNN(input_size, hidden_size, batch_first=True)
                self.linear = nn.Linear(hidden_size, output_size)

            def forward(self, x):
                out, _ = self.rnn(x)
                return self.linear(out[:, -1])  # Last timestep

        def create_problematic_gradients(model):
            """Create large gradients to simulate exploding gradient problem"""
            for param in model.parameters():
                # Set very large gradients (simulating gradient explosion)
                param.grad = torch.randn_like(param) * 1000

        # Setup model and optimizer
        model = SimpleRNN()
        optimizer = optim.SGD(model.parameters(), lr=0.01)
        criterion = nn.MSELoss()

        # Initialize gradient clipping callback
        grad_clip = GradientClipCallback(max_norm=5.0, verbose=True)
        grad_clip.set_model(model)

        # Simulate training loop with potential gradient explosion
        logger.info("Simulating training with large gradients...")

        for epoch in range(5):
            logger.info(f"\n📊 Epoch {epoch + 1}/5")

            # Simulate forward pass
            batch_size, seq_len, input_size = 32, 10, 10
            inputs = torch.randn(batch_size, seq_len, input_size)
            targets = torch.randn(batch_size, 1)

            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            # Backward pass (this would create gradients)
            loss.backward()

            # Demonstrate gradient explosion without clipping
            if epoch == 0:
                logger.info("Without clipping:")
                norm_before = get_grad_norm(model.parameters())
                logger.info(f"  Gradient norm: {norm_before:.2f}")

                # Create problematic large gradients
                create_problematic_gradients(model)
                norm_after_explosion = get_grad_norm(model.parameters())
                logger.info(f"  After explosion: {norm_after_explosion:.2f}")

            # Clip gradients using callback
            logger.info("With gradient clipping:")
            total_norm = grad_clip.on_after_backward()
            logger.info(f"  Clipped norm: {total_norm:.2f}")

            # Optimizer step
            optimizer.step()
            optimizer.zero_grad()

            # Log statistics
            stats = grad_clip.get_statistics()
            logger.info(f"  Clipping stats: {stats['clip_count']}/{stats['total_steps']} clips")

        # Final statistics
        logger.info("\n🎯 Final Statistics:")
        final_stats = grad_clip.get_statistics()
        for key, value in final_stats.items():
            if isinstance(value, float):
                logger.info(f"  {key}: {value:.4f}")
            else:
                logger.info(f"  {key}: {value}")

        logger.info("\n✅ Training completed successfully with stable gradients!")

        # Demonstrate standalone function
        logger.info("\n🔧 Demonstrating clip_grad_norm utility function")
        logger.info("=" * 60)

        # Create model with large gradients
        model = nn.Linear(10, 1)

        # Set large gradients manually
        for param in model.parameters():
            param.grad = torch.randn_like(param) * 100  # Very large

        # Measure norm before clipping
        norm_before = get_grad_norm(model.parameters())
        logger.info(f"Gradient norm before clipping: {norm_before:.2f}")

        # Clip gradients
        max_norm = 2.0
        clipped_norm = clip_grad_norm(model.parameters(), max_norm=max_norm)

        # Measure norm after clipping
        norm_after = get_grad_norm(model.parameters())
        logger.info(f"Gradient norm after clipping:  {norm_after:.2f}")
        logger.info(f"Clipped norm (reported):       {clipped_norm:.2f}")
        logger.info(f"Max allowed norm:              {max_norm}")

        # Verify clipping worked
        assert abs(norm_after - max_norm) < 0.01, "Clipping failed!"
        logger.info("✅ Clipping verification: PASSED")

    except (ImportError, OSError) as e:
        logger.warning(f"⚠️  PyTorch not available: {e}")
        logger.info("This is expected on Windows systems with DLL issues.")
        logger.info("Gradient clipping will work when PyTorch is properly installed.")
        logger.info("\n📚 Gradient Clipping Theory:")
        logger.info("1. Prevents exploding gradients by limiting norm")
        logger.info("2. Formula: clipped_grad = grad × min(1, max_norm / ||grad||)")
        logger.info("3. Essential for RNNs/LSTMs training stability")
        logger.info("4. Reduces NaN/inf losses significantly")


if __name__ == "__main__":
    demonstrate_gradient_clipping()
