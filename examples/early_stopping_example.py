"""
Example: Using Early Stopping in training loop

This demonstrates how to use EarlyStopping callback in a training loop.
"""

import sys
import logging
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.training.callbacks import EarlyStopping

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SimpleModel:
    """Simple model for demonstration"""
    def __init__(self):
        self.weights = {'linear': [1.0, 2.0, 3.0]}
    
    def state_dict(self):
        return self.weights.copy()
    
    def load_state_dict(self, state_dict):
        self.weights = state_dict.copy()


def train_with_early_stopping():
    """Demonstrate early stopping usage"""
    
    # Setup
    model = SimpleModel()
    
    # Initialize early stopping
    early_stopping = EarlyStopping(
        patience=10,
        min_delta=1e-4,
        restore_best_weights=True,
        verbose=True
    )
    early_stopping.set_model(model)
    
    # Simulate training loop
    logger.info("Starting training with early stopping...")
    
    for epoch in range(100):
        # Simulate training loss (decreasing)
        train_loss = 1.0 - epoch * 0.01
        
        # Simulate validation loss (decreasing then plateauing)
        if epoch < 50:
            val_loss = 1.0 - epoch * 0.01
        else:
            # Plateau after epoch 50
            val_loss = 0.5 + (epoch - 50) * 0.001
        
        # Check early stopping
        if early_stopping.on_epoch_end(epoch, val_loss):
            logger.info(f"\n✅ Training stopped early at epoch {epoch}")
            break
    
    # Get best metrics
    best_metrics = early_stopping.get_best_metrics()
    logger.info(f"\n📊 Best metrics:")
    for key, value in best_metrics.items():
        logger.info(f"   {key}: {value}")
    
    logger.info(f"\n✅ Final model weights: {model.weights}")


if __name__ == "__main__":
    train_with_early_stopping()

