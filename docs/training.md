# Training Module Documentation

## Overview

The training module provides utilities for model training, including callbacks for regularization and optimization.

## Callbacks

### EarlyStopping

Automatically stops training when validation loss stops improving, preventing overfitting and saving computational resources.

#### Parameters

- `patience` (int, default=10): Number of epochs with no improvement after which training stops
- `min_delta` (float, default=1e-4): Minimum change in monitored value to qualify as improvement
- `restore_best_weights` (bool, default=True): Whether to restore model weights from best epoch
- `verbose` (bool, default=True): Whether to print messages

#### Example Usage

```python
from src.training.callbacks import EarlyStopping
import torch.nn as nn
import torch.optim as optim

# Setup model and optimizer
model = YourModel()
optimizer = optim.Adam(model.parameters())
criterion = nn.MSELoss()

# Initialize early stopping
early_stopping = EarlyStopping(
    patience=10,
    min_delta=1e-4,
    restore_best_weights=True
)
early_stopping.set_model(model)

# Training loop
for epoch in range(100):
    # Training phase
    model.train()
    train_loss = 0.0
    for batch in train_loader:
        # ... training code ...
        train_loss += loss.item()

    # Validation phase
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for batch in val_loader:
            # ... validation code ...
            val_loss += loss.item()

    # Check early stopping
    if early_stopping.on_epoch_end(epoch, val_loss):
        print(f"Early stopping triggered at epoch {epoch}")
        break

# Get best metrics
best_metrics = early_stopping.get_best_metrics()
print(f"Best validation loss: {best_metrics['best_loss']:.4f}")
print(f"Best epoch: {best_metrics['best_epoch']}")
```

#### How It Works

1. **Monitors validation loss** after each epoch
2. **Tracks best loss** and saves model weights when improvement detected
3. **Increments patience counter** when no improvement
4. **Stops training** when patience counter reaches patience limit
5. **Restores best weights** (if enabled) before stopping

#### When to Use

- ✅ When training on limited data (high overfitting risk)
- ✅ When unsure about optimal number of epochs
- ✅ When computational resources are limited
- ✅ When you want automatic hyperparameter tuning

#### Best Practices

**Recommended Settings:**

- **Small datasets** (<10k samples): patience=5-10
- **Medium datasets** (10k-100k): patience=10-15
- **Large datasets** (>100k): patience=15-20
- **min_delta**: Keep small (1e-4 to 1e-3)

**Common Pitfalls:**

- ❌ **Too small patience**: May stop before finding optimum
- ❌ **Too large patience**: Defeats purpose of early stopping
- ❌ **Too large min_delta**: May miss small improvements
- ❌ **Not monitoring right metric**: Should be validation loss, not training loss

#### Integration with Existing Code

Early stopping is designed to be non-invasive:

```python
# Before (without early stopping)
for epoch in range(100):
    train_loss = train_one_epoch(model)
    val_loss = validate(model)
    print(f"Epoch {epoch}: val_loss={val_loss:.4f}")

# After (with early stopping) - minimal changes
early_stopping = EarlyStopping(patience=10)
early_stopping.set_model(model)

for epoch in range(100):
    train_loss = train_one_epoch(model)
    val_loss = validate(model)
    print(f"Epoch {epoch}: val_loss={val_loss:.4f}")

    if early_stopping.on_epoch_end(epoch, val_loss):  # Added
        break  # Added
```

#### Performance Impact

- **Memory:** Negligible (+1 model state dict)
- **Computation:** Negligible (<0.1% overhead)
- **Training time:** Reduced by 10-50% (depends on when stopping occurs)

### Gradient Clipping

Prevents exploding gradients by clipping the global norm of gradients.
Essential for training stability, especially with RNNs/LSTMs.

#### Parameters

- `max_norm` (float, default=5.0): Maximum gradient norm
- `norm_type` (float, default=2.0): Type of norm to use (L2)
- `error_if_nonfinite` (bool, default=False): Raise error for NaN/Inf gradients
- `verbose` (bool, default=True): Log clipping statistics

#### Example Usage

```python
from src.training.callbacks import GradientClipCallback

# Initialize callback
grad_clip = GradientClipCallback(max_norm=5.0)
grad_clip.set_model(model)

# In training loop
for epoch in range(num_epochs):
    optimizer.zero_grad()
    loss = compute_loss(model, data)
    loss.backward()

    # Clip gradients before optimizer step
    grad_clip.on_after_backward()

    optimizer.step()
```

#### How It Works

1. **Compute total norm**: Calculate L2 norm across all gradients
2. **Check threshold**: If norm > max_norm, calculate scaling factor
3. **Scale gradients**: Multiply all gradients by scaling factor
4. **Track statistics**: Monitor clipping frequency and norms

#### When to Use

- ✅ **RNNs/LSTMs**: Essential to prevent gradient explosion
- ✅ **Deep networks**: Helpful for training stability
- ✅ **Large learning rates**: Reduces instability risk
- ✅ **Unstable training**: Fixes NaN/inf losses

#### Best Practices

**Recommended Settings:**

- **RNNs/LSTMs**: max_norm=5.0-10.0
- **CNNs/Transformers**: max_norm=1.0-5.0
- **Small models**: max_norm=1.0-2.0
- **Large models**: max_norm=5.0-10.0

**Common Pitfalls:**

- ❌ **Too small max_norm**: May slow learning significantly
- ❌ **Too large max_norm**: Won't prevent explosion effectively
- ❌ **Wrong timing**: Must be called after backward(), before step()
- ❌ **No monitoring**: Monitor clipping ratio (<10% is good)

#### Standalone Function

For direct usage without callback:

```python
from src.utils.training_utils import clip_grad_norm

# After loss.backward()
total_norm = clip_grad_norm(model.parameters(), max_norm=5.0)

# Continue with optimizer.step()
optimizer.step()
```

#### Performance Impact

- **Memory:** Minimal (+small tracking overhead)
- **Computation:** Very low (norm calculation + scaling)
- **Training stability:** Significantly improved
- **Convergence:** Often faster and more stable

#### References

- Prechelt, L. (1998). "Early Stopping - But When?"
- Goodfellow et al. (2016). "Deep Learning", Section 7.8
- Pascanu et al. (2013). "On the difficulty of training RNNs"
