"""
Tests for training callbacks - Early Stopping.

Written BEFORE implementation (TDD approach).
"""

import pytest
from unittest.mock import MagicMock, Mock
from src.training.callbacks import EarlyStopping


class DummyModel:
    """Simple model for testing (no torch dependency)"""
    def __init__(self):
        self.weights = {'linear': [1.0, 2.0, 3.0]}
    
    def state_dict(self):
        return self.weights.copy()
    
    def load_state_dict(self, state_dict):
        self.weights = state_dict.copy()


class TestEarlyStopping:
    """Test suite for Early Stopping callback"""
    
    @pytest.fixture
    def early_stopping(self):
        """Fixture for EarlyStopping instance"""
        return EarlyStopping(patience=3, min_delta=0.01, verbose=False)
    
    def test_initialization(self, early_stopping):
        """Test 1: Proper initialization"""
        assert early_stopping.patience == 3
        assert early_stopping.min_delta == 0.01
        assert early_stopping.restore_best_weights == True
        assert early_stopping.wait == 0
        assert early_stopping.best_loss == float('inf')
    
    def test_improvement_detected(self, early_stopping):
        """Test 2: Detects improvement correctly"""
        losses = [1.0, 0.9, 0.8, 0.7]
        
        for i, loss in enumerate(losses):
            should_stop = early_stopping.on_epoch_end(epoch=i, val_loss=loss)
            assert should_stop == False  # Should NOT stop (improving)
        
        assert early_stopping.best_loss == 0.7
        assert early_stopping.wait == 0
    
    def test_early_stop_triggered(self, early_stopping):
        """Test 3: Early stopping triggers after patience epochs"""
        # Losses: improve, then plateau for 3 epochs
        losses = [1.0, 0.9, 0.8, 0.81, 0.82, 0.81]
        #                      ^^^ Best, then no improvement
        
        should_stop_history = []
        for i, loss in enumerate(losses):
            should_stop = early_stopping.on_epoch_end(epoch=i, val_loss=loss)
            should_stop_history.append(should_stop)
        
        # Should stop at last epoch (after patience=3 epochs)
        assert should_stop_history == [False, False, False, False, False, True]
        assert early_stopping.stopped_epoch == 5
        assert early_stopping.best_loss == 0.8
    
    def test_min_delta_respected(self):
        """Test 4: min_delta threshold works"""
        es = EarlyStopping(patience=2, min_delta=0.1, verbose=False)
        
        # Improvements < 0.1 should not count as improvement
        # Losses: 1.0 (best=1.0) -> 0.95 (improvement 0.05 < 0.1, doesn't count, wait=1)
        #         0.95 -> 0.92 (improvement 0.03 < 0.1, doesn't count, wait=2)
        #         0.92 -> 0.91 (improvement 0.01 < 0.1, doesn't count, wait=3 >= patience)
        # After patience=2 epochs without improvement, should stop
        losses = [1.0, 0.95, 0.92, 0.91]
        
        should_stop_history = []
        for i, loss in enumerate(losses):
            should_stop = es.on_epoch_end(epoch=i, val_loss=loss)
            should_stop_history.append(should_stop)
        
        # Should trigger after patience=2 epochs without improvement
        # Best loss is 1.0 (first epoch), wait increases for epochs 1,2,3
        # Should stop at epoch 3 (after 2 epochs without improvement)
        assert should_stop_history[-1] == True
        assert es.best_loss == 1.0  # First epoch is best (no improvement >= min_delta)
        assert es.wait >= es.patience
    
    def test_restore_best_weights(self):
        """Test 5: Best weights are restored"""
        model = DummyModel()
        es = EarlyStopping(patience=2, restore_best_weights=True, verbose=False)
        es.set_model(model)
        
        # Simulate training
        # Losses: 1.0 -> 0.8 (improvement, save weights) -> 0.9 (worse) -> 1.0 (worse)
        # Best at epoch 1 (loss=0.8)
        losses = [1.0, 0.8, 0.9, 1.0]
        
        for i, loss in enumerate(losses):
            # Modify model weights BEFORE checking early stopping
            # This simulates training that happens before validation
            model.weights['linear'] = [v + 0.1 for v in model.weights['linear']]
            
            should_stop = es.on_epoch_end(epoch=i, val_loss=loss)
        
        # Best weights should be restored (from epoch 1 when loss=0.8)
        assert should_stop == True
        assert es.best_epoch == 1
        assert es.best_loss == 0.8
        
        # Best weights should be saved at epoch 1 (after first modification)
        # Initial: [1.0, 2.0, 3.0]
        # After epoch 0: [1.1, 2.1, 3.1] (but loss=1.0, not best)
        # After epoch 1: [1.2, 2.2, 3.2] (loss=0.8, BEST - saved)
        assert es.best_weights is not None
        # Use approximate comparison due to floating point precision
        saved_weights = es.best_weights['linear']
        assert abs(saved_weights[0] - 1.2) < 0.01
        assert abs(saved_weights[1] - 2.2) < 0.01
        assert abs(saved_weights[2] - 3.2) < 0.01
    
    def test_no_improvement_from_start(self):
        """Test 6: Handle case where loss never improves"""
        es = EarlyStopping(patience=2, verbose=False)
        
        losses = [1.0, 1.1, 1.2, 1.3]  # Getting worse
        
        for i, loss in enumerate(losses):
            should_stop = es.on_epoch_end(epoch=i, val_loss=loss)
        
        assert should_stop == True
        assert es.best_loss == 1.0  # First is best
        assert es.wait == 3
    
    def test_get_best_metrics(self, early_stopping):
        """Test 7: get_best_metrics returns correct info"""
        losses = [1.0, 0.8, 0.9, 1.0]
        
        for i, loss in enumerate(losses):
            early_stopping.on_epoch_end(epoch=i, val_loss=loss)
        
        metrics = early_stopping.get_best_metrics()
        
        assert metrics['best_loss'] == 0.8
        assert metrics['best_epoch'] == 1
        assert 'stopped_epoch' in metrics
        assert 'patience_used' in metrics

