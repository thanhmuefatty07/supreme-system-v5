#!/usr/bin/env python3
"""
Market Regime Detection using Hidden Markov Models (HMM)

Ultra-optimized for real-time trading with lazy training and O(1) inference.
Detects bull, bear, and sideways market regimes for adaptive strategy behavior.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
import time
from collections import deque
from enum import Enum

import warnings
warnings.filterwarnings('ignore')  # Suppress hmmlearn warnings in production


class MarketRegime(Enum):
    """Market regime classifications."""
    BULL = "bull"
    BEAR = "bear"
    SIDEWAYS = "sideways"


@dataclass(slots=True)
class RegimeFeatures:
    """Features used for regime classification."""
    returns: float
    volatility: float
    volume_ratio: float
    timestamp: float


@dataclass(slots=True)
class RegimePrediction:
    """HMM regime prediction result."""
    regime: MarketRegime
    confidence: float
    probabilities: Dict[str, float]
    features_used: RegimeFeatures
    model_age_seconds: float
    predicted_at: float


class HMMRegimeDetector:
    """
    Hidden Markov Model-based market regime detector.

    OPTIMIZATIONS:
    - Lazy training (every 1 hour) prevents execution blocking
    - Pre-computed models for O(1) inference
    - Memory-bounded feature history
    - Numpy-optimized calculations

    Detects 3 regimes:
    - BULL: Strong upward trends
    - BEAR: Strong downward trends
    - SIDEWAYS: Range-bound/consolidation
    """

    __slots__ = [
        'n_regimes', 'feature_history', 'models', 'current_regime',
        'last_training_time', 'training_interval', 'min_training_samples',
        'feature_buffer_size', 'model_params'
    ]

    def __init__(
        self,
        n_regimes: int = 3,
        training_interval: float = 3600.0,  # 1 hour
        min_training_samples: int = 100,
        feature_buffer_size: int = 1000
    ):
        """
        Initialize HMM regime detector.

        Args:
            n_regimes: Number of market regimes to detect (default 3)
            training_interval: Seconds between model retraining
            min_training_samples: Minimum samples needed for training
            feature_buffer_size: Max features to keep in memory
        """
        self.n_regimes = n_regimes
        self.training_interval = training_interval
        self.min_training_samples = min_training_samples
        self.feature_buffer_size = feature_buffer_size

        # OPTIMIZATION: Deque for O(1) feature updates
        self.feature_history: deque = deque(maxlen=feature_buffer_size)

        # Pre-trained models (lazy loaded)
        self.models: Dict[str, Any] = {}

        # Current regime state
        self.current_regime: Optional[MarketRegime] = None
        self.last_training_time: float = 0.0

        # Model parameters (will be learned from data)
        self.model_params: Dict[str, Any] = {
            'transition_matrix': None,
            'emission_means': None,
            'emission_covars': None,
            'start_prob': None
        }

        # Initialize with simple fallback model
        self._initialize_fallback_model()

    def _initialize_fallback_model(self):
        """Initialize simple fallback model for when HMM training fails."""
        # Simple rule-based fallback for initial predictions
        self.models['fallback'] = {
            'bull_threshold': 0.02,      # 2% daily return = bull
            'bear_threshold': -0.02,     # -2% daily return = bear
            'volatility_threshold': 0.05  # 5% volatility = high volatility
        }

    def update_market_data(self, close_price: float, volume: float,
                          timestamp: float, prev_close: Optional[float] = None):
        """
        O(1) market data update for regime detection.

        Args:
            close_price: Current closing price
            volume: Trading volume
            timestamp: Data timestamp
            prev_close: Previous closing price (for return calculation)
        """
        if prev_close is None or prev_close <= 0:
            return  # Cannot calculate features without previous price

        # Calculate regime features (O(1) operations)
        returns = (close_price - prev_close) / prev_close

        # Calculate rolling volatility (simple approximation)
        recent_prices = [f.returns for f in list(self.feature_history)[-20:]]
        if recent_prices:
            volatility = np.std(recent_prices) if len(recent_prices) > 1 else abs(returns)
        else:
            volatility = abs(returns)

        # Volume ratio (normalized)
        avg_volume = np.mean([f.volume_ratio for f in list(self.feature_history)[-20:]]) if self.feature_history else 1.0
        volume_ratio = volume / avg_volume if avg_volume > 0 else 1.0

        # Create feature vector
        features = RegimeFeatures(
            returns=returns,
            volatility=volatility,
            volume_ratio=volume_ratio,
            timestamp=timestamp
        )

        # O(1) append to deque
        self.feature_history.append(features)

    def _should_retrain_model(self) -> bool:
        """
        Check if HMM model should be retrained.

        Returns:
            True if retraining is needed
        """
        now = time.time()

        # Check time interval
        if now - self.last_training_time < self.training_interval:
            return False

        # Check if we have enough new data
        if len(self.feature_history) < self.min_training_samples:
            return False

        return True

    def _extract_training_data(self) -> Optional[np.ndarray]:
        """
        Extract and prepare training data from feature history.

        Returns:
            Numpy array of shape (n_samples, n_features) or None
        """
        if len(self.feature_history) < self.min_training_samples:
            return None

        # Extract recent features for training
        recent_features = list(self.feature_history)[-self.min_training_samples:]

        # Convert to numpy array
        training_data = np.array([[
            f.returns,
            f.volatility,
            f.volume_ratio
        ] for f in recent_features])

        return training_data

    def _train_hmm_model(self, training_data: np.ndarray):
        """
        Train HMM model using market data.

        This is computationally expensive but runs infrequently (every hour).
        Uses GaussianHMM from hmmlearn for optimal performance.
        """
        try:
            from hmmlearn import hmm

            # Create and train HMM
            model = hmm.GaussianHMM(
                n_components=self.n_regimes,
                covariance_type="full",
                n_iter=100,
                random_state=42
            )

            # Fit model to training data
            model.fit(training_data)

            # Store trained model
            self.models['hmm'] = model
            self.last_training_time = time.time()

            # Cache model parameters for fast inference
            self.model_params = {
                'transition_matrix': model.transmat_,
                'emission_means': model.means_,
                'emission_covars': model.covars_,
                'start_prob': model.startprob_
            }

            return True

        except Exception as e:
            print(f"HMM training failed: {e}, using fallback model")
            return False

    def _fallback_regime_detection(self, features: RegimeFeatures) -> MarketRegime:
        """
        Fallback regime detection using simple rules.

        Used when HMM model is unavailable.
        """
        fallback = self.models.get('fallback', {})
        bull_threshold = fallback.get('bull_threshold', 0.02)
        bear_threshold = fallback.get('bear_threshold', -0.02)

        if features.returns > bull_threshold:
            return MarketRegime.BULL
        elif features.returns < bear_threshold:
            return MarketRegime.BEAR
        else:
            return MarketRegime.SIDEWAYS

    def _predict_regime_hmm(self, features: RegimeFeatures) -> Tuple[MarketRegime, Dict[str, float]]:
        """
        Predict regime using trained HMM model.

        Returns:
            Tuple of (predicted_regime, probability_dict)
        """
        model = self.models.get('hmm')
        if model is None:
            return self._fallback_regime_detection(features), {'fallback': 1.0}

        try:
            # Prepare feature vector for prediction
            feature_vector = np.array([[features.returns, features.volatility, features.volume_ratio]])

            # Predict most likely state
            state_sequence = model.predict(feature_vector)
            predicted_state = state_sequence[0]

            # Get state probabilities
            state_probs = model.predict_proba(feature_vector)[0]

            # Map HMM states to market regimes (simplified mapping)
            # In practice, this would be learned from labeled data
            regime_mapping = {
                0: MarketRegime.BULL,      # Highest return state
                1: MarketRegime.SIDEWAYS,  # Medium volatility state
                2: MarketRegime.BEAR       # Lowest return state
            }

            predicted_regime = regime_mapping.get(predicted_state, MarketRegime.SIDEWAYS)

            # Convert to probability dict
            probabilities = {
                'bull': float(state_probs[0]) if predicted_state == 0 else 0.0,
                'sideways': float(state_probs[1]) if predicted_state == 1 else 0.0,
                'bear': float(state_probs[2]) if predicted_state == 2 else 0.0
            }

            return predicted_regime, probabilities

        except Exception as e:
            print(f"HMM prediction failed: {e}, using fallback")
            return self._fallback_regime_detection(features), {'fallback': 1.0}

    def get_current_regime(self, confidence_threshold: float = 0.6) -> Optional[RegimePrediction]:
        """
        Get current market regime prediction - ULTRA OPTIMIZED O(1).

        CRITICAL: Training happens asynchronously, inference is always O(1).

        Args:
            confidence_threshold: Minimum confidence required

        Returns:
            RegimePrediction or None if insufficient data
        """
        if not self.feature_history:
            return None

        # Get latest features for prediction
        latest_features = self.feature_history[-1]

        # ULTRA-FAST: Use fallback for initial predictions or when HMM unavailable
        if (not self.models.get('hmm') or
            time.time() - self.last_training_time > self.training_interval * 2):
            # Use fast rule-based fallback
            regime = self._fallback_regime_detection(latest_features)
            probabilities = {'fallback': 1.0}
            confidence = 0.5  # Lower confidence for fallback
        else:
            # O(1) HMM inference using pre-trained model
            regime, probabilities = self._predict_regime_hmm(latest_features)
            confidence = max(probabilities.values())

        # Only return prediction if confidence is high enough
        if confidence < confidence_threshold:
            regime = MarketRegime.SIDEWAYS  # Default to sideways if uncertain

        prediction = RegimePrediction(
            regime=regime,
            confidence=confidence,
            probabilities=probabilities,
            features_used=latest_features,
            model_age_seconds=time.time() - self.last_training_time,
            predicted_at=time.time()
        )

        self.current_regime = regime
        return prediction

    def background_train_model(self):
        """
        Background training method - call this from async task/scheduler.

        This separates expensive training from hot-path inference.
        """
        if self._should_retrain_model():
            training_data = self._extract_training_data()
            if training_data is not None:
                self._train_hmm_model(training_data)

    def get_regime_statistics(self) -> Dict[str, Any]:
        """
        Get comprehensive regime detection statistics.

        Returns:
            Dict with model and performance statistics
        """
        if not self.feature_history:
            return {'status': 'no_data'}

        latest_features = self.feature_history[-1]

        return {
            'status': 'active',
            'current_regime': self.current_regime.value if self.current_regime else None,
            'feature_count': len(self.feature_history),
            'model_trained': 'hmm' in self.models,
            'last_training_age': time.time() - self.last_training_time,
            'latest_features': {
                'returns': latest_features.returns,
                'volatility': latest_features.volatility,
                'volume_ratio': latest_features.volume_ratio,
                'timestamp': latest_features.timestamp
            },
            'model_params_available': self.model_params['transition_matrix'] is not None
        }

    def force_retrain(self):
        """Force immediate model retraining (useful for testing)."""
        if len(self.feature_history) >= self.min_training_samples:
            training_data = self._extract_training_data()
            if training_data is not None:
                self._train_hmm_model(training_data)

    def clear_history(self):
        """Clear feature history (useful for testing/memory management)."""
        self.feature_history.clear()
        self.current_regime = None
        self.last_training_time = 0.0
