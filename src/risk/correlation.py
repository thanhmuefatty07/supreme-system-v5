#!/usr/bin/env python3
"""
Portfolio Correlation Risk Manager - Ultra Optimized.

Manages cross-asset correlation risk with O(1) lookups and lazy updates.
Uses Numpy for high-performance correlation matrix calculations.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set
import time
from collections import deque


@dataclass(slots=True)
class CorrelationConfig:
    """Configuration for correlation risk management."""

    lookback_period: int = 100  # Number of periods for correlation calculation
    update_interval: float = 60.0  # Recalculate matrix every 60 seconds (lazy)
    high_correlation_threshold: float = 0.7  # Threshold for high correlation
    max_correlated_positions: int = 2  # Max correlated positions before blocking
    min_data_points: int = 10  # Minimum data points needed for correlation


@dataclass(slots=True)
class CorrelationMetrics:
    """Real-time correlation risk metrics."""

    symbol: str
    correlation_penalty: float  # 1.0 = no penalty, 0.5 = half size, 0.0 = block
    high_correlation_count: int
    correlated_symbols: List[str]
    matrix_age_seconds: float
    last_updated: float


class PortfolioCorrelationManager:
    """
    Ultra-optimized portfolio correlation risk manager.

    OPTIMIZATIONS:
    - O(1) price updates using deques
    - Lazy matrix recalculation (every 60s)
    - O(1) risk lookups from cached matrix
    - Numpy vectorized correlation calculations
    - Memory-bounded history to prevent leaks
    """

    __slots__ = [
        'config', 'price_history', 'correlation_matrix',
        'symbols_list', 'last_update_time', 'metrics_cache'
    ]

    def __init__(self, config: CorrelationConfig):
        """
        Initialize correlation risk manager.

        Args:
            config: Correlation configuration
        """
        self.config = config

        # OPTIMIZATION: Use deques for O(1) append/popleft operations
        self.price_history: Dict[str, deque] = {}

        # Cached correlation matrix (lazy updated)
        self.correlation_matrix: Optional[np.ndarray] = None
        self.symbols_list: List[str] = []  # Maps matrix indices to symbols

        self.last_update_time: float = 0.0

        # Cache for recent risk calculations (reduces redundant calculations)
        self.metrics_cache: Dict[str, CorrelationMetrics] = {}

    def update_price(self, symbol: str, price: float):
        """
        O(1) price update for incremental correlation tracking.

        Args:
            symbol: Asset symbol
            price: Current price
        """
        if symbol not in self.price_history:
            # Initialize with deque for O(1) operations
            self.price_history[symbol] = deque(maxlen=self.config.lookback_period)

        # O(1) append (deque automatically handles maxlen)
        self.price_history[symbol].append(price)

    def _should_recalculate_matrix(self) -> bool:
        """
        Check if correlation matrix needs recalculation.

        Returns:
            True if matrix should be recalculated
        """
        now = time.time()
        return (now - self.last_update_time) >= self.config.update_interval

    def _recalculate_correlation_matrix(self):
        """
        Recalculate correlation matrix using Numpy.

        This is O(N² * W) but runs infrequently (every 60s).
        Numpy's corrcoef is highly optimized C code.
        """
        symbols = [sym for sym, history in self.price_history.items()
                  if len(history) >= self.config.min_data_points]

        if len(symbols) < 2:
            # Need at least 2 symbols for correlation
            return

        # Find minimum history length for alignment
        min_length = min(len(self.price_history[sym]) for sym in symbols)

        if min_length < self.config.min_data_points:
            return

        # OPTIMIZATION: Use numpy array operations (much faster than loops)
        price_arrays = []
        for sym in symbols:
            # Take last min_length prices for alignment
            prices = list(self.price_history[sym])[-min_length:]
            price_arrays.append(prices)

        # Convert to numpy array for vectorized operations
        price_matrix = np.array(price_arrays)

        # NUMPY POWER: Highly optimized correlation calculation
        self.correlation_matrix = np.corrcoef(price_matrix)
        self.symbols_list = symbols
        self.last_update_time = time.time()

        # Clear stale cache entries
        self.metrics_cache.clear()

    def get_correlation_risk(self, symbol: str, current_positions: List[str]) -> float:
        """
        ULTRA-OPTIMIZED O(1) correlation risk lookup.

        CRITICAL: Must NEVER trigger matrix recalculation on hot path.
        Matrix updates happen asynchronously in background.

        Returns position size multiplier (0.0 to 1.0):
        - 1.0: No correlation risk, full size allowed
        - 0.5: High correlation, reduce size by half
        - 0.0: Excessive correlation, block trade

        Args:
            symbol: Symbol to check
            current_positions: List of currently open positions

        Returns:
            Position size multiplier
        """
        # ULTRA-FAST CACHE CHECK: O(1) lookup
        cache_key = f"{symbol}_{'_'.join(sorted(current_positions))}"
        if cache_key in self.metrics_cache:
            cached = self.metrics_cache[cache_key]
            # Use cached result if less than 30 seconds old (balances freshness vs speed)
            if time.time() - cached.last_updated < 30.0:
                return cached.correlation_penalty

        # CRITICAL OPTIMIZATION: If no matrix available, allow full size
        # This prevents blocking trades when correlation data is unavailable
        if self.correlation_matrix is None or symbol not in self.symbols_list:
            return 1.0  # No data available, allow full size

        try:
            symbol_idx = self.symbols_list.index(symbol)
        except ValueError:
            return 1.0  # Symbol not in correlation matrix

        # ULTRA-FAST CORRELATION COUNTING: O(N) where N = current_positions
        # This is acceptable since N is typically small (< 10 positions)
        high_correlation_count = 0
        correlated_symbols = []

        for position_symbol in current_positions:
            if position_symbol == symbol:
                continue  # Skip self-correlation

            if position_symbol not in self.symbols_list:
                continue  # Position not tracked

            try:
                pos_idx = self.symbols_list.index(position_symbol)
                correlation = abs(self.correlation_matrix[symbol_idx, pos_idx])

                if correlation >= self.config.high_correlation_threshold:
                    high_correlation_count += 1
                    correlated_symbols.append(position_symbol)

            except (ValueError, IndexError):
                continue  # Skip invalid correlations

        # DETERMINE PENALTY: O(1) logic
        if high_correlation_count >= self.config.max_correlated_positions:
            penalty = 0.0  # Block trade
        elif high_correlation_count > 0:
            penalty = 0.5  # Reduce size by half
        else:
            penalty = 1.0  # No penalty

        # CACHE RESULT: O(1) storage
        metrics = CorrelationMetrics(
            symbol=symbol,
            correlation_penalty=penalty,
            high_correlation_count=high_correlation_count,
            correlated_symbols=correlated_symbols,
            matrix_age_seconds=time.time() - self.last_update_time,
            last_updated=time.time()
        )
        self.metrics_cache[cache_key] = metrics

        return penalty

    def get_correlation_matrix(self) -> Optional[np.ndarray]:
        """
        Get current correlation matrix (triggers recalculation if needed).

        Returns:
            Numpy correlation matrix or None if insufficient data
        """
        if self._should_recalculate_matrix():
            self._recalculate_correlation_matrix()

        return self.correlation_matrix

    def get_correlation_stats(self) -> Dict:
        """
        Get comprehensive correlation statistics.

        Returns:
            Dict with correlation analysis metrics
        """
        if self.correlation_matrix is None:
            return {
                'matrix_available': False,
                'symbols_tracked': len(self.price_history),
                'last_update': self.last_update_time
            }

        # Calculate matrix statistics
        correlations = self.correlation_matrix[np.triu_indices_from(self.correlation_matrix, k=1)]

        return {
            'matrix_available': True,
            'symbols_tracked': len(self.symbols_list),
            'matrix_shape': self.correlation_matrix.shape,
            'avg_correlation': float(np.mean(np.abs(correlations))),
            'max_correlation': float(np.max(np.abs(correlations))),
            'high_corr_pairs': int(np.sum(np.abs(correlations) > self.config.high_correlation_threshold)),
            'last_update': self.last_update_time,
            'age_seconds': time.time() - self.last_update_time
        }

    def clear_cache(self):
        """Clear correlation matrix and cache (useful for testing)."""
        self.correlation_matrix = None
        self.symbols_list.clear()
        self.metrics_cache.clear()
        self.last_update_time = 0.0

    def get_memory_usage_estimate(self) -> Dict:
        """
        Estimate memory usage for monitoring.

        Returns:
            Dict with memory usage estimates
        """
        total_prices = sum(len(history) for history in self.price_history.values())

        matrix_memory = 0
        if self.correlation_matrix is not None:
            matrix_memory = self.correlation_matrix.nbytes

        return {
            'price_history_count': len(self.price_history),
            'total_price_points': total_prices,
            'correlation_matrix_bytes': matrix_memory,
            'cache_entries': len(self.metrics_cache),
            'estimated_total_kb': (total_prices * 8 + matrix_memory) // 1024  # Rough estimate
        }
