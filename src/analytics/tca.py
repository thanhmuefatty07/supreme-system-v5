#!/usr/bin/env python3
"""
Transaction Cost Analysis (TCA) Module - Ultra Optimized.

Analyzes execution costs including slippage and market impact.
Optimized for O(1) recording to avoid blocking execution latency.
"""

import time
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any
import numpy as np

# Avoid circular import - define ExecutionResult interface locally


@dataclass(slots=True)
class TradeCostMetrics:
    """Memory-optimized container for trade cost data."""

    order_id: str
    symbol: str
    side: str
    decision_price: float  # Price when strategy generated signal
    arrival_price: float   # Price when order reached exchange
    execution_price: float # Actual fill price
    size: float
    timestamp: float

    # Calculated Metrics (computed on-demand)
    slippage_bps: float = 0.0
    market_impact_bps: float = 0.0


class TransactionCostAnalyzer:
    """
    Analyzes execution costs including slippage and market impact.

    OPTIMIZATIONS:
    - O(1) trade recording (non-blocking)
    - Numpy vectorized batch analysis
    - Memory-bounded history (prevents leaks)
    - Lazy calculation of complex metrics
    """

    __slots__ = ['trades', 'max_history_size', 'last_summary_time']

    def __init__(self, max_history_size: int = 10000):
        """
        Initialize TCA analyzer.

        Args:
            max_history_size: Maximum trades to keep in memory
        """
        self.trades: List[TradeCostMetrics] = []
        self.max_history_size = max_history_size
        self.last_summary_time = time.time()

    def record_trade(self, decision_price: float, execution_result: Dict[str, Any]):
        """
        Record a trade for analysis.

        CRITICAL OPTIMIZATION: O(1) operation - must not block execution!

        Args:
            decision_price: Price when strategy made decision
            execution_result: Dict result from SmartRouter execution
        """
        if execution_result.get('status') != 'FILLED':
            return  # Only analyze successful trades

        # ULTRA-FAST: Calculate slippage immediately (cheap arithmetic)
        exec_price = execution_result.get('price', 0.0)
        side = execution_result.get('side', 'buy')

        if side == 'buy':
            slippage = (exec_price - decision_price) / decision_price
        else:  # sell
            slippage = (decision_price - exec_price) / decision_price

        slippage_bps = slippage * 10000.0  # Convert to basis points

        # Create memory-efficient record
        metric = TradeCostMetrics(
            order_id=execution_result.get('order_id', ''),
            symbol=execution_result.get('symbol', ''),
            side=side,
            decision_price=decision_price,
            arrival_price=decision_price,  # Approximation (can be improved)
            execution_price=exec_price,
            size=execution_result.get('quantity', 0.0),
            timestamp=execution_result.get('timestamp', time.time()),
            slippage_bps=slippage_bps
        )

        self.trades.append(metric)

        # MEMORY MANAGEMENT: Prevent unbounded growth
        if len(self.trades) > self.max_history_size:
            # Remove oldest trades (FIFO)
            remove_count = len(self.trades) - self.max_history_size
            self.trades = self.trades[remove_count:]

    def get_summary_statistics(self) -> Dict:
        """
        Calculate aggregate statistics using Numpy vectorization.

        This is called periodically (e.g., dashboard updates), not on critical path.
        Uses batch processing for efficiency.

        Returns:
            Dict with cost analysis metrics
        """
        if not self.trades:
            return {
                'avg_slippage_bps': 0.0,
                'p95_slippage_bps': 0.0,
                'total_volume_usd': 0.0,
                'sample_size': 0,
                'analysis_timestamp': time.time()
            }

        # OPTIMIZATION: Vectorized extraction and calculation
        slippages = np.array([t.slippage_bps for t in self.trades])
        sizes = np.array([t.size for t in self.trades])
        prices = np.array([t.execution_price for t in self.trades])

        # Vectorized calculations (much faster than loops)
        total_volume_usd = np.sum(sizes * prices)
        avg_slippage = np.mean(slippages)
        p95_slippage = np.percentile(slippages, 95)

        # Calculate slippage by side for deeper analysis
        buy_trades = [t for t in self.trades if t.side == 'buy']
        sell_trades = [t for t in self.trades if t.side == 'sell']

        buy_slippage = np.mean([t.slippage_bps for t in buy_trades]) if buy_trades else 0.0
        sell_slippage = np.mean([t.slippage_bps for t in sell_trades]) if sell_trades else 0.0

        return {
            'avg_slippage_bps': float(avg_slippage),
            'p95_slippage_bps': float(p95_slippage),
            'buy_slippage_bps': float(buy_slippage),
            'sell_slippage_bps': float(sell_slippage),
            'total_volume_usd': float(total_volume_usd),
            'sample_size': len(self.trades),
            'analysis_timestamp': time.time(),
            'period_seconds': time.time() - self.last_summary_time
        }

    def get_recent_trades(self, limit: int = 100) -> List[TradeCostMetrics]:
        """
        Get most recent trades for detailed analysis.

        Args:
            limit: Maximum number of trades to return

        Returns:
            List of recent TradeCostMetrics
        """
        return self.trades[-limit:] if self.trades else []

    def calculate_market_impact(self, trade: TradeCostMetrics) -> float:
        """
        Calculate market impact for a single trade.

        This is more complex and expensive - called on-demand only.

        Args:
            trade: Trade to analyze

        Returns:
            Market impact in basis points
        """
        # Simplified market impact calculation
        # In production, this would use order book data and volume analysis
        # For now, use size-based approximation
        impact_factor = min(trade.size / 1000.0, 1.0)  # Cap at reasonable level
        return impact_factor * 5.0  # 5 bps per 1000 units traded

    def export_to_csv(self, filepath: str):
        """
        Export TCA data to CSV for external analysis.

        Args:
            filepath: Output CSV file path
        """
        if not self.trades:
            return

        import pandas as pd

        # Convert to DataFrame for easy export
        data = [{
            'order_id': t.order_id,
            'symbol': t.symbol,
            'side': t.side,
            'decision_price': t.decision_price,
            'execution_price': t.execution_price,
            'size': t.size,
            'slippage_bps': t.slippage_bps,
            'timestamp': t.timestamp
        } for t in self.trades]

        df = pd.DataFrame(data)
        df.to_csv(filepath, index=False)

    def clear_history(self):
        """Clear all trade history (useful for testing)."""
        self.trades.clear()
        self.last_summary_time = time.time()
