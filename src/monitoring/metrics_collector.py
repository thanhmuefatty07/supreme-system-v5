import statistics
import time
from dataclasses import dataclass, field
from typing import List, Dict, Optional



@dataclass
class PerformanceMetrics:
    """Snapshot of system performance."""
    total_pnl: float = 0.0
    win_rate: float = 0.0
    total_trades: int = 0
    current_drawdown: float = 0.0
    max_drawdown: float = 0.0
    avg_latency_ms: float = 0.0
    p95_latency_ms: float = 0.0
    sharpe_ratio: float = 0.0  # Placeholder for advanced calculation



class MetricsCollector:
    def __init__(self):
        self.latencies: List[float] = []
        self.trades: List[float] = []  # List of PnL per trade
        self.initial_capital: float = 0.0
        self.current_capital: float = 0.0
        self.peak_capital: float = 0.0
        self.max_drawdown: float = 0.0

        # Performance optimization: Limit buffer size
        self.max_latency_buffer = 10000

    def initialize(self, initial_capital: float):
        """Reset and set initial state."""
        self.initial_capital = float(initial_capital)
        self.current_capital = float(initial_capital)
        self.peak_capital = float(initial_capital)
        self.latencies.clear()
        self.trades.clear()
        self.max_drawdown = 0.0

    def record_latency(self, latency_ms: float):
        """Record processing time of a tick."""
        self.latencies.append(latency_ms)
        if len(self.latencies) > self.max_latency_buffer:
            self.latencies.pop(0)

    def record_trade(self, pnl: float):
        """Record a closed trade PnL."""
        self.trades.append(pnl)
        self.current_capital += pnl

        # Update Peak
        if self.current_capital > self.peak_capital:
            self.peak_capital = self.current_capital

        # Update Drawdown
        drawdown = (self.peak_capital - self.current_capital) / self.peak_capital if self.peak_capital > 0 else 0.0
        if drawdown > self.max_drawdown:
            self.max_drawdown = drawdown

    def get_metrics(self) -> PerformanceMetrics:
        """Calculate and return current metrics."""
        total_trades = len(self.trades)
        wins = len([t for t in self.trades if t > 0])
        win_rate = (wins / total_trades) if total_trades > 0 else 0.0

        total_pnl = self.current_capital - self.initial_capital

        # Latency stats
        avg_lat = statistics.mean(self.latencies) if self.latencies else 0.0
        p95_lat = statistics.quantiles(self.latencies, n=20)[18] if len(self.latencies) >= 20 else avg_lat

        current_dd = (self.peak_capital - self.current_capital) / self.peak_capital if self.peak_capital > 0 else 0.0

        return PerformanceMetrics(
            total_pnl=total_pnl,
            win_rate=win_rate,
            total_trades=total_trades,
            current_drawdown=current_dd,
            max_drawdown=self.max_drawdown,
            avg_latency_ms=avg_lat,
            p95_latency_ms=p95_lat
        )