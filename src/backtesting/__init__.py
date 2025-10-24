"""
🧪 Supreme System V5 - Production Backtesting Engine
Advanced backtesting with neuromorphic intelligence and multi-strategy support

Features:
- Historical data integration
- Multi-strategy backtesting
- Risk-adjusted performance metrics
- Walk-forward optimization
- Monte Carlo simulation
- Hardware-aware optimization
"""

__version__ = "5.0.0"
__author__ = "Supreme System V5 Team"

from .backtest_engine import BacktestEngine, BacktestConfig, BacktestResult
from .strategy_tester import StrategyTester, StrategyComparison
from .performance_analyzer import PerformanceAnalyzer, RiskMetrics
from .monte_carlo import MonteCarloSimulator

__all__ = [
    "BacktestEngine",
    "BacktestConfig", 
    "BacktestResult",
    "StrategyTester",
    "StrategyComparison",
    "PerformanceAnalyzer",
    "RiskMetrics",
    "MonteCarloSimulator"
]