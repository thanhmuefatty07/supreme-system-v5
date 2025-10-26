"""
📈 Supreme System V5 - Backtesting Module
Comprehensive backtesting framework for trading strategies

Features:
- Historical data simulation
- Performance metrics calculation
- Risk analysis and drawdown tracking
- Multi-timeframe strategy testing
- Slippage and transaction cost modeling
"""

from .backtest_engine import BacktestEngine, BacktestConfig, BacktestResult

__version__ = "5.0.0"
__author__ = "Supreme System V5 Team"

__all__ = [
    "BacktestEngine",
    "BacktestConfig", 
    "BacktestResult",
]
