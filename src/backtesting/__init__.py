"""
Supreme System V5 - Backtesting Module

Advanced backtesting framework with walk-forward optimization,
performance analysis, and strategy validation.
"""

# Import with try-except to handle relative import issues in testing
try:
    from .production_backtester import ProductionBacktester
    from .walk_forward import (
        AdvancedWalkForwardOptimizer,
        OptimizationResult,
        WalkForwardConfig,
        optimize_strategy_walk_forward,
    )
except ImportError:
    # Fallback for testing environments
    from production_backtester import ProductionBacktester
    from walk_forward import (
        AdvancedWalkForwardOptimizer,
        OptimizationResult,
        WalkForwardConfig,
        optimize_strategy_walk_forward,
    )

__all__ = [
    'ProductionBacktester',
    'AdvancedWalkForwardOptimizer',
    'WalkForwardConfig',
    'OptimizationResult',
    'optimize_strategy_walk_forward'
]
