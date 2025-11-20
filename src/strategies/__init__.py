#!/usr/bin/env python3
"""
Strategies Package - Enterprise Trading Strategies

This package contains production-ready trading strategies with:
- SMA Crossover Strategy (100% test coverage)
- RSI Strategy (78.57% test coverage)
- Breakout Strategy (basic implementation)
- Strategy Registry for management
"""

from .base_strategy import BaseStrategy, Signal
from .sma_crossover import SMACrossover
from .rsi_strategy import RSIStrategy
from .breakout_strategy import BreakoutStrategy
from .strategy_registry import StrategyRegistry

# Create global registry instance
registry = StrategyRegistry()

# Register built-in strategies
registry.register_strategy(
    name="sma_crossover",
    strategy_class=SMACrossover,
    metadata={
        "description": "Simple Moving Average Crossover Strategy",
        "type": "trend_following",
        "indicators": ["SMA"],
        "timeframes": ["1h", "4h", "1d"],
        "risk_level": "medium"
    },
    parameters={
        "fast_window": 10,
        "slow_window": 20,
        "initial_capital": 10000.0,
        "max_position_size": 0.1,
        "max_daily_loss": 0.05
    }
)

registry.register_strategy(
    name="rsi_strategy",
    strategy_class=RSIStrategy,
    metadata={
        "description": "Relative Strength Index Strategy with Divergence",
        "type": "oscillator",
        "indicators": ["RSI"],
        "timeframes": ["15m", "1h", "4h"],
        "risk_level": "medium"
    },
    parameters={
        "rsi_period": 14,
        "overbought_level": 70,
        "oversold_level": 30,
        "min_signal_strength": 0.1,
        "enable_divergence": True,
        "divergence_lookback": 5,
        "initial_capital": 10000.0,
        "max_position_size": 0.1,
        "max_daily_loss": 0.05
    }
)

registry.register_strategy(
    name="breakout_strategy",
    strategy_class=BreakoutStrategy,
    metadata={
        "description": "Breakout Trading Strategy with Volume Confirmation",
        "type": "breakout",
        "indicators": ["support_resistance", "volume"],
        "timeframes": ["1h", "4h", "1d"],
        "risk_level": "high"
    },
    parameters={
        "lookback_period": 20,
        "breakout_threshold": 0.02,
        "volume_multiplier": 1.5,
        "consolidation_period": 10,
        "require_volume_confirmation": True,
        "min_breakout_strength": 0.1,
        "initial_capital": 10000.0,
        "max_position_size": 0.1,
        "max_daily_loss": 0.05
    }
)

__all__ = [
    'BaseStrategy',
    'Signal',
    'SMACrossover',
    'RSIStrategy',
    'BreakoutStrategy',
    'StrategyRegistry',
    'registry'
]