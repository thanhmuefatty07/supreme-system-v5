#!/usr/bin/env python3
"""
Supreme System V5 - Risk Management Module

Enterprise-grade risk management with Kelly Criterion, circuit breakers,
and comprehensive position sizing controls.
"""

from .core import RiskManager
from .calculations import (
    calculate_kelly_criterion,
    apply_position_sizing,
    calculate_var_historical,
    calculate_sharpe_ratio
)
from .limits import CircuitBreaker, PositionSizeLimiter

__all__ = [
    'RiskManager',
    'CircuitBreaker',
    'PositionSizeLimiter',
    'calculate_kelly_criterion',
    'apply_position_sizing',
    'calculate_var_historical',
    'calculate_sharpe_ratio'
]

__version__ = "1.0.0"