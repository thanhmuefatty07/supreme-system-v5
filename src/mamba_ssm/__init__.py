"""
🐍 Supreme System V5 - Mamba State Space Model Module
Linear-time sequence modeling for financial time series

Features:
- O(L) computational complexity
- Long-range dependency modeling  
- Hardware-efficient selective state spaces
- Sub-quadratic attention alternative
"""

from .engine import MambaEngine, MambaSSMConfig
from .model import MambaModel, SelectiveSSM

__version__ = "5.0.0"
__author__ = "Supreme System V5 Team"

__all__ = [
    "MambaEngine",
    "MambaSSMConfig", 
    "MambaModel",
    "SelectiveSSM",
]
