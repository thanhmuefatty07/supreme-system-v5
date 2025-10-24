"""
🐍 Supreme System V5 - Mamba State Space Model Module
Linear O(L) complexity sequence modeling for trading

Features:
- Selective state space models
- Linear complexity scaling
- Long sequence processing
- Hardware acceleration ready
"""

__version__ = "5.0.0"
__author__ = "Supreme System V5 Team"

from .model import MambaSSMModel, MambaConfig
from .layers import MambaLayer, SelectiveSSM

# Export main classes  
MambaSSMEngine = MambaSSMModel  # Alias

__all__ = [
    "MambaSSMModel",
    "MambaSSMEngine",
    "MambaConfig",
    "MambaLayer",
    "SelectiveSSM"
]