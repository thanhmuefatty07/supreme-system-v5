"""
🐍 Supreme System V5 - Mamba State Space Model Module
O(L) linear complexity sequence modeling for unlimited context length
Revolutionary selective mechanism for time series processing

This module implements the breakthrough Mamba architecture:
- Selective State Space Models
- O(L) linear complexity (vs O(L²) for Transformers)
- Unlimited context length
- Hardware-efficient processing
- Real-time sequence modeling

Components:
- MambaSSMEngine: Main processing engine
- SelectiveSSM: Core selective state space implementation
- MambaBlock: Individual Mamba layer
- MambaConfig: System configuration
"""

from .engine import (
    MambaSSMEngine,
    MambaConfig,
    SelectiveSSM,
    MambaBlock,
    demo_mamba_ssm
)

__version__ = "5.0.0"
__author__ = "Supreme System V5 Team"
__email__ = "thanhmuefatty07@gmail.com"

# Module exports
__all__ = [
    "MambaSSMEngine",
    "MambaConfig",
    "SelectiveSSM",
    "MambaBlock",
    "demo_mamba_ssm"
]

# Performance specifications
PERFORMANCE_SPECS = {
    "complexity": "O(L) linear",
    "context_length": "unlimited",
    "selective_mechanism": True,
    "hardware_efficient": True,
    "memory_efficient": True
}

print("🐍 Supreme System V5 - Mamba SSM Module Loaded")
print(f"   Complexity: {PERFORMANCE_SPECS['complexity']}")
print(f"   Context Length: {PERFORMANCE_SPECS['context_length']}")
print("🚀 Revolutionary Linear Complexity Ready!")
