"""
🧠 Supreme System V5 - Neuromorphic Computing Module
Production neuromorphic processing for trading applications

Features:
- Spiking neural networks for pattern recognition
- Event-driven processing architecture  
- Hardware-aware optimization
- Real-time market data analysis
- Ultra-low power consumption
"""

__version__ = "5.0.0"
__author__ = "Supreme System V5 Team"

from .processor import NeuromorphicProcessor, NeuromorphicConfig

# Export main classes
NeuromorphicEngine = NeuromorphicProcessor  # Alias for backward compatibility

__all__ = [
    "NeuromorphicProcessor",
    "NeuromorphicEngine", 
    "NeuromorphicConfig"
]