"""
🤖 Supreme System V5 - Foundation Models Module
Zero-shot time series prediction with state-of-the-art models

Supported Models:
- TimesFM (Google Research)
- Chronos (Amazon Science) 
- PatchTST
- ForecastPFN
"""

__version__ = "5.0.0"
__author__ = "Supreme System V5 Team"

from .predictor import FoundationModelPredictor
from .engines import TimesFMEngine, ChronosEngine

# Export main classes
FoundationModelEngine = FoundationModelPredictor  # Alias

__all__ = [
    "FoundationModelPredictor",
    "FoundationModelEngine",
    "TimesFMEngine", 
    "ChronosEngine"
]