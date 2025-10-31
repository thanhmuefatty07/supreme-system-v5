# python/supreme_system_v5/strategies.py
from .utils import get_logger
from typing import Dict, Any

logger = get_logger(__name__)


class Strategy:
    """Base class for trading strategies."""

    def __init__(self, name: str):
        self.name = name
        logger.info(f"Strategy '{self.name}' base initialized.")

    def generate_signal(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Generates a mock trading signal."""
        logger.info(f"Generating mock signal for {self.name} with data: {data}")
        return {"strategy": self.name, "signal": "BUY", "confidence": 0.7}
