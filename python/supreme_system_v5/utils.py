# python/supreme_system_v5/utils.py
import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


def get_logger(name: str) -> logging.Logger:
    """Returns a logger instance with a consistent format."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    return logger


@dataclass
class Config:
    """Minimal configuration dataclass."""

    setting1: str = "default_value"
    setting2: int = 123
