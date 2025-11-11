"""
Supreme System V5 - Configuration Module

Centralized configuration management for all system components.
"""

from .config import Config, get_config, load_config

__all__ = ['Config', 'get_config', 'load_config']
