#!/usr/bin/env python3
"""
Supreme System V5 - Configuration Management

Centralized configuration management with environment variable support.
"""

import os
from pathlib import Path
from typing import Dict, Any, Optional, Union
import json
import yaml
import logging

try:
    from ..utils.constants import DEFAULT_CONFIG, LOGGING_CONFIG
    from ..utils.helpers import setup_logging
except ImportError:
    try:
        from utils.constants import DEFAULT_CONFIG, LOGGING_CONFIG
        from utils.helpers import setup_logging
    except ImportError:
        # Fallback defaults for testing
        DEFAULT_CONFIG = {}
        LOGGING_CONFIG = {}
        def setup_logging(*args, **kwargs):
            pass


class Config:
    """
    Centralized configuration management for Supreme System V5.

    Supports:
    - Environment variables
    - YAML configuration files
    - Default fallbacks
    - Type validation
    """

    def __init__(self, config_file: Optional[str] = None, setup_logging: bool = True):
        """
        Initialize configuration manager.

        Args:
            config_file: Path to YAML configuration file (optional)
            setup_logging: Whether to setup logging automatically
        """
        self._config = {}
        self._load_defaults()
        self._load_from_env()
        if config_file:
            self._load_from_file(config_file)

        # Setup logging if requested
        if setup_logging:
            self._setup_logging()

    def _load_defaults(self):
        """Load default configuration values."""
        # Use the comprehensive default config from constants
        self._config.update(DEFAULT_CONFIG.copy())

    def _setup_logging(self):
        """Setup logging based on configuration."""
        log_config = self._config.get('monitoring', {}).get('log_level', 'INFO')
        log_file = self._config.get('system', {}).get('log_file', 'logs/supreme_system.log')

        # Setup logging using utility function
        setup_logging(
            level=log_config,
            log_file=log_file
        )

    def _load_from_env(self):
        """Load configuration from environment variables."""
        # Binance API - ensure the section exists
        if 'binance' not in self._config:
            self._config['binance'] = {}
        self._config['binance']['api_key'] = os.getenv('BINANCE_API_KEY')
        self._config['binance']['api_secret'] = os.getenv('BINANCE_API_SECRET')
        self._config['binance']['testnet'] = os.getenv('BINANCE_TESTNET', 'true').lower() == 'true'

        # Trading - ensure the section exists
        if 'trading' not in self._config:
            self._config['trading'] = {}
        if 'MAX_POSITION_SIZE' in os.environ:
            self._config['trading']['max_position_size'] = float(os.getenv('MAX_POSITION_SIZE'))
        if 'STOP_LOSS_PCT' in os.environ:
            self._config['trading']['stop_loss_pct'] = float(os.getenv('STOP_LOSS_PCT'))

        # Logging - ensure the section exists
        if 'logging' not in self._config:
            self._config['logging'] = {}
        if 'LOG_LEVEL' in os.environ:
            self._config['logging']['level'] = os.getenv('LOG_LEVEL')

    def _load_from_file(self, config_file: str):
        """Load configuration from YAML file."""
        try:
            with open(config_file, 'r') as f:
                file_config = yaml.safe_load(f)
                self._merge_config(file_config)
        except FileNotFoundError:
            print(f"Warning: Configuration file {config_file} not found")
        except yaml.YAMLError as e:
            print(f"Error parsing configuration file: {e}")

    def _merge_config(self, new_config: Dict[str, Any], path: str = ""):
        """Recursively merge configuration dictionaries."""
        for key, value in new_config.items():
            full_path = f"{path}.{key}" if path else key

            if isinstance(value, dict) and key in self._config:
                if isinstance(self._config[key], dict):
                    self._merge_config(value, full_path)
                else:
                    self._config[key] = value
            else:
                self._config[key] = value

    def get(self, key: str, default: Any = None) -> Any:
        """
        Get configuration value by dot-separated key.

        Args:
            key: Dot-separated configuration key (e.g., 'binance.api_key')
            default: Default value if key not found

        Returns:
            Configuration value or default
        """
        keys = key.split('.')
        value = self._config

        try:
            for k in keys:
                value = value[k]
            return value
        except (KeyError, TypeError):
            return default

    def set(self, key: str, value: Any):
        """Set configuration value by dot-separated key."""
        keys = key.split('.')
        config = self._config

        for k in keys[:-1]:
            if k not in config:
                config[k] = {}
            config = config[k]

        config[keys[-1]] = value

    def save_to_file(self, config_file: str):
        """Save current configuration to YAML file."""
        os.makedirs(os.path.dirname(config_file), exist_ok=True)
        with open(config_file, 'w') as f:
            yaml.dump(self._config, f, default_flow_style=False)

    def get_section(self, section: str) -> Dict[str, Any]:
        """Get entire configuration section."""
        return self._config.get(section, {})

    def validate_api_credentials(self) -> bool:
        """Validate that API credentials are configured."""
        api_key = self.get('binance.api_key')
        api_secret = self.get('binance.api_secret')

        if not api_key or not api_secret:
            return False

        return len(api_key) > 0 and len(api_secret) > 0

    def create_directories(self):
        """Create necessary directories from configuration."""
        directories = [
            self.get('data.cache_dir'),
            self.get('data.historical_dir'),
            self.get('logging.file', '').rsplit('/', 1)[0] if '/' in self.get('logging.file', '') else 'logs'
        ]

        for directory in directories:
            if directory:
                Path(directory).mkdir(parents=True, exist_ok=True)

    def __str__(self) -> str:
        """String representation of configuration."""
        return f"Config(sections={list(self._config.keys())})"

    def __repr__(self) -> str:
        return self.__str__()


# Global configuration instance
config = Config()


def load_config(config_file: Optional[str] = None) -> Config:
    """
    Load configuration with optional file.

    Args:
        config_file: Path to YAML configuration file

    Returns:
        Config instance
    """
    global config
    config = Config(config_file)
    config.create_directories()
    return config


def get_config() -> Config:
    """Get global configuration instance."""
    return config
