"""
Strategy Registry and Factory Pattern Implementation

Features:
- Strategy registration and discovery
- Factory pattern for strategy instantiation
- Liskov Substitution Principle compliance
- Strategy metadata and validation
- Plugin architecture support
"""

import inspect
import logging
from typing import Dict, List, Any, Optional, Type, Callable
from abc import ABC, abstractmethod
from pathlib import Path
import importlib
import pkgutil

from .base_strategy import BaseStrategy


logger = logging.getLogger(__name__)


class StrategyRegistry:
    """
    Registry for trading strategies with factory pattern implementation.

    Supports:
    - Dynamic strategy registration
    - Metadata tracking
    - Validation
    - Plugin loading
    - Strategy discovery
    """

    def __init__(self):
        self._strategies: Dict[str, Dict[str, Any]] = {}
        self._strategy_classes: Dict[str, Type[BaseStrategy]] = {}
        self._logger = logging.getLogger(__name__)

    def register_strategy(self,
                         name: str,
                         strategy_class: Type[BaseStrategy],
                         metadata: Optional[Dict[str, Any]] = None,
                         parameters: Optional[Dict[str, Any]] = None) -> bool:
        """
        Register a strategy in the registry.

        Args:
            name: Strategy name (unique identifier)
            strategy_class: Strategy class
            metadata: Strategy metadata
            parameters: Default parameters

        Returns:
            Success status
        """
        try:
            # Validate strategy class
            if not self._validate_strategy_class(strategy_class):
                self._logger.error(f"Invalid strategy class for {name}")
                return False

            # Create registry entry
            entry = {
                'name': name,
                'class': strategy_class,
                'metadata': metadata or {},
                'parameters': parameters or {},
                'module': strategy_class.__module__,
                'version': getattr(strategy_class, '__version__', '1.0.0'),
                'description': getattr(strategy_class, '__doc__', '').strip()
            }

            # Register strategy
            self._strategies[name] = entry
            self._strategy_classes[name] = strategy_class

            self._logger.info(f"Registered strategy: {name}")
            return True

        except Exception as e:
            self._logger.error(f"Failed to register strategy {name}: {e}")
            return False

    def unregister_strategy(self, name: str) -> bool:
        """
        Unregister a strategy from the registry.

        Args:
            name: Strategy name

        Returns:
            Success status
        """
        if name in self._strategies:
            del self._strategies[name]
            del self._strategy_classes[name]
            self._logger.info(f"Unregistered strategy: {name}")
            return True

        return False

    def create_strategy(self, name: str, **kwargs) -> Optional[BaseStrategy]:
        """
        Create a strategy instance using the factory pattern.

        Args:
            name: Strategy name
            **kwargs: Strategy parameters

        Returns:
            Strategy instance or None if failed
        """
        try:
            if name not in self._strategy_classes:
                self._logger.error(f"Strategy not found: {name}")
                return None

            strategy_class = self._strategy_classes[name]
            strategy_entry = self._strategies[name]

            # Merge default parameters with provided parameters
            params = strategy_entry['parameters'].copy()
            params.update(kwargs)

            # Create instance
            strategy = strategy_class(**params)

            # Validate instance
            if not self._validate_strategy_instance(strategy):
                self._logger.error(f"Invalid strategy instance: {name}")
                return None

            self._logger.info(f"Created strategy instance: {name}")
            return strategy

        except Exception as e:
            self._logger.error(f"Failed to create strategy {name}: {e}")
            return None

    def get_strategy_info(self, name: str) -> Optional[Dict[str, Any]]:
        """
        Get strategy information.

        Args:
            name: Strategy name

        Returns:
            Strategy information or None
        """
        return self._strategies.get(name)

    def list_strategies(self, filter_by: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        List all registered strategies.

        Args:
            filter_by: Filter criteria ('type', 'author', etc.)

        Returns:
            List of strategy information
        """
        strategies = list(self._strategies.values())

        if filter_by:
            # Implement filtering logic if needed
            pass

        return strategies

    def discover_strategies(self, package_path: str = 'src.strategies') -> int:
        """
        Auto-discover and register strategies from package.

        Args:
            package_path: Package path to scan

        Returns:
            Number of strategies discovered
        """
        discovered_count = 0

        try:
            # Import package
            package = importlib.import_module(package_path)

            # Iterate through modules
            for _, module_name, _ in pkgutil.iter_modules(package.__path__):
                try:
                    # Import module
                    module = importlib.import_module(f"{package_path}.{module_name}")

                    # Find strategy classes
                    for name, obj in inspect.getmembers(module):
                        if (inspect.isclass(obj) and
                            issubclass(obj, BaseStrategy) and
                            obj != BaseStrategy):

                            # Register strategy
                            if self.register_strategy(module_name, obj):
                                discovered_count += 1

                except Exception as e:
                    self._logger.warning(f"Failed to load module {module_name}: {e}")

        except Exception as e:
            self._logger.error(f"Failed to discover strategies: {e}")

        self._logger.info(f"Discovered {discovered_count} strategies")
        return discovered_count

    def _validate_strategy_class(self, strategy_class: Type[BaseStrategy]) -> bool:
        """
        Validate strategy class implementation.

        Args:
            strategy_class: Strategy class to validate

        Returns:
            Validation result
        """
        try:
            # Check inheritance
            if not issubclass(strategy_class, BaseStrategy):
                return False

            # Check required methods
            required_methods = ['generate_signal']
            for method in required_methods:
                if not hasattr(strategy_class, method):
                    return False

                # Check method signature
                sig = inspect.signature(getattr(strategy_class, method))
                if len(sig.parameters) < 2:  # self + at least one parameter
                    return False

            # Check constructor
            init_sig = inspect.signature(strategy_class.__init__)
            # Should accept parameters beyond self

            return True

        except Exception as e:
            self._logger.error(f"Strategy class validation failed: {e}")
            return False

    def _validate_strategy_instance(self, strategy: BaseStrategy) -> bool:
        """
        Validate strategy instance.

        Args:
            strategy: Strategy instance to validate

        Returns:
            Validation result
        """
        try:
            # Check required attributes
            required_attrs = ['name', 'parameters']
            for attr in required_attrs:
                if not hasattr(strategy, attr):
                    return False

            # Test basic functionality
            test_data = pd.DataFrame({
                'timestamp': pd.date_range('2024-01-01', periods=10, freq='1H'),
                'open': [100.0] * 10,
                'high': [105.0] * 10,
                'low': [95.0] * 10,
                'close': [102.0] * 10,
                'volume': [1000] * 10
            })

            # Should not raise exception
            signal = strategy.generate_signal(test_data)

            # Signal should be properly structured
            if not isinstance(signal, dict):
                return False

            required_keys = ['action', 'symbol', 'strength', 'confidence']
            for key in required_keys:
                if key not in signal:
                    return False

            return True

        except Exception as e:
            self._logger.error(f"Strategy instance validation failed: {e}")
            return False


class StrategyFactory:
    """
    Factory pattern implementation for strategy creation.

    Provides a clean interface for strategy instantiation with
    validation, caching, and error handling.
    """

    def __init__(self, registry: Optional[StrategyRegistry] = None):
        self.registry = registry or StrategyRegistry()
        self._cache: Dict[str, BaseStrategy] = {}
        self._logger = logging.getLogger(__name__)

    def create(self, strategy_name: str, **kwargs) -> Optional[BaseStrategy]:
        """
        Create strategy instance with caching and error handling.

        Args:
            strategy_name: Name of strategy to create
            **kwargs: Strategy parameters

        Returns:
            Strategy instance or None
        """
        cache_key = f"{strategy_name}_{hash(frozenset(kwargs.items()))}"

        # Check cache first
        if cache_key in self._cache:
            self._logger.debug(f"Using cached strategy: {strategy_name}")
            return self._cache[cache_key]

        # Create new instance
        strategy = self.registry.create_strategy(strategy_name, **kwargs)

        if strategy:
            # Cache successful creation
            self._cache[cache_key] = strategy
            self._logger.info(f"Created and cached strategy: {strategy_name}")

        return strategy

    def create_multiple(self, strategy_configs: List[Dict[str, Any]]) -> List[BaseStrategy]:
        """
        Create multiple strategies from configuration list.

        Args:
            strategy_configs: List of strategy configurations

        Returns:
            List of strategy instances
        """
        strategies = []

        for config in strategy_configs:
            try:
                strategy_name = config.pop('name')
                strategy = self.create(strategy_name, **config)
                if strategy:
                    strategies.append(strategy)
            except Exception as e:
                self._logger.error(f"Failed to create strategy from config {config}: {e}")

        return strategies

    def clear_cache(self):
        """Clear strategy cache."""
        self._cache.clear()
        self._logger.info("Strategy cache cleared")


# Global registry instance
strategy_registry = StrategyRegistry()
strategy_factory = StrategyFactory(strategy_registry)


def get_strategy_registry() -> StrategyRegistry:
    """Get global strategy registry instance."""
    return strategy_registry


def get_strategy_factory() -> StrategyFactory:
    """Get global strategy factory instance."""
    return strategy_factory


def register_strategy(name: str, strategy_class: Type[BaseStrategy], **kwargs) -> bool:
    """Convenience function to register a strategy."""
    return strategy_registry.register_strategy(name, strategy_class, **kwargs)


def create_strategy(name: str, **kwargs) -> Optional[BaseStrategy]:
    """Convenience function to create a strategy."""
    return strategy_factory.create(name, **kwargs)


# Auto-discover built-in strategies
def initialize_builtin_strategies():
    """Initialize and register built-in strategies."""
    try:
        # Import strategy modules
        from . import momentum, trend_following, breakout, moving_average, mean_reversion

        # Register strategies
        strategy_registry.register_strategy(
            'momentum',
            momentum.MomentumStrategy,
            metadata={'type': 'momentum', 'description': 'Momentum-based trading strategy'},
            parameters={
                'short_period': 12,
                'long_period': 26,
                'signal_period': 9,
                'roc_period': 10,
                'trend_threshold': 0.02
            }
        )

        strategy_registry.register_strategy(
            'trend_following',
            trend_following.TrendFollowingAgent,
            metadata={'type': 'trend', 'description': 'Trend following strategy'},
            parameters={
                'short_window': 20,
                'long_window': 50,
                'adx_period': 14,
                'adx_threshold': 25,
                'rsi_period': 14,
                'rsi_overbought': 70,
                'rsi_oversold': 30,
                'volume_ma_period': 20,
                'stop_loss_pct': 0.02,
                'take_profit_pct': 0.05
            }
        )

        strategy_registry.register_strategy(
            'breakout',
            breakout.BreakoutStrategy,
            metadata={'type': 'breakout', 'description': 'Breakout trading strategy'},
            parameters={
                'lookback_period': 20,
                'breakout_threshold': 0.05,
                'volume_confirmation': True,
                'min_volume_ratio': 1.5
            }
        )

        strategy_registry.register_strategy(
            'moving_average',
            moving_average.MovingAverageStrategy,
            metadata={'type': 'mean_reversion', 'description': 'Moving average crossover strategy'},
            parameters={
                'short_window': 20,
                'long_window': 50,
                'signal_strength_threshold': 0.02
            }
        )

        strategy_registry.register_strategy(
            'mean_reversion',
            mean_reversion.MeanReversionStrategy,
            metadata={'type': 'mean_reversion', 'description': 'Mean reversion strategy'},
            parameters={
                'lookback_period': 20,
                'entry_threshold': 2.0,
                'exit_threshold': 0.5,
                'max_holding_period': 10
            }
        )

        logger.info("Initialized built-in strategies")

    except Exception as e:
        logger.error(f"Failed to initialize built-in strategies: {e}")


# Initialize on import
initialize_builtin_strategies()


if __name__ == "__main__":
    # Test strategy registry
    registry = get_strategy_registry()
    factory = get_strategy_factory()

    print("Registered strategies:")
    for strategy_info in registry.list_strategies():
        print(f"- {strategy_info['name']}: {strategy_info['metadata'].get('description', 'No description')}")

    # Test strategy creation
    print("\nTesting strategy creation:")
    momentum_strategy = factory.create('momentum')
    if momentum_strategy:
        print("✅ Successfully created momentum strategy")
    else:
        print("❌ Failed to create momentum strategy")
