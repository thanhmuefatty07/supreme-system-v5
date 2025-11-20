#!/usr/bin/env python3
"""
Simple tests for Strategy Registry
"""

import pytest
from unittest.mock import MagicMock

try:
    from src.strategies.strategy_registry import StrategyRegistry
    from src.strategies.base_strategy import BaseStrategy
except ImportError:
    pytest.skip("StrategyRegistry not available", allow_module_level=True)


class TestStrategyRegistry:
    """Basic tests for StrategyRegistry class."""

    @pytest.fixture
    def registry(self):
        """Create StrategyRegistry instance."""
        return StrategyRegistry()

    def test_initialization(self, registry):
        """Test registry initialization."""
        assert registry is not None
        assert hasattr(registry, '_strategies')

    def test_register_strategy(self, registry):
        """Test strategy registration."""
        class MockStrategy(BaseStrategy):
            """Mock strategy for testing."""
            def generate_signal(self, data):
                return None

        result = registry.register_strategy("MyStrat", MockStrategy, metadata={"test": True})
        assert result is True
        info = registry.get_strategy_info("MyStrat")
        assert info is not None
        assert info['class'] == MockStrategy

    def test_get_invalid_strategy(self, registry):
        """Test getting invalid strategy."""
        info = registry.get_strategy_info("Invalid")
        assert info is None

    def test_list_strategies(self, registry):
        """Test listing strategies."""
        class MockA(BaseStrategy):
            """Mock strategy A."""
            def generate_signal(self, data):
                return None

        class MockB(BaseStrategy):
            """Mock strategy B."""
            def generate_signal(self, data):
                return None

        registry.register_strategy("A", MockA, metadata={"test": True})
        registry.register_strategy("B", MockB, metadata={"test": True})

        strategies = registry.list_strategies()
        assert len(strategies) >= 2
        strategy_names = [s['name'] for s in strategies]
        assert "A" in strategy_names
        assert "B" in strategy_names

    def test_register_duplicate(self, registry):
        """Test registering duplicate strategy."""
        class Mock1(BaseStrategy):
            """Mock strategy 1."""
            def generate_signal(self, data):
                return None

        class Mock2(BaseStrategy):
            """Mock strategy 2."""
            def generate_signal(self, data):
                return None

        registry.register_strategy("Test", Mock1, metadata={"version": 1})
        registry.register_strategy("Test", Mock2, metadata={"version": 2})  # Should overwrite
        info = registry.get_strategy_info("Test")
        assert info['class'] == Mock2

    def test_remove_strategy(self, registry):
        """Test removing strategy."""
        class MockStrategy(BaseStrategy):
            """Mock strategy for removal test."""
            def generate_signal(self, data):
                return None

        registry.register_strategy("RemoveMe", MockStrategy, metadata={"test": True})
        assert "RemoveMe" in registry._strategy_classes

        # Test remove if method exists
        if hasattr(registry, 'unregister_strategy'):
            registry.unregister_strategy("RemoveMe")
            info = registry.get_strategy_info("RemoveMe")
            assert info is None
