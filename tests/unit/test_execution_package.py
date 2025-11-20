#!/usr/bin/env python3
"""
Comprehensive Tests for Execution Package

Tests cover all components: impact analysis, algorithmic orders, and smart router.
Target: 100% coverage for the execution module.
"""

import pytest
import asyncio
from unittest.mock import AsyncMock, MagicMock
from datetime import datetime

# Import all execution components
from src.execution.impact_analysis import (
    calculate_slippage,
    is_liquidity_sufficient,
    estimate_market_impact,
    get_optimal_execution_time
)
from src.execution.algo_orders import (
    split_iceberg_order,
    generate_twap_schedule,
    generate_vwap_schedule,
    calculate_execution_quality
)
from src.execution.router import SmartRouter, ExecutionResult


class TestImpactAnalysis:
    """Test market impact analysis functions."""

    def test_calculate_slippage_basic(self):
        """Test basic slippage calculation."""
        # Simple order book: 10 @ 100
        depth = [{'price': 100.0, 'amount': 10.0}]

        # Order smaller than available liquidity
        slippage = calculate_slippage(5.0, depth)
        assert slippage == 0.0  # No slippage

    def test_calculate_slippage_with_impact(self):
        """Test slippage with market impact."""
        # Order book with multiple levels
        depth = [
            {'price': 100.0, 'amount': 10.0},  # Level 1
            {'price': 101.0, 'amount': 10.0},  # Level 2
        ]

        # Order takes from both levels
        slippage = calculate_slippage(15.0, depth)
        expected_avg_price = (10 * 100 + 5 * 101) / 15  # 100.33
        expected_slippage = abs(100.33 - 100) / 100  # 0.0033
        assert abs(slippage - expected_slippage) < 0.0001

    def test_is_liquidity_sufficient(self):
        """Test liquidity sufficiency check."""
        depth = [
            {'price': 100.0, 'amount': 10.0},
            {'price': 101.0, 'amount': 10.0},
        ]

        # Sufficient liquidity
        sufficient, reason = is_liquidity_sufficient(5.0, depth, 0.01)
        assert sufficient is True
        assert "sufficient" in reason.lower()

        # Insufficient due to total liquidity (25 > 20 available)
        sufficient, reason = is_liquidity_sufficient(25.0, depth, 0.001)
        assert sufficient is False
        assert "liquidity" in reason.lower()

    def test_estimate_market_impact(self):
        """Test market impact estimation."""
        order_size = 1000
        avg_daily_volume = 10000

        impact = estimate_market_impact(order_size, avg_daily_volume)
        # 10% of daily volume should have some impact
        assert impact > 0
        assert impact < 1.0  # Less than 100%

        # Zero volume should give infinite impact
        infinite_impact = estimate_market_impact(100, 0)
        assert infinite_impact == float('inf')

    def test_get_optimal_execution_time(self):
        """Test optimal execution time calculation."""
        order_size = 1000
        avg_daily_volume = 10000  # 10k daily = ~7 per minute

        time_minutes = get_optimal_execution_time(order_size, avg_daily_volume)
        assert time_minutes >= 1
        assert time_minutes <= 24 * 60  # Max 1 day


class TestAlgoOrders:
    """Test algorithmic order functions."""

    def test_split_iceberg_order(self):
        """Test iceberg order splitting."""
        total_size = 100.0

        # Split by number of chunks
        chunks = split_iceberg_order(total_size, num_chunks=4)
        assert len(chunks) == 4
        assert abs(sum(chunks) - total_size) < 0.001

        # Split by max chunk size
        chunks = split_iceberg_order(total_size, max_chunk_size=30, add_noise=False)
        assert len(chunks) == 4  # 100 / 30 = 3.33 -> 4 chunks
        assert all(chunk <= 30 for chunk in chunks)
        assert abs(sum(chunks) - total_size) < 0.001

        # No splitting needed
        chunks = split_iceberg_order(10.0, max_chunk_size=50)
        assert len(chunks) == 1
        assert chunks[0] == 10.0

    def test_generate_twap_schedule(self):
        """Test TWAP schedule generation."""
        total_size = 100.0
        duration = 20  # minutes
        interval = 5   # minutes

        schedule = generate_twap_schedule(total_size, duration, interval)

        assert len(schedule) == 4  # 20 / 5 = 4 intervals
        assert all(slot['size'] == 25.0 for slot in schedule)  # 100 / 4 = 25

        # Check timing
        for i, slot in enumerate(schedule):
            assert slot['sequence'] == i + 1
            assert slot['cumulative_size'] == 25.0 * (i + 1)

    def test_generate_vwap_schedule(self):
        """Test VWAP schedule generation."""
        total_size = 100.0
        volume_profile = [1, 2, 3, 2, 1]  # Weighted towards middle

        schedule = generate_vwap_schedule(total_size, volume_profile)

        assert len(schedule) == 5
        total_weight = sum(slot['weight'] for slot in schedule)
        assert abs(total_weight - 1.0) < 0.001  # Weights should sum to 1

        total_size_check = sum(slot['size'] for slot in schedule)
        assert abs(total_size_check - total_size) < 0.001

    def test_calculate_execution_quality(self):
        """Test execution quality calculation."""
        benchmark_price = 100.0

        # Perfect execution
        executed_orders = [
            {'price': 100.0, 'size': 10.0},
            {'price': 100.0, 'size': 10.0}
        ]

        quality = calculate_execution_quality(executed_orders, benchmark_price)
        assert quality['quality_score'] > 40  # Good score for perfect execution
        assert quality['metrics']['price_improvement'] == 0.0  # No improvement

        # Better than benchmark
        executed_orders_good = [
            {'price': 99.0, 'size': 10.0},  # Better price
            {'price': 99.5, 'size': 10.0}
        ]

        quality_good = calculate_execution_quality(executed_orders_good, benchmark_price)
        assert quality_good['metrics']['price_improvement'] > 0  # Positive improvement

        # Empty orders
        quality_empty = calculate_execution_quality([], benchmark_price)
        assert quality_empty['quality_score'] == 0.0


class TestSmartRouter:
    """Test Smart Router integration."""

    @pytest.fixture
    def mock_exchange(self):
        """Create mock exchange client."""
        exchange = MagicMock()

        # Mock order book
        exchange.fetch_order_book = AsyncMock(return_value={
            'asks': [(100.0, 10.0), (101.0, 10.0)],
            'bids': [(99.0, 10.0), (98.0, 10.0)]
        })

        # Mock order creation
        exchange.create_order = AsyncMock(return_value={
            'id': 'test_order_123',
            'status': 'filled',
            'amount': 10.0,
            'price': 100.0
        })

        return exchange

    @pytest.fixture
    def router(self, mock_exchange):
        """Create SmartRouter instance."""
        config = {
            "liquidity_check": False,  # Disable for testing
            "impact_analysis": False,  # Disable for testing
            "max_slippage": 0.005,
            "iceberg_threshold": 1000,
            "max_chunk_size": 500,
            "execution_timeout": 30,
            "twap_enabled": True  # Enable TWAP
        }
        return SmartRouter(mock_exchange, config)

    def test_router_initialization(self, router):
        """Test router initialization."""
        assert router.exchange is not None
        assert router.config['max_slippage'] == 0.005
        assert router.config['iceberg_threshold'] == 1000

    @pytest.mark.asyncio
    async def test_execute_standard_order(self, router):
        """Test standard order execution."""
        result = await router.execute_order('BTC/USDT', 'buy', 5.0)

        assert result.status == 'SUCCESS'
        assert result.order_id == 'test_order_123'
        assert result.executed_size == 10.0
        assert result.avg_price == 100.0

    @pytest.mark.asyncio
    async def test_execute_large_order_iceberg(self, router):
        """Test large order triggers iceberg execution."""
        # Large order should trigger iceberg
        result = await router.execute_order('BTC/USDT', 'buy', 1500.0)

        assert result.status == 'SUCCESS'
        assert 'Iceberg execution' in result.reason

    @pytest.mark.asyncio
    async def test_execute_twap_order(self, router):
        """Test TWAP order execution."""
        result = await router.execute_order('BTC/USDT', 'buy', 100.0, order_type='twap')

        assert result.status == 'SUCCESS'
        assert 'TWAP execution' in result.reason

    @pytest.mark.asyncio
    async def test_insufficient_liquidity_rejection(self, router, mock_exchange):
        """Test order rejection due to insufficient liquidity."""
        # Create router with liquidity check enabled
        config = {
            "liquidity_check": True,  # Enable for this test
            "impact_analysis": False,
            "max_slippage": 0.005,
            "iceberg_threshold": 1000,
            "max_chunk_size": 500,
            "execution_timeout": 30,
            "twap_enabled": True
        }
        test_router = SmartRouter(mock_exchange, config)

        # Mock very limited liquidity
        mock_exchange.fetch_order_book = AsyncMock(return_value={
            'asks': [(100.0, 1.0)],  # Only 1 unit available
            'bids': [(99.0, 1.0)]
        })

        # Large order should be rejected
        result = await test_router.execute_order('BTC/USDT', 'buy', 100.0)

        assert result.status == 'REJECTED'
        assert 'liquidity' in result.reason.lower()

    def test_execution_result(self):
        """Test ExecutionResult container."""
        result = ExecutionResult('SUCCESS', 'order123', 10.0, 100.0, 'Test')

        assert result.status == 'SUCCESS'
        assert result.order_id == 'order123'
        assert result.executed_size == 10.0
        assert result.avg_price == 100.0
        assert result.reason == 'Test'

        # Test to_dict
        result_dict = result.to_dict()
        assert result_dict['status'] == 'SUCCESS'
        assert result_dict['order_id'] == 'order123'

    def test_get_execution_stats(self, router):
        """Test execution statistics."""
        stats = router.get_execution_stats()

        # Initially empty
        assert stats['total_orders'] == 0
        assert stats['success_rate'] == 0.0


class TestExecutionIntegration:
    """Integration tests for execution components."""

    def test_full_execution_workflow(self):
        """Test complete execution workflow simulation."""
        # Test slippage calculation
        depth = [{'price': 100.0, 'amount': 10.0}, {'price': 101.0, 'amount': 10.0}]
        slippage = calculate_slippage(15.0, depth)
        assert slippage > 0

        # Test liquidity check
        sufficient, _ = is_liquidity_sufficient(15.0, depth, 0.01)
        assert sufficient is True

        # Test iceberg splitting
        chunks = split_iceberg_order(100.0, max_chunk_size=30)
        assert len(chunks) >= 3
        assert abs(sum(chunks) - 100.0) < 0.001

        # Test TWAP schedule
        schedule = generate_twap_schedule(100.0, 20, 5)
        assert len(schedule) >= 3

    def test_edge_cases(self):
        """Test edge cases and error conditions."""
        # Empty depth
        slippage = calculate_slippage(10.0, [])
        assert slippage == 0.0

        sufficient, reason = is_liquidity_sufficient(10.0, [], 0.01)
        assert sufficient is False

        # Zero size
        chunks = split_iceberg_order(0.0)
        assert len(chunks) == 0

        schedule = generate_twap_schedule(0.0, 10)
        assert len(schedule) == 0

        # Invalid inputs
        impact = estimate_market_impact(100, 0)
        assert impact == float('inf')
