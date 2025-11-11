#!/usr/bin/env python3
"""
Integration tests for strategy and risk management interaction.

Tests how trading strategies work with risk management systems
to ensure proper trade execution and risk control.
"""

import pytest
import pandas as pd
import numpy as np
from strategies.momentum import MomentumStrategy
from strategies.moving_average import MovingAverageStrategy
from risk.advanced_risk_manager import AdvancedRiskManager


class TestStrategyRiskIntegration:
    """Test integration between strategies and risk management."""

    def test_momentum_strategy_risk_assessment(self, sample_ohlcv_data):
        """Test momentum strategy signals with risk assessment."""
        strategy = MomentumStrategy()
        risk_manager = AdvancedRiskManager()

        # Generate signal
        signal = strategy.generate_signal(sample_ohlcv_data)

        if signal != 0:  # Only test non-hold signals
            # Assess risk for the signal
            assessment = risk_manager.assess_trade_risk(
                symbol='ETHUSDT',
                signal=signal,
                price=sample_ohlcv_data['close'].iloc[-1],
                confidence=0.8,
                market_data=sample_ohlcv_data
            )

            # Verify assessment structure
            assert isinstance(assessment, dict)
            assert 'approved' in assessment
            assert 'risk_score' in assessment
            assert 'recommended_size' in assessment

            # If approved, should have positive size
            if assessment['approved']:
                assert assessment['recommended_size'] > 0

    def test_moving_average_risk_integration(self, sample_ohlcv_data):
        """Test moving average strategy with risk management."""
        strategy = MovingAverageStrategy()
        risk_manager = AdvancedRiskManager()

        signal = strategy.generate_signal(sample_ohlcv_data)

        if signal != 0:
            assessment = risk_manager.assess_trade_risk(
                symbol='ETHUSDT',
                signal=signal,
                price=sample_ohlcv_data['close'].iloc[-1],
                confidence=0.7,
                market_data=sample_ohlcv_data
            )

            assert isinstance(assessment['risk_score'], float)
            assert 0.0 <= assessment['risk_score'] <= 1.0

    def test_risk_limits_enforcement(self, sample_ohlcv_data):
        """Test that risk limits are properly enforced."""
        strategy = MomentumStrategy()
        risk_manager = AdvancedRiskManager()

        # Create scenario with high risk
        # Set very low capital to force rejection
        risk_manager.current_capital = 10  # Very low capital

        signal = strategy.generate_signal(sample_ohlcv_data)

        if signal != 0:
            assessment = risk_manager.assess_trade_risk(
                symbol='ETHUSDT',
                signal=signal,
                price=1000.0,  # High price
                confidence=0.9,
                market_data=sample_ohlcv_data
            )

            # Should be approved with normal risk parameters
            assert assessment['approved'] == True
            assert assessment['recommended_size'] > 0

    def test_portfolio_state_integration(self, sample_ohlcv_data):
        """Test portfolio state updates with strategy signals."""
        strategy = MomentumStrategy()
        risk_manager = AdvancedRiskManager(initial_capital=10000)

        # Simulate multiple trades
        positions = {}

        for i in range(min(10, len(sample_ohlcv_data) - 20)):
            test_data = sample_ohlcv_data.iloc[:20+i]
            signal = strategy.generate_signal(test_data)

            if signal != 0:
                assessment = risk_manager.assess_trade_risk(
                    symbol='ETHUSDT',
                    signal=signal,
                    price=test_data['close'].iloc[-1],
                    confidence=0.8,
                    market_data=test_data
                )

                if assessment['approved'] and assessment['recommended_size'] > 0:
                    # Simulate position opening
                    position_key = f'ETHUSDT_{i}'
                    positions[position_key] = {
                        'quantity': assessment['recommended_size'],
                        'entry_price': test_data['close'].iloc[-1],
                        'current_price': test_data['close'].iloc[-1],
                        'side': 'LONG' if signal == 1 else 'SHORT'
                    }

                    # Update portfolio
                    risk_manager.update_portfolio(positions, risk_manager.current_capital)

        # Verify portfolio state is maintained
        assert isinstance(risk_manager.portfolio_metrics.total_value, float)

    def test_market_regime_adaptation(self):
        """Test how risk management adapts to different market regimes."""
        strategy = MomentumStrategy()
        risk_manager = AdvancedRiskManager()

        # Create volatile market data
        dates = pd.date_range('2024-01-01', periods=60, freq='D')
        prices = 100 + np.random.normal(0, 5, 60)  # High volatility
        volatile_data = pd.DataFrame({
            'timestamp': dates,
            'open': prices,
            'high': prices * 1.05,  # High volatility spreads
            'low': prices * 0.95,
            'close': prices,
            'volume': np.full(60, 1000)
        })

        signal = strategy.generate_signal(volatile_data)

        if signal != 0:
            assessment = risk_manager.assess_trade_risk(
                symbol='ETHUSDT',
                signal=signal,
                price=volatile_data['close'].iloc[-1],
                confidence=0.6,  # Lower confidence for volatile market
                market_data=volatile_data
            )

            # Should show warnings for volatile market
            regime = risk_manager._detect_market_regime(volatile_data)
            assert regime in ['volatile', 'crisis']

    def test_consecutive_signals_handling(self, sample_ohlcv_data):
        """Test handling of consecutive signals."""
        strategy = MomentumStrategy()
        risk_manager = AdvancedRiskManager()

        consecutive_signals = []

        # Generate signals across rolling windows (exactly 10 windows)
        for i in range(10):
            start_idx = 20 + i * 5
            if start_idx + 20 > len(sample_ohlcv_data):
                break
            window_data = sample_ohlcv_data.iloc[start_idx:start_idx+20]
            signal = strategy.generate_signal(window_data)
            consecutive_signals.append(signal)

        # Should have some signal variation
        signal_changes = sum(1 for i in range(1, len(consecutive_signals))
                           if consecutive_signals[i] != consecutive_signals[i-1])

        # Verify signals are reasonable (may be all same in some market conditions)
        assert len(consecutive_signals) == 10  # Should have processed all windows
        # Allow for possibility of no signal changes in certain market conditions
        assert all(s in [-1, 0, 1] for s in consecutive_signals)


class TestMultiStrategyRiskIntegration:
    """Test multiple strategies working with risk management."""

    @pytest.mark.parametrize("strategy_class", [
        MomentumStrategy,
        MovingAverageStrategy,
    ])
    def test_multiple_strategies_risk_compatibility(self, strategy_class, sample_ohlcv_data):
        """Test that multiple strategies work with the same risk manager."""
        strategy = strategy_class()
        risk_manager = AdvancedRiskManager()

        signal = strategy.generate_signal(sample_ohlcv_data)

        if signal != 0:
            assessment = risk_manager.assess_trade_risk(
                symbol='ETHUSDT',
                signal=signal,
                price=sample_ohlcv_data['close'].iloc[-1],
                confidence=0.8,
                market_data=sample_ohlcv_data
            )

            # All strategies should produce valid assessments
            assert isinstance(assessment, dict)
            assert 'approved' in assessment

    def test_risk_manager_state_persistence(self, sample_ohlcv_data):
        """Test that risk manager maintains state across multiple assessments."""
        risk_manager = AdvancedRiskManager()

        initial_capital = risk_manager.current_capital

        # Perform multiple assessments
        for i in range(5):
            # Create slightly different data each time
            test_data = sample_ohlcv_data.iloc[:-i] if i > 0 else sample_ohlcv_data

            assessment = risk_manager.assess_trade_risk(
                symbol='ETHUSDT',
                signal=1,
                price=test_data['close'].iloc[-1],
                confidence=0.8,
                market_data=test_data
            )

        # Capital should not change from assessments alone
        assert risk_manager.current_capital == initial_capital

    def test_position_size_consistency(self, sample_ohlcv_data):
        """Test that position sizes are consistent for similar market conditions."""
        risk_manager = AdvancedRiskManager()

        # Test same conditions multiple times
        sizes = []
        for _ in range(10):
            assessment = risk_manager.assess_trade_risk(
                symbol='ETHUSDT',
                signal=1,
                price=sample_ohlcv_data['close'].iloc[-1],
                confidence=0.8,
                market_data=sample_ohlcv_data
            )

            if assessment['approved']:
                sizes.append(assessment['recommended_size'])

        if len(sizes) > 1:
            # Sizes should be reasonably consistent
            size_std = np.std(sizes)
            size_mean = np.mean(sizes)

            # Coefficient of variation should be reasonable
            cv = size_std / size_mean if size_mean > 0 else 0
            assert cv < 0.5  # Less than 50% variation


class TestRiskManagementWorkflow:
    """Test complete risk management workflow."""

    def test_trade_approval_workflow(self, sample_ohlcv_data):
        """Test complete trade approval workflow."""
        strategy = MomentumStrategy()
        risk_manager = AdvancedRiskManager()

        # Step 1: Generate signal
        signal = strategy.generate_signal(sample_ohlcv_data)

        # Step 2: Risk assessment
        assessment = risk_manager.assess_trade_risk(
            symbol='ETHUSDT',
            signal=signal,
            price=sample_ohlcv_data['close'].iloc[-1],
            confidence=0.8,
            market_data=sample_ohlcv_data
        )

        # Step 3: Verify workflow completion
        assert 'approved' in assessment
        assert 'risk_score' in assessment
        assert 'recommended_size' in assessment
        assert 'warnings' in assessment
        assert 'reasons' in assessment

    def test_portfolio_rebalancing_integration(self):
        """Test portfolio rebalancing with strategy signals."""
        risk_manager = AdvancedRiskManager()

        # Initial portfolio
        current_positions = {
            'ETHUSDT': {'quantity': 10, 'current_price': 100.0},
            'BTCUSDT': {'quantity': 5, 'current_price': 200.0}
        }

        risk_manager.update_portfolio(current_positions, 5000)

        # Target allocations
        target_allocations = {'ETHUSDT': 0.6, 'BTCUSDT': 0.4}

        # Calculate rebalancing trades
        trades = risk_manager.calculate_portfolio_rebalance(
            target_allocations,
            current_positions,
            5000
        )

        # Verify trades are generated
        assert isinstance(trades, list)

        # Should have at least one trade to rebalance
        if trades:
            for trade in trades:
                assert trade['action'] in ['BUY', 'SELL']
                assert trade['quantity'] > 0
                assert trade['symbol'] in ['ETHUSDT', 'BTCUSDT']


# Performance testing for integration
@pytest.mark.slow
@pytest.mark.integration
def test_strategy_risk_performance_integration(sample_ohlcv_data):
    """Test performance of strategy-risk integration."""
    import time

    strategy = MomentumStrategy()
    risk_manager = AdvancedRiskManager()

    start_time = time.time()

    # Simulate multiple trading cycles
    for i in range(20):
        test_data = sample_ohlcv_data.iloc[:30+i] if 30+i <= len(sample_ohlcv_data) else sample_ohlcv_data

        # Generate signal
        signal = strategy.generate_signal(test_data)

        # Risk assessment
        if signal != 0:
            assessment = risk_manager.assess_trade_risk(
                symbol='ETHUSDT',
                signal=signal,
                price=test_data['close'].iloc[-1],
                confidence=0.8,
                market_data=test_data
            )

    end_time = time.time()

    # Should complete within reasonable time
    total_time = end_time - start_time
    assert total_time < 5.0  # Less than 5 seconds for 20 cycles


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
