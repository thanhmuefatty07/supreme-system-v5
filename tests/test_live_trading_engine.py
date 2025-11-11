#!/usr/bin/env python3
"""
Tests for Supreme System V5 Live Trading Engine.

Tests live trading functionality, order execution, risk management integration,
and real-time decision making.
"""

from datetime import datetime, timedelta
from unittest.mock import MagicMock, Mock, patch

import numpy as np
import pandas as pd
import pytest

try:
    from src.trading.live_trading_engine import LiveTradingEngine
    LIVE_TRADING_AVAILABLE = True
except ImportError:
    LIVE_TRADING_AVAILABLE = False
    LiveTradingEngine = None


class TestLiveTradingEngineInitialization:
    """Test live trading engine initialization and setup."""

    @pytest.mark.skipif(not LIVE_TRADING_AVAILABLE, reason="Live trading engine not available")
    def test_live_trading_engine_init(self):
        """Test live trading engine initialization."""
        with patch('src.config.config.get_config', return_value=Mock()):
            engine = LiveTradingEngine()
            assert engine is not None
            assert hasattr(engine, 'positions')
            assert hasattr(engine, 'circuit_breaker')

    @pytest.mark.skipif(not LIVE_TRADING_AVAILABLE, reason="Live trading engine not available")
    def test_engine_configuration(self):
        """Test engine configuration loading."""
        mock_config = Mock()
        mock_config.get.return_value = "test_value"

        with patch('src.config.config.get_config', return_value=mock_config):
            engine = LiveTradingEngine()
            # Should initialize with config
            assert engine is not None

    @pytest.mark.skipif(not LIVE_TRADING_AVAILABLE, reason="Live trading engine not available")
    def test_engine_components(self):
        """Test that engine has all required components."""
        with patch('src.config.config.get_config', return_value=Mock()):
            engine = LiveTradingEngine()

            # Should have risk manager
            assert hasattr(engine, 'risk_manager')

            # Should have positions tracking
            assert hasattr(engine, 'positions')
            assert isinstance(engine.positions, dict)

            # Should have circuit breaker
            assert hasattr(engine, 'circuit_breaker')


class TestLiveTradingSignalProcessing:
    """Test signal processing and order generation."""

    @pytest.mark.skipif(not LIVE_TRADING_AVAILABLE, reason="Live trading engine not available")
    def test_signal_processing_buy_signal(self):
        """Test processing of buy signals."""
        with patch('src.config.config.get_config', return_value=Mock()):
            engine = LiveTradingEngine()

            # Mock market data
            market_data = {
                'price': 50000.0,
                'volume': 1000,
                'timestamp': datetime.now()
            }

            # Mock strategy signal
            signal = {
                'action': 'BUY',
                'symbol': 'BTCUSDT',
                'confidence': 0.8,
                'quantity': 0.01
            }

            # Mock risk assessment
            with patch.object(engine.risk_manager, 'assess_trade_risk', return_value={
                'approved': True,
                'risk_score': 0.3,
                'recommended_size': 0.01
            }):
                with patch.object(engine, '_execute_buy_order', return_value=True):
                    result = engine.execute_signal('test_strategy', 'BTCUSDT', signal, 50000.0, 0.8)

                    # Should process signal
                    assert result is not None

    @pytest.mark.skipif(not LIVE_TRADING_AVAILABLE, reason="Live trading engine not available")
    def test_signal_processing_sell_signal(self):
        """Test processing of sell signals."""
        with patch('src.config.config.get_config', return_value=Mock()):
            engine = LiveTradingEngine()

            # Add a position first
            engine.positions['BTCUSDT'] = Mock()
            engine.positions['BTCUSDT'].quantity = 0.01
            engine.positions['BTCUSDT'].entry_price = 49000.0

            signal = {
                'action': 'SELL',
                'symbol': 'BTCUSDT',
                'confidence': 0.7,
                'quantity': 0.01
            }

            with patch.object(engine.risk_manager, 'assess_trade_risk', return_value={
                'approved': True,
                'risk_score': 0.2,
                'recommended_size': 0.01
            }):
                with patch.object(engine, '_execute_sell_order', return_value=True):
                    result = engine.execute_signal('test_strategy', 'BTCUSDT', signal, 51000.0, 0.7)

                    assert result is not None

    @pytest.mark.skipif(not LIVE_TRADING_AVAILABLE, reason="Live trading engine not available")
    def test_signal_risk_rejection(self):
        """Test signal rejection due to risk limits."""
        with patch('src.config.config.get_config', return_value=Mock()):
            engine = LiveTradingEngine()

            signal = {
                'action': 'BUY',
                'symbol': 'BTCUSDT',
                'confidence': 0.9,
                'quantity': 0.1  # Large position
            }

            # Mock risk rejection
            with patch.object(engine.risk_manager, 'assess_trade_risk', return_value={
                'approved': False,
                'risk_score': 0.9,
                'reason': 'Position size too large'
            }):
                result = engine.execute_signal('test_strategy', 'BTCUSDT', signal, 50000.0, 0.9)

                # Should reject high-risk trade
                assert result is None

    @pytest.mark.skipif(not LIVE_TRADING_AVAILABLE, reason="Live trading engine not available")
    def test_signal_circuit_breaker_activation(self):
        """Test circuit breaker activation during signal processing."""
        with patch('src.config.config.get_config', return_value=Mock()):
            engine = LiveTradingEngine()

            # Force circuit breaker to open
            engine.circuit_breaker.state = type('MockState', (), {'name': 'OPEN'})()

            signal = {
                'action': 'BUY',
                'symbol': 'BTCUSDT',
                'confidence': 0.8,
                'quantity': 0.01
            }

            result = engine.execute_signal('test_strategy', 'BTCUSDT', signal, 50000.0, 0.8)

            # Should be blocked by circuit breaker
            assert result is None


class TestLiveTradingPositionManagement:
    """Test position tracking and management."""

    @pytest.mark.skipif(not LIVE_TRADING_AVAILABLE, reason="Live trading engine not available")
    def test_position_tracking(self):
        """Test position creation and tracking."""
        with patch('src.config.config.get_config', return_value=Mock()):
            engine = LiveTradingEngine()

            # Initially no positions
            assert len(engine.positions) == 0

            # Simulate adding a position
            from src.trading.live_trading_engine import LiveTradingPosition
            position = LiveTradingPosition('BTCUSDT', 'LONG', 0.01, 50000.0)
            engine.positions['BTCUSDT'] = position

            assert len(engine.positions) == 1
            assert 'BTCUSDT' in engine.positions
            assert engine.positions['BTCUSDT'].quantity == 0.01

    @pytest.mark.skipif(not LIVE_TRADING_AVAILABLE, reason="Live trading engine not available")
    def test_position_pnl_calculation(self):
        """Test profit/loss calculation for positions."""
        with patch('src.config.config.get_config', return_value=Mock()):
            engine = LiveTradingEngine()

            # Create a position
            from src.trading.live_trading_engine import LiveTradingPosition
            position = LiveTradingPosition('BTCUSDT', 'LONG', 0.01, 50000.0)
            engine.positions['BTCUSDT'] = position

            # Update with new price
            position.update_price(51000.0)

            # Calculate P&L
            pnl = position.get_unrealized_pnl()
            expected_pnl = (51000.0 - 50000.0) * 0.01  # $100 profit

            assert abs(pnl - expected_pnl) < 0.01  # Small tolerance

    @pytest.mark.skipif(not LIVE_TRADING_AVAILABLE, reason="Live trading engine not available")
    def test_position_exit_conditions(self):
        """Test position exit condition checking."""
        with patch('src.config.config.get_config', return_value=Mock()):
            engine = LiveTradingEngine()

            # Create a position
            from src.trading.live_trading_engine import LiveTradingPosition
            position = LiveTradingPosition('BTCUSDT', 'LONG', 0.01, 50000.0)
            engine.positions['BTCUSDT'] = position

            # Test stop loss
            position.stop_loss = 49000.0
            should_exit = position.should_exit(48500.0)  # Below stop loss
            assert should_exit == True

            # Test take profit
            position.take_profit = 52000.0
            should_exit = position.should_exit(52500.0)  # Above take profit
            assert should_exit == True

            # Test no exit condition
            should_exit = position.should_exit(50500.0)  # Normal price
            assert should_exit == False


class TestLiveTradingEngineIntegration:
    """Test live trading engine integration with other components."""

    @pytest.mark.skipif(not LIVE_TRADING_AVAILABLE, reason="Live trading engine not available")
    def test_engine_risk_manager_integration(self):
        """Test integration with risk manager."""
        with patch('src.config.config.get_config', return_value=Mock()):
            engine = LiveTradingEngine()

            # Test that risk manager is called during signal processing
            signal = {
                'action': 'BUY',
                'symbol': 'BTCUSDT',
                'confidence': 0.8,
                'quantity': 0.01
            }

            with patch.object(engine.risk_manager, 'assess_trade_risk') as mock_assess:
                mock_assess.return_value = {
                    'approved': True,
                    'risk_score': 0.3,
                    'recommended_size': 0.01
                }

                with patch.object(engine, '_execute_buy_order', return_value=True):
                    engine.execute_signal('test_strategy', 'BTCUSDT', signal, 50000.0, 0.8)

                    # Risk assessment should have been called
                    mock_assess.assert_called_once()

    @pytest.mark.skipif(not LIVE_TRADING_AVAILABLE, reason="Live trading engine not available")
    def test_engine_monitoring_integration(self):
        """Test integration with monitoring system."""
        with patch('src.config.config.get_config', return_value=Mock()):
            engine = LiveTradingEngine()

            # Should have monitoring capabilities
            assert hasattr(engine, 'start_monitoring')
            assert hasattr(engine, 'stop_monitoring')

    @pytest.mark.skipif(not LIVE_TRADING_AVAILABLE, reason="Live trading engine not available")
    def test_engine_error_recovery(self):
        """Test error recovery mechanisms."""
        with patch('src.config.config.get_config', return_value=Mock()):
            engine = LiveTradingEngine()

            # Test handling of network errors
            signal = {
                'action': 'BUY',
                'symbol': 'BTCUSDT',
                'confidence': 0.8,
                'quantity': 0.01
            }

            # Mock network failure
            with patch.object(engine, '_execute_buy_order', side_effect=ConnectionError("Network error")):
                result = engine.execute_signal('test_strategy', 'BTCUSDT', signal, 50000.0, 0.8)

                # Should handle network errors gracefully
                assert result is None

    @pytest.mark.skipif(not LIVE_TRADING_AVAILABLE, reason="Live trading engine not available")
    def test_engine_concurrent_signals(self):
        """Test handling of concurrent signals."""
        import threading

        with patch('src.config.config.get_config', return_value=Mock()):
            engine = LiveTradingEngine()

            results = []
            errors = []

            def process_signal(signal_id):
                try:
                    signal = {
                        'action': 'BUY',
                        'symbol': f'BTCUSDT{signal_id}',
                        'confidence': 0.8,
                        'quantity': 0.01
                    }

                    with patch.object(engine.risk_manager, 'assess_trade_risk', return_value={
                        'approved': True, 'risk_score': 0.3, 'recommended_size': 0.01
                    }):
                        with patch.object(engine, '_execute_buy_order', return_value=True):
                            result = engine.execute_signal('test_strategy', f'BTCUSDT{signal_id}', signal, 50000.0, 0.8)
                            results.append(result)
                except Exception as e:
                    errors.append(e)

            # Process multiple signals concurrently
            threads = []
            for i in range(5):
                thread = threading.Thread(target=process_signal, args=(i,))
                threads.append(thread)
                thread.start()

            # Wait for all threads
            for thread in threads:
                thread.join()

            # Should handle concurrent signals
            assert len(results) == 5
            assert len(errors) == 0


class TestLiveTradingPerformance:
    """Test live trading engine performance characteristics."""

    @pytest.mark.skipif(not LIVE_TRADING_AVAILABLE, reason="Live trading engine not available")
    def test_signal_processing_latency(self):
        """Test signal processing latency."""
        import time

        with patch('src.config.config.get_config', return_value=Mock()):
            engine = LiveTradingEngine()

            signal = {
                'action': 'BUY',
                'symbol': 'BTCUSDT',
                'confidence': 0.8,
                'quantity': 0.01
            }

            # Measure processing time
            start_time = time.time()

            with patch.object(engine.risk_manager, 'assess_trade_risk', return_value={
                'approved': True, 'risk_score': 0.3, 'recommended_size': 0.01
            }):
                with patch.object(engine, '_execute_buy_order', return_value=True):
                    result = engine.execute_signal('test_strategy', 'BTCUSDT', signal, 50000.0, 0.8)

            end_time = time.time()
            processing_time = end_time - start_time

            # Should process signals quickly (< 100ms)
            assert processing_time < 0.1, f"Processing too slow: {processing_time}s"
            assert result is not None

    @pytest.mark.skipif(not LIVE_TRADING_AVAILABLE, reason="Live trading engine not available")
    def test_memory_usage_under_load(self):
        """Test memory usage during high-frequency trading simulation."""
        import os

        import psutil

        with patch('src.config.config.get_config', return_value=Mock()):
            engine = LiveTradingEngine()

            process = psutil.Process(os.getpid())
            initial_memory = process.memory_info().rss / 1024 / 1024  # MB

            # Simulate high-frequency signals
            for i in range(100):
                signal = {
                    'action': 'BUY' if i % 2 == 0 else 'SELL',
                    'symbol': 'BTCUSDT',
                    'confidence': 0.8,
                    'quantity': 0.01
                }

                with patch.object(engine.risk_manager, 'assess_trade_risk', return_value={
                    'approved': True, 'risk_score': 0.3, 'recommended_size': 0.01
                }):
                    with patch.object(engine, '_execute_buy_order', return_value=True):
                        with patch.object(engine, '_execute_sell_order', return_value=True):
                            engine.execute_signal('test_strategy', 'BTCUSDT', signal, 50000.0, 0.8)

            final_memory = process.memory_info().rss / 1024 / 1024  # MB
            memory_delta = final_memory - initial_memory

            # Memory usage should be controlled (< 50MB increase)
            assert memory_delta < 50, f"Excessive memory usage: {memory_delta}MB"
