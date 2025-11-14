#!/usr/bin/env python3
"""
Comprehensive unit tests for Advanced strategy modules.
Generated with Codex Web for Supreme System V5.
"""

from __future__ import annotations

import math
import time
from datetime import datetime, timedelta
from typing import Any, Dict, List
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from src.strategies.base_strategy import BaseStrategy
from src.strategies.momentum import MomentumStrategy
from src.strategies.mean_reversion import MeanReversionStrategy
from src.strategies.breakout import ImprovedBreakoutStrategy
from src.strategies.moving_average import MovingAverageStrategy
from src.strategies.trend_following import TrendFollowingAgent
from src.strategies.strategy_registry import StrategyFactory, StrategyRegistry


def _build_ohlcv_dataframe(
    *,
    periods: int,
    start_price: float = 100.0,
    price_step: float = 1.0,
    volatility: float = 0.5,
    volume: int = 1_000_000,
) -> pd.DataFrame:
    """Helper to create deterministic OHLCV samples for the tests."""

    timestamps = pd.date_range(datetime(2024, 1, 1), periods=periods, freq="H")
    prices = start_price + np.arange(periods) * price_step
    noise = np.sin(np.linspace(0, np.pi, periods)) * volatility
    closes = prices + noise
    data = {
        "timestamp": timestamps,
        "open": closes + 0.1,
        "high": closes + 0.5,
        "low": closes - 0.5,
        "close": closes,
        "volume": np.full(periods, volume, dtype=float),
    }
    return pd.DataFrame(data)


class TestMomentumStrategy:
    """Unit tests focused on the MomentumStrategy decision logic."""

    def test_momentum_generate_signal_invalid_columns_returns_hold(self) -> None:
        """Ensure the strategy gracefully rejects malformed market data."""

        strategy = MomentumStrategy()
        bad_df = pd.DataFrame({"timestamp": [], "close": []})
        assert strategy.generate_signal(bad_df) == 0, "Invalid data must yield a neutral signal"

    def test_momentum_generate_signal_requires_minimum_history(self) -> None:
        """Signals should be neutral when the look-back window is not satisfied."""

        strategy = MomentumStrategy(short_period=3, long_period=6, signal_period=2, roc_period=4)
        df = _build_ohlcv_dataframe(periods=3)
        assert strategy.generate_signal(df) == 0

    def test_momentum_generate_signal_bullish_path(self) -> None:
        """Combine cached indicators pointing bullish to expect a long signal."""

        strategy = MomentumStrategy()
        df = _build_ohlcv_dataframe(periods=40)

        with patch.object(strategy, "validate_data", return_value=True), patch.object(
            strategy, "_precalculate_indicators", return_value=None
        ), patch.object(strategy, "_calculate_macd_signal_cached", return_value=1), patch.object(
            strategy, "_calculate_roc_signal_cached", return_value=1
        ), patch.object(
            strategy, "_calculate_trend_signal_cached", return_value=1
        ), patch.object(
            strategy, "_calculate_volume_confirmation", return_value=1
        ):
            assert strategy.generate_signal(df) == 1

    def test_momentum_generate_signal_bearish_path(self) -> None:
        """Combine cached indicators pointing bearish to expect a short signal."""

        strategy = MomentumStrategy(volume_confirmation=False)
        df = _build_ohlcv_dataframe(periods=50, price_step=-1.0)

        with patch.object(strategy, "validate_data", return_value=True), patch.object(
            strategy, "_precalculate_indicators", return_value=None
        ), patch.object(strategy, "_calculate_macd_signal_cached", return_value=-1), patch.object(
            strategy, "_calculate_roc_signal_cached", return_value=-1
        ), patch.object(
            strategy, "_calculate_trend_signal_cached", return_value=-1
        ):
            assert strategy.generate_signal(df) == -1

    def test_momentum_calculate_indicators_handles_nan_values(self) -> None:
        """The indicator calculation should survive NaN contaminations."""

        strategy = MomentumStrategy()
        df = _build_ohlcv_dataframe(periods=60)
        df.loc[10:12, "close"] = np.nan
        indicators = strategy.calculate_momentum_indicators(df)
        for column in ["macd_line", "macd_signal", "roc", "trend_strength"]:
            assert column in indicators.columns, f"Expected {column} in indicator output"

    def test_momentum_get_momentum_score_aggregates_signals(self) -> None:
        """Verify the score reflects the balance between bullish and bearish votes."""

        strategy = MomentumStrategy()
        df = _build_ohlcv_dataframe(periods=30)

        with patch.object(strategy, "_calculate_macd_signal", return_value=1), patch.object(
            strategy, "_calculate_roc_signal", return_value=-1
        ), patch.object(
            strategy, "_calculate_trend_signal", return_value=1
        ), patch.object(
            strategy, "_calculate_volume_confirmation", return_value=1
        ):
            score = strategy.get_momentum_score(df)
            assert math.isclose(score, 0.5), "Three bullish vs one bearish should produce 0.5"

    def test_momentum_indicator_performance_large_dataset(self) -> None:
        """Indicator generation must remain under the latency budget for big inputs."""

        strategy = MomentumStrategy()
        large_df = _build_ohlcv_dataframe(periods=1_200)
        start = time.perf_counter()
        result = strategy.calculate_momentum_indicators(large_df)
        duration = time.perf_counter() - start
        assert not result.empty
        assert duration < 1.5, "Momentum indicator calculation should be sub-1.5s"


class TestMeanReversionStrategy:
    """Validation suite for the mean-reversion signal workflow."""

    def test_mean_reversion_invalid_input_returns_hold(self) -> None:
        """Missing required OHLCV columns must disable trading signals."""

        strategy = MeanReversionStrategy()
        df = pd.DataFrame({"timestamp": pd.date_range(datetime(2024, 1, 1), periods=5), "close": [1, 2, 3, 4, 5]})
        assert strategy.generate_signal(df) == 0

    def test_mean_reversion_signal_with_insufficient_history(self) -> None:
        """Short series lacking lookback depth should produce a hold signal."""

        strategy = MeanReversionStrategy(lookback_period=20)
        df = _build_ohlcv_dataframe(periods=10)
        assert strategy.generate_signal(df) == 0

    def test_mean_reversion_bollinger_buy_signal(self) -> None:
        """Force Bollinger calculations to emit a buy recommendation."""

        strategy = MeanReversionStrategy()
        df = _build_ohlcv_dataframe(periods=50, price_step=-0.2)

        with patch.object(strategy, "_calculate_bollinger_signal", return_value=1), patch.object(
            strategy, "_calculate_rsi_signal", return_value=0
        ):
            assert strategy.generate_signal(df) == 1

    def test_mean_reversion_bollinger_sell_signal_with_rsi_confirmation(self) -> None:
        """Combine Bollinger and RSI negative readings when RSI is enabled."""

        strategy = MeanReversionStrategy(use_rsi=True)
        df = _build_ohlcv_dataframe(periods=60)

        with patch.object(strategy, "_calculate_bollinger_signal", return_value=-1), patch.object(
            strategy, "_calculate_rsi_signal", return_value=-1
        ):
            assert strategy.generate_signal(df) == -1

    def test_mean_reversion_rsi_series_handles_edge_cases(self) -> None:
        """RSI calculation should return a Series even with flat prices."""

        strategy = MeanReversionStrategy()
        prices = pd.Series([100.0] * 40)
        rsi = strategy._calculate_rsi(prices, period=14)
        assert isinstance(rsi, pd.Series)
        assert rsi.isna().sum() > 0, "Flat prices lead to NaN RSI entries"

    def test_mean_reversion_generate_signal_with_nan_data(self) -> None:
        """NaN contamination must not raise and should yield neutral signals."""

        strategy = MeanReversionStrategy(use_rsi=True)
        df = _build_ohlcv_dataframe(periods=40)
        df.loc[5:7, "close"] = np.nan

        with patch.object(strategy, "_calculate_bollinger_signal", return_value=0), patch.object(
            strategy, "_calculate_rsi_signal", return_value=0
        ):
            assert strategy.generate_signal(df) == 0


class TestMovingAverageStrategy:
    """Scenarios that exercise the moving-average crossover logic."""

    def test_moving_average_invalid_data_returns_hold(self) -> None:
        """Missing close column should suppress trade signals."""

        strategy = MovingAverageStrategy(short_window=3, long_window=5)
        df = pd.DataFrame({"timestamp": pd.date_range(datetime(2024, 1, 1), periods=5), "open": [1, 2, 3, 4, 5]})
        assert strategy.generate_signal(df) == 0

    def test_moving_average_no_crossover_returns_hold(self) -> None:
        """Parallel averages must lead to a hold decision."""

        strategy = MovingAverageStrategy(short_window=3, long_window=5)
        df = _build_ohlcv_dataframe(periods=10, price_step=0.0)
        assert strategy.generate_signal(df) == 0

    def test_moving_average_bullish_crossover(self) -> None:
        """Ensure a bullish crossover outputs a BUY signal."""

        strategy = MovingAverageStrategy(short_window=2, long_window=4)
        df = _build_ohlcv_dataframe(periods=12)

        df.loc[:, "close"] = np.concatenate((np.linspace(100, 99, 6), np.linspace(99, 110, 6)))
        signal = strategy.generate_signal(df)
        assert signal == 1

    def test_moving_average_bearish_crossover(self) -> None:
        """Ensure a bearish crossover outputs a SELL signal."""

        strategy = MovingAverageStrategy(short_window=2, long_window=4)
        df = _build_ohlcv_dataframe(periods=12)
        df.loc[:, "close"] = np.concatenate((np.linspace(100, 110, 6), np.linspace(110, 90, 6)))
        signal = strategy.generate_signal(df)
        assert signal == -1

    def test_moving_average_calculate_mas_large_dataset(self) -> None:
        """Moving average computation must scale linearly with big inputs."""

        strategy = MovingAverageStrategy(short_window=5, long_window=20)
        large_df = _build_ohlcv_dataframe(periods=1_024)
        start = time.perf_counter()
        enriched = strategy.calculate_moving_averages(large_df)
        duration = time.perf_counter() - start
        assert {"short_ma", "long_ma"}.issubset(enriched.columns)
        assert duration < 0.5


class TestTrendFollowingAgent:
    """Trend-following agent smoke tests covering signal and indicator paths."""

    def test_trend_following_initialization_sets_parameters(self) -> None:
        """Agent parameters from config must reflect on the instance."""

        config = {"short_window": 10, "long_window": 30, "adx_threshold": 20}
        agent = TrendFollowingAgent(agent_id="agent-1", config=config)
        assert agent.parameters["short_window"] == 10
        assert agent.parameters["long_window"] == 30
        assert agent.parameters["adx_threshold"] == 20

    def test_trend_following_generate_signal_insufficient_data(self) -> None:
        """A short dataset should yield a HOLD dict."""

        agent = TrendFollowingAgent(agent_id="agent-2", config={})
        df = _build_ohlcv_dataframe(periods=10)
        signal = agent.generate_signal(df)
        assert signal["action"] == "HOLD"
        assert signal["confidence"] == 0.0

    def test_trend_following_buy_path(self) -> None:
        """Mock indicators to trigger a BUY response with position sizing."""

        agent = TrendFollowingAgent(agent_id="agent-3", config={})
        df = _build_ohlcv_dataframe(periods=agent.long_window + 5)
        mocked_indicators = df.copy()
        mocked_indicators["sma_short"] = mocked_indicators["close"] + 1
        mocked_indicators["sma_long"] = mocked_indicators["close"]
        mocked_indicators["adx"] = np.full(len(df), agent.adx_threshold + 5)
        mocked_indicators["rsi"] = np.full(len(df), agent.rsi_oversold + 1)
        mocked_indicators["macd"] = np.full(len(df), 2.0)
        mocked_indicators["macd_signal"] = np.full(len(df), 1.0)
        mocked_indicators["volume"] = np.full(len(df), 2_000_000)
        mocked_indicators["volume_ma"] = np.full(len(df), 1_000_000)

        with patch.object(agent, "_calculate_indicators", return_value=mocked_indicators), patch.object(
            agent, "_determine_trend_direction", return_value="UPTREND"
        ), patch.object(agent, "_check_buy_conditions", return_value=True), patch.object(
            agent, "_calculate_position_size", return_value=25
        ):
            signal = agent.generate_signal(df, portfolio_value=50_000)
            assert signal["action"] == "BUY"
            assert signal["quantity"] == 25

    def test_trend_following_determine_trend_sideways_when_conditions_disagree(self) -> None:
        """Disagreement among indicators must fall back to SIDEWAYS."""

        agent = TrendFollowingAgent(agent_id="agent-4", config={})
        df = _build_ohlcv_dataframe(periods=agent.long_window + 1)
        df["sma_short"] = df["close"]
        df["sma_long"] = df["close"] + 1
        df["adx"] = np.full(len(df), agent.adx_threshold - 1)
        df["macd"] = np.full(len(df), 0.5)
        df["macd_signal"] = np.full(len(df), 0.6)
        trend = agent._determine_trend_direction(df)
        assert trend == "SIDEWAYS"

    def test_trend_following_position_size_bounds(self) -> None:
        """Position sizing must never return zero."""

        agent = TrendFollowingAgent(agent_id="agent-5", config={"stop_loss_pct": 0.05})
        size = agent._calculate_position_size(portfolio_value=1_000, price=200)
        assert size >= 1


class TestImprovedBreakoutStrategy:
    """Cover the complex breakout workflow with focused behavioural probes."""

    def _prepare_breakout_strategy(self) -> ImprovedBreakoutStrategy:
        strategy = ImprovedBreakoutStrategy(
            lookback_period=10,
            breakout_threshold=0.01,
            consolidation_period=5,
            max_hold_period=3,
            use_multi_timeframe=False,
            use_volume_analysis=False,
            use_pullback_detection=False,
        )
        return strategy

    def test_breakout_generate_signal_returns_hold_for_invalid_data(self) -> None:
        """Missing critical columns should result in a neutral signal."""

        strategy = self._prepare_breakout_strategy()
        df = pd.DataFrame({"timestamp": [], "close": []})
        assert strategy.generate_signal(df) == 0

    def test_breakout_generate_signal_insufficient_history(self) -> None:
        """Short data series should not produce breakout alerts."""

        strategy = self._prepare_breakout_strategy()
        df = _build_ohlcv_dataframe(periods=12)
        assert strategy.generate_signal(df) == 0

    def test_breakout_detect_breakout_bullish_path(self) -> None:
        """Mock helper layers so the detection pipeline emits a BUY."""

        strategy = self._prepare_breakout_strategy()
        df = _build_ohlcv_dataframe(periods=40)

        with patch.object(strategy, "validate_data", return_value=True), patch.object(
            strategy, "_manage_position", return_value=0
        ), patch.object(
            strategy, "_detect_breakout", return_value=1
        ):
            assert strategy.generate_signal(df) == 1

    def test_breakout_manage_position_exit_logic(self) -> None:
        """Ensure active positions use _manage_position for signals."""

        strategy = self._prepare_breakout_strategy()
        strategy.position_active = True
        with patch.object(strategy, "_manage_position", return_value=-1) as manage_mock:
            df = _build_ohlcv_dataframe(periods=20)
            signal = strategy.generate_signal(df)
            manage_mock.assert_called_once()
            assert signal == -1

    def test_breakout_parameter_management(self) -> None:
        """get/set/reset should round-trip configuration values."""

        strategy = self._prepare_breakout_strategy()
        strategy.set_parameters(lookback_period=15, breakout_threshold=0.03)
        params = strategy.get_parameters()
        assert params["lookback_period"] == 15
        assert params["breakout_threshold"] == 0.03

        strategy.reset()
        assert not strategy.position_active

    def test_breakout_performance_stats_structure(self) -> None:
        """Performance reporting should expose expected keys."""

        strategy = self._prepare_breakout_strategy()
        stats = strategy.get_performance_stats()
        for key in ["total_signals", "successful_breakouts", "false_breakouts"]:
            assert key in stats


class TestStrategyRegistryIntegration:
    """Integration coverage for the registry and factory ecosystem."""

    def test_register_and_list_strategies(self) -> None:
        """Registered strategies should appear in the listing."""

        registry = StrategyRegistry()
        assert registry.register_strategy("momentum", MomentumStrategy)
        strategies = registry.list_strategies()
        assert any(entry["name"] == "momentum" for entry in strategies)

    def test_factory_rejects_non_compliant_strategy(self) -> None:
        """Built-in strategies returning primitive signals fail validation by design."""

        registry = StrategyRegistry()
        registry.register_strategy("moving_average", MovingAverageStrategy)
        factory = StrategyFactory(registry)
        assert factory.create("moving_average") is None

    def test_registry_supports_custom_strategy_returning_dict(self) -> None:
        """A fully compliant custom strategy should pass validation."""

        class _CompliantStrategy(BaseStrategy):
            def __init__(self) -> None:
                super().__init__("compliant")

            def generate_signal(self, data: pd.DataFrame, portfolio_value: float | None = None) -> Dict[str, Any]:
                if data.empty:
                    return {"action": "HOLD", "symbol": "TEST", "strength": 0, "confidence": 0}
                return {"action": "BUY", "symbol": "TEST", "strength": 1, "confidence": 0.9}

        registry = StrategyRegistry()
        assert registry.register_strategy("compliant", _CompliantStrategy)
        factory = StrategyFactory(registry)
        strategy = factory.create("compliant")
        assert isinstance(strategy, _CompliantStrategy)

