"""AI-generated comprehensive tests for src/utils/constants.py"""

import pytest
from src.utils.constants import *


class TestConstants:
    """Comprehensive tests for constants and configuration values"""

    def test_required_ohlcv_columns(self):
        """Test REQUIRED_OHLCV_COLUMNS contains all necessary columns"""
        expected_columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
        assert REQUIRED_OHLCV_COLUMNS == expected_columns
        assert len(REQUIRED_OHLCV_COLUMNS) == 6
        assert all(isinstance(col, str) for col in REQUIRED_OHLCV_COLUMNS)

    def test_supported_intervals_minute(self):
        """Test SUPPORTED_INTERVALS contains minute intervals"""
        minute_intervals = ['1m', '3m', '5m', '15m', '30m']
        for interval in minute_intervals:
            assert interval in SUPPORTED_INTERVALS

    def test_supported_intervals_hourly(self):
        """Test SUPPORTED_INTERVALS contains hourly intervals"""
        hourly_intervals = ['1h', '2h', '4h', '6h', '8h', '12h']
        for interval in hourly_intervals:
            assert interval in SUPPORTED_INTERVALS

    def test_supported_intervals_daily_weekly(self):
        """Test SUPPORTED_INTERVALS contains daily and weekly intervals"""
        assert '1d' in SUPPORTED_INTERVALS
        assert '3d' in SUPPORTED_INTERVALS
        assert '1w' in SUPPORTED_INTERVALS
        assert '1M' in SUPPORTED_INTERVALS

    def test_supported_symbols_crypto(self):
        """Test SUPPORTED_SYMBOLS contains major crypto pairs"""
        major_pairs = ['BTCUSDT', 'ETHUSDT', 'BNBUSDT', 'ADAUSDT', 'XRPUSDT']
        for symbol in major_pairs:
            assert symbol in SUPPORTED_SYMBOLS

    def test_supported_symbols_count(self):
        """Test SUPPORTED_SYMBOLS has expected count"""
        assert len(SUPPORTED_SYMBOLS) >= 10
        assert all(symbol.endswith('USDT') for symbol in SUPPORTED_SYMBOLS)

    def test_default_stop_loss_percentage(self):
        """Test DEFAULT_STOP_LOSS_PCT is reasonable"""
        assert isinstance(DEFAULT_STOP_LOSS_PCT, (int, float))
        assert 0 < DEFAULT_STOP_LOSS_PCT < 1  # Should be percentage
        assert DEFAULT_STOP_LOSS_PCT == 0.02  # 2%

    def test_default_take_profit_percentage(self):
        """Test DEFAULT_TAKE_PROFIT_PCT is reasonable"""
        assert isinstance(DEFAULT_TAKE_PROFIT_PCT, (int, float))
        assert 0 < DEFAULT_TAKE_PROFIT_PCT < 1
        assert DEFAULT_TAKE_PROFIT_PCT == 0.05  # 5%

    def test_default_max_position_size(self):
        """Test DEFAULT_MAX_POSITION_SIZE_PCT is reasonable"""
        assert isinstance(DEFAULT_MAX_POSITION_SIZE_PCT, (int, float))
        assert 0 < DEFAULT_MAX_POSITION_SIZE_PCT <= 1
        assert DEFAULT_MAX_POSITION_SIZE_PCT == 0.10  # 10%

    def test_default_max_daily_loss(self):
        """Test DEFAULT_MAX_DAILY_LOSS_PCT is reasonable"""
        assert isinstance(DEFAULT_MAX_DAILY_LOSS_PCT, (int, float))
        assert 0 < DEFAULT_MAX_DAILY_LOSS_PCT < 1
        assert DEFAULT_MAX_DAILY_LOSS_PCT == 0.05  # 5%

    def test_default_max_portfolio_drawdown(self):
        """Test DEFAULT_MAX_PORTFOLIO_DRAWDOWN is reasonable"""
        assert isinstance(DEFAULT_MAX_PORTFOLIO_DRAWDOWN, (int, float))
        assert 0 < DEFAULT_MAX_PORTFOLIO_DRAWDOWN < 1
        assert DEFAULT_MAX_PORTFOLIO_DRAWDOWN == 0.15  # 15%

    def test_default_macd_parameters(self):
        """Test DEFAULT_MACD_* parameters"""
        assert isinstance(DEFAULT_MACD_FAST, int)
        assert isinstance(DEFAULT_MACD_SLOW, int)
        assert isinstance(DEFAULT_MACD_SIGNAL, int)
        assert DEFAULT_MACD_FAST < DEFAULT_MACD_SLOW
        assert DEFAULT_MACD_SIGNAL < DEFAULT_MACD_SLOW

    def test_default_rsi_period(self):
        """Test DEFAULT_RSI_PERIOD"""
        assert isinstance(DEFAULT_RSI_PERIOD, int)
        assert DEFAULT_RSI_PERIOD > 0

    def test_default_bollinger_bands(self):
        """Test DEFAULT_BB_* parameters"""
        assert isinstance(DEFAULT_BB_PERIOD, int)
        assert isinstance(DEFAULT_BB_STD, float)
        assert DEFAULT_BB_PERIOD > 0
        assert DEFAULT_BB_STD > 0

    def test_default_atr_period(self):
        """Test DEFAULT_ATR_PERIOD"""
        assert isinstance(DEFAULT_ATR_PERIOD, int)
        assert DEFAULT_ATR_PERIOD > 0

    def test_default_cache_ttl(self):
        """Test DEFAULT_CACHE_TTL"""
        assert isinstance(DEFAULT_CACHE_TTL, int)
        assert DEFAULT_CACHE_TTL > 0

    def test_default_data_age_limit(self):
        """Test DEFAULT_MAX_DATA_AGE_HOURS"""
        assert isinstance(DEFAULT_MAX_DATA_AGE_HOURS, int)
        assert DEFAULT_MAX_DATA_AGE_HOURS > 0

    def test_default_batch_size(self):
        """Test DEFAULT_BATCH_SIZE"""
        assert isinstance(DEFAULT_BATCH_SIZE, int)
        assert DEFAULT_BATCH_SIZE > 0

    def test_default_concurrent_requests(self):
        """Test DEFAULT_MAX_CONCURRENT_REQUESTS"""
        assert isinstance(DEFAULT_MAX_CONCURRENT_REQUESTS, int)
        assert DEFAULT_MAX_CONCURRENT_REQUESTS > 0

    def test_default_log_level(self):
        """Test DEFAULT_LOG_LEVEL"""
        assert isinstance(DEFAULT_LOG_LEVEL, str)
        assert DEFAULT_LOG_LEVEL in ['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL']

    def test_default_config_update_interval(self):
        """Test DEFAULT_CONFIG_UPDATE_INTERVAL"""
        assert isinstance(DEFAULT_CONFIG_UPDATE_INTERVAL, int)
        assert DEFAULT_CONFIG_UPDATE_INTERVAL > 0

    def test_default_health_check_interval(self):
        """Test DEFAULT_HEALTH_CHECK_INTERVAL"""
        assert isinstance(DEFAULT_HEALTH_CHECK_INTERVAL, int)
        assert DEFAULT_HEALTH_CHECK_INTERVAL > 0

    def test_target_response_time(self):
        """Test TARGET_RESPONSE_TIME_MS"""
        assert isinstance(TARGET_RESPONSE_TIME_MS, int)
        assert TARGET_RESPONSE_TIME_MS > 0

    def test_target_memory_usage(self):
        """Test TARGET_MEMORY_USAGE_MB"""
        assert isinstance(TARGET_MEMORY_USAGE_MB, int)
        assert TARGET_MEMORY_USAGE_MB > 0

    def test_target_test_coverage(self):
        """Test TARGET_TEST_COVERAGE_PCT"""
        assert isinstance(TARGET_TEST_COVERAGE_PCT, int)
        assert 0 < TARGET_TEST_COVERAGE_PCT <= 100

    def test_default_config_structure(self):
        """Test DEFAULT_CONFIG structure"""
        assert isinstance(DEFAULT_CONFIG, dict)
        assert len(DEFAULT_CONFIG) > 0

    def test_error_messages_structure(self):
        """Test ERROR_MESSAGES structure"""
        assert isinstance(ERROR_MESSAGES, dict)
        assert len(ERROR_MESSAGES) > 0
        assert all(isinstance(msg, str) for msg in ERROR_MESSAGES.values())

    def test_success_messages_structure(self):
        """Test SUCCESS_MESSAGES structure"""
        assert isinstance(SUCCESS_MESSAGES, dict)
        assert len(SUCCESS_MESSAGES) > 0
        assert all(isinstance(msg, str) for msg in SUCCESS_MESSAGES.values())

    def test_logging_config_structure(self):
        """Test LOGGING_CONFIG structure"""
        assert isinstance(LOGGING_CONFIG, dict)
        assert 'version' in LOGGING_CONFIG
        assert 'formatters' in LOGGING_CONFIG
        assert 'handlers' in LOGGING_CONFIG
        assert 'root' in LOGGING_CONFIG

    def test_performance_thresholds(self):
        """Test PERFORMANCE_THRESHOLDS structure"""
        assert isinstance(PERFORMANCE_THRESHOLDS, dict)
        assert len(PERFORMANCE_THRESHOLDS) > 0

    def test_circuit_breaker_config(self):
        """Test CIRCUIT_BREAKER_CONFIG structure"""
        assert isinstance(CIRCUIT_BREAKER_CONFIG, dict)
        assert 'failure_threshold' in CIRCUIT_BREAKER_CONFIG
        assert 'success_threshold' in CIRCUIT_BREAKER_CONFIG
        assert 'monitoring_interval' in CIRCUIT_BREAKER_CONFIG

    def test_health_check_config(self):
        """Test HEALTH_CHECK_CONFIG structure"""
        assert isinstance(HEALTH_CHECK_CONFIG, dict)
        assert len(HEALTH_CHECK_CONFIG) > 0

    def test_constants_types(self):
        """Test that all constants have appropriate types"""
        # Lists should contain appropriate types
        assert all(isinstance(interval, str) for interval in SUPPORTED_INTERVALS)
        assert all(isinstance(symbol, str) for symbol in SUPPORTED_SYMBOLS)

        # Dictionaries should have string keys
        dict_constants = [
            DEFAULT_CONFIG, ERROR_MESSAGES, SUCCESS_MESSAGES,
            LOGGING_CONFIG, PERFORMANCE_THRESHOLDS,
            CIRCUIT_BREAKER_CONFIG, HEALTH_CHECK_CONFIG
        ]

        for const_dict in dict_constants:
            assert all(isinstance(k, str) for k in const_dict.keys())

    def test_constants_reasonable_values(self):
        """Test that constants have reasonable values"""
        # Risk percentages should be positive and reasonable
        assert 0 < DEFAULT_STOP_LOSS_PCT <= 0.10  # Max 10%
        assert 0 < DEFAULT_TAKE_PROFIT_PCT <= 0.50  # Max 50%
        assert 0 < DEFAULT_MAX_POSITION_SIZE_PCT <= 1.0
        assert 0 < DEFAULT_MAX_DAILY_LOSS_PCT <= 0.20  # Max 20%

        # Technical indicators should be reasonable
        assert 5 <= DEFAULT_MACD_FAST <= 50
        assert 10 <= DEFAULT_MACD_SLOW <= 100
        assert 5 <= DEFAULT_MACD_SIGNAL <= 20
        assert 5 <= DEFAULT_RSI_PERIOD <= 30
        assert 10 <= DEFAULT_BB_PERIOD <= 50
        assert 1.0 <= DEFAULT_BB_STD <= 3.0

        # Performance targets should be reasonable
        assert TARGET_RESPONSE_TIME_MS >= 50  # At least 50ms
        assert TARGET_MEMORY_USAGE_MB >= 50   # At least 50MB
        assert TARGET_TEST_COVERAGE_PCT >= 70  # At least 70%
