"""AI-generated comprehensive tests for src/utils/constants.py"""

import pytest
from src.utils import constants


class TestConstants:
    """Comprehensive tests for constants and configuration values"""

    def test_required_ohlcv_columns(self):
        """Test REQUIRED_OHLCV_COLUMNS contains all necessary columns"""
        expected_columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
        assert constants.REQUIRED_OHLCV_COLUMNS == expected_columns
        assert len(constants.REQUIRED_OHLCV_COLUMNS) == 6
        assert all(isinstance(col, str) for col in constants.REQUIRED_OHLCV_COLUMNS)

    def test_supported_intervals_minute(self):
        """Test SUPPORTED_INTERVALS contains minute intervals"""
        minute_intervals = ['1m', '3m', '5m', '15m', '30m']
        for interval in minute_intervals:
            assert interval in constants.SUPPORTED_INTERVALS

    def test_supported_intervals_hourly(self):
        """Test SUPPORTED_INTERVALS contains hourly intervals"""
        hourly_intervals = ['1h', '2h', '4h', '6h', '8h', '12h']
        for interval in hourly_intervals:
            assert interval in constants.SUPPORTED_INTERVALS

    def test_supported_intervals_daily_weekly(self):
        """Test SUPPORTED_INTERVALS contains daily and weekly intervals"""
        assert '1d' in constants.SUPPORTED_INTERVALS
        assert '3d' in constants.SUPPORTED_INTERVALS
        assert '1w' in constants.SUPPORTED_INTERVALS
        assert '1M' in constants.SUPPORTED_INTERVALS

    def test_supported_symbols_crypto(self):
        """Test SUPPORTED_SYMBOLS contains major crypto pairs"""
        major_pairs = ['BTCUSDT', 'ETHUSDT', 'BNBUSDT', 'ADAUSDT', 'XRPUSDT']
        for symbol in major_pairs:
            assert symbol in constants.SUPPORTED_SYMBOLS

    def test_supported_symbols_count(self):
        """Test SUPPORTED_SYMBOLS has expected count"""
        assert len(constants.SUPPORTED_SYMBOLS) >= 10
        assert all(symbol.endswith('USDT') for symbol in constants.SUPPORTED_SYMBOLS)

    def test_default_stop_loss_percentage(self):
        """Test DEFAULT_STOP_LOSS_PCT is reasonable"""
        assert isinstance(constants.DEFAULT_STOP_LOSS_PCT, (int, float))
        assert 0 < constants.DEFAULT_STOP_LOSS_PCT < 1  # Should be percentage
        assert constants.DEFAULT_STOP_LOSS_PCT == 0.02  # 2%

    def test_default_take_profit_percentage(self):
        """Test DEFAULT_TAKE_PROFIT_PCT is reasonable"""
        assert isinstance(constants.DEFAULT_TAKE_PROFIT_PCT, (int, float))
        assert 0 < constants.DEFAULT_TAKE_PROFIT_PCT < 1
        assert constants.DEFAULT_TAKE_PROFIT_PCT == 0.05  # 5%

    def test_default_max_position_size(self):
        """Test DEFAULT_MAX_POSITION_SIZE_PCT is reasonable"""
        assert isinstance(constants.DEFAULT_MAX_POSITION_SIZE_PCT, (int, float))
        assert 0 < constants.DEFAULT_MAX_POSITION_SIZE_PCT <= 1
        assert constants.DEFAULT_MAX_POSITION_SIZE_PCT == 0.10  # 10%

    def test_default_max_daily_loss(self):
        """Test DEFAULT_MAX_DAILY_LOSS_PCT is reasonable"""
        assert isinstance(constants.DEFAULT_MAX_DAILY_LOSS_PCT, (int, float))
        assert 0 < constants.DEFAULT_MAX_DAILY_LOSS_PCT < 1
        assert constants.DEFAULT_MAX_DAILY_LOSS_PCT == 0.05  # 5%

    def test_default_max_portfolio_drawdown(self):
        """Test DEFAULT_MAX_PORTFOLIO_DRAWDOWN is reasonable"""
        assert isinstance(constants.DEFAULT_MAX_PORTFOLIO_DRAWDOWN, (int, float))
        assert 0 < constants.DEFAULT_MAX_PORTFOLIO_DRAWDOWN < 1
        assert constants.DEFAULT_MAX_PORTFOLIO_DRAWDOWN == 0.15  # 15%

    def test_default_macd_parameters(self):
        """Test DEFAULT_MACD_* parameters"""
        assert isinstance(constants.DEFAULT_MACD_FAST, int)
        assert isinstance(constants.DEFAULT_MACD_SLOW, int)
        assert isinstance(constants.DEFAULT_MACD_SIGNAL, int)
        assert constants.DEFAULT_MACD_FAST < constants.DEFAULT_MACD_SLOW
        assert constants.DEFAULT_MACD_SIGNAL < constants.DEFAULT_MACD_SLOW

    def test_default_rsi_period(self):
        """Test DEFAULT_RSI_PERIOD"""
        assert isinstance(constants.DEFAULT_RSI_PERIOD, int)
        assert constants.DEFAULT_RSI_PERIOD > 0

    def test_default_bollinger_bands(self):
        """Test DEFAULT_BB_* parameters"""
        assert isinstance(constants.DEFAULT_BB_PERIOD, int)
        assert isinstance(constants.DEFAULT_BB_STD, float)
        assert constants.DEFAULT_BB_PERIOD > 0
        assert constants.DEFAULT_BB_STD > 0

    def test_default_atr_period(self):
        """Test DEFAULT_ATR_PERIOD"""
        assert isinstance(constants.DEFAULT_ATR_PERIOD, int)
        assert constants.DEFAULT_ATR_PERIOD > 0

    def test_default_cache_ttl(self):
        """Test DEFAULT_CACHE_TTL"""
        assert isinstance(constants.DEFAULT_CACHE_TTL, int)
        assert constants.DEFAULT_CACHE_TTL > 0

    def test_default_data_age_limit(self):
        """Test DEFAULT_MAX_DATA_AGE_HOURS"""
        assert isinstance(constants.DEFAULT_MAX_DATA_AGE_HOURS, int)
        assert constants.DEFAULT_MAX_DATA_AGE_HOURS > 0

    def test_default_batch_size(self):
        """Test DEFAULT_BATCH_SIZE"""
        assert isinstance(constants.DEFAULT_BATCH_SIZE, int)
        assert constants.DEFAULT_BATCH_SIZE > 0

    def test_default_concurrent_requests(self):
        """Test DEFAULT_MAX_CONCURRENT_REQUESTS"""
        assert isinstance(constants.DEFAULT_MAX_CONCURRENT_REQUESTS, int)
        assert constants.DEFAULT_MAX_CONCURRENT_REQUESTS > 0

    def test_default_log_level(self):
        """Test DEFAULT_LOG_LEVEL"""
        assert isinstance(constants.DEFAULT_LOG_LEVEL, str)
        assert constants.DEFAULT_LOG_LEVEL in ['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL']

    def test_default_config_update_interval(self):
        """Test DEFAULT_CONFIG_UPDATE_INTERVAL"""
        assert isinstance(constants.DEFAULT_CONFIG_UPDATE_INTERVAL, int)
        assert constants.DEFAULT_CONFIG_UPDATE_INTERVAL > 0

    def test_default_health_check_interval(self):
        """Test DEFAULT_HEALTH_CHECK_INTERVAL"""
        assert isinstance(constants.DEFAULT_HEALTH_CHECK_INTERVAL, int)
        assert constants.DEFAULT_HEALTH_CHECK_INTERVAL > 0

    def test_target_response_time(self):
        """Test TARGET_RESPONSE_TIME_MS"""
        assert isinstance(constants.TARGET_RESPONSE_TIME_MS, int)
        assert constants.TARGET_RESPONSE_TIME_MS > 0

    def test_target_memory_usage(self):
        """Test TARGET_MEMORY_USAGE_MB"""
        assert isinstance(constants.TARGET_MEMORY_USAGE_MB, int)
        assert constants.TARGET_MEMORY_USAGE_MB > 0

    def test_target_test_coverage(self):
        """Test TARGET_TEST_COVERAGE_PCT"""
        assert isinstance(constants.TARGET_TEST_COVERAGE_PCT, int)
        assert 0 < constants.TARGET_TEST_COVERAGE_PCT <= 100

    def test_default_config_structure(self):
        """Test DEFAULT_CONFIG structure"""
        assert isinstance(constants.DEFAULT_CONFIG, dict)
        assert len(constants.DEFAULT_CONFIG) > 0

    def test_error_messages_structure(self):
        """Test ERROR_MESSAGES structure"""
        assert isinstance(constants.ERROR_MESSAGES, dict)
        assert len(constants.ERROR_MESSAGES) > 0
        assert all(isinstance(msg, str) for msg in constants.ERROR_MESSAGES.values())

    def test_success_messages_structure(self):
        """Test SUCCESS_MESSAGES structure"""
        assert isinstance(constants.SUCCESS_MESSAGES, dict)
        assert len(constants.SUCCESS_MESSAGES) > 0
        assert all(isinstance(msg, str) for msg in constants.SUCCESS_MESSAGES.values())

    def test_logging_config_structure(self):
        """Test LOGGING_CONFIG structure"""
        assert isinstance(constants.LOGGING_CONFIG, dict)
        assert 'version' in constants.LOGGING_CONFIG
        assert 'formatters' in constants.LOGGING_CONFIG
        assert 'handlers' in constants.LOGGING_CONFIG
        assert 'root' in constants.LOGGING_CONFIG

    def test_performance_thresholds(self):
        """Test PERFORMANCE_THRESHOLDS structure"""
        assert isinstance(constants.PERFORMANCE_THRESHOLDS, dict)
        assert len(constants.PERFORMANCE_THRESHOLDS) > 0

    def test_circuit_breaker_config(self):
        """Test CIRCUIT_BREAKER_CONFIG structure"""
        assert isinstance(constants.CIRCUIT_BREAKER_CONFIG, dict)
        assert 'failure_threshold' in constants.CIRCUIT_BREAKER_CONFIG
        assert 'success_threshold' in constants.CIRCUIT_BREAKER_CONFIG
        assert 'monitoring_interval' in constants.CIRCUIT_BREAKER_CONFIG

    def test_health_check_config(self):
        """Test HEALTH_CHECK_CONFIG structure"""
        assert isinstance(constants.HEALTH_CHECK_CONFIG, dict)
        assert len(constants.HEALTH_CHECK_CONFIG) > 0

    def test_constants_types(self):
        """Test that all constants have appropriate types"""
        # Lists should contain appropriate types
        assert all(isinstance(interval, str) for interval in constants.SUPPORTED_INTERVALS)
        assert all(isinstance(symbol, str) for symbol in constants.SUPPORTED_SYMBOLS)

        # Dictionaries should have string keys
        dict_constants = [
            constants.DEFAULT_CONFIG, constants.ERROR_MESSAGES, constants.SUCCESS_MESSAGES,
            constants.LOGGING_CONFIG, constants.PERFORMANCE_THRESHOLDS,
            constants.CIRCUIT_BREAKER_CONFIG, constants.HEALTH_CHECK_CONFIG
        ]

        for const_dict in dict_constants:
            assert all(isinstance(k, str) for k in const_dict.keys())

    def test_constants_reasonable_values(self):
        """Test that constants have reasonable values"""
        # Risk percentages should be positive and reasonable
        assert 0 < constants.DEFAULT_STOP_LOSS_PCT <= 0.10  # Max 10%
        assert 0 < constants.DEFAULT_TAKE_PROFIT_PCT <= 0.50  # Max 50%
        assert 0 < constants.DEFAULT_MAX_POSITION_SIZE_PCT <= 1.0
        assert 0 < constants.DEFAULT_MAX_DAILY_LOSS_PCT <= 0.20  # Max 20%

        # Technical indicators should be reasonable
        assert 5 <= constants.DEFAULT_MACD_FAST <= 50
        assert 10 <= constants.DEFAULT_MACD_SLOW <= 100
        assert 5 <= constants.DEFAULT_MACD_SIGNAL <= 20
        assert 5 <= constants.DEFAULT_RSI_PERIOD <= 30
        assert 10 <= constants.DEFAULT_BB_PERIOD <= 50
        assert 1.0 <= constants.DEFAULT_BB_STD <= 3.0

        # Performance targets should be reasonable
        assert constants.TARGET_RESPONSE_TIME_MS >= 50  # At least 50ms
        assert constants.TARGET_MEMORY_USAGE_MB >= 50   # At least 50MB
        assert constants.TARGET_TEST_COVERAGE_PCT >= 70  # At least 70%
