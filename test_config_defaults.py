#!/usr/bin/env python3
"""
Test Config & Defaults (Plug & Play Optimized Mode)
"""

import os
from dotenv import load_dotenv

def test_config_defaults():
    """Test loading configuration from .env.optimized"""

    # Load test environment
    load_dotenv('.env.optimized')

    print('🎯 Config & Defaults Test:')
    print(f'   OPTIMIZED_MODE: {os.getenv("OPTIMIZED_MODE")}')
    print(f'   EVENT_DRIVEN_PROCESSING: {os.getenv("EVENT_DRIVEN_PROCESSING")}')
    print(f'   SINGLE_SYMBOL: {os.getenv("SINGLE_SYMBOL")}')
    print(f'   PROCESS_INTERVAL_SECONDS: {os.getenv("PROCESS_INTERVAL_SECONDS")}')
    print(f'   MAX_CPU_PERCENT: {os.getenv("MAX_CPU_PERCENT")}')
    print(f'   MAX_RAM_GB: {os.getenv("MAX_RAM_GB")}')
    print(f'   TARGET_EVENT_SKIP_RATIO: {os.getenv("TARGET_EVENT_SKIP_RATIO")}')

    # Test loading into config
    config = {
        'optimized_mode': os.getenv('OPTIMIZED_MODE') == 'true',
        'event_driven': os.getenv('EVENT_DRIVEN_PROCESSING') == 'true',
        'single_symbol': os.getenv('SINGLE_SYMBOL'),
        'process_interval': int(os.getenv('PROCESS_INTERVAL_SECONDS', '30')),
        'cpu_limit': float(os.getenv('MAX_CPU_PERCENT', '88.0')),
        'ram_limit': float(os.getenv('MAX_RAM_GB', '3.86')),
        'skip_ratio_target': float(os.getenv('TARGET_EVENT_SKIP_RATIO', '0.7')),
        'technical_enabled': os.getenv('TECHNICAL_ANALYSIS_ENABLED') == 'true',
        'news_enabled': os.getenv('NEWS_ANALYSIS_ENABLED') == 'true',
        'whale_enabled': os.getenv('WHALE_TRACKING_ENABLED') == 'true',
        'monitoring_enabled': os.getenv('RESOURCE_MONITORING_ENABLED') == 'true',
    }

    print('\n✅ Config loaded successfully:')
    print(f'   Optimized mode: {config["optimized_mode"]}')
    print(f'   Event driven: {config["event_driven"]}')
    print(f'   Single symbol: {config["single_symbol"]}')
    print(f'   Process interval: {config["process_interval"]}s')
    print(f'   CPU limit: {config["cpu_limit"]}%')
    print(f'   RAM limit: {config["ram_limit"]}GB')
    print(f'   Skip ratio target: {config["skip_ratio_target"]}')
    print(f'   Technical analysis: {config["technical_enabled"]}')
    print(f'   News analysis: {config["news_enabled"]}')
    print(f'   Whale tracking: {config["whale_enabled"]}')
    print(f'   Resource monitoring: {config["monitoring_enabled"]}')

    # Validate critical settings
    assert config['optimized_mode'] == True, "OPTIMIZED_MODE should be true"
    assert config['event_driven'] == True, "EVENT_DRIVEN_PROCESSING should be true"
    assert config['single_symbol'] == 'BTC-USDT', "SINGLE_SYMBOL should be BTC-USDT"
    assert config['cpu_limit'] == 88.0, "CPU limit should be 88.0"
    assert config['ram_limit'] == 3.86, "RAM limit should be 3.86"
    assert config['skip_ratio_target'] == 0.7, "Skip ratio target should be 0.7"

    print('\n✅ Config & Defaults Task 2: PASSED')
    return True

if __name__ == "__main__":
    test_config_defaults()
