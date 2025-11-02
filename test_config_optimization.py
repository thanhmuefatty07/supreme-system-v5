#!/usr/bin/env python3
"""
Test Ultra-Optimized Config System
"""

from python.supreme_system_v5.config import ConfigManager

def test_config_manager():
    """Test the ultra-optimized configuration manager."""
    print("🎯 Ultra-Optimized Config System Test")
    print("=" * 50)

    # Initialize config manager
    config_manager = ConfigManager('.env.optimized')

    # Load and validate config
    config = config_manager.load_config()
    print("✅ Configuration loaded and validated")

    # Test key optimization parameters
    print("\n🔧 Core Optimization Settings:")
    print(f"   Optimized Mode: {config.get('OPTIMIZED_MODE')}")
    print(f"   Event Driven Processing: {config.get('EVENT_DRIVEN_PROCESSING')}")
    print(f"   Single Symbol: {config.get('SINGLE_SYMBOL')}")
    print(f"   CPU Limit: {config.get('MAX_CPU_PERCENT')}%")
    print(f"   Memory Limit: {config.get('MAX_RAM_GB')}GB")
    print(f"   Target Skip Ratio: {config.get('TARGET_EVENT_SKIP_RATIO')}")

    print("\n🎯 Advanced Performance Settings:")
    print(f"   EMA Period: {config.get('ema_period')}")
    print(f"   RSI Period: {config.get('rsi_period')}")
    print(f"   MACD Periods: {config.get('macd_fast')}/{config.get('macd_slow')}/{config.get('macd_signal')}")
    print(f"   Cache Enabled: {config.get('cache_enabled')}")
    print(f"   Cache TTL: {config.get('cache_ttl_seconds')}s")
    print(f"   Min Price Change: {config.get('min_price_change_pct')*100:.1f}%")
    print(f"   Volume Multiplier: {config.get('min_volume_multiplier')}x")

    # Test performance analysis (mock metrics)
    mock_metrics = {
        'avg_cpu_percent': 65.0,
        'avg_memory_gb': 2.8,
        'avg_indicator_latency_ms': 45.0,
        'cache_hit_ratio': 0.75,
        'skip_ratio': 0.8,
        'events_per_second': 85
    }

    print("\n📊 Performance Analysis:")
    analysis = config_manager.profiler.analyze_performance_impact(config, mock_metrics)

    print(f"   CPU Efficiency: {analysis['cpu_efficiency']['efficiency_score']:.2f}")
    print(f"   Memory Efficiency: {analysis['memory_efficiency']['efficiency_score']:.2f}")
    print(f"   Processing Efficiency: {analysis['processing_efficiency']['overall_efficiency']:.2f}")
    print(f"   Optimization Score: {analysis['optimization_score']:.1f}/100")

    if analysis['recommendations']:
        print(f"\n💡 Tuning Recommendations ({len(analysis['recommendations'])}):")
        for rec in analysis['recommendations'][:3]:  # Show top 3
            print(f"   • {rec['action']}: {rec['expected_impact']}")
    else:
        print("   ✅ No tuning recommendations needed")

    # Test config optimization
    optimized_config = config_manager.optimize_config(mock_metrics)
    print("\n🔄 Auto-Optimization Applied:")
    changes_made = sum(1 for k, v in optimized_config.items()
                      if k in config and config[k] != v)
    print(f"   Parameters adjusted: {changes_made}")

    # Save optimized config
    config_manager.save_optimized_config(optimized_config, '.env.hyper_optimized')

    print("\n✅ Ultra-Optimized Config System Test PASSED")
    return True

if __name__ == "__main__":
    test_config_manager()
