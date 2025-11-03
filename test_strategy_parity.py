#!/usr/bin/env python3
"""
Comprehensive parity validation tests for Supreme System V5 strategies.
Validates mathematical equivalence between optimized and reference implementations.
"""

import sys
import os
import numpy as np
import random
from typing import List, Dict, Any

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'python'))

from supreme_system_v5.strategies import ScalpingStrategy

def generate_historical_data(points: int = 1000, seed: int = 42) -> List[Dict[str, Any]]:
    """
    Generate realistic historical price data for testing.

    Args:
        points: Number of data points to generate
        seed: Random seed for reproducibility

    Returns:
        List of price data dictionaries
    """
    np.random.seed(seed)
    random.seed(seed)

    data = []
    base_price = 50000.0
    current_price = base_price

    for i in range(points):
        # Trend component
        trend = 100 * np.sin(i / 50)  # Slow trend

        # Noise component
        noise = np.random.normal(0, 50)

        # Micro movements
        micro = np.random.normal(0, 5)

        # Update price with momentum
        momentum = np.random.choice([-1, 0, 1], p=[0.3, 0.4, 0.3])
        current_price += momentum * 10 + trend * 0.01 + noise * 0.1 + micro

        # Ensure price stays positive
        current_price = max(current_price, 1000)

        # Generate volume
        base_volume = 1000
        volume_noise = np.random.lognormal(0, 0.5)
        volume = int(base_volume * volume_noise)

        data.append({
            'price': round(current_price, 2),
            'volume': volume,
            'timestamp': 1640995200 + i * 60  # Start from 2022-01-01, 1-minute intervals
        })

    return data

def test_strategy_parity():
    """Test parity between optimized and reference strategy implementations."""
    print("🚀 SUPREME SYSTEM V5 - STRATEGY PARITY VALIDATION")
    print("=" * 60)

    # Generate test data
    print("📊 Generating historical test data...")
    historical_data = generate_historical_data(points=2000, seed=42)
    print(f"   Generated {len(historical_data)} data points")

    # Initialize strategy with optimized configuration
    config = {
        'symbol': 'BTC-USDT',
        'ema_period': 14,
        'rsi_period': 14,
        'macd_fast': 12,
        'macd_slow': 26,
        'macd_signal': 9,
        'price_history_size': 200,
        'cache_enabled': True,
        'cache_ttl_seconds': 1.0,
        'event_config': {
            'min_price_change_pct': 0.001,
            'min_volume_multiplier': 3.0,
            'max_time_gap_seconds': 60
        }
    }

    print("🔧 Initializing ScalpingStrategy with optimized components...")
    strategy = ScalpingStrategy(config)

    # Run parity validation
    print("🔬 Running parity validation (tolerance: 1e-6)...")
    results = strategy.validate_parity_with_reference(historical_data, tolerance=1e-6)

    print(f"   Total data points: {results['total_points']}")
    print(f"   EMA parity: {'✅ PASS' if results['ema_parity'] else '❌ FAIL'}")
    print(f"   RSI parity: {'✅ PASS' if results['rsi_parity'] else '❌ FAIL'}")
    print(f"   MACD parity: {'✅ PASS' if results['macd_parity'] else '❌ FAIL'}")
    print(f"   Overall parity: {'✅ PASS' if results['parity_passed'] else '❌ FAIL'}")

    if not results['parity_passed']:
        print(f"\n⚠️  Parity violations found: {len(results['parity_violations'])}")
        print("Top 5 violations:")
        for violation in results['parity_violations'][:5]:
            print(".6f"
                  f"Ref: {violation['reference']:.6f}")
    else:
        print("\n✅ All indicators maintain mathematical parity with reference implementation!")

    # Additional validation: check that optimized components are actually used
    print("\n🔍 Component Integration Validation:")
    print(f"   OptimizedTechnicalAnalyzer: {'✅ Used' if hasattr(strategy, 'analyzer') else '❌ Missing'}")
    print(f"   SmartEventProcessor: {'✅ Used' if hasattr(strategy, 'event_processor') else '❌ Missing'}")
    print(f"   CircularBuffer: {'✅ Used' if hasattr(strategy, 'price_history') else '❌ Missing'}")

    # Test signal generation
    print("\n📈 Signal Generation Test:")
    signals_generated = 0
    for i, data_point in enumerate(historical_data[:100]):  # Test first 100 points
        signal = strategy.add_price_data(data_point['price'], data_point['volume'], data_point['timestamp'])
        if signal:
            signals_generated += 1

    print(f"   Signals generated from 100 data points: {signals_generated}")

    return results['parity_passed']

if __name__ == "__main__":
    success = test_strategy_parity()
    sys.exit(0 if success else 1)
