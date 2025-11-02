#!/usr/bin/env python3
"""
Simple Load Test for Supreme System V5.
"""

import asyncio
import time
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'python'))

from supreme_system_v5.strategies import ScalpingStrategy
from supreme_system_v5.risk import DynamicRiskManager

def generate_price_feed(symbol: str, duration_minutes: int, tick_rate: int) -> list:
    """Generate realistic price feed."""
    import math
    import random

    total_ticks = duration_minutes * 60 * tick_rate
    price_data = []

    base_price = 50000.0
    current_time = time.time()

    for i in range(total_ticks):
        trend = 200 * math.sin(i / (tick_rate * 60))
        noise = 20 * (random.random() - 0.5)
        micro_noise = 0.1 * (random.random() - 0.5)

        price = base_price + trend + noise + micro_noise
        volume = 100 + random.randint(0, 200)

        price_data.append({
            'price': round(price, 2),
            'volume': volume,
            'timestamp': current_time + (i / tick_rate)
        })

    return price_data

async def run_load_test(symbol: str, tick_rate: int, duration_minutes: int):
    """Run simple load test."""
    print(f"Supreme System V5 - Load Test: {symbol}")
    print(f"Duration: {duration_minutes} min, Rate: {tick_rate} ticks/sec")

    # Initialize components
    config = {
        'symbol': symbol,
        'position_size_pct': 0.02,
        'stop_loss_pct': 0.01,
        'take_profit_pct': 0.02,
        'ema_period': 14,
        'rsi_period': 14,
        'macd_fast': 12,
        'macd_slow': 26,
        'macd_signal': 9,
        'price_history_size': 100,
        'event_config': {
            'min_price_change_pct': 0.0005,
            'min_volume_multiplier': 2.0,
            'max_time_gap_seconds': 30
        }
    }

    risk_config = {
        'base_position_size_pct': 0.02,
        'max_position_size_pct': 0.10,
        'base_leverage': 5.0,
        'max_leverage': 50.0,
        'max_portfolio_exposure': 0.50,
        'high_confidence_threshold': 0.75,
        'medium_confidence_threshold': 0.60,
        'low_confidence_threshold': 0.45
    }

    risk_manager = DynamicRiskManager(risk_config)
    strategy = ScalpingStrategy(config, risk_manager)

    # Generate test data
    price_feed = generate_price_feed(symbol, duration_minutes, tick_rate)
    print(f"Generated {len(price_feed)} price updates")

    # Run test
    start_time = time.time()
    signals_generated = 0
    events_processed = 0

    for tick in price_feed:
        signal = strategy.add_price_data(tick['price'], tick['volume'], tick['timestamp'])
        events_processed += 1
        if signal:
            signals_generated += 1

    test_duration = time.time() - start_time
    ticks_per_second = len(price_feed) / max(test_duration, 0.001)

    print("\nResults:")
    print(f"  Events processed: {events_processed}")
    print(f"  Signals generated: {signals_generated}")
    print(f"  Test duration: {test_duration:.2f}s")
    print(f"  Ticks per second: {ticks_per_second:.1f}")
    # Acceptance criteria
    criteria_passed = 0
    criteria_total = 0

    def check_criteria(name, condition, target):
        nonlocal criteria_passed, criteria_total
        criteria_total += 1
        status = "PASS" if condition else "FAIL"
        print(f"  {status}: {name} - {target}")
        if condition:
            criteria_passed += 1

    check_criteria("Tick processing rate", ticks_per_second >= tick_rate * 0.95,
                  ".1f")
    check_criteria("Signal generation", signals_generated >= 0, "At least 0 signals")

    print(f"\nAcceptance Criteria: {criteria_passed}/{criteria_total} passed")
    return criteria_passed == criteria_total

def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(description='Simple Load Test')
    parser.add_argument('--symbol', default='BTC-USDT', help='Trading symbol')
    parser.add_argument('--rate', type=int, default=10, help='Ticks per second')
    parser.add_argument('--duration-min', type=int, default=1, help='Duration in minutes')

    args = parser.parse_args()

    # Run test
    success = asyncio.run(run_load_test(args.symbol, args.rate, args.duration_min))
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()
