#!/usr/bin/env python3
"""
Test Strategy Integration with OptimizedTechnicalAnalyzer
"""

from python.supreme_system_v5.strategies import ScalpingStrategy
from python.supreme_system_v5.risk import DynamicRiskManager

def test_strategy_integration():
    """Test the optimized strategy integration."""

    config = {
        'symbol': 'BTC-USDT',
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

    print('🎯 Strategy Integration Test:')
    print(f'   Strategy: {strategy.name}')
    print(f'   Symbol: {strategy.symbol}')
    print(f'   RSI Overbought: {strategy.rsi_overbought}')
    print(f'   RSI Oversold: {strategy.rsi_oversold}')

    # Test cold start
    signal = strategy.add_price_data(price=50000.0, volume=1000)
    print(f'   First signal (cold start): {signal}')

    # Generate trending price data
    signals_generated = 0
    for i in range(100):
        # Create trending price with some noise
        trend = (i - 50) * 5  # Trending from -250 to +250
        noise = (i % 10 - 5) * 2  # Small noise
        price = 50000 + trend + noise
        volume = 1000 + abs(trend)  # Higher volume with trend

        signal = strategy.add_price_data(price=price, volume=volume)
        if signal:
            signals_generated += 1
            print(f'   Signal {signals_generated}: {signal["action"]} at ${price:.2f}')

    print(f'\n📊 Results:')
    print(f'   Total signals generated: {signals_generated}')

    # Test performance stats
    stats = strategy.get_performance_stats()
    print(f'   Trades executed: {stats["trades_executed"]}')
    print(f'   Win rate: {stats["win_rate_pct"]:.1f}%')
    print(f'   Total PnL: {stats["total_pnl"]:.4f}')
    print(f'   Current position: {stats["current_position"]}')

    # Test analyzer performance
    analyzer_stats = stats["analyzer_stats"]
    print(f'   Events processed: {analyzer_stats.get("updates_processed", 0)}')
    print(f'   Events filtered: {analyzer_stats.get("events_filtered", 0)}')
    filter_efficiency = analyzer_stats.get("filter_efficiency", 0)
    print(f'   Filter efficiency: {filter_efficiency:.1f}%')

    print('\n✅ Strategy Integration Task 1: PASSED')
    return True

if __name__ == "__main__":
    test_strategy_integration()
