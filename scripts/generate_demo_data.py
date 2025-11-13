#!/usr/bin/env python3
"""
Generate realistic demo data for Supreme System V5 demo environment.

Generates 365 days of OHLCV data for multiple symbols with realistic
volatility and price movements.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os
from pathlib import Path
import argparse


def generate_ohlcv(symbol: str, days: int = 365, timeframe: str = '1min') -> pd.DataFrame:
    """
    Generate realistic OHLCV data for demo.
    
    Args:
        symbol: Trading symbol (e.g., 'BTC/USDT')
        days: Number of days to generate
        timeframe: Data timeframe ('1min', '5min', '1h', '1d')
    
    Returns:
        DataFrame with OHLCV data
    """
    # Base prices (realistic for each symbol)
    base_prices = {
        'BTC/USDT': 45000,
        'ETH/USDT': 2500,
        'SOL/USDT': 100,
        'BNB/USDT': 300,
        'XRP/USDT': 0.6,
        'ADA/USDT': 0.5,
        'DOGE/USDT': 0.08,
        'MATIC/USDT': 0.8,
        'DOT/USDT': 7.0,
        'AVAX/USDT': 35.0
    }
    
    # Timeframe multipliers (bars per day)
    timeframe_multipliers = {
        '1min': 24 * 60,
        '5min': 24 * 12,
        '15min': 24 * 4,
        '1h': 24,
        '4h': 6,
        '1d': 1
    }
    
    base = base_prices.get(symbol, 100)
    bars_per_day = timeframe_multipliers.get(timeframe, 24)
    total_bars = days * bars_per_day
    
    # Generate timestamps
    if timeframe == '1d':
        dates = pd.date_range(end=datetime.now(), periods=days, freq='D')
    elif timeframe == '1h':
        dates = pd.date_range(end=datetime.now(), periods=days * 24, freq='H')
    elif timeframe == '1min':
        dates = pd.date_range(end=datetime.now(), periods=days * 24 * 60, freq='1min')
    else:
        # Default to 1min and resample
        dates = pd.date_range(end=datetime.now(), periods=days * 24 * 60, freq='1min')
    
    # Generate price with realistic volatility
    # Use random walk with mean reversion
    volatility = 0.02  # 2% volatility per bar
    mean_reversion = 0.001  # Slight mean reversion
    
    returns = np.random.normal(0, volatility, len(dates))
    
    # Add mean reversion
    price_deviation = np.zeros(len(dates))
    for i in range(1, len(dates)):
        price_deviation[i] = price_deviation[i-1] * (1 - mean_reversion) + returns[i]
    
    # Generate prices
    prices = base * np.exp(np.cumsum(price_deviation))
    
    # Ensure prices stay positive and realistic
    prices = np.maximum(prices, base * 0.1)  # Min 10% of base
    prices = np.minimum(prices, base * 10)   # Max 10x base
    
    # Generate OHLCV
    df = pd.DataFrame({
        'timestamp': dates,
        'open': prices,
        'high': prices * (1 + abs(np.random.normal(0, 0.005, len(dates)))),
        'low': prices * (1 - abs(np.random.normal(0, 0.005, len(dates)))),
        'close': prices * (1 + np.random.normal(0, 0.002, len(dates))),
        'volume': np.random.uniform(1000, 10000, len(dates))
    })
    
    # Ensure high >= low and high/low are reasonable
    df['high'] = np.maximum(df['high'], df[['open', 'close']].max(axis=1))
    df['low'] = np.minimum(df['low'], df[['open', 'close']].min(axis=1))
    
    # Resample if needed
    if timeframe != '1min' and timeframe != '1h' and timeframe != '1d':
        df.set_index('timestamp', inplace=True)
        df = df.resample(timeframe).agg({
            'open': 'first',
            'high': 'max',
            'low': 'min',
            'close': 'last',
            'volume': 'sum'
        })
        df.reset_index(inplace=True)
    
    return df


def main():
    """Main function to generate demo data."""
    parser = argparse.ArgumentParser(
        description='Generate demo data for Supreme System V5'
    )
    parser.add_argument(
        '--symbols',
        type=str,
        default='BTC/USDT,ETH/USDT,SOL/USDT,BNB/USDT,XRP/USDT',
        help='Comma-separated list of symbols'
    )
    parser.add_argument(
        '--days',
        type=int,
        default=365,
        help='Number of days to generate (default: 365)'
    )
    parser.add_argument(
        '--timeframe',
        type=str,
        default='1min',
        choices=['1min', '5min', '15min', '1h', '4h', '1d'],
        help='Data timeframe (default: 1min)'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='data/demo',
        help='Output directory (default: data/demo)'
    )
    
    args = parser.parse_args()
    
    # Parse symbols
    symbols = [s.strip() for s in args.symbols.split(',')]
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"🚀 Generating demo data for {len(symbols)} symbols...")
    print(f"📅 Days: {args.days}")
    print(f"⏱️  Timeframe: {args.timeframe}")
    print(f"📁 Output: {output_dir}\n")
    
    total_candles = 0
    
    for symbol in symbols:
        print(f"📊 Generating data for {symbol}...")
        try:
            df = generate_ohlcv(symbol, days=args.days, timeframe=args.timeframe)
            
            # Save to CSV
            filename = symbol.replace('/', '_') + '.csv'
            filepath = output_dir / filename
            df.to_csv(filepath, index=False)
            
            total_candles += len(df)
            print(f"  ✅ Saved {len(df):,} candles to {filename}")
            print(f"     Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")
            print(f"     Price range: ${df['low'].min():.2f} - ${df['high'].max():.2f}")
            
        except Exception as e:
            print(f"  ❌ Error generating data for {symbol}: {e}")
    
    print(f"\n✅ Demo data generation complete!")
    print(f"📊 Total candles generated: {total_candles:,}")
    print(f"📁 Files saved to: {output_dir}")


if __name__ == '__main__':
    main()

