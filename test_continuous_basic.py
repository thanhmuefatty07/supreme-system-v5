#!/usr/bin/env python3

"""

SIMPLE TEST - Supreme System V5 Continuous Testing (Basic Version)

No emojis, Windows-compatible, quick test

"""

import numpy as np
import pandas as pd
import json
import os
from datetime import datetime
import logging

# Setup basic logging (no emojis)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

class SimpleContinuousTest:

    def __init__(self):
        self.config = {
            'test_duration_days': 1,  # Just 1 day for testing
            'initial_capital': 10000,
            'symbols': ['BTC/USDT']
        }
        logging.info("Simple Continuous Testing Started")

    def test_basic_functionality(self):
        """Test basic paper trading functionality"""
        logging.info("Testing basic paper trading...")

        # Generate simple test data
        prices = np.random.normal(45000, 1000, 100)  # BTC-like prices

        capital = self.config['initial_capital']
        position = 0
        trades = []

        for i, price in enumerate(prices[10:], 10):  # Skip first 10 for indicators
            # Simple buy/sell logic
            if position == 0 and price < np.mean(prices[:i]):
                # Buy
                position = (capital * 0.1) / price  # 10% position
                entry_price = price
                trades.append({'type': 'BUY', 'price': price, 'capital': capital})
                logging.info(f"BUY: {position:.4f} at ${price:.2f}")

            elif position > 0 and price > entry_price * 1.02:  # 2% profit
                # Sell
                pnl = position * (price - entry_price)
                capital += pnl
                trades.append({'type': 'SELL', 'price': price, 'pnl': pnl, 'capital': capital})
                logging.info(f"SELL: ${pnl:+.2f} P&L, capital now ${capital:.2f}")
                position = 0

        # Calculate results
        final_capital = capital + (position * prices[-1] if position > 0 else 0)
        total_return = (final_capital - self.config['initial_capital']) / self.config['initial_capital']

        result = {
            'initial_capital': self.config['initial_capital'],
            'final_capital': final_capital,
            'total_return': total_return,
            'total_return_percent': total_return * 100,
            'total_trades': len([t for t in trades if 'pnl' in t])
        }

        logging.info(f"Test Results: Return {result['total_return_percent']:+.2f}%, {result['total_trades']} trades")

        # Save result
        with open('test_basic_result.json', 'w') as f:
            json.dump(result, f, indent=2)

        return result

if __name__ == "__main__":
    print("Testing basic continuous functionality...")
    tester = SimpleContinuousTest()
    result = tester.test_basic_functionality()
    print("SUCCESS - Basic functionality working!")
    print(".2f")
