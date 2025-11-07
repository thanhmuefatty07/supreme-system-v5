from src.agents.base_agent import BaseTradingAgent
import pandas as pd
import numpy as np
from datetime import datetime

class TrendFollowingAgent(BaseTradingAgent):
    """
    Trend Following Strategy - Agent đầu tiên
    Simple moving average crossover
    """
    def __init__(self, agent_id, strategy_config):
        super().__init__(agent_id, strategy_config)
        self.fast_window = strategy_config.get('fast_window', 10)
        self.slow_window = strategy_config.get('slow_window', 30)
        self.position = 0
        self.portfolio_value = 100000  # Starting capital
    
    def calculate_sma(self, data, window):
        """Calculate Simple Moving Average"""
        return data.rolling(window=window).mean()
    
    def analyze_market(self, market_data):
        """
        Core trend following logic
        Buy khi fast SMA > slow SMA, Sell khi ngược lại
        """
        if len(market_data) < self.slow_window:
            return "HOLD"  # Không đủ data
        
        close_prices = market_data['Close'] if 'Close' in market_data else market_data
        
        fast_sma = self.calculate_sma(close_prices, self.fast_window)
        slow_sma = self.calculate_sma(close_prices, self.slow_window)
        
        current_fast = fast_sma.iloc[-1]
        current_slow = slow_sma.iloc[-1]
        
        if current_fast > current_slow and self.position <= 0:
            return "BUY"
        elif current_fast < current_slow and self.position >= 0:
            return "SELL"
        else:
            return "HOLD"

    def generate_trade_signal(self, market_data, portfolio_value, symbol=None):
        """Generate trade signal với position sizing"""
        base_signal = self.analyze_market(market_data)
        if base_signal in ["BUY", "SELL"]:
            # Handle MultiIndex columns from yfinance
            if isinstance(market_data.columns, pd.MultiIndex):
                current_price = market_data[('Close', symbol or 'AAPL')].iloc[-1]
            else:
                current_price = market_data['Close'].iloc[-1] if 'Close' in market_data.columns else market_data.iloc[-1]
            position_size = int((portfolio_value * 0.02) / current_price)
            return {
                "action": base_signal,
                "symbol": symbol or "AAPL",
                "quantity": position_size,
                "price": current_price,
                "timestamp": datetime.now()
            }
        return None