import pandas as pd
import numpy as np
from src.agents.base_agent import BaseTradingAgent
from datetime import datetime

class MeanReversionAgent(BaseTradingAgent):
    """
    Mean Reversion Strategy - Dựa trên RSI và Bollinger Bands
    """
    def __init__(self, agent_id, strategy_config):
        super().__init__(agent_id, strategy_config)
        self.window = strategy_config.get('window', 14)
        self.threshold = strategy_config.get('threshold', 2.0)
        self.position = 0
        self.portfolio_value = 100000
    
    def calculate_rsi(self, prices):
        """Calculate RSI indicator"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=self.window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=self.window).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def calculate_bollinger_bands(self, prices):
        """Calculate Bollinger Bands"""
        sma = prices.rolling(window=self.window).mean()
        std = prices.rolling(window=self.window).std()
        upper_band = sma + (std * self.threshold)
        lower_band = sma - (std * self.threshold)
        return upper_band, sma, lower_band
    
    def analyze_market(self, market_data):
        """Mean reversion logic với RSI và Bollinger Bands"""
        if len(market_data) < self.window:
            return "HOLD"
        close_prices = market_data['Close'] if 'Close' in market_data else market_data
        rsi = self.calculate_rsi(close_prices)
        upper_band, middle_band, lower_band = self.calculate_bollinger_bands(close_prices)
        current_price = close_prices.iloc[-1]
        current_rsi = rsi.iloc[-1] if not pd.isna(rsi.iloc[-1]) else 50
        current_lower = lower_band.iloc[-1] if not pd.isna(lower_band.iloc[-1]) else current_price
        current_upper = upper_band.iloc[-1] if not pd.isna(upper_band.iloc[-1]) else current_price
        if current_price < current_lower and current_rsi < 30 and self.position <= 0:
            return "BUY"
        elif current_price > current_upper and current_rsi > 70 and self.position >= 0:
            return "SELL"
        else:
            return "HOLD"
    
    def generate_trade_signal(self, market_data, portfolio_value, symbol=None):
        """Generate trade signal với position sizing"""
        self.portfolio_value = portfolio_value
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
        else:
            return {"action": "HOLD"}
