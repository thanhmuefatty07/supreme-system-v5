import pandas as pd
import numpy as np
from src.agents.base_agent import BaseTradingAgent

class MomentumStrategy(BaseTradingAgent):
    def __init__(self, agent_id, strategy_config):
        super().__init__(agent_id, strategy_config)
        self.momentum_window = strategy_config.get('momentum_window', 10)
        self.volume_window = strategy_config.get('volume_window', 5)
        self.rsi_period = strategy_config.get('rsi_period', 14)
        self.position = 0
    def calculate_momentum(self, prices):
        returns = prices.pct_change(periods=self.momentum_window)
        momentum = returns.rolling(window=3).mean()
        return momentum
    def calculate_volume_surge(self, volume_data):
        avg_volume = volume_data.rolling(window=self.volume_window).mean()
        volume_ratio = volume_data / avg_volume
        return volume_ratio
    def calculate_rsi(self, prices):
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=self.rsi_period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=self.rsi_period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    def analyze_market(self, market_data):
        if len(market_data) < max(self.momentum_window, self.rsi_period):
            return "HOLD"
        if isinstance(market_data.columns, pd.MultiIndex):
            close_prices = market_data[('Close', market_data.columns.levels[1][0])]
            volume_data = market_data[('Volume', market_data.columns.levels[1][0])]
        else:
            close_prices = market_data['Close']
            volume_data = market_data['Volume']
        momentum = self.calculate_momentum(close_prices)
        volume_surge = self.calculate_volume_surge(volume_data)
        rsi = self.calculate_rsi(close_prices)
        current_momentum = momentum.iloc[-1] if not pd.isna(momentum.iloc[-1]) else 0
        current_volume_ratio = volume_surge.iloc[-1] if not pd.isna(volume_surge.iloc[-1]) else 1
        current_rsi = rsi.iloc[-1] if not pd.isna(rsi.iloc[-1]) else 50
        if (current_momentum > 0.02 and current_volume_ratio > 1.5 and current_rsi < 70 and self.position <= 0):
            return "BUY"
        elif (current_momentum < -0.02 and current_volume_ratio > 1.5 and current_rsi > 30 and self.position >= 0):
            return "SELL"
        else:
            return "HOLD"
    def generate_trade_signal(self, market_data, portfolio_value, symbol=None):
        base_signal = self.analyze_market(market_data)
        if base_signal in ["BUY", "SELL"]:
            if isinstance(market_data.columns, pd.MultiIndex):
                current_price = market_data[('Close', market_data.columns.levels[1][0])].iloc[-1]
            else:
                current_price = market_data['Close'].iloc[-1]
            position_size = self.calculate_position_size(portfolio_value, current_price)
            return {
                "action": base_signal,
                "symbol": symbol or "AAPL",
                "quantity": position_size,
                "price": current_price,
                "timestamp": pd.Timestamp.now(),
                "agent_id": self.agent_id
            }
        else:
            return {"action": "HOLD"}
    def calculate_position_size(self, portfolio_value, current_price, risk_per_trade=0.02):
        risk_amount = portfolio_value * risk_per_trade
        position_size = risk_amount / current_price
        return int(position_size)
