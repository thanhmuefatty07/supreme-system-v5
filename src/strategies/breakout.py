import pandas as pd
import numpy as np
from src.agents.base_agent import BaseTradingAgent

class BreakoutStrategy(BaseTradingAgent):
    def __init__(self, agent_id, strategy_config):
        super().__init__(agent_id, strategy_config)
        self.resistance_window = strategy_config.get('resistance_window', 20)
        self.support_window = strategy_config.get('support_window', 20)
        self.atr_period = strategy_config.get('atr_period', 14)
        self.breakout_threshold = strategy_config.get('breakout_threshold', 1.0)
        self.position = 0
    def calculate_support_resistance(self, high_prices, low_prices):
        resistance = high_prices.rolling(window=self.resistance_window).max()
        support = low_prices.rolling(window=self.support_window).min()
        return resistance, support
    def calculate_atr(self, high_prices, low_prices, close_prices):
        tr1 = high_prices - low_prices
        tr2 = abs(high_prices - close_prices.shift())
        tr3 = abs(low_prices - close_prices.shift())
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(window=self.atr_period).mean()
        return atr
    def analyze_market(self, market_data):
        if len(market_data) < max(self.resistance_window, self.atr_period):
            return "HOLD"
        if isinstance(market_data.columns, pd.MultiIndex):
            high_prices = market_data[('High', market_data.columns.levels[1][0])]
            low_prices = market_data[('Low', market_data.columns.levels[1][0])]
            close_prices = market_data[('Close', market_data.columns.levels[1][0])]
        else:
            high_prices = market_data['High']
            low_prices = market_data['Low']
            close_prices = market_data['Close']
        resistance, support = self.calculate_support_resistance(high_prices, low_prices)
        atr = self.calculate_atr(high_prices, low_prices, close_prices)
        current_price = close_prices.iloc[-1]
        current_resistance = resistance.iloc[-1] if not pd.isna(resistance.iloc[-1]) else current_price
        current_support = support.iloc[-1] if not pd.isna(support.iloc[-1]) else current_price
        current_atr = atr.iloc[-1] if not pd.isna(atr.iloc[-1]) else 0
        if (current_price > current_resistance + (current_atr * self.breakout_threshold) and self.position <= 0):
            return "BUY"
        elif (current_price < current_support - (current_atr * self.breakout_threshold) and self.position >= 0):
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
