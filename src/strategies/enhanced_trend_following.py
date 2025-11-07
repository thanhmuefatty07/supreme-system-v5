from src.strategies.trend_following import TrendFollowingAgent
from datetime import datetime

class EnhancedTrendFollowingAgent(TrendFollowingAgent):
    """
    Enhanced trend following với position sizing và risk management
    """
    def calculate_position_size(self, portfolio_value, current_price, risk_per_trade=0.02):
        """Calculate position size based on risk management"""
        risk_amount = portfolio_value * risk_per_trade
        position_size = risk_amount / current_price
        return int(position_size)
    
    def generate_trade_signal(self, market_data, portfolio_value):
        """Generate enhanced trade signal với position sizing"""
        base_signal = self.analyze_market(market_data)
        if base_signal in ["BUY", "SELL"]:
            current_price = market_data['Close'].iloc[-1]
            position_size = self.calculate_position_size(portfolio_value, current_price)
            return {
                "action": base_signal,
                "symbol": "AAPL",
                "quantity": position_size,
                "price": current_price,
                "timestamp": datetime.now()
            }
        else:
            return {"action": "HOLD"}