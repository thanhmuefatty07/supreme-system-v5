from datetime import datetime

class PortfolioManager:
    """
    Real-time portfolio tracking và performance analytics
    """
    def __init__(self, order_executor):
        self.order_executor = order_executor
        self.performance_history = []
        self.daily_returns = []
        
    def update_portfolio_value(self, current_prices):
        """Calculate current portfolio value với real-time prices"""
        positions_value = 0
        for symbol, shares in self.order_executor.positions.items():
            if symbol in current_prices:
                positions_value += shares * current_prices[symbol]
        total_value = self.order_executor.cash_balance + positions_value
        self.order_executor.portfolio_value = total_value
        self.performance_history.append({
            "timestamp": datetime.now(),
            "total_value": total_value,
            "cash": self.order_executor.cash_balance,
            "positions": dict(self.order_executor.positions)
        })
        return total_value
    
    def calculate_performance_metrics(self):
        """Calculate key performance metrics"""
        if len(self.performance_history) < 2:
            return {}
        initial_value = self.performance_history[0]['total_value']
        current_value = self.performance_history[-1]['total_value']
        total_return = (current_value - initial_value) / initial_value * 100
        return {
            "total_return_percent": total_return,
            "current_portfolio_value": current_value,
            "cash_balance": self.order_executor.cash_balance,
            "active_positions": len(self.order_executor.positions),
            "total_trades": len(self.order_executor.order_history)
        }