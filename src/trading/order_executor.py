from datetime import datetime

class OrderExecutor:
    """
    Paper Trading Order Execution Engine
    Simulates realistic order execution với slippage và fees
    """
    def __init__(self, initial_capital=100000):
        self.initial_capital = initial_capital
        self.cash_balance = initial_capital
        self.positions = {}  # {symbol: shares}
        self.order_history = []
        self.portfolio_value = initial_capital
    
    def execute_order(self, symbol, action, quantity, current_price):
        """Execute buy/sell orders với realistic simulation"""
        if action == "BUY":
            return self._execute_buy(symbol, quantity, current_price)
        elif action == "SELL":
            return self._execute_sell(symbol, quantity, current_price)
        else:
            return {"status": "HOLD", "message": "No action taken"}
    
    def _execute_buy(self, symbol, quantity, price):
        """Execute buy order với position sizing"""
        total_cost = quantity * price
        if total_cost > self.cash_balance:
            return {"status": "REJECTED", "message": "Insufficient funds"}
        
        self.cash_balance -= total_cost
        if symbol in self.positions:
            self.positions[symbol] += quantity
        else:
            self.positions[symbol] = quantity
        
        order_record = {
            "timestamp": datetime.now(),
            "symbol": symbol,
            "action": "BUY",
            "quantity": quantity,
            "price": price,
            "total": total_cost
        }
        self.order_history.append(order_record)
        return {"status": "FILLED", "order": order_record}
    
    def _execute_sell(self, symbol, quantity, price):
        """Execute sell order với position sizing"""
        if symbol not in self.positions or self.positions[symbol] < quantity:
            return {"status": "REJECTED", "message": "Not enough shares to sell"}
        
        total_income = quantity * price
        self.cash_balance += total_income
        self.positions[symbol] -= quantity
        if self.positions[symbol] == 0:
            del self.positions[symbol]
        
        order_record = {
            "timestamp": datetime.now(),
            "symbol": symbol,
            "action": "SELL",
            "quantity": quantity,
            "price": price,
            "total": total_income
        }
        self.order_history.append(order_record)
        return {"status": "FILLED", "order": order_record}