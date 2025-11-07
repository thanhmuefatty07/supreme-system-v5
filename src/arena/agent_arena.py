from datetime import datetime

class AgentArena:
    """
    Multi-agent competition arena - Agents compete for performance
    """
    def __init__(self, initial_capital=100000):
        self.initial_capital = initial_capital
        self.agents = {}
        self.agent_portfolios = {}
        self.performance_leaderboard = []
    
    def register_agent(self, agent):
        """Register agent vào competition"""
        agent_id = agent.agent_id
        self.agents[agent_id] = agent
        self.agent_portfolios[agent_id] = {
            'cash': self.initial_capital,
            'positions': {},
            'portfolio_value': self.initial_capital,
            'trade_history': []
        }
    
    def run_competition(self, market_data, symbols):
        """Run multi-agent competition trên cùng market data"""
        print(f"🏟️ Starting Agent Competition with {len(self.agents)} agents")
        for timestamp, data_point in market_data.iterrows():
            current_prices = {symbol: data_point['Close'] for symbol in symbols}
            for agent_id, agent in self.agents.items():
                portfolio = self.agent_portfolios[agent_id]
                signal = agent.generate_trade_signal(
                    market_data.loc[:timestamp], 
                    portfolio['portfolio_value']
                )
                if signal and signal.get('action') in ['BUY', 'SELL']:
                    self._execute_agent_trade(agent_id, signal, current_prices)
                self._update_agent_portfolio(agent_id, current_prices)
        return self.get_final_rankings()
    
    def _execute_agent_trade(self, agent_id, signal, current_prices):
        """Execute trade for specific agent"""
        portfolio = self.agent_portfolios[agent_id]
        symbol = signal['symbol']
        action = signal['action']
        quantity = signal['quantity']
        price = current_prices.get(symbol, signal['price'])
        if action == "BUY":
            cost = quantity * price
            if cost <= portfolio['cash']:
                portfolio['cash'] -= cost
                portfolio['positions'][symbol] = portfolio['positions'].get(symbol, 0) + quantity
                portfolio['trade_history'].append({
                    'timestamp': datetime.now(),
                    'agent_id': agent_id,
                    'symbol': symbol,
                    'action': action,
                    'quantity': quantity,
                    'price': price,
                    'total': cost
                })
        elif action == "SELL":
            if symbol in portfolio['positions'] and portfolio['positions'][symbol] >= quantity:
                income = quantity * price
                portfolio['cash'] += income
                portfolio['positions'][symbol] -= quantity
                if portfolio['positions'][symbol] == 0:
                    del portfolio['positions'][symbol]
                portfolio['trade_history'].append({
                    'timestamp': datetime.now(),
                    'agent_id': agent_id,
                    'symbol': symbol,
                    'action': action,
                    'quantity': quantity,
                    'price': price,
                    'total': income
                })
    
    def _update_agent_portfolio(self, agent_id, current_prices):
        """Update agent portfolio value"""
        portfolio = self.agent_portfolios[agent_id]
        positions_value = 0
        for symbol, shares in portfolio['positions'].items():
            if symbol in current_prices:
                positions_value += shares * current_prices[symbol]
        portfolio['portfolio_value'] = portfolio['cash'] + positions_value
    
    def get_final_rankings(self):
        """Get final performance rankings"""
        rankings = []
        for agent_id, portfolio in self.agent_portfolios.items():
            return_pct = (portfolio['portfolio_value'] - self.initial_capital) / self.initial_capital * 100
            rankings.append({
                'agent_id': agent_id,
                'final_value': portfolio['portfolio_value'],
                'return_percent': return_pct,
                'total_trades': len(portfolio['trade_history'])
            })
        return sorted(rankings, key=lambda x: x['return_percent'], reverse=True)
