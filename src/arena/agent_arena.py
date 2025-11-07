from datetime import datetime

class AgentArena:
    """
    Enhanced Multi-Agent Competition Arena với Multi-Symbol Support
    """
    def __init__(self, initial_capital=100000, symbols=None):
        self.initial_capital = initial_capital
        self.agents = {}
        self.agent_portfolios = {}
        self.performance_leaderboard = []
        self.symbols = symbols or ["AAPL", "MSFT", "GOOGL", "TSLA", "NVDA", "AMD"]

    def register_agent(self, agent):
        """Register agent vào competition"""
        agent_id = agent.agent_id
        self.agents[agent_id] = agent
        print(f"✅ Registered agent: {agent_id}")

    def run_multi_symbol_competition(self, market_data_dict):
        print(f"🏟️ Starting Multi-Symbol Competition: {len(self.symbols)} symbols")
        # Initialize portfolios for each agent
        for agent_id in self.agents:
            self.agent_portfolios[agent_id] = {
                'cash': self.initial_capital,
                'positions': {symbol: 0 for symbol in self.symbols},
                'portfolio_value': self.initial_capital,
                'trade_history': []
            }
        timestamps = self._get_common_timestamps(market_data_dict)
        for timestamp in timestamps:
            current_prices = {}
            for symbol in self.symbols:
                if symbol in market_data_dict and timestamp in market_data_dict[symbol].index:
                    current_prices[symbol] = market_data_dict[symbol].loc[timestamp, 'Close']
            for agent_id, agent in self.agents.items():
                portfolio = self.agent_portfolios[agent_id]
                for symbol in self.symbols:
                    if symbol in market_data_dict:
                        symbol_data = market_data_dict[symbol].loc[:timestamp]
                        if len(symbol_data) > 20:
                            signal = agent.generate_trade_signal(
                                symbol_data, 
                                portfolio['portfolio_value'],
                                symbol=symbol
                            )
                            if signal and signal['action'] in ['BUY', 'SELL']:
                                self._execute_agent_trade(agent_id, signal, current_prices)
                self._update_agent_portfolio(agent_id, current_prices)
            if len(timestamps) % 20 == 0:
                self.update_leaderboard()
        return self.get_final_rankings()
    
    def _get_common_timestamps(self, market_data_dict):
        common_timestamps = None
        for symbol, data in market_data_dict.items():
            if common_timestamps is None:
                common_timestamps = set(data.index)
            else:
                common_timestamps = common_timestamps.intersection(set(data.index))
        return sorted(list(common_timestamps))

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

    def update_leaderboard(self):
        """Update performance leaderboard"""
        current_rankings = []
        for agent_id, portfolio in self.agent_portfolios.items():
            return_percent = ((portfolio['portfolio_value'] - self.initial_capital) / self.initial_capital) * 100
            total_trades = len(portfolio['trade_history'])
            current_rankings.append({
                'agent_id': agent_id,
                'portfolio_value': portfolio['portfolio_value'],
                'return_percent': return_percent,
                'total_trades': total_trades
            })
        self.performance_leaderboard = sorted(current_rankings, key=lambda x: x['return_percent'], reverse=True)

    def get_final_rankings(self):
        """Get final competition rankings"""
        self.update_leaderboard()
        return self.performance_leaderboard
