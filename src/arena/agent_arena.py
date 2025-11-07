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
    # ... (other Arena methods unchanged) ...
