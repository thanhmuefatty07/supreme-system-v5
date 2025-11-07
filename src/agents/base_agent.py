class BaseTradingAgent:
    """Foundation class cho all AI trading agents"""
    def __init__(self, agent_id, strategy_config):
        self.agent_id = agent_id
        self.strategy_config = strategy_config
        self.performance_metrics = {}

    def analyze_market(self, market_data):
        """Core analysis method cho agents"""
        pass
