# Simple performance metrics for backtesting/minimal agent runs
class BasicMetrics:
    def __init__(self):
        self.agent_scores = {}
    def update(self, agent_id, metrics):
        self.agent_scores[agent_id] = metrics
    def summary(self):
        return self.agent_scores
