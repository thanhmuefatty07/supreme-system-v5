from src.data.live_data_feed import LiveDataFeed
from src.strategies.trend_following import TrendFollowingAgent
from src.orchestrator.safe_orchestrator import SafeOrchestrator

def launch_phase3():
    print("🚀 PHASE 3 - PAPER TRADING DEPLOYMENT")
    print("=====================================")
    
    # 1. Initialize data feed
    symbols = ["AAPL", "MSFT", "GOOGL", "TSLA"]
    data_feed = LiveDataFeed(symbols)
    
    if data_feed.connect():
        print("✅ Data feed connected successfully")
        
        # 2. Initialize trading agents
        trend_agent = TrendFollowingAgent(
            "trend_001", 
            {"fast_window": 10, "slow_window": 30}
        )
        print("✅ Trend Following Agent initialized")
        
        # 3. Test với historical data
        test_data = data_feed.get_historical_data("AAPL", "1mo")
        if test_data is not None:
            action = trend_agent.analyze_market(test_data)
            print(f"✅ Strategy test - AAPL signal: {action}")
        
        # 4. Initialize orchestrator
        orchestrator = SafeOrchestrator(memory_budget_mb=2200)
        print("✅ Safe Orchestrator ready for paper trading")
        
        return {
            "data_feed": data_feed,
            "agents": [trend_agent],
            "orchestrator": orchestrator,
            "status": "READY"
        }
    else:
        print("❌ Data connection failed")
        return {"status": "FAILED"}

if __name__ == "__main__":
    result = launch_phase3()
    print(f"\n🎯 PHASE 3 STATUS: {result['status']}")
