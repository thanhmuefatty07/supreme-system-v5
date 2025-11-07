from src.trading.order_executor import OrderExecutor
from src.trading.portfolio_manager import PortfolioManager
from src.strategies.enhanced_trend_following import EnhancedTrendFollowingAgent
from src.strategies.mean_reversion import MeanReversionAgent
from src.arena.agent_arena import AgentArena
from src.data.live_data_feed import LiveDataFeed

def launch_complete_system():
    print("🚀 COMPLETE PAPER TRADING SYSTEM - DASHBOARD + MULTI-AGENT")
    print("==========================================================")
    arena = AgentArena(initial_capital=50000)
    trend_agent = EnhancedTrendFollowingAgent("trend_master", {'fast_window': 5, 'slow_window': 20})
    mean_reversion_agent = MeanReversionAgent("mean_reversion_pro", {'window': 14, 'threshold': 2.0})
    arena.register_agent(trend_agent)
    arena.register_agent(mean_reversion_agent)
    data_feed = LiveDataFeed(["AAPL", "MSFT", "GOOGL"])
    if data_feed.connect():
        print("✅ Data feed connected")
        historical_data = data_feed.get_historical_data("AAPL", "6mo")
        if historical_data is not None:
            print("🏟️ Starting multi-agent competition...")
            rankings = arena.run_competition(historical_data, ["AAPL"])
            print("\n🏆 COMPETITION RESULTS:")
            for i, rank in enumerate(rankings, 1):
                print(f"{i}. {rank['agent_id']}: {rank['return_percent']:.2f}% (${rank['final_value']:,.2f})")
            print("\n✅ System ready")
            print("📊 To launch dashboard: streamlit run dashboard/trading_dashboard.py")
            return {
                "arena": arena,
                "rankings": rankings,
                "status": "COMPLETE_SYSTEM_READY"
            }
if __name__ == "__main__":
    result = launch_complete_system()
    print(f"\n🎯 SYSTEM STATUS: {result['status']}")
