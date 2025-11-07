from src.arena.agent_arena import AgentArena
from src.strategies.trend_following import TrendFollowingAgent
from src.strategies.mean_reversion import MeanReversionAgent
from src.strategies.momentum import MomentumStrategy
from src.strategies.breakout import BreakoutStrategy
from src.data.live_data_feed import LiveDataFeed
import pandas as pd

def launch_enhanced_multi_symbol_system():
    print("🚀 ENHANCED MULTI-SYMBOL TRADING SYSTEM")
    print("========================================")
    tech_symbols = ["AAPL", "MSFT", "GOOGL", "TSLA", "NVDA", "AMD"]
    arena = AgentArena(initial_capital=100000, symbols=tech_symbols)
    trend_agent = TrendFollowingAgent("trend_master", {'fast_window': 10, 'slow_window': 30})
    mean_reversion_agent = MeanReversionAgent("mean_reversion_pro", {'window': 14, 'threshold': 2.0})
    momentum_agent = MomentumStrategy("momentum_trader", {'momentum_window': 10, 'volume_window': 5})
    breakout_agent = BreakoutStrategy("breakout_expert", {'resistance_window': 20, 'breakout_threshold': 1.0})
    arena.register_agent(trend_agent)
    arena.register_agent(mean_reversion_agent)
    arena.register_agent(momentum_agent)
    arena.register_agent(breakout_agent)
    print(f"✅ Registered {len(arena.agents)} trading agents")
    print(f"📊 Trading symbols: {tech_symbols}")
    data_feed = LiveDataFeed(tech_symbols)
    market_data_dict = {}
    if data_feed.connect():
        print("✅ Multi-symbol data feed connected")
        for symbol in tech_symbols:
            historical_data = data_feed.get_historical_data(symbol, "6mo")
            if historical_data is not None:
                market_data_dict[symbol] = historical_data
                print(f"   📈 {symbol}: {len(historical_data)} records")
        if len(market_data_dict) == len(tech_symbols):
            print("🏟️ Starting Enhanced Multi-Symbol Competition...")
            rankings = arena.run_multi_symbol_competition(market_data_dict)
            print("\n🏆 ENHANCED COMPETITION RESULTS")
            print("=" * 50)
            for i, rank in enumerate(rankings, 1):
                print(f"{i}. {rank['agent_id']:20} {rank['return_percent']:6.2f}% "
                      f"(${rank['portfolio_value']:8.2f}) - {rank.get('total_trades', 0)} trades")
            print("\n📊 PERFORMANCE ANALYSIS")
            best_agent = rankings[0]
            worst_agent = rankings[-1]
            print(f"🏅 Best Performer: {best_agent['agent_id']} "
                  f"({best_agent['return_percent']:.2f}%)")
            print(f"📉 Worst Performer: {worst_agent['agent_id']} "
                  f"({worst_agent['return_percent']:.2f}%)")
            print(f"📈 Performance Spread: "
                  f"{best_agent['return_percent'] - worst_agent['return_percent']:.2f}%")
            return {
                "arena": arena,
                "rankings": rankings,
                "symbols": tech_symbols,
                "status": "ENHANCED_SYSTEM_OPERATIONAL"
            }
        else:
            print("❌ Failed to fetch data for all symbols")
            return {"status": "DATA_FETCH_FAILED"}
    else:
        print("❌ Data feed connection failed")
        return {"status": "CONNECTION_FAILED"}

if __name__ == "__main__":
    result = launch_enhanced_multi_symbol_system()
    print(f"\n🎯 FINAL STATUS: {result['status']}")
