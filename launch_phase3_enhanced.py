from src.trading.order_executor import OrderExecutor
from src.trading.portfolio_manager import PortfolioManager
from src.strategies.enhanced_trend_following import EnhancedTrendFollowingAgent
from src.data.live_data_feed import LiveDataFeed

def launch_enhanced_paper_trading():
    print("🚀 ENHANCED PAPER TRADING - ORDER EXECUTION & PORTFOLIO")
    print("======================================================")
    order_executor = OrderExecutor(initial_capital=100000)
    portfolio_manager = PortfolioManager(order_executor)
    trend_agent = EnhancedTrendFollowingAgent("enhanced_trend_001", {})
    data_feed = LiveDataFeed(["AAPL"])
    if data_feed.connect():
        print("✅ Data feed connected")
        historical_data = data_feed.get_historical_data("AAPL", "3mo")
        if historical_data is not None:
            print("🧪 Running enhanced backtest...")
            for i in range(30, len(historical_data)):
                current_data = historical_data.iloc[:i]
                current_price = historical_data['Close'].iloc[i]
                signal = trend_agent.generate_trade_signal(
                    current_data, 
                    order_executor.portfolio_value
                )
                if signal["action"] != "HOLD":
                    result = order_executor.execute_order(
                        signal["symbol"],
                        signal["action"], 
                        signal["quantity"],
                        signal["price"]
                    )
                    if result["status"] == "FILLED":
                        print(f"📈 {signal['action']} {signal['quantity']} shares of {signal['symbol']} at ${signal['price']:.2f}")
                portfolio_value = portfolio_manager.update_portfolio_value(
                    {"AAPL": current_price}
                )
            metrics = portfolio_manager.calculate_performance_metrics()
            print(f"\n📊 BACKTEST RESULTS:")
            print(f"   Initial Capital: ${order_executor.initial_capital:,.2f}")
            print(f"   Final Portfolio: ${metrics['current_portfolio_value']:,.2f}")
            print(f"   Total Return: {metrics['total_return_percent']:.2f}%")
            print(f"   Total Trades: {metrics['total_trades']}")
            print(f"   Active Positions: {metrics['active_positions']}")
        return {
            "order_executor": order_executor,
            "portfolio_manager": portfolio_manager,
            "agent": trend_agent,
            "status": "ENHANCED_SYSTEM_READY"
        }
if __name__ == "__main__":
    result = launch_enhanced_paper_trading()
    print(f"\n🎯 ENHANCED SYSTEM STATUS: {result['status']}")
