from scripts.backtest_kelly_comparison import KellyComparisonBacktester
import pandas as pd

# Create simple test data
trades_df = pd.DataFrame({
    'pnl_pct': [0.01, -0.005, 0.015],  # Small wins and losses
    'win_rate_true': [0.5, 0.5, 0.5],
    'rr_ratio_true': [2.0, 2.0, 2.0]
})

backtester = KellyComparisonBacktester()
result = backtester.run_adaptive_kelly_simulation(trades_df[:1])  # Just first trade

print('First trade result:')
print(f'Position size: ${result["equity_curve"][0]:.2f}')
print(f'Final capital: ${result["final_capital"]:.2f}')
print(f'Trades executed: {result["total_trades"]}')
