#!/usr/bin/env python3
"""
Backtest Comparison Script: Adaptive Kelly vs Static Kelly vs Fixed Size

Compares three position sizing methods:
1. Adaptive Kelly: EWMA-based dynamic sizing with circuit breakers
2. Static Kelly: Fixed Kelly formula with static parameters
3. Fixed Size: Traditional fixed percentage sizing

Generates equity curves and performance metrics for comparison.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import logging
from typing import List, Dict, Any
from pathlib import Path

# Add src to path for imports
import sys
sys.path.append(str(Path(__file__).parent.parent))

from src.risk.adaptive_kelly import AdaptiveKellyRiskManager, RiskConfig

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class KellyComparisonBacktester:
    """
    Backtest engine for comparing position sizing methods.
    """

    def __init__(self, initial_capital: float = 10000.0):
        self.initial_capital = initial_capital

    def calculate_static_kelly(self, win_rate: float, reward_risk_ratio: float, mode: str = 'half') -> float:
        """Calculate static Kelly fraction."""
        if win_rate <= 0 or win_rate >= 1 or reward_risk_ratio <= 0:
            return 0.0

        kelly_fraction = win_rate - ((1 - win_rate) / reward_risk_ratio)

        # Apply mode scaling
        scaling = {'full': 1.0, 'half': 0.5, 'quarter': 0.25}.get(mode, 0.5)
        kelly_fraction *= scaling

        return max(0.0, kelly_fraction)

    def run_adaptive_kelly_simulation(self, trades_df: pd.DataFrame, mode: str = 'half') -> Dict[str, Any]:
        """
        Run simulation using Adaptive Kelly.

        Args:
            trades_df: DataFrame with columns ['pnl_pct', 'win_rate_true', 'rr_ratio_true']
            mode: Kelly mode ('full', 'half', 'quarter')

        Returns:
            Dict with equity curve and metrics
        """
        capital = self.initial_capital
        equity_curve = [capital]

        # Initialize Adaptive Kelly (UNLOCKED for true performance)
        config = RiskConfig(
            initial_win_rate=0.5,
            initial_reward_risk=2.0,
            max_daily_loss_pct=0.50,  # Very relaxed for backtest
            max_consecutive_losses=10,  # Much higher for backtest (vs 3 live)
            max_risk_per_trade=0.10,  # UNLOCKED: Allow up to 10% per trade (Kelly will scale it)
            max_position_pct=0.20      # Allow larger positions when conviction is high
        )
        risk_manager = AdaptiveKellyRiskManager(config=config, current_capital=capital)

        trades_executed = 0
        winning_trades = 0
        total_pnl = 0.0

        for idx, trade in trades_df.iterrows():
            # Check circuit breaker
            if not risk_manager.can_trade():
                logger.warning(f"Trade {idx}: Circuit breaker active - {risk_manager.halt_reason}")
                equity_curve.append(capital)
                continue

            # Get position size using Adaptive Kelly
            position_size = risk_manager.get_target_size(mode=mode)

            # Debug: log first few trades
            if idx < 5:
                logger.info(f"Trade {idx}: Kelly size ${position_size:.2f}, Can trade: {risk_manager.can_trade()}")

            if position_size <= 0:
                logger.warning(f"Trade {idx}: Kelly returned zero size")
                equity_curve.append(capital)
                continue

            # Simulate trade execution
            pnl_pct = trade['pnl_pct']
            pnl_amount = position_size * pnl_pct
            capital += pnl_amount
            total_pnl += pnl_amount

            # Update Adaptive Kelly with result
            was_win = pnl_amount > 0
            if was_win:
                winning_trades += 1
            trades_executed += 1

            risk_manager.update_performance(was_win, pnl_amount)
            risk_manager.current_capital = capital

            equity_curve.append(capital)

            # Reset daily counters periodically (simulate daily reset)
            if idx % 20 == 0:  # Every 20 trades = 1 day
                risk_manager.reset_daily()

        # Calculate metrics
        total_return = (capital - self.initial_capital) / self.initial_capital
        win_rate = winning_trades / trades_executed if trades_executed > 0 else 0
        max_drawdown = self._calculate_max_drawdown(equity_curve)
        sharpe_ratio = self._calculate_sharpe_ratio(equity_curve)

        return {
            'method': 'Adaptive Kelly',
            'equity_curve': equity_curve,
            'final_capital': capital,
            'total_return': total_return,
            'total_trades': trades_executed,
            'win_rate': win_rate,
            'total_pnl': total_pnl,
            'max_drawdown': max_drawdown,
            'sharpe_ratio': sharpe_ratio,
            'final_ewma_win_rate': risk_manager.ewma_win_rate,
            'final_ewma_rr_ratio': risk_manager.ewma_reward_risk,
            'circuit_breaker_triggers': risk_manager.is_halted
        }

    def run_static_kelly_simulation(self, trades_df: pd.DataFrame, mode: str = 'half') -> Dict[str, Any]:
        """
        Run simulation using Static Kelly (traditional Kelly formula).
        """
        capital = self.initial_capital
        equity_curve = [capital]

        trades_executed = 0
        winning_trades = 0
        total_pnl = 0.0

        for idx, trade in trades_df.iterrows():
            # Use true win rate and RR ratio for static Kelly
            win_rate = trade.get('win_rate_true', 0.5)
            rr_ratio = trade.get('rr_ratio_true', 2.0)

            kelly_fraction = self.calculate_static_kelly(win_rate, rr_ratio, mode)

            if kelly_fraction <= 0:
                equity_curve.append(capital)
                continue

            position_size = capital * kelly_fraction

            # Cap position size
            max_position = capital * 0.10  # 10% max
            position_size = min(position_size, max_position)

            # Execute trade
            pnl_pct = trade['pnl_pct']
            pnl_amount = position_size * pnl_pct
            capital += pnl_amount
            total_pnl += pnl_amount

            if pnl_amount > 0:
                winning_trades += 1
            trades_executed += 1

            equity_curve.append(capital)

        # Calculate metrics
        total_return = (capital - self.initial_capital) / self.initial_capital
        win_rate = winning_trades / trades_executed if trades_executed > 0 else 0
        max_drawdown = self._calculate_max_drawdown(equity_curve)
        sharpe_ratio = self._calculate_sharpe_ratio(equity_curve)

        return {
            'method': 'Static Kelly',
            'equity_curve': equity_curve,
            'final_capital': capital,
            'total_return': total_return,
            'total_trades': trades_executed,
            'win_rate': win_rate,
            'total_pnl': total_pnl,
            'max_drawdown': max_drawdown,
            'sharpe_ratio': sharpe_ratio
        }

    def run_fixed_size_simulation(self, trades_df: pd.DataFrame, fixed_pct: float = 0.02) -> Dict[str, Any]:
        """
        Run simulation using Fixed Size (traditional percentage-based sizing).
        """
        capital = self.initial_capital
        equity_curve = [capital]

        trades_executed = 0
        winning_trades = 0
        total_pnl = 0.0

        for idx, trade in trades_df.iterrows():
            # Fixed percentage of capital
            position_size = capital * fixed_pct

            # Execute trade
            pnl_pct = trade['pnl_pct']
            pnl_amount = position_size * pnl_pct
            capital += pnl_amount
            total_pnl += pnl_amount

            if pnl_amount > 0:
                winning_trades += 1
            trades_executed += 1

            equity_curve.append(capital)

        # Calculate metrics
        total_return = (capital - self.initial_capital) / self.initial_capital
        win_rate = winning_trades / trades_executed if trades_executed > 0 else 0
        max_drawdown = self._calculate_max_drawdown(equity_curve)
        sharpe_ratio = self._calculate_sharpe_ratio(equity_curve)

        return {
            'method': 'Fixed Size (2%)',
            'equity_curve': equity_curve,
            'final_capital': capital,
            'total_return': total_return,
            'total_trades': trades_executed,
            'win_rate': win_rate,
            'total_pnl': total_pnl,
            'max_drawdown': max_drawdown,
            'sharpe_ratio': sharpe_ratio
        }

    def _calculate_max_drawdown(self, equity_curve: List[float]) -> float:
        """Calculate maximum drawdown from equity curve."""
        peak = equity_curve[0]
        max_dd = 0.0

        for value in equity_curve:
            if value > peak:
                peak = value
            dd = (peak - value) / peak
            max_dd = max(max_dd, dd)

        return max_dd

    def _calculate_sharpe_ratio(self, equity_curve: List[float], risk_free_rate: float = 0.02) -> float:
        """Calculate Sharpe ratio from equity curve."""
        if len(equity_curve) < 2:
            return 0.0

        # Calculate daily returns
        returns = []
        for i in range(1, len(equity_curve)):
            daily_return = (equity_curve[i] - equity_curve[i-1]) / equity_curve[i-1]
            returns.append(daily_return)

        if not returns:
            return 0.0

        # Annualize (assuming daily data)
        avg_return = np.mean(returns)
        std_return = np.std(returns)

        if std_return == 0:
            return 0.0

        # Annualized Sharpe ratio
        sharpe = (avg_return - risk_free_rate/252) / std_return * np.sqrt(252)
        return sharpe

    def run_comparison(self, trades_df: pd.DataFrame, output_dir: str = 'backtest_results') -> Dict[str, Any]:
        """
        Run all three simulations and generate comparison report.
        """
        logger.info("Starting Kelly comparison backtest...")

        # Run all simulations
        results = {}

        logger.info("Running Adaptive Kelly simulation...")
        results['adaptive'] = self.run_adaptive_kelly_simulation(trades_df)

        logger.info("Running Static Kelly simulation...")
        results['static'] = self.run_static_kelly_simulation(trades_df)

        logger.info("Running Fixed Size simulation...")
        results['fixed'] = self.run_fixed_size_simulation(trades_df)

        # Generate plots and reports
        self._generate_comparison_plots(results, output_dir)
        self._generate_comparison_report(results, output_dir)

        logger.info("Backtest comparison completed!")
        return results

    def _generate_comparison_plots(self, results: Dict[str, Any], output_dir: str):
        """Generate comparison plots."""
        import os
        os.makedirs(output_dir, exist_ok=True)

        plt.figure(figsize=(15, 10))

        # Equity curves
        plt.subplot(2, 2, 1)
        for method, data in results.items():
            equity = data['equity_curve']
            plt.plot(equity, label=data['method'], linewidth=2)

        plt.title('Equity Curves Comparison')
        plt.xlabel('Trades')
        plt.ylabel('Portfolio Value ($)')
        plt.legend()
        plt.grid(True, alpha=0.3)

        # Returns comparison
        plt.subplot(2, 2, 2)
        methods = [data['method'] for data in results.values()]
        returns = [data['total_return'] * 100 for data in results.values()]

        bars = plt.bar(methods, returns, color=['blue', 'green', 'red'])
        plt.title('Total Returns Comparison')
        plt.ylabel('Total Return (%)')
        plt.xticks(rotation=45)

        # Add value labels on bars
        for bar, return_val in zip(bars, returns):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                    f'{return_val:.1f}%', ha='center', va='bottom')

        # Risk metrics
        plt.subplot(2, 2, 3)
        methods = [data['method'] for data in results.values()]
        max_dd = [data['max_drawdown'] * 100 for data in results.values()]

        bars = plt.bar(methods, max_dd, color=['blue', 'green', 'red'])
        plt.title('Maximum Drawdown Comparison')
        plt.ylabel('Max Drawdown (%)')
        plt.xticks(rotation=45)

        # Sharpe ratios
        plt.subplot(2, 2, 4)
        methods = [data['method'] for data in results.values()]
        sharpe = [data['sharpe_ratio'] for data in results.values()]

        bars = plt.bar(methods, sharpe, color=['blue', 'green', 'red'])
        plt.title('Sharpe Ratio Comparison')
        plt.ylabel('Sharpe Ratio')
        plt.xticks(rotation=45)

        plt.tight_layout()
        plt.savefig(f'{output_dir}/kelly_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()

        logger.info(f"Comparison plots saved to {output_dir}/kelly_comparison.png")

    def _generate_comparison_report(self, results: Dict[str, Any], output_dir: str):
        """Generate detailed comparison report."""
        import os
        os.makedirs(output_dir, exist_ok=True)

        report = "# Kelly Criterion Comparison Report\n\n"
        report += f"**Backtest Date:** {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
        report += f"**Initial Capital:** ${self.initial_capital:,.0f}\n"
        report += f"**Total Trades:** {len(results['adaptive']['equity_curve']) - 1}\n\n"

        # Performance table
        report += "## Performance Summary\n\n"
        report += "| Method | Final Capital | Total Return | Max Drawdown | Sharpe Ratio | Win Rate |\n"
        report += "|--------|---------------|--------------|--------------|--------------|----------|\n"

        for method, data in results.items():
            report += f"| {data['method']} | ${data['final_capital']:,.0f} | {data['total_return']*100:.1f}% | {data['max_drawdown']*100:.1f}% | {data['sharpe_ratio']:.2f} | {data['win_rate']*100:.1f}% |\n"

        report += "\n## Detailed Metrics\n\n"

        for method, data in results.items():
            report += f"### {data['method']}\n"
            report += f"- **Trades Executed:** {data['total_trades']}\n"
            report += f"- **Total P&L:** ${data['total_pnl']:,.0f}\n"

            if 'final_ewma_win_rate' in data:
                report += f"- **Final EWMA Win Rate:** {data['final_ewma_win_rate']:.3f}\n"
                report += f"- **Final EWMA R/R Ratio:** {data['final_ewma_rr_ratio']:.3f}\n"
                report += f"- **Circuit Breaker Active:** {data.get('circuit_breaker_triggers', False)}\n"

            report += "\n"

        # Save report
        with open(f'{output_dir}/kelly_comparison_report.md', 'w') as f:
            f.write(report)

        logger.info(f"Comparison report saved to {output_dir}/kelly_comparison_report.md")


def generate_synthetic_trades(num_trades: int = 1000) -> pd.DataFrame:
    """
    Generate synthetic trading data for backtesting.
    """
    np.random.seed(42)  # For reproducible results

    trades = []

    # Simulate realistic trading scenarios
    true_win_rate = 0.55  # 55% win rate
    true_rr_ratio = 2.0   # 2:1 reward to risk

    for i in range(num_trades):
        # Vary win rate slightly to simulate changing market conditions
        win_prob = true_win_rate + np.random.normal(0, 0.05)
        win_prob = np.clip(win_prob, 0.3, 0.8)  # Keep within reasonable bounds

        # Vary RR ratio
        rr_ratio = true_rr_ratio + np.random.normal(0, 0.2)
        rr_ratio = max(rr_ratio, 1.1)  # Minimum 1.1:1

        # Generate trade outcome
        is_win = np.random.random() < win_prob

        if is_win:
            # Win: gain between 0.5% to 3% of position
            pnl_pct = np.random.uniform(0.005, 0.03)
        else:
            # Loss: loss between 0.5% to 2% of position (respects RR ratio)
            pnl_pct = -np.random.uniform(0.005, 0.02)

        trades.append({
            'pnl_pct': pnl_pct,
            'win_rate_true': win_prob,
            'rr_ratio_true': rr_ratio,
            'is_win': is_win
        })

    return pd.DataFrame(trades)


def main():
    """Main execution function."""
    logger.info("Starting Kelly comparison backtest...")

    # Generate synthetic trading data
    logger.info("Generating synthetic trading data...")
    trades_df = generate_synthetic_trades(1000)

    # Run comparison
    backtester = KellyComparisonBacktester(initial_capital=10000.0)
    results = backtester.run_comparison(trades_df, output_dir='backtest_results')

    # Print summary
    logger.info("\n" + "="*60)
    logger.info("BACKTEST RESULTS SUMMARY")
    logger.info("="*60)

    for method, data in results.items():
        logger.info(f"\n{data['method']}:")
        logger.info(".2f")
        logger.info(".1f")
        logger.info(".1f")
        logger.info(".2f")

    logger.info("\n" + "="*60)
    logger.info("Backtest completed! Check 'backtest_results/' for detailed analysis.")
    logger.info("="*60)


if __name__ == '__main__':
    main()
