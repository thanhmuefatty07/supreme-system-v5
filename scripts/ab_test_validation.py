#!/usr/bin/env python3
"""
🚀 Supreme System V5 - A/B Testing Validation

Statistical validation of enhanced strategy improvements:
- Enhanced vs Baseline strategy comparison
- Statistical significance testing (p<0.05)
- Performance metrics validation
- Risk-adjusted returns analysis

Tests multiple market conditions and validates improvements.
"""

import sys
import os
import json
import random
import statistics
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Tuple
# from scipy import stats  # For statistical significance testing - using basic stats instead

# Add python path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'python'))

from supreme_system_v5.strategies import ScalpingStrategy


class ABTestValidator:
    """A/B testing framework for strategy validation"""

    def __init__(self, symbol: str = "ETH-USDT"):
        self.symbol = symbol
        self.output_dir = Path("run_artifacts")
        self.output_dir.mkdir(exist_ok=True)
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Test configurations
        self.baseline_config = {
            'symbol': symbol,
            'position_size_pct': 0.02,  # Standard 2% per trade
            'stop_loss_pct': 0.01,      # 1% stop loss
            'take_profit_pct': 0.02,    # 2% take profit
            'ema_period': 14,
            'rsi_period': 14,
            'macd_fast': 12,
            'macd_slow': 26,
            'macd_signal': 9
        }

        self.enhanced_config = {
            'symbol': symbol,
            'position_size_pct': 0.015,  # Conservative 1.5% per trade
            'stop_loss_pct': 0.008,      # 0.8% stop loss
            'take_profit_pct': 0.012,    # 1.2% take profit
            'ema_period': 14,
            'rsi_period': 14,
            'macd_fast': 12,
            'macd_slow': 26,
            'macd_signal': 9
        }

        # Test parameters
        self.num_tests = 30  # Multiple runs for statistical significance
        self.test_duration_hours = 4.0  # 4 hours per test
        self.interval_seconds = 60  # 1-minute intervals

        # Results storage
        self.baseline_results = []
        self.enhanced_results = []

    def run_ab_test(self) -> Dict[str, Any]:
        """Execute comprehensive A/B testing"""
        print("🔬 SUPREME SYSTEM V5 - A/B TESTING VALIDATION")
        print("=" * 65)
        print(f"Symbol: {self.symbol}")
        print(f"Tests: {self.num_tests} runs per strategy")
        print(f"Duration: {self.test_duration_hours} hours per test")
        print()

        # Run baseline tests
        print("📊 Running baseline strategy tests...")
        for i in range(self.num_tests):
            if (i + 1) % 5 == 0:
                print(f"   Completed {i + 1}/{self.num_tests} baseline tests")
            result = self._run_single_test(self.baseline_config, f"baseline_{i+1}")
            self.baseline_results.append(result)

        # Run enhanced tests
        print("🚀 Running enhanced strategy tests...")
        for i in range(self.num_tests):
            if (i + 1) % 5 == 0:
                print(f"   Completed {i + 1}/{self.num_tests} enhanced tests")
            result = self._run_single_test(self.enhanced_config, f"enhanced_{i+1}")
            self.enhanced_results.append(result)

        print("\n✅ A/B testing completed!")

        # Analyze results
        analysis = self._analyze_ab_results()

        # Save comprehensive results
        results = {
            'metadata': {
                'symbol': self.symbol,
                'num_tests': self.num_tests,
                'test_duration_hours': self.test_duration_hours,
                'baseline_config': self.baseline_config,
                'enhanced_config': self.enhanced_config,
                'timestamp': datetime.now().isoformat()
            },
            'baseline_results': self.baseline_results,
            'enhanced_results': self.enhanced_results,
            'analysis': analysis
        }

        self._save_results(results)
        self._generate_report(results)

        return results

    def _run_single_test(self, config: Dict[str, Any], test_id: str) -> Dict[str, Any]:
        """Run a single backtest with given configuration"""
        # Create strategy instance
        strategy = ScalpingStrategy(config)

        # Generate market data for this test
        market_data = self._generate_test_market_data()

        # Track performance
        trades = []
        portfolio_value = 10000.0  # Starting capital

        # Process each data point
        for data_point in market_data:
            signal = strategy.add_price_data(
                data_point['price'],
                data_point['volume'],
                data_point['timestamp']
            )

            if signal and signal['action'] in ['BUY', 'SELL']:
                trades.append({
                    'action': signal['action'],
                    'price': data_point['price'],
                    'timestamp': data_point['timestamp'],
                    'pnl': signal.get('pnl', 0),
                    'signal_quality': signal.get('signal_quality', 0)
                })

                # Update portfolio value (simplified)
                if 'pnl' in signal:
                    portfolio_value += signal['pnl']

        # Calculate test metrics
        test_result = self._calculate_test_metrics(trades, portfolio_value, test_id)
        return test_result

    def _generate_test_market_data(self) -> List[Dict[str, Any]]:
        """Generate realistic market data for testing"""
        data_points = int((self.test_duration_hours * 3600) / self.interval_seconds)

        # Vary market conditions across tests
        base_price = 45000 + random.uniform(-2000, 2000)  # Vary starting price
        current_price = base_price

        # Random market regime for this test
        regimes = ['trending_bullish', 'trending_bearish', 'volatile_sideways', 'ranging']
        regime = random.choice(regimes)

        # Regime parameters
        regime_params = {
            'trending_bullish': {'trend': 0.0001, 'volatility': 0.002},
            'trending_bearish': {'trend': -0.0001, 'volatility': 0.002},
            'volatile_sideways': {'trend': 0.00002, 'volatility': 0.008},
            'ranging': {'trend': 0.000005, 'volatility': 0.001}
        }

        params = regime_params[regime]

        data = []
        for i in range(data_points):
            timestamp = i * self.interval_seconds

            # Generate price movement
            trend_component = params['trend'] * random.gauss(1, 0.5)
            volatility_component = random.gauss(0, params['volatility'])
            micro_noise = random.gauss(0, 0.0005)

            price_change = trend_component + volatility_component + micro_noise
            current_price *= (1 + price_change)

            # Generate volume
            base_volume = random.uniform(100, 500)
            volume_multiplier = 1 + (abs(volatility_component) * 2)
            volume = base_volume * volume_multiplier

            data.append({
                'timestamp': timestamp,
                'price': round(current_price, 2),
                'volume': round(volume, 2),
                'regime': regime
            })

        return data

    def _calculate_test_metrics(self, trades: List[Dict[str, Any]], final_value: float, test_id: str) -> Dict[str, Any]:
        """Calculate performance metrics for a single test"""
        if not trades:
            return {
                'test_id': test_id,
                'total_trades': 0,
                'win_rate': 0.0,
                'total_pnl': 0.0,
                'avg_trade_pnl': 0.0,
                'sharpe_ratio': 0.0,
                'max_drawdown': 0.0,
                'final_value': final_value
            }

        # Calculate basic metrics
        winning_trades = [t for t in trades if t.get('pnl', 0) > 0]
        losing_trades = [t for t in trades if t.get('pnl', 0) < 0]

        total_pnl = sum(t.get('pnl', 0) for t in trades)
        win_rate = len(winning_trades) / len(trades) * 100 if trades else 0
        avg_trade_pnl = total_pnl / len(trades) if trades else 0

        # Calculate Sharpe ratio (simplified)
        pnl_values = [t.get('pnl', 0) for t in trades]
        if len(pnl_values) > 1:
            avg_return = statistics.mean(pnl_values)
            std_return = statistics.stdev(pnl_values)
            sharpe_ratio = (avg_return / std_return) * (252**0.5) if std_return > 0 else 0
        else:
            sharpe_ratio = 0

        # Calculate max drawdown (simplified)
        cumulative_pnl = 0
        peak = 0
        max_drawdown = 0

        for trade in trades:
            cumulative_pnl += trade.get('pnl', 0)
            peak = max(peak, cumulative_pnl)
            drawdown = peak - cumulative_pnl
            max_drawdown = max(max_drawdown, drawdown)

        return {
            'test_id': test_id,
            'total_trades': len(trades),
            'win_rate': win_rate,
            'total_pnl': total_pnl,
            'avg_trade_pnl': avg_trade_pnl,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'final_value': final_value
        }

    def _analyze_ab_results(self) -> Dict[str, Any]:
        """Analyze A/B test results with statistical significance"""
        print("\n📈 Analyzing A/B test results...")

        # Extract metrics for statistical testing
        baseline_win_rates = [r['win_rate'] for r in self.baseline_results]
        enhanced_win_rates = [r['win_rate'] for r in self.enhanced_results]

        baseline_sharpe = [r['sharpe_ratio'] for r in self.baseline_results]
        enhanced_sharpe = [r['sharpe_ratio'] for r in self.enhanced_results]

        baseline_pnl = [r['total_pnl'] for r in self.baseline_results]
        enhanced_pnl = [r['total_pnl'] for r in self.enhanced_results]

        # Calculate averages
        analysis = {
            'baseline_avg': {
                'win_rate': statistics.mean(baseline_win_rates),
                'sharpe_ratio': statistics.mean(baseline_sharpe),
                'total_pnl': statistics.mean(baseline_pnl),
                'total_trades': statistics.mean([r['total_trades'] for r in self.baseline_results])
            },
            'enhanced_avg': {
                'win_rate': statistics.mean(enhanced_win_rates),
                'sharpe_ratio': statistics.mean(enhanced_sharpe),
                'total_pnl': statistics.mean(enhanced_pnl),
                'total_trades': statistics.mean([r['total_trades'] for r in self.enhanced_results])
            },
            'statistical_tests': {}
        }

        # Perform basic statistical significance tests (simplified t-test approximation)
        try:
            # Win rate test
            wr_improvement = analysis['enhanced_avg']['win_rate'] - analysis['baseline_avg']['win_rate']
            wr_std = max(statistics.stdev(baseline_win_rates + enhanced_win_rates), 0.01)
            wr_t_stat = wr_improvement / (wr_std / (len(baseline_win_rates) ** 0.5))
            wr_p_value = 2 * (1 - self._normal_cdf(abs(wr_t_stat)))  # Approximate p-value

            analysis['statistical_tests']['win_rate'] = {
                't_statistic': wr_t_stat,
                'p_value': wr_p_value,
                'significant': wr_p_value < 0.05,
                'improvement': wr_improvement
            }

            # Sharpe ratio test
            sharpe_improvement = analysis['enhanced_avg']['sharpe_ratio'] - analysis['baseline_avg']['sharpe_ratio']
            sharpe_std = max(statistics.stdev(baseline_sharpe + enhanced_sharpe), 0.01)
            sharpe_t_stat = sharpe_improvement / (sharpe_std / (len(baseline_sharpe) ** 0.5))
            sharpe_p_value = 2 * (1 - self._normal_cdf(abs(sharpe_t_stat)))

            analysis['statistical_tests']['sharpe_ratio'] = {
                't_statistic': sharpe_t_stat,
                'p_value': sharpe_p_value,
                'significant': sharpe_p_value < 0.05,
                'improvement': sharpe_improvement
            }

            # PnL test
            pnl_improvement = analysis['enhanced_avg']['total_pnl'] - analysis['baseline_avg']['total_pnl']
            pnl_std = max(statistics.stdev(baseline_pnl + enhanced_pnl), 0.01)
            pnl_t_stat = pnl_improvement / (pnl_std / (len(baseline_pnl) ** 0.5))
            pnl_p_value = 2 * (1 - self._normal_cdf(abs(pnl_t_stat)))

            analysis['statistical_tests']['total_pnl'] = {
                't_statistic': pnl_t_stat,
                'p_value': pnl_p_value,
                'significant': pnl_p_value < 0.05,
                'improvement': pnl_improvement
            }

        except Exception as e:
            print(f"⚠️ Statistical test error: {e}")
            analysis['statistical_tests']['error'] = str(e)

        # Calculate confidence intervals
        analysis['confidence_intervals'] = {
            'baseline_win_rate': self._calculate_confidence_interval(baseline_win_rates),
            'enhanced_win_rate': self._calculate_confidence_interval(enhanced_win_rates),
            'baseline_sharpe': self._calculate_confidence_interval(baseline_sharpe),
            'enhanced_sharpe': self._calculate_confidence_interval(enhanced_sharpe)
        }

        return analysis

    def _calculate_confidence_interval(self, data: List[float], confidence: float = 0.95) -> Dict[str, float]:
        """Calculate confidence interval for a dataset (simplified using normal approximation)"""
        if len(data) < 2:
            return {'mean': statistics.mean(data) if data else 0, 'lower': 0, 'upper': 0}

        mean = statistics.mean(data)
        std_err = statistics.stdev(data) / (len(data) ** 0.5)

        # Use normal approximation for critical value (1.96 for 95% confidence)
        critical_value = 1.96
        margin = std_err * critical_value

        return {
            'mean': mean,
            'lower': mean - margin,
            'upper': mean + margin
        }

    def _normal_cdf(self, x: float) -> float:
        """Approximate normal cumulative distribution function using simplified approximation"""
        # Simplified approximation for p-value calculation
        # For x > 0, approximate using erf-like function
        if x < 0:
            return 1 - self._normal_cdf(-x)

        # Approximation: Φ(x) ≈ 1 - (1/√(2π)) * e^(-x²/2) / x for large x
        # For simplicity, use a basic approximation
        if x > 3:
            return 0.998  # Very close to 1
        elif x > 2:
            return 0.977
        elif x > 1:
            return 0.841
        else:
            return 0.5 + 0.4 * x  # Linear approximation for small x

    def _save_results(self, results: Dict[str, Any]) -> str:
        """Save A/B test results"""
        filename = f"ab_test_results_{self.symbol.lower().replace('-', '_')}_{self.timestamp}.json"
        filepath = self.output_dir / filename

        with open(filepath, 'w') as f:
            json.dump(results, f, indent=2, default=str)

        print(f"💾 A/B test results saved to: {filepath}")
        return str(filepath)

    def _generate_report(self, results: Dict[str, Any]) -> str:
        """Generate A/B testing report"""
        filename = f"ab_test_report_{self.symbol.lower().replace('-', '_')}_{self.timestamp}.txt"
        filepath = self.output_dir / filename

        analysis = results['analysis']
        stats_tests = analysis['statistical_tests']

        with open(filepath, 'w', encoding='utf-8') as f:
            f.write("🚀 Supreme System V5 - A/B Testing Validation Report\n")
            f.write("=" * 60)
            f.write(f"\nSymbol: {self.symbol}\n")
            f.write(f"Tests: {self.num_tests} per strategy\n")
            f.write(f"Duration: {self.test_duration_hours} hours per test\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

            f.write("📊 PERFORMANCE COMPARISON\n")
            f.write("-" * 40)
            f.write(f"\nBaseline Strategy:\n")
            f.write(".1f")
            f.write(".3f")
            f.write(".2f")
            f.write(f"\nEnhanced Strategy:\n")
            f.write(".1f")
            f.write(".3f")
            f.write(".2f")
            f.write("\n🔬 STATISTICAL SIGNIFICANCE (p<0.05)\n")
            f.write("-" * 40)

            for metric, test_results in stats_tests.items():
                if metric == 'error':
                    continue
                f.write(f"\n{metric.upper()} IMPROVEMENT:\n")
                f.write(".3f")
                f.write(".4f")
                f.write(f"   Significant: {'✅ YES' if test_results['significant'] else '❌ NO'}\n")

            f.write("\n🎯 CONCLUSION\n")
            f.write("-" * 40)

            # Overall assessment
            significant_improvements = sum(1 for test in stats_tests.values()
                                         if isinstance(test, dict) and test.get('significant', False))

            if significant_improvements >= 2:
                f.write("\n✅ ENHANCED STRATEGY VALIDATED\n")
                f.write("   Statistical significance achieved in multiple metrics\n")
                f.write("   Strategy improvements are not due to random chance\n")
            elif significant_improvements == 1:
                f.write("\n⚠️ MIXED RESULTS\n")
                f.write("   Some improvements detected, but not across all metrics\n")
                f.write("   Further testing recommended\n")
            else:
                f.write("\n❌ NO SIGNIFICANT IMPROVEMENT\n")
                f.write("   Enhanced strategy does not show statistical superiority\n")
                f.write("   Consider alternative approaches\n")

        print(f"📋 A/B test report saved to: {filepath}")
        return str(filepath)


def main():
    """Main A/B testing execution"""
    try:
        validator = ABTestValidator("ETH-USDT")
        results = validator.run_ab_test()

        analysis = results['analysis']
        stats_tests = analysis['statistical_tests']

        print("\n" + "=" * 60)
        print("🎯 A/B TESTING RESULTS SUMMARY")
        print("=" * 60)

        print("📊 Average Performance:")
        print(".1f")
        print(".3f")
        print(".2f")
        print(".1f")
        print(".3f")
        print(".2f")
        print("\n🔬 Statistical Significance (p<0.05):")
        significant_count = 0
        for metric, test_results in stats_tests.items():
            if metric == 'error':
                continue
            significant = test_results.get('significant', False)
            if significant:
                significant_count += 1
            improvement = test_results.get('improvement', 0)
            print(".3f")

        if significant_count >= 2:
            print("\n✅ SUCCESS: Enhanced strategy shows statistically significant improvements!")
        elif significant_count == 1:
            print("\n⚠️ MODERATE: Some improvements detected, further validation needed")
        else:
            print("\n❌ NEUTRAL: No significant improvements detected")
        print(f"\n📁 Results saved in: run_artifacts/")

    except Exception as e:
        print(f"❌ A/B testing error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
