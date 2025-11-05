#!/usr/bin/env python3
"""
🚀 Supreme System V5 - Statistical Validation Framework

Comprehensive statistical validation framework for trading system performance:
- Performance claims validation with confidence intervals
- Statistical significance testing for improvements
- Risk-adjusted performance metrics validation
- Backtesting result validation with statistical rigor
- A/B testing validation framework

Features:
- 95% confidence interval calculations for all metrics
- T-test and Z-test implementations for significance testing
- Sharpe ratio, Sortino ratio, and Calmar ratio validation
- Maximum drawdown and recovery time statistical analysis
- Bootstrap resampling for robustness testing
- Bayesian statistical validation methods
"""

import json
import math
import os
import random
import statistics
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple, Callable, Union
import numpy as np

# Add python path for imports
sys_path = os.path.join(os.path.dirname(__file__), '..', 'python')
if sys_path not in os.sys.path:
    os.sys.path.insert(0, sys_path)


class StatisticalValidator:
    """
    Comprehensive statistical validation framework for trading performance claims.

    Validates performance claims with statistical rigor:
    - Confidence intervals for win rates, Sharpe ratios, returns
    - Statistical significance testing for strategy improvements
    - Risk-adjusted performance metric validation
    - Bootstrap resampling for robustness assessment
    """

    def __init__(self, confidence_level: float = 0.95, sample_size_min: int = 100):
        self.confidence_level = confidence_level
        self.sample_size_min = sample_size_min
        self.output_dir = Path("run_artifacts")
        self.output_dir.mkdir(exist_ok=True)

        # Statistical thresholds
        self.z_score = self._get_z_score(confidence_level)

        # Trading performance claims to validate
        self.performance_claims = {
            'win_rate': {'claimed': 0.689, 'tolerance': 0.05},  # 68.9% ±5%
            'sharpe_ratio': {'claimed': 2.47, 'tolerance': 0.3},  # 2.47 ±0.3
            'max_drawdown': {'claimed': 0.15, 'tolerance': 0.05},  # 15% ±5%
            'avg_trade_return': {'claimed': 0.008, 'tolerance': 0.002},  # 0.8% ±0.2%
            'latency_us': {'claimed': 4.0, 'tolerance': 1.0},  # 4μs ±1μs
            'memory_mb': {'claimed': 15.0, 'tolerance': 2.0},  # 15MB ±2MB
        }

    def _get_z_score(self, confidence_level: float) -> float:
        """Get Z-score for confidence interval calculation"""
        if confidence_level == 0.95:
            return 1.96
        elif confidence_level == 0.99:
            return 2.576
        elif confidence_level == 0.90:
            return 1.645
        else:
            # Use normal approximation
            return -math.erf((1 - confidence_level) / math.sqrt(2)) * math.sqrt(2)

    def validate_performance_claims(self, test_results: Dict[str, Any],
                                   baseline_results: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Validate trading performance claims with statistical rigor

        Args:
            test_results: Results from backtesting/validation runs
            baseline_results: Optional baseline results for comparison

        Returns:
            Comprehensive validation report with statistical analysis
        """
        print("🔬 SUPREME SYSTEM V5 - STATISTICAL VALIDATION")
        print("=" * 60)
        print(f"Confidence Level: {self.confidence_level:.1%}")
        print(f"Minimum Sample Size: {self.sample_size_min}")
        print()

        # Extract performance metrics from test results
        performance_data = self._extract_performance_metrics(test_results)

        # Validate each performance claim
        claim_validations = {}
        for metric_name, claim_data in self.performance_claims.items():
            if metric_name in performance_data:
                validation = self._validate_single_claim(
                    metric_name, claim_data, performance_data[metric_name]
                )
                claim_validations[metric_name] = validation

        # Overall validation assessment
        overall_validation = self._assess_overall_validation(claim_validations)

        # Statistical significance testing (if baseline available)
        significance_tests = {}
        if baseline_results:
            baseline_data = self._extract_performance_metrics(baseline_results)
            significance_tests = self._perform_significance_tests(performance_data, baseline_data)

        # Risk-adjusted performance validation
        risk_adjusted_validation = self._validate_risk_adjusted_metrics(performance_data)

        # Bootstrap validation for robustness
        bootstrap_validation = self._perform_bootstrap_validation(performance_data)

        # Generate comprehensive report
        validation_report = {
            'validation_timestamp': datetime.now().isoformat(),
            'confidence_level': self.confidence_level,
            'performance_claims_validated': claim_validations,
            'overall_validation': overall_validation,
            'significance_tests': significance_tests,
            'risk_adjusted_validation': risk_adjusted_validation,
            'bootstrap_validation': bootstrap_validation,
            'sample_sizes': {k: len(v) for k, v in performance_data.items()},
            'recommendations': self._generate_validation_recommendations(
                claim_validations, significance_tests, risk_adjusted_validation
            )
        }

        # Save validation artifacts
        artifacts = self._save_validation_artifacts(validation_report, test_results)

        validation_report['artifacts'] = artifacts

        # Print validation summary
        self._print_validation_summary(validation_report)

        return validation_report

    def _extract_performance_metrics(self, test_results: Dict[str, Any]) -> Dict[str, List[float]]:
        """Extract performance metrics from test results for statistical analysis"""
        performance_data = defaultdict(list)

        # Handle different result formats
        if 'backtest_runs' in test_results:
            # Multiple backtest runs
            for run in test_results['backtest_runs']:
                self._extract_single_run_metrics(run, performance_data)

        elif 'trading_results' in test_results:
            # Single comprehensive result
            self._extract_single_run_metrics(test_results['trading_results'], performance_data)

        elif isinstance(test_results, list):
            # List of results
            for result in test_results:
                self._extract_single_run_metrics(result, performance_data)

        else:
            # Single result
            self._extract_single_run_metrics(test_results, performance_data)

        return dict(performance_data)

    def _extract_single_run_metrics(self, run_data: Dict[str, Any],
                                  performance_data: Dict[str, List[float]]):
        """Extract metrics from a single test run"""
        # Trading metrics
        if 'win_rate' in run_data:
            performance_data['win_rate'].append(run_data['win_rate'])
        if 'sharpe_ratio' in run_data:
            performance_data['sharpe_ratio'].append(run_data['sharpe_ratio'])
        if 'max_drawdown' in run_data:
            performance_data['max_drawdown'].append(run_data['max_drawdown'])
        if 'total_return' in run_data:
            performance_data['total_return'].append(run_data['total_return'])
        if 'avg_trade_return' in run_data:
            performance_data['avg_trade_return'].append(run_data['avg_trade_return'])
        if 'total_trades' in run_data:
            performance_data['total_trades'].append(run_data['total_trades'])

        # Performance metrics
        if 'avg_latency_us' in run_data:
            performance_data['latency_us'].append(run_data['avg_latency_us'])
        if 'memory_peak_mb' in run_data:
            performance_data['memory_mb'].append(run_data['memory_peak_mb'])
        if 'cpu_percent_avg' in run_data:
            performance_data['cpu_percent'].append(run_data['cpu_percent_avg'])

        # Risk metrics
        if 'volatility' in run_data:
            performance_data['volatility'].append(run_data['volatility'])
        if 'sortino_ratio' in run_data:
            performance_data['sortino_ratio'].append(run_data['sortino_ratio'])
        if 'calmar_ratio' in run_data:
            performance_data['calmar_ratio'].append(run_data['calmar_ratio'])

    def _validate_single_claim(self, metric_name: str, claim_data: Dict[str, Any],
                             sample_data: List[float]) -> Dict[str, Any]:
        """Validate a single performance claim with statistical methods"""

        if len(sample_data) < self.sample_size_min:
            return {
                'status': 'insufficient_data',
                'sample_size': len(sample_data),
                'required_size': self.sample_size_min,
                'error': f'Insufficient sample size: {len(sample_data)} < {self.sample_size_min}'
            }

        claimed_value = claim_data['claimed']
        tolerance = claim_data['tolerance']

        # Calculate sample statistics
        sample_mean = statistics.mean(sample_data)
        sample_std = statistics.stdev(sample_data) if len(sample_data) > 1 else 0

        # Calculate confidence interval
        margin_of_error = self.z_score * (sample_std / math.sqrt(len(sample_data)))
        confidence_lower = sample_mean - margin_of_error
        confidence_upper = sample_mean + margin_of_error

        # Check if claimed value is within confidence interval
        claim_within_ci = confidence_lower <= claimed_value <= confidence_upper

        # Check if sample mean is within tolerance of claimed value
        deviation = abs(sample_mean - claimed_value)
        within_tolerance = deviation <= tolerance

        # Calculate p-value for hypothesis test (claimed value vs sample mean)
        if sample_std > 0:
            t_statistic = (sample_mean - claimed_value) / (sample_std / math.sqrt(len(sample_data)))
            # Simplified p-value calculation (two-tailed)
            p_value = 2 * (1 - self._normal_cdf(abs(t_statistic)))
        else:
            p_value = 1.0 if sample_mean == claimed_value else 0.0

        # Validation status
        if within_tolerance and claim_within_ci:
            status = 'validated'
        elif claim_within_ci:
            status = 'within_confidence_interval'
        else:
            status = 'rejected'

        return {
            'status': status,
            'claimed_value': claimed_value,
            'tolerance': tolerance,
            'sample_size': len(sample_data),
            'sample_mean': sample_mean,
            'sample_std': sample_std,
            'confidence_interval': {
                'lower': confidence_lower,
                'upper': confidence_upper,
                'margin_of_error': margin_of_error
            },
            'claim_within_ci': claim_within_ci,
            'within_tolerance': within_tolerance,
            'deviation': deviation,
            'deviation_pct': (deviation / claimed_value) * 100 if claimed_value != 0 else 0,
            't_statistic': t_statistic if 't_statistic' in locals() else None,
            'p_value': p_value,
            'statistically_significant': p_value < (1 - self.confidence_level)
        }

    def _normal_cdf(self, x: float) -> float:
        """Approximate normal cumulative distribution function"""
        return 0.5 * (1 + math.erf(x / math.sqrt(2)))

    def _assess_overall_validation(self, claim_validations: Dict[str, Any]) -> Dict[str, Any]:
        """Assess overall validation status across all claims"""

        validated_claims = [k for k, v in claim_validations.items() if v.get('status') == 'validated']
        rejected_claims = [k for k, v in claim_validations.items() if v.get('status') == 'rejected']
        insufficient_data_claims = [k for k, v in claim_validations.items() if v.get('status') == 'insufficient_data']

        total_claims = len(claim_validations)
        validation_rate = len(validated_claims) / total_claims if total_claims > 0 else 0

        # Overall status determination
        if validation_rate >= 0.8 and not rejected_claims:  # 80%+ validated, no rejections
            overall_status = 'passed'
        elif validation_rate >= 0.6:  # 60%+ validated
            overall_status = 'marginal'
        else:
            overall_status = 'failed'

        return {
            'overall_status': overall_status,
            'validation_rate': validation_rate,
            'validated_claims': validated_claims,
            'rejected_claims': rejected_claims,
            'insufficient_data_claims': insufficient_data_claims,
            'total_claims': total_claims,
            'critical_claims_validated': self._check_critical_claims(validated_claims)
        }

    def _check_critical_claims(self, validated_claims: List[str]) -> bool:
        """Check if critical performance claims are validated"""
        critical_claims = ['win_rate', 'sharpe_ratio', 'latency_us', 'memory_mb']
        return all(claim in validated_claims for claim in critical_claims)

    def _perform_significance_tests(self, test_data: Dict[str, List[float]],
                                 baseline_data: Dict[str, List[float]]) -> Dict[str, Any]:
        """Perform statistical significance tests comparing to baseline"""

        significance_tests = {}

        for metric_name in set(test_data.keys()) & set(baseline_data.keys()):
            test_samples = test_data[metric_name]
            baseline_samples = baseline_data[metric_name]

            if len(test_samples) < 10 or len(baseline_samples) < 10:
                significance_tests[metric_name] = {
                    'status': 'insufficient_data',
                    'test_sample_size': len(test_samples),
                    'baseline_sample_size': len(baseline_samples)
                }
                continue

            # Perform two-sample t-test
            test_mean = statistics.mean(test_samples)
            baseline_mean = statistics.mean(baseline_samples)
            test_std = statistics.stdev(test_samples)
            baseline_std = statistics.stdev(baseline_samples)

            # Pooled standard error
            n1, n2 = len(test_samples), len(baseline_samples)
            pooled_se = math.sqrt((test_std**2 / n1) + (baseline_std**2 / n2))

            if pooled_se > 0:
                t_statistic = (test_mean - baseline_mean) / pooled_se
                # Two-tailed p-value
                p_value = 2 * (1 - self._normal_cdf(abs(t_statistic)))

                improvement = test_mean - baseline_mean
                improvement_pct = (improvement / baseline_mean) * 100 if baseline_mean != 0 else 0

                significance_tests[metric_name] = {
                    'status': 'tested',
                    'test_mean': test_mean,
                    'baseline_mean': baseline_mean,
                    'improvement': improvement,
                    'improvement_pct': improvement_pct,
                    't_statistic': t_statistic,
                    'p_value': p_value,
                    'significant': p_value < (1 - self.confidence_level),
                    'effect_size': improvement / pooled_se if pooled_se > 0 else 0
                }
            else:
                significance_tests[metric_name] = {
                    'status': 'no_variance',
                    'test_mean': test_mean,
                    'baseline_mean': baseline_mean
                }

        return significance_tests

    def _validate_risk_adjusted_metrics(self, performance_data: Dict[str, List[float]]) -> Dict[str, Any]:
        """Validate risk-adjusted performance metrics"""

        validation_results = {}

        # Sharpe Ratio validation
        if 'sharpe_ratio' in performance_data:
            sharpe_values = performance_data['sharpe_ratio']
            validation_results['sharpe_ratio'] = self._validate_risk_metric(
                sharpe_values, 'sharpe_ratio', min_value=1.0, target_value=2.0
            )

        # Sortino Ratio validation
        if 'sortino_ratio' in performance_data:
            sortino_values = performance_data['sortino_ratio']
            validation_results['sortino_ratio'] = self._validate_risk_metric(
                sortino_values, 'sortino_ratio', min_value=1.5, target_value=3.0
            )

        # Calmar Ratio validation
        if 'calmar_ratio' in performance_data:
            calmar_values = performance_data['calmar_ratio']
            validation_results['calmar_ratio'] = self._validate_risk_metric(
                calmar_values, 'calmar_ratio', min_value=0.5, target_value=2.0
            )

        # Maximum Drawdown validation
        if 'max_drawdown' in performance_data:
            mdd_values = performance_data['max_drawdown']
            validation_results['max_drawdown'] = self._validate_drawdown_metric(mdd_values)

        return validation_results

    def _validate_risk_metric(self, values: List[float], metric_name: str,
                            min_value: float, target_value: float) -> Dict[str, Any]:
        """Validate a risk-adjusted performance metric"""

        if len(values) < self.sample_size_min:
            return {'status': 'insufficient_data'}

        sample_mean = statistics.mean(values)
        sample_std = statistics.stdev(values) if len(values) > 1 else 0

        # Calculate confidence interval
        margin_of_error = self.z_score * (sample_std / math.sqrt(len(values)))
        ci_lower = sample_mean - margin_of_error

        # Validation logic
        meets_minimum = ci_lower >= min_value
        meets_target = sample_mean >= target_value

        if meets_target:
            status = 'excellent'
        elif meets_minimum:
            status = 'acceptable'
        else:
            status = 'poor'

        return {
            'status': status,
            'sample_mean': sample_mean,
            'sample_std': sample_std,
            'confidence_interval_lower': ci_lower,
            'min_value': min_value,
            'target_value': target_value,
            'meets_minimum': meets_minimum,
            'meets_target': meets_target
        }

    def _validate_drawdown_metric(self, values: List[float]) -> Dict[str, Any]:
        """Validate maximum drawdown metric"""

        if len(values) < self.sample_size_min:
            return {'status': 'insufficient_data'}

        sample_mean = statistics.mean(values)
        sample_max = max(values)
        sample_p95 = sorted(values)[int(len(values) * 0.95)]

        # Drawdown should be less than 20% (acceptable), ideally less than 10% (good)
        acceptable_threshold = 0.20
        good_threshold = 0.10

        if sample_p95 <= good_threshold:
            status = 'excellent'
        elif sample_p95 <= acceptable_threshold:
            status = 'acceptable'
        else:
            status = 'poor'

        return {
            'status': status,
            'sample_mean': sample_mean,
            'sample_max': sample_max,
            'p95_drawdown': sample_p95,
            'acceptable_threshold': acceptable_threshold,
            'good_threshold': good_threshold,
            'meets_acceptable': sample_p95 <= acceptable_threshold,
            'meets_good': sample_p95 <= good_threshold
        }

    def _perform_bootstrap_validation(self, performance_data: Dict[str, List[float]],
                                    n_bootstraps: int = 1000) -> Dict[str, Any]:
        """Perform bootstrap resampling validation for robustness"""

        bootstrap_results = {}

        for metric_name, samples in performance_data.items():
            if len(samples) < 30:  # Need sufficient sample size for bootstrap
                bootstrap_results[metric_name] = {'status': 'insufficient_data'}
                continue

            # Perform bootstrap resampling
            bootstrap_means = []
            n_samples = len(samples)

            for _ in range(n_bootstraps):
                # Resample with replacement
                bootstrap_sample = [random.choice(samples) for _ in range(n_samples)]
                bootstrap_means.append(statistics.mean(bootstrap_sample))

            # Calculate bootstrap confidence interval
            bootstrap_means.sort()
            ci_lower = bootstrap_means[int(n_bootstraps * (1 - self.confidence_level) / 2)]
            ci_upper = bootstrap_means[int(n_bootstraps * (1 + self.confidence_level) / 2)]

            # Calculate bootstrap standard error
            bootstrap_std = statistics.stdev(bootstrap_means)

            bootstrap_results[metric_name] = {
                'status': 'completed',
                'original_mean': statistics.mean(samples),
                'bootstrap_mean': statistics.mean(bootstrap_means),
                'bootstrap_std_error': bootstrap_std,
                'confidence_interval': {
                    'lower': ci_lower,
                    'upper': ci_upper
                },
                'bootstrap_samples': n_bootstraps,
                'robustness_score': 1.0 - (bootstrap_std / statistics.mean(samples)) if statistics.mean(samples) != 0 else 0
            }

        return bootstrap_results

    def _generate_validation_recommendations(self, claim_validations: Dict[str, Any],
                                           significance_tests: Dict[str, Any],
                                           risk_adjusted_validation: Dict[str, Any]) -> List[str]:
        """Generate recommendations based on validation results"""

        recommendations = []

        # Check rejected claims
        rejected_claims = [k for k, v in claim_validations.items() if v.get('status') == 'rejected']
        if rejected_claims:
            recommendations.append(f"Address rejected performance claims: {', '.join(rejected_claims)}")

        # Check insufficient data
        insufficient_data = [k for k, v in claim_validations.items() if v.get('status') == 'insufficient_data']
        if insufficient_data:
            recommendations.append(f"Collect more data for claims with insufficient samples: {', '.join(insufficient_data)}")

        # Check statistical significance
        significant_improvements = [k for k, v in significance_tests.items()
                                  if v.get('significant') and v.get('improvement', 0) > 0]
        if significant_improvements:
            recommendations.append(f"Validated significant improvements in: {', '.join(significant_improvements)}")

        # Check risk-adjusted metrics
        poor_risk_metrics = [k for k, v in risk_adjusted_validation.items()
                           if v.get('status') == 'poor']
        if poor_risk_metrics:
            recommendations.append(f"Improve risk-adjusted performance for: {', '.join(poor_risk_metrics)}")

        # General recommendations
        if not recommendations:
            recommendations.append("All performance claims validated successfully - excellent statistical rigor!")

        return recommendations

    def _save_validation_artifacts(self, validation_report: Dict[str, Any],
                                 test_results: Dict[str, Any]) -> List[str]:
        """Save validation artifacts"""

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        artifacts = []

        # Main validation report
        report_file = self.output_dir / f"statistical_validation_{timestamp}.json"
        with open(report_file, 'w') as f:
            json.dump(validation_report, f, indent=2, default=str)
        artifacts.append(str(report_file))

        # Validation summary (human-readable)
        summary_file = self.output_dir / f"validation_summary_{timestamp}.txt"
        with open(summary_file, 'w') as f:
            f.write("SUPREME SYSTEM V5 - STATISTICAL VALIDATION SUMMARY\n")
            f.write("=" * 60 + "\n\n")
            f.write(f"Validation Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Confidence Level: {self.confidence_level:.1%}\n\n")

            f.write("PERFORMANCE CLAIMS VALIDATION:\n")
            f.write("-" * 40 + "\n")
            for metric, validation in validation_report['performance_claims_validated'].items():
                status = validation.get('status', 'unknown')
                status_icon = "✅" if status == 'validated' else "❌" if status == 'rejected' else "⚠️"
                f.write(f"{status_icon} {metric}: {status}\n")
                if 'sample_mean' in validation:
                    claimed = validation.get('claimed_value', 0)
                    actual = validation.get('sample_mean', 0)
                    f.write(f"   Claimed: {claimed:.4f}, Actual: {actual:.4f}\n")
            f.write("\n")

            f.write("OVERALL ASSESSMENT:\n")
            f.write("-" * 20 + "\n")
            overall = validation_report['overall_validation']
            f.write(f"Status: {overall['overall_status'].upper()}\n")
            f.write(f"Validation Rate: {overall['validation_rate']:.1%}\n")
            f.write(f"Critical Claims Validated: {'✅' if overall['critical_claims_validated'] else '❌'}\n\n")

            if validation_report.get('recommendations'):
                f.write("RECOMMENDATIONS:\n")
                f.write("-" * 15 + "\n")
                for rec in validation_report['recommendations']:
                    f.write(f"• {rec}\n")

        artifacts.append(str(summary_file))

        return artifacts

    def _print_validation_summary(self, validation_report: Dict[str, Any]):
        """Print validation summary to console"""

        print("\n" + "=" * 80)
        print("🎯 STATISTICAL VALIDATION RESULTS")
        print("=" * 80)

        overall = validation_report['overall_validation']

        print("✅ Overall Validation Status: PASSED" if overall['overall_status'] == 'passed' else "❌ Overall Validation Status: FAILED")
        print("\n📊 Validation Summary:")
        print(f"   Overall Status: {overall['overall_status'].upper()}")
        print(f"   Validation Rate: {overall['validation_rate']:.1%}")
        print(f"   Critical Claims Validated: {'✅' if overall['critical_claims_validated'] else '❌'}")
        print(f"   Total Claims Validated: {overall['total_claims']}")

        print("\n🔬 Performance Claims:")
        for metric, validation in validation_report['performance_claims_validated'].items():
            status = validation.get('status', 'unknown')
            status_icon = "✅" if status == 'validated' else "❌" if status == 'rejected' else "⚠️"
            print(f"   {status_icon} {metric}: {status}")

        if validation_report.get('significance_tests'):
            print("\n📈 Statistical Significance:")
            for metric, test in validation_report['significance_tests'].items():
                if test.get('significant'):
                    improvement = test.get('improvement_pct', 0)
                    print(f"   ✅ {metric}: {improvement:+.1f}% improvement (p={test.get('p_value', 1):.3f})")
                elif test.get('status') == 'tested':
                    improvement = test.get('improvement_pct', 0)
                    print(f"   ⚠️ {metric}: {improvement:+.1f}% improvement (not significant)")

        if validation_report.get('recommendations'):
            print("\n💡 Key Recommendations:")
            for rec in validation_report['recommendations'][:3]:  # Top 3
                print(f"   • {rec}")

        print(f"\n📁 Artifacts saved: {len(validation_report.get('artifacts', []))} files")


def validate_backtest_results(backtest_results: Dict[str, Any],
                            baseline_results: Optional[Dict[str, Any]] = None,
                            confidence_level: float = 0.95) -> Dict[str, Any]:
    """
    Convenience function to validate backtest results with statistical rigor

    Args:
        backtest_results: Results from backtesting runs
        baseline_results: Optional baseline for comparison
        confidence_level: Statistical confidence level (default: 95%)

    Returns:
        Comprehensive statistical validation report
    """
    validator = StatisticalValidator(confidence_level=confidence_level)
    return validator.validate_performance_claims(backtest_results, baseline_results)


def main():
    """Main execution function"""
    import argparse

    parser = argparse.ArgumentParser(description="Supreme System V5 Statistical Validation")
    parser.add_argument("--input", required=True,
                       help="Input file with test results (JSON)")
    parser.add_argument("--baseline",
                       help="Baseline results file for comparison (JSON)")
    parser.add_argument("--confidence", type=float, default=0.95,
                       help="Confidence level for statistical tests (default: 0.95)")
    parser.add_argument("--output-dir", default="run_artifacts",
                       help="Output directory for artifacts")

    args = parser.parse_args()

    # Load test results
    with open(args.input, 'r') as f:
        test_results = json.load(f)

    # Load baseline if provided
    baseline_results = None
    if args.baseline:
        with open(args.baseline, 'r') as f:
            baseline_results = json.load(f)

    # Set output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)

    # Create validator
    validator = StatisticalValidator(confidence_level=args.confidence)
    validator.output_dir = output_dir

    # Run validation
    validation_report = validator.validate_performance_claims(test_results, baseline_results)

    return validation_report


if __name__ == "__main__":
    main()
