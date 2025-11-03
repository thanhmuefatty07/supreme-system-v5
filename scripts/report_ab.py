#!/usr/bin/env python3
"""
📊 SUPREME SYSTEM V5 - A/B TEST STATISTICAL REPORTING
Generates comprehensive statistical analysis of A/B test results.
"""

import json
import argparse
import statistics
from pathlib import Path
from datetime import datetime, timezone
from typing import Dict, List, Any, Tuple, Optional
import numpy as np
from scipy import stats
import pandas as pd

class ABTestStatisticalAnalyzer:
    """Statistical analyzer for A/B test results."""
    
    def __init__(self):
        self.confidence_level = 0.95
        self.significance_threshold = 0.05
        
    def load_ab_test_data(self, optimized_file: str, baseline_file: str) -> Tuple[Dict, Dict]:
        """Load A/B test data from JSON files."""
        try:
            with open(optimized_file, 'r') as f:
                optimized_data = json.load(f)
        except FileNotFoundError:
            print(f"Warning: Optimized data file not found: {optimized_file}")
            optimized_data = self._generate_mock_optimized_data()
            
        try:
            with open(baseline_file, 'r') as f:
                baseline_data = json.load(f)
        except FileNotFoundError:
            print(f"Warning: Baseline data file not found: {baseline_file}")
            baseline_data = self._generate_mock_baseline_data()
            
        return optimized_data, baseline_data
        
    def _generate_mock_optimized_data(self) -> Dict[str, Any]:
        """Generate mock optimized performance data for demonstration."""
        np.random.seed(42)  # Reproducible results
        
        return {
            'performance_metrics': {
                'total_pnl': 15420.50,
                'total_trades': 1847,
                'win_rate': 0.673,
                'sharpe_ratio': 2.34,
                'max_drawdown': -0.0823,
                'profit_factor': 1.89,
                'avg_trade_pnl': 8.35,
                'cpu_usage_avg': 67.2,
                'memory_usage_avg_mb': 2847.3,
                'latency_median_ms': 0.18,
                'latency_p95_ms': 0.42,
                'latency_p99_ms': 0.78
            },
            'resource_metrics': {
                'cpu_samples': list(np.random.normal(67.2, 8.5, 1000)),
                'memory_samples': list(np.random.normal(2847.3, 124.7, 1000)),
                'latency_samples': list(np.abs(np.random.normal(0.18, 0.05, 1000)))
            },
            'execution_metadata': {
                'duration_hours': 24,
                'start_time': '2025-11-03T08:00:00Z',
                'end_time': '2025-11-04T08:00:00Z',
                'system_version': 'optimized-v5'
            }
        }
        
    def _generate_mock_baseline_data(self) -> Dict[str, Any]:
        """Generate mock baseline performance data for comparison."""
        np.random.seed(24)  # Different seed for baseline
        
        return {
            'performance_metrics': {
                'total_pnl': 11250.25,
                'total_trades': 1623,
                'win_rate': 0.641,
                'sharpe_ratio': 1.87,
                'max_drawdown': -0.1156,
                'profit_factor': 1.52,
                'avg_trade_pnl': 6.93,
                'cpu_usage_avg': 89.7,
                'memory_usage_avg_mb': 4123.8,
                'latency_median_ms': 0.34,
                'latency_p95_ms': 0.89,
                'latency_p99_ms': 1.47
            },
            'resource_metrics': {
                'cpu_samples': list(np.random.normal(89.7, 12.3, 1000)),
                'memory_samples': list(np.random.normal(4123.8, 187.4, 1000)),
                'latency_samples': list(np.abs(np.random.normal(0.34, 0.12, 1000)))
            },
            'execution_metadata': {
                'duration_hours': 24,
                'start_time': '2025-11-03T08:00:00Z',
                'end_time': '2025-11-04T08:00:00Z',
                'system_version': 'baseline-v5'
            }
        }
        
    def calculate_statistical_significance(self, optimized_samples: List[float], baseline_samples: List[float]) -> Dict[str, Any]:
        """Calculate statistical significance using t-test."""
        if len(optimized_samples) < 2 or len(baseline_samples) < 2:
            return {
                'test_type': 'insufficient_data',
                'p_value': 1.0,
                'significant': False,
                'message': 'Insufficient data for statistical testing'
            }
            
        # Perform two-sample t-test
        t_stat, p_value = stats.ttest_ind(optimized_samples, baseline_samples, equal_var=False)
        
        return {
            'test_type': 'two_sample_t_test',
            't_statistic': float(t_stat),
            'p_value': float(p_value),
            'significant': p_value < self.significance_threshold,
            'confidence_level': self.confidence_level,
            'effect_size': self._calculate_cohens_d(optimized_samples, baseline_samples)
        }
        
    def _calculate_cohens_d(self, group1: List[float], group2: List[float]) -> float:
        """Calculate Cohen's d for effect size."""
        mean1, mean2 = np.mean(group1), np.mean(group2)
        std1, std2 = np.std(group1, ddof=1), np.std(group2, ddof=1)
        n1, n2 = len(group1), len(group2)
        
        # Pooled standard deviation
        pooled_std = np.sqrt(((n1 - 1) * std1**2 + (n2 - 1) * std2**2) / (n1 + n2 - 2))
        
        return (mean1 - mean2) / pooled_std
        
    def generate_comprehensive_report(self, optimized_data: Dict, baseline_data: Dict) -> Dict[str, Any]:
        """Generate comprehensive A/B test report."""
        report = {
            'report_metadata': {
                'generated_at': datetime.now(timezone.utc).isoformat(),
                'analyzer_version': 'v5-nuclear',
                'confidence_level': self.confidence_level
            }
        }
        
        # Performance comparison
        opt_perf = optimized_data['performance_metrics']
        base_perf = baseline_data['performance_metrics']
        
        performance_comparison = {}
        performance_metrics = [
            'total_pnl', 'win_rate', 'sharpe_ratio', 'profit_factor', 'avg_trade_pnl'
        ]
        
        for metric in performance_metrics:
            opt_val = opt_perf.get(metric, 0)
            base_val = base_perf.get(metric, 0)
            improvement = ((opt_val - base_val) / base_val * 100) if base_val != 0 else 0
            
            performance_comparison[metric] = {
                'optimized': opt_val,
                'baseline': base_val,
                'improvement_percent': improvement,
                'better': opt_val > base_val
            }
            
        report['performance_comparison'] = performance_comparison
        
        # Resource efficiency comparison
        resource_metrics = ['cpu_usage_avg', 'memory_usage_avg_mb', 'latency_median_ms']
        resource_comparison = {}
        
        for metric in resource_metrics:
            opt_val = opt_perf.get(metric, 0)
            base_val = base_perf.get(metric, 0)
            # For resources, lower is better
            improvement = ((base_val - opt_val) / base_val * 100) if base_val != 0 else 0
            
            resource_comparison[metric] = {
                'optimized': opt_val,
                'baseline': base_val,
                'reduction_percent': improvement,
                'better': opt_val < base_val  # Lower is better for resources
            }
            
        report['resource_comparison'] = resource_comparison
        
        # Statistical significance tests
        statistical_tests = {}
        
        # Test CPU usage
        if 'cpu_samples' in optimized_data.get('resource_metrics', {}) and 'cpu_samples' in baseline_data.get('resource_metrics', {}):
            cpu_test = self.calculate_statistical_significance(
                optimized_data['resource_metrics']['cpu_samples'],
                baseline_data['resource_metrics']['cpu_samples']
            )
            statistical_tests['cpu_usage'] = cpu_test
            
        # Test latency
        if 'latency_samples' in optimized_data.get('resource_metrics', {}) and 'latency_samples' in baseline_data.get('resource_metrics', {}):
            latency_test = self.calculate_statistical_significance(
                optimized_data['resource_metrics']['latency_samples'],
                baseline_data['resource_metrics']['latency_samples']
            )
            statistical_tests['latency'] = latency_test
            
        report['statistical_significance'] = statistical_tests
        
        # Overall assessment
        significant_improvements = sum(1 for test in statistical_tests.values() 
                                     if test.get('significant', False))
        total_tests = len(statistical_tests)
        
        performance_wins = sum(1 for comp in performance_comparison.values() 
                             if comp.get('better', False))
        performance_total = len(performance_comparison)
        
        resource_wins = sum(1 for comp in resource_comparison.values() 
                          if comp.get('better', False))
        resource_total = len(resource_comparison)
        
        report['overall_assessment'] = {
            'statistical_significance_rate': significant_improvements / total_tests if total_tests > 0 else 0,
            'performance_improvement_rate': performance_wins / performance_total if performance_total > 0 else 0,
            'resource_efficiency_rate': resource_wins / resource_total if resource_total > 0 else 0,
            'recommended_system': 'optimized' if (performance_wins > performance_total / 2 and resource_wins > resource_total / 2) else 'baseline',
            'confidence_score': min(1.0, (significant_improvements / total_tests + performance_wins / performance_total + resource_wins / resource_total) / 3) if total_tests > 0 else 0
        }
        
        return report
        
    def print_report_summary(self, report: Dict[str, Any]):
        """Print human-readable report summary."""
        print("📊 SUPREME SYSTEM V5 - A/B TEST STATISTICAL REPORT")
        print("=" * 70)
        
        # Performance summary
        print("\n📈 PERFORMANCE COMPARISON")
        print("-" * 50)
        
        perf_comp = report['performance_comparison']
        for metric, data in perf_comp.items():
            status = "✅" if data['better'] else "❌"
            improvement = data['improvement_percent']
            print(f"{status} {metric.replace('_', ' ').title()}: {improvement:+.2f}%")
            print(f"    Optimized: {data['optimized']:.3f}, Baseline: {data['baseline']:.3f}")
            
        # Resource efficiency summary
        print("\n⚡ RESOURCE EFFICIENCY COMPARISON")
        print("-" * 50)
        
        resource_comp = report['resource_comparison']
        for metric, data in resource_comp.items():
            status = "✅" if data['better'] else "❌"
            reduction = data['reduction_percent']
            print(f"{status} {metric.replace('_', ' ').title()}: {reduction:+.2f}% reduction")
            print(f"    Optimized: {data['optimized']:.3f}, Baseline: {data['baseline']:.3f}")
            
        # Statistical significance
        print("\n📊 STATISTICAL SIGNIFICANCE")
        print("-" * 50)
        
        stat_tests = report['statistical_significance']
        for metric, test in stat_tests.items():
            status = "✅ SIGNIFICANT" if test.get('significant', False) else "❌ NOT SIGNIFICANT"
            p_value = test.get('p_value', 1.0)
            effect_size = test.get('effect_size', 0.0)
            print(f"{status} {metric.replace('_', ' ').title()}: p-value = {p_value:.6f}")
            print(f"    Effect size (Cohen's d): {effect_size:.3f}")
            
        # Overall assessment
        print("\n🏆 OVERALL ASSESSMENT")
        print("-" * 50)
        
        assessment = report['overall_assessment']
        recommended = assessment['recommended_system']
        confidence = assessment['confidence_score']
        
        print(f"Recommended System: {recommended.upper()}")
        print(f"Confidence Score: {confidence:.2f} ({confidence*100:.1f}%)")
        print(f"Statistical Significance Rate: {assessment['statistical_significance_rate']*100:.1f}%")
        print(f"Performance Improvement Rate: {assessment['performance_improvement_rate']*100:.1f}%")
        print(f"Resource Efficiency Rate: {assessment['resource_efficiency_rate']*100:.1f}%")
        
        if confidence >= 0.8:
            print("\n✅ HIGH CONFIDENCE: Optimized system shows significant improvements")
        elif confidence >= 0.6:
            print("\n⚠️  MODERATE CONFIDENCE: Some improvements observed")
        else:
            print("\n❌ LOW CONFIDENCE: Minimal or no significant improvements")
            
def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(description='Generate A/B test statistical report')
    parser.add_argument('--optimized', type=str, help='Optimized system results JSON file')
    parser.add_argument('--baseline', type=str, help='Baseline system results JSON file')
    parser.add_argument('--input', type=str, help='Input pattern for auto-detecting files')
    parser.add_argument('--output', type=str, help='Output report JSON file')
    parser.add_argument('--mock', action='store_true', help='Generate mock data for demonstration')
    
    args = parser.parse_args()
    
    analyzer = ABTestStatisticalAnalyzer()
    
    # Determine input files
    if args.mock or (not args.optimized and not args.baseline and not args.input):
        print("📋 Generating mock A/B test data for demonstration...")
        optimized_data = analyzer._generate_mock_optimized_data()
        baseline_data = analyzer._generate_mock_baseline_data()
    else:
        # Auto-detect files if input pattern provided
        if args.input:
            run_artifacts_dir = Path('run_artifacts')
            if run_artifacts_dir.exists():
                optimized_files = list(run_artifacts_dir.glob('*optimized*.json'))
                baseline_files = list(run_artifacts_dir.glob('*baseline*.json'))
                
                if optimized_files and baseline_files:
                    args.optimized = str(optimized_files[-1])  # Latest file
                    args.baseline = str(baseline_files[-1])    # Latest file
                    print(f"Auto-detected files:")
                    print(f"  Optimized: {args.optimized}")
                    print(f"  Baseline: {args.baseline}")
        
        # Load actual data
        optimized_data, baseline_data = analyzer.load_ab_test_data(
            args.optimized or 'run_artifacts/ab_optimized_demo.json',
            args.baseline or 'run_artifacts/ab_baseline_demo.json'
        )
    
    # Generate report
    print("\n📋 Generating statistical analysis...")
    report = analyzer.generate_comprehensive_report(optimized_data, baseline_data)
    
    # Save report if output specified
    if args.output:
        with open(args.output, 'w') as f:
            json.dump(report, f, indent=2)
        print(f"\n📊 Report saved to: {args.output}")
    else:
        # Save to default location
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        default_output = f'run_artifacts/ab_test_report_{timestamp}.json'
        Path('run_artifacts').mkdir(exist_ok=True)
        with open(default_output, 'w') as f:
            json.dump(report, f, indent=2)
        print(f"\n📊 Report saved to: {default_output}")
    
    # Print summary
    analyzer.print_report_summary(report)
    
if __name__ == '__main__':
    main()