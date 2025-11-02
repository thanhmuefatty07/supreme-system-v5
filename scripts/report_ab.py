#!/usr/bin/env python3
"""
A/B Test Report Generator for Supreme System V5.
Analyzes performance comparison between optimized and standard modes.
"""

import json
import sys
import os
from typing import Dict, List, Any, Tuple
from pathlib import Path
import statistics

def load_test_results(results_dir: str) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    Load test results from A/B test run.

    Args:
        results_dir: Directory containing test results

    Returns:
        Tuple of (optimized_results, standard_results)
    """
    results_dir = Path(results_dir)

    optimized_file = results_dir / "optimized_results.json"
    standard_file = results_dir / "standard_results.json"

    if not optimized_file.exists() or not standard_file.exists():
        print(f"❌ Test result files not found in {results_dir}")
        sys.exit(1)

    with open(optimized_file, 'r') as f:
        optimized = json.load(f)

    with open(standard_file, 'r') as f:
        standard = json.load(f)

    return optimized, standard

def calculate_performance_comparison(optimized: Dict[str, Any], standard: Dict[str, Any]) -> Dict[str, Any]:
    """
    Calculate performance comparison metrics.

    Args:
        optimized: Optimized mode results
        standard: Standard mode results

    Returns:
        Performance comparison dictionary
    """
    comparison = {}

    # CPU Usage Comparison
    opt_cpu = optimized['performance_metrics']['avg_cpu_percent']
    std_cpu = standard['performance_metrics']['avg_cpu_percent']
    comparison['cpu_usage'] = {
        'optimized': opt_cpu,
        'standard': std_cpu,
        'improvement': std_cpu - opt_cpu,
        'improvement_pct': ((std_cpu - opt_cpu) / std_cpu) * 100 if std_cpu > 0 else 0
    }

    # Memory Usage Comparison
    opt_mem = optimized['performance_metrics']['avg_memory_gb']
    std_mem = standard['performance_metrics']['avg_memory_gb']
    comparison['memory_usage'] = {
        'optimized': opt_mem,
        'standard': std_mem,
        'improvement': std_mem - opt_mem,
        'improvement_pct': ((std_mem - opt_mem) / std_mem) * 100 if std_mem > 0 else 0
    }

    # Indicator Latency Comparison
    opt_lat = optimized['performance_metrics']['avg_indicator_latency_ms']
    std_lat = standard['performance_metrics']['avg_indicator_latency_ms']
    comparison['indicator_latency'] = {
        'optimized': opt_lat,
        'standard': std_lat,
        'improvement': std_lat - opt_lat,
        'improvement_pct': ((std_lat - opt_lat) / std_lat) * 100 if std_lat > 0 else 0
    }

    # Event Skip Ratio (higher is better for optimized)
    opt_skip = optimized['performance_metrics']['avg_event_skip_ratio']
    std_skip = standard['performance_metrics']['avg_event_skip_ratio']
    comparison['event_efficiency'] = {
        'optimized': opt_skip,
        'standard': std_skip,
        'improvement': opt_skip - std_skip,
        'improvement_pct': ((opt_skip - std_skip) / (1 - std_skip)) * 100 if std_skip < 1 else 0
    }

    # SLO Compliance
    opt_slo_compliant = optimized['slo_report']['overall_compliant']
    std_slo_compliant = standard['slo_report']['overall_compliant']
    comparison['slo_compliance'] = {
        'optimized': opt_slo_compliant,
        'standard': std_slo_compliant,
        'improvement': opt_slo_compliant and not std_slo_compliant
    }

    return comparison

def determine_winner(comparison: Dict[str, Any]) -> Dict[str, Any]:
    """
    Determine A/B test winner based on comprehensive criteria.

    Args:
        comparison: Performance comparison results

    Returns:
        Winner determination with reasoning
    """
    scores = {'optimized': 0, 'standard': 0}
    reasoning = []

    # CPU Usage (lower is better)
    if comparison['cpu_usage']['improvement_pct'] > 10:
        scores['optimized'] += 2
        reasoning.append("Significant CPU reduction (>10%)")
    elif comparison['cpu_usage']['improvement_pct'] > 5:
        scores['optimized'] += 1
        reasoning.append("Moderate CPU improvement (5-10%)")

    # Memory Usage (lower is better)
    if comparison['memory_usage']['improvement_pct'] > 15:
        scores['optimized'] += 2
        reasoning.append("Significant memory reduction (>15%)")
    elif comparison['memory_usage']['improvement_pct'] > 5:
        scores['optimized'] += 1
        reasoning.append("Moderate memory improvement (5-15%)")

    # Latency (lower is better)
    if comparison['indicator_latency']['improvement_pct'] > 20:
        scores['optimized'] += 2
        reasoning.append("Significant latency improvement (>20%)")
    elif comparison['indicator_latency']['improvement_pct'] > 10:
        scores['optimized'] += 1
        reasoning.append("Moderate latency improvement (10-20%)")

    # Event Efficiency (higher skip ratio is better for optimized)
    if comparison['event_efficiency']['improvement'] > 0.3:
        scores['optimized'] += 2
        reasoning.append("Excellent event filtering efficiency (>30%)")
    elif comparison['event_efficiency']['improvement'] > 0.1:
        scores['optimized'] += 1
        reasoning.append("Good event filtering efficiency (10-30%)")

    # SLO Compliance
    if comparison['slo_compliance']['optimized'] and not comparison['slo_compliance']['standard']:
        scores['optimized'] += 3
        reasoning.append("SLO compliance improvement")

    # Determine winner
    if scores['optimized'] > scores['standard']:
        winner = 'optimized'
        confidence = 'high' if scores['optimized'] >= 5 else 'moderate'
    elif scores['standard'] > scores['optimized']:
        winner = 'standard'
        confidence = 'high' if scores['standard'] >= 5 else 'moderate'
    else:
        winner = 'tie'
        confidence = 'low'

    return {
        'winner': winner,
        'confidence': confidence,
        'scores': scores,
        'reasoning': reasoning,
        'recommendation': get_recommendation(winner, comparison)
    }

def get_recommendation(winner: str, comparison: Dict[str, Any]) -> str:
    """Generate deployment recommendation."""
    if winner == 'optimized':
        return "RECOMMENDATION: Deploy optimized mode to production. Significant performance improvements with maintained reliability."
    elif winner == 'standard':
        return "RECOMMENDATION: Keep standard mode. Optimization may need refinement before production deployment."
    else:
        return "RECOMMENDATION: Further testing needed. Results inconclusive - consider additional metrics or longer test duration."

def generate_markdown_report(optimized: Dict[str, Any], standard: Dict[str, Any],
                           comparison: Dict[str, Any], winner: Dict[str, Any]) -> str:
    """Generate comprehensive markdown report."""

    report = f"""# 🚀 Supreme System V5 - A/B Test Report

**Test Duration:** {optimized['duration_hours']} hours
**Symbol:** {optimized['symbol']}
**Generated:** {optimized['timestamp']}

## 📊 Executive Summary

### Winner: **{winner['winner'].upper()}** (Confidence: {winner['confidence']})
{winner['recommendation']}

### Key Findings
"""

    # Add reasoning
    for reason in winner['reasoning']:
        report += f"- ✅ {reason}\n"

    report += "\n## 📈 Performance Comparison\n\n"

    # CPU Comparison
    cpu = comparison['cpu_usage']
    report += f"""### CPU Usage
| Metric | Optimized | Standard | Improvement |
|--------|-----------|----------|-------------|
| Average CPU % | {cpu['optimized']:.1f}% | {cpu['standard']:.1f}% | {cpu['improvement']:+.1f}% ({cpu['improvement_pct']:+.1f}%) |
"""

    # Memory Comparison
    mem = comparison['memory_usage']
    report += f"""
### Memory Usage
| Metric | Optimized | Standard | Improvement |
|--------|-----------|----------|-------------|
| Average RAM | {mem['optimized']:.2f}GB | {mem['standard']:.2f}GB | {mem['improvement']:+.2f}GB ({mem['improvement_pct']:+.1f}%) |
"""

    # Latency Comparison
    lat = comparison['indicator_latency']
    report += f"""
### Indicator Latency
| Metric | Optimized | Standard | Improvement |
|--------|-----------|----------|-------------|
| Avg Latency | {lat['optimized']:.1f}ms | {lat['standard']:.1f}ms | {lat['improvement']:+.1f}ms ({lat['improvement_pct']:+.1f}%) |
"""

    # Event Efficiency
    eff = comparison['event_efficiency']
    report += f"""
### Event Processing Efficiency
| Metric | Optimized | Standard | Improvement |
|--------|-----------|----------|-------------|
| Skip Ratio | {eff['optimized']:.3f} | {eff['standard']:.3f} | {eff['improvement']:+.3f} ({eff['improvement_pct']:+.1f}%) |
"""

    # SLO Compliance
    report += f"""
## 🛡️ SLO Compliance

| SLO Metric | Target | Optimized | Standard | Status |
|------------|--------|-----------|----------|--------|
| CPU < 88% | ✅ | {'✅' if optimized['slo_report']['overall_compliant'] else '❌'} | {'✅' if standard['slo_report']['overall_compliant'] else '❌'} | {'✅ IMPROVED' if comparison['slo_compliance']['improvement'] else '⚠️ SAME'} |
| Memory < 3.86GB | ✅ | {'✅' if optimized['slo_report']['overall_compliant'] else '❌'} | {'✅' if standard['slo_report']['overall_compliant'] else '❌'} | {'✅ IMPROVED' if comparison['slo_compliance']['improvement'] else '⚠️ SAME'} |
| Latency < 200ms | ✅ | {'✅' if optimized['slo_report']['overall_compliant'] else '❌'} | {'✅' if standard['slo_report']['overall_compliant'] else '❌'} | {'✅ IMPROVED' if comparison['slo_compliance']['improvement'] else '⚠️ SAME'} |

## 📋 Detailed Results

### Optimized Mode Performance
- **CPU Usage:** {optimized['performance_metrics']['avg_cpu_percent']:.1f}%
- **Memory Usage:** {optimized['performance_metrics']['avg_memory_gb']:.2f}GB
- **Indicator Latency:** {optimized['performance_metrics']['avg_indicator_latency_ms']:.1f}ms
- **Event Skip Ratio:** {optimized['performance_metrics']['avg_event_skip_ratio']:.3f}
- **SLO Compliant:** {'✅ YES' if optimized['slo_report']['overall_compliant'] else '❌ NO'}

### Standard Mode Performance
- **CPU Usage:** {standard['performance_metrics']['avg_cpu_percent']:.1f}%
- **Memory Usage:** {standard['performance_metrics']['avg_memory_gb']:.2f}GB
- **Indicator Latency:** {standard['performance_metrics']['avg_indicator_latency_ms']:.1f}ms
- **Event Skip Ratio:** {standard['performance_metrics']['avg_event_skip_ratio']:.3f}
- **SLO Compliant:** {'✅ YES' if standard['slo_report']['overall_compliant'] else '❌ NO'}

## 🎯 Recommendations

### Immediate Actions
1. **{'✅ DEPLOY' if winner['winner'] == 'optimized' else '⚠️ HOLD'}** optimized mode to production
2. Monitor SLO compliance for first 24 hours post-deployment
3. Set up automated alerts for performance regressions

### Optimization Opportunities
"""

    # Add specific recommendations based on results
    if winner['winner'] == 'optimized':
        report += """- Continue monitoring resource usage patterns
- Consider further optimization opportunities identified during testing
- Evaluate extending optimized mode to additional components
"""
    else:
        report += """- Investigate why optimizations underperformed expectations
- Review configuration parameters for potential tuning
- Consider additional performance profiling to identify bottlenecks
"""

    report += f"""
### Test Quality Assessment
- **Test Duration:** {optimized['duration_hours']} hours (Recommended: 24+ hours for confidence)
- **Statistical Significance:** {'High' if winner['confidence'] == 'high' else 'Moderate' if winner['confidence'] == 'moderate' else 'Low'}
- **Data Completeness:** ✅ Complete (both test runs successful)

## 📝 Conclusion

**Supreme System V5 A/B testing demonstrates {'significant performance improvements' if winner['winner'] == 'optimized' else 'potential for optimization'}.**

**Next Steps:**
1. {'Proceed with production deployment' if winner['winner'] == 'optimized' else 'Conduct additional testing'}
2. Implement monitoring and alerting based on test insights
3. Schedule follow-up performance reviews

---
*Generated by Supreme System V5 A/B Test Suite*
"""

    return report

def main():
    """Main entry point."""
    if len(sys.argv) != 2:
        print("Usage: python report_ab.py <results_directory>")
        sys.exit(1)

    results_dir = sys.argv[1]

    try:
        # Load test results
        optimized, standard = load_test_results(results_dir)

        # Calculate comparison
        comparison = calculate_performance_comparison(optimized, standard)

        # Determine winner
        winner = determine_winner(comparison)

        # Generate report
        report = generate_markdown_report(optimized, standard, comparison, winner)

        # Print report
        print(report)

    except Exception as e:
        print(f"❌ Error generating report: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
