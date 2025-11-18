#!/usr/bin/env python3
"""
analyze_coverage.py

Parse coverage.json and create detailed coverage analysis report.
Shows top files by impact, coverage gaps, and improvement recommendations.
"""

import json
import os
from pathlib import Path
from typing import Dict, List, Tuple

def load_coverage_data(coverage_file: str = "coverage.json") -> Dict:
    """Load coverage data from JSON file."""
    try:
        with open(coverage_file, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"Error: {coverage_file} not found. Run pytest with --cov-report=json first.")
        return {}

def calculate_file_impact(file_data: Dict) -> float:
    """Calculate impact score for a file (lines × uncovered percentage)."""
    lines = file_data['summary']['num_statements']
    covered_percent = file_data['summary']['percent_covered']

    # Impact = lines that need coverage
    uncovered_lines = lines * (100 - covered_percent) / 100
    return uncovered_lines

def analyze_coverage_gaps(coverage_data: Dict) -> List[Tuple[str, Dict]]:
    """Analyze coverage gaps and return sorted list by impact."""
    files_analysis = []

    for file_path, file_data in coverage_data.get('files', {}).items():
        if file_path.endswith('.py'):  # Only Python files
            impact = calculate_file_impact(file_data)
            summary = file_data['summary']

            analysis = {
                'file': file_path,
                'impact_score': impact,
                'total_lines': summary['num_statements'],
                'covered_lines': summary['covered_lines'],
                'coverage_percent': summary['percent_covered'],
                'missing_lines': summary['missing_lines'],
                'potential_gain': min(impact, summary['num_statements'])  # Max gain is 100%
            }

            files_analysis.append((file_path, analysis))

    # Sort by impact score (highest first)
    return sorted(files_analysis, key=lambda x: x[1]['impact_score'], reverse=True)

def generate_markdown_report(analysis: List[Tuple[str, Dict]], top_n: int = 15) -> str:
    """Generate markdown report of coverage analysis."""
    report = []

    # Header
    report.append("# Coverage Analysis Report")
    report.append("")
    report.append("## Top 15 Files by Coverage Impact")
    report.append("")
    report.append("| Rank | File | Impact Score | Total Lines | Coverage | Missing Lines | Potential Gain |")
    report.append("|------|------|-------------|------------|----------|---------------|----------------|")

    total_potential_gain = 0

    for i, (file_path, data) in enumerate(analysis[:top_n], 1):
        # Truncate long file paths
        short_path = file_path.replace('src/', '').replace('tests/', 'T/')
        if len(short_path) > 40:
            short_path = "..." + short_path[-37:]

        report.append(f"| {i:2d} | `{short_path}` | {data['impact_score']:8.1f} | {data['total_lines']:3d} | {data['coverage_percent']:5.1f}% | {data['missing_lines']:3d} | {data['potential_gain']:6.1f} |")

        total_potential_gain += data['potential_gain']

    report.append("")
    report.append(f"**Total Potential Coverage Gain:** {total_potential_gain:.1f} lines")
    report.append("")

    # Summary statistics
    total_lines = sum(data['total_lines'] for _, data in analysis)
    total_covered = sum(data['covered_lines'] for _, data in analysis)
    overall_coverage = (total_covered / total_lines * 100) if total_lines > 0 else 0

    report.append("## Summary Statistics")
    report.append("")
    report.append(f"- **Total Files Analyzed:** {len(analysis)}")
    report.append(f"- **Total Lines:** {total_lines:,}")
    report.append(f"- **Lines Covered:** {total_covered:,}")
    report.append(f"- **Overall Coverage:** {overall_coverage:.1f}%")
    report.append(f"- **Lines Needing Coverage:** {total_lines - total_covered:,}")
    report.append("")

    # Coverage distribution
    report.append("## Coverage Distribution")
    report.append("")

    ranges = [
        ("0-10%", 0, 10),
        ("10-25%", 10, 25),
        ("25-50%", 25, 50),
        ("50-75%", 50, 75),
        ("75-90%", 75, 90),
        ("90-100%", 90, 100)
    ]

    for label, min_cov, max_cov in ranges:
        count = sum(1 for _, data in analysis if min_cov <= data['coverage_percent'] < max_cov)
        if count > 0:
            report.append(f"- **{label}:** {count} files")

    # Recommendations
    report.append("")
    report.append("## Recommendations")
    report.append("")
    report.append("### Priority 1: High Impact Files")
    high_impact = [f for f, data in analysis[:5]]
    for file_path in high_impact:
        short_path = file_path.replace('src/', '').replace('tests/', 'T/')
        report.append(f"- Focus on `{short_path}` - high coverage gain potential")

    report.append("")
    report.append("### Priority 2: Test Categories")
    report.append("- **Unit Tests:** Focus on core business logic (strategies, risk management)")
    report.append("- **Integration Tests:** Critical data pipelines and trading workflows")
    report.append("- **Performance Tests:** Memory optimization and vectorized operations")

    report.append("")
    report.append("### Priority 3: Quick Wins")
    report.append("- Add basic test coverage for simple utility functions")
    report.append("- Test error handling paths in existing code")
    report.append("- Add integration tests for critical user journeys")

    return "\n".join(report)

def main():
    """Main analysis function."""
    print("🔍 Analyzing coverage data...")

    # Load coverage data
    coverage_data = load_coverage_data()

    if not coverage_data:
        print("❌ No coverage data found. Run pytest with --cov-report=json first.")
        return

    # Analyze coverage gaps
    analysis = analyze_coverage_gaps(coverage_data)

    print(f"📊 Found {len(analysis)} Python files with coverage data")

    # Generate report
    report = generate_markdown_report(analysis)

    # Save to file
    with open("COVERAGE_ANALYSIS.md", 'w', encoding='utf-8') as f:
        f.write(report)

    print("✅ Coverage analysis saved to COVERAGE_ANALYSIS.md")

    # Print top 5 to console
    print("\n🏆 Top 5 Files by Coverage Impact:")
    print("-" * 60)

    for i, (file_path, data) in enumerate(analysis[:5], 1):
        short_path = file_path.replace('src/', '').replace('tests/', 'T/')
        if len(short_path) > 35:
            short_path = "..." + short_path[-32:]

        print(f"{i}. {short_path:<35} | Impact: {data['impact_score']:6.1f} | Coverage: {data['coverage_percent']:5.1f}%")

if __name__ == "__main__":
    main()
