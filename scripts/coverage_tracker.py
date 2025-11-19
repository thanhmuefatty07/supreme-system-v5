#!/usr/bin/env python3
"""
Coverage Tracking & Monitoring Script

Tracks coverage trends and provides optimization recommendations
"""

import json
import os
from pathlib import Path
from typing import Dict, List, Tuple
from datetime import datetime, timedelta
import logging
import subprocess
import sys

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class CoverageTracker:
    """Track and analyze coverage trends"""

    def __init__(self):
        self.baseline_dir = Path("baselines")
        self.baseline_dir.mkdir(exist_ok=True)

    def run_coverage_analysis(self) -> Dict:
        """Run coverage analysis and return results"""
        try:
            # Run pytest with coverage
            result = subprocess.run([
                sys.executable, "-m", "pytest",
                "tests/", "--cov=src", "--cov-report=json",
                "--tb=no", "-q"
            ], capture_output=True, text=True, timeout=300)

            # Load coverage data
            if Path("coverage.json").exists():
                with open("coverage.json", 'r') as f:
                    coverage_data = json.load(f)
                return self._analyze_coverage(coverage_data)
            else:
                logger.error("Coverage JSON not generated")
                return {}

        except Exception as e:
            logger.error(f"Coverage analysis failed: {e}")
            return {}

    def _analyze_coverage(self, coverage_data: Dict) -> Dict:
        """Analyze coverage data"""
        analysis = {
            'timestamp': datetime.now().isoformat(),
            'total_files': 0,
            'covered_files': 0,
            'total_lines': 0,
            'covered_lines': 0,
            'coverage_percent': 0.0,
            'file_coverage': {},
            'uncovered_lines': {},
            'recommendations': []
        }

        for file_path, file_data in coverage_data.items():
            if not file_path.startswith('src/'):
                continue

            analysis['total_files'] += 1

            summary = file_data.get('summary', {})
            lines = summary.get('covered_lines', 0) + summary.get('missing_lines', 0)
            covered = summary.get('covered_lines', 0)

            if lines > 0:
                file_coverage = (covered / lines) * 100
                analysis['file_coverage'][file_path] = file_coverage

                if file_coverage > 0:
                    analysis['covered_files'] += 1

                analysis['total_lines'] += lines
                analysis['covered_lines'] += covered

                # Track uncovered lines
                missing_lines = file_data.get('missing_lines', [])
                if missing_lines:
                    analysis['uncovered_lines'][file_path] = missing_lines

        # Calculate overall coverage
        if analysis['total_lines'] > 0:
            analysis['coverage_percent'] = (analysis['covered_lines'] / analysis['total_lines']) * 100

        # Generate recommendations
        analysis['recommendations'] = self._generate_recommendations(analysis)

        return analysis

    def _generate_recommendations(self, analysis: Dict) -> List[str]:
        """Generate coverage improvement recommendations"""
        recommendations = []

        # Sort files by coverage impact (lines * uncovered percentage)
        file_impacts = []
        for file_path, coverage in analysis['file_coverage'].items():
            lines = len(analysis.get('uncovered_lines', {}).get(file_path, []))
            impact = lines * (100 - coverage) / 100
            file_impacts.append((file_path, impact, coverage))

        file_impacts.sort(key=lambda x: x[1], reverse=True)

        # Top recommendations
        for file_path, impact, coverage in file_impacts[:5]:
            if coverage < 80:
                recommendations.append(f"Focus on {file_path} (coverage: {coverage:.1f}%, impact: {impact:.0f} lines)")

        # General recommendations
        if analysis['coverage_percent'] < 50:
            recommendations.append("Overall coverage is low - prioritize unit tests for core business logic")

        if len(analysis['uncovered_lines']) > 20:
            recommendations.append("Many files have zero coverage - create basic integration tests")

        return recommendations

    def save_baseline(self, analysis: Dict):
        """Save current analysis as baseline"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"baseline_{timestamp}.json"

        with open(self.baseline_dir / filename, 'w') as f:
            json.dump(analysis, f, indent=2)

        logger.info(f"Saved baseline: {filename}")
        return filename

    def load_latest_baseline(self) -> Dict:
        """Load the most recent baseline"""
        baselines = list(self.baseline_dir.glob("baseline_*.json"))
        if not baselines:
            return {}

        latest = max(baselines, key=lambda x: x.stat().st_mtime)

        with open(latest, 'r') as f:
            return json.load(f)

    def compare_with_baseline(self, current: Dict, baseline: Dict) -> Dict:
        """Compare current analysis with baseline"""
        if not baseline:
            return {'status': 'no_baseline', 'message': 'No baseline available for comparison'}

        comparison = {
            'status': 'compared',
            'current_coverage': current.get('coverage_percent', 0),
            'baseline_coverage': baseline.get('coverage_percent', 0),
            'coverage_change': current.get('coverage_percent', 0) - baseline.get('coverage_percent', 0),
            'new_files': len(current.get('file_coverage', {})) - len(baseline.get('file_coverage', {})),
            'timestamp': current.get('timestamp')
        }

        return comparison

    def generate_report(self, analysis: Dict, comparison: Dict = None) -> str:
        """Generate comprehensive coverage report"""
        report = f"""# Coverage Analysis Report

**Generated:** {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

## Current Status

- **Overall Coverage:** {analysis.get('coverage_percent', 0):.1f}%
- **Files Analyzed:** {analysis.get('total_files', 0)}
- **Files with Coverage:** {analysis.get('covered_files', 0)}
- **Total Lines:** {analysis.get('total_lines', 0)}
- **Covered Lines:** {analysis.get('covered_lines', 0)}

"""

        if comparison:
            report += f"""## Trend Analysis

- **Previous Coverage:** {comparison.get('baseline_coverage', 0):.1f}%
- **Coverage Change:** {comparison.get('coverage_change', 0):+.1f}%
- **New Files:** {comparison.get('new_files', 0)}

"""

        # Top files by coverage
        report += "## File Coverage Ranking\n\n"
        file_coverage = analysis.get('file_coverage', {})
        sorted_files = sorted(file_coverage.items(), key=lambda x: x[1])

        report += "| File | Coverage | Status |\n"
        report += "|------|----------|--------|\n"

        for file_path, coverage in sorted_files[:10]:
            status = "🟢" if coverage >= 80 else "🟡" if coverage >= 50 else "🔴"
            report += f"| `{file_path}` | {coverage:.1f}% | {status} |\n"

        # Recommendations
        recommendations = analysis.get('recommendations', [])
        if recommendations:
            report += "\n## Recommendations\n\n"
            for rec in recommendations:
                report += f"- {rec}\n"

        return report

    def run_monitoring_cycle(self) -> Dict:
        """Run complete monitoring cycle"""
        logger.info("Starting coverage monitoring cycle")

        # Run analysis
        analysis = self.run_coverage_analysis()
        if not analysis:
            return {'status': 'failed', 'message': 'Analysis failed'}

        # Load baseline and compare
        baseline = self.load_latest_baseline()
        comparison = self.compare_with_baseline(analysis, baseline)

        # Generate report
        report = self.generate_report(analysis, comparison)

        # Save new baseline
        baseline_file = self.save_baseline(analysis)

        # Save report
        report_path = Path("coverage_monitoring_report.md")
        report_path.write_text(report, encoding='utf-8')

        result = {
            'status': 'success',
            'analysis': analysis,
            'comparison': comparison,
            'report_path': str(report_path),
            'baseline_saved': baseline_file
        }

        logger.info(f"Monitoring cycle complete: {result}")
        return result


def main():
    """Main execution"""
    print("=" * 60)
    print("  SUPREME SYSTEM V5 - COVERAGE TRACKER")
    print("=" * 60)
    print()

    tracker = CoverageTracker()

    print("📊 Running coverage analysis...")
    result = tracker.run_monitoring_cycle()

    if result['status'] == 'success':
        analysis = result['analysis']
        comparison = result.get('comparison', {})

        print("✅ Analysis Complete")
        print(f"   Coverage: {analysis.get('coverage_percent', 0):.1f}%")
        print(f"   Files: {analysis.get('covered_files', 0)}/{analysis.get('total_files', 0)}")
        print(f"   Lines: {analysis.get('covered_lines', 0)}/{analysis.get('total_lines', 0)}")

        if 'baseline_coverage' in comparison:
            change = comparison.get('coverage_change', 0)
            print(f"   Change: {change:+.1f}% from baseline")

        print(f"   Report: {result['report_path']}")
        print(f"   Baseline: {result['baseline_saved']}")

        # Check against targets
        coverage_target = float(os.getenv('COVERAGE_TARGET', 80))
        current = analysis.get('coverage_percent', 0)

        if current >= coverage_target:
            print(f"\n🎉 TARGET ACHIEVED: {current:.1f}% >= {coverage_target}%")
        else:
            gap = coverage_target - current
            print(f"\n⚠️  TARGET GAP: {gap:.1f}% remaining to reach {coverage_target}%")

    else:
        print(f"\n❌ Analysis Failed: {result.get('message', 'Unknown error')}")
        return 1

    return 0


if __name__ == "__main__":
    exit(main())

