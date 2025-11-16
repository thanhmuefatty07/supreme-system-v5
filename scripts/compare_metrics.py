"""
Compare current metrics against baseline.

Usage: python scripts/compare_metrics.py --baseline baselines/baseline_latest.json
"""

import json
import argparse
from pathlib import Path
from typing import Dict


def load_json(filepath: str) -> Dict:
    with open(filepath) as f:
        return json.load(f)


def compare_tests(baseline: Dict, current: Dict) -> Dict:
    """Compare test metrics"""
    b_tests = baseline['test_baseline']['tests']
    c_tests = current['test_baseline']['tests']
    
    return {
        'passed_change': c_tests['passed'] - b_tests['passed'],
        'failed_change': c_tests['failed'] - b_tests['failed'],
        'total_change': c_tests['total'] - b_tests['total'],
        'baseline_passed': b_tests['passed'],
        'current_passed': c_tests['passed'],
        'status': '✅ PASS' if c_tests['failed'] == 0 else '❌ FAIL'
    }


def compare_coverage(baseline: Dict, current: Dict) -> Dict:
    """Compare coverage metrics"""
    b_cov = baseline['test_baseline']['coverage'].get('total_coverage', 0)
    c_cov = current['test_baseline']['coverage'].get('total_coverage', 0)
    
    change = c_cov - b_cov
    
    return {
        'baseline': b_cov,
        'current': c_cov,
        'change': change,
        'change_pct': (change / b_cov * 100) if b_cov > 0 else 0,
        'status': '✅ MAINTAINED' if change >= 0 else '⚠️  DECREASED'
    }


def generate_report(baseline: Dict, current: Dict) -> str:
    """Generate comparison report"""
    test_comp = compare_tests(baseline, current)
    cov_comp = compare_coverage(baseline, current)
    
    report = f"""
{'='*80}
METRICS COMPARISON REPORT
{'='*80}

📅 Baseline: {baseline['captured_at']}
📅 Current:  {current['captured_at']}

📊 TEST RESULTS
{'─'*80}
  Baseline: {test_comp['baseline_passed']} passing
  Current:  {test_comp['current_passed']} passing
  Change:   {test_comp['passed_change']:+d} tests
  Status:   {test_comp['status']}

📈 COVERAGE
{'─'*80}
  Baseline: {cov_comp['baseline']:.1f}%
  Current:  {cov_comp['current']:.1f}%
  Change:   {cov_comp['change']:+.1f}% ({cov_comp['change_pct']:+.1f}%)
  Status:   {cov_comp['status']}

{'='*80}
"""
    
    # Acceptance criteria
    report += "\n✅ ACCEPTANCE CRITERIA:\n"
    criteria = []
    
    if test_comp['failed_change'] <= 0:
        criteria.append("✅ No new test failures")
    else:
        criteria.append(f"❌ New test failures: {test_comp['failed_change']}")
    
    if cov_comp['change'] >= 0:
        criteria.append("✅ Coverage maintained or improved")
    else:
        criteria.append(f"⚠️  Coverage decreased by {abs(cov_comp['change']):.1f}%")
    
    for criterion in criteria:
        report += f"  {criterion}\n"
    
    # Overall status
    all_pass = all('✅' in c for c in criteria)
    report += f"\n🎯 OVERALL: {'✅ ACCEPTABLE' if all_pass else '❌ NEEDS IMPROVEMENT'}\n"
    
    return report


def main():
    # Set UTF-8 encoding for Windows console
    import sys
    import io
    if sys.platform == 'win32':
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    
    parser = argparse.ArgumentParser(description='Compare metrics against baseline')
    parser.add_argument('--baseline', default='baselines/baseline_latest.json',
                       help='Baseline metrics file')
    parser.add_argument('--current', default=None,
                       help='Current metrics file (if None, will capture now)')
    
    args = parser.parse_args()
    
    # Load baseline
    if not Path(args.baseline).exists():
        print(f"❌ Baseline file not found: {args.baseline}")
        print("💡 Run: python scripts/capture_baseline.py first")
        return 1
    
    baseline = load_json(args.baseline)
    
    # Load or capture current
    if args.current:
        current = load_json(args.current)
    else:
        # Capture current metrics (don't fail on test failures)
        import subprocess
        result = subprocess.run(["python", "scripts/capture_baseline.py"], 
                               capture_output=True)
        # Continue even if tests failed (exit code 1)
        current = load_json("baselines/baseline_latest.json")
    
    # Generate and print report
    report = generate_report(baseline, current)
    print(report)
    
    # Save report
    report_file = Path("reports") / f"comparison_{current['captured_at'].replace(':', '-').split('.')[0]}.txt"
    report_file.parent.mkdir(exist_ok=True)
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write(report)
    
    print(f"📄 Report saved to: {report_file}")
    
    return 0


if __name__ == "__main__":
    exit(main())

