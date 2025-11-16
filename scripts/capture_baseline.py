"""
Capture baseline metrics before any improvements.

This serves as reference point for all future changes.
"""

import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path


def capture_test_baseline():
    """Run tests and capture results"""
    print("📊 Capturing test baseline...")
    
    # Run pytest with coverage
    result = subprocess.run(
        ["pytest", "tests/", "-v", "--cov=src", "--cov-report=json", "--cov-report=term"],
        capture_output=True,
        text=True
    )
    
    # Parse coverage
    coverage_data = {}
    if Path("coverage.json").exists():
        with open("coverage.json") as f:
            cov = json.load(f)
            coverage_data = {
                "total_coverage": cov["totals"]["percent_covered"],
                "lines_covered": cov["totals"]["covered_lines"],
                "lines_total": cov["totals"]["num_statements"]
            }
    
    # Count tests
    test_summary = {
        "passed": result.stdout.count(" PASSED"),
        "failed": result.stdout.count(" FAILED"),
        "skipped": result.stdout.count(" SKIPPED"),
        "total": result.stdout.count(" PASSED") + result.stdout.count(" FAILED")
    }
    
    baseline = {
        "timestamp": datetime.now().isoformat(),
        "tests": test_summary,
        "coverage": coverage_data,
        "exit_code": result.returncode,
        "stdout": result.stdout[-1000:] if len(result.stdout) > 1000 else result.stdout  # Last 1000 chars
    }
    
    return baseline


def main():
    print("=" * 80)
    print("BASELINE METRICS CAPTURE")
    print("=" * 80)
    
    try:
        git_commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"]
        ).decode().strip()
    except:
        git_commit = "unknown"
    
    baseline = {
        "captured_at": datetime.now().isoformat(),
        "git_commit": git_commit,
        "test_baseline": capture_test_baseline()
    }
    
    # Save to file
    output_file = Path("baselines") / f"baseline_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    output_file.parent.mkdir(exist_ok=True)
    
    with open(output_file, 'w') as f:
        json.dump(baseline, f, indent=2)
    
    # Also save as latest
    with open("baselines/baseline_latest.json", 'w') as f:
        json.dump(baseline, f, indent=2)
    
    print(f"\n✅ Baseline saved to: {output_file}")
    print(f"✅ Latest baseline: baselines/baseline_latest.json")
    
    # Print summary
    print("\n📊 SUMMARY:")
    print(f"   Tests: {baseline['test_baseline']['tests']['passed']} passed, "
          f"{baseline['test_baseline']['tests']['failed']} failed")
    print(f"   Coverage: {baseline['test_baseline']['coverage'].get('total_coverage', 'N/A')}%")
    
    return 0 if baseline['test_baseline']['exit_code'] == 0 else 1


if __name__ == "__main__":
    sys.exit(main())

