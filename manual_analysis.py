#!/usr/bin/env python3
"""
Manual Enterprise Analysis - When Gemini quota is exceeded
Generates comprehensive coverage analysis and test generation strategy
"""

import json
from pathlib import Path
from datetime import datetime
from typing import List, Tuple, Dict

def load_coverage_data() -> Dict:
    """Load coverage data from coverage.json"""
    try:
        with open('coverage.json', 'r') as f:
            return json.load(f)
    except Exception as e:
        print(f"❌ Error loading coverage data: {e}")
        return None

def analyze_coverage_impact(coverage_data: Dict) -> List[Tuple]:
    """Analyze files by coverage impact"""
    files = coverage_data['files']
    file_impacts = []

    for file_path, data in files.items():
        if file_path.startswith('src') or file_path.startswith('src\\'):
            coverage_pct = data['summary']['percent_covered']
            lines = data['summary']['num_statements']
            impact = (100 - coverage_pct) * lines / 100  # Potential gain
            file_impacts.append((impact, file_path, coverage_pct, lines))

    file_impacts.sort(reverse=True)
    return file_impacts

def identify_quick_wins(file_impacts: List[Tuple]) -> List[Tuple]:
    """Identify quick win opportunities"""
    # Files with < 300 lines and low current coverage
    quick_wins = []
    for impact, file_path, coverage_pct, lines in file_impacts:
        if lines < 300 and coverage_pct < 50:
            quick_wins.append((impact, file_path, coverage_pct, lines))

    return quick_wins[:5]  # Top 5 quick wins

def generate_analysis_report():
    """Generate comprehensive analysis report"""
    print("🧠 MANUAL ANALYSIS - ENTERPRISE GRADE")
    print("=" * 50)

    coverage_data = load_coverage_data()
    if not coverage_data:
        return

    totals = coverage_data['totals']
    files = coverage_data['files']

    print("📊 COVERAGE SUMMARY:")
    print(f"   Total Coverage: {totals['percent_covered']:.1f}%")
    print(f"   Lines Covered: {totals['covered_lines']:,}")
    print(f"   Total Lines: {totals['num_statements']:,}")
    print(f"   Files Analyzed: {len(files)}")
    print()

    # Analyze top impact files
    file_impacts = analyze_coverage_impact(coverage_data)

    print("🎯 TOP 20 FILES BY IMPACT (Coverage × Lines):")
    print("Rank | File                              | Coverage | Lines | Impact | Est. Effort")
    print("-" * 80)

    for i, (impact, file_path, coverage_pct, lines) in enumerate(file_impacts[:20], 1):
        effort = 'Low' if lines < 200 else 'Medium' if lines < 500 else 'High'
        short_path = file_path.replace('src/', '')[:35]
        print(f"{i:2d}   | {short_path:<35} | {coverage_pct:6.1f}% | {lines:4d} | {impact:6.1f} | {effort}")

    print()
    print("🚀 QUICK WINS IDENTIFIED (Top 5):")
    quick_wins = identify_quick_wins(file_impacts)
    for i, (_, file_path, coverage_pct, lines) in enumerate(quick_wins, 1):
        print(f"   {i}. {file_path} ({coverage_pct:.1f}% → 80%+, {lines} lines)")

    print()
    print("📈 ANALYSIS COMPLETE - READY FOR MANUAL TEST GENERATION")
    print()

    # Save to file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(f"analysis_reports/{timestamp}")
    output_dir.mkdir(parents=True, exist_ok=True)

    report_content = f"""# Manual Enterprise Analysis Report
**Generated:** {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

## Coverage Summary
- **Total Coverage:** {totals['percent_covered']:.1f}%
- **Lines Covered:** {totals['covered_lines']:,}
- **Total Lines:** {totals['num_statements']:,}
- **Files Analyzed:** {len(files)}

## Top 5 Quick Wins
"""

    for i, (_, file_path, coverage_pct, lines) in enumerate(quick_wins, 1):
        report_content += f"{i}. `{file_path}` ({coverage_pct:.1f}% → 80%+, {lines} lines)\n"

    report_content += "\n## Top 20 Files by Impact\n"
    report_content += "| Rank | File | Coverage | Lines | Impact | Effort |\n"
    report_content += "|------|------|----------|-------|--------|--------|\n"

    for i, (impact, file_path, coverage_pct, lines) in enumerate(file_impacts[:20], 1):
        effort = 'Low' if lines < 200 else 'Medium' if lines < 500 else 'High'
        short_path = file_path.replace('src/', '')[:35]
        report_content += f"| {i} | `{short_path}` | {coverage_pct:.1f}% | {lines} | {impact:.1f} | {effort} |\n"

    report_content += "\n## Next Steps\n1. Generate tests for quick wins manually\n2. Focus on high-impact, low-effort files\n3. Target 80%+ coverage per file\n4. Commit progress incrementally"

    report_path = output_dir / "MANUAL_ANALYSIS_REPORT.md"
    report_path.write_text(report_content)

    print(f"💾 Report saved: {report_path}")
    return report_path

if __name__ == "__main__":
    report_path = generate_analysis_report()
    if report_path:
        print(f"\n📋 Next: Review {report_path} and start manual test generation!")
