#!/usr/bin/env python3
import json

def check_final_coverage():
    """Check final coverage status"""
    try:
        with open('final_coverage.json', 'r') as f:
            data = json.load(f)

        totals = data['totals']
        percent = totals['percent_covered']
        covered = totals['covered_lines']
        total = covered + totals['missing_lines']
        files = len(data['files'])

        print("🎯 FINAL COVERAGE STATUS:")
        print(f"  • Coverage: {percent:.2f}%")
        print(f"  • Covered Lines: {covered}")
        print(f"  • Total Lines: {total}")
        print(f"  • Files Covered: {files}")

        return percent, covered, total, files

    except Exception as e:
        print(f"❌ Error reading coverage: {e}")
        return None

if __name__ == "__main__":
    check_final_coverage()

