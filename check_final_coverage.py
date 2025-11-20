#!/usr/bin/env python3
import json
import os

def check_final_coverage():
    """Check final coverage after all fixes"""
    # Try to find the latest coverage file
    coverage_files = ['final_total_coverage.json', 'final_coverage_after_combo.json', 'final_coverage_after_live_trading.json', 'final_coverage_after_risk.json', 'final_coverage_after_backtester.json', 'new_coverage.json', 'coverage.json']
    coverage_file = None
    for file in coverage_files:
        if os.path.exists(file):
            coverage_file = file
            break

    if not coverage_file:
        print('Coverage file not found')
        return

    try:
        with open(coverage_file, 'r') as f:
            data = json.load(f)

        totals = data['totals']
        percent = totals['percent_covered']
        covered = totals['covered_lines']
        total = covered + totals['missing_lines']

        print(f'🎯 FINAL COVERAGE AFTER BACKTESTER FIX: {percent:.2f}%')
        print(f'   Covered Lines: {covered}')
        print(f'   Total Lines: {total}')

        # Compare with initial
        if os.path.exists('new_coverage.json'):
            with open('new_coverage.json', 'r') as f:
                old_data = json.load(f)
            old_percent = old_data['totals']['percent_covered']
            improvement = percent - old_percent
            print(f'   Improvement from Data Pipeline: +{improvement:.2f}%')

        # Compare with original
        if os.path.exists('coverage.json'):
            with open('coverage.json', 'r') as f:
                orig_data = json.load(f)
            orig_percent = orig_data['totals']['percent_covered']
            total_improvement = percent - orig_percent
            print(f'   Total Improvement from 7.24%: +{total_improvement:.2f}%')

    except Exception as e:
        print(f'Error: {e}')

if __name__ == "__main__":
    check_final_coverage()

