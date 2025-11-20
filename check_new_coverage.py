#!/usr/bin/env python3
import json
import os

def check_coverage():
    """Check new coverage status"""
    if not os.path.exists('new_coverage.json'):
        print('Coverage file not found')
        return

    try:
        with open('new_coverage.json', 'r') as f:
            data = json.load(f)

        totals = data['totals']
        percent = totals['percent_covered']
        covered = totals['covered_lines']
        total = covered + totals['missing_lines']

        print(f'NEW COVERAGE: {percent:.2f}%')
        print(f'COVERED LINES: {covered}')
        print(f'TOTAL LINES: {total}')

        # Compare with old coverage
        if os.path.exists('final_coverage.json'):
            with open('final_coverage.json', 'r') as f:
                old_data = json.load(f)
            old_percent = old_data['totals']['percent_covered']
            improvement = percent - old_percent
            print(f'IMPROVEMENT: +{improvement:.2f}% from {old_percent:.2f}%')

    except Exception as e:
        print(f'Error: {e}')

if __name__ == "__main__":
    check_coverage()

