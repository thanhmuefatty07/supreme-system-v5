#!/usr/bin/env python3
import json
import os

def check_coverage():
    """Check current coverage status from coverage.json"""
    if not os.path.exists('coverage.json'):
        print("❌ coverage.json not found")
        return None

    try:
        with open('coverage.json', 'r') as f:
            data = json.load(f)

        totals = data['totals']
        percent = totals['percent_covered']
        files_count = len(data['files'])
        missing_lines = totals['missing_lines']
        covered_lines = totals['covered_lines']

        print(f"📊 CURRENT COVERAGE STATUS:")
        print(f"  • Total Coverage: {percent:.2f}%")
        print(f"  • Files Covered: {files_count}")
        print(f"  • Covered Lines: {covered_lines}")
        print(f"  • Missing Lines: {missing_lines}")

        return data

    except Exception as e:
        print(f"❌ Error reading coverage.json: {e}")
        return None

if __name__ == "__main__":
    check_coverage()
