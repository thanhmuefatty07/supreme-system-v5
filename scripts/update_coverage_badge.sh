#!/bin/bash
# Supreme System V5 - Auto-update Coverage Badge
# Usage: bash scripts/update_coverage_badge.sh

set -e

echo "🎯 Updating coverage badge..."

# Run coverage first
bash scripts/run_coverage.sh

# Parse coverage
if [ -f "coverage.json" ]; then
    COVERAGE=$(python -c "import json; print(json.load(open('coverage.json'))['totals']['percent_covered_display'])")
    
    echo "📊 Current coverage: $COVERAGE%"
    
    # Update README badge
    if [ -f "README.md" ]; then
        # Backup
        cp README.md README.md.bak
        
        # Replace badge
        sed -i "s/coverage-[0-9]*%25/coverage-${COVERAGE}%25/g" README.md
        
        echo "✅ Badge updated in README.md"
        echo "💡 Review changes and commit if correct"
        echo "💡 Backup saved to README.md.bak"
    else
        echo "❌ README.md not found"
    fi
else
    echo "❌ coverage.json not found. Run tests first."
fi
