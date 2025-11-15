#!/bin/bash
# Supreme System V5 - Coverage Report Generator
# Usage: bash scripts/run_coverage.sh

set -e

echo "🔍 Running comprehensive test coverage..."

# Activate virtual environment if exists
if [ -d "venv" ]; then
    source venv/bin/activate
fi

# Install dependencies if needed
pip install -q pytest pytest-cov coverage

# Run tests with coverage
echo "📊 Generating coverage report..."
pytest tests/ \
    --cov=src \
    --cov-report=term \
    --cov-report=html \
    --cov-report=xml \
    --cov-report=json \
    -v

# Parse coverage percentage
COVERAGE=$(python -c "import json; print(json.load(open('coverage.json'))['totals']['percent_covered_display'])")

echo ""
echo "✅ Coverage Report Generated!"
echo "📈 Total Coverage: $COVERAGE%"
echo "📁 HTML Report: htmlcov/index.html"
echo "📄 XML Report: coverage.xml"
echo "📊 JSON Report: coverage.json"
echo ""
echo "💡 Tip: Open htmlcov/index.html in browser to view detailed coverage"
echo "💡 Tip: Upload coverage.xml to Codecov for public badge"

# Generate timestamp report
echo "Coverage: $COVERAGE% | Date: $(date)" > coverage_latest.txt

echo "✅ Done! Check coverage_latest.txt for quick reference."
