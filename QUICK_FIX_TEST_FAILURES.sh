#!/bin/bash
# Quick Fix Script for Test Failures
# Date: 2025-11-13

set -e

echo "🔧 Quick Fix: Installing Missing Dependencies"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

# Install missing dependencies
pip install numba psutil memory-profiler pytest-mock pytest-timeout

echo ""
echo "✅ Dependencies installed"
echo ""
echo "🔍 Running tests to identify failures..."
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

# Run tests and collect failures
python -m pytest tests/ --tb=no -q 2>&1 | grep -E "FAILED|ERROR" > test_failures.txt || true

FAILED_COUNT=$(grep -c "FAILED" test_failures.txt || echo "0")
ERROR_COUNT=$(grep -c "ERROR" test_failures.txt || echo "0")

echo "Found: $FAILED_COUNT failed tests, $ERROR_COUNT errors"
echo ""
echo "📋 Failure summary saved to: test_failures.txt"
echo ""
echo "Next steps:"
echo "  1. Review test_failures.txt"
echo "  2. Fix import errors"
echo "  3. Fix path issues"
echo "  4. Re-run tests: pytest tests/ -v"

