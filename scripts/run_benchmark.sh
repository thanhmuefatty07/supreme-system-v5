#!/bin/bash
# Supreme System V5 - Performance Benchmark Runner
# Usage: bash scripts/run_benchmark.sh

set -e

echo "⚡ Running performance benchmarks..."

# Activate virtual environment if exists
if [ -d "venv" ]; then
    source venv/bin/activate
fi

# Install dependencies
pip install -q pytest pytest-benchmark

# Create benchmark output directory
mkdir -p benchmarks/results

# Run performance tests
echo "📊 Benchmarking strategy execution..."
pytest tests/performance/ \
    --benchmark-only \
    --benchmark-json=benchmarks/results/benchmark_$(date +%Y%m%d_%H%M%S).json \
    --benchmark-autosave \
    -v

# Parse latest benchmark
LATEST=$(ls -t benchmarks/results/*.json | head -1)

if [ -f "$LATEST" ]; then
    echo ""
    echo "✅ Benchmark Complete!"
    echo "📁 Results: $LATEST"
    echo ""
    echo "📊 Key Metrics:"
    python -c "
import json
import sys
try:
    data = json.load(open('$LATEST'))
    for bench in data.get('benchmarks', []):
        name = bench['name']
        mean = bench['stats']['mean'] * 1000  # Convert to ms
        p95 = bench['stats'].get('q95', 0) * 1000
        print(f'  • {name}: {mean:.2f}ms (P95: {p95:.2f}ms)')
except Exception as e:
    print(f'  Error parsing: {e}')
"
    echo ""
    echo "💡 Tip: Use this data for README and sales materials"
    echo "💡 Tip: Include timestamp and hardware specs for credibility"
fi

echo "✅ Done!"
