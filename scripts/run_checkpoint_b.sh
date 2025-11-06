#!/usr/bin/env bash
set -euo pipefail

mkdir -p test_results benchmark

echo "[Integration] running pipeline→orchestrator..."
python -u tests/integration/test_system_integration.py > test_results/integration_report.json

if [ -f tests/performance/realistic_benchmarks.py ]; then
  echo "[Perf] placeholder benchmarks..."
  python -u tests/performance/realistic_benchmarks.py > benchmark/perf_bench.json
fi

echo "Artifacts written to:"
echo " - test_results/integration_report.json"
echo " - benchmark/perf_bench.json (if present)"
