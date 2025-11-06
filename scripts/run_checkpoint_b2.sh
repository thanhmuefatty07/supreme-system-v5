#!/usr/bin/env bash
set -euo pipefail

mkdir -p test_results benchmark

echo "[Integration-B2] running adaptive pipeline → orchestrator..."
python -u tests/integration/test_system_integration.py > test_results/integration_report_B2.json

THR=$(jq -r '.pipeline.throughput_items_per_s' test_results/integration_report_B2.json || echo 0)
EXEC=$(jq -r '.orchestrator.executed' test_results/integration_report_B2.json || echo 0)
SKIP=$(jq -r '.orchestrator.skipped' test_results/integration_report_B2.json || echo 0)

echo "Throughput: ${THR} items/s, Executed: ${EXEC}, Skipped: ${SKIP}"

# Simple KPIs for pass/fail (tunable): throughput > 500 items/s and executed >= 2
PASS=1
awk 'BEGIN{exit !(('