#!/usr/bin/env bash
set -euo pipefail

mkdir -p benchmark test_results

echo "[Perf] Running EMA/RSI/MACD benchmarks..."
python -u tests/performance/realistic_benchmarks.py > benchmark/perf_bench.json

if command -v jq >/dev/null 2>&1; then
  IMP_EMA=$(jq -r '.results.ema_ms.improvement' benchmark/perf_bench.json)
  IMP_RSI=$(jq -r '.results.rsi_ms.improvement' benchmark/perf_bench.json)
  IMP_MACD=$(jq -r '.results.macd_ms.improvement' benchmark/perf_bench.json)
  echo "Improvements — EMA:${IMP_EMA}x, RSI:${IMP_RSI}x, MACD:${IMP_MACD}x"
fi

echo "[NLP] Loading sentiment analyzer (guarded)..."
python - <<'PY'
from src.nlp.realistic_sentiment import RealisticSentimentAnalyzer
import asyncio
async def main():
    nlp = RealisticSentimentAnalyzer(memory_budget_mb=300)
    await nlp.load()
    scores = nlp.analyze_batch(["good", "bad", "great", "terrible"])  # small sanity check
    print({"nlp_scores_len": len(scores)})
asyncio.run(main())
PY

echo "Artifacts: benchmark/perf_bench.json"
