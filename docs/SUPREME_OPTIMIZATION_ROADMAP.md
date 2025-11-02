# 🟡 REMAINING WORK & RISKS — LIVE (v4)

This page lists ONLY the remaining actionable work and unresolved risks. All completed items have been removed for clarity. Targets: CPU ≤ 88%, RAM ≤ 3.86GB (i3/4GB), scalping cadence 1–5m with adaptive gaps.

---

## 1) Strategy Integration → OptimizedTechnicalAnalyzer (PRIORITY)
- Scope:
  - Replace TechnicalIndicators by OptimizedTechnicalAnalyzer in strategies.py.
  - Enable SmartEventProcessor gating (price/volume/time significance).
  - Cap histories using CircularBuffer or deque(maxlen) with N≤1000.
- Success Criteria:
  - CPU median ↓ ≥ 35% on single-symbol 1m stream (60 min).
  - EMA/RSI/MACD parity vs reference within 1e-6.
  - Event skip ratio in [0.2, 0.8] depending on regime.
- Owner: Core Strategy Team
- ETA: 0.5–1 day

## 2) Config Defaults & Docs (Plug & Play)
- Changes:
  - Update .env.example and README to expose optimized toggles and intervals.
- Keys:
  - OPTIMIZED_MODE=true
  - EVENT_DRIVEN_PROCESSING=true
  - SINGLE_SYMBOL=BTC-USDT
  - PROCESS_INTERVAL_SECONDS=30
  - NEWS_INTERVAL_MIN=10
  - WHALE_INTERVAL_MIN=10
  - MAX_RAM_GB=3.86
  - MAX_CPU_PERCENT=88
- Success Criteria: Fresh clone can run optimized mode with minimal edits.
- Owner: Platform
- ETA: 0.25 day

## 3) Benchmarks & Load Tests (Numbers or it didn’t happen)
- Scripts:
  - scripts/bench_optimized.py — microbench EMA/RSI/MACD vs reference
  - scripts/load_single_symbol.py — 20 ticks/sec for 60 min
- Metrics:
  - indicator_update_latency_seconds
  - event_skip_ratio
  - cpu_percent_gauge, memory_in_use_bytes
  - strategy_latency_seconds (p50/p95)
- Success Criteria:
  - Median indicator update < 0.2ms; p95 < 0.5ms
  - CPU avg < 88%; RAM peak < 3.86GB; no GC stalls > 50ms
- Owner: Perf Eng
- ETA: 0.75 day

## 4) Confidence Fusion → Dynamic Risk Manager
- Scope:
  - Combine technical/news/whale confidence → composite (weights adjustable by volatility).
  - Map composite → leverage (5–50x) and position size; incorporate volatility factor.
- Success Criteria:
  - Unit-tested confidence→size/leverage curve; bounds enforced in logs.
  - Decision logs include confidence breakdown per source.
- Owner: Risk
- ETA: 0.5 day

## 5) Orchestrator Adaptive Policies
- Scope:
  - Backpressure: if CPU>88% for 60s → double intervals; recover when stable.
  - Priority: risk > whale > news > technical > patterns; preemption-safe.
- Success Criteria:
  - Stress: intervals widen then recover; no starvation; fairness visible in metrics.
- Owner: Orchestration
- ETA: 0.75 day

## 6) Observability & SLOs
- Scope:
  - Add Prometheus KPIs + Grafana dashboards for optimization metrics.
  - SLOs: CPU<88%, RAM<3.86GB, p95 cycle < 500ms, uptime 99.9%.
- Success Criteria:
  - Dashboards render KPIs; alerts verified via test fires.
- Owner: SRE
- ETA: 0.5 day

## 7) 24h Sandbox A/B (Optimized vs Baseline)
- Scope:
  - Run both modes on same symbol feed; compare PnL, DD, win rate, latency, CPU/RAM.
- Artifacts:
  - docs/reports/ab_test_{date}.md with analysis & recommendation.
- Success Criteria:
  - Optimized ≥ baseline on risk-adjusted metrics, or not worse with lower resource use.
- Owner: QA/Trading
- ETA: 1.0 day

---

# ⚠️ Unresolved Risks & Mitigations

## R1. Strategy chưa wired vào optimized engine
- Impact: Chưa đạt toàn bộ tiết kiệm CPU/RAM.
- Mitigation: Ưu tiên Task 1; thêm test parity vs reference; feature-flag fallback.

## R2. CPU spike khi burst news/whale
- Impact: Có thể vượt 88% ngắn hạn.
- Mitigation: Backpressure (Task 5), rate limit API, batch xử lý, debounce event.

## R3. Thiếu số liệu benchmark thực
- Impact: Quyết định tối ưu dựa ước lượng.
- Mitigation: Task 3 trong ngày; lưu CSV + Prom metrics; báo cáo docs/reports/.

## R4. Bộ nhớ tăng theo thời gian do lịch sử không giới hạn
- Impact: RAM vượt 3.86GB.
- Mitigation: N≤1000 cho lịch sử; CircularBuffer; weekly memory audit.

## R5. Divergence giữa approximate mode và reference
- Impact: Giảm chất lượng tín hiệu > 5%.
- Mitigation: Chỉ bật approx khi CPU cao; canary A/B; log sai số; auto rollback.

## R6. Orchestrator fairness chưa được chứng minh
- Impact: Starvation của một số pipeline khi high-load.
- Mitigation: Priority queue + quanta; metric fairness; stress test trước deploy.

---

# ▶️ Quick Start (Optimized Mode)
```bash
export OPTIMIZED_MODE=true
export EVENT_DRIVEN_PROCESSING=true
export SINGLE_SYMBOL=BTC-USDT
export PROCESS_INTERVAL_SECONDS=30
export NEWS_INTERVAL_MIN=10
export WHALE_INTERVAL_MIN=10
python -m supreme_system_v5.core
```
