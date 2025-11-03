# 🟡 REMAINING WORK & RISKS — LIVE (v5)

Only the remaining actionable work and unresolved risks are listed. Completed items have been removed. Targets: CPU ≤ 88%, RAM ≤ 3.86GB (i3/4GB), scalping cadence 1–5m with adaptive gaps.

---

## 1) Strategy Integration → OptimizedTechnicalAnalyzer (PRIORITY)
- Prompt (Do this now):
  - Replace all uses of `TechnicalIndicators` in `python/supreme_system_v5/strategies.py` with `OptimizedTechnicalAnalyzer`.
  - Inject `SmartEventProcessor` gating into the strategy loop; gate by price Δ%, volume spike multiplier, or max gap time.
  - Bound price history using `CircularBuffer(size=200)` (i3 constraint) or `deque(maxlen=1000)` as fallback.
- Technical steps:
  - Import: `from supreme_system_v5.optimized import OptimizedTechnicalAnalyzer, SmartEventProcessor, CircularBuffer`.
  - Strategy state: `self.ta = OptimizedTechnicalAnalyzer(cfg); self.events = SmartEventProcessor(evt_cfg); self.history = CircularBuffer(200)`.
  - On tick: if `self.events.should_process(price, volume, ts):` then compute; else skip.
  - Keep indicator parity unit tests against reference (tolerance ≤ 1e-6).
- Success criteria:
  - CPU median ↓ ≥ 35% on 60-min single-symbol 1m stream.
  - EMA/RSI/MACD parity ≤ 1e-6 vs reference.
  - Event skip ratio ∈ [0.2, 0.8].
- ETA: 0.5–1 day

## 2) Config Defaults & Docs (Plug & Play)
- Prompt:
  - Update `.env.example` and README to expose all optimized toggles/intervals/limits identical to `.env.optimized`.
  - Document single-symbol focus, gap scheduling (30s–10m), and resource caps.
- Technical steps:
  - Mirror keys from `.env.optimized` into `.env.example` and README snippets.
  - Validate loader handles missing keys with sane defaults.
- Success criteria: Fresh clone runs optimized mode with minimal edits.
- ETA: 0.25 day

## 3) Benchmarks & Load Tests (Numbers or it didn’t happen)
- Prompt:
  - Run micro-bench (EMA/RSI/MACD parity + latency), then 60-min load test at 10–20 tps.
  - Export CSV + Prometheus metrics for dashboards.
- Technical steps:
  - `python scripts/bench_optimized.py --samples 5000 --runs 10 --prometheus-port 9091`.
  - `python scripts/load_single_symbol.py --symbol BTC-USDT --rate 10 --duration-min 60 --prometheus-port 9092`.
  - Persist CSV in `run_artifacts/` and add Makefile targets.
- Success criteria:
  - Indicator median < 0.2ms; p95 < 0.5ms; parity pass.
  - CPU avg < 88%; RAM peak < 3.86GB; no GC stalls > 50ms.
- ETA: 0.75 day

## 4) Confidence Fusion → Dynamic Risk Manager
- Prompt:
  - Fuse `technical_confidence`, `news_confidence`, `whale_confidence` → `composite_confidence`.
  - Map to position size/leverage with volatility adjustment (bounds 5–50x).
- Technical steps:
  - Add `SignalConfidence` in risk module with weights (default Tech 0.4, News 0.35, Whale 0.25) and regime-aware scaling.
  - `size = f(confidence, volatility); leverage = clamp(g(confidence, regime), 5, 50)`; log breakdown and decisions.
  - Unit tests for monotonicity and bounds.
- Success criteria: Unit-tested curve; logs include source breakdown and final sizing/leverage.
- ETA: 0.5 day

## 5) Orchestrator Adaptive Policies
- Prompt:
  - Implement backpressure and priority queue with preemption safety.
- Technical steps:
  - If CPU>88% for 60s → multiply intervals ×2 (cap 5m); recover when <75% for 2m.
  - Priority: risk > whale > news > technical > patterns; time-slice with quanta (e.g., 50–100ms) to avoid starvation.
  - Record fairness metrics (per-component runtime/share) in Prometheus.
- Success criteria: Stress shows widening→recovery, no starvation; fairness metrics stable.
- ETA: 0.75 day

## 6) Observability & SLOs
- Prompt:
  - Add dashboards and alerts for optimization KPIs and SLOs.
- Technical steps:
  - Prom KPIs: `indicator_update_latency_seconds`, `event_skip_ratio`, `strategy_latency_seconds`, `cpu_percent_gauge`, `memory_in_use_bytes`.
  - Grafana: provision JSON dashboards; alerts for CPU>88%, RAM>3.86GB, p95>500ms, uptime<99.9%.
  - Export CSV after tests for audit.
- Success criteria: Dashboards render all KPIs; test alerts fire.
- ETA: 0.5 day

## 7) 24h Sandbox A/B (Optimized vs Baseline)
- Prompt:
  - Run optimized vs baseline on same feed for 24h; publish report.
- Technical steps:
  - Scripts: `scripts/ab_test_run.sh`, `scripts/report_ab.py` (or reuse load test with two configs).
  - Metrics: PnL, max DD, win rate, latency p50/p95, CPU/RAM.
  - Statistical significance test; recommend winner.
- Success criteria: Optimized ≥ baseline on risk-adjusted metrics, or not worse with lower resource use.
- ETA: 1.0 day

---

# ⚠️ Unresolved Risks & Mitigations

## R1. Strategy chưa wired vào optimized engine
- Impact: Chưa đạt tiết kiệm CPU/RAM đầy đủ.
- Mitigation: Ưu tiên Task 1; parity tests; feature-flag fallback.

## R2. CPU spike khi burst news/whale
- Impact: Vượt 88% ngắn hạn.
- Mitigation: Backpressure (Task 5), API rate-limit, batch + debounce.

## R3. Thiếu số liệu benchmark thực
- Impact: Quyết định dựa ước lượng.
- Mitigation: Task 3 trong ngày; lưu CSV + Prom metrics; báo cáo `docs/reports/`.

## R4. Memory growth do lịch sử không giới hạn
- Impact: RAM vượt 3.86GB.
- Mitigation: `CircularBuffer(200)`/`deque(maxlen=1000)`; weekly memory audit.

## R5. Approximate mode divergence (>5%)
- Impact: Giảm chất lượng tín hiệu.
- Mitigation: Chỉ bật khi CPU cao; canary A/B; log sai số; auto-rollback.

## R6. Orchestrator fairness chưa chứng minh
- Impact: Starvation pipeline khi high-load.
- Mitigation: Priority+quanta; fairness metrics; stress test.

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
