# SUPREME_OPTIMIZATION_ROADMAP.md – REMAINING TASKS (For Direct Repo Integration)

> **Instructions:**
> Remove all achievements, only keep unfinished or partially verified items below.
> For each task, write a clear, step-by-step professional prompt in the imperative form, optimized for code-oriented AI.

---

### 1. **Full Strategy Integration & Wiring**

- Prompt: "Comprehensively audit all trading strategies to ensure UltraOptimizedEMA, UltraOptimizedRSI, UltraOptimizedMACD, CircularBuffer, and SmartEventProcessor are fully and properly wired in replacement of all legacy TechnicalIndicators. Refactor any partial or old code paths. Add unit tests to validate parity (max tolerance 1e-6) versus reference implementations on historical data."

### 2. **Dependency & Import Consistency Validation**

- Prompt: "Automatically detect and resolve any missing imports, package dependencies, or version mismatches in the repo. Ensure requirements.txt fully covers all runtime and test dependencies. Add a self-test command that checks Python environment readiness for both Windows and Linux targets."

### 3. **Configuration Sync & Docs**

- Prompt: "Reconcile and auto-sync `.env.example` with `.env.optimized`. Ensure all critical production parameters and optimization flags are documented and easily adjustable. Update README Quick Start for optimized run commands and troubleshooting coverage, with explicit attention to hardware constraint settings."

### 4. **Benchmark & Performance Data Verification**

- Prompt: "Enforce actual benchmark runs prior to completion claims. Integrate scripts/benchoptimized.py and scripts/loadsinglesymbol.py into a make test-bench pipeline that auto-generates a performance metrics report comparing current and reference implementation (CPU, RAM, latency, accuracy parity). Block roadmap close if data missing or divergent."

### 5. **AB Testing & Monitoring Validation**

- Prompt: "Wire up A/B testing automation so every config change triggers a 24-hour head-to-head backtest of Optimized vs Baseline. Ensure automated statistical validation of PnL, drawdown, win rate, latency, CPU/RAM, with a clear pass/fail summary. Ensure Grafana dashboards and all SLO alerts (CPU, RAM, latency, uptime) are tested and exporting reports (CSV/JSON)."

### 6. **Production Entry Point & Crash Resilience**

- Prompt: "Verify that realtimebacktest.py and the main entry point are robust to all exit code -1 causes (import errors, config missing, corrupted state). Implement error diagnosis output for fast triage and full error coverage in docs."

### 7. **Commit & Reporting Protocol**

- Prompt: "Enforce that every roadmap milestone commit includes: (a) precise file/line references, (b) before–after diff summary, (c) test results artifact. All reporting must be factual, data-backed, and reference real production hardware metrics."

---

**Note:** Each prompt above must be kept until its full technical criteria are met, with real metric validation, not just passing CI or human claim. Update this file upon each roadmap milestone and after every major refactor or hardware/environment change.
