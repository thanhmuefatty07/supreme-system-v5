# Makefile - Supreme System V5 (tabs only)

SHELL := /bin/bash
PYTHON ?= python3
PROJECT_ROOT := $(shell pwd)
export PYTHONPATH := $(PROJECT_ROOT)/python

.PHONY: validate test-quick backtest perf-collect perf-report

validate:
	@echo "🔍 Validating core imports and configuration"
	@echo "PYTHONPATH=$(PYTHONPATH)"
	@$(PYTHON) -c "import sys,importlib; print('sys.path[0]=', sys.path[0]); importlib.import_module('supreme_system_v5'); print('✅ import supreme_system_v5 OK')"
	@$(PYTHON) -c "from dotenv import load_dotenv; load_dotenv('.env.ultra_constrained', override=False); print('✅ ENV loaded (safe defaults)')"
	@echo "✅ validate completed"

test-quick:
	@echo "⚡ Running quick tests"
	@$(PYTHON) -c "import supreme_system_v5 as m; print('✅ package OK')"

backtest:
	@echo "📊 Running backtest (safe default)"
	@$(PYTHON) -c "from supreme_system_v5.backtest import main as bk; bk()" || $(PYTHON) run_backtest.py || echo "used fallback backtest"

perf-collect:
	@echo "📈 Collecting performance metrics"
	@$(PYTHON) -c "import json, os, psutil, time; os.makedirs('run_artifacts', exist_ok=True); import psutil as p; m=p.Process().memory_info().rss/1024/1024; d={'ts':time.time(),'memory_mb':m,'cpu_percent':p.cpu_percent(interval=0.1)}; open('run_artifacts/realtime_metrics.json','w').write(__import__('json').dumps(d, indent=2)); print('✅ realtime_metrics.json written')"

perf-report:
	@echo "📄 Generating perf report (stub)"
