# 🚀 Supreme System V5 - Clean Ultra-Constrained Makefile
# Production-ready cryptocurrency trading bot automation
# Cleaned up - no agent references, focused on core functionality

.PHONY: help install test lint format backtest validate clean perf-report
.DEFAULT_GOAL := help

# Configuration
PYTHON := python3
PYTHONPATH := $(PWD)/python
OUTPUT_DIR := run_artifacts

# Colors for output
COLOR_RESET := \033[0m
COLOR_BOLD := \033[1m
COLOR_GREEN := \033[32m
COLOR_YELLOW := \033[33m
COLOR_RED := \033[31m
COLOR_BLUE := \033[34m
COLOR_CYAN := \033[36m

help: ## Show help
	@echo "$(COLOR_BOLD)🚀 Supreme System V5 - Ultra-Constrained Trading Bot$(COLOR_RESET)"
	@echo ""
	@echo "$(COLOR_BOLD)🎯 Quick Commands:$(COLOR_RESET)"
	@echo "  $(COLOR_GREEN)make quick-start$(COLOR_RESET)         Install + validate + backtest"
	@echo "  $(COLOR_GREEN)make validate$(COLOR_RESET)            Full system validation"
	@echo "  $(COLOR_GREEN)make backtest$(COLOR_RESET)            Quick backtest ETH-USDT"
	@echo ""
	@echo "$(COLOR_BOLD)📊 Performance:$(COLOR_RESET)"
	@echo "  $(COLOR_BLUE)make perf-report$(COLOR_RESET)         Generate performance report"
	@echo "  $(COLOR_BLUE)make monitor$(COLOR_RESET)             Resource monitoring"
	@echo "  $(COLOR_BLUE)make status$(COLOR_RESET)              System status check"
	@echo ""
	@echo "$(COLOR_BOLD)📊 Development:$(COLOR_RESET)"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | grep -v "perf-report\|monitor\|status" | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "  $(COLOR_CYAN)%-20s$(COLOR_RESET) %s\n", $$1, $$2}'

install: ## Install ultra-constrained dependencies
	@echo "$(COLOR_BOLD)📦 Installing dependencies...$(COLOR_RESET)"
	$(PYTHON) -m pip install --upgrade --no-cache-dir pip
	$(PYTHON) -m pip install --no-cache-dir -r requirements-ultra.txt
	@echo "$(COLOR_GREEN)✅ Dependencies installed$(COLOR_RESET)"

install-dev: ## Install dev dependencies
	@echo "$(COLOR_BOLD)🔧 Installing dev tools...$(COLOR_RESET)"
	$(MAKE) install
	$(PYTHON) -m pip install --no-cache-dir pytest pytest-cov black isort flake8
	@echo "$(COLOR_GREEN)✅ Dev environment ready$(COLOR_RESET)"

format: ## Auto-format code
	@echo "$(COLOR_BOLD)🎨 Formatting code...$(COLOR_RESET)"
	black python/ --line-length=120
	isort python/ --profile=black
	@echo "$(COLOR_GREEN)✅ Code formatted$(COLOR_RESET)"

lint: ## Run code quality checks
	@echo "$(COLOR_BOLD)🔍 Running code quality checks...$(COLOR_RESET)"
	-black --check python/ || echo "$(COLOR_YELLOW)⚠️ Run 'make format'$(COLOR_RESET)"
	-isort --check-only python/ || echo "$(COLOR_YELLOW)⚠️ Run 'make format'$(COLOR_RESET)"
	-flake8 python/ --max-line-length=120 --ignore=E203,W503,E501
	@echo "$(COLOR_GREEN)✅ Code quality check complete$(COLOR_RESET)"

test-quick: ## Quick smoke tests
	@echo "$(COLOR_BOLD)⚡ Quick validation...$(COLOR_RESET)"
	cd python && PYTHONPATH=$(PYTHONPATH) $(PYTHON) -c \
		"import supreme_system_v5; print('✅ Core import OK')"
	cd python && PYTHONPATH=$(PYTHONPATH) $(PYTHON) -c \
		"from supreme_system_v5.strategies import ScalpingStrategy; print('✅ Strategy OK')"
	cd python && PYTHONPATH=$(PYTHONPATH) $(PYTHON) -c \
		"from supreme_system_v5.risk import RiskManager; print('✅ Risk manager OK')"
	@echo "$(COLOR_GREEN)✅ Quick validation completed$(COLOR_RESET)"

test: ## Run test suite
	@echo "$(COLOR_BOLD)🧪 Running tests...$(COLOR_RESET)"
	@mkdir -p $(OUTPUT_DIR)
	cd python && PYTHONPATH=$(PYTHONPATH) $(PYTHON) -m pytest tests/ -v \
		--cov=supreme_system_v5 --cov-report=xml \
		--junitxml=../$(OUTPUT_DIR)/test-results.xml --timeout=300
	@echo "$(COLOR_GREEN)✅ Tests completed$(COLOR_RESET)"

backtest: ## Quick backtest validation
	@echo "$(COLOR_BOLD)📊 Running backtest...$(COLOR_RESET)"
	$(PYTHON) run_backtest.py --duration 2 --symbol ETH-USDT
	@echo "$(COLOR_GREEN)✅ Backtest completed$(COLOR_RESET)"

backtest-extended: ## Extended 24h backtest
	@echo "$(COLOR_BOLD)📈 Running 24h extended backtest...$(COLOR_RESET)"
	$(PYTHON) run_backtest.py --duration 1440 --symbol ETH-USDT
	@echo "$(COLOR_GREEN)✅ Extended backtest completed$(COLOR_RESET)"

backtest-multi: ## Multi-symbol portfolio backtest
	@echo "$(COLOR_BOLD)🔄 Multi-symbol backtest...$(COLOR_RESET)"
	$(PYTHON) run_backtest.py --duration 60 --symbols ETH-USDT,BTC-USDT,SOL-USDT
	@echo "$(COLOR_GREEN)✅ Multi-symbol backtest completed$(COLOR_RESET)"

perf-report: ## Generate performance metrics report
	@echo "$(COLOR_BOLD)📊 Generating performance report...$(COLOR_RESET)"
	$(PYTHON) scripts/collect_metrics.py --report
	@echo "$(COLOR_GREEN)✅ Performance report generated$(COLOR_RESET)"
	@echo "$(COLOR_CYAN)View report: cat $(OUTPUT_DIR)/performance_report_*.md$(COLOR_RESET)"

perf-collect: ## Collect current performance metrics
	@echo "$(COLOR_BOLD)📊 Collecting performance metrics...$(COLOR_RESET)"
	$(PYTHON) scripts/collect_metrics.py
	@echo "$(COLOR_GREEN)✅ Metrics collected$(COLOR_RESET)"
	@echo "$(COLOR_CYAN)View summary: cat $(OUTPUT_DIR)/performance_summary_*.json$(COLOR_RESET)"

validate: ## Full system validation
	@echo "$(COLOR_BOLD)🎯 Full system validation...$(COLOR_RESET)"
	$(MAKE) install-dev
	$(MAKE) format
	$(MAKE) lint
	$(MAKE) test-quick
	$(MAKE) backtest
	@echo "$(COLOR_GREEN)🎉 System validation completed!$(COLOR_RESET)"

validate-extended: ## Extended validation with performance report
	@echo "$(COLOR_BOLD)🎯 Extended system validation...$(COLOR_RESET)"
	$(MAKE) validate
	$(MAKE) perf-collect
	@echo "$(COLOR_GREEN)🎉 Extended validation completed!$(COLOR_RESET)"

quick-start: ## Complete setup and validation
	@echo "$(COLOR_BOLD)🚀 Supreme System V5 Quick Start$(COLOR_RESET)"
	$(MAKE) install-dev
	$(MAKE) test-quick
	$(MAKE) backtest
	@echo "$(COLOR_GREEN)🎉 Quick start completed!$(COLOR_RESET)"

run-paper: ## Start paper trading
	@echo "$(COLOR_BOLD)📈 Starting paper trading...$(COLOR_RESET)"
	@echo "$(COLOR_YELLOW)Press Ctrl+C to stop$(COLOR_RESET)"
	PYTHONPATH=python $(PYTHON) main.py --mode paper

run-live: ## Start live trading (REAL MONEY)
	@echo "$(COLOR_RED)⚠️ LIVE TRADING - REAL MONEY AT RISK ⚠️$(COLOR_RESET)"
	@read -p "Type 'CONFIRM_LIVE' to proceed: " confirm; \
	if [ "$$confirm" = "CONFIRM_LIVE" ]; then \
		PYTHONPATH=python $(PYTHON) main.py; \
	else \
		echo "$(COLOR_GREEN)Live trading cancelled$(COLOR_RESET)"; \
	fi

monitor: ## Resource monitoring
	@echo "$(COLOR_BOLD)📊 System monitor (Ctrl+C to stop)...$(COLOR_RESET)"
	$(PYTHON) -c "\
	import psutil, time; \
	print('🖥️ Resource Monitor - Target: <450MB RAM, <85% CPU'); \
	print('=' * 50); \
	try: \
		while True: \
			mem = psutil.virtual_memory(); \
			cpu = psutil.cpu_percent(interval=1); \
			rss = psutil.Process().memory_info().rss / (1024**2); \
			status = '🔴' if cpu > 85 or rss > 450 else '🟢'; \
			print(f'{status} CPU: {cpu:5.1f}% | Process: {rss:6.1f}MB | Available: {mem.available/1024**3:5.1f}GB', end='\r'); \
			time.sleep(2) \
	except KeyboardInterrupt: \
		print('\n🛑 Monitor stopped')"

status: ## System status
	@echo "$(COLOR_BOLD)📋 System Status$(COLOR_RESET)"
	@echo ""
	@$(PYTHON) -c "\
	import sys; \
	sys.path.insert(0, 'python'); \
	try: \
		import psutil; \
		from supreme_system_v5.strategies import ScalpingStrategy; \
		m = psutil.virtual_memory(); \
		print(f'✅ Python: {sys.version.split()[0]}'); \
		print(f'✅ RAM: {m.available/1024**3:.1f}GB available'); \
		print(f'✅ Core: ScalpingStrategy ready'); \
	except Exception as e: \
		print(f'❌ Issue: {e}')"

clean: ## Clean build artifacts
	@echo "$(COLOR_BOLD)🧹 Cleaning artifacts...$(COLOR_RESET)"
	@find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	@find . -type f -name "*.pyc" -delete 2>/dev/null || true
	@rm -rf build/ dist/ .coverage htmlcov/ .pytest_cache/ 2>/dev/null || true
	@echo "$(COLOR_GREEN)✅ Cleanup completed$(COLOR_RESET)"

# Make targets always run
.PHONY: $(shell grep -o '^[a-zA-Z_-]*:' $(MAKEFILE_LIST) | sed 's/://g' | grep -v '.PHONY')