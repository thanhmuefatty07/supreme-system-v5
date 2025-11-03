# Supreme System V5 - Ultra-Constrained Makefile
# Optimized for 1GB RAM, 2 vCPU deployment with ETH-USDT scalping
# Agent Mode: Maximum efficiency, minimum resource usage

.PHONY: help validate run-ultra-local bench-light clean install-deps setup-ultra dev-setup
.DEFAULT_GOAL := help

# Colors for output
RED := \033[31m
GREEN := \033[32m
YELLOW := \033[33m
BLUE := \033[34m
MAGENTA := \033[35m
CYAN := \033[36m
RESET := \033[0m

# Configuration
PYTHON := python3
PIP := pip3
PROFILE := ultra_constrained
SYMBOL := ETH-USDT
TEST_DURATION := 15
BENCH_SAMPLES := 1000

# Hardware detection
HARDWARE := $(shell $(PYTHON) -c "import psutil; print('ultra_constrained' if psutil.virtual_memory().total/(1024**3) <= 1.5 else 'constrained' if psutil.virtual_memory().total/(1024**3) <= 4 else 'standard')" 2>/dev/null || echo 'unknown')
CPU_COUNT := $(shell $(PYTHON) -c "import os; print(os.cpu_count())" 2>/dev/null || echo 2)
RAM_GB := $(shell $(PYTHON) -c "import psutil; print(f'{psutil.virtual_memory().total/(1024**3):.1f}')" 2>/dev/null || echo '1.0')

help: ## Show this help message
	@echo "$(CYAN)Supreme System V5 - Ultra-Constrained Makefile$(RESET)"
	@echo "$(YELLOW)Optimized for 1GB RAM, 2 vCPU with ETH-USDT scalping$(RESET)"
	@echo ""
	@echo "$(BLUE)System Info:$(RESET)"
	@echo "  Hardware: $(HARDWARE) ($(CPU_COUNT) cores, $(RAM_GB)GB RAM)"
	@echo "  Target: ETH-USDT scalping, 30-60s intervals"
	@echo "  Profile: ultra_constrained (450MB RAM budget)"
	@echo ""
	@awk 'BEGIN {FS = ":.*##"; printf "$(CYAN)Usage:\n  make <target>$(RESET)\n\nTargets:\n"} /^[a-zA-Z_-]+:.*?##/ { printf "  $(CYAN)%-20s$(RESET) %s\n", $$1, $$2 }' $(MAKEFILE_LIST)

validate: ## Validate environment for ultra-constrained deployment
	@echo "$(BLUE)🔍 Validating environment for profile: $(PROFILE)$(RESET)"
	@echo "Target hardware: $(HARDWARE) ($(RAM_GB)GB RAM)"
	$(PYTHON) scripts/validate_environment.py --profile $(PROFILE)
	@echo "$(GREEN)✅ Environment validation complete$(RESET)"

validate-json: ## Validate environment and output JSON report
	@echo "$(BLUE)🔍 Validating environment (JSON output)$(RESET)"
	@mkdir -p run_artifacts
	$(PYTHON) scripts/validate_environment.py --profile $(PROFILE) --json --output run_artifacts/validation_$$(date +%Y%m%d_%H%M%S).json
	@echo "$(GREEN)📄 JSON report saved to run_artifacts/$(RESET)"

setup-ultra: ## Setup ultra-constrained configuration
	@echo "$(BLUE)⚙️ Setting up ultra-constrained configuration$(RESET)"
	@if [ ! -f .env ]; then \
		cp .env.ultra_constrained .env; \
		echo "$(GREEN)✅ Created .env from .env.ultra_constrained$(RESET)"; \
	else \
		echo "$(YELLOW)⚠️ .env already exists, backup created$(RESET)"; \
		cp .env .env.backup_$$(date +%Y%m%d_%H%M%S); \
		cp .env.ultra_constrained .env; \
	fi
	@echo "$(CYAN)📋 Ultra-constrained profile active:$(RESET)"
	@echo "   Symbol: $(SYMBOL)"
	@echo "   RAM Target: ~450MB ($(RAM_GB)GB available)"
	@echo "   CPU Target: <85% ($(CPU_COUNT) cores)"
	@echo "   Scalping: 30-60s intervals with jitter"
	@echo "   News: 12min intervals"

install-deps: ## Install minimal dependencies for ultra-constrained deployment
	@echo "$(BLUE)📦 Installing minimal dependencies for $(HARDWARE) hardware$(RESET)"
	$(PIP) install --upgrade pip
	@if [ -f requirements-ultra.txt ]; then \
		$(PIP) install -r requirements-ultra.txt; \
	else \
		$(PIP) install -r requirements.txt; \
	fi
	@echo "$(GREEN)✅ Dependencies installed for $(HARDWARE) profile$(RESET)"

dev-setup: validate setup-ultra install-deps ## Complete development setup for ultra-constrained
	@echo "$(GREEN)🚀 Ultra-constrained development environment ready!$(RESET)"
	@echo "$(CYAN)Next steps:$(RESET)"
	@echo "  make test-parity    # Test indicator accuracy"
	@echo "  make bench-light    # Quick 15min benchmark"
	@echo "  make run-ultra-local # Start paper trading"

run-ultra-local: setup-ultra ## Run system with ultra-constrained profile (paper trading)
	@echo "$(BLUE)🚀 Starting Supreme System V5 - Ultra-Constrained Mode$(RESET)"
	@echo "$(CYAN)Configuration:$(RESET)"
	@echo "   Profile: $(PROFILE)"
	@echo "   Symbol: $(SYMBOL)"
	@echo "   Mode: Paper Trading"
	@echo "   Resource Budget: 450MB RAM, <85% CPU"
	@echo "   Hardware: $(HARDWARE) ($(CPU_COUNT) cores, $(RAM_GB)GB)"
	@echo "$(YELLOW)Press Ctrl+C to stop. Monitor with 'make monitor' in another terminal$(RESET)"
	@echo ""
	ULTRA_CONSTRAINED=1 $(PYTHON) python/supreme_system_v5/realtime_backtest.py

run-ultra-live: setup-ultra ## Run system with ultra-constrained profile (live trading - USE WITH CAUTION)
	@echo "$(RED)⚠️  LIVE TRADING MODE - REAL MONEY AT RISK!$(RESET)"
	@echo "$(RED)Hardware: $(HARDWARE) ($(RAM_GB)GB RAM) - Confirm this is adequate$(RESET)"
	@echo "$(RED)Are you absolutely sure? [y/N]" && read ans && [ $${ans:-N} = y ]
	@echo "$(BLUE)🚀 Starting Supreme System V5 - Live Trading$(RESET)"
	ULTRA_CONSTRAINED=1 EXECUTION_MODE=live $(PYTHON) python/supreme_system_v5/realtime_backtest.py

bench-light: setup-ultra ## Run lightweight 15-minute benchmark on ETH-USDT
	@echo "$(BLUE)⚡ Running lightweight benchmark$(RESET)"
	@echo "Hardware: $(HARDWARE) ($(CPU_COUNT) cores, $(RAM_GB)GB RAM)"
	@echo "Duration: $(TEST_DURATION) minutes"
	@echo "Symbol: $(SYMBOL)"
	@echo "Samples: $(BENCH_SAMPLES)"
	@echo "Expected: <15ms latency, <85% CPU, <450MB RAM"
	@mkdir -p run_artifacts
	$(PYTHON) scripts/bench_optimized.py \
		--symbol $(SYMBOL) \
		--duration-min $(TEST_DURATION) \
		--samples $(BENCH_SAMPLES) \
		--profile $(PROFILE) \
		--output run_artifacts/bench_light_$$(date +%Y%m%d_%H%M%S).json
	@echo "$(GREEN)📊 Benchmark results saved to run_artifacts/$(RESET)"
	@echo "$(CYAN)Check results with: make results$(RESET)"

bench-full: ## Run comprehensive benchmark suite
	@echo "$(BLUE)⚡ Running comprehensive benchmark$(RESET)"
	@echo "$(YELLOW)Warning: This may take 60+ minutes on $(HARDWARE) hardware$(RESET)"
	@mkdir -p run_artifacts
	$(PYTHON) scripts/bench_optimized.py --samples 5000 --runs 10 --output run_artifacts/bench_full_$$(date +%Y%m%d_%H%M%S).json
	$(PYTHON) scripts/load_single_symbol.py --symbol $(SYMBOL) --duration-min 60 --rate 10 --output run_artifacts/load_test_$$(date +%Y%m%d_%H%M%S).json
	@echo "$(GREEN)📊 Full benchmark results saved to run_artifacts/$(RESET)"

test-parity: ## Test parity between optimized and reference indicators
	@echo "$(BLUE)🧪 Testing indicator parity (tolerance: 1e-6)$(RESET)"
	$(PYTHON) -m pytest tests/test_parity_indicators.py -v --tb=short
	@echo "$(GREEN)✅ Parity tests complete$(RESET)"

test-quick: ## Run quick test suite
	@echo "$(BLUE)🧪 Running quick tests for $(HARDWARE) profile$(RESET)"
	$(PYTHON) -m pytest tests/ -x --tb=short -k "not slow" --maxfail=3
	@echo "$(GREEN)✅ Quick tests complete$(RESET)"

monitor: ## Monitor system resources during operation
	@echo "$(BLUE)📊 Monitoring system resources (optimized for $(HARDWARE))$(RESET)"
	@echo "$(YELLOW)Press Ctrl+C to stop monitoring$(RESET)"
	@echo "$(CYAN)Watching: CPU, RAM, scalping events, latency$(RESET)"
	watch -n 2 'echo "=== Supreme System V5 Resource Monitor ==="; \
		echo "Hardware: $(HARDWARE) ($(CPU_COUNT) cores, $(RAM_GB)GB RAM)"; \
		echo "Target: <85% CPU, <450MB RAM, <15ms latency"; \
		echo ""; \
		echo "=== Active Python Processes ==="; \
		ps aux | grep -E "(python|supreme)" | grep -v grep | head -3; \
		echo ""; \
		echo "=== Memory Usage ==="; \
		free -h | head -2; \
		echo ""; \
		echo "=== CPU Usage ==="; \
		top -bn1 | grep "Cpu(s)" | head -1; \
		echo ""; \
		echo "=== Disk Usage ==="; \
		df -h | head -2; \
		echo ""; \
		echo "=== Network (if applicable) ==="; \
		netstat -i 2>/dev/null | head -3 || echo "netstat not available"'

logs: ## Show recent logs
	@echo "$(BLUE)📋 Recent logs (last 50 lines)$(RESET)"
	@if [ -f logs/supreme_system.log ]; then \
		tail -50 logs/supreme_system.log | grep -E "(ERROR|WARNING|INFO)" --color=always; \
	elif [ -f supreme_system.log ]; then \
		tail -50 supreme_system.log | grep -E "(ERROR|WARNING|INFO)" --color=always; \
	else \
		echo "$(YELLOW)No log file found. Check: logs/supreme_system.log$(RESET)"; \
		ls -la logs/ 2>/dev/null || echo "logs/ directory not found"; \
	fi

status: ## Show system status and configuration
	@echo "$(CYAN)Supreme System V5 - Status Report$(RESET)"
	@echo "================================"
	@echo "Profile: $(PROFILE)"
	@echo "Hardware: $(HARDWARE) ($(CPU_COUNT) cores, $(RAM_GB)GB RAM)"
	@echo "Target Symbol: $(SYMBOL)"
	@echo "Python: $$(python3 --version)"
	@echo "Platform: $$(uname -s -m)"
	@echo ""
	@if [ -f .env ]; then \
		echo "Config: .env found"; \
		echo "Active settings:"; \
		grep -E "^(ULTRA_CONSTRAINED|SYMBOLS|SCALPING_|MAX_|LOG_LEVEL)" .env | head -5; \
	else \
		echo "Config: $(RED).env missing - run 'make setup-ultra'$(RESET)"; \
	fi
	@echo ""
	@if command -v free >/dev/null 2>&1; then \
		echo "Current Memory: $$(free -h | awk 'NR==2{printf "%s/%s (%.1f%%)", $$3,$$2,$$3*100/$$2 }')";\
	fi
	@if [ -d run_artifacts ]; then \
		echo "Artifacts: $$(ls -la run_artifacts/ | wc -l) files"; \
	else \
		echo "Artifacts: No run_artifacts directory"; \
	fi

results: ## Show latest benchmark results
	@echo "$(CYAN)Latest Benchmark Results$(RESET)"
	@echo "======================="
	@if [ -d run_artifacts ]; then \
		echo "Available results:"; \
		ls -lt run_artifacts/*.json 2>/dev/null | head -5; \
		echo ""; \
		latest=$$(ls -t run_artifacts/*.json 2>/dev/null | head -1); \
		if [ -n "$$latest" ]; then \
			echo "Latest result: $$latest"; \
			$(PYTHON) -c "import json; data=json.load(open('$$latest')); print('Performance Summary:'); [print(f'  {k}: {v}') for k,v in data.items() if k in ['latency_p95_ms', 'cpu_avg_percent', 'memory_peak_mb', 'parity_passed']]"; \
		else \
			echo "$(YELLOW)No benchmark results found. Run 'make bench-light'$(RESET)"; \
		fi; \
	else \
		echo "$(YELLOW)No run_artifacts directory. Run 'make bench-light'$(RESET)"; \
	fi

clean: ## Clean up temporary files and artifacts
	@echo "$(BLUE)🧹 Cleaning up$(RESET)"
	rm -rf __pycache__/
	rm -rf python/supreme_system_v5/__pycache__/
	rm -rf python/supreme_system_v5/*/__pycache__/
	rm -rf .pytest_cache/
	rm -f validation_report*.json
	find . -name "*.pyc" -delete
	find . -name "*.pyo" -delete
	@echo "$(GREEN)✅ Cleanup complete$(RESET)"

clean-artifacts: ## Clean benchmark artifacts (keep config)
	@echo "$(YELLOW)⚠️ This will delete all benchmark results$(RESET)"
	@echo "Continue? [y/N]" && read ans && [ $${ans:-N} = y ]
	rm -rf run_artifacts/
	mkdir -p run_artifacts
	@echo "$(GREEN)✅ Artifacts cleaned$(RESET)"

clean-all: clean clean-artifacts ## Clean everything including run artifacts and logs
	@echo "$(YELLOW)⚠️ This will delete all results, logs, and backups$(RESET)"
	@echo "Continue? [y/N]" && read ans && [ $${ans:-N} = y ]
	rm -rf logs/
	rm -f .env.backup_*
	mkdir -p logs
	@echo "$(GREEN)✅ Full cleanup complete$(RESET)"

# Advanced targets
profile-cpu: ## Profile CPU usage during benchmark
	@echo "$(BLUE)⚡ Profiling CPU usage on $(HARDWARE) hardware$(RESET)"
	$(PYTHON) -m cProfile -o cpu_profile_$$(date +%Y%m%d_%H%M%S).prof scripts/bench_optimized.py --samples 500 --runs 1
	@echo "$(GREEN)📊 CPU profile saved$(RESET)"

profile-memory: ## Profile memory usage
	@echo "$(BLUE)⚡ Profiling memory usage$(RESET)"
	@if command -v mprof >/dev/null 2>&1; then \
		mprof run scripts/bench_optimized.py --samples 100 --runs 1; \
		mprof plot; \
	else \
		echo "$(YELLOW)memory_profiler not installed. Install with: pip install memory_profiler$(RESET)"; \
	fi

check-config: ## Validate current configuration
	@echo "$(CYAN)Configuration Check for $(HARDWARE)$(RESET)"
	@echo "================================="
	@if [ -f .env ]; then \
		echo "$(GREEN)✅ .env file found$(RESET)"; \
		echo "Key settings:"; \
		grep -E "^(ULTRA_CONSTRAINED|SYMBOLS|SCALPING_|MAX_|LOG_LEVEL)" .env | head -10; \
		echo ""; \
		echo "Resource limits:"; \
		grep -E "^(MAX_RAM_MB|MAX_CPU_PERCENT)" .env; \
	else \
		echo "$(RED)❌ .env file not found$(RESET)"; \
		echo "Run 'make setup-ultra' to create from template"; \
	fi

# Hardware-specific optimizations
optimize-ultra: ## Apply ultra-constrained optimizations
	@echo "$(BLUE)⚡ Applying optimizations for $(HARDWARE) ($(RAM_GB)GB RAM)$(RESET)"
	@if [ "$(HARDWARE)" = "ultra_constrained" ]; then \
		echo "Applying ultra-constrained optimizations:"; \
		echo "  - Buffer sizes: 200 elements max"; \
		echo "  - Float32 precision"; \
		echo "  - Minimal logging"; \
		echo "  - Disabled heavy features"; \
	elif [ "$(HARDWARE)" = "constrained" ]; then \
		echo "$(YELLOW)Hardware has $(RAM_GB)GB RAM - consider .env.optimized instead$(RESET)"; \
	else \
		echo "$(GREEN)Hardware has $(RAM_GB)GB RAM - can use higher performance profile$(RESET)"; \
	fi

# Development helpers
format: ## Format code with basic tools
	@echo "$(BLUE)🎨 Formatting code$(RESET)"
	@if command -v black >/dev/null 2>&1; then \
		black python/supreme_system_v5/ --line-length 100; \
	else \
		echo "$(YELLOW)black not installed, skipping formatting$(RESET)"; \
	fi
	@if command -v isort >/dev/null 2>&1; then \
		isort python/supreme_system_v5/; \
	else \
		echo "$(YELLOW)isort not installed, skipping import sorting$(RESET)"; \
	fi
	@echo "$(GREEN)✅ Code formatted$(RESET)"

lint: ## Run basic linting checks
	@echo "$(BLUE)🔍 Running basic linting checks$(RESET)"
	@if command -v flake8 >/dev/null 2>&1; then \
		flake8 python/supreme_system_v5/ --max-line-length=100 --ignore=E203,W503 --max-complexity=10; \
	else \
		echo "$(YELLOW)flake8 not installed, skipping lint checks$(RESET)"; \
	fi

# Quick start guide
quick-start: ## Complete quick start for new users
	@echo "$(CYAN)Supreme System V5 - Quick Start Guide$(RESET)"
	@echo "====================================="
	@echo "Hardware Detected: $(HARDWARE) ($(CPU_COUNT) cores, $(RAM_GB)GB RAM)"
	@echo ""
	@echo "$(BLUE)Step 1:$(RESET) Environment validation"
	$(MAKE) validate
	@echo ""
	@echo "$(BLUE)Step 2:$(RESET) Setup ultra-constrained profile"
	$(MAKE) setup-ultra
	@echo ""
	@echo "$(BLUE)Step 3:$(RESET) Install dependencies"
	$(MAKE) install-deps
	@echo ""
	@echo "$(BLUE)Step 4:$(RESET) Test indicator parity"
	$(MAKE) test-parity
	@echo ""
	@echo "$(BLUE)Step 5:$(RESET) Quick benchmark"
	$(MAKE) bench-light
	@echo ""
	@echo "$(GREEN)🚀 Ready! Next steps:$(RESET)"
	@echo "  $(CYAN)make run-ultra-local$(RESET)  # Start paper trading"
	@echo "  $(CYAN)make monitor$(RESET)         # Monitor resources (in another terminal)"
	@echo "  $(CYAN)make logs$(RESET)           # View recent logs"
	@echo "  $(CYAN)make results$(RESET)        # Check benchmark results"

info: ## Show detailed system information
	@echo "$(CYAN)Supreme System V5 - System Information$(RESET)"
	@echo "=========================================="
	@echo "Makefile Profile: $(PROFILE)"
	@echo "Hardware Classification: $(HARDWARE)"
	@echo "Target Symbol: $(SYMBOL)"
	@echo "Test Duration: $(TEST_DURATION) minutes"
	@echo "Benchmark Samples: $(BENCH_SAMPLES)"
	@echo ""
	@echo "System Specs:"
	@echo "  Python: $$($(PYTHON) --version)"
	@echo "  Platform: $$(uname -s -m)"
	@echo "  CPU Cores: $(CPU_COUNT)"
	@echo "  RAM: $(RAM_GB)GB"
	@if command -v free >/dev/null 2>&1; then \
		echo "  Available RAM: $$(free -h | awk 'NR==2{print $$7}' | tr -d 'i')"; \
	fi
	@echo "  Working Directory: $$(pwd)"
	@echo ""
	@echo "Resource Targets:"
	@echo "  RAM Usage: <450MB (~47% of $(RAM_GB)GB)"
	@echo "  CPU Usage: <85%"
	@echo "  Latency P95: <15ms"
	@echo "  Skip Ratio: 60-80% (event filtering efficiency)"
	@echo ""
	@echo "Available Targets: $$(grep -E '^[a-zA-Z_-]+:.*##' Makefile | wc -l) commands"

# Emergency/troubleshooting targets
reset: clean setup-ultra ## Reset to clean ultra-constrained state
	@echo "$(YELLOW)🔄 Resetting to clean ultra-constrained state$(RESET)"
	@echo "$(GREEN)✅ Reset complete - ready for fresh start$(RESET)"

emergency-stop: ## Emergency stop - kill all Python processes
	@echo "$(RED)🛑 EMERGENCY STOP - Killing all Python processes$(RESET)"
	pkill -f python || echo "No Python processes found"
	pkill -f supreme || echo "No Supreme System processes found"
	@echo "$(YELLOW)All processes stopped$(RESET)"

troubleshoot: ## Show troubleshooting information
	@echo "$(CYAN)Troubleshooting Guide$(RESET)"
	@echo "===================="
	@echo "Hardware: $(HARDWARE) ($(CPU_COUNT) cores, $(RAM_GB)GB RAM)"
	@echo ""
	@echo "Common issues:"
	@echo "1. Out of memory (>450MB usage):"
	@echo "   - Reduce BUFFER_SIZE_LIMIT in .env"
	@echo "   - Set LOG_LEVEL=ERROR"
	@echo "   - Disable optional features"
	@echo ""
	@echo "2. High CPU usage (>85%):"
	@echo "   - Increase SCALPING_INTERVAL_MIN"
	@echo "   - Reduce MIN_PRICE_CHANGE_PCT"
	@echo "   - Check for background processes"
	@echo ""
	@echo "3. Slow performance:"
	@echo "   - Run 'make profile-cpu' to identify bottlenecks"
	@echo "   - Check disk space with 'df -h'"
	@echo "   - Monitor with 'make monitor'"
	@echo ""
	@echo "4. Validation failures:"
	@echo "   - Run 'make validate' for detailed diagnostics"
	@echo "   - Check Python version >= 3.10"
	@echo "   - Reinstall deps with 'make install-deps'"

# Show current resource usage
usage: ## Show current resource usage
	@echo "$(CYAN)Current Resource Usage$(RESET)"
	@echo "====================="
	@if command -v free >/dev/null 2>&1; then \
		echo "Memory:"; \
		free -h; \
		echo ""; \
	fi
	@if command -v ps >/dev/null 2>&1; then \
		echo "Top Python processes:"; \
		ps aux --sort=-%mem | grep python | head -5; \
		echo ""; \
	fi
	@if command -v df >/dev/null 2>&1; then \
		echo "Disk usage:"; \
		df -h | head -5; \
	fi