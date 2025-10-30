# Supreme System V5 - Hybrid Python + Rust Makefile
# Comprehensive build system for world's first neuromorphic trading system

.PHONY: help install install-dev install-rust build-rust test clean docker benchmark

# === VARIABLES ===
PYTHON := python3
CARGO := cargo
MATURIN := maturin
SOURCE_DIR := python/supreme_system_v5
RUST_DIR := src
TEST_DIR := tests

# Hardware detection for optimization
HARDWARE := $(shell $(PYTHON) -c "import psutil; print('i3' if psutil.virtual_memory().total/(1024**3) <= 5 else 'i5' if psutil.virtual_memory().total/(1024**3) <= 10 else 'i7')" 2>/dev/null || echo 'unknown')
CPU_COUNT := $(shell $(PYTHON) -c "import os; print(os.cpu_count())" 2>/dev/null || echo 4)

help: ## Show this help message
	@echo "Supreme System V5 - Hybrid Python + Rust Build System"
	@echo "===================================================="
	@echo ""
	@echo "Hardware detected: $(HARDWARE) ($(CPU_COUNT) cores)"
	@echo ""
	@echo "Available targets:"
	@awk 'BEGIN {FS = ":.*?## "} /^[a-zA-Z_-]+:.*?## / {printf "  \033[36m%-20s\033[0m %s\n", $$1, $$2}' $(MAKEFILE_LIST)

# === INSTALLATION ===
install: install-rust install-python ## Install complete hybrid system

install-python: ## Install Python dependencies
	@echo "🐍 Installing Python dependencies..."
	$(PYTHON) -m pip install --upgrade pip
	$(PYTHON) -m pip install maturin
	$(PYTHON) -m pip install -e . --config-settings=build-args="--release"
	@echo "✅ Python installation complete"

install-rust: ## Install Rust toolchain and dependencies
	@echo "🦀 Installing Rust toolchain..."
	@if ! command -v cargo >/dev/null 2>&1; then \
		echo "Installing Rust..."; \
		curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y; \
		source ~/.cargo/env; \
	fi
	@echo "✅ Rust toolchain ready"

install-dev: install-rust ## Install development dependencies
	@echo "🔧 Installing development dependencies..."
	$(PYTHON) -m pip install -e ".[dev]"
	pre-commit install
	@echo "✅ Development environment ready"

# === RUST COMPILATION ===
build-rust: ## Build Rust engine with optimization
	@echo "🛠️ Building Rust engine..."
	@echo "Target hardware: $(HARDWARE)"
ifeq ($(HARDWARE),i3)
	@echo "Applying i3-4GB optimizations..."
	CARGO_TARGET_CPU=generic $(MATURIN) build --release --features "python"
else ifeq ($(HARDWARE),i5)
	@echo "Applying i5-8GB optimizations..."
	CARGO_TARGET_CPU=native $(MATURIN) build --release --features "python,simd"
else
	@echo "Applying i7+ maximum performance..."
	RUSTFLAGS="-C target-cpu=native -C target-feature=+avx2" $(MATURIN) build --release --features "python,simd,parallel"
endif
	@echo "✅ Rust engine built successfully"

build-rust-dev: ## Build Rust engine for development
	@echo "🔧 Building Rust engine (development)..."
	$(MATURIN) develop --features "python"
	@echo "✅ Development Rust engine ready"

build-wheel: build-rust ## Build distribution wheel
	@echo "🎃 Building distribution wheel..."
	$(MATURIN) build --release --out dist/
	@echo "✅ Wheel built in dist/"

# === TESTING ===
test: test-rust test-python ## Run all tests

test-rust: ## Run Rust tests
	@echo "🦀 Running Rust tests..."
	$(CARGO) test --release
	@echo "✅ Rust tests completed"

test-python: ## Run Python tests
	@echo "🐍 Running Python tests..."
	pytest $(TEST_DIR) -v --tb=short
	@echo "✅ Python tests completed"

test-integration: build-rust-dev ## Run integration tests (Python + Rust)
	@echo "🔗 Running integration tests..."
	pytest $(TEST_DIR) -v -m "integration" 
	@echo "✅ Integration tests completed"

test-performance: build-rust ## Run performance benchmarks
	@echo "🏁 Running performance benchmarks..."
	$(CARGO) bench
	pytest $(TEST_DIR) -v --benchmark-only --benchmark-sort=mean
	@echo "✅ Performance benchmarks completed"

# === CODE QUALITY ===
lint: lint-rust lint-python ## Run all linters

lint-rust: ## Lint Rust code
	@echo "🦀 Linting Rust code..."
	$(CARGO) clippy --all-features -- -D warnings
	$(CARGO) fmt --check
	@echo "✅ Rust linting completed"

lint-python: ## Lint Python code
	@echo "🐍 Linting Python code..."
	ruff check $(SOURCE_DIR)
	black --check $(SOURCE_DIR)
	mypy $(SOURCE_DIR)
	@echo "✅ Python linting completed"

format: format-rust format-python ## Format all code

format-rust: ## Format Rust code
	@echo "🦀 Formatting Rust code..."
	$(CARGO) fmt

format-python: ## Format Python code  
	@echo "🐍 Formatting Python code..."
	black $(SOURCE_DIR) $(TEST_DIR)
	ruff check --fix $(SOURCE_DIR) $(TEST_DIR)

# === VALIDATION ===
validate: build-rust-dev ## Validate system functionality
	@echo "🔍 Validating Supreme System V5..."
	$(PYTHON) -c "import supreme_system_v5; print('Python import: OK')"
	$(PYTHON) -c "import supreme_engine_rs; print('Rust engine: OK')" 2>/dev/null || echo "Rust engine: Not available"
	$(PYTHON) scripts/validate_system.py
	@echo "✅ System validation completed"

# === BENCHMARKING ===
benchmark: build-rust ## Run comprehensive benchmarks
	@echo "🏁 Running comprehensive benchmarks..."
	@echo "Hardware: $(HARDWARE) - $(CPU_COUNT) cores"
	
	@echo "\n1. Rust benchmarks:"
	$(CARGO) bench --features "simd,parallel"
	
	@echo "\n2. Python benchmarks:"
	pytest $(TEST_DIR) -v --benchmark-only
	
	@echo "\n3. Integration benchmarks:"
	$(PYTHON) -c "import supreme_system_v5; print('System info:', supreme_system_v5.get_system_info())"
	$(PYTHON) -c "import supreme_system_v5; print('Benchmark:', supreme_system_v5.benchmark_system(5))"
	
	@echo "✅ All benchmarks completed"

# === DEPLOYMENT ===
docker-build: ## Build Docker image for current hardware
	@echo "🐳 Building Docker image for $(HARDWARE)..."
ifeq ($(HARDWARE),i3)
	docker build -f docker/Dockerfile.i3 -t supreme-system-v5:$(HARDWARE)-optimized .
else
	docker build -f docker/Dockerfile.standard -t supreme-system-v5:latest .
endif
	@echo "✅ Docker image built"

docker-run: docker-build ## Run system in Docker
	@echo "🚀 Running Supreme System V5 in Docker..."
	docker run -p 8000:8000 -p 9090:9090 --env-file .env supreme-system-v5:latest

# === DEVELOPMENT ===
dev-setup: install-rust build-rust-dev install-dev ## Complete development setup
	@echo "🚀 Development environment ready!"
	@echo "Next steps:"
	@echo "  1. make validate    # Check system"
	@echo "  2. make test        # Run tests"
	@echo "  3. make benchmark   # Performance test"
	@echo "  4. make run-dev     # Start development server"

run-dev: build-rust-dev ## Start development server
	@echo "💻 Starting development server..."
	uvicorn supreme_system_v5.api:app --reload --host 0.0.0.0 --port 8000

run-prod: build-rust ## Start production server
	@echo "🏭 Starting production server..."
	gunicorn supreme_system_v5.api:app -w $(CPU_COUNT) -k uvicorn.workers.UvicornWorker --bind 0.0.0.0:8000

# === MONITORING ===
monitor: ## Start monitoring stack
	@echo "📈 Starting monitoring..."
	docker-compose -f docker/docker-compose.monitoring.yml up -d
	@echo "Grafana: http://localhost:3000"
	@echo "Prometheus: http://localhost:9090"

# === RELEASE ===
release: clean build-rust build-wheel test ## Build release package
	@echo "🏆 Building release package..."
	@echo "Python version: $(shell $(PYTHON) --version)"
	@echo "Rust version: $(shell $(CARGO) --version 2>/dev/null || echo 'Not available')"
	@echo "Hardware optimized for: $(HARDWARE)"
	@echo "✅ Release package ready in dist/"

# === MAINTENANCE ===
clean: ## Clean build artifacts
	@echo "🧹 Cleaning build artifacts..."
	$(CARGO) clean
	rm -rf target/ dist/ build/ *.egg-info/
	find . -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null || true
	find . -name "*.pyc" -delete
	rm -rf .pytest_cache/ .mypy_cache/ .coverage htmlcov/
	@echo "✅ Cleanup completed"

clean-logs: ## Clean log files
	@echo "🧹 Cleaning log files..."
	rm -rf logs/ *.log
	mkdir -p logs
	@echo "✅ Logs cleaned"

update-deps: ## Update all dependencies
	@echo "🔄 Updating dependencies..."
	$(CARGO) update
	$(PYTHON) -m pip install --upgrade pip setuptools wheel
	$(PYTHON) -m pip install --upgrade -e ".[dev]"
	@echo "✅ Dependencies updated"

# === QUICK COMMANDS ===
quick-test: build-rust-dev ## Quick test (no full rebuild)
	pytest $(TEST_DIR) -v --tb=short -x

quick-bench: ## Quick benchmark (5 second test)
	$(PYTHON) -c "import supreme_system_v5; print(supreme_system_v5.benchmark_system(5))"

fix-f541: ## Fix F541 errors using comprehensive fixer
	@echo "🔧 Fixing F541 errors..."
	$(PYTHON) scripts/fix_f541_comprehensive.py --live
	@echo "✅ F541 errors fixed"

# === STATUS CHECKS ===
status: ## Show system status
	@echo "Supreme System V5 - System Status"
	@echo "================================"
	@echo "Hardware: $(HARDWARE) ($(CPU_COUNT) cores)"
	@echo "Python: $(shell $(PYTHON) --version)"
	@echo "Rust: $(shell $(CARGO) --version 2>/dev/null || echo 'Not installed')"
	@echo "Maturin: $(shell $(MATURIN) --version 2>/dev/null || echo 'Not installed')"
	@echo ""
	@echo "Rust engine status:"
	@$(PYTHON) -c "try: import supreme_engine_rs; print('  ✅ Available:', supreme_engine_rs.__version__)\nexcept: print('  ❌ Not available')" 2>/dev/null
	@echo ""
	@echo "System validation:"
	@$(PYTHON) -c "import supreme_system_v5; info=supreme_system_v5.get_system_info(); [print(f'  {k}: {v}') for k,v in info.items()]"

health: build-rust-dev ## Check system health
	@echo "🏥 System health check..."
	$(PYTHON) -c "\
	try:\
		import supreme_system_v5;\
		system = supreme_system_v5.get_system();\
		status = system.get_system_status();\
		print('System Status:', status['is_running']);\
		print('Hardware:', status['hardware_profile']);\
		print('Performance:', status['performance']);\
	except Exception as e:\
		print('Health check failed:', e)"

# === CI/CD ===
ci: clean install test lint ## Full CI pipeline
	@echo "✅ CI pipeline completed successfully"

ci-fast: format quick-test lint-python ## Fast CI checks
	@echo "✅ Fast CI completed"

# === HARDWARE-SPECIFIC TARGETS ===
optimize-i3: ## Optimize for i3-4GB systems
	@echo "⚡ Optimizing for i3-4GB systems..."
	CARGO_TARGET_CPU=generic $(MATURIN) build --release --features "python" --no-default-features
	$(PYTHON) -c "import psutil; print(f'Memory usage: {psutil.virtual_memory().percent}%')"

optimize-i5: ## Optimize for i5-8GB systems  
	@echo "⚡ Optimizing for i5-8GB systems..."
	$(MATURIN) build --release --features "python,simd"

optimize-i7: ## Optimize for i7+ systems
	@echo "⚡ Optimizing for i7+ systems..."
	RUSTFLAGS="-C target-cpu=native -C target-feature=+avx2" $(MATURIN) build --release --features "python,simd,parallel"

# === PROFILING ===
profile-rust: ## Profile Rust performance
	@echo "🔍 Profiling Rust performance..."
	$(CARGO) build --release --features "python,simd,parallel"
	perf record --call-graph=dwarf target/release/benchmark
	perf report

profile-python: build-rust-dev ## Profile Python performance
	@echo "🔍 Profiling Python performance..."
	$(PYTHON) -m cProfile -o profile_output.prof -m supreme_system_v5.benchmark
	$(PYTHON) -c "import pstats; pstats.Stats('profile_output.prof').sort_stats('cumulative').print_stats(20)"

profile-memory: ## Profile memory usage
	@echo "💾 Profiling memory usage..."
	$(PYTHON) -m memory_profiler -m supreme_system_v5.benchmark

# === MAINTENANCE ===
security: ## Run security checks
	@echo "🔒 Running security checks..."
	bandit -r $(SOURCE_DIR)
	$(CARGO) audit
	@echo "✅ Security checks completed"

# === DOCUMENTATION ===
docs: ## Build documentation
	@echo "📚 Building documentation..."
	$(CARGO) doc --no-deps --features "python,simd,parallel"
	sphinx-build -b html docs/ docs/_build/html/
	@echo "✅ Documentation built"

serve-docs: docs ## Serve documentation locally
	@echo "🌍 Serving documentation at http://localhost:8080"
	$(PYTHON) -m http.server 8080 -d docs/_build/html/

# === DEFAULT TARGET ===
.DEFAULT_GOAL := help

# === SPECIAL TARGETS ===
full-rebuild: clean install build-rust test ## Complete rebuild from scratch
	@echo "🚀 Full rebuild completed successfully!"
	@echo "System ready for production deployment."

production-build: ## Build for production deployment
	@echo "🏭 Building for production..."
	make clean
	make install-rust
	RUSTFLAGS="-C opt-level=3 -C target-cpu=native" $(MATURIN) build --release --features "python,simd,parallel"
	$(PYTHON) -m pip install dist/*.whl --force-reinstall
	make validate
	@echo "✅ Production build completed"

demonstration: build-rust-dev ## Run system demonstration
	@echo "🎤 Running Supreme System V5 demonstration..."
	$(PYTHON) -m supreme_system_v5.demo
	@echo "✅ Demonstration completed"
