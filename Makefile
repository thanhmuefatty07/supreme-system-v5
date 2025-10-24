# Supreme System V5 - Production Makefile
# Automated development, testing, and deployment workflows

.PHONY: help install install-dev install-prod clean test test-fast test-integration lint format type-check security build docker-build docker-run validate benchmark docs serve-docs

# === HELP ===
help: ## Show this help message
	@echo "Supreme System V5 - Production Development Makefile"
	@echo "=================================================="
	@echo ""
	@echo "Available targets:"
	@awk 'BEGIN {FS = ":.*?## "} /^[a-zA-Z_-]+:.*?## / {printf "  \033[36m%-20s\033[0m %s\n", $$1, $$2}' $(MAKEFILE_LIST)
	@echo ""
	@echo "Hardware Profiles:"
	@echo "  i3-4GB:  make install-i3   # Lightweight for i3-8th gen + 4GB"
	@echo "  i5-8GB:  make install      # Standard for i5+ systems"
	@echo "  i7-16GB: make install-prod # Full features for production"

# === VARIABLES ===
PYTHON := python3
PIP := pip3
POETRY := poetry
SOURCE_DIR := src
TEST_DIR := tests
DOCS_DIR := docs

# Hardware detection
HARDWARE_TYPE := $(shell python3 -c "import psutil; mem_gb = psutil.virtual_memory().total / (1024**3); print('i3' if mem_gb <= 5 else 'i5' if mem_gb <= 10 else 'i7')" 2>/dev/null || echo 'unknown')
CPU_COUNT := $(shell python3 -c "import os; print(os.cpu_count())" 2>/dev/null || echo 4)

# === INSTALLATION ===
install: ## Install for standard systems (i5+ 8GB)
	@echo "📦 Installing Supreme System V5 - Standard Configuration"
	$(PIP) install -e .
	$(PIP) install -e ".[ai,data]"
	@echo "✅ Installation complete for $(HARDWARE_TYPE) system"

install-dev: ## Install with development dependencies
	@echo "🔧 Installing Supreme System V5 - Development Configuration"
	$(PIP) install -e ".[dev,ai,data,viz,docs]"
	pre-commit install
	@echo "✅ Development environment ready"

install-prod: ## Install with full production dependencies
	@echo "🏭 Installing Supreme System V5 - Production Configuration"
	$(PIP) install -e ".[all]"
	@echo "✅ Production installation complete"

install-i3: ## Lightweight installation for i3-8th gen + 4GB
	@echo "⚡ Installing Supreme System V5 - i3 Optimized (Lightweight)"
	$(PIP) install -e .
	@echo "✅ i3-optimized installation complete (memory conserved)"

install-poetry: ## Install using Poetry (recommended)
	@echo "🎵 Installing with Poetry"
	poetry install
	poetry run pre-commit install
	@echo "✅ Poetry installation complete"

# === CLEANING ===
clean: ## Clean build artifacts and cache
	@echo "🧹 Cleaning build artifacts"
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} +
	find . -type d -name ".pytest_cache" -exec rm -rf {} +
	find . -type d -name ".mypy_cache" -exec rm -rf {} +
	rm -rf build/ dist/ .coverage htmlcov/
	rm -rf logs/ *.log
	@echo "✅ Cleanup complete"

# === TESTING ===
test: ## Run full test suite
	@echo "🧪 Running Supreme System V5 test suite"
	pytest $(TEST_DIR) -v --cov=$(SOURCE_DIR) --cov-report=html --cov-report=term
	@echo "✅ All tests completed"

test-fast: ## Run fast unit tests only
	@echo "⚡ Running fast unit tests"
	pytest $(TEST_DIR) -v -m "not slow" --tb=short
	@echo "✅ Fast tests completed"

test-integration: ## Run integration tests
	@echo "🔗 Running integration tests"
	pytest $(TEST_DIR) -v -m "integration" --tb=short
	@echo "✅ Integration tests completed"

test-hardware: ## Test hardware-specific optimizations
	@echo "💻 Testing hardware optimizations ($(HARDWARE_TYPE) detected)"
	pytest $(TEST_DIR) -v -m "hardware" --hardware=$(HARDWARE_TYPE)
	@echo "✅ Hardware tests completed"

test-real-data: ## Test with real market data APIs (requires API keys)
	@echo "📊 Testing with real market data"
	pytest $(TEST_DIR) -v -m "real_data" --tb=short
	@echo "✅ Real data tests completed"

# === CODE QUALITY ===
lint: ## Run linting (ruff + flake8)
	@echo "🔍 Running linters"
	ruff check $(SOURCE_DIR) $(TEST_DIR)
	flake8 $(SOURCE_DIR) $(TEST_DIR) --max-line-length=88
	@echo "✅ Linting complete"

format: ## Format code (black + ruff)
	@echo "🎨 Formatting code"
	black $(SOURCE_DIR) $(TEST_DIR)
	ruff check --fix $(SOURCE_DIR) $(TEST_DIR)
	@echo "✅ Code formatting complete"

type-check: ## Run type checking (mypy)
	@echo "🔍 Type checking with mypy"
	mypy $(SOURCE_DIR)
	@echo "✅ Type checking complete"

security: ## Run security checks (bandit)
	@echo "🔒 Running security analysis"
	bandit -r $(SOURCE_DIR) -f json -o security-report.json
	bandit -r $(SOURCE_DIR)
	@echo "✅ Security analysis complete"

check: format lint type-check security ## Run all code quality checks
	@echo "✅ All quality checks completed"

# === VALIDATION ===
validate: ## Validate system for production deployment
	@echo "🔍 Running Supreme System V5 production validation"
	python scripts/validate_system.py
	@echo "✅ System validation complete"

benchmark: ## Run performance benchmarks
	@echo "🏁 Running performance benchmarks on $(HARDWARE_TYPE)"
	pytest $(TEST_DIR) -v --benchmark-only --benchmark-sort=mean
	@echo "✅ Benchmarks completed"

# === BUILDING ===
build: clean ## Build distribution packages
	@echo "🛠️ Building Supreme System V5 packages"
	python -m build
	@echo "✅ Build complete"

build-wheel: ## Build wheel package only
	@echo "🍡 Building wheel package"
	python -m build --wheel
	@echo "✅ Wheel build complete"

# === DOCKER ===
docker-build: ## Build Docker image
	@echo "🐳 Building Docker image"
	docker build -t supreme-system-v5:latest .
	@echo "✅ Docker image built"

docker-build-i3: ## Build optimized Docker image for i3 systems
	@echo "🐳 Building i3-optimized Docker image"
	docker build -f Dockerfile.i3 -t supreme-system-v5:i3-optimized .
	@echo "✅ i3-optimized Docker image built"

docker-run: ## Run Docker container
	@echo "🚀 Running Supreme System V5 in Docker"
	docker run -p 8000:8000 -p 9090:9090 --env-file .env supreme-system-v5:latest

docker-run-i3: ## Run i3-optimized Docker container
	@echo "🚀 Running i3-optimized Supreme System V5"
	docker run -p 8000:8000 --env-file .env supreme-system-v5:i3-optimized

docker-compose-dev: ## Start development environment with Docker Compose
	@echo "🚀 Starting development environment"
	docker-compose -f docker-compose.dev.yml up -d
	@echo "✅ Development environment started"

docker-compose-prod: ## Start production environment with Docker Compose
	@echo "🏭 Starting production environment"
	docker-compose -f docker-compose.yml -f docker-compose.prod.yml up -d
	@echo "✅ Production environment started"

# === DOCUMENTATION ===
docs: ## Build documentation
	@echo "📚 Building documentation"
	cd $(DOCS_DIR) && make html
	@echo "✅ Documentation built"

serve-docs: ## Serve documentation locally
	@echo "🌍 Serving documentation at http://localhost:8080"
	cd $(DOCS_DIR)/_build/html && python -m http.server 8080

# === DEVELOPMENT WORKFLOW ===
dev: install-dev ## Quick development setup
	@echo "🚀 Supreme System V5 development environment ready!"
	@echo "Next steps:"
	@echo "  1. Copy .env.example to .env and configure"
	@echo "  2. Run 'make validate' to check system"
	@echo "  3. Run 'make test-fast' for quick testing"
	@echo "  4. Run 'make run-dev' to start development server"

run-dev: ## Start development server
	@echo "💻 Starting Supreme System V5 development server"
	uvicorn supreme_system_v5.api.main:app --reload --host 0.0.0.0 --port 8000

run-prod: ## Start production server
	@echo "🏭 Starting Supreme System V5 production server"
	gunicorn supreme_system_v5.api.main:app -w $(CPU_COUNT) -k uvicorn.workers.UvicornWorker --bind 0.0.0.0:8000

# === CONTINUOUS INTEGRATION ===
ci: check test validate ## Run full CI pipeline
	@echo "✅ Continuous integration pipeline completed"

ci-fast: format lint test-fast ## Run fast CI checks
	@echo "✅ Fast CI pipeline completed"

# === DATABASE ===
db-migrate: ## Run database migrations
	@echo "📦 Running database migrations"
	alembic upgrade head
	@echo "✅ Database migrations completed"

db-reset: ## Reset database (WARNING: destroys data)
	@echo "⚠️ Resetting database (this will destroy all data)"
	@read -p "Are you sure? [y/N] " -n 1 -r; echo; if [[ $$REPLY =~ ^[Yy]$$ ]]; then alembic downgrade base && alembic upgrade head; fi

# === MONITORING ===
monitoring: ## Start monitoring stack
	@echo "📈 Starting monitoring stack"
	docker-compose -f docker-compose.monitoring.yml up -d
	@echo "✅ Monitoring stack started (Prometheus + Grafana)"

# === UTILITIES ===
env-example: ## Generate .env.example from current .env
	@echo "⚙️ Generating .env.example"
	sed 's/=.*/=/' .env > .env.example
	@echo "✅ .env.example generated"

requirements: ## Generate requirements.txt from pyproject.toml
	@echo "📦 Generating requirements.txt"
	pip-compile --resolver=backtracking
	@echo "✅ requirements.txt generated"

# === RELEASE ===
release-patch: ## Bump patch version and create release
	@echo "🏷️ Creating patch release"
	bump2version patch
	git push && git push --tags
	@echo "✅ Patch release created"

release-minor: ## Bump minor version and create release
	@echo "🏷️ Creating minor release"
	bump2version minor
	git push && git push --tags
	@echo "✅ Minor release created"

release-major: ## Bump major version and create release
	@echo "🏷️ Creating major release"
	bump2version major
	git push && git push --tags
	@echo "✅ Major release created"

# === DEFAULT TARGET ===
.DEFAULT_GOAL := help