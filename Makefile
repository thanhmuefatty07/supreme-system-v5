.PHONY: help install test lint format clean build docs

help:
	@echo "Available commands:"
	@echo "  install     Install dependencies"
	@echo "  test        Run tests with coverage"
	@echo "  lint        Run linting (flake8, mypy)"
	@echo "  format      Format code with black"
	@echo "  clean       Clean build artifacts"
	@echo "  build       Build package"
	@echo "  docs        Generate documentation"

install:
	pip install -r requirements.txt
	pip install -e .
	pip install pytest pytest-cov black flake8 mypy pre-commit

install-dev: install
	pip install -r requirements-dev.txt
	pre-commit install

test:
	pytest tests/ -v --cov=src --cov-report=term-missing

test-quick:
	python test_basic_functionality.py

lint:
	flake8 src tests
	mypy src --ignore-missing-imports

format:
	black src tests

format-check:
	black --check --diff src tests

clean:
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} +
	rm -rf .coverage htmlcov .pytest_cache .mypy_cache
	rm -rf build/ dist/

build:
	python setup.py sdist bdist_wheel

docs:
	@echo "Documentation generation not yet implemented"

ci: lint format-check test

pre-commit:
	pre-commit run --all-files

# Development workflow
dev-setup: install-dev pre-commit

dev-test: format lint test

# Quick development cycle
dev: format test-quick
