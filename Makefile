.PHONY: help install install-dev test lint format type-check clean pre-commit build

help: ## Show this help message
	@echo "Available commands:"
	@awk 'BEGIN {FS = ":.*##"} /^[a-zA-Z_-]+:.*##/ {printf "  \033[36m%-15s\033[0m %s\n", $$1, $$2}' $(MAKEFILE_LIST)

install: ## Install the package
	uv sync

install-dev: ## Install the package with development dependencies
	uv sync --all-extras --dev

update-deps: ## Update dependencies to their latest versions
	uv sync --upgrade

test: ## Run tests
	uv run pytest tests/ -v

test-cov: ## Run tests with coverage
	uv run pytest tests/ -v --cov=src/eir_auto_gp --cov-report=html --cov-report=term

lint: ## Run ruff linter
	uv run ruff check .

lint-fix: ## Run ruff linter with auto-fix
	uv run ruff check . --fix --unsafe-fixes

format: ## Format code with ruff
	uv run ruff format .

format-check: ## Check if code is formatted
	uv run ruff format --check .

type-check: ## Run ty type checking
	uvx ty check src/eir_auto_gp

pre-commit: ## Run pre-commit hooks on all files
	uv run pre-commit run --all-files

pre-commit-install: ## Install pre-commit hooks
	uv run pre-commit install

clean: ## Clean up build artifacts
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info/
	rm -rf .pytest_cache/
	rm -rf .coverage
	rm -rf htmlcov/
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete

build: ## Build the package
	uv build

check-all: lint format-check type-check test ## Run all checks (lint, format, type-check, test)

fix-and-check-all: lint-fix format type-check test ## Fix issues and run all checks (lint, format, type-check, test)

ci: check-all ## Run all CI checks locally