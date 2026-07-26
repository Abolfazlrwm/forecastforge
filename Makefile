.DEFAULT_GOAL := help
.PHONY: help install lint format typecheck test clean run-precommit

PYTHON ?= python3.12
PIP ?= $(PYTHON) -m pip

help: ## Show this help message.
	@echo "ForecastForge developer commands:"
	@echo ""
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-16s\033[0m %s\n", $$1, $$2}'

install: ## Install the package with development dependencies (editable).
	$(PIP) install --upgrade pip
	$(PIP) install -e ".[dev,docs]"
	pre-commit install

lint: ## Run Ruff lint checks (no changes applied).
	ruff check .

format: ## Auto-format code with Ruff (rewrites files).
	ruff format .
	ruff check . --fix

typecheck: ## Run MyPy static type checks in strict mode.
	mypy src

test: ## Run the pytest suite with coverage.
	pytest

clean: ## Remove caches, build artifacts, and coverage output.
	rm -rf .pytest_cache .mypy_cache .ruff_cache .coverage coverage.xml htmlcov
	rm -rf build dist *.egg-info
	find . -type d -name "__pycache__" -not -path "./.venv/*" -exec rm -rf {} +

run-precommit: ## Run all pre-commit hooks against the entire repository.
	pre-commit run --all-files
