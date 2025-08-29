.PHONY: install lint format test test-unit test-e2e diagrams clean help

help:  ## Show this help message
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

install:  ## Install the package and dev dependencies
	uv sync --extra dev

format:  ## Format code with black and isort
	uv run black .
	uv run isort .

lint:  ## Run all linters
	uv run black --check .
	uv run isort --check-only .
	uv run flake8 .

typecheck:  ## Run mypy type checking (optional)
	uv run mypy .

test:  ## Run core functionality tests
	uv run pytest tests/unit/test_research_planner.py tests/unit/test_research_executor.py tests/unit/test_synthesis_agent.py -v

test-unit:  ## Run unit tests only
	uv run pytest tests/unit -v

test-e2e:  ## Run e2e tests only
	uv run pytest tests/e2e -v

test-all:  ## Run all tests (may have some failures)
	uv run pytest tests/ -v

clean:  ## Clean up cache files
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} +
	find . -type d -name ".pytest_cache" -exec rm -rf {} +
	find . -type d -name ".mypy_cache" -exec rm -rf {} +

check: format lint test  ## Run format, lint, and test