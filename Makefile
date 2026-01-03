# Makefile for langgraph-ollama-local
# Local agent building at scale using Ollama

.PHONY: help install install-dev install-all test test-quick test-cov lint format clean examples check pre-commit

# Default target
help:
	@echo "LangGraph Ollama Local - Development Commands"
	@echo ""
	@echo "Installation:"
	@echo "  make install        Install package in editable mode"
	@echo "  make install-dev    Install with development dependencies"
	@echo "  make install-all    Install with all optional dependencies"
	@echo ""
	@echo "Testing:"
	@echo "  make test           Run tests with coverage"
	@echo "  make test-quick     Run tests without coverage (faster)"
	@echo "  make test-cov       Run tests and generate HTML coverage report"
	@echo "  make test-int       Run integration tests (requires Ollama)"
	@echo ""
	@echo "Code Quality:"
	@echo "  make lint           Run linting (ruff + mypy)"
	@echo "  make format         Format code (ruff format)"
	@echo "  make check          Run all checks (lint + test)"
	@echo "  make pre-commit     Run pre-commit hooks on all files"
	@echo ""
	@echo "Examples:"
	@echo "  make examples       Run example notebooks"
	@echo "  make jupyter        Start Jupyter Lab"
	@echo ""
	@echo "Utilities:"
	@echo "  make clean          Remove build artifacts"
	@echo "  make ollama-check   Check Ollama connection"

# =============================================================================
# Installation
# =============================================================================

install:
	pip install -e .

install-dev:
	pip install -e ".[dev]"
	pre-commit install

install-all:
	pip install -e ".[all]"
	pre-commit install

# =============================================================================
# Testing
# =============================================================================

test:
	pytest tests/ -v --cov=langgraph_ollama_local --cov-report=term-missing

test-quick:
	pytest tests/ -v -x --tb=short

test-cov:
	pytest tests/ -v --cov=langgraph_ollama_local --cov-report=html
	@echo "Coverage report generated in htmlcov/index.html"

test-int:
	pytest tests/integration/ -v -m integration

# =============================================================================
# Code Quality
# =============================================================================

lint:
	ruff check langgraph_ollama_local/ tests/
	mypy langgraph_ollama_local/

format:
	ruff format langgraph_ollama_local/ tests/
	ruff check --fix langgraph_ollama_local/ tests/

check: lint test

pre-commit:
	pre-commit run --all-files

# =============================================================================
# Examples
# =============================================================================

examples:
	@echo "Running example notebooks..."
	@if [ -d "examples" ]; then \
		for nb in examples/*.ipynb; do \
			echo "Running $$nb..."; \
			jupyter nbconvert --to notebook --execute "$$nb" --inplace || true; \
		done; \
	else \
		echo "No examples directory found."; \
	fi

jupyter:
	jupyter lab --notebook-dir=examples

# =============================================================================
# Utilities
# =============================================================================

clean:
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info/
	rm -rf .pytest_cache/
	rm -rf .mypy_cache/
	rm -rf .ruff_cache/
	rm -rf htmlcov/
	rm -rf .coverage
	rm -rf .checkpoints/
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true

ollama-check:
	@echo "Checking Ollama connection..."
	@python -c "from langgraph_ollama_local import LocalAgentConfig; \
		config = LocalAgentConfig(); \
		print(f'Ollama URL: {config.ollama.base_url}'); \
		print(f'Model: {config.ollama.model}'); \
		import httpx; \
		r = httpx.get(f'{config.ollama.base_url}/api/tags', timeout=5); \
		print(f'Connection: OK ({len(r.json().get(\"models\", []))} models available)')" \
		2>/dev/null || echo "Connection: FAILED (is Ollama running?)"

# =============================================================================
# Development Helpers
# =============================================================================

# Create a new agent module from template
new-agent:
	@read -p "Agent name (e.g., 'my_agent'): " name; \
	echo "Creating agents/$$name.py..."; \
	mkdir -p langgraph_ollama_local/agents; \
	echo '"""' > langgraph_ollama_local/agents/$$name.py; \
	echo "$$name agent implementation." >> langgraph_ollama_local/agents/$$name.py; \
	echo '"""' >> langgraph_ollama_local/agents/$$name.py; \
	echo "" >> langgraph_ollama_local/agents/$$name.py; \
	echo "Created langgraph_ollama_local/agents/$$name.py"

# Show current configuration
show-config:
	@python -c "from langgraph_ollama_local import LocalAgentConfig; \
		import json; \
		config = LocalAgentConfig(); \
		print('Current Configuration:'); \
		print(f'  Ollama Host: {config.ollama.host}'); \
		print(f'  Ollama Port: {config.ollama.port}'); \
		print(f'  Ollama Model: {config.ollama.model}'); \
		print(f'  Ollama Timeout: {config.ollama.timeout}s'); \
		print(f'  LangGraph Recursion Limit: {config.langgraph.recursion_limit}'); \
		print(f'  Checkpoint Dir: {config.langgraph.checkpoint_dir}')"
