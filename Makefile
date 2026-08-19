UV      ?= uv
RUFF    ?= $(UV) run --group dev ruff
PYRIGHT ?= $(UV) run --group dev pyright

.DEFAULT_GOAL := qa

.PHONY: sync format lint typecheck test qa clean

sync:
	$(UV) sync --frozen

format:
	$(RUFF) format .

lint:
	$(RUFF) check .

typecheck:
	$(PYRIGHT)

test:
	$(UV) run --group dev pytest -m "not slow"

qa:
	$(RUFF) check .
	$(RUFF) format --check .
	$(PYRIGHT)
	$(UV) run --group dev pytest -m "not slow"

clean:
	rm -rf .ruff_cache .mypy_cache .pyright __pycache__ .pytest_tmp
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
	find . -type f \( -name "*.pyc" -o -name "*.pyo" \) -delete
