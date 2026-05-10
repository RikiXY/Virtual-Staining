UV      ?= uv
PYTHON  ?= $(UV) run python
RUFF    ?= $(UV) run --group dev ruff
PYRIGHT ?= $(UV) run --group dev pyright

CONFIG ?= config/runs/example.yaml

.DEFAULT_GOAL := help

help:
	@printf "\nTargets:\n"
	@printf "  %-24s %s\n" "dataset" "Build dataset_train/val/test from CONFIG"
	@printf "  %-24s %s\n" "train" "Train Pix2Pix from CONFIG"
	@printf "  %-24s %s\n" "infer" "Run inference via vs-infer CLI from CONFIG"
	@printf "  %-24s %s\n" "evaluate" "Evaluate generated outputs from CONFIG"
	@printf "  %-24s %s\n" "complete-run" "Run dataset, train, infer, evaluate from CONFIG"
	@printf "  %-24s %s\n" "sync" "Sync uv dependencies from uv.lock"
	@printf "  %-24s %s\n" "format" "Format Python files with ruff"
	@printf "  %-24s %s\n" "lint" "Check Python files with ruff"
	@printf "  %-24s %s\n" "format-check" "Check formatting without applying changes"
	@printf "  %-24s %s\n" "check-types" "Run pyright type checker"
	@printf "  %-24s %s\n" "test" "Run pytest"
	@printf "  %-24s %s\n" "qa" "Run checks and tests"
	@printf "  %-24s %s\n" "clean" "Remove local caches"
	@printf "\nExperiment configuration policy:\n"
	@printf "  %-24s %s\n" "CONFIG" "$(CONFIG)"
	@printf "  Put dataset paths, run names, image sizes, epochs, seeds,"
	@printf " checkpoints, and evaluation paths in YAML.\n"
	@printf "\nExamples:\n"
	@printf "  make dataset CONFIG=config/runs/example.yaml\n"
	@printf "  make train CONFIG=config/runs/example.yaml\n"
	@printf "  make infer CONFIG=config/runs/example.yaml\n"
	@printf "  make evaluate CONFIG=config/runs/example.yaml\n"
	@printf "  make complete-run CONFIG=config/runs/example.yaml\n"
	@printf "\n"

require-config:
	@test -n "$(CONFIG)" || (echo "CONFIG is required, e.g. CONFIG=config/runs/example.yaml"; exit 1)
	@test -f "$(CONFIG)" || (echo "CONFIG file not found: $(CONFIG)"; exit 1)

sync:
	$(UV) sync --frozen

dataset: require-config
	$(PYTHON) src/prepare_dataset.py --config $(CONFIG)

train: require-config
	$(PYTHON) src/pix2pix.py train --config $(CONFIG)

infer: require-config
	$(PYTHON) -m virtual_staining.cli.infer --config $(CONFIG)

evaluate: require-config
	$(PYTHON) tools/evaluate_generation.py dataset --config $(CONFIG)

complete-run: require-config
	$(MAKE) dataset CONFIG=$(CONFIG)
	$(MAKE) train CONFIG=$(CONFIG)
	$(MAKE) infer CONFIG=$(CONFIG)
	$(MAKE) evaluate CONFIG=$(CONFIG)

format:
	$(RUFF) format .

lint:
	$(RUFF) check .

format-check:
	$(RUFF) format --check .

check-types:
	$(PYRIGHT)

test:
	$(UV) run --group dev pytest

qa:
	$(RUFF) check .
	$(RUFF) format --check .
	$(PYRIGHT)
	$(UV) run --group dev pytest

clean:
	rm -rf .ruff_cache .mypy_cache .pyright __pycache__ .pytest_tmp
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
	find . -type f \( -name "*.pyc" -o -name "*.pyo" \) -delete
