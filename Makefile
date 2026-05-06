UV      ?= uv
PYTHON  ?= $(UV) run python
RUFF    ?= ruff
PYRIGHT ?= $(UV) run pyright

CONFIG ?= config/runs/example.yaml

.DEFAULT_GOAL := help

help:
	@printf "\nTargets:\n"
	@printf "  %-24s %s\n" "prepare-dataset" "Build dataset_train/val/test from CONFIG"
	@printf "  %-24s %s\n" "train" "Train Pix2Pix from CONFIG"
	@printf "  %-24s %s\n" "infer" "Run inference from CONFIG"
	@printf "  %-24s %s\n" "evaluate" "Evaluate generated outputs from CONFIG"
	@printf "  %-24s %s\n" "complete-run" "Run train, infer, evaluate from CONFIG"
	@printf "  %-24s %s\n" "test" "Run pytest"
	@printf "  %-24s %s\n" "check" "Run lint, format-check, check-types"
	@printf "  %-24s %s\n" "qa" "Run check and test"
	@printf "  %-24s %s\n" "sync / lock / clean" "Dependency and cleanup helpers"
	@printf "\nExperiment configuration policy:\n"
	@printf "  %-24s %s\n" "CONFIG" "$(CONFIG)"
	@printf "  Put dataset paths, run names, image sizes, epochs, seeds,"
	@printf " checkpoints, and evaluation paths in YAML.\n"
	@printf "  Do not pass experiment parameters such as DATASET, RUN_NAME,"
	@printf " IMAGE_SIZE, EPOCHS, SEED, or CHECKPOINT to make.\n"
	@printf "\nExamples:\n"
	@printf "  make prepare-dataset CONFIG=config/runs/example.yaml\n"
	@printf "  make train CONFIG=config/runs/example.yaml\n"
	@printf "  make infer CONFIG=config/runs/example.yaml\n"
	@printf "  make evaluate CONFIG=config/runs/example.yaml\n"
	@printf "  make complete-run CONFIG=config/runs/example.yaml\n"
	@printf "  make test\n"
	@printf "\n"

require-config:
	@test -n "$(CONFIG)" || (echo "CONFIG is required, e.g. CONFIG=config/runs/example.yaml"; exit 1)
	@test -f "$(CONFIG)" || (echo "CONFIG file not found: $(CONFIG)"; exit 1)

sync:
	$(UV) sync --frozen

lock:
	$(UV) lock

prepare-dataset: require-config
	$(PYTHON) src/prepare_dataset.py --config $(CONFIG)

train: require-config
	$(PYTHON) src/pix2pix.py train --config $(CONFIG)

infer: require-config
	$(PYTHON) src/pix2pix.py test --config $(CONFIG)

evaluate: require-config
	$(PYTHON) tools/evaluate_generation.py dataset --config $(CONFIG)

complete-run: require-config
	$(MAKE) train CONFIG=$(CONFIG)
	$(MAKE) infer CONFIG=$(CONFIG)
	$(MAKE) evaluate CONFIG=$(CONFIG)

test:
	$(UV) run --group dev pytest

lint:
	$(RUFF) check .

format:
	$(RUFF) format .

format-check:
	$(RUFF) format --check .

check-types:
	$(PYRIGHT)

check: lint format-check check-types

qa: check test

clean:
	rm -rf .ruff_cache .mypy_cache .pyright __pycache__ .pytest_tmp
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
	find . -type f \( -name "*.pyc" -o -name "*.pyo" \) -delete
