UV      ?= uv
PYTHON  ?= $(UV) run python
RUFF    ?= $(UV) run --group dev ruff
PYRIGHT ?= $(UV) run --group dev pyright

.DEFAULT_GOAL := help

help:
	@printf "\nTargets:\n"
	@printf "  %-24s %s\n" "dataset" "Build splits/train|val|test from CONFIG"
	@printf "  %-24s %s\n" "train" "Train Pix2Pix from CONFIG"
	@printf "  %-24s %s\n" "infer" "Run inference via vs-infer CLI from CONFIG"
	@printf "  %-24s %s\n" "infer-images" "Run inference on INPUT_PATH file or directory"
	@printf "  %-24s %s\n" "evaluate" "Evaluate outputs via vs-evaluate CLI from CONFIG"
	@printf "  %-24s %s\n" "complete-run" "Run dataset, train, infer, evaluate from CONFIG"
	@printf "  %-24s %s\n" "run-queue" "Run a queue YAML from QUEUE"
	@printf "  %-24s %s\n" "compare" "Compare metric distributions from CONFIG"
	@printf "  %-24s %s\n" "render-panels" "Render diagnostic panels from CONFIG"
	@printf "  %-24s %s\n" "evaluate-single" "Evaluate one target/generated pair"
	@printf "  %-24s %s\n" "organize" "Export ranked sample files from CONFIG"
	@printf "  %-24s %s\n" "sync" "Sync uv dependencies from uv.lock"
	@printf "  %-24s %s\n" "format" "Format Python files with ruff"
	@printf "  %-24s %s\n" "lint" "Check Python files with ruff"
	@printf "  %-24s %s\n" "format-check" "Check formatting without applying changes"
	@printf "  %-24s %s\n" "check-types" "Run pyright type checker"
	@printf "  %-24s %s\n" "test" "Run pytest"
	@printf "  %-24s %s\n" "test-slow" "Run only slow pytest tests"
	@printf "  %-24s %s\n" "qa" "Run checks and tests"
	@printf "  %-24s %s\n" "clean" "Remove local caches"
	@printf "\nExperiment configuration policy:\n"
	@printf "  %-24s %s\n" "CONFIG" "Required for experiment and config-driven utility targets"
	@printf "  %-24s %s\n" "INPUT_PATH" "Required for infer-images"
	@printf "  %-24s %s\n" "OUTPUT_PATH" "Optional for infer-images"
	@printf "  %-24s %s\n" "MODE" "Optional image mode: auto|resize|tile"
	@printf "  %-24s %s\n" "TILE_OVERLAP" "Optional tile overlap"
	@printf "  %-24s %s\n" "OUTPUT_FORMAT" "Optional image output format"
	@printf "  %-24s %s\n" "TARGET_IMAGE" "Required for evaluate-single"
	@printf "  %-24s %s\n" "GENERATED_IMAGE" "Required for evaluate-single"
	@printf "  %-24s %s\n" "OUTPUT_DIR" "Optional for evaluate-single"
	@printf "  Put dataset paths, run names, image sizes, epochs, seeds,"
	@printf " checkpoints, and evaluation paths in YAML.\n"
	@printf "  Accepted patches are written under dataset_root/splits/<split>/.\n"
	@printf "\nExamples:\n"
	@printf "  make dataset CONFIG=config/runs/local/my_run.yaml\n"
	@printf "  make train CONFIG=config/runs/local/my_run.yaml\n"
	@printf "  make infer CONFIG=config/runs/local/my_run.yaml\n"
	@printf "  make infer-images CONFIG=config/runs/local/my_run.yaml INPUT_PATH=examples\n"
	@printf "  make evaluate CONFIG=config/runs/local/my_run.yaml\n"
	@printf "  make evaluate-single TARGET_IMAGE=..._target.tif GENERATED_IMAGE=..._target_generated.tif\n"
	@printf "  make complete-run CONFIG=config/runs/local/my_run.yaml\n"
	@printf "  make run-queue QUEUE=config/queues/example.yaml\n"
	@printf "\n"

require-config:
	@test -n "$(CONFIG)" || (echo "CONFIG is required, e.g. CONFIG=config/runs/example.yaml"; exit 1)
	@test -f "$(CONFIG)" || (echo "CONFIG file not found: $(CONFIG)"; exit 1)

require-input-path:
	@test -n "$(INPUT_PATH)" || (echo "INPUT_PATH is required, e.g. INPUT_PATH=examples"; exit 1)
	@test -e "$(INPUT_PATH)" || (echo "INPUT_PATH not found: $(INPUT_PATH)"; exit 1)

require-queue:
	@test -n "$(QUEUE)" || (echo "QUEUE is required, e.g. QUEUE=config/queues/example.yaml"; exit 1)
	@test -f "$(QUEUE)" || (echo "QUEUE file not found: $(QUEUE)"; exit 1)

require-target-image:
	@test -n "$(TARGET_IMAGE)" || (echo "TARGET_IMAGE is required, e.g. TARGET_IMAGE=local_workspace/datasets/my_run/splits/test/00512_09216_target.tif"; exit 1)
	@test -f "$(TARGET_IMAGE)" || (echo "TARGET_IMAGE file not found: $(TARGET_IMAGE)"; exit 1)

require-generated-image:
	@test -n "$(GENERATED_IMAGE)" || (echo "GENERATED_IMAGE is required, e.g. GENERATED_IMAGE=local_workspace/results/my_run/artifacts/output_test/00512_09216_target_generated.tif"; exit 1)
	@test -f "$(GENERATED_IMAGE)" || (echo "GENERATED_IMAGE file not found: $(GENERATED_IMAGE)"; exit 1)

sync:
	$(UV) sync --frozen

dataset: require-config
	$(UV) run vs-prepare --config $(CONFIG)

train: require-config
	$(UV) run vs-train --config $(CONFIG)

infer: require-config
	$(UV) run vs-infer --config $(CONFIG)

infer-images: require-config require-input-path
	$(UV) run vs-infer-images --config $(CONFIG) --input $(INPUT_PATH) $(if $(OUTPUT_PATH),--output $(OUTPUT_PATH),) $(if $(MODE),--mode $(MODE),) $(if $(TILE_OVERLAP),--tile-overlap $(TILE_OVERLAP),) $(if $(OUTPUT_FORMAT),--output-format $(OUTPUT_FORMAT),) $(if $(RECURSIVE),--recursive,)

evaluate: require-config
	$(UV) run vs-evaluate --config $(CONFIG)

complete-run: require-config
	$(UV) run vs-complete-run --config $(CONFIG)

run-queue: require-queue
	$(UV) run vs-run-queue --queue $(QUEUE)

compare: require-config
	$(UV) run vs-compare --config $(CONFIG)

render-panels: require-config
	$(UV) run vs-render-panels --config $(CONFIG)

evaluate-single: require-target-image require-generated-image
	$(UV) run vs-evaluate-single --target-image "$(TARGET_IMAGE)" --generated-image "$(GENERATED_IMAGE)" $(if $(OUTPUT_DIR),--output-dir "$(OUTPUT_DIR)",)

organize: require-config
	$(UV) run vs-organize --config $(CONFIG)

format:
	$(RUFF) format .

lint:
	$(RUFF) check .

format-check:
	$(RUFF) format --check .

check-types:
	$(PYRIGHT)

test:
	$(UV) run --group dev pytest -m "not slow"

test-slow:
	$(UV) run --group dev pytest -m "slow"

qa:
	$(RUFF) check .
	$(RUFF) format --check .
	$(PYRIGHT)
	$(UV) run --group dev pytest -m "not slow"

clean:
	rm -rf .ruff_cache .mypy_cache .pyright __pycache__ .pytest_tmp
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
	find . -type f \( -name "*.pyc" -o -name "*.pyo" \) -delete
