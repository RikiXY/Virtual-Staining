UV      ?= uv
PYTHON  ?= $(UV) run python
RUFF    ?= ruff
PYRIGHT ?= pyright

ENV_FILE ?= .env.make

ifneq ("$(wildcard $(ENV_FILE))","")
include $(ENV_FILE)
endif

# Optional args
EPOCHS ?= 100
SEED ?= 42
L1 ?= 25
CHECKPOINT ?=
SOURCE_NAME ?= source.tif
TARGET_NAME ?= target.tif
SAVE_MASKS ?= 0
IMAGE_SIZE ?= 512 512
GRID_MOVEMENT ?= 512 512
MARGIN ?= 200

# Compare args
RUN_NAME_A ?=
RUN_NAME_B ?=
LABEL_A ?= $(RUN_NAME_A)
LABEL_B ?= $(RUN_NAME_B)
COMPARE_COLUMN ?= ssim
COMPARE_DIRECTION ?= higher-is-better

# Derived paths
DATASET_ROOT      := local_workspace/datasets/$(DATASET)
RESULTS_PATH      := local_workspace/results
RUN_PATH          := $(RESULTS_PATH)/$(RUN_NAME)
CHECKPOINT_DIR    := $(RUN_PATH)/checkpoints
DATASET_TEST_PATH := $(DATASET_ROOT)/dataset_test
OUTPUT_TEST_PATH  := $(RUN_PATH)/output_test
COMPARE_OUTPUT    ?= $(RESULTS_PATH)/comparisons/$(RUN_NAME_A)_vs_$(RUN_NAME_B)/$(COMPARE_COLUMN)

# Auto-pick the highest checkpoint if not provided explicitly
AUTO_CHECKPOINT := $(lastword $(sort $(wildcard $(CHECKPOINT_DIR)/ep*.pth)))
CHECKPOINT_RESOLVED := $(if $(CHECKPOINT),$(CHECKPOINT),$(AUTO_CHECKPOINT))
SAVE_MASKS_FLAG := $(if $(filter 1 true yes,$(SAVE_MASKS)),--save-masks,)

.DEFAULT_GOAL := help

help:
	@printf "\nCommon targets:\n"
	@printf "  make prepare-dataset                  Build dataset_train/val/test from source+target images\n"
	@printf "  make train                            Train Pix2Pix\n"
	@printf "  make infer                            Run inference with a trained checkpoint\n"
	@printf "  make evaluate                         Evaluate generated outputs\n"
	@printf "  make compare-unpaired                 Compare two independent metric distributions\n"
	@printf "  make compare-paired                   Compare two runs on the same test samples\n"
	@printf "  make run-all                          Run train, infer, and evaluate sequentially\n"
	@printf "  make test                             Run the unit test suite (pytest)\n"
	@printf "  make sync                             Install exact versions from uv.lock (reproducible)\n"
	@printf "  make lock                             Re-resolve dependencies and update uv.lock\n"
	@printf "  make lint                             Run Ruff lints\n"
	@printf "  make format                           Format code with Ruff\n"
	@printf "  make format-check                     Check formatting without changing files\n"
	@printf "  make check-types                      Run Pyright\n"
	@printf "  make check                            Run lint + format-check + check-types\n"
	@printf "  make clean                            Remove common caches\n"
	@printf "\nConfig:\n"
	@printf "  ENV_FILE=.env.make                    Optional Makefile env file (default: $(ENV_FILE))\n"
	@printf "  DATASET=inv_512                       Dataset name (can come from $(ENV_FILE))\n"
	@printf "  RUN_NAME=inv_P-512_L1-37              Run name (can come from $(ENV_FILE))\n"
	@printf "\nOptional:\n"
	@printf "  SOURCE_NAME=<file>                    Source image filename (default: $(SOURCE_NAME))\n"
	@printf "  TARGET_NAME=<file>                    Target image filename (default: $(TARGET_NAME))\n"
	@printf "  SAVE_MASKS=1                          Save patch-level masks in subimages/\n"
	@printf "  IMAGE_SIZE=\"W H\"                      Patch size for prepare-dataset (default: $(IMAGE_SIZE))\n"
	@printf "  GRID_MOVEMENT=\"X Y\"                   Grid movement for prepare-dataset (default: $(GRID_MOVEMENT))\n"
	@printf "  MARGIN=<n>                            Crop margin for prepare-dataset (default: $(MARGIN))\n"
	@printf "  EPOCHS=<n>                            Training epochs (default: $(EPOCHS))\n"
	@printf "  SEED=<n>                              Training seed (default: $(SEED))\n"
	@printf "  L1=<n>                                L1 loss weight for training (default: $(L1))\n"
	@printf "  CHECKPOINT=<path>                     Optional explicit checkpoint for infer\n"
	@printf "  RUN_NAME_A=<name>                     First run name for compare targets\n"
	@printf "  RUN_NAME_B=<name>                     Second run name for compare targets\n"
	@printf "  LABEL_A=<label>                       Display label for run A (default: RUN_NAME_A)\n"
	@printf "  LABEL_B=<label>                       Display label for run B (default: RUN_NAME_B)\n"
	@printf "  COMPARE_COLUMN=<col>                  Metric column to compare (default: $(COMPARE_COLUMN))\n"
	@printf "  COMPARE_DIRECTION=<dir>               higher-is-better or lower-is-better (default: $(COMPARE_DIRECTION))\n"
	@printf "  COMPARE_OUTPUT=<path>                 Output directory for compare results\n"
	@printf "\nExamples:\n"
	@printf "  cp .env.make.example .env.make\n"
	@printf "  make prepare-dataset SOURCE_NAME=source.tif TARGET_NAME=target.tif\n"
	@printf "  make train\n"
	@printf "  make infer\n"
	@printf "  make evaluate\n"
	@printf "  make run-all\n"
	@printf "  make test\n"
	@printf "  make train DATASET=inv_1024 RUN_NAME=inv_P-1024_L1-50\n"
	@printf "  make train DATASET=inv_512 RUN_NAME=inv_debug EPOCHS=10 SEED=123 L1=37\n"
	@printf "  make infer DATASET=inv_512 RUN_NAME=inv_P-512_L1-37 CHECKPOINT=local_workspace/results/inv_P-512_L1-37/checkpoints/ep042.pth\n"
	@printf "  make compare-paired RUN_NAME_A=run_l1_25 RUN_NAME_B=run_l1_50 COMPARE_COLUMN=ssim\n"
	@printf "  make compare-unpaired RUN_NAME_A=run_a RUN_NAME_B=run_b COMPARE_COLUMN=mae COMPARE_DIRECTION=lower-is-better\n"
	@printf "\n"

require-config:
	@test -n "$(DATASET)" || (echo "DATASET is empty. Set it once, e.g. export DATASET=inv_512"; exit 1)
	@test -n "$(RUN_NAME)" || (echo "RUN_NAME is empty. Set it once, e.g. export RUN_NAME=inv_P-512_L1-37"; exit 1)

require-dataset:
	@test -n "$(DATASET)" || (echo "DATASET is empty. Set it once, e.g. export DATASET=inv_512"; exit 1)

require-compare:
	@test -n "$(RUN_NAME_A)" || (echo "RUN_NAME_A is empty. Set it, e.g. RUN_NAME_A=run_a"; exit 1)
	@test -n "$(RUN_NAME_B)" || (echo "RUN_NAME_B is empty. Set it, e.g. RUN_NAME_B=run_b"; exit 1)

sync:
	$(UV) sync --frozen

lock:
	$(UV) lock

prepare-dataset: require-dataset
	$(PYTHON) src/prepare_dataset.py --path $(DATASET_ROOT) --source-name $(SOURCE_NAME) --target-name $(TARGET_NAME) --seed $(SEED) --image-size $(IMAGE_SIZE) --grid-movement $(GRID_MOVEMENT) --margin $(MARGIN) $(SAVE_MASKS_FLAG)

train: require-config
	$(PYTHON) src/pix2pix.py train --dataset-root $(DATASET_ROOT)/ --run-name $(RUN_NAME) --results-path $(RESULTS_PATH) --epochs $(EPOCHS) --seed $(SEED) --l1-weight $(L1)

infer: require-config
	@test -n "$(CHECKPOINT_RESOLVED)" || (echo "No checkpoint found in $(CHECKPOINT_DIR). Set CHECKPOINT=/path/to/file.pth if needed."; exit 1)
	$(PYTHON) src/pix2pix.py test --dataset-root $(DATASET_ROOT)/ --run-path $(RUN_PATH) --checkpoint $(CHECKPOINT_RESOLVED)

evaluate: require-config
	$(PYTHON) tools/evaluate_generation.py dataset --target-dir $(DATASET_TEST_PATH) --generated-dir $(OUTPUT_TEST_PATH) --save-graphs

compare-unpaired: require-compare
	$(PYTHON) tools/compare_distributions.py unpaired \
		--csv-a $(RESULTS_PATH)/$(RUN_NAME_A)/evaluation \
		--csv-b $(RESULTS_PATH)/$(RUN_NAME_B)/evaluation \
		--label-a $(LABEL_A) \
		--label-b $(LABEL_B) \
		--column $(COMPARE_COLUMN) \
		--$(COMPARE_DIRECTION) \
		--output-dir $(COMPARE_OUTPUT)

compare-paired: require-compare
	$(PYTHON) tools/compare_distributions.py paired \
		--csv-a $(RESULTS_PATH)/$(RUN_NAME_A)/evaluation \
		--csv-b $(RESULTS_PATH)/$(RUN_NAME_B)/evaluation \
		--label-a $(LABEL_A) \
		--label-b $(LABEL_B) \
		--column $(COMPARE_COLUMN) \
		--$(COMPARE_DIRECTION) \
		--output-dir $(COMPARE_OUTPUT)

run-all:
	$(MAKE) train
	$(MAKE) infer
	$(MAKE) evaluate

test:
	$(UV) run --group dev pytest

lint:
	$(RUFF) check .

format:
	$(RUFF) format .

format-check:
	$(RUFF) format --check .

check-types:
	$(PYRIGHT) .

check: lint format-check check-types

clean:
	rm -rf .ruff_cache .mypy_cache .pyright __pycache__ .pytest_tmp
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
	find . -type f \( -name "*.pyc" -o -name "*.pyo" \) -delete
