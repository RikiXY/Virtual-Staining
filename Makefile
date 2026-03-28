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
PREPARE_LANG ?= en
IMAGE_SIZE ?= 512 512
GRID_MOVEMENT ?= 512 512
MARGIN ?= 200

# Derived paths
DATASET_ROOT      := local_workspace/datasets/$(DATASET)
RESULTS_PATH      := local_workspace/results
RUN_PATH          := $(RESULTS_PATH)/$(RUN_NAME)
CHECKPOINT_DIR    := $(RUN_PATH)/checkpoints
DATASET_TEST_PATH := $(DATASET_ROOT)/dataset_test
OUTPUT_TEST_PATH  := $(RUN_PATH)/output_test

# Auto-pick the highest checkpoint if not provided explicitly
AUTO_CHECKPOINT := $(lastword $(sort $(wildcard $(CHECKPOINT_DIR)/ep*.pth)))
CHECKPOINT_RESOLVED := $(if $(CHECKPOINT),$(CHECKPOINT),$(AUTO_CHECKPOINT))
SAVE_MASKS_FLAG := $(if $(filter 1 true yes,$(SAVE_MASKS)),--save-masks,)

.DEFAULT_GOAL := help

help:
	@printf "\nCommon targets:\n"
	@printf "  make prepare-dataset                  Build dataset_train/val/test from source+target images\n"
	@printf "  make train                            Train Pix2Pix\n"
	@printf "  make test                             Run test inference\n"
	@printf "  make evaluate                         Evaluate generated outputs\n"
	@printf "  make run-all                          Run train, test, and evaluate sequentially\n"
	@printf "  make sync                             Sync project dependencies\n"
	@printf "  make lock                             Refresh uv.lock\n"
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
	@printf "  PREPARE_LANG={en,it}                  Messages language for prepare-dataset (default: $(PREPARE_LANG))\n"
	@printf "  IMAGE_SIZE=\"W H\"                      Patch size for prepare-dataset (default: $(IMAGE_SIZE))\n"
	@printf "  GRID_MOVEMENT=\"X Y\"                   Grid movement for prepare-dataset (default: $(GRID_MOVEMENT))\n"
	@printf "  MARGIN=<n>                            Crop margin for prepare-dataset (default: $(MARGIN))\n"
	@printf "  EPOCHS=<n>                            Training epochs (default: $(EPOCHS))\n"
	@printf "  SEED=<n>                              Training seed (default: $(SEED))\n"
	@printf "  L1=<n>                                L1 loss weight for training (default: $(L1))\n"
	@printf "  CHECKPOINT=<path>                     Optional explicit checkpoint for test\n"
	@printf "\nExamples:\n"
	@printf "  cp .env.make.example .env.make\n"
	@printf "  make prepare-dataset SOURCE_NAME=source.tif TARGET_NAME=target.tif\n"
	@printf "  make train\n"
	@printf "  make test\n"
	@printf "  make evaluate\n"
	@printf "  make run-all\n"
	@printf "  make train DATASET=inv_1024 RUN_NAME=inv_P-1024_L1-50\n"
	@printf "  make train DATASET=inv_512 RUN_NAME=inv_debug EPOCHS=10 SEED=123 L1=37\n"
	@printf "  make test DATASET=inv_512 RUN_NAME=inv_P-512_L1-37 CHECKPOINT=local_workspace/results/inv_P-512_L1-37/checkpoints/ep042.pth\n"
	@printf "\n"

require-config:
	@test -n "$(DATASET)" || (echo "DATASET is empty. Set it once, e.g. export DATASET=inv_512"; exit 1)
	@test -n "$(RUN_NAME)" || (echo "RUN_NAME is empty. Set it once, e.g. export RUN_NAME=inv_P-512_L1-37"; exit 1)

require-dataset:
	@test -n "$(DATASET)" || (echo "DATASET is empty. Set it once, e.g. export DATASET=inv_512"; exit 1)

sync:
	$(UV) sync

lock:
	$(UV) lock

prepare-dataset: require-dataset
	$(PYTHON) src/prepare_dataset.py --path $(DATASET_ROOT) --source-name $(SOURCE_NAME) --target-name $(TARGET_NAME) --seed $(SEED) --lang $(PREPARE_LANG) --image-size $(IMAGE_SIZE) --grid-movement $(GRID_MOVEMENT) --margin $(MARGIN) $(SAVE_MASKS_FLAG)

train: require-config
	$(PYTHON) src/pix2pix.py train --dataset-root $(DATASET_ROOT)/ --run-name $(RUN_NAME) --results-path $(RESULTS_PATH) --epochs $(EPOCHS) --seed $(SEED) --l1-lambda $(L1)

test: require-config
	@test -n "$(CHECKPOINT_RESOLVED)" || (echo "No checkpoint found in $(CHECKPOINT_DIR). Set CHECKPOINT=/path/to/file.pth if needed."; exit 1)
	$(PYTHON) src/pix2pix.py test --dataset-root $(DATASET_ROOT)/ --run-path $(RUN_PATH) --checkpoint $(CHECKPOINT_RESOLVED)

evaluate: require-config
	$(PYTHON) tools/evaluate_generation.py dataset $(DATASET_TEST_PATH) $(OUTPUT_TEST_PATH) --save-graphs

run-all:
	$(MAKE) train
	$(MAKE) test
	$(MAKE) evaluate

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
	rm -rf .ruff_cache .mypy_cache .pyright __pycache__
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
	find . -type f \( -name "*.pyc" -o -name "*.pyo" \) -delete
