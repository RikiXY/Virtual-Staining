UV      ?= uv
PYTHON  ?= $(UV) run python
RUFF    ?= ruff
PYRIGHT ?= pyright

ENV_FILE ?= .env.make

ifneq ("$(wildcard $(ENV_FILE))","")
include $(ENV_FILE)
endif

# Config files loaded by default by the Python entrypoints
PREPROCESSING_CONFIG ?= config/preprocessing.yaml
TRAIN_CONFIG ?= config/train.yaml

# Optional env overrides
DATASET ?=
DATASET_ROOT ?= $(if $(DATASET),local_workspace/datasets/$(DATASET),)
RESULTS_PATH ?=
RUN_NAME ?=
EPOCHS ?=
SEED ?=
L1_WEIGHT ?=
CHECKPOINT ?=
SOURCE_NAME ?=
TARGET_NAME ?=
SAVE_MASKS ?=
IMAGE_SIZE ?=
GRID_MOVEMENT ?=
MARGIN ?=
BATCH_SIZE ?=
NUM_WORKERS ?=
LR_G ?=
LR_D ?=
BETA1 ?=
BETA2 ?=
CHECKPOINT_RATE ?=
VALIDATE_RATE ?=
LOG_RATE ?=
RESUME ?=

# Compare args
RUN_NAME_A ?=
RUN_NAME_B ?=
LABEL_A ?= $(RUN_NAME_A)
LABEL_B ?= $(RUN_NAME_B)
COMPARE_COLUMN ?= ssim
COMPARE_DIRECTION ?= higher-is-better

# Derived paths
RESULTS_ROOT      := $(if $(RESULTS_PATH),$(RESULTS_PATH),local_workspace/results)
RUN_PATH          := $(RESULTS_ROOT)/$(RUN_NAME)
CHECKPOINT_DIR    := $(RUN_PATH)/checkpoints
DATASET_TEST_PATH := $(DATASET_ROOT)/dataset_test
OUTPUT_TEST_PATH  := $(RUN_PATH)/output_test
COMPARE_OUTPUT    ?= $(RESULTS_ROOT)/comparisons/$(RUN_NAME_A)_vs_$(RUN_NAME_B)/$(COMPARE_COLUMN)

# Auto-pick the highest checkpoint if not provided explicitly
AUTO_CHECKPOINT := $(lastword $(sort $(wildcard $(CHECKPOINT_DIR)/ep*.pth)))
CHECKPOINT_RESOLVED := $(if $(CHECKPOINT),$(CHECKPOINT),$(AUTO_CHECKPOINT))

arg = $(if $($(1)),$(2) $($(1)))
flag = $(if $(filter 1 true yes,$($(1))),$(2))

PREPARE_CONFIG_ARGS = \
	$(call arg,DATASET_ROOT,--path) \
	$(call arg,SOURCE_NAME,--source-name) \
	$(call arg,TARGET_NAME,--target-name) \
	$(call arg,SEED,--seed) \
	$(call arg,IMAGE_SIZE,--image-size) \
	$(call arg,GRID_MOVEMENT,--grid-movement) \
	$(call arg,MARGIN,--margin) \
	$(call flag,SAVE_MASKS,--save-masks) \
	$(call arg,MIN_FOREGROUND_RATIO,--min-foreground-ratio) \
	$(call arg,MAX_WHITE_RATIO,--max-white-ratio) \
	$(call arg,WHITE_THRESHOLD,--white-threshold) \
	$(call arg,MAX_LARGEST_WHITE_COMPONENT_RATIO,--max-largest-white-component-ratio)

TRAIN_CONFIG_ARGS = \
	$(call arg,DATASET_ROOT,--dataset-root) \
	$(call arg,RESULTS_PATH,--results-path) \
	$(call arg,RUN_NAME,--run-name) \
	$(call arg,EPOCHS,--epochs) \
	$(call arg,SEED,--seed) \
	$(call arg,IMAGE_SIZE,--image-size) \
	$(call arg,BATCH_SIZE,--batch-size) \
	$(call arg,NUM_WORKERS,--num-workers) \
	$(call arg,LR_G,--lr-g) \
	$(call arg,LR_D,--lr-d) \
	$(call arg,BETA1,--beta1) \
	$(call arg,BETA2,--beta2) \
	$(call arg,L1_WEIGHT,--l1-weight) \
	$(call arg,CHECKPOINT_RATE,--checkpoint-rate) \
	$(call arg,VALIDATE_RATE,--validate-rate) \
	$(call arg,LOG_RATE,--log-rate) \
	$(call arg,RESUME,--resume)

.DEFAULT_GOAL := help

help:
	@printf "\nTargets:\n"
	@printf "  %-24s %s\n" "prepare-dataset" "Build dataset_train/val/test"
	@printf "  %-24s %s\n" "train" "Train Pix2Pix"
	@printf "  %-24s %s\n" "infer" "Run inference with a checkpoint"
	@printf "  %-24s %s\n" "evaluate" "Evaluate generated outputs"
	@printf "  %-24s %s\n" "compare-unpaired" "Compare independent metric distributions"
	@printf "  %-24s %s\n" "compare-paired" "Compare paired metric distributions"
	@printf "  %-24s %s\n" "run-all" "Run train, infer, evaluate"
	@printf "  %-24s %s\n" "test" "Run pytest"
	@printf "  %-24s %s\n" "check" "Run lint, format-check, check-types"
	@printf "  %-24s %s\n" "sync / lock / clean" "Dependency and cleanup helpers"
	@printf "\nConfig files:\n"
	@printf "  %-30s %s\n" "ENV_FILE" "$(ENV_FILE)"
	@printf "  %-30s %s\n" "PREPROCESSING_CONFIG" "$(PREPROCESSING_CONFIG)"
	@printf "  %-30s %s\n" "TRAIN_CONFIG" "$(TRAIN_CONFIG)"
	@printf "\nDataset/run overrides:\n"
	@printf "  %-30s %s\n" "DATASET" "Derives DATASET_ROOT=local_workspace/datasets/<name>"
	@printf "  %-30s %s\n" "DATASET_ROOT" "Passed as --path / --dataset-root"
	@printf "  %-30s %s\n" "RESULTS_PATH" "Passed as --results-path"
	@printf "  %-30s %s\n" "RUN_NAME" "Passed as --run-name"
	@printf "\nPreprocessing overrides:\n"
	@printf "  %-30s %s\n" "SOURCE_NAME TARGET_NAME" "Input image filenames"
	@printf "  %-30s %s\n" "IMAGE_SIZE" "Patch/training size, e.g. \"512 512\""
	@printf "  %-30s %s\n" "GRID_MOVEMENT MARGIN" "Patch grid and crop margin"
	@printf "  %-30s %s\n" "SAVE_MASKS=1" "Enable --save-masks"
	@printf "\nTraining overrides:\n"
	@printf "  %-30s %s\n" "EPOCHS SEED BATCH_SIZE" "Basic run controls"
	@printf "  %-30s %s\n" "LR_G LR_D BETA1 BETA2" "Optimizer settings"
	@printf "  %-30s %s\n" "L1_WEIGHT" "Passed as --l1-weight"
	@printf "  %-30s %s\n" "NUM_WORKERS" "DataLoader workers"
	@printf "  %-30s %s\n" "CHECKPOINT_RATE VALIDATE_RATE" "Epoch intervals"
	@printf "  %-30s %s\n" "LOG_RATE RESUME" "Logging interval and resume checkpoint"
	@printf "\nExtra args:\n"
	@printf "  %-30s %s\n" "PREPARE_ARGS" "Appended after generated prepare args"
	@printf "  %-30s %s\n" "TRAIN_ARGS" "Appended after generated train args"
	@printf "  %-30s %s\n" "CHECKPOINT" "Explicit checkpoint for infer"
	@printf "\nComparison overrides:\n"
	@printf "  %-30s %s\n" "RUN_NAME_A RUN_NAME_B" "Runs to compare"
	@printf "  %-30s %s\n" "LABEL_A LABEL_B" "Display labels"
	@printf "  %-30s %s\n" "COMPARE_COLUMN" "$(COMPARE_COLUMN)"
	@printf "  %-30s %s\n" "COMPARE_DIRECTION" "$(COMPARE_DIRECTION)"
	@printf "  %-30s %s\n" "COMPARE_OUTPUT" "Output directory"
	@printf "\nExamples:\n"
	@printf "  cp .env.make.example .env.make\n"
	@printf "  make prepare-dataset DATASET=inv_512 SOURCE_NAME=source.tif TARGET_NAME=target.tif\n"
	@printf "  make train\n"
	@printf "  make infer\n"
	@printf "  make evaluate\n"
	@printf "  make run-all\n"
	@printf "  make test\n"
	@printf "  make train DATASET=inv_1024 RUN_NAME=inv_P-1024_L1-50\n"
	@printf "  make train DATASET=inv_512 RUN_NAME=inv_debug EPOCHS=10 SEED=123 L1_WEIGHT=37\n"
	@printf "  make infer DATASET=inv_512 RUN_NAME=inv_P-512_L1-37 CHECKPOINT=local_workspace/results/inv_P-512_L1-37/checkpoints/ep042.pth\n"
	@printf "  make compare-paired RUN_NAME_A=run_l1_25 RUN_NAME_B=run_l1_50 COMPARE_COLUMN=ssim\n"
	@printf "  make compare-unpaired RUN_NAME_A=run_a RUN_NAME_B=run_b COMPARE_COLUMN=mae COMPARE_DIRECTION=lower-is-better\n"
	@printf "\n"

require-config:
	@test -n "$(DATASET_ROOT)" || (echo "DATASET_ROOT is empty. Set DATASET=inv_512 or DATASET_ROOT=local_workspace/datasets/inv_512"; exit 1)
	@test -n "$(RUN_NAME)" || (echo "RUN_NAME is empty. Set it once, e.g. export RUN_NAME=inv_P-512_L1-37"; exit 1)

require-dataset:
	@test -n "$(DATASET_ROOT)" || (echo "DATASET_ROOT is empty. Set DATASET=inv_512 or DATASET_ROOT=local_workspace/datasets/inv_512"; exit 1)

require-compare:
	@test -n "$(RUN_NAME_A)" || (echo "RUN_NAME_A is empty. Set it, e.g. RUN_NAME_A=run_a"; exit 1)
	@test -n "$(RUN_NAME_B)" || (echo "RUN_NAME_B is empty. Set it, e.g. RUN_NAME_B=run_b"; exit 1)

sync:
	$(UV) sync --frozen

lock:
	$(UV) lock

prepare-dataset:
	$(PYTHON) src/prepare_dataset.py --config $(PREPROCESSING_CONFIG) $(PREPARE_CONFIG_ARGS) $(PREPARE_ARGS)

train:
	$(PYTHON) src/pix2pix.py train --config $(TRAIN_CONFIG) $(TRAIN_CONFIG_ARGS) $(TRAIN_ARGS)

infer: require-config
	@test -n "$(CHECKPOINT_RESOLVED)" || (echo "No checkpoint found in $(CHECKPOINT_DIR). Set CHECKPOINT=/path/to/file.pth if needed."; exit 1)
	$(PYTHON) src/pix2pix.py test --dataset-root $(DATASET_ROOT)/ --run-path $(RUN_PATH) --checkpoint $(CHECKPOINT_RESOLVED)

evaluate: require-config
	$(PYTHON) tools/evaluate_generation.py dataset --target-dir $(DATASET_TEST_PATH) --generated-dir $(OUTPUT_TEST_PATH) --save-graphs

compare-unpaired: require-compare
	$(PYTHON) tools/compare_distributions.py unpaired \
		--csv-a $(RESULTS_ROOT)/$(RUN_NAME_A)/evaluation \
		--csv-b $(RESULTS_ROOT)/$(RUN_NAME_B)/evaluation \
		--label-a $(LABEL_A) \
		--label-b $(LABEL_B) \
		--column $(COMPARE_COLUMN) \
		--$(COMPARE_DIRECTION) \
		--output-dir $(COMPARE_OUTPUT)

compare-paired: require-compare
	$(PYTHON) tools/compare_distributions.py paired \
		--csv-a $(RESULTS_ROOT)/$(RUN_NAME_A)/evaluation \
		--csv-b $(RESULTS_ROOT)/$(RUN_NAME_B)/evaluation \
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
