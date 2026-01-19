.PHONY: test test-cov test-fast lint format clean help install sync \
        prepare train predict visualize \
        deepjeb-prepare deepjeb-train deepjeb-train-mini deepjeb-predict deepjeb-visualize \
        hdwia-prepare hdwia-train hdwia-train-mini hdwia-predict hdwia-visualize

# Variables with defaults
TASK ?= deepjeb
CONFIG ?= train
MODEL ?=
OVERRIDES ?=
CKPT ?=
RESULTS ?=

# Default target
help:
	@echo "GNN Learning Project - Available Commands"
	@echo ""
	@echo "============================================"
	@echo "Data Preparation"
	@echo "============================================"
	@echo "  make prepare TASK=<task>"
	@echo ""
	@echo "  Examples:"
	@echo "    make prepare TASK=deepjeb"
	@echo "    make prepare TASK=hdwia"
	@echo ""
	@echo "============================================"
	@echo "Training"
	@echo "============================================"
	@echo "  make train TASK=<task> [CONFIG=<config>] [MODEL=<model>] [OVERRIDES='...']"
	@echo ""
	@echo "  Available Tasks: deepjeb, hdwia"
	@echo "  Available Models: gcn, gatv2"
	@echo ""
	@echo "  Examples:"
	@echo "    make train TASK=deepjeb"
	@echo "    make train TASK=deepjeb CONFIG=train-mini"
	@echo "    make train TASK=deepjeb MODEL=gcn"
	@echo "    make train TASK=hdwia OVERRIDES='training.max_epochs=50'"
	@echo ""
	@echo "============================================"
	@echo "Inference"
	@echo "============================================"
	@echo "  make predict TASK=<task> CKPT=<checkpoint_path>"
	@echo "  make visualize TASK=<task> RESULTS=<results_path>"
	@echo ""
	@echo "  Examples:"
	@echo "    make predict TASK=deepjeb CKPT=outputs/best.ckpt"
	@echo "    make visualize TASK=deepjeb RESULTS=outputs/results.pt"
	@echo ""
	@echo "============================================"
	@echo "Development"
	@echo "============================================"
	@echo "  make test              - Run all tests"
	@echo "  make test-cov          - Run tests with coverage"
	@echo "  make test-fast         - Run tests excluding slow tests"
	@echo "  make lint              - Lint code with autopep8"
	@echo "  make format            - Format code with autopep8"
	@echo ""
	@echo "============================================"
	@echo "Utilities"
	@echo "============================================"
	@echo "  make install           - Install dependencies"
	@echo "  make sync              - Sync dependencies"
	@echo "  make clean             - Clean build artifacts"
	@echo ""

# =============================================================================
# Core Commands (Parameterized)
# =============================================================================

prepare:
	@echo "Preparing $(TASK) dataset..."
	uv run python -m tasks.$(TASK) prepare \
		$(if $(MODEL),-o model=$(MODEL)) \
		$(if $(OVERRIDES),$(addprefix -o ,$(OVERRIDES)))

train:
	@echo "Training $(TASK) model..."
	uv run python -m tasks.$(TASK) train \
		$(if $(CONFIG),-c $(CONFIG)) \
		$(if $(MODEL),-o model=$(MODEL)) \
		$(if $(OVERRIDES),$(addprefix -o ,$(OVERRIDES)))

predict:
ifndef CKPT
	$(error CKPT is required. Usage: make predict TASK=<task> CKPT=path/to/checkpoint.ckpt)
endif
	@echo "Running $(TASK) prediction..."
	uv run python -m tasks.$(TASK) predict -m $(CKPT)

visualize:
ifndef RESULTS
	$(error RESULTS is required. Usage: make visualize TASK=<task> RESULTS=path/to/results.pt)
endif
	@echo "Generating $(TASK) visualizations..."
	uv run python -m tasks.$(TASK) visualize -r $(RESULTS)

# =============================================================================
# DeepJEB Task - Convenience Shortcuts
# =============================================================================

deepjeb-prepare:
	$(MAKE) prepare TASK=deepjeb

deepjeb-train:
	$(MAKE) train TASK=deepjeb

deepjeb-train-mini:
	$(MAKE) train TASK=deepjeb CONFIG=train-mini

deepjeb-predict:
ifndef CKPT
	$(error CKPT is required. Usage: make deepjeb-predict CKPT=path/to/checkpoint.ckpt)
endif
	$(MAKE) predict TASK=deepjeb CKPT=$(CKPT)

deepjeb-visualize:
ifndef RESULTS
	$(error RESULTS is required. Usage: make deepjeb-visualize RESULTS=path/to/results.pt)
endif
	$(MAKE) visualize TASK=deepjeb RESULTS=$(RESULTS)

# =============================================================================
# HDWIA Task - Convenience Shortcuts
# =============================================================================

hdwia-prepare:
	$(MAKE) prepare TASK=hdwia

hdwia-train:
	$(MAKE) train TASK=hdwia

hdwia-train-mini:
	$(MAKE) train TASK=hdwia CONFIG=train-mini

hdwia-predict:
ifndef CKPT
	$(error CKPT is required. Usage: make hdwia-predict CKPT=path/to/checkpoint.ckpt)
endif
	$(MAKE) predict TASK=hdwia CKPT=$(CKPT)

hdwia-visualize:
ifndef RESULTS
	$(error RESULTS is required. Usage: make hdwia-visualize RESULTS=path/to/results.pt)
endif
	$(MAKE) visualize TASK=hdwia RESULTS=$(RESULTS)

# =============================================================================
# Testing
# =============================================================================

test:
	@echo "Running all tests..."
	uv run pytest

test-cov:
	@echo "Running tests with coverage..."
	uv run pytest --cov=. --cov-report=html --cov-report=term-missing
	@echo ""
	@echo "Coverage report generated: htmlcov/index.html"

test-fast:
	@echo "Running tests excluding slow tests..."
	uv run pytest -m "not slow"

# =============================================================================
# Linting & Formatting
# =============================================================================

lint:
	@echo "Running linter..."
	uv run autopep8 --diff --recursive --aggressive .

format:
	@echo "Formatting code..."
	uv run autopep8 --in-place --recursive --aggressive .

# =============================================================================
# Utilities
# =============================================================================

clean:
	@echo "Cleaning cache files..."
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name .pytest_cache -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name .ruff_cache -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
	find . -type d -name htmlcov -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name ".coverage" -delete
	@echo "Clean complete"

sync:
	@echo "Syncing dependencies..."
	uv sync

install:
	@echo "Installing project..."
	uv sync
	@echo "Installation complete"
