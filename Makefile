.SUFFIXES:
.DELETE_ON_ERROR:

SHELL       := /usr/bin/env bash
.SHELLFLAGS := -eu -o pipefail -c

MAKEFLAGS += --warn-undefined-variables
MAKEFLAGS += --no-builtin-rules --no-builtin-variables
MAKEFLAGS += --output-sync=target

.DEFAULT_GOAL := help

# =============================================================================
### Help
# =============================================================================

.PHONY: help
help: ## Show this help
	@grep -E '^(###[ ].+|[a-zA-Z0-9_%/-]+:.*##[^#])' $(firstword $(MAKEFILE_LIST)) \
		| sed -E \
		-e 's|^### (.+)|\x1b[1;36m\1\x1b[0m|' \
		-e 's|^([a-zA-Z0-9_%/-]+):.*## (.+)|  \x1b[32m\1\x1b[0m:\2|' \
		| awk -F: '{ \
		if ($$0 !~ /:/) { printf "\n%s\n", $$0 } \
		else { printf "  %-20s %s\n", $$1, $$2 } \
		}'

# =============================================================================
# Environment
# =============================================================================

DOTENV := .env

ifneq ($(wildcard $(DOTENV)),)
	include $(DOTENV)
	export
endif

# =============================================================================
# Logging
# =============================================================================

BOLD   := \033[1m
CYAN   := \033[36m
GREEN  := \033[32m
YELLOW := \033[33m
RED    := \033[31m
RESET  := \033[0m

#open Suppress --warn-undefined-variables false positives for $(call) arguments
1 :=
2 :=
3 :=

define _log_raw
{ \
	_tag="[$(2)]"; \
	_msg="$(3)"; \
	if _c=$$(tput cols 2>/dev/null); then _cols=$$_c; else _cols=80; fi; \
	_max=$$(( _cols - $${#_tag} - 4 )); \
	if [ $${#_msg} -gt $$_max ] && [ $$_max -gt 0 ]; then \
	_msg="$${_msg:0:$$_max}..."; \
	fi; \
	printf "$(BOLD)$(1)%s$(RESET) %s\n" "$$_tag" "$$_msg" >&2; \
	}
endef

log_info = $(call _log_raw,$(CYAN),INFO,$(1))
log_ok   = $(call _log_raw,$(GREEN),DONE,$(1))
log_warn = $(call _log_raw,$(YELLOW),WARN,$(1))
log_err  = $(call _log_raw,$(RED),FAIL,$(1))

# =============================================================================
# File Helpers
# =============================================================================

# add line if absent, appending at the end -- dedups without a global sort,
# so an existing structured file keeps its order. Useful for gitignore and a-like.
define add_line
grep --quiet --line-regexp --fixed-strings -- $(1) $(2) 2>/dev/null || echo $(1) >> $(2)
endef

# del line if present, preserving the file order (no global sort),
# useful for gitignore and a-like
define del_line
if [[ -e $(2) ]]; then sed --in-place '\,\b$(1)\b,d' $(2); fi
endef

# =============================================================================
# Configuration
# =============================================================================

PYTHON_VERSION_FILE := .python-version
VENV                := .venv
VENV_STAMP          := $(VENV)/pyvenv.cfg
GITIGNORE           := .gitignore

PYMANAGER := uv
PYVENV    := $(PYMANAGER) venv
PYSYNC    := $(PYMANAGER) sync
PYINSTALL := $(PYMANAGER) pip install
PYRUN     := $(PYMANAGER) run python
PYTEST    := $(PYMANAGER) run pytest
PYMYPY    := $(PYMANAGER) run mypy
PYIMPORTS := $(PYMANAGER) run lint-imports
PYBUILD   := $(PYMANAGER) build
PYAPP     := $(PYMANAGER) run gravityml

# =============================================================================
### Virtual Environment
# =============================================================================

.PHONY: venv
venv: $(VENV_STAMP) ## Build local .venv with uv from .python-version

$(VENV_STAMP): $(PYTHON_VERSION_FILE)
	@if ! command -v $(PYMANAGER) >/dev/null 2>&1; then \
		$(call log_err,$(PYMANAGER) not found on PATH); \
		exit 1; \
	fi
	@$(call log_info,Creating $(VENV) with Python $$(cat $(PYTHON_VERSION_FILE)))
	@$(PYVENV) --python "$$(cat $(PYTHON_VERSION_FILE))" $(VENV)
	@$(call add_line,$(VENV),$(GITIGNORE))
	@$(call log_ok,$(VENV) ready)

.PHONY: sync
sync: venv ## Sync dependencies into .venv from the lockfile
	@$(call log_info,Syncing dependencies with $(PYSYNC))
	@$(PYSYNC)
	@$(call log_ok,dependencies synced)

.PHONY: install
install: venv ## Install the projeck into .venv
	@$(call log_info,Installing project with $(PYINSTALL))
	@$(PYINSTALL) -e .
	@$(call log_ok,project installed)

# =============================================================================
### Quality
# =============================================================================

.PHONY: test
test: ## Run the full test suite (doctests + coverage)
	@$(call log_info,Running tests with coverage)
	@$(PYTEST) --cov=gravityml --cov-report=term-missing
	@$(call log_ok,tests passed)

.PHONY: test-quick
test-quick: ## Run the test suite quietly (per TDD cycle)
	@$(PYTEST) -q

.PHONY: typecheck
typecheck: ## Static type-check src with mypy
	@$(call log_info,Type-checking with mypy)
	@$(PYMYPY)
	@$(call log_ok,types clean)

.PHONY: lint-imports
lint-imports: ## Check onion-ring import contracts with import-linter
	@$(call log_info,Checking import contracts)
	@$(PYIMPORTS)
	@$(call log_ok,imports clean)

.PHONY: lint
lint: lint-imports ## Lint, format-check (ruff) and import contracts
	@$(call log_info,Linting with ruff)
	@$(PYMANAGER) run ruff check src tests
	@$(PYMANAGER) run ruff format --check src tests
	@$(call log_ok,lint clean)

.PHONY: format
format: ## Auto-format and fix lint with ruff (src + tests)
	@$(call log_info,Formatting with ruff)
	@$(PYMANAGER) run ruff format src tests
	@$(PYMANAGER) run ruff check --fix src tests
	@$(call log_ok,formatted)

# =============================================================================
### Run
# =============================================================================

.PHONY: run
run: ## Start the gravityml package (uv run gravityml) and show its help menu
	@$(PYAPP) --help

# =============================================================================
### Build
# =============================================================================

.PHONY: build
build: test | dist ## Build the sdist + wheel into dist/ (tests must pass first)
	@$(call log_info,Building sdist + wheel with $(PYBUILD))
	@$(PYBUILD)
	@$(call log_ok,artifacts in dist/)

# Order-only prerequisite of build: ensure the output dir exists and is ignored.
dist:
	@mkdir -p $@
	@$(call add_line,$@,$(GITIGNORE))
