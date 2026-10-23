.PHONY: install install-lanl install-all lint typecheck test demo report preflight clean

VENV := .venv
# uv through the Windows launcher where `uv` is often not on PATH, uv itself elsewhere.
ifeq ($(OS),Windows_NT)
UV ?= py -3.11 -m uv
else
UV ?= uv
endif
PY := $(UV) run --python $(VENV)

# `--locked` everywhere below is load-bearing: it fails instead of silently
# re-resolving if uv.lock and pyproject.toml disagree, which is what keeps
# `reports/demo/` reproducible across machines and time instead of only on
# the machine that happened to generate it. See the comment in
# .github/workflows/ci.yml for the failure this was protecting against.

# What CI installs: `make demo` and the full test suite, deep and graph models
# included (CPU-only torch). NOT enough for `dvc repro`, which needs DVC itself
# from the `tracking` extra: see install-lanl.
install:
	$(UV) sync --extra classical --extra deep --extra graph --extra dev --locked --python 3.11

# What the real-dataset run needs: the catalog including the autoencoders, plus DVC and MLflow.
install-lanl:
	$(UV) sync --extra classical --extra deep --extra tracking --extra dev --locked --python 3.11

install-all:
	$(UV) sync --extra all --locked --python 3.11

lint:
	$(PY) ruff check src tests scripts
	$(PY) ruff format --check src tests scripts

typecheck:
	$(PY) mypy src

test:
	$(PY) pytest

demo:
	$(PY) authbench demo

# Disk, memory and dependency budget for a full LANL run, before anything is
# downloaded. Prints what each stage will need and exits non-zero if this
# machine cannot hold it.
preflight:
	$(PY) authbench preflight

report: demo
	@echo "Tables and figures written under reports/ (see reports/tables, reports/figures)."

clean:
	rm -rf outputs multirun .pytest_cache .ruff_cache .mypy_cache htmlcov .coverage
