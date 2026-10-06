.PHONY: install install-lanl install-all lint typecheck test demo report preflight clean

VENV := .venv
PY := py -3.11 -m uv run --python $(VENV)

# `--locked` everywhere below is load-bearing: it fails instead of silently
# re-resolving if uv.lock and pyproject.toml disagree, which is what keeps
# `reports/demo/` reproducible across machines and time instead of only on
# the machine that happened to generate it. See the comment in
# .github/workflows/ci.yml for the failure this was protecting against.

# Enough for `make demo` and the test suite. NOT enough for `dvc repro`: that
# needs DVC itself, which lives in the `tracking` extra — see install-lanl.
install:
	py -3.11 -m uv sync --extra classical --extra dev --locked --python 3.11

# What the real-dataset run needs: the classical models, plus DVC and MLflow.
install-lanl:
	py -3.11 -m uv sync --extra classical --extra tracking --extra dev --locked --python 3.11

install-all:
	py -3.11 -m uv sync --extra all --locked --python 3.11

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
