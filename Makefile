.PHONY: install install-lanl install-all lint typecheck test demo report preflight clean

VENV := .venv
PY := py -3.11 -m uv run --python $(VENV)

# Enough for `make demo` and the test suite. NOT enough for `dvc repro`: that
# needs DVC itself, which lives in the `tracking` extra — see install-lanl.
install:
	py -3.11 -m uv venv --python 3.11 $(VENV)
	py -3.11 -m uv pip install -e ".[classical,dev]" --python $(VENV)

# What the real-dataset run needs: the classical models, plus DVC and MLflow.
install-lanl:
	py -3.11 -m uv venv --python 3.11 $(VENV)
	py -3.11 -m uv pip install -e ".[classical,tracking,dev]" --python $(VENV)

install-all:
	py -3.11 -m uv venv --python 3.11 $(VENV)
	py -3.11 -m uv pip install -e ".[all]" --python $(VENV)

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
