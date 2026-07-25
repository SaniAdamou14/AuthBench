.PHONY: install install-all lint typecheck test demo report clean

VENV := .venv
PY := py -3.11 -m uv run --python $(VENV)

install:
	py -3.11 -m uv venv --python 3.11 $(VENV)
	py -3.11 -m uv pip install -e ".[classical,dev]" --python $(VENV)

install-all:
	py -3.11 -m uv venv --python 3.11 $(VENV)
	py -3.11 -m uv pip install -e ".[all]" --python $(VENV)

lint:
	$(PY) ruff check src tests
	$(PY) ruff format --check src tests

typecheck:
	$(PY) mypy src

test:
	$(PY) pytest

demo:
	$(PY) authbench demo

report: demo
	@echo "Tables and figures written under reports/ (see reports/tables, reports/figures)."

clean:
	rm -rf outputs multirun .pytest_cache .ruff_cache .mypy_cache htmlcov .coverage
