"""Experiment tracking, optional by construction.

MLflow lives in the `tracking` extra, but `train_eval` imported it at module
level — so `make install` (which installs `classical,dev`) produced a tree
where the README's own `dvc repro` died on `ModuleNotFoundError: mlflow`
before reading a single row. A run whose *results* do not depend on MLflow
must not fail to produce them because a bookkeeping dependency is absent.

`open_tracker` returns a real MLflow-backed tracker when MLflow is importable
and a no-op one otherwise, logging which it chose. The no-op branch is loud
in the log and invisible in the tables, which is the right way round: the
metrics written to `reports/tables/` are the deliverable, and MLflow is a
convenience on top of them.
"""

from __future__ import annotations

import contextlib
import logging
from collections.abc import Iterator
from typing import Any, Protocol

logger = logging.getLogger(__name__)


class Tracker(Protocol):
    """What `train_eval` needs from an experiment tracker. Nothing else."""

    def run(self, run_name: str) -> contextlib.AbstractContextManager[Any]: ...

    def log_param(self, key: str, value: Any) -> None: ...

    def log_metric(self, key: str, value: float) -> None: ...


class NullTracker:
    """Discards everything. Used when MLflow is not installed."""

    def run(self, run_name: str) -> contextlib.AbstractContextManager[Any]:
        return contextlib.nullcontext()

    def log_param(self, key: str, value: Any) -> None:
        return None

    def log_metric(self, key: str, value: float) -> None:
        return None


class MlflowTracker:
    """Thin adapter over the MLflow fluent API."""

    def __init__(self, tracking_uri: str, experiment_name: str) -> None:
        import mlflow

        self._mlflow = mlflow
        mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_experiment(experiment_name)

    @contextlib.contextmanager
    def run(self, run_name: str) -> Iterator[Any]:
        with self._mlflow.start_run(run_name=run_name) as active:
            yield active

    def log_param(self, key: str, value: Any) -> None:
        self._mlflow.log_param(key, value)

    def log_metric(self, key: str, value: float) -> None:
        self._mlflow.log_metric(key, value)


def open_tracker(tracking_uri: str, experiment_name: str) -> Tracker:
    """An MLflow tracker if one can be opened, else a no-op.

    Every failure is caught, not just `ImportError`. The narrower version was a
    mistake with a concrete cost: MLflow 3.15 refuses a `file:./mlruns`
    backend outright — "in maintenance mode", raising `MlflowException` — so
    the whole `train_eval` stage died at startup, before reading a row, because
    of a bookkeeping backend the results do not depend on.

    A tracker that cannot be opened is a tracker that is not used. The run
    still produces `reports/tables/`, and the reason is logged loudly.
    """
    try:
        tracker = MlflowTracker(tracking_uri, experiment_name)
    except ImportError:
        logger.warning(
            "MLflow is not installed — run metrics will be written to reports/tables/ but "
            "not tracked. Install the extra with `uv pip install -e '.[tracking]'` to enable "
            "tracking at %s.",
            tracking_uri,
        )
        return NullTracker()
    except Exception as exc:  # noqa: BLE001 - tracking must never fail the run
        logger.warning(
            "MLflow could not be opened at %s (%s: %s) — continuing without tracking. "
            "Results are unaffected and still written to reports/tables/.",
            tracking_uri,
            type(exc).__name__,
            exc,
        )
        return NullTracker()
    logger.info("MLflow tracking to %s (experiment %r).", tracking_uri, experiment_name)
    return tracker
