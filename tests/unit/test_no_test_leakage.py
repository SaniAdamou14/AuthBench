"""US-107: mechanical guarantee that no `models/` or `features/` module can
import the test-partition loader. This is an architecture test, not a
convention — it must fail loudly if the import ever appears.
"""

from __future__ import annotations

import ast
from pathlib import Path

import authbench

FORBIDDEN_NAME = "get_test_split"
SCANNED_PACKAGES = ["models", "features"]


def _iter_python_files(package: str) -> list[Path]:
    root = Path(authbench.__file__).parent / package
    return sorted(root.rglob("*.py"))


def _imports_forbidden_name(path: Path) -> bool:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and any(
            alias.name == FORBIDDEN_NAME for alias in node.names
        ):
            return True
        if isinstance(node, ast.Import) and any(
            alias.name.endswith(FORBIDDEN_NAME) for alias in node.names
        ):
            return True
        if isinstance(node, ast.Attribute) and node.attr == FORBIDDEN_NAME:
            return True
        if isinstance(node, ast.Name) and node.id == FORBIDDEN_NAME:
            return True
    return False


def test_no_module_under_models_or_features_imports_test_loader() -> None:
    offenders = []
    for package in SCANNED_PACKAGES:
        for path in _iter_python_files(package):
            if _imports_forbidden_name(path):
                offenders.append(str(path))

    assert not offenders, (
        f"{FORBIDDEN_NAME} referenced under models/ or features/: {offenders}. "
        "Training/feature code must never see the test partition (US-107)."
    )
