"""No test depends on the free space of the machine running it.

`ingest.download.require_free_space` guards every download and conversion with
2 GB of headroom, so on a nearly full disk tests that convert a few kilobytes
failed for a reason unrelated to the code. Tests of the guard itself replace
`shutil.disk_usage` again, which takes precedence over this default.
"""

from __future__ import annotations

import types

import pytest


@pytest.fixture(autouse=True)
def _ample_disk(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "authbench.ingest.download.shutil.disk_usage",
        lambda _: types.SimpleNamespace(total=2**50, used=0, free=2**45),
    )
