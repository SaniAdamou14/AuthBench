"""US-101: the failure modes of a multi-GB download nobody watches.

Every test here corresponds to a way the LANL fetch has to fail *loudly and
early* rather than after two hours: no disk, a gate that stopped returning
tokens, a server saying "not now", and a partial file that no longer matches
the resource.
"""

from __future__ import annotations

import types
from pathlib import Path

import pytest
import requests

from authbench.ingest.download import (
    DISK_SAFETY_MARGIN_BYTES,
    ChecksumMismatchError,
    ConcurrentDownloadError,
    InsufficientDiskSpaceError,
    InvalidFenceTokenError,
    download_with_resume,
    fetch_lanl_fence_token,
    lanl_file_url,
    require_free_space,
    sha256_of,
)

PAYLOAD = b"1,U1@DOM1,U1@DOM1,C1,C2,Negotiate,Batch,LogOn,Success\n" * 64


class _FakeResponse:
    def __init__(self, status_code: int, body: bytes = b"", headers: dict[str, str] | None = None):
        self.status_code = status_code
        self._body = body
        self.headers = headers or {}
        self.text = body.decode(errors="replace")

    def __enter__(self) -> _FakeResponse:
        return self

    def __exit__(self, *exc: object) -> None:
        return None

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise requests.exceptions.HTTPError(f"HTTP {self.status_code}")

    def iter_content(self, chunk_size: int = 1) -> list[bytes]:
        return [self._body[i : i + chunk_size] for i in range(0, len(self._body), chunk_size)]


class _FakeSession:
    """A session that replays a scripted list of responses."""

    def __init__(self, responses: list[_FakeResponse]) -> None:
        self._responses = list(responses)
        self.requests: list[dict[str, str]] = []

    def get(self, url: str, **kwargs: object) -> _FakeResponse:
        self.requests.append({"url": url, **{k: str(v) for k, v in kwargs.items()}})
        return self._responses.pop(0)


def _full_body_response() -> _FakeResponse:
    return _FakeResponse(200, PAYLOAD, {"content-length": str(len(PAYLOAD))})


def test_require_free_space_reports_the_shortfall_in_gigabytes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        "authbench.ingest.download.shutil.disk_usage",
        lambda _: types.SimpleNamespace(total=0, used=0, free=1_000_000),
    )

    with pytest.raises(InsufficientDiskSpaceError) as excinfo:
        require_free_space(tmp_path / "auth.txt.gz", 5 * 1024**3, what="downloading auth")

    message = str(excinfo.value)
    assert "downloading auth" in message
    assert "0.0 GB" in message  # what the volume has


def test_free_space_is_checked_before_a_single_byte_is_written(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    dest = tmp_path / "auth.txt.gz"
    monkeypatch.setattr(
        "authbench.ingest.download.shutil.disk_usage",
        lambda _: types.SimpleNamespace(total=0, used=0, free=DISK_SAFETY_MARGIN_BYTES),
    )
    session = _FakeSession([_full_body_response()])

    with pytest.raises(InsufficientDiskSpaceError):
        download_with_resume("http://x/auth", dest, None, session=session)  # type: ignore[arg-type]

    assert not dest.exists() or dest.stat().st_size == 0
    assert not (tmp_path / "auth.txt.gz.lock").exists(), "the lock must not survive the failure"


def test_a_retryable_status_is_retried_and_then_succeeds(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A 503 halfway through a five-hour transfer is the server saying "not
    now". Treating it as fatal throws away everything already downloaded."""
    monkeypatch.setattr("authbench.ingest.download.time.sleep", lambda _: None)
    session = _FakeSession([_FakeResponse(503), _full_body_response()])

    result = download_with_resume(
        "http://x/auth",
        tmp_path / "auth.txt.gz",
        None,
        session=session,  # type: ignore[arg-type]
        retry_backoff_s=0.0,
    )

    assert result.bytes_downloaded == len(PAYLOAD)
    assert (tmp_path / "auth.txt.gz").read_bytes() == PAYLOAD


def test_a_non_retryable_status_aborts_immediately(tmp_path: Path) -> None:
    session = _FakeSession([_FakeResponse(404)])

    with pytest.raises(requests.exceptions.HTTPError):
        download_with_resume("http://x/auth", tmp_path / "auth.txt.gz", None, session=session)  # type: ignore[arg-type]


def test_a_checksum_mismatch_aborts_the_pipeline(tmp_path: Path) -> None:
    session = _FakeSession([_full_body_response()])

    with pytest.raises(ChecksumMismatchError):
        download_with_resume("http://x/auth", tmp_path / "auth.txt.gz", "0" * 64, session=session)  # type: ignore[arg-type]


def test_a_file_already_present_and_valid_is_not_downloaded_again(tmp_path: Path) -> None:
    dest = tmp_path / "auth.txt.gz"
    dest.write_bytes(PAYLOAD)
    session = _FakeSession([])  # any request would IndexError

    result = download_with_resume("http://x/auth", dest, sha256_of(dest), session=session)  # type: ignore[arg-type]

    assert result.skipped is True
    assert result.bytes_downloaded == 0


def test_a_stale_lock_file_fails_loudly_rather_than_corrupting_the_partial(
    tmp_path: Path,
) -> None:
    dest = tmp_path / "auth.txt.gz"
    (tmp_path / "auth.txt.gz.lock").touch()

    with pytest.raises(ConcurrentDownloadError, match="already exists"):
        download_with_resume("http://x/auth", dest, None, session=_FakeSession([]))  # type: ignore[arg-type]


def test_a_gate_that_returns_html_is_not_mistaken_for_a_token() -> None:
    """Without this the HTML lands in every file URL and the run fails several
    layers down with a 404 that points at the wrong cause."""
    session = _FakeSession([_FakeResponse(200, b"<!DOCTYPE html><html><body>Access</body></html>")])

    with pytest.raises(InvalidFenceTokenError, match="did not return a usable token"):
        fetch_lanl_fence_token("a@b.org", "research", session=session)  # type: ignore[arg-type]


def test_an_empty_gate_response_is_rejected() -> None:
    session = _FakeSession([_FakeResponse(200, b"   \n")])

    with pytest.raises(InvalidFenceTokenError):
        fetch_lanl_fence_token("a@b.org", "research", session=session)  # type: ignore[arg-type]


def test_a_plausible_token_is_accepted_and_used_to_build_the_file_url() -> None:
    session = _FakeSession([_FakeResponse(200, b"  abc123-DEF_456.789  \n")])

    token = fetch_lanl_fence_token("a@b.org", "research", session=session)  # type: ignore[arg-type]

    assert token == "abc123-DEF_456.789"
    assert lanl_file_url(token, "auth.txt.gz").endswith(f"/data-fence/{token}/cyber1/auth.txt.gz")
