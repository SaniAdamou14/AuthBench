"""Resumable, checksum-verified download of the LANL source files (US-101).

Uses HTTP Range requests to resume after interruption, and refuses to
re-download a file whose SHA-256 already matches the recorded value.
"""

from __future__ import annotations

import contextlib
import hashlib
import logging
import os
import re
import time
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import requests
from tqdm import tqdm

logger = logging.getLogger(__name__)

_CHUNK_SIZE = 1024 * 1024  # 1 MiB
_TRANSIENT_ERRORS = (
    requests.exceptions.ChunkedEncodingError,
    requests.exceptions.ConnectionError,
    requests.exceptions.Timeout,
)
_CONTENT_RANGE_START_RE = re.compile(r"bytes (\d+)-")


class ChecksumMismatchError(RuntimeError):
    """Raised when a downloaded file's SHA-256 does not match the recorded value."""


class ConcurrentDownloadError(RuntimeError):
    """Raised when another process already holds the lock for this destination.

    Two uncoordinated writers appending to the same partial file is exactly
    how a multi-GB download gets silently corrupted (duplicated bytes past
    the real end of file) — this makes that mistake fail loudly instead.
    """


class CorruptPartialDownloadError(RuntimeError):
    """Raised when the partial file on disk is inconsistent with what the
    server reports (e.g. already larger than the resource's real size, or a
    206 response whose `Content-Range` start doesn't match what was
    requested). Safer to stop and ask for the file to be removed than to
    guess how to repair it.
    """


@contextlib.contextmanager
def _exclusive_lock(dest: Path) -> Iterator[None]:
    """A simple cross-process lock file: fails fast if another process is
    already downloading to the same `dest`, rather than letting two writers
    silently corrupt the same partial file.
    """
    lock_path = dest.with_name(dest.name + ".lock")
    try:
        fd = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        os.close(fd)
    except FileExistsError as exc:
        raise ConcurrentDownloadError(
            f"{lock_path} already exists — another download of {dest.name} is likely in "
            "progress. If you're sure that's not the case (e.g. a previous run crashed "
            "without cleaning up), delete the lock file and retry."
        ) from exc

    try:
        yield
    finally:
        lock_path.unlink(missing_ok=True)


LANL_FENCE_BASE = "https://csr.lanl.gov"


def fetch_lanl_fence_token(
    email: str, usage: str, *, session: requests.Session | None = None, timeout_s: float = 20.0
) -> str:
    """LANL's cyber1 dataset is gated behind a click-through data-use form
    (`csr.lanl.gov/data/cyber1/`, see `js/fence.js`): submitting an email and
    a usage statement returns a token used to build the real file URLs
    (`{LANL_FENCE_BASE}/data-fence/{token}/cyber1/<file>`). There is no
    stable, permanent URL for the raw files — the site changed from the
    plain static URLs the original spec assumed.
    """
    session = session or requests.Session()
    response = session.get(
        f"{LANL_FENCE_BASE}/data-fence/token",
        params={"email": email, "usage": usage},
        timeout=timeout_s,
    )
    response.raise_for_status()
    return response.text.strip()


def lanl_file_url(token: str, filename: str) -> str:
    return f"{LANL_FENCE_BASE}/data-fence/{token}/cyber1/{filename}"


@dataclass
class DownloadResult:
    path: Path
    sha256: str
    bytes_downloaded: int
    resumed: bool
    skipped: bool  # True if the file was already present and valid


def sha256_of(path: Path, chunk_size: int = _CHUNK_SIZE) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _download_attempt(
    url: str, dest: Path, session: requests.Session, timeout_s: float
) -> tuple[bool, int]:
    """One resume-aware GET. Returns (resumed, bytes_downloaded_this_attempt).
    Raises one of `_TRANSIENT_ERRORS` on a dropped connection — the caller
    decides whether to retry.
    """
    resume_from = dest.stat().st_size if dest.exists() else 0
    headers = {"Range": f"bytes={resume_from}-"} if resume_from else {}

    with session.get(url, headers=headers, stream=True, timeout=timeout_s) as response:
        if response.status_code == 416:
            raise CorruptPartialDownloadError(
                f"{dest.name}: server rejected resuming from byte {resume_from} (416 Range Not "
                "Satisfiable) — the partial file is almost certainly larger than the real "
                "resource (e.g. from a previous concurrent/corrupted download). Delete "
                f"{dest} and restart."
            )
        response.raise_for_status()  # a 404/error page is not valid file content

        resumed = response.status_code == 206
        if resumed:
            # Don't just trust the 206 status — some proxies return it while
            # actually ignoring the Range and serving from byte 0. Only treat
            # it as a real resume if Content-Range confirms the same start
            # offset we asked for; otherwise fall back to a clean restart
            # rather than silently appending a duplicate stream (exactly how
            # the file grew past its real size the last time).
            content_range = response.headers.get("Content-Range", "")
            match = _CONTENT_RANGE_START_RE.match(content_range)
            if not match or int(match.group(1)) != resume_from:
                logger.warning(
                    "%s: server returned 206 but Content-Range %r doesn't confirm resuming "
                    "from byte %d — restarting this file from scratch.",
                    dest.name,
                    content_range,
                    resume_from,
                )
                resumed = False
        if resume_from and not resumed:
            # Server ignored the Range request — restart from scratch to stay correct.
            resume_from = 0

        mode = "ab" if resumed else "wb"
        total = int(response.headers.get("content-length", 0)) + resume_from
        bytes_downloaded = 0

        with (
            dest.open(mode) as f,
            tqdm(
                total=total or None,
                initial=resume_from,
                unit="B",
                unit_scale=True,
                desc=dest.name,
            ) as bar,
        ):
            for chunk in response.iter_content(chunk_size=_CHUNK_SIZE):
                if not chunk:
                    continue
                f.write(chunk)
                bytes_downloaded += len(chunk)
                bar.update(len(chunk))

    return resumed, bytes_downloaded


def download_with_resume(
    url: str,
    dest: Path,
    expected_sha256: str | None,
    *,
    session: requests.Session | None = None,
    timeout_s: float = 30.0,
    max_retries: int = 8,
    retry_backoff_s: float = 5.0,
) -> DownloadResult:
    """Download `url` to `dest`, resuming from `dest`'s current size if it
    exists, and verifying `expected_sha256` on completion.

    If `dest` already exists and its checksum matches `expected_sha256`,
    nothing is downloaded (US-101: "relancer la commande ... ne retélécharge
    rien"). A dropped connection mid-transfer (`_TRANSIENT_ERRORS` — common on
    a multi-GB download) is retried up to `max_retries` times, each retry
    resuming from the partially-written file's current size rather than
    starting over.
    """
    dest.parent.mkdir(parents=True, exist_ok=True)
    session = session or requests.Session()

    if dest.exists() and expected_sha256 is not None:
        existing_hash = sha256_of(dest)
        if existing_hash == expected_sha256:
            logger.info("%s already present and valid, skipping download", dest.name)
            return DownloadResult(dest, existing_hash, 0, resumed=False, skipped=True)

    resumed = False
    total_bytes_downloaded = 0
    with _exclusive_lock(dest):
        for attempt in range(1, max_retries + 1):
            try:
                resumed, bytes_this_attempt = _download_attempt(url, dest, session, timeout_s)
                total_bytes_downloaded += bytes_this_attempt
                break
            except _TRANSIENT_ERRORS as exc:
                if attempt == max_retries:
                    raise
                logger.warning(
                    "%s: transient error on attempt %d/%d (%s), resuming from %d bytes in %.0fs",
                    dest.name,
                    attempt,
                    max_retries,
                    exc,
                    dest.stat().st_size if dest.exists() else 0,
                    retry_backoff_s,
                )
                time.sleep(retry_backoff_s)

    final_hash = sha256_of(dest)
    if expected_sha256 is not None and final_hash != expected_sha256:
        raise ChecksumMismatchError(
            f"{dest.name}: expected sha256={expected_sha256}, got {final_hash}. "
            "Pipeline aborted — see US-101."
        )

    return DownloadResult(dest, final_hash, total_bytes_downloaded, resumed=resumed, skipped=False)
