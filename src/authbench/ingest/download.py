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
import shutil
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

# 5xx and 429 are the server saying "not now", not "never" — on a multi-hour,
# multi-GB transfer they are as routine as a dropped socket, and treating them
# as fatal throws away everything already downloaded in this attempt.
_RETRYABLE_STATUS_CODES = frozenset({408, 425, 429, 500, 502, 503, 504})

# Headroom left on the destination volume after the download. Filling a system
# disk to the last byte is its own failure mode, and the pipeline still has to
# write Parquet next.
DISK_SAFETY_MARGIN_BYTES = 2 * 1024**3  # 2 GiB

# The fence token is an opaque path segment in the file URL, so it can only
# be URL-safe characters and is never empty or HTML.
_TOKEN_RE = re.compile(r"[A-Za-z0-9._~\-]{8,256}")


class ChecksumMismatchError(RuntimeError):
    """Raised when a downloaded file's SHA-256 does not match the recorded value."""


class InsufficientDiskSpaceError(RuntimeError):
    """Raised before a download starts when the destination volume cannot hold
    the file.

    A multi-GB transfer that dies on `No space left on device` after two hours
    leaves a partial file, a stale lock and no useful message. The size is
    known from the server's `Content-Length` before the first byte is written,
    so this is checkable up front — and it is, because on a laptop system
    disk it is the most likely way this stage fails.
    """


class InvalidFenceTokenError(RuntimeError):
    """Raised when LANL's data-use gate returns something that is not a token
    (an HTML page, an empty body, an interstitial). Without this check the
    non-token is pasted into every file URL and the run fails later with a
    404 that points at the wrong cause.
    """


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


def free_space_bytes(path: Path) -> int:
    """Free bytes on the volume that will hold `path`.

    Walks up to the nearest existing ancestor, so this answers for a
    destination directory that has not been created yet.
    """
    probe = path if path.exists() else path.parent
    while not probe.exists() and probe != probe.parent:
        probe = probe.parent
    return shutil.disk_usage(probe).free


def require_free_space(
    path: Path, needed_bytes: int, *, margin_bytes: int = DISK_SAFETY_MARGIN_BYTES, what: str = ""
) -> None:
    """Raise `InsufficientDiskSpaceError` unless `path`'s volume has
    `needed_bytes` plus `margin_bytes` free."""
    available = free_space_bytes(path)
    required = needed_bytes + margin_bytes
    if available < required:
        label = f"{what}: " if what else ""
        raise InsufficientDiskSpaceError(
            f"{label}{path} needs {required / 1e9:.1f} GB free "
            f"({needed_bytes / 1e9:.1f} GB of data + {margin_bytes / 1e9:.1f} GB headroom) "
            f"but the volume has {available / 1e9:.1f} GB. Free up space, or point the "
            "destination at another drive with --out-dir."
        )


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
    token = response.text.strip()

    # A gate that has changed shape, an interstitial, or a captcha page all
    # come back as HTTP 200 with a body. Pasted into a URL that body yields a
    # 404 several layers down, and the user is left debugging the wrong thing.
    # The token is an opaque URL path segment, so anything that cannot be one
    # is rejected here, where the real cause is still visible.
    if not _TOKEN_RE.fullmatch(token):
        preview = " ".join(token[:200].split())
        raise InvalidFenceTokenError(
            f"{LANL_FENCE_BASE}/data-fence/token did not return a usable token for "
            f"{email!r}. Got {len(token)} characters starting with: {preview!r}. "
            "The data-use gate at https://csr.lanl.gov/data/cyber1/ has most likely "
            "changed — open it in a browser, accept the form, and check the URL the "
            "download links point at."
        )
    return token


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
        if response.status_code in _RETRYABLE_STATUS_CODES:
            # Surfaced as a transient error so the caller's retry/backoff loop
            # handles it exactly like a dropped socket, resuming from what is
            # already on disk instead of discarding hours of transfer.
            raise requests.exceptions.ConnectionError(
                f"{dest.name}: HTTP {response.status_code} from {url} (retryable)"
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
        remaining = int(response.headers.get("content-length", 0))
        total = remaining + resume_from
        bytes_downloaded = 0

        # Checked here rather than by the caller because this is the first
        # point at which the size is known: LANL publishes none, and the
        # server only reveals it in the response headers.
        already_on_disk = resume_from if resumed else 0
        require_free_space(dest, max(0, total - already_on_disk), what=f"downloading {dest.name}")

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
