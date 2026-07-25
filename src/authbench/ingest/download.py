"""Resumable, checksum-verified download of the LANL source files (US-101).

Uses HTTP Range requests to resume after interruption, and refuses to
re-download a file whose SHA-256 already matches the recorded value.
"""

from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass
from pathlib import Path

import requests
from tqdm import tqdm

logger = logging.getLogger(__name__)

_CHUNK_SIZE = 1024 * 1024  # 1 MiB


class ChecksumMismatchError(RuntimeError):
    """Raised when a downloaded file's SHA-256 does not match the recorded value."""


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


def download_with_resume(
    url: str,
    dest: Path,
    expected_sha256: str | None,
    *,
    session: requests.Session | None = None,
    timeout_s: float = 30.0,
) -> DownloadResult:
    """Download `url` to `dest`, resuming from `dest`'s current size if it
    exists, and verifying `expected_sha256` on completion.

    If `dest` already exists and its checksum matches `expected_sha256`,
    nothing is downloaded (US-101: "relancer la commande ... ne retélécharge
    rien").
    """
    dest.parent.mkdir(parents=True, exist_ok=True)
    session = session or requests.Session()

    if dest.exists() and expected_sha256 is not None:
        existing_hash = sha256_of(dest)
        if existing_hash == expected_sha256:
            logger.info("%s already present and valid, skipping download", dest.name)
            return DownloadResult(dest, existing_hash, 0, resumed=False, skipped=True)

    resume_from = dest.stat().st_size if dest.exists() else 0
    headers = {"Range": f"bytes={resume_from}-"} if resume_from else {}

    with session.get(url, headers=headers, stream=True, timeout=timeout_s) as response:
        resumed = response.status_code == 206
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

    final_hash = sha256_of(dest)
    if expected_sha256 is not None and final_hash != expected_sha256:
        raise ChecksumMismatchError(
            f"{dest.name}: expected sha256={expected_sha256}, got {final_hash}. "
            "Pipeline aborted — see US-101."
        )

    return DownloadResult(dest, final_hash, bytes_downloaded, resumed=resumed, skipped=False)
