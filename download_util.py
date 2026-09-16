"""Resumable, interrupt-safe file downloads. Stdlib only, no Qt/torch.

Shared by the ``tools/audiocpp/fetch_*.py`` scripts and by in-app
background fetching (see ``audiocpp_backend``). Design points:

* Downloads go to ``<dest>.part`` and are atomically renamed only after
  size + SHA256 verification, so an interrupted run (Ctrl+C, crash,
  cancel) never leaves a corrupt ``dest`` behind.
* Re-running resumes via HTTP ``Range`` from the existing ``.part`` size.
  Servers that ignore ``Range`` trigger a clean restart.
* Progress and cancellation flow through callbacks/events, so both a
  console bar and the engine NRT inbox can consume them; the worker never
  touches UI or audio state directly.
"""

import hashlib
import os
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path

CHUNK_SIZE = 1024 * 1024
CONNECT_TIMEOUT = 30


class DownloadCancelled(Exception):
    """Raised when cancel_event is set. The .part file is kept for resume."""


class DownloadError(RuntimeError):
    """Raised for integrity failures and unrecoverable network errors."""


@dataclass
class DownloadSpec:
    url: str
    dest: Path
    size: int | None = None       # expected final bytes; None = unknown
    sha256: str | None = None     # expected hex digest (lowercase); None = skip
    label: str = ""               # human-readable name for progress output

    def __post_init__(self):
        self.dest = Path(self.dest)
        if self.sha256 is not None:
            self.sha256 = self.sha256.lower()
        if not self.label:
            self.label = self.dest.name


def format_bytes(n):
    """Format a byte count for progress output. None -> '?'."""
    if n is None:
        return "?"
    value = float(n)
    for unit in ("B", "KB", "MB", "GB"):
        if value < 1024 or unit == "GB":
            return f"{value:.1f} {unit}" if unit != "B" else f"{int(value)} B"
        value /= 1024


def _sha256_of(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _verify(path, size, sha256):
    """True if an existing file matches all known expectations."""
    try:
        if not path.is_file():
            return False
        if size is not None and path.stat().st_size != size:
            return False
        if sha256 is not None and _sha256_of(path) != sha256:
            return False
        return True
    except OSError:
        return False


def _parse_total(response, resume_from, fallback_size):
    """Best-effort total byte count from Content-Range / Content-Length."""
    cr = response.headers.get("Content-Range")
    if cr:
        # "bytes 100-999/2877774656"
        try:
            return int(cr.rsplit("/", 1)[-1])
        except ValueError:
            pass
    length = response.headers.get("Content-Length")
    if length:
        try:
            return resume_from + int(length)
        except ValueError:
            pass
    return fallback_size


def _open_request(url, resume_from):
    req = urllib.request.Request(url, headers={"User-Agent": "ANode-fetch"})
    if resume_from > 0:
        req.add_header("Range", f"bytes={resume_from}-")
    return urllib.request.urlopen(req, timeout=CONNECT_TIMEOUT)


def _download_one(spec, progress_cb, cancel_event):
    dest, part = spec.dest, spec.dest.with_name(spec.dest.name + ".part")
    if _verify(dest, spec.size, spec.sha256):
        if progress_cb:
            progress_cb(spec.label, spec.size or 0, spec.size, "present")
        return dest

    resume_from = part.stat().st_size if part.is_file() else 0
    if cancel_event is not None and cancel_event.is_set():
        raise DownloadCancelled(f"cancelled before {spec.label}")

    last_error = None
    for attempt in range(1, 4):
        try:
            return _attempt(spec, part, resume_from, progress_cb, cancel_event)
        except DownloadCancelled:
            raise
        except DownloadError:
            raise
        except Exception as e:  # transient network failure: retry
            last_error = e
            resume_from = part.stat().st_size if part.is_file() else 0
            time.sleep(min(2 ** attempt, 8))
    raise DownloadError(f"download failed after 3 attempts ({spec.label}): {last_error}")


def _attempt(spec, part, resume_from, progress_cb, cancel_event):
    dest = spec.dest
    part.parent.mkdir(parents=True, exist_ok=True)
    try:
        response = _open_request(spec.url, resume_from)
    except urllib.error.HTTPError as e:
        if e.code == 416 and resume_from > 0:
            # Range unsatisfiable: the part may already be complete.
            if _verify(part, spec.size, spec.sha256):
                os.replace(part, dest)
                if progress_cb:
                    progress_cb(spec.label, spec.size or 0, spec.size, "done")
                return dest
            part.unlink(missing_ok=True)
            return _attempt(spec, part, 0, progress_cb, cancel_event)
        if e.code >= 500:
            raise RuntimeError(f"HTTP {e.code} for {spec.url}")  # transient: retried
        raise DownloadError(f"HTTP {e.code} for {spec.url}")
    code = response.getcode()
    if code == 206 and resume_from > 0:
        mode, done = "ab", resume_from
    else:
        # Server ignored Range (or fresh start): restart cleanly.
        mode, done = "wb", 0
    total = _parse_total(response, done, spec.size)
    try:
        with open(part, mode) as f:
            while True:
                if cancel_event is not None and cancel_event.is_set():
                    raise DownloadCancelled(f"cancelled during {spec.label}")
                chunk = response.read(CHUNK_SIZE)
                if not chunk:
                    break
                f.write(chunk)
                done += len(chunk)
                if progress_cb:
                    progress_cb(spec.label, done, total, "downloading")
    finally:
        try:
            response.close()
        except Exception:
            pass
    if progress_cb:
        progress_cb(spec.label, done, total, "verifying")
    if spec.size is not None and part.stat().st_size != spec.size:
        part.unlink(missing_ok=True)
        raise DownloadError(
            f"size mismatch for {spec.label}: "
            f"{part.stat().st_size if part.exists() else 0} != {spec.size}"
        )
    if spec.sha256 is not None and _sha256_of(part) != spec.sha256:
        part.unlink(missing_ok=True)
        raise DownloadError(f"SHA256 mismatch for {spec.label}")
    os.replace(part, dest)
    if progress_cb:
        progress_cb(spec.label, done, total, "done")
    return dest


def fetch_all(specs, progress_cb=None, cancel_event=None):
    """Download every spec, resuming partial files. Returns dest paths.

    Raises :class:`DownloadCancelled` (part files kept) or
    :class:`DownloadError`. Never leaves a corrupt ``dest``: ``dest`` is
    only written by atomic rename after verification.
    """
    return [_download_one(spec, progress_cb, cancel_event) for spec in specs]


class ConsoleProgress:
    """Single-line stderr progress bar for the fetch scripts."""

    def __init__(self, out=None, min_interval=0.25):
        self.out = out if out is not None else sys.stderr
        self.min_interval = min_interval
        self._last = 0.0
        self._labels = set()

    def __call__(self, label, done, total, state):
        now = time.monotonic()
        if state == "present" and label not in self._labels:
            self.out.write(f"{label}: already present, skipping\n")
            self._labels.add(label)
            return
        if state in ("verifying", "done") or now - self._last >= self.min_interval:
            self._last = now
            if total:
                pct = 100.0 * done / total if total else 0.0
                line = f"\r{label}: {format_bytes(done)} / {format_bytes(total)} ({pct:.1f}%)"
            else:
                line = f"\r{label}: {format_bytes(done)}"
            if state == "done":
                line += " done\n"
            elif state == "verifying":
                line += " verifying…\n"
            self.out.write(line)
            self.out.flush()
