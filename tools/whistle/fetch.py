#!/usr/bin/env python3
"""Fetch the Whistle model + native engine into models/Whistle/.

Interrupt-safe and resumable: Ctrl+C keeps `.part` files and re-running
continues where it stopped. Stdlib only; run from the repo root:

    python tools/whistle/fetch.py

Downloads the pinned whistle.cact (16.9 MB), its Apache-2.0 LICENSE and the
pinned cactus-needle engine wheel, then unpacks needle/libneedle3.dll out of
the wheel. Whistle is Apache-2.0: fetch-at-runtime only, never committed.
Never downloads audio.cpp binaries, CUDA libraries, or pip packages.
"""
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from audiocpp_backend import REPO_ROOT as _ROOT, missing_specs
from download_util import (
    ConsoleProgress,
    DownloadCancelled,
    DownloadError,
    fetch_all,
)
from whistle_backend import (
    DEFAULT_MODEL_DIR,
    extract_engine_wheel,
    whistle_fetch_specs,
)


def main() -> int:
    model_dir = _ROOT / DEFAULT_MODEL_DIR
    specs = whistle_fetch_specs(model_dir)
    pending = missing_specs(specs)
    if not pending:
        print("All Whistle model files already present.")
        return _extract(model_dir)
    try:
        fetch_all(pending, progress_cb=ConsoleProgress())
    except DownloadCancelled:
        print("\nInterrupted — re-run this script to resume.")
        return 130
    except DownloadError as e:
        print(f"\nFailed: {e}")
        return 1
    print("Whistle and the needle engine are Apache-2.0; "
          "see models/Whistle/LICENSE.")
    return _extract(model_dir)


def _extract(model_dir) -> int:
    """Unpack the engine library; the weights are only usable with it."""
    try:
        library = extract_engine_wheel(model_dir)
    except DownloadError as e:
        print(f"\nFailed: {e}")
        return 1
    print(f"Ready: {library}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())