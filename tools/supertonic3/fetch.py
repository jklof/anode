#!/usr/bin/env python3
"""Fetch Supertonic 3 model assets into models/Supertonic3/.

Interrupt-safe and resumable: Ctrl+C keeps `.part` files and re-running
continues where it stopped. Stdlib only; run from the repo root:

    python tools/supertonic3/fetch.py

Downloads the pinned archive revision (four ONNX models + configs +
ten voice styles + LICENSE, ~398 MB). Model weights are under the
OpenRAIL-M license (see models/Supertonic3/LICENSE for its terms):
fetch-at-runtime only, never committed. Never
downloads audio.cpp binaries, CUDA libraries, or pip packages.
"""
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from audiocpp_backend import REPO_ROOT as _ROOT, missing_specs
from supertonic3_backend import supertonic3_fetch_specs
from download_util import (
    ConsoleProgress,
    DownloadCancelled,
    DownloadError,
    fetch_all,
)


def main() -> int:
    specs = supertonic3_fetch_specs(_ROOT / "models" / "Supertonic3")
    pending = missing_specs(specs)
    if not pending:
        print("All Supertonic 3 model files already present.")
        return 0
    try:
        fetch_all(pending, progress_cb=ConsoleProgress())
    except DownloadCancelled:
        print("\nInterrupted — re-run this script to resume.")
        return 130
    except DownloadError as e:
        print(f"\nFailed: {e}")
        return 1
    print("Model weights are under the OpenRAIL-M license; "
          "see tools/supertonic3/README.md.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
