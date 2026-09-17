#!/usr/bin/env python3
"""Fetch SheetSage2 GGUF weights into models/SheetSage2-GGUF/.

Interrupt-safe and resumable: Ctrl+C keeps the `.part` file and re-running
continues where it stopped. Stdlib only; run from the repo root:

    python tools/audiocpp/fetch_sheetsage2_gguf.py

Single self-contained file (no sidecars upstream). No publisher hash is
available, so integrity is checked by exact file size. Weights are
CC BY-NC 4.0 (non-commercial).
"""
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from audiocpp_backend import REPO_ROOT as _ROOT, missing_specs, sheetsage_fetch_specs
from download_util import (
    ConsoleProgress,
    DownloadCancelled,
    DownloadError,
    fetch_all,
)


def main() -> int:
    specs = sheetsage_fetch_specs(_ROOT / "models" / "SheetSage2-GGUF")
    pending = missing_specs(specs)
    if not pending:
        print("SheetSage2 model file already present.")
        return 0
    try:
        fetch_all(pending, progress_cb=ConsoleProgress())
    except DownloadCancelled:
        print("\nInterrupted — re-run this script to resume.")
        return 130
    except DownloadError as e:
        print(f"\nFailed: {e}")
        return 1
    print("Weights are CC BY-NC 4.0 (non-commercial); see models/SheetSage2-GGUF/README.md.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
