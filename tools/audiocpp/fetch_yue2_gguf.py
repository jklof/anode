#!/usr/bin/env python3
"""Fetch YuE2 GGUF weights into models/YuE2-3B-GGUF/.

Interrupt-safe and resumable: Ctrl+C keeps `.part` files and re-running
continues where it stopped. Stdlib only; run from the repo root:

    python tools/audiocpp/fetch_yue2_gguf.py

Downloads the Q4_K_M main model (~2.7 GiB, measured ~4.3 GB peak VRAM for
a 77 s song on an 8 GB laptop GPU) plus the F16 VAE and required sidecars.
Weights are CC BY-NC 4.0 (non-commercial).
"""
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from audiocpp_backend import REPO_ROOT as _ROOT, missing_specs, yue2_fetch_specs
from download_util import (
    ConsoleProgress,
    DownloadCancelled,
    DownloadError,
    fetch_all,
)


def main() -> int:
    specs = yue2_fetch_specs(_ROOT / "models" / "YuE2-3B-GGUF")
    pending = missing_specs(specs)
    if not pending:
        print("All YuE2 model files already present.")
        return 0
    try:
        fetch_all(pending, progress_cb=ConsoleProgress())
    except DownloadCancelled:
        print("\nInterrupted — re-run this script to resume.")
        return 130
    except DownloadError as e:
        print(f"\nFailed: {e}")
        return 1
    print("Weights are CC BY-NC 4.0 (non-commercial); see models/YuE2-3B-GGUF/README.md.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
