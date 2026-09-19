#!/usr/bin/env python3
"""Fetch Qwen3 Forced Aligner GGUF weights into models/Qwen3-ForcedAligner-GGUF/.

Interrupt-safe and resumable: Ctrl+C keeps `.part` files and re-running
continues where it stopped. Stdlib only; run from the repo root:

    python tools/audiocpp/fetch_qwen3_align_gguf.py

Downloads the default 0.6B Q8 profile (~1.1 GiB, SHA256-verified single
self-contained file with embedded sidecars). Weights are Apache 2.0.
"""
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from audiocpp_backend import REPO_ROOT as _ROOT, missing_specs, qwen3_align_fetch_specs
from download_util import (
    ConsoleProgress,
    DownloadCancelled,
    DownloadError,
    fetch_all,
)


def main() -> int:
    specs = qwen3_align_fetch_specs(_ROOT / "models" / "Qwen3-ForcedAligner-GGUF")
    pending = missing_specs(specs)
    if not pending:
        print("Qwen3 Forced Aligner model file already present.")
        return 0
    try:
        fetch_all(pending, progress_cb=ConsoleProgress())
    except DownloadCancelled:
        print("\nInterrupted — re-run this script to resume.")
        return 130
    except DownloadError as e:
        print(f"\nFailed: {e}")
        return 1
    print("Weights are Apache 2.0.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
