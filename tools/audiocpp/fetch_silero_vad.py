#!/usr/bin/env python3
"""Fetch the Silero VAD asset into models/Silero-VAD/.

Interrupt-safe and resumable: Ctrl+C keeps the `.part` file and re-running
continues where it stopped. Stdlib only; run from the repo root:

    python tools/audiocpp/fetch_silero_vad.py

Tiny (~1.2 MiB) bundled chunking model the audio.cpp sidecar needs for
Qwen3-ASR word timestamps (VAD-chunk first, align per chunk). Pinned to the
runtime version; integrity is size-checked (upstream publishes no hash).
Silero VAD is MIT-licensed.
"""
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from audiocpp_backend import REPO_ROOT as _ROOT, missing_specs, silero_vad_fetch_specs
from download_util import (
    ConsoleProgress,
    DownloadCancelled,
    DownloadError,
    fetch_all,
)


def main() -> int:
    specs = silero_vad_fetch_specs(_ROOT / "models" / "Silero-VAD")
    pending = missing_specs(specs)
    if not pending:
        print("Silero VAD asset already present.")
        return 0
    try:
        fetch_all(pending, progress_cb=ConsoleProgress())
    except DownloadCancelled:
        print("\nInterrupted — re-run this script to resume.")
        return 130
    except DownloadError as e:
        print(f"\nFailed: {e}")
        return 1
    print("Silero VAD is MIT-licensed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
