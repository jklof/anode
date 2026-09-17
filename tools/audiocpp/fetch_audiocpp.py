#!/usr/bin/env python3
"""Fetch the pinned audio.cpp runtime into tools/audiocpp/bin/.

Interrupt-safe and resumable: Ctrl+C keeps a `.part` file and re-running
continues where it stopped. Stdlib only; run from the repo root:

    python tools/audiocpp/fetch_audiocpp.py
"""
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from audiocpp_backend import (
    AudioCppRuntime,
    install_runtime_archives,
    missing_specs,
    runtime_fetch_specs,
)
from download_util import (
    ConsoleProgress,
    DownloadCancelled,
    DownloadError,
    fetch_all,
)


def main() -> int:
    runtime = AudioCppRuntime()
    if runtime.cli.exists():
        print(f"{runtime.cli} already present; delete tools/audiocpp/bin to refetch.")
        return 0
    try:
        specs, staging = runtime_fetch_specs()
    except RuntimeError as e:
        print(e)
        return 2
    pending = missing_specs(specs)
    if not pending:
        install_runtime_archives(staging, runtime.bin_dir)
    else:
        try:
            fetch_all(pending, progress_cb=ConsoleProgress())
        except DownloadCancelled:
            print("\nInterrupted — re-run this script to resume.")
            return 130
        except DownloadError as e:
            print(f"\nFailed: {e}")
            return 1
        install_runtime_archives(staging, runtime.bin_dir)
    print(f"ready: {runtime.cli}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
