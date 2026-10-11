#!/usr/bin/env python3
"""Fetch YuE2 torch weights into models/YuE2-Torch/.

Interrupt-safe and resumable: Ctrl+C keeps `.part` files and re-running
continues where it stopped. Run from the repo root:

    python tools/audiocpp/fetch_yue2_torch.py

Downloads the BF16 safetensors checkpoint (the tokenizer is embedded, so no
separate tokenizer or sidecar files are needed). Full content hashes verify
in this worker process. Weights are CC BY-NC 4.0 (non-commercial).
"""
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from audiocpp_backend import (REPO_ROOT as _ROOT, missing_specs,
                              yue2_torch_fetch_specs)
from download_util import (
    ConsoleProgress,
    DownloadCancelled,
    DownloadError,
    fetch_all,
)


def main() -> int:
    specs = yue2_torch_fetch_specs(_ROOT / "models" / "YuE2-Torch")
    pending = missing_specs(specs)
    if not pending:
        print("All YuE2 torch model files already present.")
        return 0
    try:
        fetch_all(pending, progress_cb=ConsoleProgress())
    except DownloadCancelled:
        print("\nInterrupted — re-run this script to resume.")
        return 130
    except DownloadError as e:
        print(f"\nFailed: {e}")
        return 1
    print("Weights are CC BY-NC 4.0 (non-commercial); see models/YuE2-Torch/README.md.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
