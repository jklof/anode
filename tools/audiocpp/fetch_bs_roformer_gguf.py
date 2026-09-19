#!/usr/bin/env python3
"""Fetch BS-RoFormer GGUF weights into models/BS-RoFormer-ep368-GGUF/.

Interrupt-safe and resumable: Ctrl+C keeps the `.part` file and re-running
continues where it stopped. Stdlib only; run from the repo root:

    python tools/audiocpp/fetch_bs_roformer_gguf.py

Downloads the default ep368 Q8 package (~165 MB, SHA256-verified single
self-contained file with embedded package spec + config). Upstream
checkpoint: model_bs_roformer_ep_368_sdr_12.9628.ckpt lineage (ViperX/UVR,
MIT — credit UVR and its developers); BS-RoFormer architecture is MIT
(lucidrains/BS-RoFormer). GGUF conversion by audio-cpp.
"""
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from audiocpp_backend import REPO_ROOT as _ROOT, missing_specs, bs_roformer_fetch_specs
from download_util import (
    ConsoleProgress,
    DownloadCancelled,
    DownloadError,
    fetch_all,
)


def main() -> int:
    specs = bs_roformer_fetch_specs(_ROOT / "models" / "BS-RoFormer-ep368-GGUF")
    pending = missing_specs(specs)
    if not pending:
        print("BS-RoFormer model file already present.")
        return 0
    try:
        fetch_all(pending, progress_cb=ConsoleProgress())
    except DownloadCancelled:
        print("\nInterrupted — re-run this script to resume.")
        return 130
    except DownloadError as e:
        print(f"\nFailed: {e}")
        return 1
    print("Upstream checkpoint lineage ViperX/UVR, MIT — credit UVR and its developers; "
          "see models/BS-RoFormer-ep368-GGUF/README.md.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
