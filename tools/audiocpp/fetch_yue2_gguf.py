#!/usr/bin/env python3
"""Fetch YuE2 GGUF weights into models/YuE2-3B-GGUF/.

Community quantizations (ngquocvinh/YuE2-3B-GGUF) consumed through the
in-tree audio.cpp runtime (tools/audiocpp/). Stdlib only. Verifies every
file against the publisher's SHA256SUMS. Downloads the Q4_K_M main model
(~2.7 GiB, measured ~4.3 GB peak VRAM for a 77 s song on an 8 GB laptop
GPU) plus the F16 VAE and required sidecars.
"""
import hashlib
import shutil
import sys
import urllib.request
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
DEST = REPO_ROOT / "models" / "YuE2-3B-GGUF"
BASE = "https://huggingface.co/ngquocvinh/YuE2-3B-GGUF/resolve/main"

FILES = [
    # (repo path, size bytes, sha256)
    ("yue2-3b-q4_k_m.gguf", 2877774656,
     "24315e53105cb3418b095d1e42e7276bf54b17100c51a43a4d0e37efdeed823d"),
    ("yue2-vae-f16.gguf", 265226496,
     "81e05a79e78ce5cb8deb1d87e17b73e5d205b440be2bd510ae62acf37b300ddb"),
    ("sidecars/yue2-model-config.json", 959,
     "ad3477bbef890bf98ae196c1e4b44779494a6231c4ab66f32708eabadf265329"),
    ("sidecars/yue2-generation-config.json", 466,
     "203830cebde6e3644eb291925362990d198e66c3b6006cd11f1aaa21904bcc61"),
    ("sidecars/yue2-qwen.tiktoken", 2561218,
     "b2b1b8dfb5cc5f024bafc373121c6aba3f66f9a5a0269e243470a1de16a33186"),
    ("sidecars/yue2-vae-config.json", 1378,
     "f0191bb9694009956de44e0c361a6f1334760be4c8f848e599bde242a54a0970"),
]


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _fetch(rel: str, size: int, sha256: str) -> None:
    dest = DEST / rel
    if dest.exists():
        if dest.stat().st_size == size and _sha256(dest) == sha256:
            print(f"present: {rel}")
            return
        print(f"replacing stale {rel}")
        dest.unlink()
    dest.parent.mkdir(parents=True, exist_ok=True)
    url = f"{BASE}/{rel}"
    for attempt in (1, 2, 3):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "ANode-fetch"})
            with urllib.request.urlopen(req) as r, open(dest, "wb") as f:
                shutil.copyfileobj(r, f, length=4 * 1024 * 1024)
            break
        except Exception as e:
            if attempt == 3:
                raise RuntimeError(f"download failed: {url}: {e}")
            print(f"retry {attempt}/3 after: {e}", flush=True)
    if dest.stat().st_size != size or _sha256(dest) != sha256:
        dest.unlink(missing_ok=True)
        raise RuntimeError(f"integrity check failed for {rel}")
    print(f"verified {rel}")


def main() -> int:
    for rel, size, sha256 in FILES:
        print(f"fetching {rel} ...", flush=True)
        _fetch(rel, size, sha256)
    print(f"ready: {DEST}")
    print("Weights are CC BY-NC 4.0 (non-commercial); see models/YuE2-3B-GGUF/README.md.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
