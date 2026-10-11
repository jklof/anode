#!/usr/bin/env python3
"""Fetch YuE2 torch weights into models/YuE2-Torch/ (stdlib only).

Duplicates ~80 lines of download machinery deliberately so the worker stays
usable without ANode (extractability over DRY). Node-side fetching reuses
the repo machinery (audiocpp_backend.yue2_torch_fetch_specs); the URL/sha
table below mirrors it — keep both in sync.

    python tools/yue2_torch/fetch.py

Weights are CC BY-NC 4.0 (non-commercial).
"""

import hashlib
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

CHUNK = 1024 * 1024

# (relative path, URL, size bytes or None, sha256 or None).
# NOTE: no tokenizer entry — it ships embedded in the checkpoint as the
# text_encoders.yue2_tokenizer_json tensor (the Qwen .tiktoken sidecar is
# a base vocab that lacks the codec/special ids).
TABLE = [
    ("yue2_3b_bf16.safetensors",
     "https://huggingface.co/Comfy-Org/YuE2/resolve/main"
     "/checkpoints/yue2_3b_bf16.safetensors",
     None, None),
]


def _sha256_of(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _verify(path, size, sha):
    try:
        if not path.is_file():
            return False
        if size is not None and path.stat().st_size != size:
            return False
        if sha is not None and _sha256_of(path) != sha.lower():
            return False
        return True
    except OSError:
        return False


def _fetch(url, dest, size, sha):
    dest = Path(dest)
    part = dest.with_name(dest.name + ".part")
    if _verify(dest, size, sha):
        print(f"{dest.name}: already present, skipping")
        return
    resume = part.stat().st_size if part.is_file() else 0
    req = urllib.request.Request(url, headers={"User-Agent": "ANode-fetch"})
    if resume:
        req.add_header("Range", f"bytes={resume}-")
    try:
        resp = urllib.request.urlopen(req, timeout=30)
    except urllib.error.HTTPError as e:
        print(f"HTTP {e.code} for {url}", file=sys.stderr)
        return 1
    mode, done = ("ab", resume) if resp.getcode() == 206 and resume \
        else ("wb", 0)
    total = size
    t0, last = time.monotonic(), 0.0
    with open(part, mode) as f:
        while True:
            chunk = resp.read(CHUNK)
            if not chunk:
                break
            f.write(chunk)
            done += len(chunk)
            now = time.monotonic()
            if now - last >= 0.25:
                last = now
                if total:
                    print(f"\r{dest.name}: {done / 1e9:.2f} / "
                          f"{total / 1e9:.2f} GB ({100.0 * done / total:.1f}%)",
                          end="", flush=True)
    print()
    if size is not None and part.stat().st_size != size:
        part.unlink(missing_ok=True)
        print(f"size mismatch for {dest.name}", file=sys.stderr)
        return 1
    if sha is not None and _sha256_of(part) != sha.lower():
        part.unlink(missing_ok=True)
        print(f"SHA256 mismatch for {dest.name}", file=sys.stderr)
        return 1
    part.replace(dest)
    print(f"{dest.name}: done in {time.monotonic() - t0:.0f}s")
    return 0


def main():
    root = Path(__file__).resolve().parents[2]
    model_dir = root / "models" / "YuE2-Torch"
    model_dir.mkdir(parents=True, exist_ok=True)
    rc = 0
    for rel, url, size, sha in TABLE:
        rc |= _fetch(url, model_dir / rel, size, sha) or 0
    if not rc:
        print("Weights are CC BY-NC 4.0 (non-commercial); "
              "see models/YuE2-Torch/README.md.")
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
