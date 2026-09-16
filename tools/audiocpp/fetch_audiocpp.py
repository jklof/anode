#!/usr/bin/env python3
"""Fetch the pinned audio.cpp runtime into tools/audiocpp/bin/.

Stdlib only. Verifies size + SHA256 of every archive before extracting.
Currently supports Windows x64 + CUDA 13.3 (the Blackwell/sm_120-capable
track). Other platforms: download manually from the releases page, see
tools/audiocpp/README.md.
"""
import hashlib
import shutil
import sys
import urllib.request
import zipfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
DEST = REPO_ROOT / "tools" / "audiocpp" / "bin"
TAG = "v0.8.0"
BASE = f"https://github.com/0xShug0/audio.cpp/releases/download/{TAG}"

ASSETS = {
    "win32": [
        (
            f"{BASE}/audio-v0.8.0-bin-windows-x64-cuda13.3.zip",
            269563691,
            "98ccbc3f5e6c73a6ffffa7ae32e1e204f4d12b15cac8a7901fb6be24d943fde6",
        ),
        (
            f"{BASE}/audio-v0.8.0-cudart-windows-x64-cuda13.3.zip",
            575457454,
            "17fc9b2098d8167b6207be838e15187412f070429b6174b7353404e7131897fc",
        ),
    ],
}


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _download(url: str, dest: Path, size: int, sha256: str) -> None:
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
    actual_size = dest.stat().st_size
    if actual_size != size:
        raise RuntimeError(f"size mismatch for {dest.name}: {actual_size} != {size}")
    actual_sha = _sha256(dest)
    if actual_sha != sha256:
        raise RuntimeError(f"sha256 mismatch for {dest.name}: {actual_sha}")
    print(f"verified {dest.name} ({actual_size} bytes)")


def main() -> int:
    assets = ASSETS.get(sys.platform)
    if not assets:
        print(f"No pinned audio.cpp assets for platform {sys.platform!r}.")
        print("Download manually from "
              "https://github.com/0xShug0/audio.cpp/releases and extract")
        print("under tools/audiocpp/bin/ so the CLI sits next to its backend DLLs.")
        return 2
    DEST.mkdir(parents=True, exist_ok=True)
    cli = DEST / ("audiocpp_cli.exe" if sys.platform == "win32" else "audiocpp_cli")
    if cli.exists():
        print(f"{cli} already present; delete tools/audiocpp/bin to refetch.")
        return 0
    tmp = DEST / "_dl"
    tmp.mkdir(exist_ok=True)
    try:
        for url, size, sha256 in assets:
            archive = tmp / url.rsplit("/", 1)[-1]
            print(f"downloading {archive.name} ...", flush=True)
            _download(url, archive, size, sha256)
            with zipfile.ZipFile(archive) as z:
                z.extractall(DEST)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    if not cli.exists():
        raise RuntimeError(f"extract finished but {cli} is missing")
    print(f"ready: {cli}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
