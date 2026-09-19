# tools/audiocpp — in-tree audio.cpp native runtime

`audio.cpp` (v0.8.1) is a C++ inference runtime for local audio models
(TTS, ASR, music, …). ANode drives it as a **sidecar subprocess** for
offline generation jobs that cannot run on the real-time audio thread
(see `audiocpp_backend.py` at the repo root).

* `bin/` — extracted `audiocpp_cli` + backend DLLs. **Git-ignored.**
  Fetch with `fetch_audiocpp.py` (verifies SHA256).
* `fetch_audiocpp.py` — downloads the pinned release + CUDA runtime.
  Stdlib only: `python fetch_audiocpp.py` from the repo root.
* `fetch_yue2_gguf.py` — downloads the YuE2 GGUF model + sidecars into
  `models/YuE2-3B-GGUF/` (verifies against the publisher's SHA256SUMS).

Currently only **Windows x64 + CUDA** is auto-fetched (CUDA 13.3 track,
required for Blackwell/sm_120 GPUs such as the RTX 5070). Other
platforms install manually: download the matching v0.8.1 asset from the
[releases page](https://github.com/0xShug0/audio.cpp/releases) and extract
it under `bin/` so `audiocpp_cli` sits next to its backend libraries:

* Linux x64 — `audio-v0.8.1-bin-ubuntu-x64-cpu-portable.tar.gz` (works
  everywhere), `-cuda12.8-colab`, or `-vulkan(-portable)` for GPU builds.
* macOS — `audio-v0.8.1-bin-macos-arm64-metal.tar.gz` (Apple Silicon) or
  `audio-v0.8.1-bin-macos-x64-metal.tar.gz` (Intel).

On Unix also make the CLI executable (`chmod +x tools/audiocpp/bin/audiocpp_cli`);
tarballs usually preserve the bit, manual extraction sometimes does not.
CPU-only installs work out of the box: the nodes pick `--backend cpu`
automatically when torch sees no CUDA GPU.
