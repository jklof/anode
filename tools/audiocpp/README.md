# tools/audiocpp — in-tree audio.cpp native runtime

`audio.cpp` (v0.8.0) is a C++ inference runtime for local audio models
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
platforms: see the
[releases page](https://github.com/0xShug0/audio.cpp/releases) and place
the extracted tree under `bin/` so `audiocpp_cli(.exe)` sits next to its
backend libraries.
