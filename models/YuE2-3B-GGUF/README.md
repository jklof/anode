# models/YuE2-3B-GGUF — YuE2 song-generation weights (GGUF)

Community GGUF quantizations of `m-a-p/YuE2-3B`
(`ngquocvinh/YuE2-3B-GGUF`), consumed through the in-tree audio.cpp
runtime in `tools/audiocpp/` by the YuE2 Song Generator node.

* `yue2-3b-q4_k_m.gguf` + `yue2-vae-f16.gguf` — fetched, **git-ignored**.
* `sidecars/` — model/generation/VAE configs + tokenizer. Fetched, git-ignored.
* `outputs/<date>_<seed>/` — per-generation provenance (wav, request,
  metrics). Local records, never committed.

Fetch everything with (stdlib only, verifies SHA256):

```bash
python tools/audiocpp/fetch_yue2_gguf.py
```

**License:** model weights are **CC BY-NC 4.0** (non-commercial). Do not
use generated-model weights commercially without separate permission
from the rights holders.
