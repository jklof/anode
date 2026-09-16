# models/SheetSage2-GGUF — SheetSage2 transcription weights (GGUF)

Native GGUF (`audio-cpp/SheetSage2-GGUF`, self-contained, no sidecars),
consumed through the in-tree audio.cpp runtime in `tools/audiocpp/` by
the SheetSage2 Transcriber node (audio recording -> ABC melody score).

* `sheetsage2-orig.gguf` — fetched, **git-ignored**.
* `outputs/<date>_<name>/` — per-transcription provenance (score.abc,
  score-melody.abc, input echo, run meta). Local records, never committed.

Fetch with (stdlib only, resumable):

```bash
python tools/audiocpp/fetch_sheetsage2_gguf.py
```

**License:** model weights are **CC BY-NC 4.0** (non-commercial). Do not
use them commercially without separate permission from the rights holders.
