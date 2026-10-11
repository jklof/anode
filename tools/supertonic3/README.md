# tools/supertonic3 — Supertonic 3 local TTS assets

Supertonic 3 is a local ONNX text-to-speech system (~99 M parameters,
31 languages + `na` fallback, 44.1 kHz output). ANode drives the
upstream Python SDK in a supervised child subprocess (`worker.py`)
launched by an NRT worker — never on the real-time audio thread.

## Pins (verified 2026-10-11)

* SDK source `supertone-oss-archive/supertonic-py`
  @ `df0f9686dac7fbbde391b759e2ee5286a3737622`
* Model `supertone-oss-archive/supertonic-3`
  @ `aafc6e32416a594460b32413efc49d7fe4ce6d46`

Source code is MIT; model weights are under the OpenRAIL-M license —
see `models/Supertonic3/LICENSE` for its restrictions and obligations.
Fetch-at-runtime only — never committed, bundled, or redistributed.

## Setup

```bash
conda activate anode-dev
python -m pip install "git+https://github.com/supertone-oss-archive/supertonic-py.git@df0f9686dac7fbbde391b759e2ee5286a3737622"
python tools/supertonic3/fetch.py
```

`fetch.py` downloads the pinned revision into `models/Supertonic3/`
(four ONNX models ~398 MB + `onnx/tts.json`, `onnx/unicode_indexer.json`,
ten `voice_styles/*.json`, `LICENSE`). All assets are SHA-256 verified
(ONNX against the upstream LFS oids; small files against independently
established hashes of the pinned revision). Neither node
construction, Apply, Generate, nor `load_state` downloads anything —
use the node Download button or this script explicitly. Concurrent
downloads to the same model directory are serialized in-app by a
per-destination NRT lock; do not run this script concurrently with an
in-app Download to the same directory.

Python 3.11 / Windows CPU is the verified target. Runtime imports stay
lazy/child-only: the nodes register in the palette without the SDK, and
synthesis without it reports an installation hint.

## Model format

Voice styles are JSON with `style_ttl` (`dims [1, 50, 256]`, 12800
floats) and `style_dp` (`dims [1, 8, 16]`, 128 floats), plus a
`metadata` mapping (source file, sample rates, extraction time).
Values are preserved exactly — never normalized or blended here.
JSON structure alone cannot prove v3 suitability: shipped presets are
the pinned v3 voices; compatibility of externally supplied files is
the user's responsibility. Files with explicitly contradictory version
metadata (v1/v2 markers) are rejected.

## Offline use

After the one-time fetch + SDK install above, synthesis runs fully
offline (no network). The child loads local assets with
`auto_download=False`, synthesizes with explicit language/speed/steps/
pause settings, writes `speech.wav` at the model rate (~44.1 kHz), and
writes `request.json` / `metrics.json` provenance. SamplePlayer converts
to the 48 kHz stereo engine format on its own NRT worker — no resampler
in block processing.

## Limitations

* No recording-based cloning, model training, embedding blending,
  GPU mode, streaming synthesis, or persistent inference server.
  The Voice Designer imports existing voice identities; it does not
  extract a voice from reference audio.
* Expression tags (`<laugh>`, `<breath>`, `<sigh>`, `<surprise>`,
  `<scream>`, `<throatclear>`, `<sad>`, `<angry>`, `<cough>`, `<yawn>`)
  are preserved verbatim but work best in Korean/English/Japanese and
  may be ignored by the model (upstream limitation).
* Each generation starts a fresh child process (cold start per
  request); optimize only after measurement.
* Commercial-use terms are defined by the OpenRAIL-M license; see the
  distributed LICENSE.

## Test commands

```bash
conda activate anode-dev
python -m pytest tests/test_supertonic3_backend.py tests/test_supertonic3_voice_designer.py tests/test_supertonic3_tts.py -q
python tools/supertonic3/fetch.py --help  # (no flag; prints usage via argparse error)
```

Real-model acceptance (needs assets + SDK): English/Finnish, all ten
preset previews, expression tags, multi-chunk text, an imported known
v3 style, cancelled inference, offline execution, and playback under
CPU load. Record startup latency, memory, synthesis RTF, and underruns.
