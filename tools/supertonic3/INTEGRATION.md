# Feeding Supertonic 3 into any PCM-input program

Supertonic 3 is a local ONNX text-to-speech system. Its Python SDK turns
text into a plain PCM buffer, which you can hand to anything that
accepts PCM: an audio callback, a file writer, a network stream, a game
engine, another TTS/DSP pipeline. No server, no API calls, no GPU needed.

## 1. What you get out

| Property     | Value                                              |
| ------------ | -------------------------------------------------- |
| Sample rate  | 44 100 Hz (`tts.sample_rate`)                      |
| Channels     | 1 (mono)                                           |
| Dtype        | `float32` NumPy array, shape `(1, N)`              |
| Range        | roughly −1…1                                       |
| Duration     | `N / 44100` seconds (authoritative; see §5)        |
| Voice        | fixed preset or imported voice-style JSON          |
| Languages    | 31 codes plus `na` (language-agnostic fallback)    |

The SDK handles long-text chunking internally (300 chars per chunk, 120
for Korean) and bakes the inter-chunk silence into the returned buffer,
so one `synthesize()` call yields one ready-to-play utterance.

## 2. One-time setup (pinned, reproducible)

Use the archived, pinned revisions — not floating `main`:

```bash
# Python 3.11, then:
python -m pip install \
  "git+https://github.com/supertone-oss-archive/supertonic-py.git@df0f9686dac7fbbde391b759e2ee5286a3737622"

# Model weights (~398 MB) into a local directory, e.g. ./assets:
hf download supertone-oss-archive/supertonic-3 \
  --revision aafc6e32416a594460b32413efc49d7fe4ce6d46 \
  --local-dir assets
```

`assets/` must then contain `onnx/*.onnx`, `onnx/tts.json`,
`onnx/unicode_indexer.json`, and `voice_styles/*.json`. Source code is
MIT; the weights are under the OpenRAIL-M license — see
`assets/LICENSE` for its restrictions and obligations.

## 3. Minimal synthesis

```python
from supertonic import TTS

# Build once and reuse: construction loads four ONNX sessions (cold start).
tts = TTS(model="supertonic-3", model_dir="assets", auto_download=False,
          intra_op_num_threads=2, inter_op_num_threads=1)

style = tts.get_voice_style("M1")          # presets M1–M5, F1–F5
# style = tts.get_voice_style_from_path("my_voice.json")  # imported voice

wav, dur = tts.synthesize("Hello, world!", voice_style=style, lang="en",
                          total_steps=8, speed=1.05, silence_duration=0.3)
# wav: float32 (1, N) @ 44100 Hz. dur may be an array — see §5.

tts.save_audio(wav, "output.wav")          # 16-bit WAV via soundfile
```

Parameter bounds enforced by the SDK: `speed` 0.7–2.0, `total_steps`
1–100, `silence_duration` ≥ 0, text ≤ 100 000 chars per call.
Expression tags (`<laugh>`, `<breath>`, `<sigh>`, …) are passed through
verbatim in the text; support varies by language upstream.

## 4. Adapting the PCM to your host

**Sample rate.** If your engine runs at e.g. 48 kHz, resample the
buffer once, up front, on a background thread — never inside a
real-time callback. Any quality resampler works (`resampy`, `scipy`,
`libsamplerate`, `soundfile` + external tool).

**Channels.** Duplicate the mono row for stereo (`np.repeat(wav, 2,
axis=0)` or your framework's mono→stereo helper); truncate leading
channels if your bus is narrower.

**Dtype/layout.** Most PCM consumers want interleaved (`wav.T`,
contiguous) int16 or float32. Convert once:
`np.ascontiguousarray((wav.T * 32767).astype(np.int16))`.

**Feeding a block callback.** Copy from the pre-rendered buffer with a
read cursor — never call `synthesize()` on the audio thread. It blocks
for seconds (model inference) and allocates.

## 5. Robustness notes (learned integrating it here)

- **Reuse one `TTS` instance.** Construction cost (ONNX session load)
  is paid once; per-utterance calls then only pay inference.
- **Synthesize off the audio thread.** Dispatch text to a worker
  thread/queue, publish the finished buffer or file path, and let the
  audio side consume it lock-free (or via a short ring buffer with
  silence on underrun).
- **Derive duration from the buffer**, not the SDK return: some paths
  return `dur` as an array. `frames = wav.shape[1];
  seconds = frames / sample_rate` is always correct.
- **Validate before playing**: shape `(1, N)` with `N > 0`, finite
  samples, positive rate. Reject empty/whitespace-only input before
  calling (the SDK raises `ValueError`).
- **Threads**: `intra_op_num_threads=2, inter_op_num_threads=1` is a
  sane CPU starting point; raise only after measuring contention with
  your engine.
- **Pass `auto_download=False`** with an explicit `model_dir` so a
  missing-assets situation fails fast with a clear error instead of
  silently hitting the network from a background thread.
- **Voice files** are JSON with `style_ttl` (`dims [1, 50, 256]`) and
  `style_dp` (`dims [1, 8, 16]`); validate dims, exact element counts,
  and finite float values before use, and keep the values bit-exact.

## 6. Checklist

1. `import supertonic` works in the interpreter your program uses.
2. `assets/` holds the pinned revision (all four `.onnx` + configs +
   voice styles + LICENSE).
3. A short English synthesis to `output.wav` plays correctly.
4. Your resample/mono→stereo path is exercised with a 44.1 kHz fixture.
5. Synthesis runs on a worker; the audio thread only copies PCM.
