# Feeding audio into Whistle from any application

Whistle is a local speech-to-text model: 16.9 MB, CPU-only, Apache-2.0,
English/German/French/Spanish/Italian/Dutch/Polish. It reads audio and
returns text, so it plugs into the *input* side of a program: a live
caption line, a transcript column, a search index, a command router. No
server, no API key, no GPU, no network after the one-time fetch.

Everything below was measured on a single Windows CPU core with the pinned
model. The performance section (§5) is the part most worth reading before
you commit to a design — batch and streaming differ by roughly 9x and only
one of them is real-time.

## 1. What you get out

| Property        | Value                                                    |
| --------------- | -------------------------------------------------------- |
| Input rate      | **16 000 Hz, mono** (hard requirement — see §4)          |
| Input dtype     | `float32` samples in −1…1, or a 16 kHz WAV path           |
| `text`          | Committed transcript; `""` for silence or steady noise    |
| `language`      | Detected `en de fr es it nl pl`, or `""` for no speech     |
| `ttft_ms`       | Milliseconds from call to first token                     |
| `decode_tps`    | Decoder tokens/second after the first token               |
| `words`         | Optional: `{word, start, end, probability}` per word       |
| Streaming step  | `{text, words, pending, language, received, pass_ms}`      |

Two-part output is worth understanding up front. In streaming mode `text`
holds only what the decoder has **committed**; the unconfirmed tail lives in
`pending`. Join the committed parts to build a transcript; render `pending`
separately if you want live captions to look responsive.

`probability` per word is not a confidence score for the utterance — it is
the decoder's probability for that word. Use it to grey out uncertain words,
not to gate the transcript.

## 2. One-time setup (pinned, reproducible)

`pip install cactus-needle` pulls a pure-Python package (its only runtime
dependency is `huggingface_hub`). The native engine and the weights are
**not** in the wheel — fetch them yourself so the install cannot silently
reach the network later:

```bash
python -m pip install "cactus-needle[mic]==3.2.1"

# Weights (16 919 407 B, sha256 b6e02f04…d461dffeb)
curl -L -o whistle.cact \
  https://huggingface.co/Cactus-Compute/whistle/resolve/main/whistle.cact

# Engine: the platform wheel carries needle/libneedle3.dll as its only
# relevant member. Windows x64 shown; substitute your platform tag.
#   win_amd64 | win_arm64 | macosx_11_0_arm64
#   macosx_11_0_x86_64  | manylinux2014_x86_64 | manylinux2014_aarch64
curl -L -o engine.whl \
  https://huggingface.co/Cactus-Compute/needle3/resolve/main/python/cactus_needle-3.3.1-py3-none-win_amd64.whl
unzip -j engine.whl 'needle/libneedle3.*' -d .
```

Verify before trusting either file:

| File                        | Bytes   | sha256                                                          |
| --------------------------- | ------- | --------------------------------------------------------------- |
| `whistle.cact`              | 16919407 | `b6e02f048568ac5d01a2042556c658061e699acbc0aa2a1439f52f3d461dffeb` |
| `cactus_needle-3.3.1…whl`   | 699518  | `4e50c5fa983c8737de200d9e96af937dd8135371ca557fe8c7c9eb1adba39f5a` |
| `libneedle3.dll` (in wheel) | 1531392 | `58d22ea22266094a3ef9213c02b9ab7cd06507642440e610cfbd41ac489298c3` |

Both the package pin and the engine pin matter and they are **different
numbers** (`3.2.1` vs `3.3.1`) — that split is upstream's own. The `.cact`
container format is versioned against the engine, so a mismatch fails at
load with an opaque "unknown .cact format tag" rather than degrading. Move
the two pins together, never one alone.

Licensing is Apache-2.0 for weights and engine (see the fetched
`LICENSE`). Fetch at runtime; do not commit or redistribute the binaries.

## 3. Minimal transcription

```python
import os

os.environ["NEEDLE3_LIB_PATH"]        = "/path/to/libneedle3.dll"
os.environ["NEEDLE_WHISTLE_WEIGHTS"]  = "/path/to/whistle.cact"
os.environ["HF_HUB_OFFLINE"]          = "1"   # fail fast instead of fetching
os.environ["NEEDLE_TELEMETRY"]        = "0"   # upstream usage counters are ON
os.environ["DO_NOT_TRACK"]            = "1"   # by default; opt out explicitly

import needle

model = needle.Whistle(weights="/path/to/whistle.cact")   # ~0.03 s

r = model.transcribe("clip.wav", language="en", word_timestamps=True)
r["text"]                    # "turn off the kitchen lights"
r["language"]                # "en"
r["words"][0]                # {'word': 'So', 'start': 0.32, 'end': 0.48,
                            #  'probability': 0.629}
```

`HF_HUB_OFFLINE=1` matters. Without it, a path typo makes the package
silently download from Hugging Face — from whatever thread called in.
`NEEDLE_TELEMETRY`/`DO_NOT_TRACK` matter for a different reason: they are
**not** on by default, `needle.stream()` calls `_track()` on every call, and
`HF_HUB_OFFLINE` does not cover that path.

## 4. Getting audio in — the part that bites

Whistle is strict about sample rate and **silent** about it. Feeding it
48 kHz samples does not raise; it produces a fluent, confident, wrong
transcript. Measured on a known clip:

| What you pass                                  | Result                                   |
| ---------------------------------------------- | ---------------------------------------- |
| 16 kHz array                                   | `"So you guys know that Canada is not joining the EU, right?"` |
| **48 kHz array** (played 3x slow)              | `"I don't know what you're talking about."` |
| 48 kHz WAV **path**                            | `"See, guys, know that Canada is not joining the EU, right?"` |

The third row is the trap in miniature: the *same file* works when passed as
a path and silently corrupts when passed as an array, because the path
branch resamples via `soxr` and the array branch does not.

So resample explicitly and assert:

```python
audio = decode_and_resample(path, target_sr=16000, target_channels=1)
assert audio.dtype == np.float32 and audio.ndim == 2 and audio.shape[0] == 1
model.transcribe(audio)
```

Notes on each entry point:

- **WAV path** — stdlib `wave` only. 16 kHz needs nothing extra; any other
  rate needs `soxr`, i.e. the `[mic]` extra.
- **Raw samples** — `bytes`, `array('f')`, a NumPy array, or any iterable.
  A NumPy array is `astype("float32").tobytes()`, so a `float64` or
  non-contiguous array costs a copy. There is **no rate check**.
- **Stereo** is not downmixed. `_read_wav` averages WAV channels; an array
  you pass is consumed verbatim, so a stereo array is read as if it were
  mono at twice the length.
- **Duration** — the docs say "at most 30 s", but longer input is windowed
  internally and works: a verified 179 s input returned 2490 characters in
  one call.

## 5. Streaming, and the one number that governs your design

Batch `transcribe()` is fast. Streaming `model.stream(chunks)` is roughly
9x slower per second of audio, because the decoder commits a word only when
**two passes agree** — so every pass re-decodes a trailing window. That
overlap is the entire cost model:

```python
for step in model.stream(chunks, language="en"):
    committed += (" " if committed and step["text"] else "") + step["text"]
    render(committed, pending=step["pending"])
```

`chunks` is an iterable of 16 kHz mono blocks; the generator is pulled by
the engine, so terminate it (on a stop flag) and the stream flushes its own
tail.

Measured on 40 s of speech, one CPU core:

| chunk | wall | realtime factor | mean `pass_ms` | transcript |
| ----- | ---- | --------------- | -------------- | ---------- |
| 1 s   | 56.7 s | **0.71x**     | 1382           | 542 chars  |
| 2 s   | 32.8 s | 1.22x           | 1560           | 546 chars  |
| 4 s   | 21.2 s | 1.89x           | 1926           | 549 chars  |
| 8 s   | 16.5 s | 2.42x           | 2750           | 549 chars  |

The transcript is identical at every chunk size. **Chunk length buys
throughput with latency and nothing else**, because the fixed re-decode
window is amortized over more new audio.

Consequences:

- **1 s is below real time on one core.** Do not ship it as a default.
- **4 s is the shortest size with real headroom** (1.89x) and is a sane
  default; 8 s is safer on slower machines.
- **A decode pass takes seconds, not milliseconds.** Size your
  producer→consumer buffer accordingly (see §7) — this is the mistake most
  likely to corrupt your audio.

If you cannot keep up, **drop the oldest audio and say so**. A transcript
that reads as one continuous sentence but is missing speech is worse than
one that admits the gap, and at these speeds the gap is easy to produce.

## 6. Process and threading rules

- **The engine is process-global and not thread-safe.** One `Whistle` per
  process, touched by one thread. Constructing more does not reload — the
  module caches the handle and skips `needle_load` when the weights path
  matches.
- **Weights cannot be unloaded.** Once bound, they stay mapped for the life
  of the process. Plan for a permanent 17 MB footprint, and note that a
  tuned `.cact` cannot be swapped back out.
- **Never call the engine from an audio callback.** It allocates, blocks for
  1–4 s, and holds the GIL in the native layer. Feed it from a worker
  thread and publish results elsewhere.
- **Environment variables are process-global.** Set the five in §3 once,
  before the first call, and leave them alone. Churning them between
  sessions risks a mismatch with the model already mapped in the library.

## 7. Robustness notes (learned integrating it here)

- **Size your handoff buffer in seconds, not blocks.** The obvious
  producer/consumer ring from a pitch tracker holds 64 blocks, sized for a
  ~10 ms analysis window. A Whistle pass costs up to 4 s, during which
  hundreds of blocks arrive. Two distinct failures follow, and only the
  second is obvious:
  - *Silent corruption*: if your consumer holds references to blocks while
    the producer cycles a smaller pool round-robin, slots are overwritten
    while still being read. You get wrong audio, not an error. The pool must
    exceed one whole chunk in blocks.
  - *Dropped audio*: if the ring is smaller than one decode pass, the excess
    arriving mid-pass is discarded. Count and report it.
  A pool covering the largest chunk plus one pass (13 s ≈ 1219 blocks at
  48 kHz/512-frame blocks) fixed both and took a 7 s hole out of a 30 s
  transcript. 2.6 MB is a trivial price.
- **Reuse one `Whistle` instance.** Construction is ~0.03 s, but the native
  handle and weights are process-global, so there is nothing to gain by
  rebuilding.
- **Validate inputs**: finite samples, non-empty, 16 kHz, mono. Empty or
  whitespace audio returns `""` rather than raising, so an empty result is
  normal (silence) and must not be treated as a failure.
- **Decide how silence behaves.** Silence and steady noise return an empty
  transcript rather than an invented sentence — good. Decide whether that
  is "no output", "clear the line", or "keep the last caption".
- **Only one model per process.** `embed(audio)` (encoder rows, one per
  80 ms frame) needs the same instance; there is no second runtime.
- **Pin `language`** on known-language audio; detection costs accuracy.

## 8. Checklist

1. `import needle` works in the interpreter your program actually uses.
2. `whistle.cact` and `libneedle3.*` are present and SHA-256 verified.
3. `HF_HUB_OFFLINE=1`, `NEEDLE_TELEMETRY=0`, `DO_NOT_TRACK=1` set before the
   first call.
4. A known clip transcribed through a **16 kHz mono array** returns the
   expected sentence.
5. The same clip through a 48 kHz array is a deliberate, asserted error —
   not silently plausible garbage.
6. Chunk length chosen from §5; your handoff buffer sized for a multi-second
   pass and for one whole chunk.
7. Drops are counted and surfaced, not swallowed.
8. The engine is touched by exactly one thread, off the audio callback.