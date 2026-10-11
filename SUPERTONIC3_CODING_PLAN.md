# Coding-agent handoff: Voice Designer + Supertonic 3 TTS

Date: 2026-10-11. Implement the two-node integration described here, following `AGENTS.md`. This plan supersedes naming and scope ambiguities in `SUPERTONIC3_DESIGN.md`; that document supplies research context and upstream links.

## Deliverable and scope

Deliver two registered nodes and their Qt widgets:

1. **Supertonic Voice Designer** (`Supertonic3VoiceDesigner`): choose a preset or import a compatible v3 voice-style JSON, validate it, and save/publish a reusable profile.
2. **Supertonic 3 TTS** (`Supertonic3TTS`): consume text and a voice profile, generate a kept WAV on NRT, and publish its path plus a completion pulse.

Use the existing SamplePlayer for preview/playback and existing vocal effects for pitch/formant tuning. Provide an example patch containing all three nodes. No recording-based cloning, model training, embedding blending, semantic embedding sliders, GPU mode, streaming synthesis, or persistent inference server in this implementation. Blending is a later experiment requiring listening tests.

Preview means Apply on Voice Designer triggers downstream TTS with a short editable test sentence, then SamplePlayer plays the result. The designer's action label should be **Apply voice**, with help explaining the connected preview chain. Do not promise an independent Preview button that cannot work without the connected TTS/player. Generated voice identity and post-playback pitch/formant effects remain separate.

## Inspect first

Read actual implementations before modifying: `base.py` (ports, parameters, `TriggerPulseMixin`, NRT hooks), `core.py` (`NRTExecutor`, command boundaries), `audiocpp_backend.py` (`AudioCppJob` and subprocess supervision), `offline_job_widget.py`, `download_util.py`, `audio_io.py`, `plugins/sample_player.py`, `plugins/qwen3_asr_transcriber.py`, and `plugins/yue2_song_generator.py` (UI parameter flush ordering). Find the controller's Save flush and UI parameter proxy behavior. Inspect existing test fixtures and plugin registration conventions.

Do not refactor the shared job architecture or existing nodes. Preserve unrelated user changes. No commit or push without an explicit request.

## Files to add/change

| File | Responsibility |
| --- | --- |
| `supertonic3_backend.py` | Pinned constants, fetch specs, request validation, profile preparation, artifact handling, worker argv and supervision |
| `tools/supertonic3/worker.py` | Child-only SDK import, CPU inference, WAV and metrics output |
| `tools/supertonic3/fetch.py` | Explicit resumable model download CLI using shared fetch helpers |
| `tools/supertonic3/README.md` | Setup, pins, model format, offline use, limitations, test commands |
| `plugins/supertonic3_voice_designer.py` | Designer node and configured custom widget |
| `plugins/supertonic3_tts.py` | TTS node and custom widget with multiline text editor |
| `tests/test_supertonic3_backend.py` | Validation, request/argv, fetching and worker lifecycle contracts |
| `tests/test_supertonic3_voice_designer.py` | Designer lifecycle/publication/save-load |
| `tests/test_supertonic3_tts.py` | TTS lifecycle, controls, cancellation and chain integration |
| `patches/supertonic3_voice_preview.json` | Working preview patch serialized using the actual graph schema |
| `environment.yml`, `.gitignore`, `README.md` | Pinned SDK dependency, ignored assets/artifacts, feature documentation |

Keep reusable node bases outside `plugins/`; if a helper Node subclass must live there, mark it abstract to prevent palette registration.

## Step 1: pin and verify the runtime/assets

Research pins: SDK source `supertone-oss-archive/supertonic-py@df0f9686dac7fbbde391b759e2ee5286a3737622`; model `supertone-oss-archive/supertonic-3@aafc6e32416a594460b32413efc49d7fe4ce6d46`. Verify availability and compatibility before encoding them. Obtain exact asset sizes and SHA-256 values from the pinned archive; use upstream LFS metadata where available and independently establish hashes for ordinary files. Do not invent integrity values or silently switch to mutable `main`.

Required assets under `models/Supertonic3`: four ONNX models in `onnx/`, `onnx/tts.json`, `onnx/unicode_indexer.json`, ten JSON styles in `voice_styles/`, and license/attribution. Add precise ignore rules for downloaded assets and local generated profiles/audio; leave documentation trackable.

Use `DownloadSpec`/`fetch_all` and the existing explicit download policy. Neither node construction, Apply, Generate, nor load_state may download models. Download only fetches model assets; never audio.cpp binaries, CUDA libraries, or pip packages. Both nodes can expose the shared explicit Download button; avoid file corruption if two requests overlap by following the downloader's actual concurrency guarantees or adding narrow destination coordination on NRT.

Add a reproducible SDK dependency in `environment.yml`. Prefer a published pinned artifact only if equivalent to the verified source; otherwise use the source commit pin. Verify Python 3.11/Windows CPU dependency compatibility. Runtime imports stay lazy/child-only so a missing SDK doesn't prevent palette registration. Show an installation hint when synthesis is attempted without it.

Source code license is MIT and model license is OpenRAIL-M. Include correct notices/help; do not copy the audio.cpp weight license wording. Do not bundle weights.

## Step 2: implement profile validation/preparation

Create a pure/worker-side profile pipeline accepting either a shipped preset or an imported style JSON. Read bounded input, validate dictionary structure, numeric data, finite float32-representable values, dimensions, and exact element counts against the pinned v3 expectations. Reject booleans as numeric data and reject oversized/unreasonable dimensions before allocation. Preserve values; do not normalize or randomly edit embeddings.

The format contains `style_ttl` and `style_dp`, each with `dims` and `data`. Validate with the SDK-compatible schema, including actual nesting in the pinned preset files. Do not infer voice characteristics from individual coordinates.

JSON structure cannot reliably distinguish v2 from v3. Label import **Supertonic 3 voice-style JSON** and explain that compatibility is the user's responsibility for externally supplied files. Reject explicitly contradictory version metadata; record unknown provenance honestly rather than claiming it is verified v3. This corrects the research document's overly strict suggestion to require provenance that existing external files may not provide.

Write a unique kept `voice_style.json` and companion provenance containing source/preset, backend version/pins, source hash, output hash, and display name. Return only paths and small metadata. Save named designer settings through normal patch parameters. Overwriting the same published path is prohibited: each Apply creates a new path so downstream file consumers can distinguish replacements.

## Step 3: implement the supervised synthesis worker

Build a plain immutable request containing text/URI snapshot, selected style source, model directory, language, speed, steps, pause duration, threads, request ID, and output directory. UI/control validation is limited to cheap value checks; all filesystem reads, hashing, asset checks, JSON parsing, and imports occur on NRT/child.

Launch `[sys.executable, worker_absolute_path, "--request", request_absolute_path]` with the existing `run_audiocpp_subprocess(..., prog="supertonic3")`. Use a list argv, inherited activated environment, hidden Windows window, finite timeout, and no shell. Ensure the worker can import repository helpers regardless of working directory, using the established sidecar pattern.

Inside the child:

1. Resolve bounded UTF-8 text input and validate settings; preserve supported expression tags. Use SDK-compatible total text limits and finite numeric bounds. Empty text and invalid connected URIs fail.
2. Load local assets using `TTS(model="supertonic-3", model_dir=..., auto_download=False, intra_op_num_threads=threads, inter_op_num_threads=1)`.
3. Load the chosen profile or preset and call `synthesize` with explicit language/settings. Defaults: English, speed 1.05, 8 steps, 0.3 seconds chunk silence; initial thread count 2.
4. Validate waveform `(1, N)`, positive length/rate, and finite samples. Handle duration as an array if returned that way; derive authoritative file duration from frames/sample rate. Respect the SDK's duration trimming/output-save behavior: inspect it before choosing how to write padded waveform tails.
5. Write WAV via existing audio helpers at the actual model rate, approximately 44.1 kHz. Do not resample here solely for playback; SamplePlayer already handles 48 kHz stereo conversion on NRT.
6. Write request/metrics provenance including SDK/model pins, profile hash, settings, actual duration/rate, elapsed time, and output hash. Keep bounded error output; avoid verbose per-chunk logging.

The parent worker checks completion/output, rechecks cancellation, and promotes a completed run into a unique kept directory. Remove partial directories on failure/cancellation on NRT. Return no sessions, subprocess handles, tensors, or open files to the engine. Initially each generation starts a fresh child; optimize only after measured need.

## Step 4: implement the two nodes

Both subclass `AudioCppJob`, use `OfflineJobWidget`, declare category `Offline`, and retain `TriggerPulseMixin` processing. Override fetch payload creation so it includes an empty runtime list and model-only specs. Never call `AudioCppRuntime` for readiness.

### Voice Designer contract

Inputs: audio `trigger_in`. Outputs: URI `voice_style`; mono audio `done`.

Parameters: `voice` (M1–M5/F1–F5), optional `voice_style_file` (overrides preset), `profile_name`, `model_dir`, transient `apply`, `download`, `cancel`, persisted `last_profile`. Set `RUN_PARAM="apply"` and suitable job tag/status labels. Apply snapshots controls, prepares the profile on NRT, publishes its URI, then arms the done pulse.

### TTS contract

Inputs: audio `trigger_in`, URI `text_uri`, URI `voice_style_uri`. Outputs: URI `audio`; mono audio `done`.

Parameters: multiline `text`, fixed `language` menu with 31 codes plus `na`, fallback `voice`, optional `voice_style_file`, `speed` 0.7–2.0, `total_steps` 1–100, `silence_duration` 0–5 seconds, `threads` 1–8, `model_dir`, transient `generate`, `download`, `cancel`, persisted `last_wav`.

Keep language/preset menu order fixed for saved-patch compatibility. Resolve profile precedence: connected `voice_style_uri`, file parameter, preset. Resolve text precedence: connected `text_uri`, inline text. Connected-empty inputs reject rather than silently use fallback. Snapshot the URI at dispatch; read files only on the worker.

### Acceptance and lifecycle rules

- Trigger edges queue engine parameter commands; never call param.set/sync or construct a job from `process()`.
- Publish file URI before arming one-block done pulse. Clear output every block; `(1, BLOCK_SIZE)` never changes.
- Associate each run with a request ID and cancellation token. Supersession gets existing NRT epoch protection. Cancel must additionally reject a successful result already queued before cancellation: it does not automatically advance the epoch. Capture immutable token/ID in the submitted spec rather than consulting worker-shared live node state.
- New Apply/Generate/Download, deletion, and reload invalidate publication rights. Stale progress cannot revert status or report completion. Follow existing progress tags and never submit another node NRT job just to report progress.
- Worker completion callbacks only install small metadata/paths and update status. No joins, I/O, hashing, or destructive cleanup in callbacks. Successful but stale kept artifacts can remain unpublished; remove partials worker-side.
- Failure/cancel retains the previous successful URI, emits no success pulse, and shows useful status.
- Save last paths and controls, not pending jobs; restore with graph attached and NRT artifact validation. Restore neither generates/downloads nor pulses/autoplays. Explicit Generate with a connected restored profile still works. Pending pulse and edge state clear on transport restart.

## Step 5: UI and working patch

Provide distinct labels for source selection, imported profiles, Apply voice, Generate speech, Download, Cancel, and status. Text editor changes must flow through the proxy queue and flush before Generate and Save; test the actual ordering rather than assuming an action hook guarantees it. Workers never access Qt.

Use normal node telemetry for active/busy/error state and kept artifact paths. Designer help must explain that imports load existing identities and that this is not reference-audio cloning. TTS help documents URI precedence and expression tags. Avoid SDK/internal details in ordinary controls.

Example patch: Designer `voice_style` → TTS `voice_style_uri`; Designer `done` → TTS `trigger_in`; TTS `audio` → SamplePlayer `uri_in`; TTS `done` → SamplePlayer `trigger_in`; SamplePlayer → output. Give TTS a short test sentence. Apply previews the selected voice when the engine runs. With engine stopped, Apply saves the profile but no block pulse propagates; document manual Generate/start behavior instead of adding a hidden control-thread auto-run path. Save/reload the patch and verify typed wires using the real schema.

## Step 6: verification and completion

Run all Python/pytest commands with `conda run -n anode-dev ...` or an activated `anode-dev` shell. Locate conda if absent from PATH. Direct invocation of the environment python.exe without activation is forbidden by AGENTS.md.

Use fake SDK/subprocess outputs for normal automated tests; no model/network downloads in CI. Test:

- Model-only pinned fetch and integrity/partial asset failures; missing SDK does not break node discovery.
- Imported profile shape/size/non-finite rejection, precedence, exact value preservation, unique artifact paths, honest version provenance.
- Immutable text/settings/URI snapshot, bounded file input, expression-tag preservation, waveform/duration handling, and no automatic fetch.
- Apply/Generate twice, cancellation after result queueing, Download supersession, deletion/reload, stale progress, worker failure/timeout and process reaping.
- Publication-before-pulse, one pulse only, zeroed later blocks, correct buffer shape, transport reset.
- Save/load graph attachment, missing artifacts, no accidental generation or pulse, pending UI text committed before Save/Generate.
- Full Designer → TTS → SamplePlayer chain with fake synthesized 44.1 kHz WAV: asynchronous load/autoplay, finite stereo 48 kHz output, EOF silence without stale audio.

Run focused new tests after each logical slice, then relevant existing tests (`test_sample_player.py`, `test_qwen3_asr_transcriber.py`, `test_download_util.py`, `test_audiocpp_subprocess.py`) and the full suite when practical. Report exact commands/results and any skipped checks. Do not duplicate existing supervisor tests unless the new integration changes its behavior.

Real-model acceptance, when assets/dependencies are available: English/Finnish, all ten preset previews, expression tags, multi-chunk text, imported known v3 style, cancelled inference, offline execution, and audio playback under CPU load. Record startup latency, memory, synthesis RTF, and underruns. Actual synthesis is not a prerequisite for mock lifecycle tests, but mark integration empirically unverified if real-model checks could not run.

The coder's final handoff must list changed files, setup/download commands, example patch usage, checks actually run, and remaining empirical limitations. Deliver the implementation locally; do not commit/push or send messages to another chat without authorization.
