# Supertonic 3 implementation review

Reviewed 2026-10-11. Implementation inspected locally. Independently ran the three new test modules through activated `conda run -n anode-dev`: **65 passed in 5.47s**. The full-suite result of 1040 passed / 3 skipped is the coder's report, not independently rerun in this review. No real-model inference performed. No implementation files modified.

## Changes required

### P1: coordinate downloads targeting the same model directory

Both nodes' `_build_fetch_payload()` return the full shared manifest and inherit `_fetch_nrt()` without destination coordination. `download_util._download_one()` uses one fixed `<dest>.part` path per asset; `_attempt()` opens it in `wb`/`ab` and later unlinks or replaces it. Pressing Download on the designer and TTS at the same time, or restarting a download before the cancelled worker has exited, lets independent workers write/truncate/rename the same partial file. This can fail verification, erase resumable progress, or cause a missing-part error. Small size-only assets can also be accepted without content integrity after overlapping writes.

Fix with narrow NRT-side coordination for the resolved model destination, keeping lock waits away from engine/control callbacks and cancellation responsive. Coordinate the CLI as well if claiming concurrent process safety. Recheck readiness after obtaining ownership. Add a deterministic two-worker test using fake downloads and a barrier; ensure no two writers own the same `.part` simultaneously and cancellation doesn't damage the other download.

Locations: `plugins/supertonic3_tts.py:271`, `plugins/supertonic3_voice_designer.py:185`, `download_util.py:112` and `:163`.

### P2: reject failed completions after Cancel, not only successful ones

In both `on_nrt_complete()` branches, the request identifier is read only from a dictionary payload. NRTExecutor delivers worker exceptions as exception objects without a request ID. Cancel clears `_active_request_id` but does not advance the NRT epoch. If an ordinary exception has already been queued, or the worker throws an ordinary error during cancellation, the current-epoch failure still reaches `_fail()` and replaces Cancelled with Error. The existing request-ID guard cannot reject it because `request_id is None`.

Preserve request identity on terminal failures, or reject failure callbacks when no request is still active (accounting for the executor's epoch filtering). Test actual submission/cancel/drain ordering with an ordinary queued exception, for both nodes. A cancelled job must retain Cancelled, previous URI, and no pulse. Review equivalent cancellation handling for fetch progress and restore jobs, which currently have no request-ID guard.

Locations: `plugins/supertonic3_tts.py:364`–`:383`, `plugins/supertonic3_voice_designer.py:251`–`:273`.

### P2: replace tests that bypass the integration being claimed

No new test calls `tools/supertonic3/worker.py:main()`; synthesis tests replace the entire subprocess with a function that writes a fixture WAV. They verify parent artifact handling but not SDK arguments, local-only construction, worker voice/profile resolution, waveform validation, or metrics writing. A fake SDK can exercise all these without downloading weights.

`test_run_nrt_no_automatic_fetch()` ends with `assert called == [] or True`, which always passes; it doesn't establish that fetching was avoided. `test_worker_failure_and_timeout_reap()` simulates one RuntimeError, never a timeout or process reaping. Existing supervisor tests can cover reaping, but the test name/report should accurately describe coverage.

`test_designer_tts_player_chain()` manually assigns both URI outputs, calls the SamplePlayer loader directly, and sets `_is_playing = True`. It never processes the designer/TTS completion pulses or applies the resulting engine command. Thus it does not verify Apply → Generate → asynchronous load/autoplay, which is the main user workflow.

Add worker-main tests using fake SDK/local helpers and an engine-command/NRT-drain chain test with fake synthesis. Assert playback begins without directly setting `_is_playing`, and that publication precedes pulse delivery. Add the planned Save/editor flush test using the real proxy/controller boundary; the current widget test uses a fake proxy and checks only call order.

Locations: `tests/test_supertonic3_tts.py:303`, `:331`, `:334`, `:437`, `:498`.

### P2: remove unsupported non-commercial license claims

Implementation docstrings, node descriptions, root README, fetch CLI, and tools README call OpenRAIL-M non-commercial; the tools README additionally says commercial use requires separate permission. OpenRAIL-M is not itself a non-commercial license. The original published Supertonic 3 license grants use/distribution rights subject to its specified restrictions, without a blanket non-commercial condition.

Use accurate wording: model weights are under OpenRAIL-M; see the distributed license for restrictions and obligations. Verify the precise archived LICENSE before making any different claim about the pinned archive. The pinned archive license could not be retrieved through the review's web tool, so do not substitute an assumption about it.

Source inspected: https://huggingface.co/Supertone/supertonic-3/blob/main/LICENSE

Examples: `supertonic3_backend.py:20`, `tools/supertonic3/README.md:71`, `README.md:64`, both plugin descriptions, and fetch.py.

## Additional plan gaps

- Small config/style/license files use size-only verification despite the plan asking for content hashes. Obtain hashes from the pinned small files and add them to the manifest; schema/length alone cannot detect same-size corruption or changes to voice identity.
- `load_state()` performs filesystem existence checks on the engine/control boundary instead of always delegating artifact validation to NRT. It also leaves saved transient trigger parameters unchanged. Move the checks to relink workers and explicitly reset Apply/Generate/Download/Cancel on restore. Fresh construction prevents most accidental runs today, but it does not fulfill the planned restoration contract.
- The bounded readers stat the file and then call unbounded `read_bytes()`. Read at most cap+1 bytes so a file growing after stat or an unusual source cannot exceed the advertised memory bound.

## What is sound

Heavy SDK inference is isolated in a child; CPU thread limits and `auto_download=False` match the pinned SDK. URI precedence, unique profile paths, mono completion buffers, use of existing SamplePlayer, and keeping inference/file processing outside `process()` fit ANode's architecture. The pinned SDK's `save_audio()` writes the returned waveform directly, so this review found no required duration-trimming change.

Resolve the findings and rerun focused checks before accepting the integration. Then perform real-model English/Finnish/preset and preview-chain acceptance when assets are available; throughput, quality, and audio-load behavior remain empirically unverified.
