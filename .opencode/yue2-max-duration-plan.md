# YuE2 `max_duration` (seconds) — implementation plan for coding agent

## 0. Goal and locked decisions

Add a user-facing **song length cap in seconds** to the YuE2 offline node, mirroring ComfyUI semantics, to bound peak VRAM on 8 GB Windows (RTX 5070, WDDM) and avoid the audio.cpp whole-arena paging cliff.

Locked by user:

- UI unit: **seconds (float)**, not tokens.
- Default: **240 s** (fits most songs; no back-compat freeze required — old uncapped behavior ~360 s may change).
- Range: **30–900 s** (Comfy allows 0.04–900; songs need ≥30 s; hard upper bound 900 s matches Comfy).
- Truncation policy: **warn + keep** (Comfy semantics). A song that hits the cap is kept with a visible `(cut at cap)` note, not an error.
- `abc_max_tokens`: **out of scope** for this change. Leave ABC planning at CLI default (4096). Only cap the semantic/music stage.

Background (do not re-investigate; already confirmed):

- Comfy `YuE2GenerateMusic(max_duration 360 s default, max 900)` → `max_tokens = round(seconds * 25)` (`FRAMES_PER_SECOND = 25`, `comfy/text_encoders/yue2.py`), `min(requested, CONTEXT 24576 − prompt_len)`, auto-log when reduced, `(history, truncated)` flag; `EmptyYuE2LatentAudio` sizes `(B,64,seconds*25)` from actual `yue2_frames`.
- ANode today sends no `semantic_max_tokens` (`audiocpp_backend.py:_build_yue2_argv`, `plugins/yue2_song_generator.py:_build_spec`), so every run uses the CLI default (~9000 tokens ≈ 360 s) with full KV + NAR + full-song VAE peak.
- Root perf difference (confirmed by user observation): Comfy streams layers/chunks over PCIe (linear slowdown, Shared RAM usage far above 8 GB but finishes); audio.cpp pages whole preallocated ggml CUDA arenas via WDDM Shared fallback (cliff + stalls). This cap **reduces how often we page**; it does not re-architect ggml paging. Arena-knob exposure (`*_arena_mb`, `vae_weight_type`, `attention_tile_rows`) is an explicitly separate follow-up, not part of this task.

## 1. Conversion contract (single source of truth)

- `YUE2_FPS = 25` (semantic frames per second; matches Comfy `FRAMES_PER_SECOND` and upstream 25 fps codec).
- `tokens = max(1, round(seconds * 25))`. Examples: 30 s → 750; 240 s → 6000; 900 s → 22500.
- `CONTEXT = 24576` (YuE2 `max_position_embeddings`). Effective budget is `min(tokens, CONTEXT − prompt_len)`; the CLI/sidecar enforces the context fit (same as Comfy). ANode does **not** pre-compute prompt length (no tokenizer on the control thread); it sends the requested value and surfaces any CLI-side reduction/truncation.
- Transmit as upstream request option `semantic_max_tokens` (see `model_specs/yue2.json`: `semantic_max_tokens`, `min 1`, plus `export_semantic` flag). Always send it explicitly (new default 240 s → 6000 is a deliberate behavior change; do not use the omit-when-default pattern here).

## 2. File changes

### 2.1 `audiocpp_backend.py`

1. Add module constant near `YUE2_ATTENTION_MODES` (~line 804):
   `YUE2_FPS = 25` with comment (semantic codec frames/sec; Comfy parity).
2. `_build_yue2_argv(...)` (~line 816):
   - New kwargs: `semantic_max_tokens=None, export_semantic=False`.
   - Validate: `semantic_max_tokens` must be `int` (not bool) and `>= 1`, else `ValueError`. `export_semantic` is bool.
   - Emit when set: `--request-option semantic_max_tokens={N}`; when `export_semantic` true: `--request-option export_semantic=true`.
   - Keep pure function, no process/env changes. Existing `--threads 8`, `--backend`, weight/attention handling untouched.
3. `run_yue2_gen(...)` (~line 903):
   - New kwargs: `semantic_max_tokens=None, export_semantic=True` (default True for YuE2 so truncation is observable; the CLI only writes `semantic.json` under `--out-dir` when the flag is set — verify against `audio.cpp/docs/models/yue2.md` at implementation time).
   - Pass through to `_build_yue2_argv`.
   - After success (after existing `score.abc` handling ~lines 959–963): read `out_dir/semantic.json` **best-effort, never fail the job if missing/unparseable** (older CLI, `cot` edge cases). Parse tolerant shape: docs describe a flat int array payload with meta `frames` + `truncated`. Implement helper `_read_semantic_info(out_dir)` returning `{"frames": int|None, "truncated": bool|None}`; on any `OSError`/`ValueError`/`KeyError` return `{"frames": None, "truncated": None}`.
   - Extend return dict: `{"wav", "metrics", "score", "semantic"}` where `semantic` is the dict above. Update docstring.
   - Do not change launch/monitor logic (`run_audiocpp_subprocess`, `_terminate`), timeouts, or metrics parsing.
4. Add/extend pure argv-builder unit coverage (see §3). No GPU needed.

### 2.2 `plugins/yue2_song_generator.py`

1. Constants (~lines 58–62): add `DEFAULT_MAX_DURATION = 240.0`, `MIN_MAX_DURATION = 30.0`, `MAX_MAX_DURATION = 900.0`. Import `YUE2_FPS` from `audiocpp_backend`.
2. `YuE2Widget.PARAM_KEYS` (~line 102): add `"max_duration"` in a sensible order (after `"cot"`, before LoRA keys). No widget-code change needed (auto float widget).
3. `__init__` params (~line 221, after `num_inference_steps` / before `attention`):
   `self.add_float_param("max_duration", DEFAULT_MAX_DURATION, MIN_MAX_DURATION, MAX_MAX_DURATION, unit="s", help="Max song length in seconds (upper bound; the song may end earlier and is cut with a warning if it needs longer). Shorter caps use less VRAM.")`
4. `_build_spec` (~line 330):
   - Read `seconds = float(self.params["max_duration"].value)`; validate finite and `30 <= seconds <= 900` else `ValueError` with clear message.
   - `tokens = max(1, round(seconds * YUE2_FPS))`; put `"max_duration": seconds, "semantic_max_tokens": tokens` in spec.
5. `_running_status` (~line 369): append cap, e.g. `f"… , cap {seconds:.0f}s"`.
6. `_run_nrt` (~line 419): pass `semantic_max_tokens=spec["semantic_max_tokens"]` (and rely on `run_yue2_gen` default `export_semantic=True`) through to `run_yue2_gen`. Capture `gen.get("semantic")` and include `"semantic": gen.get("semantic") or {"frames": None, "truncated": None}` plus `"max_duration": spec["max_duration]`, `"semantic_max_tokens": spec["semantic_max_tokens"]` in the returned dict.
7. `_keep_song` (~line 481 `request.json`): add `"max_duration": spec.get("max_duration")`, `"semantic_max_tokens": spec.get("semantic_max_tokens")`, `"semantic": <gen semantic info or None>`. Keep all existing keys unchanged (saved-patch provenance compat).
8. `on_nrt_complete` `tag == "gen"` (~line 521):
   - Read `result.get("semantic") or {}`; `truncated = sem.get("truncated") is True`.
   - Status detail: existing `f"{secs:.1f} s song (seed …, cot…)"` plus `", cut at {max_duration:.0f}s cap"` when truncated, else unchanged. Keep `RTF`/`+ plan` suffix logic. Never error on truncation (warn + keep). `logger.warning` when truncated.
   - Telemetry (`get_telemetry`, ~line 624): include truncation note in `detail` the same way; keep `busy`/`score_text` behavior.
9. `load_state` (~line 599): old patches lack `max_duration` — after `super().load_state(data)`, `if "max_duration" not in self.params:` cannot happen (param always registered); instead guard value validity: if stored value missing/out of range (e.g. `0` from older default), reset to `DEFAULT_MAX_DURATION` via set+sync. Simplest robust form:
   ```
   try: v = float(self.params["max_duration"].value)
   except ...: v = DEFAULT...
   if not (MIN... <= v <= MAX...): self.params["max_duration"].set(DEFAULT...); sync()
   ```
10. `description` (~line 143): add one sentence: `Max duration caps the song length (default 240 s; the song may end earlier and is cut with a warning if longer) and bounds peak VRAM.` Keep all existing text (patch-compat for docs/tests asserting substrings like `"audio.cpp"`).

### 2.3 UI registration

No extra UI code: `PARAM_KEYS` addition is sufficient. Verify the auto float widget renders with `unit="s"` (same pattern as `guidance_scale`).

## 3. Tests (Windows-runnable, no GPU)

Target files: `tests/test_yue2_song_generator.py`, `tests/test_audiocpp_subprocess.py`.

1. **Conversion**: 30 s→750, 240 s→6000, 900 s→22500 tokens; rounding check (e.g. 120.5 s→3012 or 3013 — assert `round()` semantics, pick one and document).
2. **Argv builder**: `semantic_max_tokens=6000` emits exactly `--request-option semantic_max_tokens=6000`; `export_semantic=True` emits `--request-option export_semantic=true`; both absent by default-when-None (backend-level default preserved); invalid (`0`, `-1`, `True`, `3.5`, `"6000"`) raises `ValueError`.
3. **`run_yue2_gen` with stubbed subprocess** (existing stub pattern): `semantic.json` present → `result["semantic"] == {"frames":…, "truncated":…}`; missing/unparseable → `{"frames": None, "truncated": None}` and job still succeeds.
4. **Node spec**: `_build_spec` maps 240 s→6000; out-of-range/NaN/inf raises `ValueError`; `request.json` contains the three new keys.
5. **`on_nrt_complete`**: truncated True → status contains `cut at 240s cap`, URIs still published, no pulse lost; truncated None/False → old format unchanged.
6. **Save/load**: node with `max_duration=120`, `to_json` → new node → `load_state` keeps 120; old patch dict without the key loads with 240.0 and does not raise.
7. Keep existing GPU-gated test (`ANODE_YUE2_GPU=1`) untouched.

Run on the Windows box per `AGENTS.md §14` (conda env is source of truth; `conda` may not be on PATH in non-interactive shells):

```
conda run -n anode-dev pytest tests/test_yue2_song_generator.py tests/test_audiocpp_subprocess.py -q
```

then the full suite when practical: `conda run -n anode-dev pytest -q`.

## 4. Manual verification (RTX 5070 8 GB, Windows)

1. Generate a previously-failing long song at 240 s cap; confirm completion + `cut at cap` note when applicable, and `request.json`/`metrics.json` kept.
2. Compare Task Manager Dedicated vs Shared against the old uncapped run; expect lower peak / no grind-to-halt for typical songs.
3. Short-song regression: 60–90 s song unaffected in length/quality.

## 5. Explicitly out of scope

- `abc_max_tokens` UI, arena `*_mb` knobs, `vae_weight_type`, `attention_tile_rows`, backend/offload changes, sidecar version bump (`AUDIOCPP_PIN_VERSION`), MP3/WAV handling, trigger/pulse semantics.

## 6. Acceptance criteria

- [ ] `max_duration` float param visible (30–900 s, default 240 s, unit `s`), in `PARAM_KEYS`, help text set.
- [ ] CLI receives `semantic_max_tokens=round(s*25)` + `export_semantic=true`; builder tests pass.
- [ ] Truncation surfaces as `cut at Ns cap` warning in status/telemetry, song kept, URIs + pulse intact; missing `semantic.json` never fails the job.
- [ ] `request.json` records `max_duration/semantic_max_tokens/semantic`; old patches load with 240 s default.
- [ ] Target test files + full suite green on the Windows box.
