# YuE2 Torch node, fully independent of audiocpp — implementation plan

## 0. Goal and locked decisions

New node type `YuE2TorchSongGenerator` in new file `plugins/yue2_torch_song_generator.py`,
in-process torch, BF16-only phase 1, old `YuE2SongGenerator` untouched. The new node
imports **nothing** from `audiocpp_backend`. Achieved by extracting a generic base first.

Locked: in-process (not child process), new node alongside old, BF16-only phase 1,
`max_duration` included from the start (240 s default / 30–900 s / `tokens=round(s*25)` /
warn+keep truncation). Deferred: tiled VAE, int8/kitchen/prefetch, LoRA, `abc_max_tokens`,
other families (BS-RoFormer → Qwen3-ASR → Aligner → SheetSage2), sidecar deletion.

## 1. Step 1 — extract `offline_job_base.py` (zero behavior change)

New top-level module `offline_job_base.py` (beside `audiocpp_backend.py`; add
`"offline_job_base"` to `plugin_system._PROJECT_MODULES`; lives outside `plugins/`
so never palette-registered per AGENTS.md). Imports only `base`, `download_util`,
stdlib (`json/logging/threading/pathlib/queue/shutil/tempfile`).

Move verbatim from `audiocpp_backend.py`:
- `__init__` (1010–1015), `submit_job` (1020–1026), `_cancel_job` (1028–1033),
  `remove` (1035–1036)
- `RUN_PARAM/RUN_TAG/RUNNING_STATES/ACTION_VERB` (1005–1008) + `on_ui_param_change`
  dispatch (1102–1157) incl. deliberate re-stage pattern
- `_restage` (1041–1043), `_fail` (1052–1056), `_need_engine` (1058–1063),
  `_busy_flag` (1095–1100)
- Hooks (names/signatures unchanged): `_build_spec`, `_running_status`,
  `_spec_warnings`, `_run_nrt`, `_on_job_submitted`, `_on_job_cancelled`,
  `_model_fetch_specs`, `_cleanup_discarded`
- Fetch orchestration: `_FETCH_PROGRESS_PCT/_BYTES` (968–972), `_spec_to_dict`
  (1162–1165), `_fetch_nrt` model leg + throttle closure (1198–1233),
  `_format_fetch_progress` (1243–1256), `_on_fetch_progress/_complete` (1258–1274)
- `_push_telemetry` (1276–1303), `_refresh_ui` (1305–1322)
- `new_run_dir` (1333–1338), `keep_artifacts` (1340–1349), `write_json` (1351–1355)
- `on_nrt_discarded → _cleanup_discarded` (1324–1328); keep tag strings
  `"fetch"/"fetch_progress"` byte-identical (consumed by `core.py`)
- `GenerationCancelled` → canonical `JobCancelled`, keep alias
- `missing_specs` helper

Extension points: `_runtime_fetch_payload()` / `_install_runtime()` hooks, default
empty; `_resolve_model_dir` default impl (convention: `params["model_dir"]` +
repo-root fallback), overridable.

`audiocpp_backend.AudioCppJob(OfflineJobBase)`: thin subclass adding only
`AudioCppRuntime` legs (`_build_fetch_payload` runtime leg 1171–1192,
`_fetch_nrt` runtime leg 1234–1240), `default_backend`, `_parse_metrics`,
all `_build_*_argv` + `run_*`, fetch specs, pins. Keep re-exports
(`from offline_job_base import …`) so existing tests/monkeypatches
(`backend.subprocess`, `backend._terminate`, `AudioCppRuntime`,
`AUDIOCPP_PIN_VERSION`, ABC helpers) pass unmodified.

ABC/lyrics pure utils (`abc_section_names`, `lyric_section_names`,
`score_lyrics_fit_warning`, `strip_abc_chords`, ~252–313): move to `abc_score.py` /
`lyric_fit.py` with re-exports from both backend modules (tests pin
`backend.*` names — update tests or re-export; prefer re-export in step 1,
migrate tests in step 3).

Acceptance step 1: full suite green, no user-visible change.

## 2. Step 2 — new node `plugins/yue2_torch_song_generator.py`

Module `yue2_torch_song_generator`; class `YuE2TorchSongGenerator(OfflineJobBase)`
defined in-file (registry rule `plugin_system.py:75-83`); widget
`YuE2TorchWidget` with `NODE_CLASS_NAME="YuE2TorchSongGenerator"` verbatim.
Imports: `offline_job_base`, `base`, `download_util`, `audio_io`, torch worker —
never `audiocpp_backend`. Category `Offline`, label `YuE2 Song Generator (Torch BF16)`.

Ports (same names, per-instance): `trigger_in`, `abc_uri`/`lyrics_uri`,
`song`/`plan` URIs, `ready` pulse (`PULSE_OUTPUT_NAME="ready"`).

Params (BF16-only): `style`, `lyrics_file`, `abc_file`, `format`, `cot`,
`guidance_scale`, `num_inference_steps`, `max_duration` (float s, 30–900,
default 240), `seed`/`auto_seed`, `model_dir="models/YuE2-Torch"`,
`generate/download/cancel`, `last_wav`/`last_plan`. No `profile`/`weight_type`/
LoRA/`abc_max_tokens`.

Worker (middle path, eager BF16, free-per-job): copied Comfy math
(`ldm/yue2/model.py`, `text_encoders/yue2.py` pure part, `ldm/audio/autoencoder.py`,
llama-subset, timestep_embedding) + ~100-line shims (ops→torch.nn,
attention→SDPA, prefetch→no-op). `_run_nrt`: lazy-load
`yue2_3b_bf16.safetensors` (~7.8 GB, Comfy-Org/YuE2, CC BY-NC 4.0) →
ABC `_generate` → semantic `_generate` (`tokens=max(1,round(s*25))`,
budget clamp `min(requested, 24576−prompt)`, `truncated` flag) →
`_acoustic_conditioning` chunked (CPU-staged KV) → NAR 8-step midpoint DiT per
`yue2_chunks` (cancel per chunk/step → `JobCancelled`) → full-song VAE decode
(spike; tiling phase 2) → `song.wav` float32 + `score.abc?` + `metrics`
(`rtf`, `seconds`, `truncated`) → existing `load_song_file`/`_keep_song`/
`request.json` flow. `on_nrt_complete`: `cut at Ns cap` warn+keep, URIs + pulse,
anti-ghost plan clear. `load_state` relinks file only (never auto-loads 7.8 GB).

Fetch: `YUE2_TORCH_HF_BASE` + `checkpoints/yue2_3b_bf16.safetensors` →
`models/YuE2-Torch/` via `DownloadSpec`/`missing_specs`/`fetch_all` +
`tools/yue2_torch/fetch_yue2_torch.py` + Download button; never auto-download;
`.gitignore` lines; provenance records torch+comfy-rev.

Guardrails: dedicated single-slot torch executor, VRAM preflight in
`_build_spec` (reject-only), `del+empty_cache+gc` retire via `submit_detached`,
epoch-guarded install + `_cleanup_discarded`, bounded telemetry. Accepted
residual: OOM sticky (survivable), segfault fatal — sidecar node stays fallback.

Deps (spike): existing `torch` (+CUDA build on Windows box), add
`safetensors>=0.4.2`, `tokenizers>=0.13.3` (conda-forge vs pip decided on target
box; this Linux box is CPU torch 2.8 — spike validated by pure tests here,
e2e GPU-gated there).

## 3. Step 3 — tests + docs

Pure (no GPU): tokens math (30→750, 240→6000, 900→22500), `distribution`
masks/`min_tokens`/penalty/`temperature==0`, `chunk_ranges` split + raise,
budget-clamp + `truncated`, `forward` frame/chunk-match error, spec validation,
save/load round-trip (new type key), QSpinBox int guard. New file
`tests/test_yue2_torch_song_generator.py` mirroring existing patterns
(fixture `NODE_REGISTRY.get`, `sys.modules["yue2_torch_song_generator"]` live
helper). Step-1 extraction covered by unchanged full suite; migrate
`backend.*` test imports to new homes here if re-export was used.

Docs: node `description` (torch BF16, cap semantics, CC BY-NC-4.0),
param helps, fetch README. Run on Windows box:
`conda run -n anode-dev pytest tests/test_yue2_torch_song_generator.py -q`,
then full `conda run -n anode-dev pytest -q`.

## 4. Out of scope

Tiled VAE, int8/kitchen/aimdo/graphs, lowvram modes, LoRA, `abc_max_tokens`,
SheetSage2/Qwen/BS-Roformer ports, sidecar deletion, MP3/pulse semantics changes.

## 5. Acceptance

- [ ] New node appears alongside old; old untouched; new imports nothing
  audiocpp (`grep audiocpp_backend plugins/yue2_torch_song_generator.py` empty,
  add CI grep).
- [ ] Step-1 suite green with zero behavior change.
- [ ] Short `cot=off` BF16 render + relink pass GPU-gated; truncation shows
  `cut at cap`, song kept; missing `semantic.json` never fails job.
- [ ] Old patches open old node; new patches round-trip new node.
