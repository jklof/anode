# YuE2 torch worker (self-contained) + old-node engine switch — implementation plan

## 0. Background: why replace audio.cpp for YuE2

ANode generates songs via an **external C++ sidecar**: `tools/audiocpp/` ships a
pinned `audiocpp_cli` binary (v0.8.1, Windows x64 + CUDA track) that runs the
YuE2-3B model from GGUF files. The `YuE2SongGenerator` node (`plugins/
yue2_song_generator.py`) launches one sidecar subprocess per song on an NRT
background worker and keeps the finished WAV plus provenance (`request.json` /
`metrics.json`). Audio.cpp preallocates large contiguous CUDA arenas (weights +
KV cache + graph workspaces, ~20 GiB reserved by default). On an 8 GB Windows
GPU (RTX 5070, WDDM driver model) a long song spills into WDDM shared memory
and generation **grinds to a halt** (page-fault thrashing per token), while
ComfyUI's native PyTorch YuE2 path on the same card spills gracefully and
finishes in reasonable time — it streams layers/chunks over PCIe (roughly
linear slowdown) instead of paging whole arenas.

This plan builds a **self-contained PyTorch YuE2 worker** replicating ComfyUI's
streaming architecture (chunked acoustic conditioning, per-chunk diffusion,
small VRAM-resident working set), integrated through the existing node behind an
engine switch. The sidecar stays as the default until torch proves out.

Key terms for readers new to this repo:
- **NRT worker**: background thread-pool job (`NRTExecutor`, `core.py`) for
  minutes-long blocking work. The audio thread never does I/O, model loading,
  or subprocess supervision (`AGENTS.md` §§4–6).
- **`AudioCppJob`** (`audiocpp_backend.py`): node base class owning the
  generate/cancel/download trigger dispatch, NRT submit, fetch-with-progress,
  temp-dir/keep-dir artifact helpers, and telemetry push. Subclasses implement
  `_build_spec` (validate params → worker spec dict) and `_run_nrt`
  (blocking worker) plus an `on_nrt_complete` branch.
- **Spec/run-dir/keep flow**: `_build_spec` snapshots UI params;
  `_run_nrt(cancel_event, spec)` runs in a temp `run_dir`, then moves exactly
  one song file plus `plan.abc`/`request.json`/`metrics.json` into a
  timestamped keep dir; `on_nrt_complete` publishes `song`/`plan` URI outputs
  and fires a one-block `ready` pulse. Cancel is a `threading.Event` set from
  the control thread; the worker polls it and the supervisor terminates the
  child process (`_terminate`: terminate → wait 5 s → kill).
- **ComfyUI**: reference PyTorch implementation of YuE2, merged upstream Sep
  2026 (PR `Comfy-Org/ComfyUI#16250`, `yue2` branch). Source of the vendored
  math (see §1). Upstream model: `m-a-p/YuE2-3B` (Apache-2.0 code, CC BY-NC-4.0
  weights); Comfy repack used here: `Comfy-Org/YuE2` file
  `checkpoints/yue2_3b_bf16.safetensors` (~7.8 GB, AR + NAR + VAE + tokenizer
  in one file).

Prior work already shipped (do not redo):
- `max_duration` song-length cap: float seconds param (30–900 s, default
  240 s) → CLI `semantic_max_tokens = max(1, round(seconds * 25))`
  (`YUE2_FPS = 25`, Comfy parity); truncation is warn+keep with a
  `cut at Ns cap` note; `semantic.json` (`{frames, truncated}`) parsed
  best-effort. The torch worker implements the same semantics natively.
- Truncation/`semantic.json` plumbing (`_read_semantic_info`, `request.json`
  `semantic` key) is engine-agnostic — reuse as-is.

## 1. Goal and locked decisions

Build a standalone PyTorch YuE2 worker and expose it through the **existing**
`YuE2SongGenerator` node via an `engine` menu (no second node type). The worker
imports **nothing** from ANode modules, runs without the engine, and speaks a
CLI-compatible subset of `audiocpp_cli` so the node change is an argv-target swap.

Locked:
- **Old node, engine menu**: `engine` = `sidecar (GGUF)` (default, current
  behavior) vs `torch (BF16)` (opt-in). Default stays `sidecar` until torch
  proves out; flip later.
- **Worker layout, simple first**: `tools/yue2_torch/` + single model dir
  `models/YuE2-Torch/` holding `yue2_3b_bf16.safetensors` (CC BY-NC 4.0,
  fetch-at-runtime, never bundled). No Comfy `checkpoints/` + `audio_encoders/`
  mirroring until SheetSage2 enters scope.
- **Dependencies**: worker may use anything in the shared conda env plus
  justified additions. Concrete adds: `safetensors>=0.4.2`,
  `tokenizers>=0.13.3`; `tqdm`, `einops`, `numpy` OK. No `comfy-kitchen` /
  `comfy-aimdo` / triton in phase 1 (eager BF16 + `torch.sdpa` only).
- **Phase 1 scope**: BF16, eager, full-song VAE decode. Deferred: tiled VAE,
  int8, prefetch/CUDA-graphs, lowvram/offload modes, LoRA adapters,
  `abc_max_tokens`, other model families (BS-RoFormer → Qwen3-ASR → Aligner →
  SheetSage2), sidecar deletion, `offline_job_base` extraction (unneeded while
  only one family migrates — the node stays on `AudioCppJob`).

## 2. Worker: `tools/yue2_torch/` (zero ANode imports)

```
tools/yue2_torch/
  run.py        # CLI entry (argparse only here): parse args, run pipeline,
                # write outputs, print metrics.*, exit codes. No torch import
                # at module top (fast --help); die promptly on SIGTERM
                # (process death is the cancel mechanism).
  cli.py        # PURE builder: spec dict -> argv list. No torch/heavy imports,
                # so the node + tests import it without a GPU.
  pipeline.py   # stage orchestration: plan → semantic → acoustic → vocode.
                # Owns the NAR sampler: midpoint-Euler flow matching,
                # num_inference_steps=8 default (sidecar parity), deterministic
                # torch.Generator(seed) noise per chunk.
  ar.py         # vendored Comfy math (GPL headers kept — GPL is the preferred
                # license for this work): consts (EOD=151643, ABC_START/END=
                # 151847/151848, MUSIC_START/END=151851/151852, CODEC_OFFSET=
                # 151853/SIZE=32768, CONTEXT=24576, FPS=25, INSTRUCTIONS),
                # distribution(), chunk_ranges() (size=(ctx-prefix-3)//2,
                # raise if <1 frame fits), YuE2Tokenizer, AR generate loop
                # (plain KV tuples, dual-batch CFG with 1.01 default for
                # cot=off else 1.0, repetition penalty, min_tokens 32 ABC /
                # 200 semantic, penalty windows 100 / 50, legacy_off for
                # cot=off). External-ABC conditioning: prefix construction
                # must support abc_file input, not just generated plans.
  dit.py        # vendored NAR DiT (comfy/ldm/yue2/model.py: model_config vocab
                # 184704/hidden 2048/inter 6144/28L/16Q+8KV/merged_qkv+mlp;
                # per-chunk forward: pad, vae2llm+time+pos, rope, prefix
                # permute, layer loop, llm2vae slice). Prefetch calls stripped.
  vae.py        # vendored Oobleck VAE (full-song decode phase 1; tiled later).
  llama.py      # extracted subset only: Qwen3_8BConfig, RMSNorm, merged-qkv
                # Attention, merged-mlp SwiGLU MLP, TransformerBlock, minimal
                # Llama2_ (init_kv_cache/forward/lm_head), RoPE helpers,
                # plain FixedKV. Other configs, graphs, VBAR, spec-decode
                # stay out.
  fetch.py      # STDLIB-ONLY downloader (urllib + hashlib + Range resume +
                # size/sha verify) for without-Anode use. Node-side fetch
                # reuses repo machinery (see §4); ~80 lines duplicated
                # deliberately — extractability over DRY.
  README.md     # env notes, CLI reference, model license (CC BY-NC 4.0).
```

Comfy sources to vendor from (pin the exact revision in a header comment when
copying; re-check `distribution`/`chunk_ranges`/config constants against it):
`comfy/text_encoders/yue2.py`, `comfy/ldm/yue2/model.py`,
`comfy/ldm/audio/autoencoder.py` (Oobleck VAE), relevant subset of
`comfy/text_encoders/llama.py`, all at `Comfy-Org/ComfyUI` `master`
(post-#16250). Keep Comfy/GPL and original adaptation headers (M·A·P YuE2
Apache-2.0 note, stable-audio-tools note) intact.

CI grep enforced: any `import base|core|audiocpp_backend|audio_io|plugin_system`
under `tools/yue2_torch/` fails the build.

### CLI contract (drop-in subset of `audiocpp_cli`)

```
run.py --model <dir> --lyrics <text> --request-option style=... \
  --request-option cot=off|melody|full [--request-option abc_file=...] \
  --request-option semantic_max_tokens=N [--request-option guidance_scale=G] \
  [--request-option num_inference_steps=K] --seed N \
  --out song.wav --out-dir run_dir --metrics
```

Strict option parsing: unknown `--request-option` keys fail loudly (catches
typos; mirrors sidecar strictness). `export_semantic` needs **no flag**: the
worker always writes the tiny `semantic.json` under `--out-dir`.

Stdout: `metrics.<k>=<float>` lines (parsed by the existing `_parse_metrics`)
plus `progress.stage=<load|plan|acoustic|vocode>` heartbeats (log-only for now;
supervisor stage timeouts later). Files: `song.wav` (float32 PCM via stdlib
`wave`, 48 kHz stereo = native VAE rate, **no resampling** in worker),
`score.abc` when the model planned (cot=melody/full without input abc),
`semantic.json` (`{frames, truncated}`).
Exit codes: 0 ok / 2 usage / 3 model-missing / 4 CUDA OOM / 1 other.

## 3. Node changes (`plugins/yue2_song_generator.py` only)

- New `engine` menu param (`["sidecar (GGUF)", "torch (BF16)"]`, default 0) +
  `torch_model_dir` string param (default `models/YuE2-Torch`). Both added to
  `YuE2Widget.PARAM_KEYS` (engine next to `profile`, model dir next to
  `model_dir`). `profile`/`weight_type` help text gains "(sidecar only)".
  Download button label must reflect the active engine's payload
  (sidecar `~3.8 GB` vs torch `~7.8 GB`).
- `_build_spec`: branch on engine. **Sidecar path byte-identical to today.**
  Torch path: same validation rules for style/lyrics/abc/cot/seed/guidance/
  steps/`max_duration`; `tokens=max(1,round(s*25))`; resolve torch model dir +
  require `yue2_3b_bf16.safetensors` (missing → ValueError with fetch hint).
  Spec carries `engine="torch"`, worker python (`sys.executable` — same shared
  conda env, no second interpreter to discover), `tools/yue2_torch/run.py`,
  `semantic_max_tokens`. LoRA params on the torch path raise
  `ValueError("…not supported on the torch engine yet")` — never silently
  ignore them. `profile`/`weight_type`/`attention` are sidecar-only and ignored
  on the torch path (help text says so; eager SDPA is always used phase 1).
- `_run_nrt`: branch. Torch path writes `run_dir/spec.json` (lyrics/style text
  by file, not argv — avoids OS command-line limits), builds argv via worker
  `cli.py` builder, then supervises with the **same** poll/cancel/timeout/
  terminate/tail semantics as `run_audiocpp_subprocess`. That runner hardcodes
  two `"audio.cpp"` strings (launch-failure and nonzero-exit errors), so
  generalize them with the existing `task_label` mechanism (or a small
  `prog` parameter) as part of this change — all five sidecar families keep
  passing their current labels, tests updated accordingly. Tail then identical:
  `load_song_file` → `plan_text` → `_keep_song` (+`semantic`) → return dict
  (+`engine` key for status/provenance).
- `_keep_song` `request.json`: add `engine`; torch path records
  `runtime: {torch: <version>, comfy_rev: <sha>, model_sha: <…>}` instead of
  the `audio_cpp` pin. Keep dir: torch outputs under
  `models/YuE2-Torch/outputs/` (per-engine dirs avoid cross-backend confusion).
- `on_nrt_complete` / `get_telemetry` / `load_state`: engine-agnostic already
  except status text — append `(torch)` or `(sidecar)` to `_status_detail` so
  VRAM reports are attributable. Extend the `load_state` reset-to-default
  guard to the new params (same pattern as `max_duration`). Old patches
  without the `engine` key load as `sidecar`.
- `_model_fetch_specs`: branch — sidecar returns `yue2_fetch_specs`, torch
  returns new `yue2_torch_fetch_specs` (same `DownloadSpec` machinery,
  `missing_specs` verify, throttled progress, never auto-download).
- Description: one sentence for the engine switch + BF16 VRAM note.

## 4. Fetch

- Node path (primary): new `YUE2_TORCH_*` specs in `audiocpp_backend.py` next
  to the existing ones + a `tools/audiocpp/fetch_yue2_torch_gguf.py`-style
  sibling script. Reuses `DownloadSpec`/`fetch_all`/resume/verify/progress.
- Standalone path: `tools/yue2_torch/fetch.py` (stdlib only) for without-Anode.
- `.gitignore`: `models/YuE2-Torch/*.safetensors`, tokenizer cache, `outputs/`.

## 5. Tests

- Worker-standalone (no engine fixtures, no GPU) in new
  `tests/test_yue2_torch_worker.py`: `tokens` math, `distribution`
  masks/`min_tokens`/penalty/`temperature==0`, `chunk_ranges` split + raise,
  budget-clamp + `truncated`, DiT frame/chunk-match error, `cli.py` builder
  (flags emitted/omitted/rejected), `fetch.py` URL/sha table, zero-Anode-import
  CI grep test.
- Node (stubbed Popen, existing pattern in `tests/test_yue2_song_generator.py`):
  engine menu defaults sidecar; sidecar spec/argv byte-identical to today;
  torch spec maps 240 s→6000 with argv targeting `run.py` + spec file; torch
  `on_nrt_complete` truncation note + `(torch)` tag; LoRA-on-torch raises;
  save/load round-trip with new params; old patch (no `engine` key) loads as
  sidecar.
- GPU-gated e2e (`ANODE_YUE2_TORCH_GPU=1`, Windows box): short `cot=off` torch
  render, relink, truncation note. Skipped in CI here.

Run: `conda run -n anode-dev pytest tests/test_yue2_torch_worker.py tests/test_yue2_song_generator.py -q`, then full suite. (`conda` may not be on PATH in
non-interactive shells; `README.md`/`environment.yml` are the source of truth
for the env name — never invoke the env python by absolute path, it breaks
native-library discovery.)

## 6. Out of scope

Tiled VAE, int8/kitchen/aimdo/graphs, lowvram/offload modes, LoRA,
`abc_max_tokens`, SheetSage2/Qwen/BS-Roformer ports, sidecar deletion,
`offline_job_base` extraction (unneeded — node stays on `AudioCppJob`;
revisit when the second family migrates), MP3/pulse semantics.

## 7. Acceptance

- [ ] `tools/yue2_torch/run.py --help` works with only the package dir present
  (no engine imports); zero-Anode-import CI grep green.
- [ ] Node defaults unchanged (sidecar, same argv as today — covered by
  existing tests); torch opt-in renders short `cot=off` song GPU-gated with
  `cut at cap` + `(torch)` note.
- [ ] LoRA + torch raises loudly; old patches load as sidecar; new engine
  choice round-trips save/load.
- [ ] Target test files + full suite green on the Windows box.
- [ ] No weights committed; fetch-at-runtime only; provenance in `request.json`.
