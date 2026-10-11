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
  `checkpoints/yue2_3b_bf16.safetensors` (~7.8 GB: AR + NAR + VAE weights
  plus the HF tokenizer.json embedded as the
  `text_encoders.yue2_tokenizer_json` tensor — verified in the file).

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
imports **nothing** from ANode modules, runs without the engine, and is driven
via a `spec.json` file (lyrics/style text by file, not argv — avoids OS
command-line limits), so the node change is an argv-target swap from the
sidecar CLI to `tools/yue2_torch/run.py`.

Locked:
- **Old node, engine menu**: `engine` = `sidecar (GGUF)` (default, current
  behavior) vs `torch (BF16)` (opt-in). Default stays `sidecar` until torch
  proves out; flip later.
- **Worker layout, simple first**: `tools/yue2_torch/` + single model dir
  `models/YuE2-Torch/` holding `yue2_3b_bf16.safetensors` (CC BY-NC 4.0,
  fetch-at-runtime, never bundled — tokenizer embedded, no second file).
  No Comfy `checkpoints/` + `audio_encoders/`
  mirroring until SheetSage2 enters scope.
- **Dependencies**: worker may use anything in the shared conda env plus
  justified additions. Concrete adds: `safetensors>=0.4.2`,
  `tokenizers>=0.13.3`; `tqdm`, `einops`, `numpy` OK (add the missing ones to
  `environment.yml` — none of `safetensors`/`tokenizers`/`einops`/`tqdm` are
  in there today). No `comfy-kitchen` /
  `comfy-aimdo` / triton in phase 1 (eager BF16 + `torch.sdpa` only).
- **Phase 1 scope**: BF16, eager, full-song VAE decode, **coarse sequential
  stage loading** (AR → `to('cpu')`/`del` + `empty_cache()` → DiT chunk loop →
  `to('cpu')`/`del` + `empty_cache()` → VAE decode). All ~7.8 GB resident at
  once exceeds 8 GB WDDM with KV/activations (the exact thrash §0 diagnoses),
  so this swap is required phase 1, not a later optimization. Deferred: tiled
  VAE, per-layer streaming, int8, prefetch/CUDA-graphs, lowvram/offload modes,
  LoRA adapters,
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
  pipeline.py   # stage orchestration: tokenize → ABC plan → semantic →
                # KV-conditioning prefill → per-chunk DiT sampling → VAE.
                # Flow-matching Euler/DPM-2 via dit.sample(); default
                # sampler dpm_2 @ num_inference_steps=32, scheduler
                # sgm_uniform (Comfy blueprint "Music Cover (YuE2)"
                # parity: KSampler(sampler_name='dpm_2', steps=32, cfg=1.0,
                # denoise=1.0)). Deterministic torch.Generator(seed) noise.
                # Owns the VRAM plan: AR section on GPU → tokens+KV context →
                # AR to CPU/del/empty_cache → DiT section on GPU → latents →
                # DiT to CPU/del/empty_cache → VAE decoder on GPU → decode
                # (peak ~AR / ~DiT, never all-resident; sections are read
                # per-stage from the file, never held whole).
  ar.py         # exact port of Comfy sampling logic (keep headers intact):
                # consts (EOD=151643, ABC_START/END=151847/151848,
                # MUSIC_START/END=151851/151852, CODEC_OFFSET=151853/
                # SIZE=32768, CONTEXT=24576, FPS=25), INSTRUCTIONS sentences,
                # tokenize_with_weights prompt layout ([EOD]+prompt+
                # [ABC_START], dual negative), distribution() (allowed-range
                # masking, end-token floor, scatter penalty, top-k/top-p,
                # greedy passthrough), chunk_ranges() (size=(ctx-prefix-3)//2,
                # raise if <1 frame fits), two-phase _generate (ABC plan with
                # min 32, semantic with min 200, dual-batch CFG 1.01 for
                # cot=off else 1.0, penalty windows 100/50, legacy_off for
                # cot=off), context-budget clamp (fit_budget). Tokenizer is
                # the HF tokenizer.json EMBEDDED in the checkpoint
                # (text_encoders.yue2_tokenizer_json tensor, via tokenizers
                # package) — the .tiktoken sidecar is the wrong vocab.
  dit.py        # exact port of comfy/ldm/yue2/model.py YuE2 DiT (shared
                # yue2_config; vae2llm 64→hidden WITH bias, llm2vae back,
                # sinusoidal-256 timestep MLP ([cos, sin] concat order per
                # comfy/ldm/modules/diffusionmodules/util.py — transposing the
                # halves silently detaches the model from the noise level and
                # drops the fundamental), learned latent pos embed;
                # per-chunk forward: pad +1, vae2llm+time+pos, rope,
                # prefix-KV permute, layer loop with (k,v,index) caches,
                # norm+llm2vae slice). Plus flow_sigmas() (sgm_uniform) and
                # euler_sample()/dpm2_sample() (Comfy k_diffusion ports).
  vae.py        # exact port of Comfy Oobleck VAE decoder with the YuE2 VAE
                # config (48 kHz, 1920x, latent 64, ch 64, snake, no tanh;
                # halo-tiled decode past 1024 frames per the config's
                # core/halo grid — full-song fp32 decode needs ~21 GB
                # transient on long songs. Sections load strict=True).
  llama.py      # exact port of the YuE2-relevant Comfy llama subset:
                # Qwen3_8BConfig (+yue2_config override), RMSNorm,
                # split-half RoPE (manual form of the kitchen kernel),
                # merged-qkv Attention (gemma3 q/k norms, tuple caches),
                # merged-mlp SwiGLU, TransformerBlock, Llama2_
                # (init_kv_cache/forward/logits). No graphs/VBAR/spec-decode.
  fetch.py      # STDLIB-ONLY downloader (urllib + hashlib + Range resume +
                # size/sha verify) for without-Anode use. Node-side fetch
                # reuses repo machinery (see §4); ~80 lines duplicated
                # deliberately — extractability over DRY.
  README.md     # env notes, CLI reference, model license (CC BY-NC 4.0).
```

Comfy sources vendored from `Comfy-Org/ComfyUI` @
`6cff1e974f8212c6b55581ebfa4f51a544f0a077` (master 2026-09-15, post-#16250;
recorded as COMFY_REV in pipeline.py + README):
`comfy/text_encoders/yue2.py`, `comfy/ldm/yue2/model.py`,
`comfy/ldm/audio/autoencoder.py` (Oobleck VAE), relevant subset of
`comfy/text_encoders/llama.py`. Keep original adaptation headers (M·A·P
YuE2 Apache-2.0 note, stable-audio-tools note) intact. Verified: all
three weight sections strict-load against the real checkpoint, and the
checkpoint inventory (28+28 layers, embed 184704x2048, VAE strides)
matches the ported configs.

CI grep enforced (word-boundary regex, avoids `base64` false positives):
`\b(import|from)\s+(base|core|audiocpp_backend|audio_io|plugin_system)\b`
under `tools/yue2_torch/` fails the build.

### CLI contract (`spec.json`-only — deliberate divergence from sidecar)

```
run.py --spec <run_dir/spec.json> --out song.wav --out-dir run_dir --metrics
```

`spec.json` keys: `lyrics, style, cot, seed, semantic_max_tokens,
guidance_scale?, num_inference_steps?, abc_file?, model_dir,
temperature?, top_p?, top_k?, repetition_penalty?` (sampler keys default
to Comfy parity: 1.0/0.95/100/1.2). Unknown keys
fail loudly (exit 2, catches typos; mirrors sidecar strictness).
`export_semantic` needs **no flag**: the
worker always writes the tiny `semantic.json` under `--out-dir`.

Stdout: `metrics.<k>=<float>` lines only (parsed by the existing
`_parse_metrics`). No `progress.stage=` heartbeats in phase 1:
`run_audiocpp_subprocess` buffers via `communicate()` and returns one tail,
so streaming progress would be invisible until exit — defer to a later
streaming-supervisor change. Files: `song.wav` (float32 PCM via
`soundfile.write(..., subtype="FLOAT")` — the subtype must be explicit, as
libsndfile defaults `.wav` to PCM_16 and would hard-clip the decoder's >1.0
peaks; same pattern as `audio_io.write_wav_file`; stdlib `wave`
on Python 3.11 writes format `0x0001` integer PCM and would mislabel float32
bytes, so it must not be used — at the VAE native rate; the node
resamples/dups to engine 48 kHz stereo on load via `load_song_file`, so the
worker does no resampling),
`score.abc` when the model planned (cot=melody/full without input abc),
`semantic.json` (`{frames, truncated}`).
Exit codes: 0 ok / 2 usage / 3 model-missing / 4 CUDA OOM / 1 other.
OOM (4) surfaces through the supervisor's nonzero-exit `RuntimeError` and
the node maps it to a "lower max_duration / close VRAM users" hint.

## 3. Node changes (`plugins/yue2_song_generator.py` only)

- New `engine` menu param (`["sidecar (GGUF)", "torch (BF16)"]`, default 0) +
  `torch_model_dir` string param (default `models/YuE2-Torch`, resolved
  repo-relative via a `_resolve_torch_model_dir` mirroring
  `_resolve_model_dir`). Only `engine` is added to
  `YuE2Widget.PARAM_KEYS` (next to `profile`); `torch_model_dir` stays out,
  mirroring `model_dir` which is deliberately absent from `PARAM_KEYS` to keep
  the widget compact. Keep the sidecar's `num_inference_steps` parameter at
  its 8-step default, and add a separate `torch_steps` parameter at 32. The
  widget shows only the active engine's step control, so the number displayed
  is always the number sent to that engine; torch receives the explicit value
  rather than interpreting 8 as a sentinel. Existing torch patches migrate
  their former shared step value (8 -> 32; non-default values are preserved).
  `profile`/`weight_type`/`attention` help text gains
  "(sidecar only)". Download button label reflects the active engine's
  payload (sidecar `~3.8 GB` vs torch `~7.8 GB`) via a
  `YuE2Widget._update_download_label(engine_idx)` helper called from both
  `__init__` (initial param value — `update_from_params` alone leaves the
  patch-load label stale until a snapshot arrives) and an
  `update_from_params` override
  (`DOWNLOAD_LABEL` alone is a class constant in `offline_job_widget.py` and
  cannot do this). When `engine==torch`, the `profile`/`weight_type`/
  `attention` rows are visually disabled (`setEnabled(False)`) or relabeled
  inactive.
- `_build_spec`: branch on engine. **Sidecar path byte-identical to today.**
  Torch path: same validation rules for style/lyrics/abc/cot/seed/guidance/
  steps/`max_duration`; `tokens=max(1,round(s*25))`; resolve torch model dir +
  require `yue2_3b_bf16.safetensors` (missing → ValueError with fetch hint).
  The tokenizer is read from the checkpoint's embedded
  `text_encoders.yue2_tokenizer_json` tensor at runtime (sibling
  `tokenizer.json` fallback); no separate fetch.
  Spec carries `engine="torch"`, worker python (`sys.executable` — same shared
  conda env, no second interpreter to discover), `tools/yue2_torch/run.py`,
  `semantic_max_tokens`. LoRA params on the torch path raise
  `ValueError("…not supported on the torch engine yet")` — never silently
  ignore them. `profile` is **ignored** on the torch path (torch is always
  BF16; rejecting non-default would wrongly reject an intentional BF16 pick
  and mislabel the Q4_K_M default). Non-default `weight_type`/`attention`
  are warn-ignored on the torch path (recorded in
  `spec["ignored_sidecar_opts"]`, surfaced via `_spec_warnings` as a ⚠ status
  note and in `request.json` — a stale sidecar setting must not block torch
  generation); eager SDPA is always used phase 1.
  AR truncation: stop after `semantic_max_tokens` frames on every `cot` mode,
  then vocode the partial, write `semantic.json {frames, truncated:true}`,
  warn+keep with the existing `cut at Ns cap` note.
- `_run_nrt`: branch. Torch path writes `run_dir/spec.json` (lyrics/style text
  by file, not argv — avoids OS command-line limits), builds argv via worker
  `cli.py` builder (`spec dict → [--spec, --out, --out-dir, --metrics]`),
  then supervises with the **same** poll/cancel/timeout/
  terminate/tail semantics as `run_audiocpp_subprocess` (Windows
  `TerminateProcess` is prompt death; no SIGTERM handler needed). That runner
  hardcodes two `"audio.cpp"` strings (launch-failure and nonzero-exit
  errors), so generalize them with a `prog="audio.cpp"` default parameter —
  all five sidecar families keep byte-identical messages with zero test
  churn, torch passes `prog="yue2-torch"`. Tail then identical:
  `load_song_file` (resample/dup to engine format) → `plan_text` →
  `_keep_song` (+`semantic`) → return dict
  (+`engine` key for status/provenance).
- `_keep_song` `request.json`: add `engine`; torch path records
  `runtime: {torch: <torch.__version__>, comfy_rev: <pinned sha from header>,
  model: {size, sha (from fetch spec, verified at fetch — never rehashed per
  run)}}` instead of
  the `audio_cpp` pin. Keep dir: torch outputs under
  `models/YuE2-Torch/outputs/` (per-engine dirs avoid cross-backend confusion).
  MP3 handling is unchanged (`_keep_song` encodes from the tensor, so torch
  MP3 keeps work for free).
- `on_nrt_complete` / `get_telemetry` / `load_state`: engine-agnostic already
  except status text — append `(torch)` or `(sidecar)` to `_status_detail` so
  VRAM reports are attributable. Extend the `load_state` reset-to-default
  guard to the new params (same pattern as `max_duration`). Old patches
  without the `engine` key load as `sidecar`.
- `_model_fetch_specs`: branch — sidecar returns `yue2_fetch_specs`, torch
  returns new `yue2_torch_fetch_specs` (weights only — tokenizer is
  embedded; same `DownloadSpec` machinery, throttled progress, never
  auto-download).
  `_build_fetch_payload` conditional: torch engine returns `runtime=[]`
  (`runtime_note` reports background checkpoint SHA-256 verification) — the
  ~845 MB `audiocpp_cli` archives must not fetch for torch. The complete
  model spec goes to the NRT worker, which checks an existing checkpoint's
  size and full SHA-256 before accepting it; the engine command thread never
  hashes the multi-GB file.
- Description: one sentence for the engine switch + BF16 VRAM note.

## 4. Fetch

- Node path (primary): new `YUE2_TORCH_*` specs in `audiocpp_backend.py` next
  to the existing ones (exact HF URL + size + sha256 pinned at
  implementation for `yue2_3b_bf16.safetensors`) + a
  `tools/audiocpp/fetch_yue2_torch.py`-style
  sibling script. Reuses `DownloadSpec`/`fetch_all`/resume/verify/progress
  (full hash verified in NRT; see §3).
- Standalone path: `tools/yue2_torch/fetch.py` (stdlib only) for without-Anode.
- `.gitignore`: `models/YuE2-Torch/*.safetensors`, tokenizer cache,
  `models/YuE2-Torch/outputs/`.

## 5. Tests

- Worker-standalone (no engine fixtures, no GPU) in new
  `tests/test_yue2_torch_worker.py`: `tokens` math, Comfy `INSTRUCTIONS` +
  prompt layout, `distribution` (allowed-range masking/end floor/scatter
  penalty/top-k/greedy-scores), `chunk_ranges` split + raise, `fit_budget`
  clamp/flag/no-room, tiny-CPU llama/DiT/VAE shape + KV-growth tests, AR
  `_generate` smoke, embedded-tokenizer roundtrip (gated on the fetched
  checkpoint), `cli.py` builder + sampler-key validation,
  `fetch.py` URL/sha table (weights only), `soundfile` (not stdlib `wave`)
  WAV-write check, zero-Anode-import CI grep test (word-boundary regex).
  Strict section-load verification vs the real checkpoint lives in a
  scratch probe (AR/DiT/VAE `strict=True` green).
- Node (stubbed Popen, existing pattern in `tests/test_yue2_song_generator.py`):
  engine menu defaults sidecar; sidecar spec/argv byte-identical to today
  (unchanged `prog` default keeps the `"audio.cpp …"` error strings);
  torch spec maps 240 s→6000, emits explicit 32-step default (or selected
  `torch_steps`), and targets `run.py` + spec file containing full texts;
  torch fetch payload has `runtime==[]` and verifies the checkpoint in the
  NRT worker, including same-size existing files; download
  button label follows `engine` (init + update); torch
  `on_nrt_complete` truncation note + `(torch)` tag; LoRA-on-torch raises;
  non-default `weight_type`/`attention` warn-and-ignore (⚠ note +
  `request.json`) while `profile` is ignored; save/load round-trip
  with new params; old patch (no `engine` key) loads as
  sidecar.
- GPU-gated e2e (`ANODE_YUE2_TORCH_GPU=1`, Windows box): short `cot=off` torch
  render through `run.py` main (24 frames, 1 DiT step) asserting the WAV
  (48 kHz stereo, samples == frames*1920) and `semantic.json` — green on
  the 8 GB box (~38 s). This layer caught the bf16/RoPE-dtype, multinomial
  shape, and per-chunk-KV bugs that unit tests cannot reach. Long-song
  full-decode VRAM vs length is out of scope for the gate (tiled VAE later).

Run: `conda run -n anode-dev pytest tests/test_yue2_torch_worker.py tests/test_yue2_song_generator.py -q`, then full suite. (`conda` may not be on PATH in
non-interactive shells; `README.md`/`environment.yml` are the source of truth
for the env name — never invoke the env python by absolute path, it breaks
native-library discovery.)

## 6. Out of scope

Per-layer streaming, int8/kitchen/aimdo/graphs, lowvram/offload modes, LoRA,
`abc_max_tokens`, SheetSage2/Qwen/BS-Roformer ports, sidecar deletion,
`offline_job_base` extraction (unneeded — node stays on `AudioCppJob`;
revisit when the second family migrates), streaming progress heartbeats /
supervisor stage timeouts. (`_keep_song` MP3-encode from the tensor is reused
as-is, so torch MP3 keeps work for free — only CLI-side `--out-format` is
irrelevant.)

## 7. Acceptance

- [ ] `tools/yue2_torch/run.py --help` works with only the package dir present
  (no engine imports; no torch import at module top); zero-Anode-import CI
  grep green.
- [ ] Node defaults unchanged (sidecar, same argv as today — covered by
  existing tests); torch opt-in renders short `cot=off` song GPU-gated with
  `cut at cap` + `(torch)` note.
- [ ] LoRA + torch raises loudly; non-default `weight_type`/`attention`
  warn-and-ignore (⚠ status note + `request.json` record), `profile` ignored
  with "(sidecar only)" help + disabled widget rows; old patches load as
  sidecar; new engine choice round-trips save/load; torch Download fetches
  no runtime and never hashes 7.8 GB on the command thread.
- [ ] Target test files + full suite green on the Windows box.
- [ ] No weights committed; fetch-at-runtime only; provenance in `request.json`
  (torch version + pinned Comfy rev + model size/sha from fetch spec).
