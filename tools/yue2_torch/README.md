# YuE2 torch worker (`tools/yue2_torch/`)

Self-contained PyTorch YuE2-3B song generation (ComfyUI streaming
architecture: chunked acoustic conditioning, per-chunk diffusion, small
VRAM-resident working set). Zero ANode imports — the node drives it as a
subprocess with a `spec.json` file.

## Environment

Shared `anode-dev` conda env plus `safetensors`, `tokenizers`, `einops`,
`tqdm` (see `environment.yml`). No `comfy-kitchen` / `comfy-aimdo` /
triton in phase 1 (eager BF16 + `torch.sdpa` only).

## CLI reference

```
python tools/yue2_torch/run.py --spec <run_dir/spec.json> \
  --out song.wav --out-dir run_dir --metrics [--seed N]
```

`spec.json` keys: `lyrics, style, cot (off|melody|full), seed,
semantic_max_tokens, guidance_scale?, num_inference_steps?, abc_file?,
model_dir`. Unknown keys fail (exit 2).

Stdout: `metrics.<k>=<float>` lines (parsed by the node's
`_parse_metrics`). Files: `song.wav` (float32 via `soundfile` at the VAE
native rate — the node resamples to 48 kHz stereo on load),
`score.abc` when the model planned one, `semantic.json`
(`{frames, truncated}`).

Exit codes: 0 ok / 2 usage / 3 model-missing / 4 CUDA OOM / 1 other.

`sampler` (`"dpm_2"` default, or `"euler"`) is accepted as an optional
spec key; `num_inference_steps` defaults to 32 when absent.

## Sampling recipe (Comfy parity — do not "simplify")

`YuE2` is a `ModelType.FLOW` model whose `sampling_settings` are
`{"multiplier": 1.0}` with the default `shift=1.0`, so **sigma == timestep
== t in (0, 1]** and sigmas run from `sigma_max=1.0` (pure noise) down to
`0.0` (clean). Concretely:

* sigmas come from Comfy's `normal_scheduler(..., sgm=True)`
  (`sgm_uniform`): a linspace `sigma_max -> sigma_min` with a trailing
  `0.0`, i.e. **descending**, so every Euler/DPM `dt` is **negative**.
* the model consumes the current sample `x` directly
  (`CONST.calculate_input`) and returns velocity `v`, giving
  `x0 = x - sigma*v`; hence `to_d = (x - x0)/sigma == v` and a step is
  `x += v * (sigma_{i+1} - sigma_i)`.
* defaults are `dpm_2` @ **32** steps, matching the official Comfy
  blueprint (`KSampler(sampler_name='dpm_2', scheduler='sgm_uniform',
  steps=32, cfg=1.0, denoise=1.0)`). `euler` is also implemented and is
  ~2x cheaper in NFE.

`timestep_embedding` **must** concatenate `[cos(args), sin(args)]`
(`comfy/ldm/modules/diffusionmodules/util.py`). The halves feed a linear
embedder, so transposing them silently detaches the DiT from the noise
level: the flow trajectory collapses, the output loses its fundamental and
sounds like a tape running fast. Regression guard:
`test_timestep_embedding_is_cos_then_sin`.

`song.wav` is written with an explicit `subtype="FLOAT"`; libsndfile
defaults `.wav` to PCM_16, which quantizes and hard-clips the decoder's
>1.0 peaks.

## VRAM plan (required phase 1)

AR on GPU → tokens → AR to CPU/`del`/`empty_cache()` → DiT on GPU →
chunk latents → DiT to CPU/`del`/`empty_cache()` → VAE on GPU → decode.
Peak is ~AR / ~DiT, never all-resident (~7.8 GB weights alone exceed an
8 GB WDDM card with KV/activations).

Measured on an 8 GB card (2000-frame song, cot=full): ar-loaded 4.33,
semantic peak 4.86, conditioning peak 5.01, DiT/VAE lower — flat curve,
no growth anomaly. Rule of thumb: AR needs ~4.3 GiB weights + ~112 KiB
per token single-batch (~224 KiB dual-batch with cot=off CFG) + ~1 GiB
headroom. So on 8 GB: prompt + budget up to ~9k tokens single-batch
(cot=melody/full) or ~7k dual-batch (cot=off). A 4270-token prompt with
a 6000 budget (~10k dual ≈ 7.5+ GiB) spills WDDM shared memory and dies
slowly — shorten the lyrics/ABC input, lower max_duration, or use
cot=full. The worker prints a pre-flight estimate and per-stage
live/peak GiB to stderr for exactly this diagnosis.

DiT attention runs full-chunk (Comfy parity — flash/mem-efficient sdpa
confirmed working on sm_120, ~0.03 GB transient at 6000 frames).
Long VAE inputs decode halo-tiled (1024-frame cores + 16-frame halos
per `yue2-vae-config.json`, halo outputs trimmed, cores concatenated to
exactly `frames * 1920` samples); short inputs take the exact
full-decode path. `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` is
set in `run.py` on Linux only (unsupported + warning-spam on Windows).
OOMs abort with the stage name (`[ar-plan]`, `[ar-semantic]`,
`[ar-conditioning]`, `[dit-sample]`, `[vae-decode]`) plus live GiB, and
the peak lands in `metrics.vram_peak_mb`.

## Model files (`models/YuE2-Torch/`, fetch-at-runtime)

* `yue2_3b_bf16.safetensors` — `Comfy-Org/YuE2`
  `checkpoints/yue2_3b_bf16.safetensors` (size/sha pinned after the first
  verified download; presence-only until then). Holds the AR planner
  (`text_encoders.model.*`), the DiT (`model.diffusion_model.*`), the VAE
  (`vae.*`), and the tokenizer (`text_encoders.yue2_tokenizer_json`).
* No separate tokenizer file: the HF `tokenizer.json` is embedded in the
  checkpoint (decoded via the `tokenizers` package).

Fetch via the node Download button or:

```
python tools/audiocpp/fetch_yue2_torch.py   # repo machinery
python tools/yue2_torch/fetch.py            # stdlib only, without ANode
```

## Sources vendored from

`Comfy-Org/ComfyUI` @ `6cff1e974f8212c6b55581ebfa4f51a544f0a077`
(master 2026-09-15, post-#16250). Files: `comfy/text_encoders/yue2.py` →
`ar.py`, `comfy/ldm/yue2/model.py` → `dit.py`,
`comfy/ldm/audio/autoencoder.py` (Oobleck VAE) → `vae.py`, relevant
subset of `comfy/text_encoders/llama.py` → `llama.py`. Original headers
kept intact. Comfy-only machinery stripped (no `comfy.*` /
`comfy_kitchen` imports, graphs, prefetch, progress): plain
`nn.Linear`/`nn.Embedding`, manual split-half RoPE, `torch.sdpa`.

## Model license

Weights are CC BY-NC 4.0 (non-commercial) — fetch at runtime only, never
commit or bundle. Commercial use needs separate permission.
