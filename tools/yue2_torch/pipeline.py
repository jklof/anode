"""Stage orchestration for the YuE2 torch worker.

Mirrors the ComfyUI YuE2 flow (``YuE2Tokenizer.tokenize_with_weights`` +
``YuE2TEModel.generate``/``encode_token_weights`` + DiT sampling + VAE
decode) without any ``comfy.*`` dependency:

1. tokenize (HF tokenizer embedded in the checkpoint) -> prefix/negative
2. ABC plan sampling (cot=melody/full without input ABC; 256 tokens max)
3. semantic sampling (dual-batch CFG, end-token, budget clamp)
4. KV-conditioning prefill (per-layer caches stacked on CPU)
5. per-chunk DiT midpoint-Euler sampling (8 steps)
6. full-song VAE decode -> soundfile float32 48 kHz stereo

VRAM plan (required phase 1 on 8 GB WDDM): each stage reads only its own
checkpoint section, builds its model on GPU, then releases
(``to("cpu")``/``del``/``empty_cache()``) before the next stage. Peak is
~AR (~6.4 GB with dual-batch KV) / ~DiT / ~VAE, never all-resident.
Per-layer streaming and tiled VAE stay deferred (plan §6).
"""

import json
import time
from pathlib import Path

try:
    from . import ar as _ar
    from . import dit as _dit
    from . import vae as _vae
    from .cli import validate_spec as _validate_spec
    from .llama import Llama2_, yue2_config
except ImportError:  # standalone file loading (tests import by path)
    import importlib.util as _ilu
    import sys as _sys

    def _sibling(_name):
        _mod_name = f"yue2_torch_{_name}"
        _mod = _sys.modules.get(_mod_name)
        if _mod is None:
            _spec = _ilu.spec_from_file_location(
                _mod_name,
                str(Path(__file__).resolve().parent / f"{_name}.py"))
            _mod = _ilu.module_from_spec(_spec)
            _sys.modules[_mod_name] = _mod
            _spec.loader.exec_module(_mod)
        return _mod

    _ar = _sibling("ar")
    _dit = _sibling("dit")
    _vae = _sibling("vae")
    _validate_spec = _sibling("cli").validate_spec
    _llama = _sibling("llama")
    Llama2_, yue2_config = _llama.Llama2_, _llama.yue2_config

WEIGHTS_FILE = "yue2_3b_bf16.safetensors"
TOKENIZER_KEY = "text_encoders.yue2_tokenizer_json"
# Comfy blueprint "Music Cover (YuE2)": KSampler(sampler_name='dpm_2',
# scheduler='sgm_uniform', steps=32, cfg=1.0, denoise=1.0).
DEFAULT_STEPS = 32
DEFAULT_SAMPLER = "dpm_2"
ABC_MAX_TOKENS = 256
COMFY_REV = "6cff1e974f8212c6b55581ebfa4f51a544f0a077"


def _section_tensors(model_dir, prefix, exclude=()):
    """Read one checkpoint section into CPU tensors (header + slice only).

    Never materializes the full 7.8 GB file: each stage pulls exactly the
    keys it owns, so VRAM staging starts at the file-read level.
    """
    from safetensors import safe_open
    path = Path(model_dir) / WEIGHTS_FILE
    if not path.exists():
        raise FileNotFoundError(
            f"torch weights not found: {path} "
            "(run tools/audiocpp/fetch_yue2_torch.py)")
    out = {}
    with safe_open(str(path), framework="pt", device="cpu") as f:
        for key in f.keys():
            if key.startswith(prefix) and key not in exclude:
                out[key[len(prefix):]] = f.get_tensor(key)
    if not out:
        raise RuntimeError(f"no {prefix}* tensors in {path}")
    return out


def _device():
    import torch
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _release(*objs):
    """Move stage models back to CPU and flush the CUDA cache.

    NOTE: this cannot drop the caller's bindings (``del`` on ``*objs``
    only drops the local tuple) — call sites must ``del`` names
    explicitly after calling this.
    """
    import torch
    for o in objs:
        try:
            if hasattr(o, "to"):
                o.to("cpu")
        except Exception:
            pass
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _vram_gb():
    import torch
    if not torch.cuda.is_available():
        return 0.0
    return torch.cuda.memory_allocated() / 1e9


def _vram_peak_gb():
    import torch
    if not torch.cuda.is_available():
        return 0.0
    return torch.cuda.max_memory_allocated() / 1e9


def _vram_log(stage):
    print(f"yue2-torch vram [{stage}]: live {_vram_gb():.2f} GiB, "
          f"peak {_vram_peak_gb():.2f} GiB", flush=True)


def _oom(stage, detail=""):
    """Annotate a CUDA OOM with the pipeline stage + live usage."""
    return RuntimeError(
        f"[{stage}] CUDA out of memory ({detail}; "
        f"{_vram_gb():.1f} GiB live). Lower max_duration / close VRAM "
        "users".rstrip())


# AR footprint model for the 8 GB pre-flight (measured/derived, bf16):
# ~5.0 GiB weights (embed + lm_head + 28 layers) + per-token dual/single
# KV cache + 1.0 GiB headroom (CUDA context, transients, allocator slack).
AR_WEIGHTS_GB = 5.0
AR_KV_GB_PER_TOKEN = 115.0 / 1e6
AR_HEADROOM_GB = 1.0
AR_VRAM_BUDGET_GB = 7.0


def check_ar_fit(prefix_len, budget, dual, context=24576):
    """Estimate the AR stage footprint; warn (never block) when over.

    A 4270-token prompt with a 6000 budget was observed spilling WDDM
    shared memory to 23 GiB live before dying — but the reference
    (audio.cpp/Comfy) handles such prompts on 8 GB, so an estimate must
    not refuse the run. It warns with the numbers and remedies; the
    stage-tagged OOM is the backstop. Returns the estimated GiB.
    """
    import sys
    total = int(prefix_len) + int(budget)
    est = (AR_WEIGHTS_GB
           + total * AR_KV_GB_PER_TOKEN * (2 if dual else 1)
           + AR_HEADROOM_GB)
    if est > AR_VRAM_BUDGET_GB:
        print(f"yue2-torch: prompt ({prefix_len} tokens) + budget "
              f"({budget} tokens) estimates ~{est:.1f} GiB AR VRAM, over "
              f"the {AR_VRAM_BUDGET_GB:.1f} GiB 8 GB guideline: if this "
              f"OOMS, shorten the lyrics/ABC input or lower max_duration",
              file=sys.stderr, flush=True)
    return est


def run_pipeline(spec, out_wav, out_dir, *, num_inference_steps=None,
                 progress_cb=None):
    """Run plan -> semantic -> acoustic -> vocode. Returns result dict.

    ``spec`` is the validated worker spec (see ``cli.validate_spec``).
    Writes ``song.wav`` (soundfile float32 48 kHz stereo),
    ``score.abc`` when the model planned one, and always ``semantic.json``.
    Returns ``{"wav", "score"|None, "semantic", "metrics", "native_sr"}``.
    """
    import torch
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    _validate_spec(spec)
    t0 = time.monotonic()
    model_dir = Path(spec["model_dir"])
    out_wav, out_dir = Path(out_wav), Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    steps = spec.get("num_inference_steps") \
        if spec.get("num_inference_steps") is not None \
        else (num_inference_steps or DEFAULT_STEPS)
    sampler = spec.get("sampler") or DEFAULT_SAMPLER
    requested = int(spec["semantic_max_tokens"])
    cot = spec["cot"]
    seed = int(spec["seed"])
    guidance = spec.get("guidance_scale")
    temperature = spec.get("temperature", _ar.DEFAULT_TEMPERATURE)
    top_p = spec.get("top_p", _ar.DEFAULT_TOP_P)
    top_k = spec.get("top_k", _ar.DEFAULT_TOP_K)
    rep = spec.get("repetition_penalty",
                   _ar.DEFAULT_REPETITION_PENALTY)
    cfg = 1.01 if cot == "off" and guidance is None \
        else (guidance if guidance is not None else 1.0)
    legacy_off = cot == "off"

    tokenizer = _ar.YuE2Tokenizer.from_model_dir(model_dir)
    _ar.bind_tokenizer(tokenizer)
    abc_text = ""
    if spec.get("abc_file"):
        abc_text = Path(spec["abc_file"]).read_text(encoding="utf-8")
    lyrics = spec["lyrics"]
    if not lyrics.strip():
        raise ValueError("lyrics must be nonempty text")
    tokens = _ar.tokenize_with_weights(
        spec["style"], lyrics, abc_text, cot=cot, seed=seed,
        max_tokens=requested, temperature=temperature, top_p=top_p,
        top_k=top_k, repetition_penalty=rep, cfg_scale=cfg)
    device = _device()
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    if progress_cb:
        progress_cb("plan", 0.0)

    # -- stage 1: AR model on GPU ----------------------------------------
    ar_tensors = _section_tensors(
        model_dir, "text_encoders.model.", exclude=(TOKENIZER_KEY,))
    ar_model = Llama2_(yue2_config())
    ar_model.load_state_dict(ar_tensors, strict=True)
    del ar_tensors
    ar_model.to(device=device, dtype=dtype).eval()
    _vram_log("ar-loaded")

    plan_text = None
    try:
        # Inference-only (Comfy parity): no autograd graphs are built or
        # retained anywhere in generation.
        with torch.inference_mode():
            # -- ABC plan phase (generated unless input score was given) --
            abc_ids = list(tokens["abc_ids"])
            if cot in ("melody", "full") and not abc_ids:
                try:
                    abc_ids, _ = _ar._generate(
                        ar_model, tokens["prefix"], seed, ABC_MAX_TOKENS,
                        "abc", device, dtype, negative=tokens["negative"],
                        cfg_scale=cfg, legacy_off=False,
                        temperature=temperature, top_p=top_p,
                        top_k=top_k, repetition_penalty=rep,
                        penalty_window=_ar.DEFAULT_PENALTY_WINDOW,
                        min_tokens=min(32, ABC_MAX_TOKENS))
                except RuntimeError as e:
                    if "out of memory" in str(e).lower():
                        raise _oom("ar-plan",
                                   f"{len(tokens['prefix'])} prompt tokens")
                    raise
                plan_text = tokenizer.decode(abc_ids) if abc_ids else None
                if plan_text is not None and not plan_text.strip():
                    plan_text = None
            _vram_log("abc-done")
            # -- semantic phase ------------------------------------------
            sem_prefix = (tokens["prefix"] + abc_ids
                          + [_ar.ABC_END, _ar.MUSIC_START])
            sem_negative = tokens["negative"] + (
                [_ar.MUSIC_START] if cot == "off"
                else [_ar.ABC_START] + abc_ids
                + [_ar.ABC_END, _ar.MUSIC_START])
            budget, ctx_reduced = _ar.fit_budget(
                requested, sem_prefix, sem_negative)
            if device.type == "cuda":
                check_ar_fit(len(sem_prefix), budget, dual=cfg != 1.0)
            print(f"yue2-torch generate: cot={cot} prefix={len(sem_prefix)} "
                  f"budget={budget} dual={cfg != 1.0} "
                  f"cfg={cfg} temp={temperature} top_p={top_p} top_k={top_k}",
                  flush=True)
            try:
                semantic, ar_truncated = _ar._generate(
                    ar_model, sem_prefix, seed, budget, "semantic", device,
                    dtype, negative=sem_negative, cfg_scale=cfg,
                    legacy_off=legacy_off, temperature=temperature,
                    top_p=top_p, top_k=top_k,
                    repetition_penalty=rep,
                    penalty_window=_ar.SEMANTIC_PENALTY_WINDOW,
                    min_tokens=min(200, budget))
            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    raise _oom("ar-semantic",
                               f"budget {budget}, prefix {len(sem_prefix)}")
                raise
            truncated = bool(ar_truncated or ctx_reduced
                             or budget < requested)
            _vram_log("semantic-done")
            # -- KV-conditioning prefills, one per acoustic chunk ---------
            # (Comfy _acoustic_conditioning: each chunk prefills
            # prefix + own slice + [MUSIC_END] independently and stacks the
            # per-layer KV; the DiT slices KV per chunk via 4-tuple ranges.)
            ranges = _ar.chunk_ranges(len(semantic), len(sem_prefix))
            total = sum(len(sem_prefix) + end - start + 1
                        for start, end in ranges)
            cfg_model = ar_model.config
            context = torch.empty(
                1, total, cfg_model.num_hidden_layers, 2,
                cfg_model.num_key_value_heads, cfg_model.head_dim,
                device="cpu", dtype=torch.float32)
            chunks = []
            offset = 0
            try:
                for start, end in ranges:
                    ids = sem_prefix + semantic[start:end] + [_ar.MUSIC_END]
                    caches = ar_model.init_kv_cache(1, len(ids), device,
                                                    dtype)
                    idt = torch.tensor([ids], device=device,
                                       dtype=torch.long)
                    _, _, caches = ar_model(idt, past_key_values=caches)
                    for i, (k, v, _idx) in enumerate(caches):
                        context[0, offset:offset + len(ids), i, 0] = \
                            k[0].transpose(0, 1).float().cpu()
                        context[0, offset:offset + len(ids), i, 1] = \
                            v[0].transpose(0, 1).float().cpu()
                    chunks.append((start, end, offset, offset + len(ids)))
                    offset += len(ids)
                    del caches
            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    raise _oom("ar-conditioning",
                               f"{len(semantic)} frames")
                raise
            _vram_log("conditioning-done")
    finally:
        _release(ar_model)
        del ar_model
    if progress_cb:
        progress_cb("acoustic", 0.5)

    # -- stage 2: DiT sampling on GPU ------------------------------------
    dit_tensors = _section_tensors(model_dir, "model.diffusion_model.")
    dit_model = _dit.YuE2DiT()
    dit_model.load_state_dict(dit_tensors, strict=True)
    del dit_tensors
    dit_model.to(device=device, dtype=dtype).eval()
    _vram_log("dit-loaded")
    try:
        latents = _dit.sample(
            dit_model, context.to(device=device, dtype=dtype), chunks,
            len(semantic), steps=steps, seed=seed, sampler=sampler,
            device=device, dtype=dtype)
    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            raise _oom("dit-sample",
                       f"{len(semantic)} frames, {steps} steps")
        raise
    finally:
        _release(dit_model, context)
        del dit_model, context
    _vram_log("dit-done")
    if progress_cb:
        progress_cb("vocode", 0.8)

    # -- stage 3: VAE decode (tiled past 1024 frames) ---------------------
    vae_tensors = _section_tensors(model_dir, "vae.")
    vae_model = _vae.AudioOobleckVAE.load_decoder(
        {"vae." + k: v for k, v in vae_tensors.items()},
        device="cpu", dtype=None)
    del vae_tensors
    vae_model.to(device=device, dtype=torch.float32).eval()
    _vram_log("vae-loaded")
    latents_f32 = latents.to(device=device, dtype=torch.float32)
    del latents
    audio = None
    try:
        with torch.inference_mode():
            if latents_f32.shape[-1] > _vae.DECODE_CORE_FRAMES:
                audio = _vae.decode_tiled(vae_model, latents_f32)
            else:
                audio = vae_model.decode(latents_f32)
        audio_np = audio.float().cpu().numpy()
    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            raise _oom("vae-decode", f"{latents_f32.shape[-1]} frames")
        raise
    finally:
        _release(vae_model)
        del vae_model, latents_f32, audio
    if progress_cb:
        progress_cb("done", 1.0)

    # -- outputs (soundfile float32; stdlib wave cannot write float32) ----
    # subtype must be explicit: libsndfile defaults .wav to PCM_16, which
    # quantizes and hard-clips the decoder's >1.0 peaks.
    import soundfile as sf
    import numpy as np
    native_sr = _vae.NATIVE_SR
    data = np.ascontiguousarray(audio_np[0].T, dtype=np.float32)
    if data.shape[1] == 1:
        data = np.repeat(data, 2, axis=1)
    out_wav.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(out_wav), data, native_sr, subtype="FLOAT")

    score_path = None
    if plan_text is not None:
        score_path = str(out_dir / "score.abc")
        Path(score_path).write_text(plan_text, encoding="utf-8")

    semantic_info = {"frames": len(semantic), "truncated": truncated}
    (out_dir / "semantic.json").write_text(
        json.dumps(semantic_info), encoding="utf-8")

    wall_s = max(1e-6, time.monotonic() - t0)
    dur_s = data.shape[0] / float(native_sr)
    metrics = {"wall_ms": wall_s * 1000.0,
               "audio_duration_ms": dur_s * 1000.0,
               "rtf": wall_s / max(1e-6, dur_s)}
    if torch.cuda.is_available():
        metrics["vram_peak_mb"] = torch.cuda.max_memory_allocated() / 1e6
    # Non-float metrics ride along as strings; the node parses k=v lines
    # and keeps them for provenance.
    metrics["sampler"] = sampler
    metrics["steps"] = float(steps)
    return {"wav": str(out_wav), "score": score_path,
            "semantic": semantic_info, "metrics": metrics,
            "native_sr": native_sr}
