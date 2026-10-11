"""Tests for the YuE2 torch worker (tools/yue2_torch/) + node engine switch.

Worker-standalone tests run without a GPU (pure functions and tiny
random-weight CPU models only; the subprocess is never launched, and the
7.8 GB checkpoint is touched only by the gated embedded-tokenizer test).
Node tests stub the model dir with placeholder files and drive
_build_spec paths directly.
"""
import importlib.util
import math
import os
import re
import subprocess
import sys
import time
from pathlib import Path

import pytest
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
WORKER_DIR = REPO_ROOT / "tools" / "yue2_torch"
CHECKPOINT = REPO_ROOT / "models" / "YuE2-Torch" / "yue2_3b_bf16.safetensors"

needs_checkpoint = pytest.mark.skipif(
    not CHECKPOINT.exists(),
    reason="needs the fetched torch checkpoint")


def _load(name):
    mod_name = f"yue2_torch_{name}"
    mod = sys.modules.get(mod_name)
    if mod is None:
        spec = importlib.util.spec_from_file_location(
            mod_name, str(WORKER_DIR / f"{name}.py"))
        mod = importlib.util.module_from_spec(spec)
        sys.modules[mod_name] = mod
        spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def cli():
    return _load("cli")


@pytest.fixture(scope="module")
def ar():
    return _load("ar")


@pytest.fixture(scope="module")
def dit():
    return _load("dit")


@pytest.fixture(scope="module")
def vae():
    return _load("vae")


@pytest.fixture(scope="module")
def llama():
    return _load("llama")


@pytest.fixture(scope="module")
def node_cls():
    import plugin_system
    plugin_system.load_plugins("plugins")
    cls = plugin_system.NODE_REGISTRY.get("YuE2SongGenerator")
    assert cls is not None
    return cls


def _live_node_module():
    return sys.modules["yue2_song_generator"]


class _StubTokenizer:
    """Stand-in HF tokenizer (ids are byte values + 1000)."""

    def encode(self, text):
        return [b + 1000 for b in text.encode("utf-8")]

    def decode(self, ids, skip_special_tokens=True):
        return bytes(i - 1000 for i in ids).decode(
            "utf-8", errors="replace")


@pytest.fixture()
def stub_tokenizer(ar):
    tok = _StubTokenizer()
    ar.bind_tokenizer(tok)
    yield tok
    ar.bind_tokenizer(None)


# ----------------------------------------------------------------------
# worker: tokens math + instructions
# ----------------------------------------------------------------------
def test_tokens_math(cli):
    assert cli.tokens_for_seconds(240) == 6000
    assert cli.tokens_for_seconds(30) == 750
    assert cli.tokens_for_seconds(0) == 1
    assert cli.tokens_for_seconds(900) == 22500


def test_instructions_match_comfy(ar):
    assert set(ar.INSTRUCTIONS) == {"off", "melody", "full"}
    assert ar.INSTRUCTIONS["off"].startswith("Generate music with codec")
    assert "melody-only ABC" in ar.INSTRUCTIONS["melody"]
    assert "chord-annotated ABC" in ar.INSTRUCTIONS["full"]


def test_tokenize_prompt_layout(ar, stub_tokenizer):
    bundle = ar.tokenize_with_weights("pop", "[Verse]\nla\n", "",
                                      cot="full", seed=7, max_tokens=6000)
    assert bundle["prefix"][0] == ar.EOD
    assert bundle["prefix"][-1] == ar.ABC_START
    assert bundle["negative"][0] == ar.EOD
    assert bundle["negative"] != bundle["prefix"]
    assert bundle["cfg_scale"] == 1.0
    assert bundle["seed"] == 7 and bundle["max_tokens"] == 6000
    off = ar.tokenize_with_weights("pop", "la", "", cot="off")
    assert off["cfg_scale"] == 1.01


# ----------------------------------------------------------------------
# worker: distribution (Comfy parity)
# ----------------------------------------------------------------------
def _logits(n=184704, fill=-100.0):
    return torch.full((1, n), fill)


def test_distribution_abc_masks_to_text_and_end(ar):
    logits = _logits()
    logits[0, 42] = 5.0
    logits[0, ar.CODEC_OFFSET + 7] = 9.0  # must be masked in abc phase
    scores = ar.distribution(logits, [], 10, "abc", temperature=1.0,
                             top_p=1.0, top_k=0, min_tokens=0)
    assert scores[0, 42].item() == pytest.approx(5.0)
    assert scores[0, ar.CODEC_OFFSET + 7].item() == float("-inf")
    # Masking is additive: the allowed end id keeps its input logit.
    assert scores[0, ar.ABC_END].item() == pytest.approx(-100.0)


def test_distribution_semantic_masks_to_codec_and_end(ar):
    logits = _logits()
    logits[0, 42] = 5.0  # text id: masked in semantic phase
    logits[0, ar.CODEC_OFFSET + 7] = 3.0
    scores = ar.distribution(logits, [], 300, "semantic", temperature=1.0,
                             top_p=1.0, top_k=0, min_tokens=0)
    assert scores[0, 42].item() == float("-inf")
    assert scores[0, ar.CODEC_OFFSET + 7].item() == pytest.approx(3.0)
    assert scores[0, ar.MUSIC_END].item() == pytest.approx(-100.0)


def test_distribution_end_token_floor(ar):
    logits = _logits()
    scores = ar.distribution(logits, [], 5, "abc", temperature=1.0,
                             top_p=1.0, top_k=0, min_tokens=32)
    assert scores[0, ar.ABC_END].item() == float("-inf")
    scores = ar.distribution(logits, [], 32, "abc", temperature=1.0,
                             top_p=1.0, top_k=0, min_tokens=32)
    assert scores[0, ar.ABC_END].item() == pytest.approx(-100.0)


def test_distribution_greedy_returns_scores(ar):
    logits = _logits()
    logits[0, ar.CODEC_OFFSET + 3] = 4.0
    scores = ar.distribution(logits, [], 300, "semantic", temperature=0,
                             top_p=1.0, top_k=0, min_tokens=0)
    assert scores.argmax(-1).item() == ar.CODEC_OFFSET + 3


def test_distribution_top_k_filters(ar):
    logits = _logits()
    logits[0, ar.CODEC_OFFSET + 1] = 3.0
    logits[0, ar.CODEC_OFFSET + 2] = 2.0
    logits[0, ar.CODEC_OFFSET + 3] = 1.0
    scores = ar.distribution(logits, [], 300, "semantic", temperature=1.0,
                             top_p=1.0, top_k=2, min_tokens=0)
    assert scores[0, ar.CODEC_OFFSET + 3].item() == float("-inf")
    assert scores[0, ar.CODEC_OFFSET + 1].item() > float("-inf")


def test_distribution_repetition_penalty(ar):
    logits = _logits()
    logits[0, ar.CODEC_OFFSET + 5] = 4.0
    plain = ar.distribution(logits, [ar.CODEC_OFFSET + 5] * 60, 300,
                            "semantic", temperature=1.0, top_p=1.0,
                            top_k=0, repetition_penalty=1.0,
                            penalty_window=50, min_tokens=0)
    penalized = ar.distribution(logits, [ar.CODEC_OFFSET + 5] * 60, 300,
                                "semantic", temperature=1.0, top_p=1.0,
                                top_k=0, repetition_penalty=1.2,
                                penalty_window=50, min_tokens=0)
    assert penalized[0, ar.CODEC_OFFSET + 5].item() < \
        plain[0, ar.CODEC_OFFSET + 5].item()


# ----------------------------------------------------------------------
# worker: chunk_ranges + fit_budget
# ----------------------------------------------------------------------
def test_chunk_ranges_split(ar):
    ranges = ar.chunk_ranges(10, 100)
    assert ranges == [(0, 10)]
    ranges = ar.chunk_ranges(10, 100, context=117)
    # size = (117-100-3)//2 = 7 -> [(0,7),(7,10)]
    assert ranges == [(0, 7), (7, 10)]


def test_chunk_ranges_raise(ar):
    with pytest.raises(ValueError):
        ar.chunk_ranges(10, 10, context=10)
    with pytest.raises(ValueError):
        ar.chunk_ranges(0, 10)


def test_fit_budget_clamps_and_flags(ar):
    budget, reduced = ar.fit_budget(6000, [0] * 100, [0] * 50)
    assert (budget, reduced) == (6000, False)
    budget, reduced = ar.fit_budget(6000, [0] * 24000, [0] * 50)
    assert budget == 24576 - 24000 and reduced is True
    with pytest.raises(ValueError, match="no room"):
        ar.fit_budget(6000, [0] * 24576, [0] * 50)


# ----------------------------------------------------------------------
# worker: cli builder
# ----------------------------------------------------------------------
def _spec(**kw):
    spec = {"lyrics": "[Verse]\nla\n", "style": "pop", "cot": "full",
            "seed": 7, "semantic_max_tokens": 6000, "model_dir": "m"}
    spec.update(kw)
    return spec


def test_cli_builder_emits_file_based_argv(cli):
    argv = cli.build_argv("py", "run.py", spec_path="s.json",
                          out_wav="o.wav", out_dir="run")
    assert argv == ["py", "run.py", "--spec", "s.json", "--out", "o.wav",
                    "--out-dir", "run", "--metrics"]
    assert "lyrics" not in argv and "[Verse]" not in str(argv)


def test_cli_builder_rejects(cli):
    with pytest.raises(ValueError, match="unknown spec"):
        cli.validate_spec(_spec(bogus=1))
    with pytest.raises(ValueError, match="cot"):
        cli.validate_spec(_spec(cot="bogus"))
    with pytest.raises(ValueError, match="ABC score"):
        cli.validate_spec(_spec(cot="off", abc_file="s.abc"))
    with pytest.raises(ValueError, match="seed"):
        cli.validate_spec(_spec(seed=-1))
    with pytest.raises(ValueError, match="lyrics"):
        cli.validate_spec(_spec(lyrics="  "))
    with pytest.raises(ValueError, match="temperature"):
        cli.validate_spec(_spec(temperature=5.0))
    with pytest.raises(ValueError, match="top_p"):
        cli.validate_spec(_spec(top_p=0.0))
    with pytest.raises(ValueError, match="top_k"):
        cli.validate_spec(_spec(top_k=-1))
    with pytest.raises(ValueError, match="repetition_penalty"):
        cli.validate_spec(_spec(repetition_penalty=0.5))
    # Optionals omitted stay valid.
    cli.validate_spec(_spec())


def test_worker_spec_shape_matches_node_builder(cli):
    """The node writes lyrics/style TEXT into spec.json (file paths never
    leave the node); the worker must accept exactly that shape."""
    worker_spec = {"lyrics": "[Verse]\nla\n", "style": "pop", "cot": "full",
                   "seed": 7, "semantic_max_tokens": 6000,
                   "model_dir": "m"}
    cli.validate_spec(worker_spec)


def test_pipeline_reads_lyrics_text_not_path():
    """Regression: run_pipeline consumed spec['lyrics_file'], which the
    worker spec never carries (KeyError on every torch run)."""
    src = (WORKER_DIR / "pipeline.py").read_text(encoding="utf-8")
    assert 'spec["lyrics_file"]' not in src
    assert 'spec["lyrics"]' in src


# ----------------------------------------------------------------------
# worker: llama core (tiny CPU model)
# ----------------------------------------------------------------------
def _tiny_config(llama, **kw):
    # Full vocab width: distribution() indexes codec/special ids, so the
    # head must be full-size even in the tiny model (2x189 MB fp32, fine).
    cfg = {"vocab_size": 184704, "hidden_size": 256,
           "intermediate_size": 512, "num_hidden_layers": 2,
           "num_attention_heads": 2, "num_key_value_heads": 1,
           "max_position_embeddings": 64, "lm_head": True,
           "merged_qkv": True, "merged_mlp": True}
    cfg.update(kw)
    return llama.Qwen3_8BConfig(**cfg)


def test_rmsnorm_math(llama):
    norm = llama.RMSNorm(4, eps=0.0)
    with torch.no_grad():
        norm.weight.fill_(1.0)
    x = torch.tensor([[1.0, 2.0, 3.0, 4.0]])
    out = norm(x)
    assert torch.allclose(out, x / math_sqrt_mean(x), atol=1e-5)


def math_sqrt_mean(x):
    import math
    return math.sqrt((x.float() ** 2).mean().item())


def test_rope_shapes(llama):
    cos, sin = llama.precompute_freqs_cis(128, torch.tensor([[0, 1, 2, 3]]),
                                          1000000.0)
    assert cos.shape == (1, 1, 4, 64) and sin.shape == (1, 1, 4, 64)
    x = torch.randn(1, 2, 4, 128)
    assert llama.apply_rope(x, (cos, sin)).shape == x.shape


cuda_needed = pytest.mark.skipif(not torch.cuda.is_available(),
                                 reason="needs CUDA")


@cuda_needed
def test_rope_device_matches_cpu(llama):
    pos = torch.tensor([[0, 1, 2, 3]])
    cos_cpu, sin_cpu = llama.precompute_freqs_cis(128, pos, 1000000.0,
                                                  device="cpu")
    cos_cu, sin_cu = llama.precompute_freqs_cis(128, pos.cuda(), 1000000.0,
                                                device="cuda")
    assert cos_cu.device.type == "cuda"
    assert torch.allclose(cos_cu.cpu(), cos_cpu, atol=1e-5)
    assert torch.allclose(sin_cu.cpu(), sin_cpu, atol=1e-5)


@cuda_needed
def test_tiny_prefill_on_cuda(llama):
    cfg = _tiny_config(llama)
    model = llama.Llama2_(cfg).eval().cuda()
    ids = torch.tensor([[10, 20, 30]], device="cuda")
    caches = model.init_kv_cache(1, 16, "cuda", torch.float32)
    hidden, _, caches = model(ids, past_key_values=caches)
    assert hidden.shape == (1, 3, 256) and hidden.device.type == "cuda"
    assert caches[0][2] == 3
    del model
    torch.cuda.empty_cache()


def test_llama_forward_and_kv_growth(llama):
    cfg = _tiny_config(llama)
    model = llama.Llama2_(cfg).eval()
    ids = torch.tensor([[10, 20, 30]])
    hidden, _, no_cache = model(ids)
    assert hidden.shape == (1, 3, 256)
    assert no_cache == []
    caches = model.init_kv_cache(1, 16, "cpu", torch.float32)
    hidden, _, caches = model(ids, past_key_values=caches)
    assert caches[0][2] == 3  # index advanced past the prefill
    nxt, _, caches = model(torch.tensor([[40]]), past_key_values=caches)
    assert nxt.shape == (1, 1, 256) and caches[0][2] == 4
    assert model.logits(hidden).shape == (1, 184704)


# ----------------------------------------------------------------------
# worker: DiT + VAE shapes (tiny CPU models)
# ----------------------------------------------------------------------
def test_dit_forward_shape(dit, llama):
    cfg = llama.yue2_config(num_hidden_layers=2, hidden_size=256,
                            intermediate_size=512, num_attention_heads=2,
                            num_key_value_heads=1,
                            max_position_embeddings=64)
    model = dit.YuE2DiT(config=cfg).eval()
    frames, ar_len = 4, 6
    x = torch.randn(1, 64, frames)
    t = torch.tensor([0.5])
    context = torch.randn(1, ar_len, 2, 2, 1, 128)
    chunks = [(0, frames, 0, ar_len)]
    with torch.no_grad():
        out = model(x, t, context, chunks)
    assert out.shape == (1, 64, frames)


def test_flow_sigmas_match_comfy_sgm_uniform(dit):
    """Comfy ``normal_scheduler(sgm=True)``: sigma_max -> sigma_min, +0.0."""
    steps = 8
    sigmas = dit.flow_sigmas(steps)
    # linspace(1.0, 0.001, steps + 1)[:-1] with a trailing 0.0 appended
    assert len(sigmas) == steps + 1
    assert sigmas[0] == pytest.approx(dit.FLOW_SIGMA_MAX)
    assert sigmas[-2] == pytest.approx(
        dit.FLOW_SIGMA_MIN + (dit.FLOW_SIGMA_MAX - dit.FLOW_SIGMA_MIN) / steps)
    assert sigmas[-1] == 0.0
    assert all(a > b for a, b in zip(sigmas, sigmas[1:]))


def test_euler_matches_comfy_velocity_integration(dit, llama):
    """Euler is x += v*dsigma with a negative dt, one NFE per step.

    Comfy ``sample_euler``: ``to_d = (x - denoised)/sigma`` and
    ``denoised = x - sigma*v`` (CONST), so ``to_d == v``.
    """
    cfg = llama.yue2_config(num_hidden_layers=1, hidden_size=256,
                            intermediate_size=512, num_attention_heads=2,
                            num_key_value_heads=1,
                            max_position_embeddings=64)
    model = dit.YuE2DiT(config=cfg).eval()
    frames = 4
    context = torch.randn(1, 6, 2, 2, 1, 128)
    chunks = [(0, frames, 0, 6)]
    seen = []

    def fake(x, t, context, yue2_chunks):
        seen.append(float(t))
        return torch.ones_like(x)

    model.forward = fake
    try:
        out = dit.euler_sample(model, context, chunks, frames, steps=4,
                               seed=3)
    finally:
        del model.forward
    # one NFE per Euler step
    assert len(seen) == 4
    # timesteps handed to the model are the sigmas, descending 1 -> ~0
    assert seen == pytest.approx(dit.flow_sigmas(4)[:-1])
    # constant velocity v=1 integrated over a total span of -1.0, so the
    # sample must land exactly 1.0 below where it started
    g = torch.Generator().manual_seed(3)
    x0 = torch.randn((1, 64, frames), generator=g)
    assert torch.allclose(out.float(), x0 - 1.0, atol=1e-5)


def test_dpm2_uses_midpoint_and_lands_on_x0(dit, llama):
    """dpm_2 takes a geometric midpoint, and the final step lands on x0."""
    cfg = llama.yue2_config(num_hidden_layers=1, hidden_size=256,
                            intermediate_size=512, num_attention_heads=2,
                            num_key_value_heads=1,
                            max_position_embeddings=64)
    model = dit.YuE2DiT(config=cfg).eval()
    frames = 4
    context = torch.randn(1, 6, 2, 2, 1, 128)
    chunks = [(0, frames, 0, 6)]
    seen = []
    calls = {"n": 0}

    def fake(x, t, context, yue2_chunks):
        seen.append(float(t))
        calls["n"] += 1
        return torch.ones_like(x)

    model.forward = fake
    try:
        out = dit.dpm2_sample(model, context, chunks, frames, steps=4,
                              seed=3)
    finally:
        del model.forward
    sigmas = dit.flow_sigmas(4)
    # 2 NFE for each of the first steps-1, 1 NFE for the final step
    assert calls["n"] == 2 * (len(sigmas) - 2) + 1
    # midpoints are geometric means of neighbouring sigmas
    for i in range(len(sigmas) - 2):
        expected = math.sqrt(sigmas[i] * sigmas[i + 1])
        assert seen[2 * i + 1] == pytest.approx(expected, rel=1e-6)
    # constant velocity v=1 integrates to a total span of -1.0
    g = torch.Generator().manual_seed(3)
    x0 = torch.randn((1, 64, frames), generator=g)
    assert torch.allclose(out.float(), x0 - 1.0, atol=1e-5)


def test_sample_dispatch_rejects_unknown(dit, llama):
    cfg = llama.yue2_config(num_hidden_layers=1, hidden_size=256,
                            intermediate_size=512, num_attention_heads=2,
                            num_key_value_heads=1,
                            max_position_embeddings=64)
    model = dit.YuE2DiT(config=cfg).eval()
    with pytest.raises(ValueError, match="unknown sampler"):
        dit.sample(model, torch.randn(1, 6, 2, 2, 1, 128), [(0, 4, 0, 6)], 4,
                   steps=2, sampler="karras")


def test_sampler_names_stay_in_sync_with_dit(cli, dit):
    """cli.py must stay torch-free, so it mirrors dit.SAMPLERS literally."""
    assert set(cli._SAMPLERS) == set(dit.SAMPLERS)
    spec = _spec(sampler="dpm_2")
    cli.validate_spec(spec)          # accepted
    with pytest.raises(ValueError, match="sampler must be one of"):
        cli.validate_spec(_spec(sampler="karras"))


def test_timestep_embedding_is_cos_then_sin(dit):
    """Comfy concatenates [cos, sin]; the halves must not be transposed.

    A transposed embedding silently detaches the DiT from the noise level,
    which destroys the diffusion trajectory and drops the fundamental.
    """
    t = torch.tensor([0.5])
    emb = dit.timestep_embedding(t, 256)
    assert emb.shape == (1, 256)
    half = 128
    cos_part, sin_part = emb[:, :half], emb[:, half:]
    # t=0.5 -> cos(0.5*1) = 0.8776 in the leading half
    assert float(cos_part[0, 0]) == pytest.approx(
        float(torch.cos(torch.tensor(0.5))), abs=1e-5)
    assert float(sin_part[0, 0]) == pytest.approx(
        float(torch.sin(torch.tensor(0.5))), abs=1e-5)
    # and the embedding must actually depend on t
    a = dit.timestep_embedding(torch.tensor([0.1]), 256)
    b = dit.timestep_embedding(torch.tensor([0.9]), 256)
    assert (a - b).abs().max() > 0.1


def test_written_wav_is_float_not_pcm16(tmp_path):
    """libsndfile defaults .wav to PCM_16, which clips >1.0 peaks."""
    import numpy as np
    import soundfile as sf

    path = tmp_path / "song.wav"
    data = np.array([[1.4, -1.4], [0.25, -0.25]], dtype=np.float32)
    sf.write(str(path), data, 48000, subtype="FLOAT")
    assert sf.info(str(path)).subtype == "FLOAT"
    back, _ = sf.read(str(path), dtype="float32", always_2d=True)
    assert back.max() == pytest.approx(1.4, abs=1e-6)


def test_dit_frame_mismatch_raises(dit, llama):
    cfg = llama.yue2_config(num_hidden_layers=1, hidden_size=256,
                            intermediate_size=512, num_attention_heads=2,
                            num_key_value_heads=1,
                            max_position_embeddings=64)
    model = dit.YuE2DiT(config=cfg).eval()
    with pytest.raises(ValueError, match="must match"):
        model(torch.randn(1, 64, 4), torch.tensor([0.5]),
              torch.randn(1, 6, 1, 2, 1, 128), [(0, 5, 0, 6)])
    with pytest.raises(ValueError, match="64 channels"):
        model(torch.randn(1, 8, 4), torch.tensor([0.5]),
              torch.randn(1, 6, 1, 2, 1, 128), [(0, 4, 0, 6)])


def test_vae_decode_shape(vae):
    model = vae.AudioOobleckVAE().eval()
    with torch.no_grad():
        out = model.decode(torch.randn(1, 64, 4))
    assert out.shape == (1, 2, 4 * vae.UPSAMPLE)
    assert vae.NATIVE_SR == 48000 and vae.UPSAMPLE == 1920
    with pytest.raises(ValueError, match="latents must be"):
        model.decode(torch.randn(1, 32, 4))


def test_vae_tiled_matches_full_on_short_input(vae):
    model = vae.AudioOobleckVAE().eval()
    lat = torch.randn(1, 64, 8)
    with torch.no_grad():
        assert torch.equal(vae.decode_tiled(model, lat),
                           model.decode(lat))


def test_vae_tiled_long_input_exact_length(vae):
    model = vae.AudioOobleckVAE().eval()
    frames, core, halo = 40, 16, 4
    lat = torch.randn(1, 64, frames)
    with torch.no_grad():
        t1 = vae.decode_tiled(model, lat, core=core, halo=halo)
        t2 = vae.decode_tiled(model, lat, core=core, halo=halo)
    assert t1.shape == (1, 2, frames * vae.UPSAMPLE)
    assert torch.isfinite(t1).all()
    assert torch.equal(t1, t2)  # deterministic tiling


def test_oom_helper_tags_stage():
    pipeline = _load("pipeline")
    err = pipeline._oom("vae-decode", "6000 frames")
    assert "[vae-decode]" in str(err) and "6000 frames" in str(err)


def test_ar_preflight_passes_small():
    pipeline = _load("pipeline")
    est = pipeline.check_ar_fit(300, 6000, dual=False)
    assert est < pipeline.AR_VRAM_BUDGET_GB


def test_ar_preflight_warns_never_blocks():
    # The reference handles large prompts on 8 GB, so the estimate must
    # warn (not raise): the reported 4270-token + 6000-budget case.
    pipeline = _load("pipeline")
    est = pipeline.check_ar_fit(4270, 6000, dual=False)
    assert est > pipeline.AR_VRAM_BUDGET_GB
    est2 = pipeline.check_ar_fit(4270, 6000, dual=True)
    assert est2 > est


# ----------------------------------------------------------------------
# worker: AR generate smoke (tiny CPU model, greedy)
# ----------------------------------------------------------------------
def test_generate_loop_smoke(ar, llama):
    cfg = _tiny_config(llama)
    model = llama.Llama2_(cfg).eval()
    device, dtype = "cpu", torch.float32
    with torch.no_grad():
        history, truncated = ar._generate(
            model, [ar.EOD, 10, 20, ar.ABC_START], seed=0,
            max_tokens=8, phase="abc", device=device, dtype=dtype,
            negative=[ar.EOD, 30], cfg_scale=1.0, legacy_off=False,
            temperature=0, top_p=1.0, top_k=0, repetition_penalty=1.0,
            penalty_window=10, min_tokens=0)
    assert isinstance(history, list) and len(history) <= 8
    assert isinstance(truncated, bool)


# ----------------------------------------------------------------------
# worker: embedded tokenizer (needs the fetched checkpoint)
# ----------------------------------------------------------------------
@needs_checkpoint
def test_embedded_tokenizer_roundtrip(ar):
    tok = ar.YuE2Tokenizer.from_checkpoint(REPO_ROOT / "models/YuE2-Torch")
    ids = tok.encode("English, indie pop")
    assert len(ids) > 0 and all(0 <= i < 184704 for i in ids)
    assert "indie" in tok.decode(ids)


@needs_checkpoint
def test_embedded_tokenizer_prompt_layout(ar):
    tok = ar.YuE2Tokenizer.from_checkpoint(REPO_ROOT / "models/YuE2-Torch")
    ar.bind_tokenizer(tok)
    try:
        bundle = ar.tokenize_with_weights("pop", "[Verse]\nla\n", "",
                                          cot="full", seed=7, max_tokens=6000)
    finally:
        ar.bind_tokenizer(None)
    assert bundle["prefix"][0] == ar.EOD
    assert bundle["prefix"][-1] == ar.ABC_START
    assert bundle["negative"][0] == ar.EOD


# ----------------------------------------------------------------------
# worker: fetch table
# ----------------------------------------------------------------------
def test_fetch_table_weights_only():
    fetch = _load("fetch")
    names = [rel for rel, _, _, _ in fetch.TABLE]
    assert names == ["yue2_3b_bf16.safetensors"]


# ----------------------------------------------------------------------
# worker: zero-Anode-import grep
# ----------------------------------------------------------------------
def test_zero_anode_imports():
    pat = re.compile(
        r"\b(import|from)\s+(base|core|audiocpp_backend|audio_io|plugin_system)\b")
    offenders = []
    for path in sorted(WORKER_DIR.glob("*.py")):
        text = path.read_text(encoding="utf-8")
        if pat.search(text):
            offenders.append(path.name)
    assert offenders == []


# ----------------------------------------------------------------------
# worker: run.py --help fast, no torch at top
# ----------------------------------------------------------------------
def test_run_help_fast():
    src = (WORKER_DIR / "run.py").read_text(encoding="utf-8")
    head = src.split("def main")[0]
    assert "import torch" not in head
    env_python = sys.executable
    t0 = time.monotonic()
    proc = subprocess.run([env_python, str(WORKER_DIR / "run.py"), "--help"],
                          capture_output=True, text=True, timeout=60)
    assert proc.returncode == 0
    assert "--spec" in proc.stdout
    assert time.monotonic() - t0 < 30


# ----------------------------------------------------------------------
# node: engine switch
# ----------------------------------------------------------------------
def _torch_model_dir(tmp_path):
    d = tmp_path / "YuE2-Torch"
    d.mkdir(parents=True, exist_ok=True)
    (d / "yue2_3b_bf16.safetensors").write_bytes(b"weights-stub")
    return d


def _lyrics(tmp_path):
    p = tmp_path / "lyrics.txt"
    p.write_text("[Verse]\nla\n", encoding="utf-8")
    return p


def _make_torch_node(node_cls, tmp_path):
    node = node_cls()
    node.params["torch_model_dir"].set(str(_torch_model_dir(tmp_path)))
    node.params["torch_model_dir"].sync()
    node.params["lyrics_file"].set(str(_lyrics(tmp_path)))
    node.params["lyrics_file"].sync()
    node.params["engine"].set(1)
    node.params["engine"].sync()
    return node


def test_engine_defaults_sidecar(node_cls):
    node = node_cls()
    assert list(node.params["engine"].meta["items"]) == [
        "sidecar (GGUF)", "torch (BF16)"]
    assert int(node.params["engine"].value) == 0
    assert node._engine_idx() == 0


def test_torch_spec_maps_cap_and_targets_runpy(node_cls, tmp_path):
    node = _make_torch_node(node_cls, tmp_path)
    spec = node._build_spec()
    assert spec["engine"] == "torch"
    assert spec["semantic_max_tokens"] == 6000
    assert spec["run_py"].endswith("run.py")
    assert spec["worker_python"]


def test_torch_argv_targets_runpy_with_spec_file(node_cls, tmp_path):
    node = _make_torch_node(node_cls, tmp_path)
    spec = node._build_spec()
    cli_mod = node._torch_cli()
    argv = cli_mod.build_argv(spec["worker_python"], spec["run_py"],
                              spec_path="run/spec.json",
                              out_wav="run/song.wav", out_dir="run")
    assert argv[1].endswith("run.py")
    assert "--spec" in argv and "--metrics" in argv


def test_lora_on_torch_raises(node_cls, tmp_path):
    node = _make_torch_node(node_cls, tmp_path)
    node.params["ar_lora"].set(str(tmp_path / "a.safetensors"))
    node.params["ar_lora"].sync()
    (tmp_path / "a.safetensors").write_bytes(b"x")
    with pytest.raises(ValueError, match="torch engine"):
        node._build_spec()


def test_sidecar_only_opts_warn_ignored_profile_ignored(
        node_cls, tmp_path):
    node = _make_torch_node(node_cls, tmp_path)
    # profile is ignored: even BF16 passes.
    node.params["profile"].set(2)
    node.params["profile"].sync()
    spec = node._build_spec()
    assert spec["engine"] == "torch"
    assert spec["ignored_sidecar_opts"] == []
    # Stale sidecar perf knobs warn-and-ignore instead of blocking.
    node.params["weight_type"].set(1)
    node.params["weight_type"].sync()
    node.params["attention"].set(2)
    node.params["attention"].sync()
    spec = node._build_spec()
    assert spec["ignored_sidecar_opts"] == [
        "weight_type=q4_0 (low VRAM)", "attention=eager"]
    warns = node._spec_warnings(spec)
    assert len(warns) == 1 and "ignores sidecar-only" in warns[0]
    # Defaults stay quiet.
    node.params["weight_type"].set(0)
    node.params["weight_type"].sync()
    node.params["attention"].set(0)
    node.params["attention"].sync()
    assert node._spec_warnings(node._build_spec()) == []


def test_torch_missing_weights_raise(node_cls, tmp_path):
    node = node_cls()
    empty = tmp_path / "empty-torch"
    empty.mkdir()
    node.params["torch_model_dir"].set(str(empty))
    node.params["torch_model_dir"].sync()
    node.params["lyrics_file"].set(str(_lyrics(tmp_path)))
    node.params["lyrics_file"].sync()
    node.params["engine"].set(1)
    node.params["engine"].sync()
    with pytest.raises(ValueError, match="Torch weights"):
        node._build_spec()


def test_torch_fetch_payload_skips_runtime(node_cls, tmp_path):
    import audiocpp_backend as backend
    # Spec table lists the weights file only (tokenizer is embedded).
    empty = tmp_path / "empty-torch"
    empty.mkdir()
    assert [s.dest.name for s in
            backend.yue2_torch_fetch_specs(empty)] == [
        "yue2_3b_bf16.safetensors"]
    node = _make_torch_node(node_cls, tmp_path)
    payload = node._build_fetch_payload()
    assert payload["runtime"] == []
    assert "torch engine" in payload["runtime_note"]


def test_torch_complete_tags_torch_and_cut(node_cls):
    from base import CHANNELS, SAMPLE_RATE
    node = node_cls()
    data = torch.linspace(0, 1, SAMPLE_RATE, dtype=torch.float32
                          ).unsqueeze(0).repeat(CHANNELS, 1).contiguous()
    node.on_nrt_complete("gen", True, {
        "audio": data, "wav": "t.wav", "metrics": {"rtf": 1.5},
        "seed": 3, "cot": "off", "engine": "torch",
        "semantic": {"frames": 6000, "truncated": True},
        "max_duration": 240.0})
    assert "[torch]" in node._status_detail
    assert "cut at 240s cap" in node._status_detail


def test_save_load_round_trips_engine(node_cls, tmp_path):
    node = _make_torch_node(node_cls, tmp_path)
    snapshot = node.to_dict()
    node2 = node_cls()
    node2.load_state(snapshot)
    assert int(node2.params["engine"].value) == 1


def test_old_patch_without_engine_loads_as_sidecar(node_cls):
    node = node_cls()
    snapshot = node.to_dict()
    # to_dict params shape: {name: value}; dropping engine keys simulates
    # a patch saved before the engine switch.
    snapshot["params"].pop("engine", None)
    snapshot["params"].pop("torch_model_dir", None)
    node2 = node_cls()
    node2.load_state(snapshot)
    assert node2._engine_idx() == 0


# ----------------------------------------------------------------------
# node: download label + torch_model_dir not in widget keys
# ----------------------------------------------------------------------
def test_torch_model_dir_not_in_widget_keys(node_cls):
    mod = _live_node_module()
    assert "torch_model_dir" not in mod.YuE2Widget.PARAM_KEYS
    assert "engine" in mod.YuE2Widget.PARAM_KEYS


@pytest.fixture(scope="module")
def qapp():
    import os
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PySide6.QtCore import QCoreApplication
    from PySide6.QtWidgets import QApplication
    inst = QCoreApplication.instance()
    if inst is None:
        return QApplication([])
    if isinstance(inst, QApplication):
        return inst
    pytest.skip("a bare QCoreApplication is active; QWidget tests cannot run")


def test_download_label_follows_engine(qapp, node_cls):
    mod = _live_node_module()

    class _StubProxy:
        def __init__(self):
            self.calls = []
            self.widgets = {}
            self.node_item = type("NI", (), {"params": {"engine": {"value": 0}}})()

        def set_parameter(self, name, value):
            self.calls.append((name, value))

        def create_param_widget(self, name):
            from PySide6.QtWidgets import QWidget

            class _W(QWidget):
                def update_from_backend(self, value):
                    self._backend_value = value

            w = _W()
            self.widgets[name] = w
            return w

    widget = mod.YuE2Widget(_StubProxy())
    assert "3.8" in widget.btn_download.text()
    widget.update_from_params({"engine": 1})
    assert "7.8" in widget.btn_download.text()
    widget.update_from_params({"engine": 0})
    assert "3.8" in widget.btn_download.text()


gpu_e2e = pytest.mark.skipif(
    os.environ.get("ANODE_YUE2_TORCH_GPU") != "1" or not CHECKPOINT.exists()
    or not torch.cuda.is_available(),
    reason="needs ANODE_YUE2_TORCH_GPU=1 + fetched weights + CUDA")


@gpu_e2e
def test_gpu_e2e_cot_off_short_song(tmp_path):
    """Full user path with real weights on CUDA: run.py main -> pipeline.

    Minimal work (cot=off, 24 semantic frames, 1 DiT step) but maximal
    path coverage: embedded tokenizer, bf16 AR prefill + dual-batch CFG
    decode, KV-conditioning stack, bf16 DiT chunk forward, fp32 VAE
    decode, soundfile write, semantic.json. This is the layer that was
    missing — every prior torch-backend failure came from code only
    reachable here.
    """
    import json
    run_mod = _load("run")
    lyrics = tmp_path / "lyrics.txt"
    lyrics.write_text("[Verse]\nSoft morning light.\n", encoding="utf-8")
    model_dir = REPO_ROOT / "models" / "YuE2-Torch"
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    spec = {"lyrics": lyrics.read_text(encoding="utf-8"),
            "style": "English, indie pop",
            "cot": "off", "seed": 831001, "semantic_max_tokens": 24,
            "num_inference_steps": 1, "model_dir": str(model_dir)}
    spec_path = run_dir / "spec.json"
    spec_path.write_text(json.dumps(spec), encoding="utf-8")
    out_wav = run_dir / "song.wav"
    rc = run_mod.main(["--spec", str(spec_path), "--out", str(out_wav),
                       "--out-dir", str(run_dir), "--metrics"])
    assert rc == 0
    assert out_wav.exists()
    import soundfile as sf
    data, sr = sf.read(str(out_wav), dtype="float32", always_2d=True)
    assert sr == 48000 and data.shape[1] == 2 and data.shape[0] > 0
    semantic = json.loads((run_dir / "semantic.json").read_text(
        encoding="utf-8"))
    assert semantic["frames"] == data.shape[0] // 1920
    assert not (run_dir / "score.abc").exists()  # cot=off plans nothing
