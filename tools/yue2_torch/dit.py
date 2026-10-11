"""NAR diffusion transformer (acoustic branch) for the YuE2 torch worker.

Exact port of ComfyUI ``comfy/ldm/yue2/model.py`` (post-PR-#16250
``master``; revision pinned in README): shared ``yue2_config`` (vocab
184704 / hidden 2048 / inter 6144 / 28 layers / 16 Q + 8 KV heads /
merged qkv+mlp), ``TimestepEmbedder`` (sinusoidal-256 MLP), learned
``AudioPositionEmbedding``, and the chunked ``YuE2.forward`` (pad +1
frame each side, vae2llm + time + pos, prefix-KV permute, layer loop
with ``(key, value, ar_length)`` tuple caches, llm2vae slice).

Stripped: ``comfy.*`` imports, prefetch queues, interruption hooks,
``optimized_attention`` dispatch (plain ``torch.sdpa`` inside
``llama.py``). Linear biases match the checkpoint exactly (vae2llm,
llm2vae and both timestep MLPs carry bias; transformer linears do not).

Provides Euler and DPM2 flow samplers over the DiT. DPM2 with 32 steps is
the worker default and matches the ComfyUI YuE2 blueprint; Euler remains an
explicit lower-cost option. Both use the ``ModelSamplingDiscreteFlow``
schedule (sigmas 1.0 -> 0.0, model timestep == sigma) and deterministic
``torch.Generator(seed)`` noise.
"""

import math

import torch
import torch.nn as nn

try:
    from .llama import (RMSNorm, TransformerBlock, precompute_freqs_cis,
                        yue2_config)
except ImportError:  # standalone file loading (tests import by path)
    import importlib.util as _ilu
    import sys as _sys
    from pathlib import Path as _Path

    def _sibling(_name):
        _mod_name = f"yue2_torch_{_name}"
        _mod = _sys.modules.get(_mod_name)
        if _mod is None:
            _spec = _ilu.spec_from_file_location(
                _mod_name, str(_Path(__file__).resolve().parent
                               / f"{_name}.py"))
            _mod = _ilu.module_from_spec(_spec)
            _sys.modules[_mod_name] = _mod
            _spec.loader.exec_module(_mod)
        return _mod

    _llama = _sibling("llama")
    RMSNorm, TransformerBlock = _llama.RMSNorm, _llama.TransformerBlock
    precompute_freqs_cis, yue2_config = (_llama.precompute_freqs_cis,
                                        _llama.yue2_config)


def timestep_embedding(t, dim=256, max_period=10000):
    """Sinusoidal timestep embedding (Comfy diffusionmodules.util).

    Order matters: Comfy concatenates ``[cos(args), sin(args)]``.
    """
    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period) * torch.arange(half, dtype=torch.float32,
                                             device=t.device) / half)
    args = t.float()[:, None] * freqs[None, :]
    return torch.cat([torch.cos(args), torch.sin(args)], dim=-1)


class TimestepEmbedder(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(256, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size),
        )

    def forward(self, t, dtype):
        return self.mlp(timestep_embedding(t, 256).to(dtype))


class AudioPositionEmbedding(nn.Module):
    def __init__(self, frames, hidden_size):
        super().__init__()
        self.register_buffer("pe", torch.empty(frames, hidden_size))

    def forward(self, length, x):
        return self.pe[:length].to(x.dtype)


class YuE2DiT(nn.Module):
    def __init__(self, config=None):
        super().__init__()
        self.config = config or yue2_config()
        config = self.config
        self.model = nn.Module()
        self.model.layers = nn.ModuleList(
            TransformerBlock(config, i)
            for i in range(config.num_hidden_layers))
        self.model.norm = RMSNorm(config.hidden_size,
                                    eps=config.rms_norm_eps)
        self.vae2llm = nn.Linear(64, config.hidden_size)
        self.llm2vae = nn.Linear(config.hidden_size, 64)
        self.time_embedder = TimestepEmbedder(config.hidden_size)
        self.latent_pos_embed = AudioPositionEmbedding(
            config.max_position_embeddings, config.hidden_size)

    def forward(self, x, timestep, context, yue2_chunks):
        """Denoise latents ``x`` (B, 64, F) conditioned on AR KV ``context``.

        ``timestep`` is (B,) float flow time; ``context`` is the stacked
        per-layer KV ``(B, T, L, 2, K, H)``; ``yue2_chunks`` maps latent
        ranges to KV ranges. Raises ValueError when the latent duration
        does not match the chunk map.
        """
        batch, channels, frames = x.shape
        if channels != 64:
            raise ValueError(
                f"YuE2 latents must have 64 channels, got {channels}")
        if frames != yue2_chunks[-1][1]:
            raise ValueError("YuE2 latent duration must match the seconds "
                             "output of YuE2 Text Encode.")
        config = self.config
        time = self.time_embedder(timestep.to(x.dtype), x.dtype)[:, None]
        output = torch.empty_like(x)
        for start, end, kv_start, kv_end in yue2_chunks:
            ar_length = kv_end - kv_start
            length = end - start + 2
            state = nn.functional.pad(
                x[..., start:end].transpose(1, 2), (0, 0, 1, 1))
            state = self.vae2llm(state) + time \
                + self.latent_pos_embed(length, x)[None]
            positions = torch.arange(ar_length, ar_length + length,
                                     device=x.device)[None]
            rope = precompute_freqs_cis(config.head_dim, positions,
                                        config.rope_theta, device=x.device)
            prefix = context[:, kv_start:kv_end].reshape(
                batch, ar_length, config.num_hidden_layers, 2,
                config.num_key_value_heads, config.head_dim)
            prefix = prefix.permute(2, 3, 0, 4, 1, 5)
            for index, layer in enumerate(self.model.layers):
                state, _ = layer(
                    state, freqs_cis=rope,
                    past_key_value=(prefix[index, 0], prefix[index, 1],
                                    ar_length))
            output[..., start:end] = self.llm2vae(
                self.model.norm(state))[:, 1:-1].transpose(1, 2)
        return output


# ``ModelSamplingDiscreteFlow`` for YuE2: shift 1.0 (default),
# multiplier 1.0 (``sampling_settings = {"multiplier": 1.0}``), so
# ``sigma(t) == timestep(t) == t`` and sigmas run 0.001 .. 1.0.
FLOW_SIGMA_MAX = 1.0
FLOW_SIGMA_MIN = 0.001


def flow_sigmas(steps, sgm=True):
    """Comfy ``normal_scheduler`` sigmas for a FLOW model.

    Mirrors ``comfy.samplers.normal_scheduler(model_sampling, steps, sgm)``:
    ``sgm_uniform`` is ``sgm=True``. For YuE2 ``sigma(t) == t``, so this is
    a linspace from ``sigma_max`` to ``sigma_min`` (plus a trailing 0.0).
    """
    steps = max(2, int(steps))
    start, end = FLOW_SIGMA_MAX, FLOW_SIGMA_MIN
    if sgm:
        # linspace(start, end, steps + 1)[:-1] then append 0.0
        ts = [start + (end - start) * i / steps for i in range(steps)]
    else:
        ts = [start + (end - start) * i / (steps - 1) for i in range(steps)]
    return ts + [0.0]


def _velocity(model, x, sigma, context, yue2_chunks, device):
    """One NFE: model input is the current sample, output is velocity.

    Comfy's ``CONST.calculate_input`` returns the sample unchanged and
    ``CONST.calculate_denoised`` is ``x - sigma * v``, so
    ``to_d = (x - denoised)/sigma == v``.
    """
    t = torch.full((1,), sigma, device=device)
    return model(x, t, context, yue2_chunks)


def _initial_noise(frames, seed, device, dtype):
    generator = torch.Generator(device=device).manual_seed(int(seed))
    x = torch.randn((1, 64, frames), generator=generator, device=device,
                    dtype=torch.float32).to(dtype)
    return x


def euler_sample(model, context, yue2_chunks, frames, *, steps=32, seed=0,
                 device="cpu", dtype=torch.float32):
    """ComfyUI ``k_diffusion.sample_euler`` for a FLOW model.

    ``x_{i+1} = x_i + v_i * (sigma_{i+1} - sigma_i)`` with sigmas
    descending ``sigma_max`` -> 0.0 (negative dt); the final step lands
    exactly on the x0 prediction. One NFE per step.
    """
    x = _initial_noise(frames, seed, device, dtype)
    sigmas = flow_sigmas(steps)
    with torch.inference_mode():
        for i in range(len(sigmas) - 1):
            sigma = sigmas[i]
            dt = sigmas[i + 1] - sigma
            v = _velocity(model, x, sigma, context, yue2_chunks, device)
            x = (x.float() + v.float() * dt).to(dtype)
    return x


def dpm2_sample(model, context, yue2_chunks, frames, *, steps=32, seed=0,
                device="cpu", dtype=torch.float32):
    """ComfyUI ``k_diffusion.sample_dpm_2`` for a FLOW model.

    Second-order DPM-Solver: a midpoint evaluation reuses the previous
    derivative, so it is ~2x the NFE of Euler for the same step count.
    This is the sampler the official YuE2 blueprint ships
    (``KSampler(sampler_name='dpm_2', scheduler='sgm_uniform', steps=32)``).

    The final step (next sigma == 0) degenerates to a plain Euler step so
    the result lands exactly on the x0 prediction.
    """
    x = _initial_noise(frames, seed, device, dtype)
    sigmas = flow_sigmas(steps)
    with torch.inference_mode():
        for i in range(len(sigmas) - 1):
            sigma_hat = sigmas[i]
            sigma_next = sigmas[i + 1]
            d = _velocity(model, x, sigma_hat, context, yue2_chunks, device)
            if sigma_next == 0:
                x = (x.float() + d.float() * (sigma_next - sigma_hat)).to(
                    dtype)
                break
            # geometric midpoint in log-sigma space
            sigma_mid = math.exp(math.log(sigma_hat)
                                 + 0.5 * (math.log(sigma_next)
                                          - math.log(sigma_hat)))
            x_2 = (x.float() + d.float() * (sigma_mid - sigma_hat)).to(dtype)
            d_2 = _velocity(model, x_2, sigma_mid, context, yue2_chunks,
                            device)
            x = (x.float() + d_2.float() * (sigma_next - sigma_hat)).to(
                dtype)
    return x


SAMPLERS = {"euler": euler_sample, "dpm_2": dpm2_sample}


def sample(model, context, yue2_chunks, frames, *, steps=32, seed=0,
           sampler="dpm_2", device="cpu", dtype=torch.float32):
    """Dispatch to a Comfy-named sampler."""
    if sampler not in SAMPLERS:
        raise ValueError(f"unknown sampler {sampler!r}, "
                         f"expected one of {sorted(SAMPLERS)}")
    return SAMPLERS[sampler](model, context, yue2_chunks, frames,
                             steps=steps, seed=seed, device=device,
                             dtype=dtype)
