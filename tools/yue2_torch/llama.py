"""Llama/Qwen3 transformer core for the YuE2 torch worker.

Exact port of the YuE2-relevant subset of ComfyUI
``comfy/text_encoders/llama.py`` (post-PR-#16250 ``master``; revision pinned
in README). Comfy-only machinery is stripped: no ``comfy.*`` /
``comfy_kitchen`` imports, no CUDA graphs / VBAR / spec-decode, no
sliding-window variants, plain ``nn.Linear``/``nn.Embedding`` (Comfy ``ops``
default to the same dense modules here), ``torch.sdpa`` for attention.

What is kept, exactly:
``Qwen3_8BConfig`` fields (+ ``yue2_config()`` mirroring Comfy
``comfy/ldm/yue2/model.py::model_config``), ``RMSNorm``, ``FixedKV`` (plain
tuple caches ``(key, value, index)`` only), ``precompute_freqs_cis`` /
``rope_matrix`` / ``apply_rope`` (split-half NeoX rotation), ``Attention``
(merged-qkv, gemma3 q/k norms), ``MLP`` (merged gate_up, silu), 
``TransformerBlock`` (input/post norms, ``(x, present)`` return),
``Llama2_`` (embed/layers/norm/lm_head, ``init_kv_cache``,
causal+padding mask construction, ``(hidden, None, caches)`` return).

Module attribute names intentionally match the checkpoint key suffixes
(``text_encoders.model.*`` for AR, ``model.diffusion_model.model.*`` for
the DiT) so weights load with ``strict=True``.
"""

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class Qwen3_8BConfig:
    vocab_size: int = 151936
    hidden_size: int = 4096
    intermediate_size: int = 12288
    num_hidden_layers: int = 36
    num_attention_heads: int = 32
    num_key_value_heads: int = 8
    max_position_embeddings: int = 40960
    rms_norm_eps: float = 1e-6
    rope_theta: float = 1000000.0
    transformer_type: str = "llama"
    head_dim: int = 128
    rms_norm_add: bool = False
    mlp_activation: str = "silu"
    qkv_bias: bool = False
    rope_dims = None
    q_norm = "gemma3"
    k_norm = "gemma3"
    rope_scale = None
    final_norm: bool = True
    lm_head: bool = True
    fixed_kv: bool = False
    merged_qkv: bool = False
    merged_mlp: bool = False
    stop_tokens = (151643, 151645)


def yue2_config(**overrides):
    """Mirror Comfy ``comfy/ldm/yue2/model.py::model_config``.

    Shared by the AR planner and the NAR DiT: vocab 184704 / hidden 2048 /
    inter 6144 / 28 layers / 16 Q + 8 KV heads / merged qkv+mlp. Verified
    against the checkpoint (28 layer stacks, embed 184704x2048).
    """
    base = {
        "vocab_size": 184704, "hidden_size": 2048,
        "intermediate_size": 6144, "num_hidden_layers": 28,
        "num_attention_heads": 16, "num_key_value_heads": 8,
        "max_position_embeddings": 24576,
        "merged_qkv": True, "merged_mlp": True,
    }
    base.update(overrides)
    return Qwen3_8BConfig(**base)


class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.empty(dim))

    def forward(self, x):
        return (x * torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True)
                                + self.eps) * self.weight).to(x.dtype)


def precompute_freqs_cis(head_dim, position_ids, theta=1000000.0,
                         device=None):
    """Return (cos, sin), each (1, 1, L, head_dim//2).

    Single-theta path of Comfy ``precompute_freqs_cis`` (YuE2 never uses
    rope_scale/rope_dims/MRoPE). Comfy builds the full-width embedding by
    duplicating freqs, so both halves are identical and the half-width
    pair here is exactly equivalent for the split-half rotation.
    """
    inv_freq = 1.0 / (theta ** (torch.arange(0, head_dim, 2,
                                                    device=device).float()
                                / head_dim))
    freqs = (inv_freq[None, :, None].float()
             @ position_ids[:, None, :].float()).transpose(1, 2)
    emb = torch.cat((freqs, freqs), dim=-1)
    half = emb.shape[-1] // 2
    cos, sin = emb.cos()[..., :half], emb.sin()[..., :half]
    if device is not None:
        cos, sin = cos.to(device), sin.to(device)
    return cos.unsqueeze(1), sin.unsqueeze(1)


def rope_matrix(freqs_cis):
    cos, sin = freqs_cis
    half = sin.shape[-1]
    matrix = torch.stack((cos, -sin, sin, cos), dim=-1)
    return matrix.reshape(*matrix.shape[:-1], 2, 2)


def apply_rope(x, freqs_cis):
    """Split-half (NeoX-style) rotary embedding.

    Manual form of ``comfy_kitchen.apply_rope_split_half``: with the 2x2
    blocks (cos, -sin / sin, cos) this is
    ``cat([x1*cos - x2*sin, x1*sin + x2*cos])`` over channel halves.
    ``x`` is (B, H, L, D); cos/sin broadcast as (1, 1, L, D/2) and are
    cast to ``x``'s dtype first — without this, bf16 activations meet
    float32 tables and downstream sdpa sees mixed dtypes.
    """
    cos, sin = freqs_cis
    cos, sin = cos.to(x.dtype), sin.to(x.dtype)
    half = x.shape[-1] // 2
    x1, x2 = x[..., :half], x[..., half:]
    return torch.cat([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)


class FixedKV:
    """Placeholder type only: this worker uses plain ``(key, value, index)``
    tuple caches (Comfy's non-graph path). Graph FixedKV is out of scope."""


class Attention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.num_heads = config.num_attention_heads
        self.num_kv_heads = config.num_key_value_heads
        self.hidden_size = config.hidden_size
        self.head_dim = config.head_dim
        self.inner_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        self.merged_qkv = getattr(config, "merged_qkv", False)
        if self.merged_qkv:
            self.qkv_proj = nn.Linear(config.hidden_size,
                                      self.inner_size + self.kv_size * 2,
                                      bias=config.qkv_bias)
        else:  # pragma: no cover - YuE2 always uses merged_qkv
            raise NotImplementedError("YuE2 requires merged_qkv")
        self.o_proj = nn.Linear(self.inner_size, config.hidden_size,
                                bias=False)
        self.q_norm = None
        self.k_norm = None
        if config.q_norm == "gemma3":
            self.q_norm = RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        if config.k_norm == "gemma3":
            self.k_norm = RMSNorm(self.head_dim, eps=config.rms_norm_eps)

    def forward(self, hidden_states, attention_mask=None, freqs_cis=None,
                past_key_value=None):
        b, t, _ = hidden_states.shape
        xq, xk, xv = self.qkv_proj(hidden_states).split(
            (self.inner_size, self.kv_size, self.kv_size), dim=-1)
        xq = xq.view(b, t, self.num_heads, self.head_dim).transpose(1, 2)
        xk = xk.view(b, t, self.num_kv_heads, self.head_dim).transpose(1, 2)
        xv = xv.view(b, t, self.num_kv_heads, self.head_dim).transpose(1, 2)
        if self.q_norm is not None:
            xq = self.q_norm(xq)
        if self.k_norm is not None:
            xk = self.k_norm(xk)
        xq, xk = apply_rope(xq, freqs_cis), apply_rope(xk, freqs_cis)

        present = None
        if past_key_value is not None:
            # Plain tuple cache path (Comfy's non-FixedKV branch).
            index = 0
            if len(past_key_value) > 0:
                past_key, past_value, index = past_key_value
                if past_key.shape[2] >= index + xk.shape[2]:
                    past_key[:, :, index:index + xk.shape[2]] = xk
                    past_value[:, :, index:index + xv.shape[2]] = xv
                    xk = past_key[:, :, :index + t]
                    xv = past_value[:, :, :index + t]
                    present = (past_key, past_value, index + t)
                else:
                    xk = torch.cat((past_key[:, :, :index], xk), dim=2)
                    xv = torch.cat((past_value[:, :, :index], xv), dim=2)
                    present = (xk, xv, index + t)
            else:
                present = (xk, xv, t)

        if self.num_heads != self.num_kv_heads:
            rep = self.num_heads // self.num_kv_heads
            k, v = xk.repeat_interleave(rep, dim=1), xv.repeat_interleave(rep, dim=1)
        else:
            k, v = xk, xv
        out = F.scaled_dot_product_attention(xq, k, v, attn_mask=attention_mask)
        return self.o_proj(out.transpose(1, 2).reshape(b, t, -1)), present


class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.merged_mlp = getattr(config, "merged_mlp", False)
        if not self.merged_mlp:  # pragma: no cover - YuE2 always merged
            raise NotImplementedError("YuE2 requires merged_mlp")
        self.gate_up_proj = nn.Linear(config.hidden_size,
                                      config.intermediate_size * 2,
                                      bias=False)
        self.down_proj = nn.Linear(config.intermediate_size,
                                   config.hidden_size, bias=False)

    def forward(self, x):
        gate, up = self.gate_up_proj(x).chunk(2, dim=-1)
        return self.down_proj(F.silu(gate) * up)


class TransformerBlock(nn.Module):
    def __init__(self, config, index=0):
        super().__init__()
        self.index = index
        self.self_attn = Attention(config)
        self.mlp = MLP(config)
        self.input_layernorm = RMSNorm(config.hidden_size,
                                       eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size,
                                                eps=config.rms_norm_eps)

    def forward(self, x, attention_mask=None, freqs_cis=None,
                past_key_value=None):
        residual = x
        x, present = self.self_attn(
            self.input_layernorm(x), attention_mask=attention_mask,
            freqs_cis=freqs_cis, past_key_value=past_key_value)
        x = residual + x
        residual = x
        x = self.post_attention_layernorm(x)
        x = self.mlp(x)
        return residual + x, present


class Llama2_(nn.Module):
    """AR planner / KV-cache owner (Comfy ``Llama2_`` minus graphs)."""

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.embed_tokens = nn.Embedding(config.vocab_size,
                                         config.hidden_size)
        self.layers = nn.ModuleList(
            TransformerBlock(config, index=i)
            for i in range(config.num_hidden_layers))
        self.norm = RMSNorm(config.hidden_size,
                            eps=config.rms_norm_eps) \
            if config.final_norm else None
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size,
                                 bias=False) if config.lm_head else None

    @staticmethod
    def get_past_len(past_key_values):
        return past_key_values[0][2] if past_key_values else 0

    def init_kv_cache(self, batch, capacity, device, dtype):
        caches = []
        for _ in range(self.config.num_hidden_layers):
            key = torch.empty((batch, self.config.num_key_value_heads,
                               capacity, self.config.head_dim),
                              device=device, dtype=dtype)
            caches.append((key, torch.empty_like(key), 0))
        return caches

    def compute_freqs_cis(self, position_ids, device):
        return precompute_freqs_cis(self.config.head_dim, position_ids,
                                    self.config.rope_theta, device=device)

    def forward(self, input_ids, position_ids=None, attention_mask=None,
                past_key_values=None):
        """Embed + stack + norm. Returns ``(hidden, None, caches)``.

        ``attention_mask`` is (B, L) padding validity (1 = keep); a causal
        mask is added whenever the step extends the sequence (``seq > 1``),
        mirroring Comfy. ``past_key_values`` is a per-layer list of
        ``(key, value, index)`` tuples; without it ``caches`` is ``[]``.
        """
        x = self.embed_tokens(input_ids)
        seq_len = x.shape[1]
        past_len = self.get_past_len(past_key_values) \
            if past_key_values else 0
        if position_ids is None:
            position_ids = torch.arange(past_len, past_len + seq_len,
                                        device=x.device).unsqueeze(0)
        freqs_cis = self.compute_freqs_cis(position_ids, x.device)

        mask = None
        if attention_mask is not None:
            keep = attention_mask.to(x.dtype).reshape(
                attention_mask.shape[0], 1, 1, -1)
            mask = (1.0 - keep) * (torch.finfo(x.dtype).min / 4)
            mask = mask.expand(attention_mask.shape[0], 1, seq_len,
                               attention_mask.shape[-1]).contiguous()
        if seq_len > 1:
            causal = torch.empty(past_len + seq_len, past_len + seq_len,
                                 dtype=x.dtype, device=x.device
                                 ).fill_(torch.finfo(x.dtype).min / 4).triu_(1)
            causal = causal[-seq_len:, :]
            mask = causal if mask is None else mask + causal

        next_caches = list(past_key_values) if past_key_values else []
        for i, layer in enumerate(self.layers):
            past_kv = past_key_values[i] if past_key_values else None
            x, current = layer(x, attention_mask=mask, freqs_cis=freqs_cis,
                               past_key_value=past_kv)
            if next_caches:
                next_caches[i] = current
        if self.norm is not None:
            x = self.norm(x)
        return x, None, next_caches

    def logits(self, hidden):
        """Last-position logits as (B, V) (Comfy ``BaseQwen3.logits``)."""
        return self.lm_head(hidden[:, -1])
