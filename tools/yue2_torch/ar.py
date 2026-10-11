"""AR (planning/semantic) stage for the YuE2 torch worker.

Exact port of the sampling-relevant parts of ComfyUI
``comfy/text_encoders/yue2.py`` (post-PR-#16250 ``master``; revision pinned
in README). Comfy-only machinery is stripped: no ``comfy.*`` imports, no
progress bars / interruption hooks (cancel is process death), no CUDA
graphs (plain tuple KV caches from ``llama.py``), no VRAM estimators.

Kept exactly: vocabulary/framing constants, ``INSTRUCTIONS`` sentences,
``tokenize_with_weights`` prompt layout (``[EOD] + prompt + [ABC_START]``),
``distribution`` (allowed-range masking, end-token floor, scatter penalty,
top-k/top-p, greedy passthrough), ``chunk_ranges`` (size + error),
dual-batch CFG (``logits[1:] + cfg*(logits[:1]-logits[1:])``), the
two-phase generate loop (ABC plan, then semantic with ``legacy_off`` for
``cot=off``), context-budget clamping, and Comfy sampling defaults
(temperature 1.0, top_p 0.95, top_k 100, repetition 1.2).

The tokenizer is the HuggingFace ``tokenizer.json`` embedded in the
checkpoint as the ``text_encoders.yue2_tokenizer_json`` uint8 tensor
(decoded via the ``tokenizers`` package) — NOT the ``.tiktoken`` sidecar,
whose 151643-entry Qwen base vocab lacks the codec/special ids the
184704-wide AR head emits.
"""

import torch

# NOTE: the model object is injected (see _generate); this module never
# imports llama/dit directly, so it stays importable standalone.

# -- vocabulary / framing constants (Comfy parity) ---------------------------
EOD = 151643
ABC_START = 151847
ABC_END = 151848
MUSIC_START = 151851
MUSIC_END = 151852
CODEC_OFFSET = 151853
CODEC_SIZE = 32768
CONTEXT = 24576
FPS = 25

INSTRUCTIONS = {
    "off": "Generate music with codec tokens from the given conditions.",
    "melody": "Generate a melody-only ABC transcription without chord symbols, then generate music with codec tokens from the given conditions.",
    "full": "Generate a chord-annotated ABC transcription, then generate music with codec tokens from the given conditions.",
}

# Comfy sampling defaults (tokenize_with_weights / generate call sites).
DEFAULT_TEMPERATURE = 1.0
DEFAULT_TOP_P = 0.95
DEFAULT_TOP_K = 100
DEFAULT_REPETITION_PENALTY = 1.2
DEFAULT_PENALTY_WINDOW = 100
SEMANTIC_PENALTY_WINDOW = 50


class YuE2Tokenizer:
    """HuggingFace tokenizer for YuE2 text/ABC streams.

    Resolution order: ``text_encoders.yue2_tokenizer_json`` tensor embedded
    in the checkpoint (exact Comfy source) -> sibling ``tokenizer.json``
    in the model dir (manual export fallback).
    """

    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    @classmethod
    def from_checkpoint(cls, model_dir):
        from pathlib import Path
        from safetensors import safe_open
        path = Path(model_dir) / "yue2_3b_bf16.safetensors"
        if not path.exists():
            raise FileNotFoundError(
                f"torch weights not found: {path} "
                "(run tools/audiocpp/fetch_yue2_torch.py)")
        key = "text_encoders.yue2_tokenizer_json"
        try:
            with safe_open(str(path), framework="pt", device="cpu") as f:
                raw = f.get_tensor(key).numpy().tobytes()
        except Exception as e:
            raise RuntimeError(
                f"embedded tokenizer {key!r} unreadable in {path}: {e}")
        return cls.from_json_bytes(raw)

    @classmethod
    def from_json_bytes(cls, raw):
        from tokenizers import Tokenizer
        if isinstance(raw, (bytes, bytearray)):
            raw = bytes(raw).decode("utf-8")
        return cls(Tokenizer.from_str(raw))

    @classmethod
    def from_json_file(cls, path):
        from pathlib import Path
        return cls.from_json_bytes(Path(path).read_bytes())

    @classmethod
    def from_model_dir(cls, model_dir):
        """Embedded checkpoint tokenizer, else sibling ``tokenizer.json``."""
        from pathlib import Path
        try:
            return cls.from_checkpoint(model_dir)
        except (FileNotFoundError, RuntimeError):
            fallback = Path(model_dir) / "tokenizer.json"
            if fallback.exists():
                return cls.from_json_file(fallback)
            raise

    def encode(self, text):
        return self.tokenizer.encode(text).ids

    def decode(self, ids, skip_special_tokens=True):
        return self.tokenizer.decode(list(ids),
                                     skip_special_tokens=skip_special_tokens)


def tokenize_with_weights(style_text, lyrics_text, abc_text="", cot="full",
                          seed=0, max_tokens=9000,
                          temperature=DEFAULT_TEMPERATURE,
                          top_p=DEFAULT_TOP_P, top_k=DEFAULT_TOP_K,
                          repetition_penalty=DEFAULT_REPETITION_PENALTY,
                          penalty_window=DEFAULT_PENALTY_WINDOW,
                          cfg_scale=None):
    """Mirror Comfy ``YuE2Tokenizer.tokenize_with_weights``.

    Returns the sampling bundle consumed by the generate loop.
    """
    if cot not in INSTRUCTIONS:
        raise ValueError(f"cot must be off/melody/full, got {cot!r}")
    tokenizer = _active_tokenizer()
    prompt = (f"{INSTRUCTIONS[cot]}\n[Tags]\n{style_text}\n[Lyrics]\n"
              f"{lyrics_text}\n")
    return {
        "prefix": [EOD] + tokenizer.encode(prompt) + [ABC_START],
        "negative": [EOD] + tokenizer.encode(INSTRUCTIONS[cot]),
        "abc_ids": tokenizer.encode(abc_text or ""),
        "cot": cot,
        "seed": seed,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "top_p": top_p,
        "top_k": top_k,
        "repetition_penalty": repetition_penalty,
        "penalty_window": penalty_window,
        "cfg_scale": cfg_scale if cfg_scale is not None
        else (1.01 if cot == "off" else 1.0),
    }


_TOKENIZER = None


def bind_tokenizer(tokenizer):
    """Bind the process-wide tokenizer used by ``tokenize_with_weights``."""
    global _TOKENIZER
    _TOKENIZER = tokenizer


def _active_tokenizer():
    if _TOKENIZER is None:
        raise RuntimeError("no tokenizer bound (call bind_tokenizer first)")
    return _TOKENIZER


def distribution(logits, history, step, phase, temperature=1.0, top_p=1.0,
                 top_k=0, repetition_penalty=1.0, penalty_window=50,
                 min_tokens=0, legacy_off=False):
    """Mirror Comfy ``distribution``; return sampling scores.

    ``temperature == 0`` returns the masked scores unchanged (the caller
    argmaxes). ``top_k <= 0`` disables top-k filtering.
    """
    scores = logits.clone() if legacy_off else logits.float().clone()
    end = ABC_END if phase == "abc" else MUSIC_END
    allowed = torch.full_like(scores, -torch.inf)
    if phase == "abc":
        allowed[..., :EOD] = 0
    else:
        allowed[..., CODEC_OFFSET:CODEC_OFFSET + CODEC_SIZE] = 0
    allowed[..., end] = 0
    scores += allowed
    if step < min_tokens:
        scores[..., end] = -torch.inf
    if repetition_penalty != 1.0 and history:
        recent = torch.tensor([history[-penalty_window:]], dtype=torch.long,
                              device=scores.device)
        counts = torch.zeros_like(scores)
        counts.scatter_add_(-1, recent,
                            torch.ones_like(recent, dtype=scores.dtype))
        penalty = repetition_penalty ** counts
        scores = torch.where(scores < 0, scores * penalty, scores / penalty)
    if temperature == 0:
        return scores
    scores /= temperature
    if top_k and top_k > 0:
        threshold = scores.topk(min(top_k, scores.shape[-1])
                                ).values[..., -1, None]
        scores.masked_fill_(scores < threshold, -torch.inf)
    if top_p < 1:
        values, indices = scores.sort(descending=True)
        probabilities = values.softmax(-1)
        removed = probabilities.cumsum(-1) - probabilities > top_p
        removed[..., :3 if legacy_off else 1] = False
        values.masked_fill_(removed, -torch.inf)
        scores = values.scatter(-1, indices, values)
    return scores


def chunk_ranges(frames, prefix_tokens, context=CONTEXT):
    """Mirror Comfy ``chunk_ranges``.

    ``size = (context - prefix - 3) // 2``; raises when there is nothing
    to split or no room for even one frame.
    """
    size = (int(context) - int(prefix_tokens) - 3) // 2
    if int(frames) < 1 or size < 1:
        raise ValueError("YuE2 needs music tokens and enough context "
                         "for at least one acoustic frame.")
    frames = int(frames)
    return [(start, min(start + size, frames))
            for start in range(0, frames, size)]


def fit_budget(requested, prefix, negative, context=CONTEXT):
    """Clamp the semantic budget to the context room (Comfy parity).

    Returns ``(budget, reduced)`` where ``reduced`` flags a clamp below
    the requested cap. Raises Comfy's no-room error when the prompt
    itself leaves no space for music.
    """
    requested = int(requested)
    budget = min(requested, int(context) - max(len(prefix), len(negative)))
    if budget < 1 or len(prefix) + 5 > int(context):
        raise ValueError("YuE2 prompt leaves no room for music; shorten "
                         "the style, lyrics, or ABC.")
    return budget, budget < requested


def _prefill_pair(model, prefix, negative, capacity, device, dtype):
    """Left-pad the CFG pair, prefill, return (logits, caches, positions)."""
    prefixes = [prefix] if negative is None else [prefix, negative]
    length = max(map(len, prefixes))
    ids = torch.tensor([[0] * (length - len(p)) + p for p in prefixes],
                       device=device, dtype=torch.long)
    mask = positions = None
    if any(len(p) != length for p in prefixes):
        mask = torch.ones((len(prefixes), capacity), device=device,
                          dtype=torch.long)
        for index, p in enumerate(prefixes):
            mask[index, :length - len(p)] = 0
        positions = mask[:, :length].cumsum(-1).sub_(1).clamp_min_(0)
    caches = model.init_kv_cache(len(prefixes), capacity, device, dtype)
    hidden, _, caches = model(ids, position_ids=positions,
                              attention_mask=mask[:, :length]
                              if mask is not None else None,
                              past_key_values=caches)
    logits = model.logits(hidden.to(dtype) if dtype is not None else hidden)
    pos = [len(p) for p in prefixes]
    return logits, caches, pos


def _generate(model, prefix, seed, max_tokens, phase, device, dtype,
              negative=None, cfg_scale=1.0, legacy_off=False,
              temperature=1.0, top_p=1.0, top_k=0, repetition_penalty=1.0,
              penalty_window=50, min_tokens=0):
    """Mirror Comfy ``YuE2TEModel._generate`` (no graphs/progress).

    Returns ``(history, truncated)``; the end token is NOT appended to
    history. Dual-batch CFG when ``cfg_scale != 1.0``.
    """
    end = ABC_END if phase == "abc" else MUSIC_END
    generator = torch.Generator(device=device).manual_seed(int(seed))
    dual = cfg_scale != 1.0
    prefixes = [prefix, negative] if dual else [prefix]
    prefix_len = max(map(len, prefixes))
    logits, caches, positions = _prefill_pair(
        model, prefix, negative if dual else None,
        prefix_len + max_tokens, device, dtype)
    positions = [[p] for p in positions]
    decode = torch.empty((len(prefixes), 1), device=device, dtype=torch.long)
    history = []
    log_every = 500 if str(device) != "cpu" else 0
    for _step in range(max_tokens):
        guided = logits if not dual else \
            logits[1:] + cfg_scale * (logits[:1] - logits[1:])
        scores = distribution(guided, history, len(history), phase,
                              temperature=temperature, top_p=top_p,
                              top_k=top_k,
                              repetition_penalty=repetition_penalty,
                              penalty_window=penalty_window,
                              min_tokens=min_tokens,
                              legacy_off=legacy_off)
        if temperature == 0:
            token = int(scores.argmax(-1, keepdim=True).item())
        else:
            # float32 probabilities: legacy_off scores stay in model
            # dtype (bf16), which multinomial may refuse.
            probs = scores.softmax(-1).float()
            token = int(torch.multinomial(probs, 1,
                                          generator=generator).item())
        if token == end:
            return history, False
        history.append(token)
        if log_every and len(history) % log_every == 0:
            live = torch.cuda.memory_allocated(device) / 1e9
            print(f"yue2-torch generate [{phase}]: step {len(history)}/"
                  f"{max_tokens}, live {live:.2f} GiB", flush=True)
        decode.copy_(torch.tensor([[token]] * len(prefixes), device=device))
        pos = torch.tensor(positions, device=device, dtype=torch.long)
        hidden, _, caches = model(decode, position_ids=pos,
                                  past_key_values=caches)
        logits = model.logits(hidden)
        positions = [[p[0] + 1] for p in positions]
    return history, True
