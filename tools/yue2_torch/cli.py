"""Pure argv/spec builder for the YuE2 torch worker (no torch/heavy imports).

The node and the tests import this module without a GPU. All validation
mirrors the sidecar rules so both engines reject the same bad inputs.
"""

COT_MODES = ("off", "melody", "full")
YUE2_FPS = 25

SPEC_REQUIRED = ("lyrics", "style", "cot", "seed", "semantic_max_tokens",
                 "model_dir")
SPEC_OPTIONAL = ("guidance_scale", "num_inference_steps", "abc_file",
                 "temperature", "top_p", "top_k", "repetition_penalty",
                 "sampler")
KNOWN_SPEC_KEYS = frozenset(SPEC_REQUIRED + SPEC_OPTIONAL + ("out_format",))

# Mirrors dit.SAMPLERS keys. Kept literal so this module stays importable
# without torch (the node imports it to build argv with no GPU present);
# test_sampler_names_stay_in_sync_with_dit guards the duplication.
_SAMPLERS = ("dpm_2", "euler")


def tokens_for_seconds(seconds, fps=YUE2_FPS):
    """Song-length cap: seconds -> semantic frames (Comfy parity)."""
    return max(1, round(float(seconds) * fps))


def validate_spec(spec):
    """Validate a torch worker spec dict. Raises ValueError."""
    unknown = sorted(set(spec) - KNOWN_SPEC_KEYS)
    if unknown:
        raise ValueError(f"unknown spec keys: {unknown}")
    for key in SPEC_REQUIRED:
        if key not in spec:
            raise ValueError(f"spec missing required key: {key!r}")
    lyrics, style = spec["lyrics"], spec["style"]
    if not lyrics or not str(lyrics).strip():
        raise ValueError("lyrics must be nonempty text")
    if not style or not str(style).strip():
        raise ValueError("style must be nonempty text")
    if spec["cot"] not in COT_MODES:
        raise ValueError(f"cot must be off/melody/full, got {spec['cot']!r}")
    seed = spec["seed"]
    if not isinstance(seed, int) or isinstance(seed, bool) \
            or not 0 <= seed < 2 ** 63:
        raise ValueError(f"seed must be an integer in [0, 2**63), got {seed!r}")
    tokens = spec["semantic_max_tokens"]
    if isinstance(tokens, bool) or not isinstance(tokens, int) or tokens < 1:
        raise ValueError(
            "semantic_max_tokens must be a positive integer, "
            f"got {tokens!r}")
    if spec.get("abc_file") and spec["cot"] == "off":
        raise ValueError("an ABC score requires cot=melody or cot=full")
    guidance = spec.get("guidance_scale")
    if guidance is not None:
        if (isinstance(guidance, bool)
                or not isinstance(guidance, (int, float))
                or not 0 <= guidance <= 20
                or guidance != guidance):  # NaN guard, no math import
            raise ValueError(
                "guidance_scale must be a finite number in [0, 20], "
                f"got {guidance!r}")
    steps = spec.get("num_inference_steps")
    if steps is not None:
        if (not isinstance(steps, int) or isinstance(steps, bool)
                or steps < 1):
            raise ValueError(
                "num_inference_steps must be a positive integer, "
                f"got {steps!r}")
    sampler = spec.get("sampler")
    if sampler is not None and sampler not in _SAMPLERS:
        raise ValueError(
            f"sampler must be one of {sorted(_SAMPLERS)}, got {sampler!r}")
    temperature = spec.get("temperature")
    if temperature is not None:
        if (isinstance(temperature, bool)
                or not isinstance(temperature, (int, float))
                or not 0 <= temperature <= 2
                or temperature != temperature):
            raise ValueError(
                f"temperature must be a finite number in [0, 2], "
                f"got {temperature!r}")
    top_p = spec.get("top_p")
    if top_p is not None:
        if (isinstance(top_p, bool)
                or not isinstance(top_p, (int, float))
                or not 0 < top_p <= 1):
            raise ValueError(
                f"top_p must be a number in (0, 1], got {top_p!r}")
    top_k = spec.get("top_k")
    if top_k is not None:
        if (not isinstance(top_k, int) or isinstance(top_k, bool)
                or top_k < 0):
            raise ValueError(
                "top_k must be an integer >= 0 (0 disables filtering), "
                f"got {top_k!r}")
    rep = spec.get("repetition_penalty")
    if rep is not None:
        if (isinstance(rep, bool)
                or not isinstance(rep, (int, float))
                or not rep >= 1.0 or rep != rep):
            raise ValueError(
                f"repetition_penalty must be a finite number >= 1.0, "
                f"got {rep!r}")
    return spec


def build_argv(python, run_py, *, spec_path, out_wav, out_dir):
    """Build the worker subprocess argv (pure, no process).

    File-based only: lyrics/style travel inside ``spec_path``, never on the
    command line (Windows ~8k cmdline limit).
    """
    return [str(python), str(run_py), "--spec", str(spec_path),
            "--out", str(out_wav), "--out-dir", str(out_dir), "--metrics"]
