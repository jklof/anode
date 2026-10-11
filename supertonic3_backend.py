"""Shared Supertonic 3 helpers (no Qt, no torch, no audio processing).

Supertonic 3 is a local ONNX text-to-speech system (~99 M parameters,
31 languages + ``na`` fallback, 44.1 kHz output). ANode drives the
upstream Python SDK in a supervised child subprocess launched by an NRT
worker (AGENTS.md sections 4 and 6) — never in ``process()``.

This module owns the pins, the model fetch manifest, pure request/argv
builders, bounded profile validation/preparation, and waveform checks.
Heavy SDK imports stay in ``tools/supertonic3/worker.py`` (child-only)
so a missing SDK never breaks palette registration.

Pins (verified 2026-10-11):

* SDK source ``supertone-oss-archive/supertonic-py`` @
  ``df0f9686dac7fbbde391b759e2ee5286a3737622``.
* Model ``supertone-oss-archive/supertonic-3`` @
  ``aafc6e32416a594460b32413efc49d7fe4ce6d46``.

Source code is MIT; model weights are under the OpenRAIL-M license —
see the distributed LICENSE for its restrictions and obligations.
Fetch-at-runtime only — never bundled (see .gitignore).
"""

import hashlib
import json
import math
import sys
import threading
from pathlib import Path

from download_util import DownloadCancelled, DownloadSpec

REPO_ROOT = Path(__file__).resolve().parent

SUPERTONIC3_SDK_REPO = "supertone-oss-archive/supertonic-py"
SUPERTONIC3_SDK_PIN = "df0f9686dac7fbbde391b759e2ee5286a3737622"
SUPERTONIC3_SDK_URL = (
    f"https://github.com/{SUPERTONIC3_SDK_REPO}.git@{SUPERTONIC3_SDK_PIN}"
)
SUPERTONIC3_SDK_PIP_SPEC = (
    f"supertonic @ git+https://github.com/{SUPERTONIC3_SDK_REPO}.git"
    f"@{SUPERTONIC3_SDK_PIN}"
)

SUPERTONIC3_MODEL_REPO = "supertone-oss-archive/supertonic-3"
SUPERTONIC3_MODEL_PIN = "aafc6e32416a594460b32413efc49d7fe4ce6d46"
SUPERTONIC3_HF_BASE = (
    f"https://huggingface.co/{SUPERTONIC3_MODEL_REPO}"
    f"/resolve/{SUPERTONIC3_MODEL_PIN}"
)

# (repo-relative path, size bytes, sha256).
# ONNX sizes/hashes are the upstream LFS oids (== file SHA-256) from the
# pinned archive tree API. Small JSON/license hashes were independently
# established by downloading the pinned revision and hashing the bytes
# (2026-10-11); sizes match the archive listing. Do not invent hashes here.
SUPERTONIC3_FILES = [
    ("onnx/duration_predictor.onnx", 3700147,
     "c3eb91414d5ff8a7a239b7fe9e34e7e2bf8a8140d8375ffb14718b1c639325db"),
    ("onnx/text_encoder.onnx", 36416150,
     "c7befd5ea8c3119769e8a6c1486c4edc6a3bc8365c67621c881bbb774b9902ff"),
    ("onnx/vector_estimator.onnx", 256534781,
     "883ac868ea0275ef0e991524dc64f16b3c0376efd7c320af6b53f5b780d7c61c"),
    ("onnx/vocoder.onnx", 101424195,
     "085de76dd8e8d5836d6ca66826601f615939218f90e519f70ee8a36ed2a4c4ba"),
    ("onnx/tts.json", 8253,
     "42078d3aef1cd43ab43021f3c54f47d2d75ceb4e75f627f118890128b06a0d09"),
    ("onnx/unicode_indexer.json", 277676,
     "9bf7346e43883a81f8645c81224f786d43c5b57f3641f6e7671a7d6c493cb24f"),
    ("voice_styles/M1.json", 291748,
     "e35604687f5d23694b8e91593a93eec0e4eca6c0b02bb8ed69139ab2ea6b0a5b"),
    ("voice_styles/M2.json", 292055,
     "b76cbf62bac707c710cf0ae5aba5e31eea1a6339a9734bfae33ab98499534a50"),
    ("voice_styles/M3.json", 290198,
     "ea1ac35ccb91b0d7ecad533a2fbd0eec10c91513d8951e3b25fbba99954e159b"),
    ("voice_styles/M4.json", 291522,
     "ca8eefad4fcd989c9379032ff3e50738adc547eeb5e221b82593a6d7b3bac303"),
    ("voice_styles/M5.json", 291469,
     "dd22b92740314321f8ae11c5e87f8dd60d060f15dd3a632b5adf77f471f77af2"),
    ("voice_styles/F1.json", 292046,
     "bbdec6ee00231c2c742ad05483df5334cab3b52fda3ba38e6a07059c4563dbc2"),
    ("voice_styles/F2.json", 292423,
     "7c722c6a72707b1a77f035d67f0d1351ba187738e06f7683e8c72b1df3477fc6"),
    ("voice_styles/F3.json", 290794,
     "12f6ef2573baa2defa1128069cb59f203e3ab67c92af77b42df8a0e3a2f7c6ab"),
    ("voice_styles/F4.json", 291808,
     "c2fa764c1225a76dfc3e2c73e8aa4f70d9ee48793860eb34c295fff01c2e032b"),
    ("voice_styles/F5.json", 291479,
     "45966e73316415626cf41a7d1c6f3b4c70dbc1ba2bee5c1978ef0ce33244fc8d"),
    ("LICENSE", 15007,
     "0d944a9110fed9a9602d60e0423a272903e7bd21ab060490774efc77c2275e9f"),
]

# Fixed preset order for saved-patch compatibility (M1–M5, F1–F5).
VOICES = ("M1", "M2", "M3", "M4", "M5", "F1", "F2", "F3", "F4", "F5")

# Fixed language order for saved-patch compatibility: the SDK's
# SUPPORTED_LANGUAGES order plus the "na" fallback last.
LANGUAGES = (
    "en", "ko", "ja", "ar", "bg", "cs", "da", "de", "el", "es",
    "et", "fi", "fr", "hi", "hr", "hu", "id", "it", "lt", "lv",
    "nl", "pl", "pt", "ro", "ru", "sk", "sl", "sv", "tr", "uk",
    "vi", "na",
)
LANGUAGE_NAMES = {
    "en": "English", "ko": "Korean", "ja": "Japanese", "ar": "Arabic",
    "bg": "Bulgarian", "cs": "Czech", "da": "Danish", "de": "German",
    "el": "Greek", "es": "Spanish", "et": "Estonian", "fi": "Finnish",
    "fr": "French", "hi": "Hindi", "hr": "Croatian", "hu": "Hungarian",
    "id": "Indonesian", "it": "Italian", "lt": "Lithuanian",
    "lv": "Latvian", "nl": "Dutch", "pl": "Polish", "pt": "Portuguese",
    "ro": "Romanian", "ru": "Russian", "sk": "Slovak", "sl": "Slovenian",
    "sv": "Swedish", "tr": "Turkish", "uk": "Ukrainian",
    "vi": "Vietnamese", "na": "Auto (language-agnostic)",
}


def language_menu_items():
    """Display strings for the fixed language menu (code order pinned)."""
    return [f"{LANGUAGE_NAMES.get(c, c)} ({c})" for c in LANGUAGES]


EXPRESSION_TAGS = (
    "<laugh>", "<breath>", "<sigh>", "<surprise>", "<scream>",
    "<throatclear>", "<sad>", "<angry>", "<cough>", "<yawn>",
)

DEFAULT_LANGUAGE = "en"
DEFAULT_VOICE = "M1"
DEFAULT_SPEED = 1.05
DEFAULT_TOTAL_STEPS = 8
DEFAULT_SILENCE_DURATION = 0.3
DEFAULT_THREADS = 2
DEFAULT_MODEL_DIR = "models/Supertonic3"

SPEED_MIN, SPEED_MAX = 0.7, 2.0
STEPS_MIN, STEPS_MAX = 1, 100
SILENCE_MIN, SILENCE_MAX = 0.0, 5.0
THREADS_MIN, THREADS_MAX = 1, 8

MAX_TEXT_LENGTH = 100000  # SDK MAX_TEXT_LENGTH (characters)
MAX_TEXT_BYTES = 512 * 1024  # bounded file/URI text reads
MAX_PROFILE_BYTES = 8 * 1024 * 1024  # bounded voice-style JSON reads

# Pinned v3 voice-style geometry (measured from the archived presets:
# style_ttl [1, 50, 256] = 12800 floats, style_dp [1, 8, 16] = 128).
STYLE_TTL_DIMS = (1, 50, 256)
STYLE_DP_DIMS = (1, 8, 16)
MAX_DIMS_PRODUCT = 1024 * 1024  # pre-allocation sanity cap
MAX_DIM_EXTENT = 4096

WORKER_TIMEOUT_S = 600
FETCH_LABEL_PREFIX = "Supertonic3"
FLOAT32_MAX = 3.4028235e38


def supertonic3_fetch_specs(model_dir):
    """DownloadSpecs for the Supertonic 3 model assets (model-only)."""
    model_dir = Path(model_dir)
    return [DownloadSpec(f"{SUPERTONIC3_HF_BASE}/{rel}", model_dir / rel,
                         size, sha, label=f"{FETCH_LABEL_PREFIX}: {rel}")
            for rel, size, sha in SUPERTONIC3_FILES]


def required_asset_paths(model_dir):
    """Repo-relative asset paths a worker must verify before synthesis."""
    return [rel for rel, _, _ in SUPERTONIC3_FILES]


def check_assets(model_dir):
    """Cheap asset check. Returns (ok, missing_rel_paths). No process."""
    from download_util import _verify
    model_dir = Path(model_dir)
    missing = []
    for rel, size, sha in SUPERTONIC3_FILES:
        if not _verify(model_dir / rel, size, sha):
            missing.append(rel)
    return (len(missing) == 0), missing


# ----------------------------------------------------------------------
# fetch destination coordination (NRT-side only)
# ----------------------------------------------------------------------
# Both Supertonic nodes share one model directory, and download_util uses
# one fixed ``<dest>.part`` path per asset. Two NRT fetch workers writing
# the same partial file can fail verification, erase resumable progress,
# or hit missing-part errors. Coordinate on the resolved model directory
# with per-destination locks held only by NRT workers: engine/control
# callbacks never wait, and waits stay responsive to cancellation.
_FETCH_LOCKS_GUARD = threading.Lock()
_FETCH_LOCKS = {}


def fetch_lock_for(model_dir):
    """Return the shared lock for a resolved model directory. Pure."""
    key = str(Path(model_dir).absolute())
    with _FETCH_LOCKS_GUARD:
        lock = _FETCH_LOCKS.get(key)
        if lock is None:
            lock = threading.Lock()
            _FETCH_LOCKS[key] = lock
        return lock


def acquire_fetch_lock(lock, cancel_event, timeout_s=0.05):
    """Acquire ``lock`` while staying responsive to cancellation. NRT only.

    Polls with a short timeout so a cancelled wait raises DownloadCancelled
    instead of blocking the worker indefinitely.
    """
    while True:
        if cancel_event is not None and cancel_event.is_set():
            raise DownloadCancelled("download cancelled while waiting")
        if lock.acquire(timeout=timeout_s):
            return True


def result_request_id(result):
    """Request identity for an NRT completion payload. Pure.

    Success payloads carry ``request_id`` in their dict; worker failures
    are tagged with ``_supertonic3_request_id`` by the node's ``_run_nrt``
    wrapper so a Cancel can still reject them. Returns None when unknown.
    """
    if isinstance(result, dict):
        return result.get("request_id")
    return getattr(result, "_supertonic3_request_id", None)


def tag_exception_request_id(exc, request_id):
    """Attach request identity to a worker exception. Pure, never raises."""
    try:
        setattr(exc, "_supertonic3_request_id", request_id)
    except Exception:
        pass
    return exc


# ----------------------------------------------------------------------
# profile validation / preparation (pure + bounded file I/O for NRT/child)
# ----------------------------------------------------------------------

def _is_number(x):
    return isinstance(x, (int, float)) and not isinstance(x, bool)


def _validate_dims(dims, expected, field):
    if not isinstance(dims, (list, tuple)):
        raise ValueError(f"{field}.dims must be a list of integers")
    if any(isinstance(d, bool) or not isinstance(d, int) for d in dims):
        raise ValueError(f"{field}.dims must be integers (got {dims!r})")
    if len(dims) < 1 or len(dims) > 4:
        raise ValueError(f"{field}.dims has unreasonable rank {len(dims)}")
    if any(d < 1 or d > MAX_DIM_EXTENT for d in dims):
        raise ValueError(f"{field}.dims has unreasonable extent {list(dims)!r}")
    product = 1
    for d in dims:
        product *= d
        if product > MAX_DIMS_PRODUCT:
            raise ValueError(
                f"{field}.dims product exceeds the sanity cap "
                f"({MAX_DIMS_PRODUCT}) — refusing to allocate")
    if tuple(dims) != tuple(expected):
        raise ValueError(
            f"{field}.dims {list(dims)!r} does not match the pinned v3 "
            f"expectation {list(expected)!r}")
    return product


def _count_and_validate_data(data, expected_count, field):
    """Iterative leaf walk: no recursion, no large temporaries.

    Rejects booleans, non-numeric leaves, non-finite values and values
    outside the float32-representable range. Returns the leaf count.
    """
    if not isinstance(data, list):
        raise ValueError(f"{field}.data must be a JSON array")
    count = 0
    stack = [iter(data)]
    # Each stack entry is an iterator over one nesting level.
    while stack:
        try:
            item = next(stack[-1])
        except StopIteration:
            stack.pop()
            continue
        if isinstance(item, list):
            if len(stack) > 8:
                raise ValueError(f"{field}.data is nested too deeply")
            stack.append(iter(item))
            continue
        if not _is_number(item):
            raise ValueError(
                f"{field}.data must contain only numbers "
                f"(got {type(item).__name__})")
        try:
            f = float(item)
        except (TypeError, ValueError):
            raise ValueError(f"{field}.data contains an invalid number")
        if not math.isfinite(f):
            raise ValueError(f"{field}.data contains a non-finite value")
        if abs(f) > FLOAT32_MAX:
            raise ValueError(
                f"{field}.data contains a value outside float32 range")
        count += 1
        if count > expected_count:
            raise ValueError(
                f"{field}.data has more elements than dims allow "
                f"({expected_count})")
    if count != expected_count:
        raise ValueError(
            f"{field}.data has {count} elements but dims require "
            f"{expected_count}")
    return count


_CONTRADICTORY_VERSIONS = frozenset({
    "1", "2", "v1", "v2", "supertonic", "supertonic-1", "supertonic-2",
    "supertonic1", "supertonic2",
})


def _check_version_metadata(obj):
    """Reject explicitly contradictory version metadata. Pure."""
    candidates = []
    if isinstance(obj, dict):
        for key in ("version", "model", "model_name", "supertonic",
                    "supertonic_version"):
            if key in obj:
                candidates.append(obj[key])
        meta = obj.get("metadata")
        if isinstance(meta, dict):
            for key in ("version", "model", "model_name", "supertonic",
                        "supertonic_version"):
                if key in meta:
                    candidates.append(meta[key])
    for value in candidates:
        if isinstance(value, bool):
            continue
        if isinstance(value, int):
            if value in (1, 2):
                raise ValueError(
                    f"voice style targets Supertonic v{value}, not v3")
            continue
        if isinstance(value, str):
            norm = value.strip().lower().replace(" ", "").replace("-", "")
            # "supertonic3"/"v3"/"3" are fine; anything matching the v1/v2
            # set (after normalization) is contradictory.
            norm_plain = norm.replace("supertonic", "").replace("v", "")
            if norm in {"supertonic1", "supertonic2", "supertonic",
                       "1", "2", "v1", "v2"} or norm_plain in {"1", "2"}:
                raise ValueError(
                    f"voice style targets {value!r}, not Supertonic 3")
    return None


def validate_profile_dict(obj):
    """Validate a parsed voice-style JSON object. Pure (no I/O).

    Accepts either a shipped preset or an imported style JSON: a dict
    with ``style_ttl`` and ``style_dp``, each a dict with ``dims`` and
    ``data``. Values are preserved exactly (no normalization).

    JSON structure cannot reliably distinguish v2 from v3, so unknown
    provenance is recorded honestly instead of claimed as verified v3.
    Returns ``{"ttl_count", "dp_count", "provenance"}``.
    Raises ValueError with a human-readable message on rejection.
    """
    if not isinstance(obj, dict):
        raise ValueError("voice style must be a JSON object")
    for field, expected in (("style_ttl", STYLE_TTL_DIMS),
                            ("style_dp", STYLE_DP_DIMS)):
        section = obj.get(field)
        if not isinstance(section, dict):
            raise ValueError(
                f"voice style is missing {field!r} "
                "(expected dims/data mapping)")
        if "dims" not in section or "data" not in section:
            raise ValueError(
                f"voice style {field!r} needs 'dims' and 'data' fields")
        product = _validate_dims(section["dims"], expected, field)
        _count_and_validate_data(section["data"], product, field)
    _check_version_metadata(obj)
    meta = obj.get("metadata")
    provenance = {
        "metadata": dict(meta) if isinstance(meta, dict) else {},
        "verified_v3": False,
        "note": ("JSON structure cannot prove v3 suitability: "
                 "compatibility of externally supplied files is the "
                 "user's responsibility."),
    }
    return {"ttl_count": STYLE_TTL_DIMS[0] * STYLE_TTL_DIMS[1] * STYLE_TTL_DIMS[2],
            "dp_count": STYLE_DP_DIMS[0] * STYLE_DP_DIMS[1] * STYLE_DP_DIMS[2],
            "provenance": provenance}


def load_and_validate_profile_file(path, max_bytes=MAX_PROFILE_BYTES):
    """Bounded read + parse + validate of a voice-style JSON file.

    NRT/worker-side only (filesystem I/O). Returns
    ``(obj, source_sha256, validation)``. Raises ValueError/OSError
    with human-readable messages.
    """
    path = Path(path)
    try:
        size = path.stat().st_size
    except OSError:
        raise ValueError(f"voice style file not found: {path}")
    if size > max_bytes:
        raise ValueError(
            f"voice style file is too large ({size} bytes, "
            f"cap {max_bytes})")
    try:
        with open(path, "rb") as f:
            raw = f.read(max_bytes + 1)
    except OSError as e:
        raise ValueError(f"cannot read voice style file {path}: {e}")
    if len(raw) > max_bytes:
        raise ValueError(
            f"voice style file is too large (over {max_bytes} bytes)")
    source_hash = hashlib.sha256(raw).hexdigest()
    try:
        obj = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, ValueError) as e:
        raise ValueError(f"voice style file is not valid UTF-8 JSON: {e}")
    validation = validate_profile_dict(obj)
    return obj, source_hash, validation


def _sanitize_profile_name(name, fallback="voice"):
    safe = "".join(c if (c.isalnum() or c in ("-", "_")) else "_"
                   for c in str(name or "").strip())
    safe = safe.strip("_") or fallback
    return safe[:48]


def prepare_profile(*, profile_obj, source, source_hash, preset_name,
                    display_name, profiles_dir):
    """Write a unique kept voice_style.json + provenance. NRT only.

    ``source`` is ``"preset:<NAME>"`` or ``"file:<path>"``. Never
    overwrites: each call creates a new timestamped directory so
    downstream file consumers can distinguish replacements. Returns
    ``{"profile_path", "provenance_path", "source_hash",
    "output_hash", "display_name"}`` with absolute path strings.
    """
    import datetime
    import uuid
    profiles_dir = Path(profiles_dir)
    stamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    safe = _sanitize_profile_name(display_name or preset_name or "voice")
    keep = profiles_dir / f"{stamp}_{safe}_{uuid.uuid4().hex[:8]}"
    keep.mkdir(parents=True, exist_ok=True)
    profile_path = keep / "voice_style.json"
    # Preserve values exactly: dump the validated object as-is.
    profile_path.write_text(
        json.dumps(profile_obj, indent=2) + "\n", encoding="utf-8")
    output_hash = hashlib.sha256(profile_path.read_bytes()).hexdigest()
    provenance = {
        "display_name": display_name or preset_name or safe,
        "source": source,
        "preset": preset_name,
        "backend": "supertonic3",
        "sdk": {"repo": SUPERTONIC3_SDK_REPO, "pin": SUPERTONIC3_SDK_PIN},
        "model": {"repo": SUPERTONIC3_MODEL_REPO, "pin": SUPERTONIC3_MODEL_PIN},
        "source_hash": source_hash,
        "output_hash": output_hash,
        "style_ttl_dims": list(STYLE_TTL_DIMS),
        "style_dp_dims": list(STYLE_DP_DIMS),
    }
    provenance_path = keep / "provenance.json"
    provenance_path.write_text(
        json.dumps(provenance, indent=2) + "\n", encoding="utf-8")
    return {"profile_path": str(profile_path),
            "provenance_path": str(provenance_path),
            "source_hash": source_hash,
            "output_hash": output_hash,
            "display_name": provenance["display_name"]}


# ----------------------------------------------------------------------
# TTS request / settings validation (cheap UI/control-side checks only)
# ----------------------------------------------------------------------

def validate_tts_settings(*, language, speed, total_steps,
                          silence_duration, threads):
    """Cheap value checks for the engine/control thread. Pure."""
    if language not in LANGUAGES:
        raise ValueError(
            f"language must be one of {list(LANGUAGES)}, got {language!r}")
    if isinstance(speed, bool) or not isinstance(speed, (int, float)):
        raise ValueError(f"speed must be a number, got {speed!r}")
    if not math.isfinite(float(speed)) or not SPEED_MIN <= float(speed) <= SPEED_MAX:
        raise ValueError(
            f"speed must be in [{SPEED_MIN}, {SPEED_MAX}], got {speed!r}")
    if isinstance(total_steps, bool) or not isinstance(total_steps, int):
        raise ValueError(
            f"total_steps must be an integer, got {total_steps!r}")
    if not STEPS_MIN <= total_steps <= STEPS_MAX:
        raise ValueError(
            f"total_steps must be in [{STEPS_MIN}, {STEPS_MAX}], "
            f"got {total_steps!r}")
    if isinstance(silence_duration, bool) or not isinstance(
            silence_duration, (int, float)):
        raise ValueError(
            f"silence_duration must be a number, got {silence_duration!r}")
    if (not math.isfinite(float(silence_duration))
            or not SILENCE_MIN <= float(silence_duration) <= SILENCE_MAX):
        raise ValueError(
            f"silence_duration must be in [{SILENCE_MIN}, {SILENCE_MAX}] s, "
            f"got {silence_duration!r}")
    if isinstance(threads, bool) or not isinstance(threads, int):
        raise ValueError(f"threads must be an integer, got {threads!r}")
    if not THREADS_MIN <= threads <= THREADS_MAX:
        raise ValueError(
            f"threads must be in [{THREADS_MIN}, {THREADS_MAX}], "
            f"got {threads!r}")


def pick_text_source(*, inline_text, uri_connected, uri_snapshot):
    """Text precedence: connected text_uri wins over inline text. Pure.

    A connected-but-empty URI rejects instead of silently falling back.
    Returns ("uri"|"inline", value).
    """
    if uri_connected:
        if not uri_snapshot:
            raise ValueError(
                "Wired text is empty — publish a text file first, "
                "then generate.")
        return "uri", uri_snapshot
    if not inline_text or not str(inline_text).strip():
        raise ValueError("Text is empty — type text or wire a text file.")
    return "inline", str(inline_text)


def pick_profile_source(*, preset_idx, file_param,
                        uri_connected=False, uri_snapshot=""):
    """Profile precedence: connected URI, file param, then preset. Pure.

    Connected-empty inputs reject rather than silently use the fallback.
    Returns ("uri"|"file"|"preset", value).
    """
    if uri_connected:
        if not uri_snapshot:
            raise ValueError(
                "Wired voice style is empty — apply a voice first, "
                "then generate.")
        return "uri", uri_snapshot
    if file_param:
        return "file", file_param
    try:
        preset = VOICES[int(preset_idx)]
    except (TypeError, ValueError, IndexError):
        raise ValueError(f"unknown voice preset index {preset_idx!r}")
    return "preset", preset


def read_bounded_text_file(path, max_bytes=MAX_TEXT_BYTES):
    """Bounded UTF-8 text read for NRT/child contexts. Returns str."""
    path = Path(path)
    try:
        size = path.stat().st_size
    except OSError:
        raise ValueError(f"text file not found: {path}")
    if size > max_bytes:
        raise ValueError(
            f"text file is too large ({size} bytes, cap {max_bytes})")
    try:
        with open(path, "rb") as f:
            raw = f.read(max_bytes + 1)
    except OSError as e:
        raise ValueError(f"cannot read text file {path}: {e}")
    if len(raw) > max_bytes:
        raise ValueError(f"text file is too large (over {max_bytes} bytes)")
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as e:
        raise ValueError(f"text file is not valid UTF-8: {e}")
    if len(text) > MAX_TEXT_LENGTH:
        raise ValueError(
            f"text has {len(text)} characters (cap {MAX_TEXT_LENGTH})")
    if not text.strip():
        raise ValueError(f"text file is empty: {path}")
    return text


def build_tts_request(*, text_kind, text_value, style_kind, style_value,
                      model_dir, language, speed, total_steps,
                      silence_duration, threads, request_id):
    """Immutable worker spec snapshot. Pure (no I/O).

    ``text_kind`` is ``"inline"``/``"uri"``; ``style_kind`` is
    ``"preset"``/``"file"``/``"uri"``. Only cheap value checks run here;
    filesystem reads, hashing, asset checks and imports belong to NRT/child.
    """
    validate_tts_settings(language=language, speed=speed,
                          total_steps=total_steps,
                          silence_duration=silence_duration, threads=threads)
    if text_kind not in ("inline", "uri"):
        raise ValueError(f"bad text_kind {text_kind!r}")
    if style_kind not in ("preset", "file", "uri"):
        raise ValueError(f"bad style_kind {style_kind!r}")
    if text_kind == "inline":
        if not text_value or not str(text_value).strip():
            raise ValueError("Text is empty.")
        if len(str(text_value)) > MAX_TEXT_LENGTH:
            raise ValueError(
                f"text has {len(str(text_value))} characters "
                f"(cap {MAX_TEXT_LENGTH})")
    else:
        if not text_value:
            raise ValueError("Wired text snapshot is empty.")
    if style_kind == "preset" and text_value is not None:
        pass
    if style_kind == "preset" and style_value not in VOICES:
        raise ValueError(f"unknown voice preset {style_value!r}")
    if style_kind in ("file", "uri") and not style_value:
        raise ValueError("voice style snapshot is empty.")
    if not request_id or not str(request_id).strip():
        raise ValueError("request_id is required")
    return {"text_kind": text_kind, "text_value": str(text_value),
            "style_kind": style_kind, "style_value": str(style_value),
            "model_dir": str(model_dir), "language": language,
            "speed": float(speed), "total_steps": int(total_steps),
            "silence_duration": float(silence_duration),
            "threads": int(threads), "request_id": str(request_id)}


def build_worker_argv(worker_path, request_path):
    """Pure argv builder: [sys.executable, worker, --request, request]."""
    worker = str(Path(worker_path).resolve())
    request = str(Path(request_path).resolve())
    if not worker or not request:
        raise ValueError("worker and request paths are required")
    return [sys.executable, worker, "--request", request]


def validate_waveform(wav, sample_rate):
    """Validate a synthesized waveform. Pure. Returns duration seconds."""
    import numpy as np
    if sample_rate is None or isinstance(sample_rate, bool):
        raise ValueError(f"bad sample rate {sample_rate!r}")
    try:
        rate = int(sample_rate)
    except (TypeError, ValueError):
        raise ValueError(f"bad sample rate {sample_rate!r}")
    if rate <= 0:
        raise ValueError(f"bad sample rate {rate!r}")
    arr = np.asarray(wav)
    if arr.ndim != 2 or arr.shape[0] != 1 or arr.shape[1] < 1:
        raise ValueError(f"expected waveform shape (1, N>0), got {arr.shape}")
    try:
        import numpy as _np
        if not bool(_np.all(_np.isfinite(arr))):
            raise ValueError("waveform contains non-finite samples")
    except ValueError:
        raise
    except Exception as e:
        raise ValueError(f"cannot validate waveform: {e}")
    return float(arr.shape[1]) / float(rate)


def scalar_duration(duration, frames, sample_rate):
    """Authoritative duration from frames/rate; tolerates SDK arrays."""
    if frames is not None and sample_rate:
        try:
            return float(int(frames)) / float(int(sample_rate))
        except (TypeError, ValueError):
            pass
    try:
        import numpy as _np
        arr = _np.asarray(duration).ravel()
        if arr.size >= 1:
            return float(arr[0])
    except Exception:
        pass
    try:
        return float(duration)
    except (TypeError, ValueError):
        raise ValueError(f"bad duration {duration!r}")
