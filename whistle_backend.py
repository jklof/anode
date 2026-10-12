"""Shared Whistle realtime speech-to-text helpers (no Qt, no torch, no audio
processing).

Whistle is Cactus Compute's 16.9 MB CPU speech-to-text model: English,
German, French, Spanish, Italian, Dutch and Polish, with word timestamps
(start / end / probability), keyword biasing and a streaming decoder that
commits the words two passes agree on (~1 s latency).

It is NOT an audio.cpp model. It ships as a single ``whistle.cact``
container read in place by ``libneedle3`` — the native engine shipped in the
``cactus-needle`` pip package — so ANode drives it from a background NRT
worker (AGENTS.md sections 4 and 6), never from ``process()``. Keeping the
pins, the fetch manifest and the engine extraction here means the node file
stays a lifecycle machine and the fetch script shares one source of truth
(``supertonic3_backend.py`` is the same shape).

Pins (verified 2026-10-12):

* Python package ``cactus-needle[mic] == 3.2.1`` (``MODEL_PIN``).
* Native engine wheel ``cactus_needle-3.3.1-py3-none-win_amd64.whl``
  (``ENGINE_PIN``), from which ``needle/libneedle3.dll`` is extracted.

``MODEL_PIN``/``ENGINE_PIN`` are deliberately coupled: the ``.cact``
container format is versioned together with the engine that maps it, so a
mismatched pin fails at ``needle_load`` with an opaque
"unknown .cact format tag" error rather than degrading gracefully. Bump
both pins and every size/hash below in one deliberate change, never one
alone.

Licensing: Apache-2.0 for both the weights and the engine — see the fetched
``models/Whistle/LICENSE``. Fetch-at-runtime only: never committed, never
bundled, never redistributed (see .gitignore).

Once the assets are on disk inference is fully local: nothing in the
streaming path opens a socket. The node additionally pins
``HF_HUB_OFFLINE=1`` before the first load, so even a mistaken cache miss in
``cactus-needle`` fails fast instead of reaching the network from a
background thread.
"""

import hashlib
import sys
import zipfile
from pathlib import Path

from download_util import DownloadError, DownloadSpec

REPO_ROOT = Path(__file__).resolve().parent

WHISTLE_HF_BASE = "https://huggingface.co/Cactus-Compute/whistle/resolve/main"
ENGINE_HF_BASE = "https://huggingface.co/Cactus-Compute/needle3/resolve/main"

MODEL_PIN = "3.2.1"
ENGINE_PIN = "3.3.1"

# Fixed language order for saved-patch compatibility: upstream's LANGUAGES
# tuple. Index 0 of the node's menu is auto-detect, so a menu index is a
# position in whistle_language_menu(), not in this tuple.
WHISTLE_LANGUAGES = ("en", "de", "fr", "es", "it", "nl", "pl")
AUTO_LANGUAGE_LABEL = "Auto (detect)"

DEFAULT_MODEL_DIR = "models/Whistle"
FETCH_LABEL_PREFIX = "Whistle"

# Whistle audio format: 16 kHz mono, whatever the engine delivers.
WHISTLE_SAMPLE_RATE = 16000

# ----------------------------------------------------------------------
# model assets
# ----------------------------------------------------------------------
# (repo-relative path, size bytes, sha256). The .cact hash is the upstream LFS
# oid (== file SHA-256). LICENSE is a regular git blob upstream, so no content
# hash is published for it and integrity is size-checked only — the same
# policy the Silero VAD spec uses.
WHISTLE_FILES = [
    ("whistle.cact", 16919407,
     "b6e02f048568ac5d01a2042556c658061e699acbc0aa2a1439f52f3d461dffeb"),
    ("LICENSE", 11358, None),
]

# ----------------------------------------------------------------------
# native engine
# ----------------------------------------------------------------------
# cactus-needle ships the v3 engine as libneedle3.* inside a platform wheel
# of the pure-Python package. We fetch the wheel (so one manifest entry
# covers both the pin and the integrity of the shipped bytes) and extract the
# single member we load through NEEDLE3_LIB_PATH.
ENGINE_WHEEL_DIR = "bin"
ENGINE_LIB_STEM = "libneedle3"

# Platform tag -> (wheel size, wheel sha256). The URL is derived from
# ENGINE_PIN. ONLY Windows x64 is pinned here, and therefore the only
# platform the Download button / fetch script auto-fetch — the same policy
# the audio.cpp sidecar follows (AGENTS.md section 6). Other platforms
# install bin/ manually; add a tuple here once a build is verified rather
# than letting an unverified platform silently pull the wrong wheel.
ENGINE_WHEELS = {
    "win_amd64": (
        699518,
        "4e50c5fa983c8737de200d9e96af937dd8135371ca557fe8c7c9eb1adba39f5a",
    ),
}

# sys.platform + platform.machine() -> upstream wheel tag. Kept as a map so
# a new platform is a data change, not a code change.
ENGINE_PLATFORM_TAGS = {
    ("win32", "AMD64"): "win_amd64",
    ("win32", "arm64"): "win_arm64",
    ("darwin", "arm64"): "macosx_11_0_arm64",
    ("darwin", "x86_64"): "macosx_11_0_x86_64",
    ("linux", "x86_64"): "manylinux2014_x86_64",
    ("linux", "arm64"): "manylinux2014_aarch64",
}

# Bytes of the extracted member, as verified for the Windows x64 wheel. Used
# as the post-extract integrity gate: a truncated or wrong-platform library
# would otherwise fail deep inside ctypes.CDLL with an unreadable message.
ENGINE_LIB_SIZE = 1531392
ENGINE_LIB_SHA256 = (
    "58d22ea22266094a3ef9213c02b9ab7cd06507642440e610cfbd41ac489298c3"
)

_LIB_SUFFIX = {"win32": ".dll", "darwin": ".dylib"}


def engine_library_name():
    """Platform-appropriate library file name (libneedle3.dll/.dylib/.so)."""
    return ENGINE_LIB_STEM + _LIB_SUFFIX.get(sys.platform, ".so")


def engine_platform_tag():
    """Upstream wheel tag for this machine, or None when unknown."""
    import platform
    key = (sys.platform, platform.machine())
    tag = ENGINE_PLATFORM_TAGS.get(key)
    if tag is None:
        # Normalise the common alternative spellings rather than failing on a
        # cosmetic difference (AMD64 vs x86_64 vs amd64).
        arch = key[1].lower().replace("amd64", "x86_64")
        for (plat, machine), candidate in ENGINE_PLATFORM_TAGS.items():
            if plat == key[0] and machine.lower() == arch:
                return candidate
    return tag


def weights_path(model_dir):
    """Path of the .cact weights container inside ``model_dir``."""
    return Path(model_dir) / WHISTLE_FILES[0][0]


def engine_library_path(model_dir):
    """Path of the extracted native engine inside ``model_dir``."""
    return Path(model_dir) / ENGINE_WHEEL_DIR / engine_library_name()


def engine_wheel_path(model_dir, tag=None):
    """Path the fetched engine wheel is stored at inside ``model_dir``."""
    tag = tag or engine_platform_tag()
    name = f"cactus_needle-{ENGINE_PIN}-py3-none-{tag}.whl"
    return Path(model_dir) / ENGINE_WHEEL_DIR / name


# ----------------------------------------------------------------------
# language menu
# ----------------------------------------------------------------------
def whistle_language_menu():
    """Menu items for the node's language parameter (order is patch format)."""
    return [AUTO_LANGUAGE_LABEL] + [code.upper() for code in WHISTLE_LANGUAGES]


def language_code(index):
    """Menu index -> ISO code, or None for auto-detect. Unknown -> None."""
    try:
        idx = int(index)
    except (TypeError, ValueError):
        return None
    if idx <= 0 or idx > len(WHISTLE_LANGUAGES):
        return None
    return WHISTLE_LANGUAGES[idx - 1]


def parse_keywords(raw):
    """Split the keywords param into a list. Pure.

    Accepts commas and/or newlines as separators so a pasted word list works
    either way. Empty terms are dropped; an empty result means "no biasing",
    which upstream treats as None.
    """
    text = str(raw or "").replace("\r", "\n").replace("\n", ",")
    return [term.strip() for term in text.split(",") if term.strip()]


# ----------------------------------------------------------------------
# fetch manifest / readiness
# ----------------------------------------------------------------------
def whistle_fetch_specs(model_dir):
    """DownloadSpecs for the Whistle assets (weights + engine wheel).

    Weights and LICENSE land directly in ``model_dir``; the engine wheel lands
    in ``model_dir/bin/`` and is unpacked by :func:`extract_engine_wheel`.
    """
    model_dir = Path(model_dir)
    specs = [
        DownloadSpec(f"{WHISTLE_HF_BASE}/{rel}", model_dir / rel, size, sha,
                     label=f"{FETCH_LABEL_PREFIX}: {rel}")
        for rel, size, sha in WHISTLE_FILES
    ]
    tag = engine_platform_tag()
    pinned = ENGINE_WHEELS.get(tag) if tag else None
    if pinned is not None:
        size, sha = pinned
        url = f"{ENGINE_HF_BASE}/python/cactus_needle-{ENGINE_PIN}-py3-none-{tag}.whl"
        specs.append(DownloadSpec(
            url, engine_wheel_path(model_dir, tag), size, sha,
            label=f"{FETCH_LABEL_PREFIX} engine: {engine_library_name()}"))
    return specs


def check_assets(model_dir):
    """Cheap readiness check. Returns ``(ok, missing_repo_relative_paths)``.

    No subprocess, no model load — just the same size/hash verification the
    fetch used, so a truncated or swapped file is caught before the engine
    tries to map it. The extracted library is size+hash verified on the
    platform whose bytes were pinned (Windows x64); elsewhere only existence
    is checked so a manually installed build is accepted without inventing a
    hash for it.
    """
    from download_util import _verify

    model_dir = Path(model_dir)
    missing = []
    for rel, size, sha in WHISTLE_FILES:
        if not _verify(model_dir / rel, size, sha):
            missing.append(rel)
    lib = engine_library_path(model_dir)
    if sys.platform == "win32":
        if not _verify(lib, ENGINE_LIB_SIZE, ENGINE_LIB_SHA256):
            missing.append(_as_relpath(lib, model_dir))
    elif not lib.is_file():
        missing.append(_as_relpath(lib, model_dir))
    return (len(missing) == 0), missing


def assets_ready(model_dir):
    """``check_assets`` reduced to a boolean, for the widget's Download button.

    Existence only: a full hash of 17 MB of weights on every UI tick would be
    the wrong trade for "is the button worth enabling".
    """
    model_dir = Path(model_dir)
    return weights_path(model_dir).is_file() and engine_library_path(model_dir).is_file()


def _as_relpath(path, root):
    try:
        return str(Path(path).relative_to(root))
    except ValueError:
        return str(path)


def extract_engine_wheel(model_dir, cancel_event=None):
    """Unpack ``needle/libneedle3.*`` out of the fetched wheel into ``bin/``.

    Returns the library path. Raises :class:`DownloadError` on a missing
    wheel, a missing member or a failed integrity check; nothing is written
    unless the extracted bytes verify, and a partially written library is
    removed, because a corrupt DLL surfaces as an unreadable ctypes error at
    the first inference instead of here.
    """
    if cancel_event is not None and cancel_event.is_set():
        raise DownloadError("engine extraction cancelled")

    model_dir = Path(model_dir)
    tag = engine_platform_tag()
    pinned = ENGINE_WHEELS.get(tag) if tag else None
    if pinned is None:
        target = engine_library_path(model_dir)
        raise DownloadError(
            f"no pinned Whistle engine wheel for {tag or sys.platform}; "
            f"place the engine at {target} manually "
            "(see whistle_backend.ENGINE_WHEELS)")

    wheel = engine_wheel_path(model_dir, tag)
    if not wheel.is_file():
        raise DownloadError(
            f"engine wheel not found: {wheel} — run tools/whistle/fetch.py")

    member = f"needle/{engine_library_name()}"
    with zipfile.ZipFile(wheel) as archive:
        try:
            data = archive.read(member)
        except KeyError:
            raise DownloadError(
                f"{wheel.name} does not contain {member}") from None

    if len(data) != ENGINE_LIB_SIZE or hashlib.sha256(data).hexdigest() != ENGINE_LIB_SHA256:
        raise DownloadError(
            f"extracted {member} from {wheel.name} failed verification "
            f"({len(data)} bytes, sha256 {hashlib.sha256(data).hexdigest()})")

    out = engine_library_path(model_dir)
    out.parent.mkdir(parents=True, exist_ok=True)
    staged = out.with_name(out.name + ".part")
    try:
        staged.write_bytes(data)
        staged.replace(out)
    except OSError:
        try:
            staged.unlink(missing_ok=True)
        except OSError:
            pass
        raise
    return out