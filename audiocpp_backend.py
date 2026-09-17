"""Shared client for the in-tree audio.cpp native runtime (tools/audiocpp/).

audio.cpp is a C++ inference runtime for local audio models. ANode drives
it as a sidecar *subprocess* for offline jobs (song generation, score
transcription, …) that must never run on the real-time audio thread
(AGENTS.md sections 4 and 6).

This module has no Qt dependency and performs no audio processing itself:

* :class:`AudioCppRuntime` — locates the `audiocpp_cli` binary and the
  model directories, plus a cheap health check.
* :func:`run_yue2_gen` — blocking YuE2 generation call, designed to run
  as an NRT worker function (``Node.submit_nrt``). Supports cooperative
  cancellation so a superseded generation does not burn GPU time.
* :class:`AudioCppJob` — :class:`base.Node` base with generate/cancel
  plumbing shared by all audio.cpp-backed nodes. Lives here (not under
  ``plugins/``) so it is not auto-registered in the node palette.

Sidecar policy (see also AGENTS.md section 6):

* ``AUDIOCPP_PIN_VERSION`` is the single pinned runtime. Bump it only
  deliberately: update the pin, the archive hashes below, and the fetch
  scripts, then re-run the pure argv-builder tests (no GPU needed) since
  CLI flags drift between releases.
* Auto-fetch covers Windows x64 + CUDA 13.3 only (the Blackwell/sm_120
  track); other platforms place an extracted tree under ``tools/audiocpp/
  bin/`` manually. The runtime is never auto-downloaded — only the node
  Download button and the ``tools/audiocpp/fetch_*.py`` scripts fetch.
* Model weights are CC BY-NC 4.0 (non-commercial): fetch-at-runtime only,
  never committed (see .gitignore), never bundled or redistributed. Kept
  outputs carry ``request.json`` / ``metrics.json`` provenance. Commercial
  use needs separate permission from the rights holders.
"""

import json
import logging
import os
import queue
import re
import shutil
import subprocess
import tempfile
import threading
from pathlib import Path

from base import Node
from download_util import DownloadCancelled, DownloadSpec, fetch_all, format_bytes

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parent
AUDIOCPP_PIN_VERSION = "v0.8.0"

AUDIOCPP_RELEASE_BASE = (
    f"https://github.com/0xShug0/audio.cpp/releases/download/{AUDIOCPP_PIN_VERSION}"
)
# Pinned runtime archives (Windows x64 + CUDA 13.3 track: the
# Blackwell/sm_120-capable build). (filename, size bytes, sha256).
RUNTIME_ARCHIVES = [
    ("audio-v0.8.0-bin-windows-x64-cuda13.3.zip", 269563691,
     "98ccbc3f5e6c73a6ffffa7ae32e1e204f4d12b15cac8a7901fb6be24d943fde6"),
    ("audio-v0.8.0-cudart-windows-x64-cuda13.3.zip", 575457454,
     "17fc9b2098d8167b6207be838e15187412f070429b6174b7353404e7131897fc"),
]

YUE2_HF_BASE = "https://huggingface.co/ngquocvinh/YuE2-3B-GGUF/resolve/main"
# (repo-relative path, size bytes, sha256). Weights are CC BY-NC 4.0.
YUE2_FILES = [
    ("yue2-3b-q4_k_m.gguf", 2877774656,
     "24315e53105cb3418b095d1e42e7276bf54b17100c51a43a4d0e37efdeed823d"),
    ("yue2-vae-f16.gguf", 265226496,
     "81e05a79e78ce5cb8deb1d87e17b73e5d205b440be2bd510ae62acf37b300ddb"),
    ("sidecars/yue2-model-config.json", 959,
     "ad3477bbef890bf98ae196c1e4b44779494a6231c4ab66f32708eabadf265329"),
    ("sidecars/yue2-generation-config.json", 466,
     "203830cebde6e3644eb291925362990d198e66c3b6006cd11f1aaa21904bcc61"),
    ("sidecars/yue2-qwen.tiktoken", 2561218,
     "b2b1b8dfb5cc5f024bafc373121c6aba3f66f9a5a0269e243470a1de16a33186"),
    ("sidecars/yue2-vae-config.json", 1378,
     "f0191bb9694009956de44e0c361a6f1334760be4c8f848e599bde242a54a0970"),
]


def runtime_fetch_specs(root=None):
    """DownloadSpecs for the runtime archives + their staging directory.

    Archives extract into ``bin/`` via :func:`install_runtime_archives`.
    Raises RuntimeError on platforms without a pinned build.
    """
    import sys
    if sys.platform != "win32":
        raise RuntimeError(
            "No pinned audio.cpp build for this platform. Download manually "
            "from https://github.com/0xShug0/audio.cpp/releases and extract "
            "under tools/audiocpp/bin/ (see tools/audiocpp/README.md)."
        )
    root = Path(root) if root is not None else REPO_ROOT / "tools" / "audiocpp"
    staging = root / "_dl"
    specs = [DownloadSpec(f"{AUDIOCPP_RELEASE_BASE}/{name}", staging / name,
                          size, sha, label=f"audio.cpp runtime: {name}")
             for name, size, sha in RUNTIME_ARCHIVES]
    return specs, staging


def install_runtime_archives(staging, bin_dir):
    """Extract verified runtime zips into bin_dir. NRT/worker-side only
    (disk I/O); removes the staging directory afterwards."""
    import zipfile
    staging, bin_dir = Path(staging), Path(bin_dir)
    bin_dir.mkdir(parents=True, exist_ok=True)
    for archive in sorted(staging.glob("*.zip")):
        with zipfile.ZipFile(archive) as z:
            z.extractall(bin_dir)
    shutil.rmtree(staging, ignore_errors=True)
    return bin_dir


def yue2_fetch_specs(model_dir):
    """DownloadSpecs for the YuE2 GGUF model + sidecars."""
    model_dir = Path(model_dir)
    return [DownloadSpec(f"{YUE2_HF_BASE}/{rel}", model_dir / rel,
                         size, sha, label=f"YuE2 model: {rel}")
            for rel, size, sha in YUE2_FILES]


def missing_specs(specs):
    """Subset of specs whose destination does not verify. Cheap checks."""
    from download_util import _verify
    return [s for s in specs
            if not _verify(s.dest, s.size, s.sha256)]


SHEETSAGE_HF_BASE = "https://huggingface.co/audio-cpp/SheetSage2-GGUF/resolve/main"
# (repo-relative path, size bytes, sha256). Upstream publishes no hash for
# this file, so integrity is size-checked only.
SHEETSAGE_FILES = [
    ("sheetsage2-orig.gguf", 2708224512, None),
]
SHEETSAGE_GGUF = "sheetsage2-orig.gguf"
SHEETSAGE_DEFAULT_MAX_TOKENS = 5120
# Weight dtypes the CLI actually accepts (its --help also names q8_0, but
# v0.8.0 rejects it at runtime).
SHEETSAGE_WEIGHT_TYPES = ("native", "f32", "f16", "bf16", "q4_0", "q4_k")


def sheetsage_fetch_specs(model_dir):
    """DownloadSpecs for the SheetSage2 GGUF model (self-contained)."""
    model_dir = Path(model_dir)
    return [DownloadSpec(f"{SHEETSAGE_HF_BASE}/{rel}", model_dir / rel,
                         size, sha, label=f"SheetSage2 model: {rel}")
            for rel, size, sha in SHEETSAGE_FILES]


_HEADER_LINE_RE = re.compile(r"^[A-Za-z]:")
_CHORD_RE = re.compile(r'"[^"\n]*"')
_LYRIC_SECTION_RE = re.compile(r"^\s*\[[^\[\]\n]{1,40}\]\s*$")


def abc_section_names(abc_text):
    """Section labels from `% name` comment lines, in order. Pure function."""
    sections = []
    for line in abc_text.splitlines():
        stripped = line.strip()
        if stripped.startswith("%") and len(stripped) > 1:
            sections.append(stripped[1:].strip())
    return sections


def lyric_section_names(lyrics_text):
    """Section tags from `[Verse]`-style header lines, in order. Pure function."""
    return [line.strip().strip("[]").strip()
            for line in lyrics_text.splitlines()
            if _LYRIC_SECTION_RE.match(line)]


def score_lyrics_fit_warning(*, abc_sections, lyric_sections):
    """Warn when a cover score and lyrics are shaped too differently.

    The model must stretch/cram syllables onto the plan when the section
    counts diverge, which garbles vocals. Returns a human-readable warning
    or None when the fit looks sane. Pure function; warn-only, never reject.
    """
    n_abc, n_lyr = len(abc_sections), len(lyric_sections)
    if n_abc == 0:
        return None
    if n_lyr == 0:
        if n_abc >= 3:
            return (f"score has {n_abc} sections but the lyrics have no "
                    f"[section] tags: add tags (e.g. [Verse]/[Chorus]) matching "
                    f"the score ({', '.join(abc_sections[:4])}"
                    f"{'…' if n_abc > 4 else ''}) or the words may garble")
        return None
    if abs(n_abc - n_lyr) >= 2 and max(n_abc, n_lyr) >= 2 * min(n_abc, n_lyr):
        return (f"score has {n_abc} sections but lyrics have {n_lyr}: "
                f"match [tags] to the score or the words may garble")
    return None


def strip_abc_chords(abc_text):
    """Return the ABC with chord symbols removed (melody-only variant).

    Chord symbols are `"Name"` quoted strings on music lines. Header lines
    (``X:``/``T:``/``K:``/…), ``%`` comments and ``w:`` lyric lines pass
    through untouched. Pure function; the YuE2 cover recipe wants a
    chord-free score for ``cot=melody``.
    """
    out = []
    for line in abc_text.splitlines():
        stripped = line.strip()
        if (not stripped or stripped.startswith("%")
                or stripped.startswith("w:") or _HEADER_LINE_RE.match(stripped)):
            out.append(line)
        else:
            out.append(_CHORD_RE.sub("", line))
    return "\n".join(out) + "\n"


def _build_sheetsage_argv(cli, model_dir, *, audio_wav, weight_type,
                          max_tokens, out_abc):
    """Pure argv builder (no process). Tested without a GPU."""
    if weight_type not in SHEETSAGE_WEIGHT_TYPES:
        raise ValueError(f"weight_type must be one of {SHEETSAGE_WEIGHT_TYPES}")
    if (not isinstance(max_tokens, int) or isinstance(max_tokens, bool)
            or max_tokens < 1):
        raise ValueError("max_tokens must be a positive integer")
    return [
        str(cli), "--task", "midi", "--family", "sheetsage2",
        "--model", str(model_dir), "--backend", "cuda", "--threads", "8",
        "--session-option", f"sheetsage2.weight_type={weight_type}",
        "--request-option", f"max_tokens={max_tokens}",
        "--audio", str(audio_wav),
        "--text-out", str(out_abc), "--metrics",
    ]


def run_sheetsage_transcribe(cli, model_dir, *, audio_wav, weight_type="native",
                             max_tokens=SHEETSAGE_DEFAULT_MAX_TOKENS,
                             out_abc, cancel_event=None, timeout_s=3600):
    """Transcribe one recording to an ABC score file. Blocking; NRT only.

    Returns ``{"abc": str, "metrics": dict}``. Raises
    :class:`GenerationCancelled` on cancellation, ``RuntimeError`` /
    ``ValueError`` / ``FileNotFoundError`` on failure.
    """
    cli, model_dir = Path(cli), Path(model_dir)
    audio_wav, out_abc = Path(audio_wav), Path(out_abc)
    if not cli.exists():
        raise FileNotFoundError(f"audiocpp_cli not found: {cli}")
    if not (model_dir / SHEETSAGE_GGUF).exists():
        raise FileNotFoundError(
            f"SheetSage2 weights not downloaded: {model_dir / SHEETSAGE_GGUF} "
            "(run tools/audiocpp/fetch_sheetsage2_gguf.py)"
        )
    if not audio_wav.exists():
        raise FileNotFoundError(f"input audio not found: {audio_wav}")
    argv = _build_sheetsage_argv(cli, model_dir, audio_wav=audio_wav,
                                 weight_type=weight_type, max_tokens=max_tokens,
                                 out_abc=out_abc)
    out_abc.parent.mkdir(parents=True, exist_ok=True)
    try:
        proc = subprocess.Popen(
            argv, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, **_no_window_kwargs(),
        )
    except OSError as e:
        raise RuntimeError(f"failed to launch audio.cpp: {e}")
    try:
        elapsed = 0.0
        step = 0.5
        output_tail = ""
        while True:
            if cancel_event is not None and cancel_event.is_set():
                raise GenerationCancelled("transcription cancelled")
            try:
                # May be retried after TimeoutExpired; returns full output.
                output_tail, _ = proc.communicate(timeout=step)
                break
            except subprocess.TimeoutExpired:
                elapsed += step
                if elapsed >= timeout_s:
                    raise TimeoutError(
                        f"SheetSage2 transcription exceeded {timeout_s}s; terminating"
                    )
    finally:
        if proc.poll() is None:
            try:
                proc.terminate()
            except OSError:
                pass
    if proc.returncode != 0:
        raise RuntimeError(
            f"audio.cpp exited with code {proc.returncode}: "
            f"{output_tail[-2000:].strip()}"
        )
    if not out_abc.exists():
        raise RuntimeError("audio.cpp reported success but wrote no ABC file")
    return {"abc": str(out_abc), "metrics": _parse_metrics(output_tail)}


class GenerationCancelled(RuntimeError):
    """Raised when a generation job is cancelled before producing output."""


class AudioCppRuntime:
    """Locates the in-tree audio.cpp runtime. Cheap filesystem checks only."""

    def __init__(self, root=None):
        self.root = Path(root) if root is not None else REPO_ROOT / "tools" / "audiocpp"
        self.bin_dir = self.root / "bin"
        self.cli = self.bin_dir / ("audiocpp_cli.exe" if os.name == "nt" else "audiocpp_cli")

    def check_ready(self):
        """Return (ok, hint). Never spawns a process."""
        if not self.cli.exists():
            return False, (
                "audio.cpp runtime not found. Run from the repo root: "
                "python tools/audiocpp/fetch_audiocpp.py"
            )
        return True, ""

    def health(self, timeout_s=60):
        """Run `--list-devices`. Returns (ok, output). Only called off the
        audio thread (NRT worker or control thread)."""
        ok, hint = self.check_ready()
        if not ok:
            return False, hint
        try:
            proc = subprocess.run(
                [str(self.cli), "--list-devices"],
                capture_output=True, text=True, timeout=timeout_s,
                **_no_window_kwargs(),
            )
        except (OSError, subprocess.SubprocessError) as e:
            return False, f"audio.cpp health check failed: {e}"
        if proc.returncode != 0:
            return False, f"audio.cpp health check failed: {proc.stderr[-500:]}"
        return True, proc.stdout


def _no_window_kwargs():
    if os.name == "nt" and hasattr(subprocess, "CREATE_NO_WINDOW"):
        return {"creationflags": subprocess.CREATE_NO_WINDOW}
    return {}


def _build_yue2_argv(cli, model_dir, *, lyrics, style, cot, seed,
                     main_gguf, vae_gguf, abc_file, out_wav):
    """Pure argv builder (no process). Tested without a GPU."""
    if cot not in ("off", "melody", "full"):
        raise ValueError(f"cot must be off/melody/full, got {cot!r}")
    if not isinstance(seed, int) or isinstance(seed, bool) or not 0 <= seed < 2 ** 63:
        raise ValueError(f"seed must be an integer in [0, 2**63), got {seed!r}")
    if not lyrics or not lyrics.strip():
        raise ValueError("lyrics must be nonempty text")
    if not style or not style.strip():
        raise ValueError("style must be nonempty text")
    argv = [
        str(cli), "--task", "gen", "--family", "yue2",
        "--model", str(model_dir), "--backend", "cuda", "--threads", "8",
        "--lyrics", lyrics,
        "--request-option", f"style={style}",
        "--request-option", f"cot={cot}",
        "--session-option", f"yue2.model_gguf={main_gguf}",
        "--session-option", f"yue2.vae_gguf={vae_gguf}",
        "--seed", str(seed),
        "--out", str(out_wav), "--metrics",
    ]
    if abc_file:
        if cot == "off":
            raise ValueError("an ABC score requires cot=melody or cot=full")
        argv += ["--request-option", f"abc_file={abc_file}"]
    return argv


_METRIC_RE = re.compile(r"^metrics\.(\w+)=([0-9.]+)\s*$")


def _parse_metrics(text):
    metrics = {}
    for line in text.splitlines():
        m = _METRIC_RE.match(line.strip())
        if m:
            try:
                metrics[m.group(1)] = float(m.group(2))
            except ValueError:
                pass
    return metrics


def run_yue2_gen(cli, model_dir, *, lyrics, style, cot="full", seed=831001,
                 main_gguf="yue2-3b-q4_k_m.gguf", vae_gguf="yue2-vae-f16.gguf",
                 abc_file=None, out_wav, cancel_event=None, timeout_s=1800):
    """Run one YuE2 generation. Blocking; call only from an NRT worker.

    Returns ``{"wav": str, "metrics": dict}``. Raises
    :class:`GenerationCancelled` on cancellation, ``RuntimeError`` /
    ``ValueError`` / ``FileNotFoundError`` on failure.
    """
    cli, model_dir, out_wav = Path(cli), Path(model_dir), Path(out_wav)
    if not cli.exists():
        raise FileNotFoundError(f"audiocpp_cli not found: {cli}")
    main_path, vae_path = model_dir / main_gguf, model_dir / vae_gguf
    if not main_path.exists():
        raise FileNotFoundError(
            f"GGUF profile not downloaded: {main_path} "
            "(run tools/audiocpp/fetch_yue2_gguf.py)"
        )
    if not vae_path.exists():
        raise FileNotFoundError(f"VAE GGUF not found: {vae_path}")
    if abc_file and not Path(abc_file).exists():
        raise FileNotFoundError(f"ABC score file not found: {abc_file}")
    argv = _build_yue2_argv(cli, model_dir, lyrics=lyrics, style=style,
                            cot=cot, seed=seed, main_gguf=main_gguf,
                            vae_gguf=vae_gguf, abc_file=abc_file, out_wav=out_wav)
    out_wav.parent.mkdir(parents=True, exist_ok=True)
    try:
        proc = subprocess.Popen(
            argv, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, **_no_window_kwargs(),
        )
    except OSError as e:
        raise RuntimeError(f"failed to launch audio.cpp: {e}")
    try:
        elapsed = 0.0
        step = 0.5
        output_tail = ""
        while True:
            if cancel_event is not None and cancel_event.is_set():
                raise GenerationCancelled("generation cancelled")
            try:
                # May be retried after TimeoutExpired; returns full output.
                output_tail, _ = proc.communicate(timeout=step)
                break
            except subprocess.TimeoutExpired:
                elapsed += step
                if elapsed >= timeout_s:
                    raise TimeoutError(
                        f"YuE2 generation exceeded {timeout_s}s; terminating"
                    )
    finally:
        if proc.poll() is None:
            try:
                proc.terminate()
            except OSError:
                pass
    if proc.returncode != 0:
        raise RuntimeError(
            f"audio.cpp exited with code {proc.returncode}: "
            f"{output_tail[-2000:].strip()}"
        )
    if not out_wav.exists():
        raise RuntimeError("audio.cpp reported success but wrote no WAV file")
    return {"wav": str(out_wav), "metrics": _parse_metrics(output_tail)}


class AudioCppJob(Node):
    """Node base with generate/cancel plumbing for audio.cpp sidecar jobs.

    A new job cancels the previous one first: a superseded multi-minute
    GPU job must not keep burning VRAM behind the replacement. The cancel
    event is set from the engine/control thread; the pool-thread worker
    terminates its own subprocess — ``Popen.terminate()`` is never called
    from the audio path.

    Shared job machine: the run/download/cancel transient-trigger dispatch
    (``on_ui_param_change``), the resumable runtime+model fetch
    (``_build_fetch_payload`` / ``_fetch_nrt``) and the fetch-side
    ``on_nrt_complete`` branches live here. Subclasses only provide:

    * ``RUN_PARAM`` / ``RUN_TAG`` / ``RUNNING_STATES`` / ``ACTION_VERB``,
    * ``_build_spec()`` — snapshot committed params, raise ValueError,
    * ``_running_status(spec)`` — (status, detail) while the job runs,
    * ``_spec_warnings(spec)`` — non-fatal fit warnings (warn-only),
    * ``_run_nrt(cancel_event, spec)`` — the blocking worker,
    * their job-specific ``on_nrt_complete`` branch (delegating the
      ``fetch`` / ``fetch_progress`` tags to the base helpers).

    ``is_offline`` marks the node for the offline visual language
    (AGENTS.md section 6): dashed URI wires, OFFLINE header badge, and a
    help-panel badge. Offline nodes never produce per-block signal from
    the job itself — they publish file paths + a one-block ready pulse.
    """

    is_offline = True

    RUN_PARAM = "generate"
    RUN_TAG = "gen"
    RUNNING_STATES = ("Generating", "Downloading")
    ACTION_VERB = "Generate"

    def __init__(self, name=""):
        super().__init__(name)
        self._job_lock = threading.Lock()
        self._job_cancel = None
        self._status = "Idle"
        self._status_detail = ""

    # ------------------------------------------------------------------
    # job control (engine/control thread)
    # ------------------------------------------------------------------
    def submit_job(self, tag, fn, *args):
        """Cancel any in-flight job, then submit ``fn(cancel_event, *args)``."""
        self._cancel_job()
        cancel = threading.Event()
        with self._job_lock:
            self._job_cancel = cancel
        self.submit_nrt(fn, cancel, *args, tag=tag)

    def _cancel_job(self):
        with self._job_lock:
            cancel = self._job_cancel
            self._job_cancel = None
        if cancel is not None:
            cancel.set()

    def remove(self):
        self._cancel_job()

    # ------------------------------------------------------------------
    # shared parameter / status helpers (engine/control thread)
    # ------------------------------------------------------------------
    def _restage(self, name, value):
        self.params[name].set(value)
        self.params[name].sync()

    def _resolve_model_dir(self):
        raw = self.params["model_dir"].value
        d = Path(raw)
        if not d.is_absolute():
            d = REPO_ROOT / d
        return d

    def _fail(self, message):
        self._status = "Error"
        self._status_detail = message
        self.error_msg = message
        logger.error(f"{type(self).__name__} {self.name}: {message}")

    def _need_engine(self):
        if getattr(self, "graph", None) is None or self.graph.engine is None:
            self._fail("Node is not attached to an engine.")
            self._refresh_ui()
            return False
        return True

    # ------------------------------------------------------------------
    # shared run/download/cancel dispatch (engine/control thread)
    # ------------------------------------------------------------------
    def _build_spec(self):
        """Snapshot committed params into a worker spec. Raise ValueError."""
        raise NotImplementedError

    def _spec_warnings(self, spec):
        """Non-fatal fit warnings shown alongside the running status.
        Subclass hook; warn-only, never reject. Runs on the engine/control
        thread at trigger time (small text reads only, same class as the
        existence checks in _build_spec)."""
        return []

    def _running_status(self, spec):
        """(status, detail) to show while the job runs. Subclass hook."""
        return self.RUNNING_STATES[0], "Running…"

    def _run_nrt(self, cancel_event, spec):
        """Blocking worker. Subclass hook (submitted as RUN_TAG)."""
        raise NotImplementedError

    def _on_job_submitted(self, spec):
        """Post-submit hook (e.g. start an elapsed-time clock)."""

    def _on_job_cancelled(self):
        """Post-cancel hook (e.g. stop an elapsed-time clock). The NRT
        completion may be discarded rather than delivered, so clock-like
        state must stop here, not in on_nrt_complete."""

    def on_ui_param_change(self, param_name: str):
        if param_name == self.RUN_PARAM:
            if not self.params[self.RUN_PARAM].value:
                return
            # Deliberate re-stage of the transient trigger (AGENTS.md §5).
            self._restage(self.RUN_PARAM, False)
            if not self._need_engine():
                return
            try:
                spec = self._build_spec()
            except ValueError as e:
                self._fail(str(e))
                self._refresh_ui()
                return
            self.error_msg = None
            self._status, self._status_detail = self._running_status(spec)
            for warning in self._spec_warnings(spec):
                logger.warning(f"{type(self).__name__} {self.name}: {warning}")
                self._status_detail += f" (⚠ {warning})"
            self._on_job_submitted(spec)
            self.submit_job(self.RUN_TAG, self._run_nrt, spec)
            self._refresh_ui()
        elif param_name == "cancel":
            if self.params["cancel"].value:
                self._restage("cancel", False)
                self._cancel_job()
                self._on_job_cancelled()
                if self._status in self.RUNNING_STATES:
                    self._status = "Cancelled"
                    self._status_detail = "Cancelled by user"
                self._refresh_ui()
        elif param_name == "download":
            if not self.params["download"].value:
                return
            self._restage("download", False)
            if not self._need_engine():
                return
            try:
                payload = self._build_fetch_payload()
            except ValueError as e:
                self._fail(str(e))
                self._refresh_ui()
                return
            if not payload["runtime"] and not payload["models"]:
                self.error_msg = None
                self._status = "Idle"
                self._status_detail = "Everything already downloaded"
                self._refresh_ui()
                return
            self.error_msg = None
            self._status = "Downloading"
            self._status_detail = "Starting download…"
            self.submit_job("fetch", self._fetch_nrt, payload)
            self._refresh_ui()

    # ------------------------------------------------------------------
    # shared fetch (engine/control thread + NRT worker)
    # ------------------------------------------------------------------
    @staticmethod
    def _spec_to_dict(spec):
        return {"url": spec.url, "dest": str(spec.dest), "size": spec.size,
                "sha256": spec.sha256, "label": spec.label}

    def _model_fetch_specs(self, model_dir):
        """DownloadSpecs for this node's model weights. Subclass hook."""
        raise NotImplementedError

    def _build_fetch_payload(self):
        """Plain-data fetch plan for the NRT worker. Raises ValueError."""
        runtime = AudioCppRuntime()
        model_dir = self._resolve_model_dir()
        try:
            runtime_specs, staging = runtime_fetch_specs(runtime.root)
        except RuntimeError as e:
            raise ValueError(str(e))
        return {
            "runtime": [self._spec_to_dict(s)
                        for s in missing_specs(runtime_specs)
                        if not runtime.cli.exists()],
            "staging": str(staging),
            "bin_dir": str(runtime.bin_dir),
            "models": [self._spec_to_dict(s)
                       for s in missing_specs(self._model_fetch_specs(model_dir))],
        }

    def _fetch_nrt(self, cancel_event, payload):
        """Download missing runtime/models with inbox progress. NRT only."""
        epoch = self._nrt_epoch
        inbox = self._nrt_inbox

        def _progress(label, done, total, state):
            if inbox is None:
                return
            try:
                inbox.put((epoch, "fetch_progress", True,
                           {"label": label, "done": done, "total": total,
                            "state": state}))
            except Exception:
                pass

        fetched = []
        model_specs = [DownloadSpec(s["url"], Path(s["dest"]), s["size"],
                                    s["sha256"], s["label"])
                       for s in payload["models"]]
        if model_specs:
            fetch_all(model_specs, progress_cb=_progress, cancel_event=cancel_event)
            fetched += [s.label for s in model_specs]
        runtime_specs = [DownloadSpec(s["url"], Path(s["dest"]), s["size"],
                                      s["sha256"], s["label"])
                         for s in payload["runtime"]]
        if runtime_specs:
            fetch_all(runtime_specs, progress_cb=_progress, cancel_event=cancel_event)
            install_runtime_archives(Path(payload["staging"]), Path(payload["bin_dir"]))
            fetched += [s.label for s in runtime_specs]
        return {"fetched": fetched}

    @staticmethod
    def _format_fetch_progress(progress):
        label = progress.get("label", "download")
        done, total, state = (progress.get("done", 0), progress.get("total"),
                              progress.get("state", ""))
        if state == "present":
            return f"{label}: already present"
        if state == "verifying":
            return f"{label}: verifying…"
        if total:
            pct = 100.0 * done / total
            return (f"{label}: {pct:.1f}% "
                    f"({format_bytes(done)} / {format_bytes(total)})")
        return f"{label}: {format_bytes(done)}"

    def _on_fetch_progress(self, ok, result):
        if ok:
            self._status = "Downloading"
            self._status_detail = self._format_fetch_progress(result)

    def _on_fetch_complete(self, ok, result):
        if not ok:
            if isinstance(result, (GenerationCancelled, DownloadCancelled)):
                self._status = "Cancelled"
                self._status_detail = "Download cancelled — re-run to resume"
                self.error_msg = None
            else:
                self._fail(f"Download failed: {result}")
            return
        self.error_msg = None
        self._status = "Idle"
        self._status_detail = f"Download complete — press {self.ACTION_VERB}"

    def _push_telemetry(self):
        """Deliver the current get_telemetry() snapshot to the UI now.

        Engine telemetry normally reaches widgets on the ~100 ms tick
        (running) or via NRT-drain coupling (stopped, controller.py).
        Button presses that produce no NRT traffic — e.g. Download with
        everything already fetched, or a rejected Generate — would
        otherwise leave the widget labels frozen, so the param handler
        pushes explicitly. Same {"type": "telemetry"} schema the engine
        itself emits; dropped when the output queue is full. Call only
        from engine/control contexts (param handlers, NRT completion),
        never from process().
        """
        graph = getattr(self, "graph", None)
        engine = getattr(graph, "engine", None) if graph is not None else None
        if engine is None:
            return
        try:
            data = self.get_telemetry()
        except Exception:
            return
        if not data:
            return
        try:
            engine.output_queue.put_nowait(
                {"type": "telemetry", "node_data": {self.id: data}})
        except queue.Full:
            pass

    def _refresh_ui(self):
        """Push telemetry now AND refresh the node header/snapshot state.

        _push_telemetry() only updates custom-widget labels; the red error
        badge (and param confirmations) come from snapshots, which the
        engine emits for structural ops, on error change while running, or
        on NRT drain while stopped — none of which fire for a bare button
        press. So a cleared error_msg would keep its red box (or a fresh
        one stay invisible) until something else snapshots. Requesting a
        snapshot here makes press->feedback immediate in both engine
        states. Button-press rare; same cost class as structural ops.
        """
        self._push_telemetry()
        graph = getattr(self, "graph", None)
        engine = getattr(graph, "engine", None) if graph is not None else None
        emit = getattr(engine, "_emit_snapshot", None)
        if callable(emit):
            emit()

    def on_nrt_discarded(self, tag, ok, result):
        self._cleanup_discarded(tag, ok, result)

    def _cleanup_discarded(self, tag, ok, result):
        """Subclass hook: release resources carried by a superseded result."""

    # ------------------------------------------------------------------
    # artifact helpers (NRT worker only)
    # ------------------------------------------------------------------
    @staticmethod
    def new_run_dir(model_dir, prefix):
        """Fresh per-generation directory for CLI output. Caller owns it."""
        outputs = Path(model_dir) / "outputs"
        outputs.mkdir(parents=True, exist_ok=True)
        return Path(tempfile.mkdtemp(prefix=f"{prefix}_", dir=str(outputs)))

    @staticmethod
    def keep_artifacts(run_dir, keep_dir, names):
        """Move named artifacts into a timestamped keep directory. Returns it."""
        keep_dir.mkdir(parents=True, exist_ok=True)
        for name in names:
            src = Path(run_dir) / name
            if src.exists():
                shutil.move(str(src), str(keep_dir / name))
        shutil.rmtree(run_dir, ignore_errors=True)
        return keep_dir

    @staticmethod
    def write_json(path, value):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
